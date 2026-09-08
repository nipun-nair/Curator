"""The orchestration layer: a LangGraph StateGraph with one real cycle.

    START -> router -+-> explainer -> END          (chitchat: never wake retrieval)
                     `-> retriever -> critic -+-> explainer -> END
                            ^                 |
                            `-----------------'  (revise, bounded by max_critic_rounds)

Why a graph rather than a chain: the critic edge is conditional and backwards. Chains
cannot express "go get more evidence and try again" without an outer while-loop that
nobody can inspect. Here the loop is a first-class edge, the bound is config, and the
number of times it fired lands in the trace.

`Curator.answer()` is the single entry point used by the CLI, the tests and the eval
harness — so what CI runs is what a user runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from .agents import critic as critic_mod
from .agents import explainer as explainer_mod
from .agents import retriever as retriever_mod
from .agents.router import make_router
from .agents.state import GraphState
from .config import Config
from .llm import LLM, build_llm
from .schemas import Answer, Evidence, Intent, TurnTrace
from .tools.client import Toolbox


@dataclass
class Curator:
    cfg: Config
    toolbox: Toolbox
    llm: LLM | None = None
    router_llm: LLM | None = None

    def __post_init__(self) -> None:
        self._router = make_router(self.cfg, self.llm, self.router_llm)
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def _node_router(self, state: GraphState) -> dict[str, Any]:
        route = self._router(state["query"], state.get("history", []))
        return {"route": route, "search_query": route.topic or state["query"], "critic_rounds": 0}

    def _node_retriever(self, state: GraphState) -> dict[str, Any]:
        route = state["route"]
        hint = state.get("verdict").missing_info if state.get("verdict") else ""
        evidence, calls = retriever_mod.retrieve(
            self.llm,
            self.toolbox,
            state.get("search_query", state["query"]),
            route.constraints,
            max_steps=self.cfg.agents.max_tool_steps,
            hint=hint,
        )
        # Union across rounds: a second pass should add evidence, never lose it.
        merged: dict[str, Evidence] = {e.paper.paper_id: e for e in state.get("evidence", [])}
        for e in evidence:
            merged.setdefault(e.paper.paper_id, e)
        trace = state.get("trace") or TurnTrace()
        trace.tool_calls.extend(calls)
        return {"evidence": list(merged.values()), "trace": trace}

    def _node_critic(self, state: GraphState) -> dict[str, Any]:
        verdict = critic_mod.review(
            self.llm, state["query"], state.get("evidence", []), state["route"].constraints
        )
        return {
            "verdict": verdict,
            "verdicts": [verdict],
            "critic_rounds": state.get("critic_rounds", 0) + 1,
        }

    def _node_explainer(self, state: GraphState) -> dict[str, Any]:
        route = state["route"]
        if route.intent is Intent.CHITCHAT:
            return {"answer": Answer(reply="Happy to help — what topic should I search?")}
        rejected = set(state["verdict"].rejected_ids) if state.get("verdict") else set()
        evidence = [e for e in state.get("evidence", []) if e.paper.paper_id not in rejected]
        answer = explainer_mod.compose(
            self.llm, state["query"], route.intent, evidence, k=route.constraints.k
        )
        return {"answer": answer}

    # ------------------------------------------------------------------ edges
    def _after_router(self, state: GraphState) -> str:
        return "explainer" if state["route"].intent is Intent.CHITCHAT else "retriever"

    def _after_critic(self, state: GraphState) -> str:
        verdict = state["verdict"]
        if verdict.revise and state.get("critic_rounds", 0) < self.cfg.agents.max_critic_rounds:
            return "retriever"          # the cycle
        return "explainer"

    def _build(self):
        g = StateGraph(GraphState)
        g.add_node("router", self._node_router)
        g.add_node("retriever", self._node_retriever)
        g.add_node("explainer", self._node_explainer)
        g.add_edge(START, "router")

        if self.cfg.agents.single_agent:
            # Ablation: no critic, no cycle, one retrieval pass. The point of comparison.
            g.add_conditional_edges("router", self._after_router, ["retriever", "explainer"])
            g.add_edge("retriever", "explainer")
        else:
            g.add_node("critic", self._node_critic)
            g.add_conditional_edges("router", self._after_router, ["retriever", "explainer"])
            if self.cfg.agents.use_critic:
                g.add_edge("retriever", "critic")
                g.add_conditional_edges("critic", self._after_critic, ["retriever", "explainer"])
            else:
                g.add_edge("retriever", "explainer")
        g.add_edge("explainer", END)
        return g.compile()

    # ------------------------------------------------------------------ api
    def answer(
        self, query: str, user_id: str = "demo", history: list[dict[str, str]] | None = None
    ) -> tuple[Answer, TurnTrace]:
        t0 = time.perf_counter()
        usage_before = (
            (self.llm.usage.calls, self.llm.usage.prompt_chars, self.llm.usage.completion_chars)
            if self.llm
            else (0, 0, 0)
        )
        state: GraphState = {
            "query": query,
            "user_id": user_id,
            "history": history or [],
            "trace": TurnTrace(),
        }
        # recursion_limit is the last-resort circuit breaker: even a mis-wired edge
        # cannot spin. Budget = nodes per pass * allowed passes, plus slack.
        out = self.graph.invoke(state, {"recursion_limit": 6 + 3 * self.cfg.agents.max_critic_rounds})

        trace: TurnTrace = out.get("trace") or TurnTrace()
        trace.router = out.get("route")
        trace.evidence_ids = [e.paper.paper_id for e in out.get("evidence", [])]
        trace.verdicts = out.get("verdicts", [])
        trace.critic_rounds = out.get("critic_rounds", 0)
        trace.tool_results = list(self.toolbox.history)
        trace.latency_ms = (time.perf_counter() - t0) * 1000
        if self.llm:
            trace.llm_calls = self.llm.usage.calls - usage_before[0]
            trace.prompt_chars = self.llm.usage.prompt_chars - usage_before[1]
            trace.completion_chars = self.llm.usage.completion_chars - usage_before[2]
        self.toolbox.history.clear()
        return out.get("answer", Answer(reply="")), trace


def build_curator(cfg: Config, toolbox: Toolbox) -> Curator:
    """Wire the LLM backends declared in config. The router may use a different
    backend from the writer — that is the whole point of the LoRA router."""
    llm = build_llm(cfg.llm)
    router_llm = None
    if cfg.agents.router == "lora" and cfg.llm.adapter_path:
        from copy import deepcopy

        rcfg = deepcopy(cfg.llm)
        rcfg.backend = "hf_local"
        router_llm = build_llm(rcfg)
    return Curator(cfg=cfg, toolbox=toolbox, llm=llm, router_llm=router_llm)
