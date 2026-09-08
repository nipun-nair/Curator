"""End-to-end graph behaviour against the deterministic mock backend.

These are the tests that would have caught every wiring bug I hit while building it:
a chitchat turn that woke the retriever, a critic that looped forever, and an explainer
that happily cited a paper id the retriever never saw.
"""

from __future__ import annotations

import json

from curator.graph import Curator
from curator.llm import MockLLM
from curator.schemas import Intent


def mock_with(rules):
    from curator.config import LLMConfig

    return MockLLM(LLMConfig(backend="mock", max_retries=0), rules=rules)


STEP_SEARCH = json.dumps({"thought": "search", "action": "search_papers",
                          "arguments": {"query": "retrieval augmented generation", "k": 5}})
STEP_FINISH = json.dumps({"action": "finish", "arguments": {}})
GOOD_CRITIC = json.dumps({"grounded": True, "revise": False, "rejected_ids": [], "notes": "ok"})


def test_chitchat_never_calls_a_tool(cfg, toolbox):
    llm = mock_with([(".*", STEP_FINISH)])
    curator = Curator(cfg=cfg, toolbox=toolbox, llm=llm)
    answer, trace = curator.answer("thanks, that was great")
    assert trace.router.intent is Intent.CHITCHAT
    assert trace.tool_calls == []
    assert answer.recommendations == []


def test_happy_path_retrieves_and_grounds(cfg, toolbox):
    llm = mock_with(
        [
            (r"retrieval agent", STEP_SEARCH),
            (r"You review a set", GOOD_CRITIC),
            (r"You write the final", json.dumps({
                "reply": "Here are some papers.",
                "recommendations": [
                    {"paper_id": "FIX-0001", "why": "directly on topic", "citations": ["FIX-0001"]}
                ],
            })),
        ]
    )
    curator = Curator(cfg=cfg, toolbox=toolbox, llm=llm)
    answer, trace = curator.answer("papers on retrieval augmented generation")
    assert trace.router.intent is Intent.RECOMMEND
    assert "FIX-0001" in trace.evidence_ids
    assert [r.paper_id for r in answer.recommendations] == ["FIX-0001"]


def test_hallucinated_citations_are_dropped(cfg, toolbox):
    llm = mock_with(
        [
            (r"retrieval agent", STEP_SEARCH),
            (r"You review a set", GOOD_CRITIC),
            (r"You write the final", json.dumps({
                "reply": "Here you go.",
                "recommendations": [
                    {"paper_id": "FIX-0001", "why": "real", "citations": ["FIX-0001"]},
                    {"paper_id": "ARXIV-9999", "why": "invented", "citations": ["ARXIV-9999"]},
                ],
            })),
        ]
    )
    curator = Curator(cfg=cfg, toolbox=toolbox, llm=llm)
    answer, _ = curator.answer("papers on retrieval augmented generation")
    ids = [r.paper_id for r in answer.recommendations]
    assert "FIX-0001" in ids and "ARXIV-9999" not in ids


def test_critic_loop_is_bounded(cfg, toolbox):
    """A critic that always demands revision must still terminate."""
    always_revise = json.dumps(
        {"grounded": False, "revise": True, "missing_info": "more", "rejected_ids": []}
    )
    llm = mock_with(
        [
            (r"retrieval agent", STEP_SEARCH),
            (r"You review a set", always_revise),
            (r"You write the final", json.dumps({"reply": "thin evidence", "recommendations": []})),
        ]
    )
    curator = Curator(cfg=cfg, toolbox=toolbox, llm=llm)
    _, trace = curator.answer("papers on retrieval augmented generation")
    assert trace.critic_rounds <= cfg.agents.max_critic_rounds


def test_bad_tool_name_is_recovered_from(cfg, toolbox):
    """The retriever must reject an unknown tool and keep going, not crash the turn."""
    bad = json.dumps({"action": "hack_the_db", "arguments": {}})
    llm = mock_with(
        [
            (r"REJECTED", STEP_SEARCH),          # second step, after the rejection
            (r"retrieval agent", bad),           # first step
            (r"You review a set", GOOD_CRITIC),
            (r"You write the final", json.dumps({"reply": "ok", "recommendations": []})),
        ]
    )
    curator = Curator(cfg=cfg, toolbox=toolbox, llm=llm)
    _, trace = curator.answer("papers on retrieval augmented generation")
    assert "hack_the_db" not in [c.name for c in trace.tool_calls]
    assert trace.evidence_ids  # it recovered and still retrieved


def test_malformed_json_falls_back_not_crashes(cfg, toolbox):
    llm = mock_with([(r".*", "I am not JSON at all, sorry.")])
    curator = Curator(cfg=cfg, toolbox=toolbox, llm=llm)
    answer, trace = curator.answer("papers on retrieval augmented generation")
    assert trace.evidence_ids            # fell back to a plain search
    assert answer.recommendations        # fell back to the template writer


def test_constraints_are_enforced_end_to_end(cfg, toolbox):
    llm = mock_with(
        [
            (r"retrieval agent", STEP_SEARCH),
            (r"You review a set", GOOD_CRITIC),
            (r"You write the final", json.dumps({"reply": "x", "recommendations": []})),
        ]
    )
    curator = Curator(cfg=cfg, toolbox=toolbox, llm=llm)
    _, trace = curator.answer(
        "papers on retrieval augmented generation since 2024, no surveys"
    )
    assert trace.router.constraints.year_min == 2024
    assert trace.router.constraints.exclude_surveys is True


def test_single_agent_ablation_skips_the_critic(cfg, toolbox):
    from copy import deepcopy

    ab = deepcopy(cfg)
    ab.agents.single_agent = True
    ab.agents.use_critic = False
    llm = mock_with(
        [
            (r"retrieval agent", STEP_SEARCH),
            (r"You write the final", json.dumps({"reply": "x", "recommendations": []})),
        ]
    )
    _, trace = Curator(cfg=ab, toolbox=toolbox, llm=llm).answer("rag papers")
    assert trace.critic_rounds == 0
