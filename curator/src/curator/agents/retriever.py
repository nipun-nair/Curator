"""Retriever agent — a bounded tool-calling loop over the MCP toolbox.

The model is not handed a search function; it is handed the *discovered* tool schemas
and asked which to call next. That is real function calling: the loop validates the
requested name against what the server advertises, validates the arguments against the
JSON schema, executes, feeds the observation back, and stops on `finish` or on the
step budget — whichever comes first.

The step budget is the reliability mechanism. An agent that cannot loop forever cannot
burn a GPU hour on a typo.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import LLM
from ..schemas import Constraints, Evidence, Paper, ToolCall
from ..tools.client import Toolbox

SYSTEM = """You are the retrieval agent of a research-paper recommender.
You have these tools:

{tools}

Decide the single next action. Search broadly first, then fetch the abstracts of the
papers you intend to recommend so the writer can quote real content. Call `finish`
when you have enough evidence — usually after one or two searches plus abstracts.

Never invent a paper_id. Only ids returned by a tool exist."""


class Step(BaseModel):
    """The model's next move. `thought` is kept short on purpose: it is logged, and
    long chains of reasoning here mostly buy latency."""

    thought: str = Field(default="", max_length=400)
    action: str  # a tool name, or "finish"
    arguments: dict = Field(default_factory=dict)


def _validate(step: Step, toolbox: Toolbox) -> str | None:
    """Return an error string if the model asked for something impossible."""
    if step.action == "finish":
        return None
    if step.action not in toolbox.names():
        return f"Unknown tool {step.action!r}. Available: {sorted(toolbox.names())}"
    schema = next(t["parameters"] for t in toolbox.schemas() if t["name"] == step.action)
    required = set((schema or {}).get("required", []))
    missing = required - set(step.arguments)
    if missing:
        return f"Missing required arguments for {step.action}: {sorted(missing)}"
    unknown = set(step.arguments) - set((schema or {}).get("properties", {}))
    if unknown:
        return f"Unknown arguments for {step.action}: {sorted(unknown)}"
    return None


def retrieve(
    llm: LLM,
    toolbox: Toolbox,
    query: str,
    constraints: Constraints,
    max_steps: int = 4,
    hint: str = "",
) -> tuple[list[Evidence], list[ToolCall]]:
    """Run the loop; return the evidence found and the calls attempted."""
    system = SYSTEM.format(tools=toolbox.prompt_block())
    scratch: list[str] = []
    calls: list[ToolCall] = []
    found: dict[str, Evidence] = {}
    abstracts: dict[str, str] = {}

    goal = f"Information need: {query}\nConstraints: {constraints.model_dump_json()}"
    if hint:
        goal += f"\nThe reviewer says this is still missing: {hint}"

    for _ in range(max_steps):
        prompt = goal + "\n\nHistory:\n" + ("\n".join(scratch) or "(nothing yet)") + "\n\nNext step:"
        try:
            step = llm.structured(prompt, Step, system=system)
        except Exception as exc:
            scratch.append(f"ERROR malformed step ({exc}); falling back to a plain search")
            step = Step(action="search_papers", arguments={"query": query, "k": constraints.k})

        if err := _validate(step, toolbox):
            scratch.append(f"REJECTED {step.action}: {err}")
            continue
        if step.action == "finish":
            break

        calls.append(ToolCall(name=step.action, arguments=step.arguments))
        result = toolbox.call(step.action, step.arguments)
        if not result.ok:
            scratch.append(f"{step.action} failed: {result.error}")
            continue

        _absorb(result.name, result.content, found, abstracts)
        scratch.append(f"{step.action}({step.arguments}) -> {_summarise(result.content)}")

    # Backfill abstracts the agent forgot to fetch: the critic needs text to check.
    for pid, ev in list(found.items()):
        if not ev.paper.abstract and pid not in abstracts:
            r = toolbox.call("get_paper", {"paper_id": pid})
            if r.ok and isinstance(r.content, dict):
                ev.paper.abstract = r.content.get("abstract", "")

    for pid, abstract in abstracts.items():
        if pid in found:
            found[pid].paper.abstract = abstract

    return list(found.values()), calls


def _absorb(name: str, content, found: dict[str, Evidence], abstracts: dict[str, str]) -> None:
    """Turn whatever a tool returned into Evidence, without trusting its shape."""
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or "paper_id" not in item:
                continue
            pid = item["paper_id"]
            if pid not in found:
                found[pid] = Evidence(
                    paper=Paper(
                        paper_id=pid,
                        title=item.get("title", ""),
                        abstract=item.get("abstract", ""),
                        year=item.get("year"),
                        categories=item.get("categories", []) or [],
                    ),
                    score=float(item.get("score", 0.0)),
                    source="tool",
                    retrieved_by=name,
                )
    elif isinstance(content, dict) and "paper_id" in content:
        pid = content["paper_id"]
        abstracts[pid] = content.get("abstract", "")
        if pid not in found:
            found[pid] = Evidence(paper=Paper(**{
                k: v for k, v in content.items() if k in Paper.model_fields
            }), source="tool", retrieved_by=name)


def _summarise(content) -> str:
    if isinstance(content, list):
        return f"{len(content)} results: " + ", ".join(
            f"{c.get('paper_id')} {str(c.get('title', ''))[:60]}"
            for c in content[:8]
            if isinstance(c, dict)
        )
    if isinstance(content, dict):
        return f"{content.get('paper_id', '?')}: {str(content.get('abstract', ''))[:300]}"
    return str(content)[:300]
