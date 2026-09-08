"""Critic agent — the reliability gate, and the reason this is a *graph* not a chain.

It answers two questions about the retrieved set:
  1. is there enough evidence here to answer the request at all?
  2. does any candidate fail a stated constraint?

If either fails it sets `revise=True` with a `missing_info` string, and the graph edges
send control back to the retriever with that string as a hint. `max_critic_rounds`
caps the cycle, and every round is counted in the trace so the eval can report how
often the loop actually fires and what it buys.

There is also a deterministic pre-check that needs no model at all — constraint
violations are arithmetic, and spending a generation on them would be silly.
"""

from __future__ import annotations

from ..llm import LLM
from ..schemas import Constraints, CriticVerdict, Evidence

SYSTEM = """You review a set of retrieved papers before they are shown to a user.
Be strict but not perfectionist: the goal is that every recommendation can be
justified from the abstracts present, not that the set is optimal.

Set revise=true ONLY if the evidence cannot support an answer — wrong topic entirely,
or fewer than two usable papers. If you set revise=true, `missing_info` must say what
to search for next, as a search phrase.
List in rejected_ids any paper that is off-topic or violates a stated constraint."""


def deterministic_check(evidence: list[Evidence], cons: Constraints) -> CriticVerdict:
    """Cheap, exact, model-free. Runs first; a model is only consulted if this passes."""
    rejected = []
    for e in evidence:
        p = e.paper
        if cons.year_min and (p.year or 0) < cons.year_min:
            rejected.append(p.paper_id)
        elif cons.year_max and (p.year or 9999) > cons.year_max:
            rejected.append(p.paper_id)
        elif cons.exclude_surveys and "survey" in p.title.lower():
            rejected.append(p.paper_id)
    survivors = [e for e in evidence if e.paper.paper_id not in rejected]
    if len(survivors) < 2:
        return CriticVerdict(
            grounded=False,
            revise=True,
            missing_info="not enough papers satisfy the stated constraints",
            rejected_ids=rejected,
            notes="deterministic constraint check",
        )
    return CriticVerdict(
        grounded=True, revise=False, rejected_ids=rejected, notes="deterministic checks passed"
    )


def review(
    llm: LLM | None, query: str, evidence: list[Evidence], cons: Constraints
) -> CriticVerdict:
    pre = deterministic_check(evidence, cons)
    if pre.revise or llm is None:
        return pre

    listing = "\n\n".join(
        f"[{e.paper.paper_id}] {e.paper.title} ({e.paper.year})\n{e.paper.abstract[:600]}"
        for e in evidence
        if e.paper.paper_id not in pre.rejected_ids
    )
    prompt = (
        f"User asked: {query}\nStated constraints: {cons.model_dump_json()}\n\n"
        f"Retrieved papers:\n{listing}\n\nReview this set."
    )
    try:
        verdict = llm.structured(prompt, CriticVerdict, system=SYSTEM)
    except Exception as exc:
        # A broken critic must not block an otherwise fine answer: fail open, but say so.
        return CriticVerdict(
            grounded=True, revise=False, rejected_ids=pre.rejected_ids,
            notes=f"critic unavailable, passed through ({type(exc).__name__})",
        )
    verdict.rejected_ids = sorted(set(verdict.rejected_ids) | set(pre.rejected_ids))
    # Guard against a critic that rejects everything and then says it is fine.
    if len(evidence) - len(verdict.rejected_ids) < 1:
        verdict.revise = True
        verdict.missing_info = verdict.missing_info or query
    return verdict
