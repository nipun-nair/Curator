"""Explainer agent — writes the user-facing turn, with citations that must resolve.

Grounding is enforced twice: the prompt asks for `[paper_id]` citations, and
`enforce_grounding` then drops any recommendation whose id is not in the evidence set.
The drop rate is a reported metric, not a silent repair — if a model is inventing ids,
the eval should show it.
"""

from __future__ import annotations

from ..llm import LLM
from ..schemas import Answer, Evidence, Intent, Recommendation

SYSTEM = """You write the final reply of a research-paper recommender.

Rules:
- Recommend only papers from the evidence below, by their exact paper_id.
- Every `why` must be supported by that paper's abstract. One or two sentences.
- Say plainly when the evidence is thin rather than padding the list.
- `reply` is 2-4 sentences of connective prose. Do not restate every `why`."""


def compose(
    llm: LLM | None, query: str, intent: Intent, evidence: list[Evidence], k: int = 5
) -> Answer:
    usable = evidence[:k]
    if not usable:
        return Answer(
            reply="I could not find papers matching that in the indexed corpus. "
            "Try broadening the topic or relaxing the year constraint.",
            recommendations=[],
        )
    if llm is None:
        return _fallback(usable)

    listing = "\n\n".join(
        f"[{e.paper.paper_id}] {e.paper.title} ({e.paper.year})\n{e.paper.abstract[:800]}"
        for e in usable
    )
    prompt = f"User asked: {query}\nIntent: {intent.value}\n\nEvidence:\n{listing}\n\nWrite the reply."
    try:
        answer = llm.structured(prompt, Answer, system=SYSTEM)
    except Exception:
        return _fallback(usable)
    return enforce_grounding(answer, usable)


def enforce_grounding(answer: Answer, evidence: list[Evidence]) -> Answer:
    """Drop hallucinated ids. Returns a new Answer; the caller records how many went."""
    valid = {e.paper.paper_id for e in evidence}
    kept = []
    for rec in answer.recommendations:
        if rec.paper_id in valid:
            rec.citations = sorted(set(rec.citations) & valid) or [rec.paper_id]
            kept.append(rec)
    answer.recommendations = kept
    return answer


def _fallback(evidence: list[Evidence]) -> Answer:
    """Template answer used when no LLM is configured or generation failed. Still
    correct, still cited — degraded, not broken."""
    recs = [
        Recommendation(
            paper_id=e.paper.paper_id,
            why=(e.paper.abstract or e.paper.title)[:200].strip(),
            citations=[e.paper.paper_id],
        )
        for e in evidence
    ]
    return Answer(reply=f"Found {len(recs)} papers matching your request.", recommendations=recs)
