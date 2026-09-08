"""Graph state. One TypedDict, no hidden globals — anything an agent needs to hand
to the next one is here and therefore visible in the trace."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from ..schemas import Answer, CriticVerdict, Evidence, RouterOutput, TurnTrace


def _extend(a: list[Any], b: list[Any]) -> list[Any]:
    """Reducer for accumulating channels across the critic loop."""
    return [*a, *b]


class GraphState(TypedDict, total=False):
    # inputs
    query: str
    user_id: str
    history: list[dict[str, str]]

    # router
    route: RouterOutput

    # retrieval
    search_query: str
    evidence: list[Evidence]

    # critic loop
    verdict: CriticVerdict
    verdicts: Annotated[list[CriticVerdict], _extend]
    critic_rounds: int

    # output
    answer: Answer
    trace: TurnTrace
