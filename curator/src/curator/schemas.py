"""Typed contracts for every hop in the system.

Every agent boundary is a Pydantic model. That is the reliability story: an agent
never hands another agent free text it has to re-parse, and a malformed generation
is a caught ValidationError with a repair retry rather than a silent wrong answer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Intent(StrEnum):
    RECOMMEND = "recommend"   # "find me papers on X"
    REFINE = "refine"         # "more recent ones", "less survey-y"
    COMPARE = "compare"       # "how does A differ from B"
    EXPLAIN = "explain"       # "why did you suggest that one"
    CHITCHAT = "chitchat"     # everything the retriever should not be woken up for


class Constraints(BaseModel):
    """Slots the router extracts. All optional — absence means 'no constraint'."""

    year_min: int | None = None
    year_max: int | None = None
    exclude_surveys: bool = False
    must_have_code: bool = False
    k: int = Field(default=5, ge=1, le=20)


class RouterOutput(BaseModel):
    """What the (optionally LoRA-tuned) router emits. Small, closed, easy to score."""

    intent: Intent
    topic: str = ""
    constraints: Constraints = Field(default_factory=Constraints)

    @field_validator("topic")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    url: str = ""


class Evidence(BaseModel):
    """A retrieved paper plus the provenance of *how* it was retrieved.

    `source` is what makes the ablations legible: you can see at eval time whether a
    win came from BM25, from the dense index, or only from the fusion of both.
    """

    paper: Paper
    score: float = 0.0
    source: Literal["bm25", "dense", "rrf", "tool"] = "rrf"
    retrieved_by: str = "search_papers"


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    name: str
    ok: bool
    latency_ms: float = 0.0
    content: Any = None
    error: str | None = None


class Recommendation(BaseModel):
    paper_id: str
    why: str
    citations: list[str] = Field(default_factory=list)


class CriticVerdict(BaseModel):
    """The reliability gate. `revise` is the only thing that can re-open retrieval."""

    grounded: bool
    revise: bool = False
    missing_info: str = ""
    rejected_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class Answer(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)


class TurnTrace(BaseModel):
    """Everything the eval harness needs, captured per turn. No hidden state."""

    router: RouterOutput | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    critic_rounds: int = 0
    verdicts: list[CriticVerdict] = Field(default_factory=list)
    latency_ms: float = 0.0
    llm_calls: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    errors: list[str] = Field(default_factory=list)
