"""Router agent — intent classification + slot filling.

Three interchangeable implementations, which is the whole point of this node:

  rules  a keyword baseline. Fast, free, and the floor every learned model must beat.
  llm    zero/few-shot prompting of the general chat model.
  lora   a 0.6B base model with a QLoRA adapter trained on `train/build_router_data.py`
         output. Same interface, ~10x cheaper per turn than `llm`.

Because all three emit `RouterOutput`, the ablation is a config flag and the eval
harness scores them with the same intent-F1 and slot-accuracy code.
"""

from __future__ import annotations

import re

from ..config import Config
from ..llm import LLM
from ..schemas import Constraints, Intent, RouterOutput

SYSTEM = """You are the router of a research-paper recommender.
Classify the user's turn into exactly one intent and extract search constraints.

intents:
  recommend — the user wants papers on a topic
  refine    — the user is adjusting a previous request (newer, fewer surveys, more applied)
  compare   — the user asks how two papers or approaches differ
  explain   — the user asks why something was recommended
  chitchat  — greetings, thanks, off-topic; no retrieval needed

Set `topic` to a self-contained search phrase, resolving pronouns from the history.
Leave a constraint null unless the user actually stated it."""

FEWSHOT = """Examples
user: any good work on retrieval augmented generation for legal text?
-> {"intent":"recommend","topic":"retrieval augmented generation for legal documents","constraints":{"k":5}}

user: same thing but only since 2024 and skip the surveys
-> {"intent":"refine","topic":"retrieval augmented generation for legal documents","constraints":{"year_min":2024,"exclude_surveys":true,"k":5}}

user: thanks, that's great
-> {"intent":"chitchat","topic":"","constraints":{}}
"""

_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def route_rules(query: str, history: list[dict[str, str]] | None = None) -> RouterOutput:
    """Deterministic baseline. No model, no network — also the fallback when a
    generation fails validation twice."""
    q = query.lower().strip()
    cons = Constraints()
    if m := _YEAR.search(q):
        year = int(m.group(0))
        if any(w in q for w in ("since", "after", "newer", "recent", "from")):
            cons.year_min = year
        elif any(w in q for w in ("before", "until", "up to")):
            cons.year_max = year
    if any(w in q for w in ("no survey", "skip the survey", "not a survey", "non-survey")):
        cons.exclude_surveys = True
    if m := re.search(r"\b(\d{1,2})\s+(?:papers|results|suggestions)\b", q):
        cons.k = min(int(m.group(1)), 20)

    if not q or any(q.startswith(w) for w in ("hi", "hello", "hey", "thanks", "thank you", "bye")):
        intent = Intent.CHITCHAT
    elif any(w in q for w in ("why did you", "why that", "explain why", "how come you")):
        intent = Intent.EXPLAIN
    elif any(w in q for w in ("difference between", "compare", "versus", " vs ")):
        intent = Intent.COMPARE
    elif any(w in q for w in ("same but", "more recent", "instead", "narrower", "broader", "less ")):
        intent = Intent.REFINE
    else:
        intent = Intent.RECOMMEND

    topic = "" if intent is Intent.CHITCHAT else re.sub(r"[?!.]+$", "", query).strip()
    if intent is Intent.REFINE and history:
        prev = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
        topic = f"{prev} {topic}".strip()
    return RouterOutput(intent=intent, topic=topic, constraints=cons)


def route_llm(llm: LLM, query: str, history: list[dict[str, str]] | None = None) -> RouterOutput:
    hist = "\n".join(f"{h['role']}: {h['content']}" for h in (history or [])[-4:])
    user = f"{FEWSHOT}\n\nConversation so far:\n{hist or '(none)'}\n\nuser: {query}\n->"
    try:
        return llm.structured(user, RouterOutput, system=SYSTEM)
    except Exception:
        # Two failed structured attempts already happened inside `structured`.
        # Degrade to the rules baseline rather than failing the turn.
        return route_rules(query, history)


def make_router(cfg: Config, llm: LLM | None, lora_llm: LLM | None = None):
    """Returns a callable(query, history) -> RouterOutput according to config."""
    mode = cfg.agents.router
    if mode == "rules" or llm is None:
        return route_rules
    if mode == "lora":
        target = lora_llm or llm
        return lambda q, h=None: route_llm(target, q, h)
    return lambda q, h=None: route_llm(llm, q, h)
