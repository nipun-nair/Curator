"""Metrics. Four families, because "it works" is not a number.

  retrieval   recall@k, nDCG@k, MRR — is the right paper in the set at all?
  routing     macro intent F1, slot accuracy — is the cheap model doing its job?
  generation  groundedness, citation precision, hallucinated-id rate — is the answer
              actually supported by what was retrieved?
  systems     latency, llm calls, tool-call validity, critic-loop rate — what does a
              turn cost, and how often does the reliability machinery fire?

Everything here is pure and deterministic so it can be unit-tested without a model.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from statistics import mean, median


# --------------------------------------------------------------------- retrieval
def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    return len(set(retrieved[:k]) & rel) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k


def mrr(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for i, pid in enumerate(retrieved, start=1):
        if pid in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-gain nDCG. Ideal DCG uses min(|relevant|, k) hits at the top."""
    rel = set(relevant)
    if not rel:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1) for i, pid in enumerate(retrieved[:k], start=1) if pid in rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal else 0.0


# ----------------------------------------------------------------------- routing
def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    scores = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred, strict=True) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p != label)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return mean(scores) if scores else 0.0


def slot_accuracy(gold: dict, pred: dict, keys: Sequence[str] = ("year_min", "year_max", "exclude_surveys")) -> float:
    """Exact match per slot, averaged. Missing gold slot means 'must be absent/false'."""
    hits = 0
    for key in keys:
        want = gold.get(key)
        got = pred.get(key)
        if want in (None, False) and got in (None, False):
            hits += 1
        elif want == got:
            hits += 1
    return hits / len(keys)


# -------------------------------------------------------------------- generation
def groundedness(citations: Iterable[str], evidence_ids: Iterable[str]) -> float:
    """Fraction of cited ids that were actually retrieved this turn."""
    cites = list(citations)
    if not cites:
        return float("nan")
    ev = set(evidence_ids)
    return sum(1 for c in cites if c in ev) / len(cites)


def hallucinated_ids(citations: Iterable[str], evidence_ids: Iterable[str]) -> list[str]:
    ev = set(evidence_ids)
    return sorted({c for c in citations if c not in ev})


def constraint_violations(papers: Iterable[dict], constraints: dict) -> int:
    """Recommendations that break a constraint the user explicitly stated."""
    bad = 0
    for p in papers:
        if constraints.get("year_min") and (p.get("year") or 0) < constraints["year_min"]:
            bad += 1
        elif constraints.get("year_max") and (p.get("year") or 9999) > constraints["year_max"]:
            bad += 1
        elif constraints.get("exclude_surveys") and "survey" in (p.get("title", "")).lower():
            bad += 1
    return bad


# ----------------------------------------------------------------------- systems
def tool_call_validity(results: Sequence[dict]) -> float:
    if not results:
        return float("nan")
    return sum(1 for r in results if r.get("ok")) / len(results)


def aggregate(values: Iterable[float]) -> dict[str, float]:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "median": float("nan"), "n": 0}
    return {"mean": round(mean(vals), 4), "median": round(median(vals), 4), "n": len(vals)}
