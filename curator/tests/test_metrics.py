"""Metrics are the thing every claim rests on, so they get tested against hand
computed values rather than against themselves."""

from __future__ import annotations

import math

import metrics as M


def test_recall_and_precision():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = ["b", "e", "z"]
    assert M.recall_at_k(retrieved, relevant, 5) == 2 / 3
    assert M.recall_at_k(retrieved, relevant, 2) == 1 / 3
    assert M.precision_at_k(retrieved, relevant, 5) == 2 / 5


def test_mrr_first_hit_only():
    assert M.mrr(["x", "y", "a"], ["a", "y"]) == 0.5
    assert M.mrr(["x"], ["a"]) == 0.0


def test_ndcg_perfect_and_reversed():
    rel = ["a", "b"]
    assert M.ndcg_at_k(["a", "b", "c"], rel, 3) == 1.0
    worse = M.ndcg_at_k(["c", "a", "b"], rel, 3)
    assert 0.0 < worse < 1.0


def test_ndcg_matches_hand_computation():
    # one hit at rank 2 -> DCG = 1/log2(3); IDCG (1 relevant) = 1/log2(2) = 1
    got = M.ndcg_at_k(["x", "a", "y"], ["a"], 3)
    assert math.isclose(got, 1 / math.log2(3), rel_tol=1e-9)


def test_macro_f1_is_not_accuracy():
    gold = ["recommend"] * 8 + ["chitchat"] * 2
    always_majority = ["recommend"] * 10
    # 80% accurate, but the minority class scores zero -> macro F1 must be well below.
    assert M.macro_f1(gold, always_majority) < 0.5


def test_slot_accuracy_treats_absent_as_false():
    assert M.slot_accuracy({}, {"year_min": None, "exclude_surveys": False}) == 1.0
    assert M.slot_accuracy({"year_min": 2024}, {"year_min": 2024}) == 1.0
    assert M.slot_accuracy({"year_min": 2024}, {"year_min": 2020}) < 1.0


def test_groundedness_and_hallucination():
    assert M.groundedness(["a", "b"], ["a", "b", "c"]) == 1.0
    assert M.groundedness(["a", "zz"], ["a"]) == 0.5
    assert M.hallucinated_ids(["a", "zz"], ["a"]) == ["zz"]
    assert math.isnan(M.groundedness([], ["a"]))


def test_constraint_violations():
    papers = [
        {"paper_id": "1", "title": "A Survey of Things", "year": 2025},
        {"paper_id": "2", "title": "Real Work", "year": 2020},
    ]
    assert M.constraint_violations(papers, {"exclude_surveys": True}) == 1
    assert M.constraint_violations(papers, {"year_min": 2024}) == 1
    assert M.constraint_violations(papers, {}) == 0


def test_aggregate_ignores_nan():
    out = M.aggregate([1.0, 3.0, float("nan")])
    assert out["n"] == 2 and out["mean"] == 2.0
