"""The evaluation harness. One command, one immutable run directory, one metrics.json.

Reproducibility contract for every run:
  * all seeds set, temperature 0, greedy decoding
  * the full resolved config is written next to the results, plus its fingerprint
  * git SHA + dirty flag + installed versions of every dependency that touches a number
  * per-example records are kept, not just aggregates, so a regression is diffable

    python eval/run_eval.py --config configs/ablations/no_critic.yaml
    python eval/run_eval.py --config configs/ablations/ci_mock.yaml --limit 6
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import metrics as M  # noqa: E402  (eval/metrics.py)

from curator.config import load_config  # noqa: E402
from curator.graph import build_curator  # noqa: E402
from curator.retrieval.store import HybridIndex, load_corpus  # noqa: E402
from curator.tools import server as tool_server  # noqa: E402
from curator.tools.client import Toolbox  # noqa: E402

TRACKED = [
    "langgraph", "langchain-core", "mcp", "sentence-transformers",
    "rank-bm25", "qdrant-client", "numpy", "pydantic",
]


def provenance() -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except Exception:
            return "unavailable"

    versions = {}
    for pkg in TRACKED:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not-installed"
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def set_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def load_golden(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def evaluate(config_path: str | None, limit: int | None, corpus: str | None, tag: str) -> Path:
    cfg = load_config(config_path)
    if corpus:
        cfg.retrieval.corpus_path = corpus
    if limit:
        cfg.eval.limit = limit
    set_seeds(cfg.eval.seed)

    golden = load_golden(ROOT / cfg.eval.golden_path, cfg.eval.limit)

    # Index in-process and inject it, so the tool server and the harness are provably
    # looking at the same corpus.
    index = HybridIndex(cfg.retrieval).build(load_corpus(cfg.retrieval.corpus_path))
    tool_server.set_index(index)

    records: list[dict] = []
    t_start = time.perf_counter()
    with Toolbox(tool_server.server) as toolbox:
        curator = build_curator(cfg, toolbox)
        for row in golden:
            try:
                answer, trace = curator.answer(row["query"], history=row.get("history", []))
                err = None
            except Exception as exc:  # one bad example must not void the run
                answer, trace, err = None, None, f"{type(exc).__name__}: {exc}"

            rec = {"id": row["id"], "query": row["query"], "error": err}
            if answer is not None and trace is not None:
                rec_ids = [r.paper_id for r in answer.recommendations]
                cites = [c for r in answer.recommendations for c in r.citations]
                papers = [
                    index.by_id[p].model_dump() for p in rec_ids if p in index.by_id
                ]
                rec |= {
                    "predicted_intent": trace.router.intent.value if trace.router else None,
                    "gold_intent": row.get("intent"),
                    "predicted_constraints": trace.router.constraints.model_dump() if trace.router else {},
                    "gold_constraints": row.get("constraints", {}),
                    "retrieved_ids": trace.evidence_ids,
                    "recommended_ids": rec_ids,
                    "relevant_ids": row.get("relevant_ids", []),
                    "recall@k": M.recall_at_k(trace.evidence_ids, row.get("relevant_ids", []), cfg.eval.k),
                    "precision@k": M.precision_at_k(rec_ids, row.get("relevant_ids", []), cfg.eval.k),
                    "ndcg@k": M.ndcg_at_k(trace.evidence_ids, row.get("relevant_ids", []), cfg.eval.k),
                    "mrr": M.mrr(trace.evidence_ids, row.get("relevant_ids", [])),
                    "groundedness": M.groundedness(cites, trace.evidence_ids),
                    "hallucinated_ids": M.hallucinated_ids(cites, trace.evidence_ids),
                    "constraint_violations": M.constraint_violations(papers, row.get("constraints", {})),
                    "slot_accuracy": M.slot_accuracy(
                        row.get("constraints", {}),
                        trace.router.constraints.model_dump() if trace.router else {},
                    ),
                    "tool_calls": len(trace.tool_calls),
                    "tool_validity": M.tool_call_validity([r.model_dump() for r in trace.tool_results]),
                    "critic_rounds": trace.critic_rounds,
                    "latency_ms": trace.latency_ms,
                    "llm_calls": trace.llm_calls,
                    "completion_chars": trace.completion_chars,
                    "reply": answer.reply,
                }
            records.append(rec)

    ok = [r for r in records if not r.get("error")]
    summary = {
        "config": cfg.name,
        "fingerprint": cfg.fingerprint(),
        "n": len(records),
        "errors": len(records) - len(ok),
        "wall_clock_s": round(time.perf_counter() - t_start, 2),
        "retrieval": {
            f"recall@{cfg.eval.k}": M.aggregate(r["recall@k"] for r in ok),
            f"ndcg@{cfg.eval.k}": M.aggregate(r["ndcg@k"] for r in ok),
            "mrr": M.aggregate(r["mrr"] for r in ok),
        },
        "routing": {
            "intent_macro_f1": round(
                M.macro_f1(
                    [r["gold_intent"] for r in ok if r.get("gold_intent")],
                    [r["predicted_intent"] or "none" for r in ok if r.get("gold_intent")],
                ),
                4,
            ),
            "slot_accuracy": M.aggregate(r["slot_accuracy"] for r in ok),
        },
        "generation": {
            "groundedness": M.aggregate(r["groundedness"] for r in ok),
            "hallucinated_id_rate": round(
                sum(1 for r in ok if r["hallucinated_ids"]) / max(len(ok), 1), 4
            ),
            "constraint_violation_rate": round(
                sum(1 for r in ok if r["constraint_violations"]) / max(len(ok), 1), 4
            ),
        },
        "systems": {
            "latency_ms": M.aggregate(r["latency_ms"] for r in ok),
            "llm_calls": M.aggregate(r["llm_calls"] for r in ok),
            "tool_calls": M.aggregate(r["tool_calls"] for r in ok),
            "tool_validity": M.aggregate(r["tool_validity"] for r in ok),
            "critic_loop_rate": round(
                sum(1 for r in ok if r["critic_rounds"] > 1) / max(len(ok), 1), 4
            ),
        },
        "provenance": provenance(),
    }

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / cfg.eval.runs_dir / f"{stamp}-{tag or cfg.name}-{cfg.fingerprint()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    with open(run_dir / "records.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")

    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2))
    print(f"\nrun written to {run_dir.relative_to(ROOT)}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--corpus", default=None, help="override corpus path")
    ap.add_argument("--tag", default="", help="label for the run directory")
    args = ap.parse_args()
    evaluate(args.config, args.limit, args.corpus, args.tag)


if __name__ == "__main__":
    main()
