"""Score the router in isolation: base vs prompted vs LoRA, on the held-out split.

Isolating this from the end-to-end eval matters. If end-to-end accuracy moves you want
to know whether the router got better or the writer did, and this is the only place
that answers it cleanly.

    python train/eval_router.py --adapter checkpoints/router-qlora
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import metrics as M  # noqa: E402

from curator.agents.router import route_llm, route_rules  # noqa: E402
from curator.config import LLMConfig  # noqa: E402
from curator.llm import build_llm  # noqa: E402
from curator.schemas import RouterOutput  # noqa: E402


def load_split(path: Path) -> list[tuple[str, dict]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        rows.append((msgs[1]["content"], json.loads(msgs[2]["content"])))
    return rows


def score(name: str, predict, rows) -> dict:
    gold_i, pred_i, slots, lat = [], [], [], []
    for text, gold in rows:
        t0 = time.perf_counter()
        try:
            out: RouterOutput = predict(text)
        except Exception:
            continue
        lat.append((time.perf_counter() - t0) * 1000)
        gold_i.append(gold["intent"])
        pred_i.append(out.intent.value)
        slots.append(M.slot_accuracy(gold.get("constraints", {}), out.constraints.model_dump()))
    return {
        "system": name,
        "n": len(gold_i),
        "intent_macro_f1": round(M.macro_f1(gold_i, pred_i), 4),
        "intent_accuracy": round(
            sum(1 for g, p in zip(gold_i, pred_i, strict=True) if g == p) / max(len(gold_i), 1), 4
        ),
        "slot_accuracy": M.aggregate(slots),
        "latency_ms": M.aggregate(lat),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "router" / "eval.jsonl"))
    ap.add_argument("--adapter", default=str(ROOT / "checkpoints" / "router-qlora"))
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompted", default="qwen3:4b", help="ollama model for the prompted baseline")
    ap.add_argument("--skip-prompted", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs" / "router_comparison.json"))
    args = ap.parse_args()

    rows = load_split(Path(args.data))
    print(f"{len(rows)} held-out examples")
    results = [score("rules", lambda t: route_rules(t), rows)]

    if not args.skip_prompted:
        big = build_llm(LLMConfig(backend="ollama", model=args.prompted, max_tokens=200))
        results.append(score(f"prompted:{args.prompted}", lambda t: route_llm(big, t), rows))

    if Path(args.adapter).exists():
        base = build_llm(LLMConfig(backend="hf_local", model=args.base, max_tokens=200))
        tuned = build_llm(
            LLMConfig(backend="hf_local", model=args.base, adapter_path=args.adapter, max_tokens=200)
        )
        results.append(score(f"base:{args.base}", lambda t: route_llm(base, t), rows))
        results.append(score(f"qlora:{args.base}", lambda t: route_llm(tuned, t), rows))
    else:
        print(f"no adapter at {args.adapter}; skipping the LoRA rows")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    for r in results:
        print(
            f"{r['system']:<28} F1 {r['intent_macro_f1']:.3f}  "
            f"slots {r['slot_accuracy']['mean']:.3f}  p50 {r['latency_ms']['median']:.0f} ms"
        )


if __name__ == "__main__":
    main()
