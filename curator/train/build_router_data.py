"""Build the router's SFT dataset.

Two sources, mixed:
  templates    combinatorial generation over intents x topics x constraint phrasings.
               Free, exactly labelled, and it covers the slot space evenly — which
               prompting a big model will not do.
  distillation optional: ask the large chat model to label held-out real queries, then
               keep only the ones where its output validates against RouterOutput.

The eval split is generated from disjoint topics so a good score cannot come from
memorising the topic strings.

    python train/build_router_data.py --n 3000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAIN_TOPICS = [
    "retrieval augmented generation", "multi-agent orchestration", "low rank adaptation",
    "vector databases", "speech recognition for low resource languages", "graph neural networks",
    "instruction tuning", "chain of thought prompting", "knowledge distillation",
    "conversational recommendation", "dense passage retrieval", "model quantization",
    "reinforcement learning from human feedback", "code generation benchmarks",
    "federated learning", "named entity recognition", "prompt injection defence",
]
EVAL_TOPICS = [  # deliberately disjoint from TRAIN_TOPICS
    "mixture of experts routing", "clinical text summarisation", "table question answering",
    "speculative decoding", "cross lingual transfer", "agent memory architectures",
]

RECOMMEND = [
    "any good papers on {t}?", "what should I read about {t}", "find me work on {t}",
    "recommend {k} papers about {t}", "I'm looking for research on {t}",
    "show me the key papers in {t}", "got anything on {t}?",
]
REFINE = [
    "same but only since {y}", "narrower please, more applied {t}",
    "more recent ones, nothing before {y}", "less theoretical, and skip the surveys",
    "can you drop the surveys and keep only {y} onwards",
]
COMPARE = [
    "what's the difference between {t} and {t2}", "compare {t} with {t2}",
    "how does {t} stack up against {t2}", "{t} vs {t2}, which is better studied",
]
EXPLAIN = [
    "why did you suggest the second one", "explain why that paper is relevant",
    "how come you picked those", "why is the first one a good fit",
]
CHITCHAT = [
    "hi", "hello there", "thanks, that helps", "thank you", "cheers", "ok bye",
    "what can you do", "who made you",
]

SYSTEM = (
    "You are the router of a research-paper recommender. Classify the intent "
    "(recommend|refine|compare|explain|chitchat) and extract constraints. JSON only."
)


def make_example(rng: random.Random, topics: list[str]) -> dict:
    intent = rng.choices(
        ["recommend", "refine", "compare", "explain", "chitchat"],
        weights=[0.4, 0.2, 0.15, 0.1, 0.15],
    )[0]
    topic = rng.choice(topics)
    other = rng.choice([t for t in topics if t != topic])
    year = rng.choice([2021, 2022, 2023, 2024, 2025])
    k = rng.choice([3, 5, 8, 10])
    cons: dict = {}

    if intent == "recommend":
        text = rng.choice(RECOMMEND).format(t=topic, k=k)
        if "{k}" in "".join(RECOMMEND) and str(k) in text:
            cons["k"] = k
        out_topic = topic
    elif intent == "refine":
        text = rng.choice(REFINE).format(t=topic, y=year)
        if str(year) in text:
            cons["year_min"] = year
        if "survey" in text:
            cons["exclude_surveys"] = True
        out_topic = topic
    elif intent == "compare":
        text = rng.choice(COMPARE).format(t=topic, t2=other)
        out_topic = f"{topic} versus {other}"
    elif intent == "explain":
        text = rng.choice(EXPLAIN)
        out_topic = ""
    else:
        text = rng.choice(CHITCHAT)
        out_topic = ""

    target = {
        "intent": intent,
        "topic": out_topic,
        "constraints": {"k": cons.get("k", 5)} | {k2: v for k2, v in cons.items() if k2 != "k"},
    }
    # TRL's SFTTrainer consumes the `messages` column directly.
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))},
        ]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", default=str(ROOT / "data" / "router"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split, n, topics in (
        ("train", args.n, TRAIN_TOPICS),
        ("eval", args.n_eval, EVAL_TOPICS),
    ):
        seen: set[str] = set()
        rows = []
        while len(rows) < n:
            ex = make_example(rng, topics)
            key = ex["messages"][1]["content"]
            if key in seen:
                continue
            seen.add(key)
            rows.append(ex)
            if len(seen) > n * 20:  # template space exhausted
                break
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"{split}: {len(rows)} examples -> {path}")


if __name__ == "__main__":
    main()
