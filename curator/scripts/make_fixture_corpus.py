"""Generate the offline fixture corpus + golden set.

These are SYNTHETIC records (ids prefixed FIX-) with generic titles and abstracts.
They exist so that `make test` and `make eval` run end-to-end with no network, no
GPU and no API key — and so the eval harness itself can be unit-tested against
known-correct relevance judgements.

They are not real papers and must never be reported as results. For real numbers:
    python scripts/fetch_corpus.py --categories cs.CL cs.IR --max 5000
    python scripts/build_judgments.py            # pooled + manually adjudicated
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Six topic clusters with deliberately disjoint vocabulary, so retrieval quality is
# measurable and a fusion win over either single retriever is visible.
CLUSTERS: dict[str, dict] = {
    "rag": {
        "terms": ["retrieval augmented generation", "grounding", "passage retrieval", "citation"],
        "titles": [
            "Retrieval Augmented Generation for Long Context Question Answering",
            "Grounding Language Model Answers in Retrieved Passages",
            "Citation Faithfulness in Retrieval Augmented Assistants",
            "Chunking Strategies for Passage Retrieval Pipelines",
            "A Survey of Retrieval Augmented Generation Architectures",
        ],
        "cats": ["cs.CL", "cs.IR"],
    },
    "agents": {
        "terms": ["multi-agent", "orchestration", "tool use", "planning", "delegation"],
        "titles": [
            "Multi-Agent Orchestration for Complex Tool Use",
            "Planner Executor Critic Architectures for Language Agents",
            "Delegation Policies in Cooperative Language Agent Teams",
            "Bounded Loops and Circuit Breakers for Autonomous Agents",
            "A Survey of Multi-Agent Language Model Systems",
        ],
        "cats": ["cs.AI", "cs.MA"],
    },
    "peft": {
        "terms": ["low rank adaptation", "parameter efficient", "quantized", "adapter", "fine-tuning"],
        "titles": [
            "Low Rank Adaptation of Quantized Language Models",
            "Parameter Efficient Fine Tuning under Tight Memory Budgets",
            "Adapter Merging for Multi Task Language Models",
            "Rank Selection Heuristics for Low Rank Adapters",
            "Quantization Aware Adapter Training at Four Bits",
        ],
        "cats": ["cs.LG", "cs.CL"],
    },
    "vectordb": {
        "terms": ["vector database", "approximate nearest neighbour", "embedding index", "hnsw"],
        "titles": [
            "Approximate Nearest Neighbour Search for Embedding Indexes",
            "Vector Database Design for Hybrid Sparse Dense Retrieval",
            "Graph Based Indexes for Billion Scale Embedding Search",
            "Payload Filtering in Vector Databases",
            "Reciprocal Rank Fusion of Sparse and Dense Rankings",
        ],
        "cats": ["cs.IR", "cs.DB"],
    },
    "eval": {
        "terms": ["evaluation", "reliability", "reproducibility", "benchmark", "judge"],
        "titles": [
            "Reliability and Variance in Language Model Evaluation",
            "Reproducibility Checklists for Language Model Benchmarks",
            "Model as Judge Agreement with Human Annotators",
            "Measuring Groundedness and Attribution in Generated Answers",
            "A Survey of Evaluation Practice for Generative Systems",
        ],
        "cats": ["cs.CL", "cs.LG"],
    },
    "convrec": {
        "terms": ["conversational recommendation", "preference elicitation", "user simulator", "cold start"],
        "titles": [
            "Conversational Recommendation with Preference Elicitation",
            "User Simulators for Evaluating Conversational Recommenders",
            "Cold Start Strategies in Dialogue Based Recommendation",
            "Explanation Quality in Conversational Recommender Systems",
            "Multi Turn Preference Tracking for Recommendation Dialogue",
        ],
        "cats": ["cs.IR", "cs.CL"],
    },
}

YEARS = [2021, 2022, 2023, 2024, 2025]


def build_corpus() -> list[dict]:
    papers: list[dict] = []
    n = 0
    for cluster, spec in CLUSTERS.items():
        for i, title in enumerate(spec["titles"]):
            n += 1
            terms = ", ".join(spec["terms"])
            abstract = (
                f"SYNTHETIC FIXTURE RECORD. This entry describes work on {terms}. "
                f"We study {title.lower()} and report controlled experiments on public "
                f"benchmarks. Our analysis isolates the contribution of each component and "
                f"discusses failure modes relevant to {spec['terms'][0]}."
            )
            papers.append(
                {
                    "paper_id": f"FIX-{n:04d}",
                    "title": title,
                    "abstract": abstract,
                    "year": YEARS[i % len(YEARS)],
                    "authors": [f"Author {chr(65 + i)}", "Author Z"],
                    "categories": spec["cats"],
                    "url": f"https://example.invalid/fixture/FIX-{n:04d}",
                    "_cluster": cluster,
                }
            )
    return papers


GOLDEN_QUERIES: list[tuple[str, str, str, dict]] = [
    # (query, cluster, intent, constraints)
    ("papers on retrieval augmented generation and grounding answers", "rag", "recommend", {}),
    ("how do people keep citations faithful in RAG systems", "rag", "recommend", {}),
    ("recent work on retrieval augmented generation, nothing before 2024 and no surveys",
     "rag", "recommend", {"year_min": 2024, "exclude_surveys": True}),
    ("multi-agent orchestration and tool use for language agents", "agents", "recommend", {}),
    ("anything on planner critic architectures for agents", "agents", "recommend", {}),
    ("low rank adaptation of quantized models", "peft", "recommend", {}),
    ("parameter efficient fine tuning on a small memory budget", "peft", "recommend", {}),
    ("vector database design and approximate nearest neighbour search", "vectordb", "recommend", {}),
    ("how is sparse and dense retrieval fused", "vectordb", "recommend", {}),
    ("evaluation reliability and reproducibility of generative systems", "eval", "recommend", {}),
    ("measuring groundedness and attribution in generated answers", "eval", "recommend", {}),
    ("conversational recommendation and preference elicitation", "convrec", "recommend", {}),
    ("user simulators for evaluating conversational recommenders", "convrec", "recommend", {}),
    ("same topic but only 2025 papers", "convrec", "refine", {"year_min": 2025}),
    ("what is the difference between LoRA and QLoRA", "peft", "compare", {}),
    ("why did you suggest that first paper", "peft", "explain", {}),
    ("thanks, that was useful", "", "chitchat", {}),
    ("hi there", "", "chitchat", {}),
]


def build_golden(papers: list[dict]) -> list[dict]:
    by_cluster: dict[str, list[dict]] = {}
    for p in papers:
        by_cluster.setdefault(p["_cluster"], []).append(p)

    rows = []
    for i, (query, cluster, intent, cons) in enumerate(GOLDEN_QUERIES, start=1):
        relevant = [p["paper_id"] for p in by_cluster.get(cluster, [])]
        if cons.get("year_min"):
            relevant = [
                p["paper_id"]
                for p in by_cluster.get(cluster, [])
                if (p["year"] or 0) >= cons["year_min"]
            ]
        if cons.get("exclude_surveys"):
            relevant = [
                pid
                for pid in relevant
                if "survey" not in next(p["title"] for p in papers if p["paper_id"] == pid).lower()
            ]
        rows.append(
            {
                "id": f"q{i:03d}",
                "query": query,
                "intent": intent,
                "constraints": cons,
                "relevant_ids": relevant,
                "history": (
                    [{"role": "user", "content": "conversational recommendation and preference elicitation"}]
                    if intent == "refine"
                    else []
                ),
            }
        )
    return rows


def main() -> None:
    papers = build_corpus()
    golden = build_golden(papers)

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "eval").mkdir(exist_ok=True)

    with open(ROOT / "data" / "fixture_corpus.jsonl", "w", encoding="utf-8") as fh:
        for p in papers:
            fh.write(json.dumps({k: v for k, v in p.items() if not k.startswith("_")}) + "\n")
    with open(ROOT / "eval" / "golden.jsonl", "w", encoding="utf-8") as fh:
        for row in golden:
            fh.write(json.dumps(row) + "\n")

    print(f"wrote {len(papers)} fixture papers and {len(golden)} golden queries")


if __name__ == "__main__":
    main()
