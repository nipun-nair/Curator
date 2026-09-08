# Curator — project spec

A portfolio project sized for two weekends that puts every bullet in the job ad on the
same table, in a way a reviewer can verify in ten minutes.

---

## 1. The idea in one paragraph

A conversational recommender for research papers. You ask in plain language across
several turns ("work on RAG for legal text" → "same but only since 2024, no surveys"),
and a small team of agents routes the request, retrieves evidence through an MCP tool
server, verifies the evidence before anything is written, and answers with citations
that are checked against what was actually retrieved. Around it sits an evaluation
harness that scores retrieval, routing, grounding and cost, and a set of ablations that
answer *which part of the machinery is earning its keep*.

**Why this domain.** It gives every bullet somewhere honest to live without contrivance,
the corpus is free and legally clean (arXiv API), relevance is judgeable by one person in
an afternoon, and it is close enough to conversational recommendation that the research
framing is genuine rather than decorative.

---

## 2. Bullet-to-component map

This is the table to put at the top of the README's "what this demonstrates" if a
reviewer is skimming.

| Job ad bullet | Where it lives | What makes it non-trivial |
|---|---|---|
| Design and orchestration of AI agents or multi-agent systems | `src/curator/graph.py`, `agents/` | Four agents with distinct contracts and one **backward** edge — a critic that can re-open retrieval, bounded at 2 rounds. Not a linear pipeline wearing a costume. |
| Agent frameworks (LangGraph, LangChain, AutoGen, CrewAI) | LangGraph `StateGraph`, typed state, conditional edges, `recursion_limit` | The cycle is a first-class edge, so it is inspectable and bounded rather than a hidden `while` loop. |
| Tool integration, function calling, tool calling, MCP | `tools/server.py`, `tools/client.py` | A real MCP server (mcp 2.x `MCPServer`), tools discovered at runtime, the retriever's prompt **generated** from that discovery, arguments validated against the advertised JSON schema before dispatch. |
| LLMs, RAG, embeddings, vector databases | `retrieval/store.py` | Hybrid BM25 ⊕ dense with Reciprocal Rank Fusion, bge-small embeddings, Qdrant or a zero-infra numpy store — and an ablation that says whether fusion actually beat either half. |
| Deployment via Ollama, vLLM, Hugging Face | `llm.py` | One interface, four backends, switched by config. The same eval runs against Ollama and vLLM, which is the only way to claim the numbers are a property of the system. |
| Model adaptation via LoRA / QLoRA | `train/` | A 0.6B router with a ~10 MB QLoRA adapter, trained on a combinatorially generated SFT set with a **topic-disjoint** eval split, scored against prompting a 4B model. |
| Evaluation, reliability, reproducibility | `eval/`, `.github/workflows/ci.yml` | Four metric families, immutable run dirs with git SHA and package versions, seeded and greedy, generated results table, and CI that fails on a grounding regression. |

**The one thing that ties it together:** every bullet is answered by a *measurement*, not
an assertion. The LoRA router is not "I fine-tuned a model" — it is a row in a table next
to the prompted baseline, with latency alongside accuracy.

---

## 3. What the reviewer sees in ten minutes

Order matters. Optimise the repo for this sequence:

1. `README.md` — architecture diagram, then the ablation table.
2. `make test` — 26 tests, no GPU, no network, no API key. Green in under two seconds.
3. `make eval CONFIG=configs/ablations/ci_mock.yaml` — the full graph end to end,
   deterministic, writes a run directory.
4. `RESULTS.md` — generated, with a git SHA on it.
5. `src/curator/graph.py` — 150 lines, the cycle visible at a glance.

If any of those five steps needs a caveat from you to work, fix that before adding a
feature. A repo that runs on a stranger's laptop beats a cleverer one that does not.

---

## 4. Build plan

### Weekend 1 — the spine (works end to end, unimpressively)

**Saturday**

| block | work | done when |
|---|---|---|
| 1 | Scaffold, `schemas.py`, `config.py`. Decide the contracts *first*. | `Config().fingerprint()` returns a hash |
| 2 | `scripts/fetch_corpus.py` → 3–4k arXiv abstracts | `make corpus-stats` prints a sane year and category spread |
| 3 | `retrieval/store.py`: BM25, then dense, then RRF | a search for a topic you know returns the paper you were thinking of |
| 4 | `tools/server.py`: the MCP server + `make tools` | five tools listed with schemas |

**Sunday**

| block | work | done when |
|---|---|---|
| 5 | `llm.py` with `ollama` + `mock` | `MockLLM` drives a fake turn |
| 6 | Router (rules first, then prompted) | intents correct on 10 hand-written turns |
| 7 | Retriever tool loop + `graph.py` (no critic yet) | `curator ask "..."` answers with real ids |
| 8 | `eval/metrics.py` + `run_eval.py`, golden set v1 (~20 queries) | one run directory with recall@5 in it |

End of Weekend 1 you have a *measurable baseline*. That is the milestone; polish is not.

### Weekend 2 — the parts that are actually the portfolio

**Saturday**

| block | work | done when |
|---|---|---|
| 9 | Critic + the backward edge + bounds | a turn shows `critic_rounds: 2` and still terminates |
| 10 | `enforce_grounding`, hallucinated-id metric | the metric is non-zero for at least one config (find out where it breaks) |
| 11 | Ablation configs + `report.py` | `RESULTS.md` has five rows |
| 12 | vLLM backend, re-run one config against it | two runs, same golden set, different serving stack |

**Sunday**

| block | work | done when |
|---|---|---|
| 13 | `build_router_data.py` (3k examples, disjoint eval topics) | train/eval splits on disk |
| 14 | `train_router_qlora.py` — 20–40 min on a Colab T4 or a 3060 | adapter in `checkpoints/` |
| 15 | `eval_router.py`: rules vs prompted-4B vs base-0.6B vs tuned-0.6B | four rows with F1 **and** p50 latency |
| 16 | CI, README, honest-limitations section | green badge, generated results table |

**Cut list, in order, if a weekend goes sideways.** Qdrant (the numpy store is
equivalent for 4k papers) → the vLLM run → the `compare`/`explain` intents → the profile
tools. Do **not** cut: the critic loop, the grounding check, the eval harness, or the
LoRA router. Those four are the project.

---

## 5. Design decisions worth defending in an interview

Have an answer ready for each. These are the questions a good interviewer asks.

**"Why multi-agent? A single prompt would do this."** Often true, which is why the
`single_agent` ablation exists and why you should report it even when it is close. The
defensible claim is narrow: separating retrieval from verification lets you bound and
measure each, and it is the critic that moves the constraint-violation rate — not the
number of agents. If the ablation shows no gain, say that; a candidate who measured and
reported a null result is more credible than one with an unfalsifiable architecture.

**"Why MCP instead of plain functions?"** Three reasons, in order of honesty: the tool
surface becomes a versioned contract with a schema the agent discovers rather than
imports; the same server is reusable by any MCP client, so the retrieval layer outlives
this repo; and it forces a real failure boundary, so tool errors are counted data rather
than exceptions unwinding through agent code. The cost is a process boundary and some
async plumbing — say that too.

**"Why fine-tune a router rather than prompt?"** Latency and cost per turn, at equal or
better accuracy on a closed label set with structured output. That is exactly the shape
of task where a small tuned model wins, and exactly the claim your `eval_router.py` table
either supports or refutes.

**"How do you know it is not hallucinating?"** You do not, in general — you *bound* it.
Citations are checked against retrieved ids, the unresolvable ones are dropped and
counted, and CI fails when that rate rises. That is a weaker and much more defensible
claim than "it does not hallucinate".

**"What would you do with another month?"** A cross-family critic (a critic from the same
model family shares the writer's blind spots); a user simulator for multi-turn eval
instead of single-turn golden queries; and human agreement numbers on the relevance
judgements, since every retrieval metric inherits their noise.

---

## 6. Where the honest limits are

Write these in the README yourself before a reviewer finds them.

- The committed fixture corpus is synthetic. Numbers on it are wiring checks.
- Relevance judgements are pooled and single-annotator; report your own re-judge
  agreement.
- Single-turn golden queries under-test the thing the system is *for* (multi-turn
  refinement). The `refine` examples with history are a gesture at this, not a solution.
- Model-as-judge critic shares a family with the writer.
- ~4k papers is a toy index; nothing here says anything about ANN behaviour at scale, and
  the Qdrant path exists to show the integration, not to prove throughput.

A limitations section that names the real weaknesses reads as competence. One that says
"future work: more data" reads as filler.

---

## 7. Stack, pinned

Verified against PyPI on 2026-09-08.

| package | version | note |
|---|---|---|
| `langgraph` | 1.2.11 | `StateGraph`, conditional edges, `recursion_limit` |
| `mcp` | 2.2.0 | **2.x renamed `FastMCP` → `MCPServer`**; fields are snake_case (`input_schema`, `is_error`). 1.x snippets on the web will not run. |
| `sentence-transformers` | 6.0.1 | `BAAI/bge-small-en-v1.5`, query-side instruction prefix |
| `qdrant-client` | 1.19.0 | optional |
| `trl` | 1.12.0 | `SFTTrainer(model=…, peft_config=…, quantization_config=…)` |
| `peft` | 0.20.0 | `LoraConfig`, r=16, α=32 |
| `transformers` | 5.16.1 | `BitsAndBytesConfig` NF4 |
| `vllm` | 0.28.0 | OpenAI-compatible server |

Hardware: everything except the fine-tune runs on a laptop CPU. The QLoRA run wants
~6 GB VRAM — a free Colab T4 is enough for a 0.6B base at r=16.
