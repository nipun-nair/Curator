.PHONY: help setup fixture corpus index chat tools eval eval-all report test lint train-router serve-vllm serve-mcp qdrant clean

PY ?= python
CONFIG ?= configs/base.yaml

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

setup:  ## install the package and dev extras
	$(PY) -m pip install -e ".[dev]"

fixture:  ## regenerate the synthetic corpus + golden set (offline, no network)
	$(PY) scripts/make_fixture_corpus.py

corpus:  ## fetch a real arXiv corpus into data/corpus.jsonl
	$(PY) scripts/fetch_corpus.py --categories cs.CL cs.IR cs.AI --max 4000

index:  ## build the hybrid index
	$(PY) -m curator.cli index --config $(CONFIG)

tools:  ## show what the MCP server advertises
	$(PY) -m curator.cli tools --config $(CONFIG)

serve-mcp:  ## run the tool server standalone (point Claude Desktop or any MCP client at this)
	$(PY) -m curator.tools.server

chat:  ## interactive session
	$(PY) -m curator.cli chat --config $(CONFIG)

eval:  ## score one config
	$(PY) eval/run_eval.py --config $(CONFIG)

eval-all:  ## run every ablation, then regenerate RESULTS.md
	@for c in configs/ablations/*.yaml; do \
		echo "=== $$c"; $(PY) eval/run_eval.py --config $$c || exit 1; \
	done
	$(PY) eval/report.py

report:  ## regenerate RESULTS.md from runs/
	$(PY) eval/report.py

test:  ## offline test suite (mock backend, no GPU, no network)
	$(PY) -m pytest tests/ -q

lint:
	ruff check src tests eval train scripts

train-router:  ## build the SFT set and QLoRA fine-tune the router
	$(PY) train/build_router_data.py --n 3000
	$(PY) train/train_router_qlora.py --epochs 2
	$(PY) train/eval_router.py --adapter checkpoints/router-qlora

serve-vllm:  ## serve the writer model with vLLM (OpenAI-compatible)
	vllm serve Qwen/Qwen3-4B-Instruct --port 8000 --max-model-len 8192

qdrant:  ## start the vector database
	docker compose up -d qdrant

clean:
	rm -rf runs/* data/index checkpoints .pytest_cache
