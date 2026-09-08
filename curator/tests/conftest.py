from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from curator.config import Config, LLMConfig  # noqa: E402
from curator.retrieval.store import HybridIndex, load_corpus  # noqa: E402
from curator.tools import server as tool_server  # noqa: E402
from curator.tools.client import Toolbox  # noqa: E402

FIXTURE_CORPUS = ROOT / "data" / "fixture_corpus.jsonl"


@pytest.fixture(scope="session")
def cfg() -> Config:
    """BM25-only + mock LLM: the whole suite runs with no model download."""
    c = Config()
    c.name = "test"
    c.retrieval.mode = "bm25"
    c.retrieval.corpus_path = str(FIXTURE_CORPUS)
    c.llm = LLMConfig(backend="mock", temperature=0.0, max_retries=0)
    c.agents.router = "rules"
    return c


@pytest.fixture(scope="session")
def index(cfg) -> HybridIndex:
    idx = HybridIndex(cfg.retrieval).build(load_corpus(cfg.retrieval.corpus_path))
    tool_server.set_index(idx)
    return idx


@pytest.fixture
def toolbox(index):
    with Toolbox(tool_server.server) as tb:
        yield tb


@pytest.fixture
def golden() -> list[dict]:
    path = ROOT / "eval" / "golden.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
