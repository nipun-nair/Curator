"""Config-as-code. One YAML per experiment; every run records the config it used.

Reproducibility rule for this repo: nothing that changes a result is allowed to live
in a function default. It lives here, it gets hashed, and the hash goes in the run
directory name.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LLMConfig:
    backend: str = "ollama"          # ollama | openai_compat (vLLM) | hf_local | mock
    model: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0          # 0.0 for every eval run. Non-negotiable.
    max_tokens: int = 768
    timeout_s: float = 60.0
    max_retries: int = 2
    # hf_local only:
    adapter_path: str | None = None   # LoRA adapter dir produced by train/train_router_qlora.py
    load_in_4bit: bool = True


@dataclass
class RetrievalConfig:
    corpus_path: str = "data/corpus.jsonl"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    store: str = "numpy"              # numpy (zero-infra) | qdrant
    qdrant_url: str = "http://localhost:6333"
    collection: str = "curator_papers"
    mode: str = "hybrid"              # hybrid | dense | bm25
    top_k: int = 5
    candidate_k: int = 30             # per-retriever depth before fusion
    rrf_k: int = 60                   # reciprocal-rank-fusion constant
    index_dir: str = "data/index"


@dataclass
class AgentConfig:
    router: str = "llm"               # llm | lora | rules
    use_critic: bool = True
    max_critic_rounds: int = 2        # circuit breaker: the graph cannot loop forever
    max_tool_steps: int = 4           # circuit breaker on the retriever's tool loop
    single_agent: bool = False        # ablation: collapse everything into one prompt
    profile_path: str = "data/profiles.json"


@dataclass
class EvalConfig:
    golden_path: str = "eval/golden.jsonl"
    k: int = 5
    limit: int | None = None
    seed: int = 20260908
    runs_dir: str = "runs"


@dataclass
class Config:
    name: str = "base"
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable short hash of everything that can change a number."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:10]


def _merge(base: Any, overrides: dict[str, Any]) -> Any:
    """Recursively apply a dict of overrides onto a nested dataclass."""
    if not is_dataclass(base):
        return overrides
    valid = {f.name: f for f in fields(base)}
    for key, val in overrides.items():
        if key not in valid:
            raise KeyError(f"unknown config key: {key!r} (valid: {sorted(valid)})")
        current = getattr(base, key)
        if is_dataclass(current) and isinstance(val, dict):
            _merge(current, val)
        else:
            setattr(base, key, val)
    return base


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load `configs/base.yaml`, then an optional experiment YAML, then env, then kwargs.

    An experiment YAML may declare `extends: base.yaml` so ablation files stay to the
    three lines that actually differ from the baseline.
    """
    cfg = Config()
    base = REPO_ROOT / "configs" / "base.yaml"
    if base.exists():
        _merge(cfg, yaml.safe_load(base.read_text()) or {})

    if path is not None:
        p = Path(path)
        if not p.exists():
            p = REPO_ROOT / "configs" / str(path)
        data = yaml.safe_load(p.read_text()) or {}
        data.pop("extends", None)  # base is always loaded first; this is documentation
        _merge(cfg, data)

    # Env escape hatch, for CI and for "run the same config against vLLM".
    if v := os.getenv("CURATOR_LLM_BACKEND"):
        cfg.llm.backend = v
    if v := os.getenv("CURATOR_LLM_MODEL"):
        cfg.llm.model = v
    if v := os.getenv("CURATOR_LLM_BASE_URL"):
        cfg.llm.base_url = v

    if overrides:
        _merge(cfg, overrides)
    return cfg
