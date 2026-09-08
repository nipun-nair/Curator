"""One interface, four deployment paths, plus structured decoding with a repair loop.

    ollama         local dev, `ollama serve`
    openai_compat  vLLM (`vllm serve ...`) or any OpenAI-shaped endpoint
    hf_local       transformers + an optional PEFT LoRA adapter (the tuned router)
    mock           deterministic, offline; what CI and the unit tests run against

Swapping backend is a one-line config change, which is the point: the same eval
harness scores the Ollama build and the vLLM build without touching agent code.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import LLMConfig

T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    """Cheap, backend-agnostic cost proxy. Characters, not tokens, so no tokenizer
    needs to be loaded just to compare two runs."""

    calls: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    latency_ms: float = 0.0


class LLM:
    """Base class. Subclasses implement `_complete`."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.usage = Usage()

    def _complete(self, system: str, user: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def generate(self, user: str, system: str = "") -> str:
        t0 = time.perf_counter()
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                out = self._complete(system, user)
                break
            except Exception as exc:  # transport-level retry with backoff
                last_err = exc
                if attempt == self.cfg.max_retries:
                    raise
                time.sleep(0.5 * (2**attempt))
        else:  # pragma: no cover
            raise RuntimeError(str(last_err))
        self.usage.calls += 1
        self.usage.prompt_chars += len(system) + len(user)
        self.usage.completion_chars += len(out)
        self.usage.latency_ms += (time.perf_counter() - t0) * 1000
        return out

    # ---- structured output ------------------------------------------------
    def structured(self, user: str, schema: type[T], system: str = "") -> T:
        """Generate, parse, validate; on failure show the model its own error once.

        This is the difference between a demo and something you would put on a rota.
        A ValidationError here is *recoverable*; an unparseable second attempt raises
        and the caller falls back to a deterministic default.
        """
        instruction = (
            f"{system}\n\nRespond with JSON only, matching this schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        ).strip()
        raw = self.generate(user, instruction)
        try:
            return schema.model_validate(_extract_json(raw))
        except (ValidationError, ValueError) as exc:
            repair = (
                f"{user}\n\nYour previous answer was invalid.\n"
                f"Previous answer:\n{raw[:1200]}\n\nError:\n{exc}\n\n"
                "Return corrected JSON only."
            )
            raw2 = self.generate(repair, instruction)
            return schema.model_validate(_extract_json(raw2))


def _extract_json(text: str) -> Any:
    """Models wrap JSON in prose and fences. Pull the first balanced object out."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    # strip <think>...</think> blocks emitted by reasoning models
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"unbalanced JSON in model output: {text[:200]!r}")


class OllamaLLM(LLM):
    def _complete(self, system: str, user: str) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        r = httpx.post(
            f"{self.cfg.base_url}/api/chat",
            json={
                "model": self.cfg.model,
                "messages": msgs,
                "stream": False,
                "options": {
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.max_tokens,
                    "seed": 0,
                },
            },
            timeout=self.cfg.timeout_s,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


class OpenAICompatLLM(LLM):
    """vLLM: `vllm serve Qwen/Qwen3-4B-Instruct --port 8000` -> base_url .../v1"""

    def _complete(self, system: str, user: str) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        headers = {"Content-Type": "application/json"}
        if key := os.getenv(self.cfg.api_key_env):
            headers["Authorization"] = f"Bearer {key}"
        r = httpx.post(
            f"{self.cfg.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": self.cfg.model,
                "messages": msgs,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
                "seed": 0,
            },
            timeout=self.cfg.timeout_s,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class HFLocalLLM(LLM):
    """In-process transformers, with an optional LoRA adapter merged on top.

    Used for the fine-tuned router: a 0.6B base + a 12MB adapter beats prompting a
    4B model at intent+slot extraction, at a fraction of the latency.
    """

    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)
        import torch  # noqa: PLC0415 — heavy import, deliberately lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        quant = None
        if cfg.load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model, quantization_config=quant, dtype="auto", device_map="auto"
        )
        if cfg.adapter_path:
            from peft import PeftModel  # noqa: PLC0415

            self.model = PeftModel.from_pretrained(self.model, cfg.adapter_path)
        self.model.eval()

    def _complete(self, system: str, user: str) -> str:
        import torch  # noqa: PLC0415

        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=self.cfg.max_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=max(self.cfg.temperature, 1e-5),
            )
        return self.tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)


class MockLLM(LLM):
    """Deterministic offline backend.

    It is not a stub for its own sake: CI runs the full graph against it, so a broken
    edge, a bad prompt template or a schema drift fails the build without a GPU.
    Rules are matched in order; the first hit wins.
    """

    def __init__(self, cfg: LLMConfig, rules: list[tuple[str, str]] | None = None):
        super().__init__(cfg)
        self.rules: list[tuple[str, str]] = rules or []
        self.transcript: list[tuple[str, str]] = []

    def _complete(self, system: str, user: str) -> str:
        self.transcript.append((system, user))
        blob = f"{system}\n{user}"
        for pattern, response in self.rules:
            if re.search(pattern, blob, flags=re.IGNORECASE | re.DOTALL):
                return response
        return "{}"


def build_llm(cfg: LLMConfig) -> LLM:
    match cfg.backend:
        case "ollama":
            return OllamaLLM(cfg)
        case "openai_compat" | "vllm":
            return OpenAICompatLLM(cfg)
        case "hf_local":
            return HFLocalLLM(cfg)
        case "mock":
            return MockLLM(cfg)
        case _:
            raise ValueError(f"unknown llm backend: {cfg.backend}")
