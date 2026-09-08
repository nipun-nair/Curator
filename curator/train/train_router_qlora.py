"""QLoRA fine-tune of the router. TRL SFTTrainer + PEFT LoraConfig + 4-bit NF4.

The claim this script exists to support: a 0.6B model with a ~10 MB adapter matches or
beats a 4B model prompted zero-shot at intent + slot extraction, at a fraction of the
latency and memory — so the orchestrator can afford to call it on every turn.

Fits in ~6 GB of VRAM. On CPU it will run but slowly; use --max-steps to smoke-test.

    python train/build_router_data.py --n 3000
    python train/train_router_qlora.py --epochs 2
    python train/eval_router.py --adapter checkpoints/router-qlora

Then score it inside the full system:
    python eval/run_eval.py --config configs/ablations/router_lora.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", default=str(ROOT / "data" / "router"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "router-qlora"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--no-4bit", action="store_true", help="plain LoRA instead of QLoRA")
    args = ap.parse_args()

    # Heavy imports stay inside main so `pip install -e .` without [train] still works.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_dataset(
        "json",
        data_files={
            "train": f"{args.data}/train.jsonl",
            "eval": f"{args.data}/eval.jsonl",
        },
    )
    print({k: len(v) for k, v in data.items()})

    quant = None
    if not args.no_4bit and torch.cuda.is_available():
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # r=16 on attention + MLP projections. Rank is the knob worth ablating: r in
    # {8,16,32} against router accuracy is a two-line change and a real result.
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    sft_config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        completion_only_loss=True,  # train on the JSON, not on the user's question
    )

    trainer = SFTTrainer(
        model=args.model,
        args=sft_config,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        processing_class=AutoTokenizer.from_pretrained(args.model),
        peft_config=peft_config,
        quantization_config=quant,
    )
    trainer.train()
    trainer.save_model(args.out)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"adapter saved to {args.out}")
    print("now: python eval/run_eval.py --config configs/ablations/router_lora.yaml")


if __name__ == "__main__":
    main()
