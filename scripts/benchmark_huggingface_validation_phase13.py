"""Run Phase 13 Marco 7 validation benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from saint.adapters.huggingface_validation import run_hf_phase13_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", default="data/phase13_tiny_corpus.txt")
    parser.add_argument("--out", default="runs/phase13_marco7_validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--saint-lr", type=float, default=1e-3)
    parser.add_argument("--lora-lr", type=float, default=5e-3)
    parser.add_argument("--full-lr", type=float, default=1e-4)
    args = parser.parse_args()

    result = run_hf_phase13_validation(
        args.model,
        args.corpus,
        args.out,
        steps=args.steps,
        budget=args.budget,
        lora_rank=args.lora_rank,
        saint_learning_rate=args.saint_lr,
        lora_learning_rate=args.lora_lr,
        full_learning_rate=args.full_lr,
        device=args.device,
        batch_size=args.batch_size,
    )
    out = Path(args.out)
    print(f"train_examples={result['train_examples']}")
    print(f"validation_examples={result['validation_examples']}")
    print(f"json={out / 'validation_results.json'}")
    print(f"markdown={out / 'validation_results.md'}")


if __name__ == "__main__":
    main()
