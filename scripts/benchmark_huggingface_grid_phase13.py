"""Run Phase 13 Marco 8 hyperparameter grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from saint.adapters.huggingface_grid import run_hf_phase13_grid


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", default="data/phase13_tiny_corpus.txt")
    parser.add_argument("--out", default="runs/phase13_marco8_grid")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--saint-budgets", default="8,16")
    parser.add_argument("--saint-lrs", default="0.001,0.005")
    parser.add_argument("--lora-ranks", default="2,4")
    parser.add_argument("--lora-lrs", default="0.001,0.005")
    args = parser.parse_args()

    result = run_hf_phase13_grid(
        args.model,
        args.corpus,
        args.out,
        steps=args.steps,
        saint_budgets=_ints(args.saint_budgets),
        saint_lrs=_floats(args.saint_lrs),
        lora_ranks=_ints(args.lora_ranks),
        lora_lrs=_floats(args.lora_lrs),
        device=args.device,
        batch_size=args.batch_size,
    )
    out = Path(args.out)
    print(f"rows={len(result['rows'])}")
    print(f"json={out / 'grid_results.json'}")
    print(f"markdown={out / 'grid_results.md'}")


if __name__ == "__main__":
    main()
