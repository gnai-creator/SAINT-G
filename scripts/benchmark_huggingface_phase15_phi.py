"""Benchmark SAINT Phi deltas: Delta W = A Phi B."""

from __future__ import annotations

import argparse
from copy import copy
from json import dumps
from pathlib import Path
from typing import Any

from benchmark_huggingface_phase15_compare import (
    _ints,
    _items,
    _lora_rank,
    _memory_items,
    _row_from_saint,
    _run_saint_subprocess,
)


MARCO12_GAIN = 6.233305e-04


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _gain(row: dict[str, Any]) -> float:
    return float(row.get("validation_gain_per_parameter") or 0.0)


def _delta(row: dict[str, Any]) -> float | None:
    value = row.get("validation_loss_delta")
    return float(value) if value is not None else None


def _case_args(args, *, target: str, budget: int, variant: str, memory: str):
    namespace = copy(args)
    namespace.target_names = target
    namespace.budget = budget
    namespace.routing_method = "activation_phi_validation_rerank"
    namespace.phi_variant = variant
    namespace.out = str(
        Path(args.out) / f"phi_{variant}_b{budget}_{_safe(target)}_{_safe(memory)}"
    )
    return namespace


def _run_phi(args, *, target: str, budget: int, variant: str, memory: str):
    values = _case_args(args, target=target, budget=budget, variant=variant, memory=memory)
    result = _run_saint_subprocess(values, budget=budget, max_memory=memory)
    row = _row_from_saint(result, budget=budget, max_memory=memory)
    row.update(
        {
            "method": "saint_phi_delta",
            "target": target,
            "phi_rank": args.phi_rank,
            "phi_variant": variant,
            "baseline_marco12_gain_per_parameter": MARCO12_GAIN,
            "beats_marco12_gain_per_parameter": _gain(row) >= MARCO12_GAIN,
        }
    )
    return row


def _lora(args, targets: list[str]) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        for rank in _ints(args.lora_ranks):
            try:
                namespace = copy(args)
                namespace.target_names = target
                namespace.out = str(Path(args.out) / f"lora_{_safe(target)}_r{rank}")
                row = _lora_rank(namespace, rank=rank)
                row["target"] = target
                rows.append(row)
            except Exception as exc:  # pragma: no cover - large-model diagnostic.
                rows.append(
                    {
                        "method": f"lora_rank{rank}_train_only",
                        "target": target,
                        "rank": rank,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    return rows


def _best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("method") == "saint_phi_delta" and row.get("status") == "ok"
    ]
    return max(candidates, key=_gain) if candidates else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = _best(rows)
    lora_gain = max(
        [
            _gain(row)
            for row in rows
            if str(row.get("method", "")).startswith("lora_rank")
            and row.get("status") == "ok"
        ]
        or [0.0]
    )
    return {
        "best_phi": best,
        "best_phi_validation_gain_per_parameter": _gain(best or {}),
        "best_lora_validation_gain_per_parameter": lora_gain,
        "beats_marco12": bool(best and _gain(best) >= MARCO12_GAIN),
        "criterion": "compare Phi against Marco 12 layer 1 v_proj budget 32",
    }


def _markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 15 Marco 13 Phi Sweep",
        "",
        "| method | target | budget | phi | val delta | val gain/param | params | status |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {target} | {budget} | {phi} | {delta} | {gain} | {params} | {status} |".format(
                method=row.get("method", ""),
                target=row.get("target", ""),
                budget="" if row.get("budget") is None else row.get("budget"),
                phi=row.get("phi_variant", ""),
                delta="" if _delta(row) is None else f"{_delta(row):.6f}",
                gain=f"{_gain(row):.6e}",
                params="" if row.get("parameter_count") is None else row.get("parameter_count"),
                status=row.get("status", ""),
            )
        )
    lines.extend(["", "```json", dumps(summary, indent=2), "```"])
    return "\n".join(lines) + "\n"


def run(args) -> dict[str, Any]:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    targets = _items(args.target_names)
    for memory in _memory_items(args.max_memories):
        for target in targets:
            for budget in _ints(args.budgets):
                for variant in _csv(args.phi_variants):
                    rows.append(
                        _run_phi(
                            args,
                            target=target,
                            budget=budget,
                            variant=variant,
                            memory=memory,
                        )
                    )
    if not args.skip_lora:
        rows.extend(_lora(args, targets))
    summary = _summary(rows)
    result = {"model": args.model, "rows": rows, "summary": summary}
    (root / "phase15_phi_results.json").write_text(dumps(result, indent=2), encoding="utf-8")
    (root / "phase15_phi_results.md").write_text(_markdown(rows, summary), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", default="data/tinyshakespeare_phase13.txt")
    parser.add_argument("--out", default="runs/phase15_marco13_phi")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--budgets", default="32")
    parser.add_argument("--max-memories", default="0=14GiB,cpu=64GiB")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-texts", type=int, default=3)
    parser.add_argument("--validation-texts", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--routing-method", default="activation_phi_validation_rerank")
    parser.add_argument("--routing-max-length", type=int, default=4)
    parser.add_argument("--routing-batch-size", type=int, default=1)
    parser.add_argument("--routing-block-size", type=int, default=4)
    parser.add_argument("--target-names", default="model.layers.1.self_attn.v_proj.weight")
    parser.add_argument("--target-device", default="cuda")
    parser.add_argument("--max-cuda-gb", type=float, default=23.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--validate-during-train", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--validation-rerank-multiplier", type=int, default=4)
    parser.add_argument("--validation-rerank-chunk-size", type=int, default=256)
    parser.add_argument("--validation-probe-epsilon", type=float, default=1e-3)
    parser.add_argument("--validation-rerank-max-candidates", type=int, default=32)
    parser.add_argument("--validation-rerank-batch-size", type=int, default=8)
    parser.add_argument("--structured-prototype-count", type=int, default=1)
    parser.add_argument("--structured-prototype-mode", default="weight_sign")
    parser.add_argument("--structured-scale-granularity", default="block")
    parser.add_argument("--phi-rank", type=int, default=4)
    parser.add_argument(
        "--phi-variants",
        default="dense,diagonal,upper_triangular,block_diagonal,kronecker,hadamard,codebook_2x2,codebook_4x4",
    )
    parser.add_argument("--phi-variant", default="dense")
    parser.add_argument("--hf-device-map", default="auto")
    parser.add_argument("--lora-max-memory", default="0=14GiB,cpu=64GiB")
    parser.add_argument("--lora-learning-rate", type=float, default=0.001)
    parser.add_argument("--lora-ranks", default="1")
    parser.add_argument("--lora-b-init-scale", type=float, default=0.0)
    parser.add_argument("--skip-lora", action="store_true")
    args = parser.parse_args()
    print(dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
