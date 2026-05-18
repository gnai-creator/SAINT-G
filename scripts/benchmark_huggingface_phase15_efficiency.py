"""Phase 15 efficiency sweep for extreme-compression SAINT runs."""

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


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _label(value: str) -> str:
    safe = []
    for char in value:
        safe.append(char if char.isalnum() else "_")
    return "".join(safe).strip("_")


def _gain(row: dict[str, Any]) -> float:
    return float(row.get("validation_gain_per_parameter") or 0.0)


def _validation_delta(row: dict[str, Any]) -> float | None:
    value = row.get("validation_loss_delta")
    return float(value) if value is not None else None


def _with_case(args, **updates):
    namespace = copy(args)
    for key, value in updates.items():
        setattr(namespace, key, value)
    return namespace


def _saint_case(args, *, target: str, budget: int, mode: str, granularity: str, memory: str):
    case_label = "_".join(
        [
            f"b{budget}",
            f"t{_label(target)}",
            f"pm{mode}",
            f"sg{granularity}",
            _label(memory),
        ]
    )
    case_args = _with_case(
        args,
        target_names=target,
        structured_prototype_mode=mode,
        structured_scale_granularity=granularity,
        out=str(Path(args.out) / case_label),
    )
    result = _run_saint_subprocess(case_args, budget=budget, max_memory=memory)
    row = _row_from_saint(result, budget=budget, max_memory=memory)
    row.update(
        {
            "target": target,
            "structured_prototype_mode": mode,
            "structured_scale_granularity": granularity,
            "structured_prototype_count": args.structured_prototype_count,
        }
    )
    return row


def _lora_rows(args, targets: list[str]) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        for rank in _ints(args.lora_ranks):
            try:
                namespace = _with_case(
                    args,
                    target_names=target,
                    out=str(Path(args.out) / f"lora_{_label(target)}_r{rank}"),
                )
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


def _annotate_equivalent_lora(rows: list[dict[str, Any]]) -> None:
    lora_by_target = {
        row.get("target"): row
        for row in rows
        if str(row.get("method", "")).startswith("lora_rank")
        and row.get("status") == "ok"
    }
    for row in rows:
        if row.get("method") != "saint_train_only" or row.get("status") != "ok":
            continue
        lora = lora_by_target.get(row.get("target"))
        if not lora:
            continue
        params = int(row.get("parameter_count") or 0)
        lora_gain = _gain(lora)
        estimate = -lora_gain * params
        row["lora_equivalent_parameter_validation_delta_estimate"] = estimate
        row["lora_equivalent_parameter_gain_per_parameter"] = lora_gain
        delta = _validation_delta(row)
        row["beats_lora_equivalent_parameter_estimate"] = (
            delta is not None and delta <= estimate
        )


def _best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("method") == "saint_train_only"
        and row.get("status") == "ok"
        and row.get("validation_loss_delta") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=_gain)


def _phase_decision(rows: list[dict[str, Any]], best: dict[str, Any] | None) -> dict[str, Any]:
    if not best:
        return {"passed_efficiency_gate": False, "reason": "no successful SAINT row"}
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
        "passed_efficiency_gate": _gain(best) > 0.0 and _gain(best) >= lora_gain,
        "best_saint_validation_gain_per_parameter": _gain(best),
        "best_lora_validation_gain_per_parameter": lora_gain,
        "criterion": "efficiency per parameter, not absolute LoRA validation loss",
    }


def _markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 15 Marco 12 Efficiency Sweep",
        "",
        "| method | target | budget | mode | scale | val delta | val gain/param | params | status |",
        "|---|---|---:|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {target} | {budget} | {mode} | {scale} | {delta} | {gain} | {params} | {status} |".format(
                method=row.get("method", ""),
                target=row.get("target", ""),
                budget="" if row.get("budget") is None else row.get("budget"),
                mode=row.get("structured_prototype_mode", ""),
                scale=row.get("structured_scale_granularity", ""),
                delta="" if _validation_delta(row) is None else f"{_validation_delta(row):.6f}",
                gain=f"{_gain(row):.6e}",
                params="" if row.get("parameter_count") is None else row.get("parameter_count"),
                status=row.get("status", ""),
            )
        )
    lines.extend(["", "## Summary", "", "```json", dumps(summary, indent=2), "```"])
    return "\n".join(lines) + "\n"


def run(args) -> dict[str, Any]:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    targets = _items(args.target_names)
    for memory in _memory_items(args.max_memories):
        for target in targets:
            for budget in _ints(args.budgets):
                for mode in _csv(args.structured_prototype_modes):
                    for granularity in _csv(args.structured_scale_granularities):
                        rows.append(
                            _saint_case(
                                args,
                                target=target,
                                budget=budget,
                                mode=mode,
                                granularity=granularity,
                                memory=memory,
                            )
                        )
    if not args.skip_lora:
        rows.extend(_lora_rows(args, targets))
    _annotate_equivalent_lora(rows)
    best = _best(rows)
    summary = {
        "best_saint": best,
        "phase15_decision": _phase_decision(rows, best),
    }
    result = {"model": args.model, "rows": rows, "summary": summary}
    (root / "phase15_efficiency_results.json").write_text(
        dumps(result, indent=2),
        encoding="utf-8",
    )
    (root / "phase15_efficiency_results.md").write_text(
        _markdown(rows, summary),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", default="data/tinyshakespeare_phase13.txt")
    parser.add_argument("--out", default="runs/phase15_marco12_efficiency")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--budgets", default="16,32,64")
    parser.add_argument("--max-memories", default="0=14GiB,cpu=64GiB")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-texts", type=int, default=3)
    parser.add_argument("--validation-texts", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--routing-method", default="activation_structured_block_validation_rerank")
    parser.add_argument("--routing-max-length", type=int, default=4)
    parser.add_argument("--routing-batch-size", type=int, default=1)
    parser.add_argument("--routing-block-size", type=int, default=4)
    parser.add_argument("--target-names", default="model.layers.2.self_attn.v_proj.weight")
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
    parser.add_argument("--structured-prototype-count", type=int, default=2)
    parser.add_argument("--structured-prototype-modes", default="activation,weight_value")
    parser.add_argument("--structured-scale-granularities", default="block,row,col")
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
