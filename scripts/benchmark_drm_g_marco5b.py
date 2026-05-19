"""DRM-G Marco 5B retention benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from saint.config import RuntimeConfig
from saint.runtime.runner import train_runtime


LINEAR_SEQUENCE = [
    {"target_module": "blocks.1.attn.out_proj", "projection_init": "gradient"},
    {"target_module": "blocks.2.attn.out_proj", "projection_init": "gradient"},
    {"target_module": "blocks.3.attn.out_proj", "projection_init": "gradient"},
]


def _metadata(seed: int, validation_batches: int, batch_size: int) -> dict[str, Any]:
    return {
        "baseline_config": "configs/baselines/small_3.5M.yaml",
        "device": "cpu",
        "batch_size": batch_size,
        "seq_len": 8,
        "phi_rank": 8,
        "graft_scale": 1.0,
        "learning_rate": 0.005,
        "use_real_tokens": True,
        "real_data_dir": "data/baseline",
        "validation_split": "val",
        "validation_batches": validation_batches,
        "data_seed": seed,
        "validation_seed": 3000 + seed,
        "old_validation_seed": 1000 + seed,
        "require_beats_dense": True,
        "min_validation_gain": 0.0,
        "min_gain_per_parameter": 0.0,
        "defer_gain_floor": -0.00005,
        "grafts": LINEAR_SEQUENCE,
    }


def _run(seed: int, out_dir: Path, validation_batches: int, batch_size: int) -> dict[str, Any]:
    run_dir = out_dir / f"linear_retention_seed{seed}"
    config = RuntimeConfig(
        experiment_name=f"drm_g_marco5b_retention_seed{seed}",
        task="drm_transformer",
        method="drm_g_saint_phi_progressive",
        output_dir=str(run_dir),
        seed=seed,
        steps=2,
        parameter_budget=128,
        metadata=_metadata(seed, validation_batches, batch_size),
    )
    manifest = train_runtime(config)
    meta = manifest["metadata"]
    checkpoint_bytes = sum(int(item.get("bytes", 0)) for item in manifest.get("files", []))
    return {
        "seed": seed,
        "base_loss": meta["base_loss"],
        "final_loss": meta["final_loss"],
        "sequence_gain": meta["sequence_gain"],
        "gain_per_parameter": meta["sequence_gain_per_parameter"],
        "old_regression": meta["old_regression"],
        "approved_grafts": meta["approved_grafts"],
        "rejected_grafts": meta["rejected_grafts"],
        "deferred_grafts": meta["deferred_grafts"],
        "approval_rate": meta["approval_rate"],
        "conflict_count": meta["conflict_count"],
        "checkpoint_bytes": checkpoint_bytes,
        "routing_s": meta["routing_s"],
        "train_s": meta["train_s"],
        "eval_s": meta["eval_s"],
        "rows": meta["progressive_rows"],
        "run_dir": str(run_dir),
    }


def _summary(rows: list[dict[str, Any]], max_old_regression: float) -> dict[str, Any]:
    positive = [row for row in rows if row["sequence_gain"] > 0.0]
    retained = [
        row for row in positive
        if row["old_regression"] <= max_old_regression and row["approved_grafts"] >= 1
    ]
    best = max(rows, key=lambda row: row["gain_per_parameter"])
    return {
        "run_count": len(rows),
        "positive_runs": len(positive),
        "retention_passed_runs": len(retained),
        "max_old_regression": max_old_regression,
        "best": best,
        "phase_5b_passed": bool(retained),
    }


def _markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# DRM-G Marco 5B Retention",
        "",
        f"- run_count: {summary['run_count']}",
        f"- positive_runs: {summary['positive_runs']}",
        f"- retention_passed_runs: {summary['retention_passed_runs']}",
        f"- max_old_regression: {summary['max_old_regression']}",
        f"- phase_5b_passed: {summary['phase_5b_passed']}",
        "",
        "| seed | gain | gain/param | old_reg | approved | rejected | deferred |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {sequence_gain:.6f} | {gain_per_parameter:.6e} | "
            "{old_regression:.6f} | {approved_grafts} | {rejected_grafts} | "
            "{deferred_grafts} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/drm_g_marco5b_retention")
    parser.add_argument("--seeds", nargs="*", type=int, default=[31, 32, 33, 34])
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-old-regression", type=float, default=0.0002)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _run(seed, out_dir, args.validation_batches, args.batch_size)
        for seed in args.seeds
    ]
    rows.sort(key=lambda row: row["gain_per_parameter"], reverse=True)
    summary = _summary(rows, args.max_old_regression)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(_markdown(rows, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
