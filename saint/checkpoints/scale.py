"""Scale validation helpers for checkpoint sharding experiments."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import tracemalloc
from typing import Any

from saint.checkpoints.robust import read_matrix_payload_entry, write_matrix_payload


def synthetic_delta_payload(
    *,
    matrix_count: int = 4,
    rows: int = 128,
    cols: int = 128,
) -> dict[str, list[list[float]]]:
    payload = {}
    for matrix_index in range(matrix_count):
        matrix = []
        for row in range(rows):
            matrix.append(
                [
                    ((matrix_index + 1) * 0.001)
                    + ((row % 17) - 8) * 0.0003
                    + ((col % 19) - 9) * 0.0002
                    for col in range(cols)
                ]
            )
        payload[f"matrix_{matrix_index:03d}"] = matrix
    return payload


def _max_abs_error(
    expected: dict[str, list[list[float]]],
    actual: dict[str, list[list[float]]],
) -> float:
    max_error = 0.0
    for name, matrix in expected.items():
        other = actual[name]
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                max_error = max(max_error, abs(value - other[row_index][col_index]))
    return max_error


def benchmark_large_shards(
    run_dir: str | Path,
    *,
    matrix_count: int = 4,
    rows: int = 128,
    cols: int = 128,
    dtype: str = "float16",
    shard_bytes: int = 8192,
) -> dict[str, Any]:
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = synthetic_delta_payload(
        matrix_count=matrix_count,
        rows=rows,
        cols=cols,
    )

    write_start = perf_counter()
    entry = write_matrix_payload(
        target / "large_deltas.saintbin",
        payload,
        dtype=dtype,
        shard_bytes=shard_bytes,
    )
    write_elapsed_s = perf_counter() - write_start

    tracemalloc.start()
    read_start = perf_counter()
    restored = read_matrix_payload_entry(target, entry)
    read_elapsed_s = perf_counter() - read_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    value_count = matrix_count * rows * cols
    return {
        "format": entry["format"],
        "dtype": dtype,
        "matrix_count": matrix_count,
        "rows": rows,
        "cols": cols,
        "value_count": value_count,
        "shard_bytes": shard_bytes,
        "shard_count": int(entry.get("shard_count", 1)),
        "payload_bytes": int(entry["bytes"]),
        "write_elapsed_s": write_elapsed_s,
        "read_elapsed_s": read_elapsed_s,
        "read_peak_bytes": peak_bytes,
        "checksum_validated": True,
        "max_abs_error": _max_abs_error(payload, restored),
    }


def benchmark_partial_shard_read(
    run_dir: str | Path,
    *,
    matrix_count: int = 8,
    rows: int = 256,
    cols: int = 256,
    selected_count: int = 2,
    dtype: str = "float16",
    shard_bytes: int = 65536,
) -> dict[str, Any]:
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = synthetic_delta_payload(
        matrix_count=matrix_count,
        rows=rows,
        cols=cols,
    )
    entry = write_matrix_payload(
        target / "partial_deltas.saintbin",
        payload,
        dtype=dtype,
        shard_bytes=shard_bytes,
    )
    selected = {
        f"matrix_{index:03d}"
        for index in range(min(selected_count, matrix_count))
    }

    tracemalloc.start()
    full_start = perf_counter()
    full = read_matrix_payload_entry(target, entry)
    full_elapsed_s = perf_counter() - full_start
    _, full_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    partial_start = perf_counter()
    partial = read_matrix_payload_entry(target, entry, matrix_names=selected)
    partial_elapsed_s = perf_counter() - partial_start
    _, partial_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    selected_values = sum(
        len(row)
        for name in selected
        for row in payload[name]
    )
    return {
        "format": entry["format"],
        "dtype": dtype,
        "matrix_count": matrix_count,
        "selected_count": len(selected),
        "full_matrix_count": len(full),
        "partial_matrix_count": len(partial),
        "selected_value_count": selected_values,
        "shard_count": int(entry.get("shard_count", 1)),
        "full_read_elapsed_s": full_elapsed_s,
        "full_read_peak_bytes": full_peak_bytes,
        "partial_read_elapsed_s": partial_elapsed_s,
        "partial_read_peak_bytes": partial_peak_bytes,
        "partial_keys": sorted(partial),
        "max_abs_error": _max_abs_error(
            {name: payload[name] for name in selected},
            partial,
        ),
    }


def benchmark_dtype_io(
    run_dir: str | Path,
    *,
    matrix_count: int = 4,
    rows: int = 128,
    cols: int = 128,
    shard_bytes: int = 32768,
    dtypes: tuple[str, ...] = ("float32", "float16", "bfloat16", "int8"),
) -> dict[str, Any]:
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = synthetic_delta_payload(
        matrix_count=matrix_count,
        rows=rows,
        cols=cols,
    )
    results = []
    for dtype in dtypes:
        dtype_dir = target / dtype
        dtype_dir.mkdir(parents=True, exist_ok=True)
        write_start = perf_counter()
        entry = write_matrix_payload(
            dtype_dir / "dtype_deltas.saintbin",
            payload,
            dtype=dtype,
            shard_bytes=shard_bytes,
        )
        write_elapsed_s = perf_counter() - write_start

        tracemalloc.start()
        read_start = perf_counter()
        restored = read_matrix_payload_entry(dtype_dir, entry)
        read_elapsed_s = perf_counter() - read_start
        _, read_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append(
            {
                "dtype": dtype,
                "format": entry["format"],
                "payload_bytes": int(entry["bytes"]),
                "shard_count": int(entry.get("shard_count", 1)),
                "write_elapsed_s": write_elapsed_s,
                "read_elapsed_s": read_elapsed_s,
                "read_peak_bytes": read_peak_bytes,
                "max_abs_error": _max_abs_error(payload, restored),
            }
        )
    baseline_bytes = next(
        item["payload_bytes"] for item in results if item["dtype"] == "float32"
    )
    for item in results:
        item["size_ratio_vs_float32"] = item["payload_bytes"] / baseline_bytes
    return {
        "matrix_count": matrix_count,
        "rows": rows,
        "cols": cols,
        "value_count": matrix_count * rows * cols,
        "shard_bytes": shard_bytes,
        "results": results,
    }


def benchmark_dtype_quality(
    run_dir: str | Path,
    *,
    dtypes: tuple[str, ...] = ("float32", "float16", "bfloat16", "int8"),
) -> dict[str, Any]:
    from saint.config import RuntimeConfig
    from saint.runtime import merge_runtime, train_runtime
    from saint.transformer.model import distillation_loss

    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    results = []
    baseline_loss = None
    for dtype in dtypes:
        output_dir = target / dtype
        config = RuntimeConfig(
            experiment_name=f"phase12e_{dtype}",
            output_dir=str(output_dir),
            task="mini_transformer",
            method="mini_saint_dynamic_delta",
            steps=1,
            parameter_budget=24,
            seed=31,
            metadata={
                "checkpoint_dtype": dtype,
                "checkpoint_shard_bytes": 512,
            },
        )
        train_result = train_runtime(config)
        merged = merge_runtime(output_dir)
        from saint.adapters import make_task

        task = make_task(config)
        merged_loss = distillation_loss(
            merged["merged_weights"],
            task.target_weights,
            task.test_sequences,
        )
        if dtype == "float32":
            baseline_loss = merged_loss
        results.append(
            {
                "dtype": dtype,
                "train_loss": train_result["train_loss"],
                "checkpoint_test_loss": train_result["test_loss"],
                "merged_loss": merged_loss,
                "loss_delta_vs_float32": 0.0 if baseline_loss is None else merged_loss - baseline_loss,
                "delta_format": next(
                    item["format"]
                    for item in train_result["files"]
                    if item["payload"] == "delta"
                ),
                "payload_bytes": next(
                    int(item["bytes"])
                    for item in train_result["files"]
                    if item["payload"] == "delta"
                ),
            }
        )
    baseline_loss = baseline_loss if baseline_loss is not None else 0.0
    for item in results:
        item["loss_delta_vs_float32"] = item["merged_loss"] - baseline_loss
    return {
        "task": "mini_transformer",
        "method": "mini_saint_dynamic_delta",
        "baseline_dtype": "float32",
        "results": results,
    }


__all__ = [
    "benchmark_dtype_io",
    "benchmark_dtype_quality",
    "benchmark_large_shards",
    "benchmark_partial_shard_read",
    "synthetic_delta_payload",
]
