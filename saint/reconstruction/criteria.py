"""Automatic success/failure criteria for reconstruction benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from .benchmark import BenchmarkResult


@dataclass(frozen=True)
class CriteriaDecision:
    passed: bool
    reason: str
    method_name: str
    avg_relative_l1_error: float
    avg_compression_ratio: float


def evaluate_method_against_thresholds(
    results: list[BenchmarkResult],
    *,
    method_name: str,
    max_avg_relative_l1_error: float,
    min_avg_compression_ratio: float,
) -> CriteriaDecision:
    """Evaluate one method against simple aggregate thresholds."""

    selected = [result for result in results if result.method_name == method_name]
    if not selected:
        return CriteriaDecision(
            passed=False,
            reason=f"method not found: {method_name}",
            method_name=method_name,
            avg_relative_l1_error=float("inf"),
            avg_compression_ratio=0.0,
        )

    avg_error = sum(result.relative_l1_error for result in selected) / len(selected)
    avg_compression = sum(result.compression_ratio for result in selected) / len(selected)
    passed = (
        avg_error <= max_avg_relative_l1_error
        and avg_compression >= min_avg_compression_ratio
    )
    reason = (
        "passed"
        if passed
        else (
            f"failed thresholds: avg_error={avg_error:.4f} "
            f"max={max_avg_relative_l1_error:.4f}, "
            f"avg_compression={avg_compression:.4f} "
            f"min={min_avg_compression_ratio:.4f}"
        )
    )
    return CriteriaDecision(
        passed=passed,
        reason=reason,
        method_name=method_name,
        avg_relative_l1_error=avg_error,
        avg_compression_ratio=avg_compression,
    )
