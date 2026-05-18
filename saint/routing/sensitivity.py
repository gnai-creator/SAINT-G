"""Sensitivity-aware region routing."""

from __future__ import annotations

from saint.blocks.metrics import reconstruction_error
from saint.blocks.partition import Matrix
from saint.reconstruction.baselines import block_codebook_reconstruction
from saint.reconstruction.matrix_ops import shape, zeros
from saint.routing.regions import (
    count_matrix_values,
    region_l1,
    slice_region,
    write_region,
    zero_like,
)
from saint.routing.types import RegionCandidate, RoutedRegion, RoutingPlan


def sensitivity_candidates(
    region: Matrix,
    *,
    candidate_block_sizes: tuple[int, ...],
    quantization_step: float,
    error_weight: float,
    parameter_weight: float,
    sensitivity: float,
    include_freeze: bool,
    include_free_delta: bool,
) -> list[RegionCandidate]:
    original_values = count_matrix_values(region)
    candidates: list[RegionCandidate] = []

    if include_freeze:
        reconstructed = zero_like(region)
        error = reconstruction_error(region, reconstructed).relative_l1_error
        candidates.append(
            RegionCandidate(
                method="freeze",
                reconstructed=reconstructed,
                relative_l1_error=error,
                parameter_count=0,
                score=error_weight * sensitivity * error,
            )
        )

    for block_size in candidate_block_sizes:
        result = block_codebook_reconstruction(
            region,
            block_size=block_size,
            signature_mode="quantized",
            quantization_step=quantization_step,
        )
        error = reconstruction_error(region, result.reconstructed).relative_l1_error
        parameter_ratio = result.parameter_count / original_values
        candidates.append(
            RegionCandidate(
                method=f"codebook_{block_size}",
                reconstructed=result.reconstructed,
                relative_l1_error=error,
                parameter_count=result.parameter_count,
                score=(
                    error_weight * sensitivity * error
                    + parameter_weight * parameter_ratio
                ),
            )
        )

    if include_free_delta:
        rows, cols = shape(region)
        params = rows * cols
        candidates.append(
            RegionCandidate(
                method="free_delta",
                reconstructed=[[float(value) for value in row] for row in region],
                relative_l1_error=0.0,
                parameter_count=params,
                score=parameter_weight * (params / original_values),
            )
        )

    return sorted(candidates, key=lambda candidate: (candidate.score, candidate.parameter_count))


def route_matrix_regions_by_sensitivity_budget(
    matrix: Matrix,
    *,
    region_size: int = 8,
    candidate_block_sizes: tuple[int, ...] = (4, 2),
    quantization_step: float = 0.1,
    error_weight: float = 1.0,
    parameter_weight: float = 0.5,
    method_budgets: dict[str, float] | None = None,
    target_compression: float = 1.1,
    include_freeze: bool = True,
    include_free_delta: bool = True,
) -> tuple[RoutingPlan, list[list[float]]]:
    """Route high-sensitivity regions first under hard per-method budgets."""

    rows, cols = shape(matrix)
    region_entries = []
    max_l1 = 0.0
    for row_start in range(0, rows, region_size):
        for col_start in range(0, cols, region_size):
            region = slice_region(matrix, row_start, col_start, region_size, region_size)
            sensitivity_raw = region_l1(region)
            max_l1 = max(max_l1, sensitivity_raw)
            region_entries.append((row_start, col_start, region, sensitivity_raw))

    region_count = len(region_entries)
    budgets = method_budgets or {
        "free_delta": 0.05,
        "codebook_2": 0.20,
    }
    max_counts = {
        method: max(1, int(region_count * fraction)) if fraction > 0 else 0
        for method, fraction in budgets.items()
    }
    used_counts: dict[str, int] = {}
    reconstructed = zeros(rows, cols)
    routed_regions: list[RoutedRegion] = []

    for row_start, col_start, region, sensitivity_raw in sorted(
        region_entries,
        key=lambda entry: entry[3],
        reverse=True,
    ):
        sensitivity = sensitivity_raw / max_l1 if max_l1 > 0 else 0.0
        candidates = sensitivity_candidates(
            region,
            candidate_block_sizes=candidate_block_sizes,
            quantization_step=quantization_step,
            error_weight=error_weight,
            parameter_weight=parameter_weight,
            sensitivity=sensitivity,
            include_freeze=include_freeze,
            include_free_delta=include_free_delta,
        )
        selected = next(
            (candidate for candidate in candidates if candidate.method == "freeze"),
            candidates[0],
        )
        for candidate in candidates:
            limit = max_counts.get(candidate.method)
            if limit is None or used_counts.get(candidate.method, 0) < limit:
                selected = candidate
                break

        used_counts[selected.method] = used_counts.get(selected.method, 0) + 1
        write_region(reconstructed, row_start, col_start, selected.reconstructed)
        height, width = shape(region)
        routed_regions.append(
            RoutedRegion(
                row_start=row_start,
                col_start=col_start,
                height=height,
                width=width,
                method=selected.method,
                relative_l1_error=selected.relative_l1_error,
                parameter_count=selected.parameter_count,
            )
        )

    total_parameter_count = sum(region.parameter_count for region in routed_regions)
    original_values = rows * cols
    compression_ratio = (
        original_values / total_parameter_count
        if total_parameter_count > 0
        else float("inf")
    )
    plan = RoutingPlan(
        regions=tuple(routed_regions),
        total_parameter_count=total_parameter_count,
        metadata={
            "region_size": region_size,
            "candidate_block_sizes": candidate_block_sizes,
            "quantization_step": quantization_step,
            "error_weight": error_weight,
            "parameter_weight": parameter_weight,
            "method_budgets": budgets,
            "target_compression": target_compression,
            "target_met": compression_ratio >= target_compression,
            "include_freeze": include_freeze,
            "include_free_delta": include_free_delta,
        },
    )
    return plan, reconstructed
