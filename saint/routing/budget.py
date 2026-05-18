"""Budget-first region routing."""

from __future__ import annotations

from saint.blocks.metrics import reconstruction_error
from saint.blocks.partition import Matrix
from saint.reconstruction.baselines import (
    ReconstructionResult,
    block_codebook_reconstruction,
)
from saint.reconstruction.matrix_ops import shape, zeros
from saint.routing.regions import count_matrix_values, slice_region, write_region
from saint.routing.types import RoutedRegion, RoutingPlan, WeightSearchResult


def budget_region_candidate(
    region: Matrix,
    *,
    candidate_block_sizes: tuple[int, ...],
    quantization_step: float,
    error_weight: float,
    parameter_weight: float,
    include_free_delta: bool,
) -> tuple[str, list[list[float]], float, int]:
    original_values = count_matrix_values(region)
    candidates = []
    for block_size in candidate_block_sizes:
        result = block_codebook_reconstruction(
            region,
            block_size=block_size,
            signature_mode="quantized",
            quantization_step=quantization_step,
        )
        error = reconstruction_error(region, result.reconstructed).relative_l1_error
        parameter_ratio = result.parameter_count / original_values
        score = error_weight * error + parameter_weight * parameter_ratio
        candidates.append(
            (
                f"codebook_{block_size}",
                result.reconstructed,
                error,
                result.parameter_count,
                score,
            )
        )

    if include_free_delta:
        rows, cols = shape(region)
        params = rows * cols
        candidates.append(
            (
                "free_delta",
                [[float(value) for value in row] for row in region],
                0.0,
                params,
                parameter_weight * (params / original_values),
            )
        )

    method, reconstructed, error, params, _score = min(
        candidates,
        key=lambda candidate: (candidate[4], candidate[3], candidate[2]),
    )
    return method, reconstructed, error, params


def route_matrix_regions_by_budget(
    matrix: Matrix,
    *,
    region_size: int = 8,
    candidate_block_sizes: tuple[int, ...] = (4, 2),
    quantization_step: float = 0.05,
    error_weight: float = 1.0,
    parameter_weight: float = 0.25,
    target_compression: float = 1.0,
    include_free_delta: bool = True,
) -> tuple[RoutingPlan, list[list[float]]]:
    """Route regions by score, then report target compression success."""

    rows, cols = shape(matrix)
    reconstructed = zeros(rows, cols)
    regions: list[RoutedRegion] = []

    for row_start in range(0, rows, region_size):
        for col_start in range(0, cols, region_size):
            region = slice_region(
                matrix,
                row_start,
                col_start,
                region_size,
                region_size,
            )
            height, width = shape(region)
            method, region_recon, error, params = budget_region_candidate(
                region,
                candidate_block_sizes=candidate_block_sizes,
                quantization_step=quantization_step,
                error_weight=error_weight,
                parameter_weight=parameter_weight,
                include_free_delta=include_free_delta,
            )
            write_region(reconstructed, row_start, col_start, region_recon)
            regions.append(
                RoutedRegion(
                    row_start=row_start,
                    col_start=col_start,
                    height=height,
                    width=width,
                    method=method,
                    relative_l1_error=error,
                    parameter_count=params,
                )
            )

    total_parameter_count = sum(region.parameter_count for region in regions)
    original_values = rows * cols
    compression_ratio = (
        original_values / total_parameter_count
        if total_parameter_count > 0
        else 0.0
    )
    plan = RoutingPlan(
        regions=tuple(regions),
        total_parameter_count=total_parameter_count,
        metadata={
            "region_size": region_size,
            "candidate_block_sizes": candidate_block_sizes,
            "quantization_step": quantization_step,
            "error_weight": error_weight,
            "parameter_weight": parameter_weight,
            "target_compression": target_compression,
            "target_met": compression_ratio >= target_compression,
            "include_free_delta": include_free_delta,
        },
    )
    return plan, reconstructed


def select_budget_search_result(
    matrix: Matrix,
    *,
    search_results: list[WeightSearchResult],
    target_compression: float,
    max_relative_l1_error: float,
) -> WeightSearchResult:
    feasible = [entry for entry in search_results if entry.target_met]
    if feasible:
        return max(
            feasible,
            key=lambda entry: (entry.compression_ratio, -entry.relative_l1_error),
        )
    return min(
        search_results,
        key=lambda entry: (
            max(entry.relative_l1_error - max_relative_l1_error, 0.0)
            + max(target_compression - entry.compression_ratio, 0.0),
            entry.relative_l1_error,
            -entry.compression_ratio,
        ),
    )


def make_weight_search_result(
    matrix: Matrix,
    *,
    parameter_weight: float,
    result: ReconstructionResult,
    target_compression: float,
    max_relative_l1_error: float,
) -> WeightSearchResult:
    rows, cols = shape(matrix)
    original_values = rows * cols
    error = reconstruction_error(matrix, result.reconstructed).relative_l1_error
    compression = (
        original_values / result.parameter_count
        if result.parameter_count > 0
        else 0.0
    )
    return WeightSearchResult(
        parameter_weight=parameter_weight,
        result=result,
        relative_l1_error=error,
        compression_ratio=compression,
        target_met=(
            error <= max_relative_l1_error
            and compression >= target_compression
        ),
    )
