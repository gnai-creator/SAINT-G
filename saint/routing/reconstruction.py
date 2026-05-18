"""Reconstruction wrappers for routing policies."""

from __future__ import annotations

from time import perf_counter

from saint.blocks.metrics import reconstruction_error
from saint.blocks.partition import Matrix
from saint.reconstruction.baselines import ReconstructionResult
from saint.routing.budget import (
    make_weight_search_result,
    route_matrix_regions_by_budget,
    select_budget_search_result,
)
from saint.routing.quality import route_matrix_regions
from saint.routing.regions import method_counts
from saint.routing.sensitivity import route_matrix_regions_by_sensitivity_budget


def routed_codebook_reconstruction(
    matrix: Matrix,
    *,
    region_size: int = 8,
    candidate_block_sizes: tuple[int, ...] = (4, 2),
    error_threshold: float = 0.1,
    quantization_step: float = 0.05,
) -> ReconstructionResult:
    """Reconstruction baseline using the heuristic region router."""

    start = perf_counter()
    plan, reconstructed = route_matrix_regions(
        matrix,
        region_size=region_size,
        candidate_block_sizes=candidate_block_sizes,
        error_threshold=error_threshold,
        quantization_step=quantization_step,
    )
    return ReconstructionResult(
        name="routed_quality_first",
        reconstructed=reconstructed,
        parameter_count=plan.total_parameter_count,
        elapsed_s=perf_counter() - start,
        metadata={
            **plan.metadata,
            "region_count": len(plan.regions),
            "method_counts": method_counts(plan.regions),
        },
    )


def routed_budget_reconstruction(
    matrix: Matrix,
    *,
    region_size: int = 8,
    candidate_block_sizes: tuple[int, ...] = (4, 2),
    quantization_step: float = 0.05,
    error_weight: float = 1.0,
    parameter_weight: float = 0.25,
    target_compression: float = 1.0,
    include_free_delta: bool = True,
) -> ReconstructionResult:
    """Reconstruction baseline using score = error + parameter penalty."""

    start = perf_counter()
    plan, reconstructed = route_matrix_regions_by_budget(
        matrix,
        region_size=region_size,
        candidate_block_sizes=candidate_block_sizes,
        quantization_step=quantization_step,
        error_weight=error_weight,
        parameter_weight=parameter_weight,
        target_compression=target_compression,
        include_free_delta=include_free_delta,
    )
    return ReconstructionResult(
        name="routed_budget_first",
        reconstructed=reconstructed,
        parameter_count=plan.total_parameter_count,
        elapsed_s=perf_counter() - start,
        metadata={
            **plan.metadata,
            "region_count": len(plan.regions),
            "method_counts": method_counts(plan.regions),
        },
    )


def search_routed_budget_reconstruction(
    matrix: Matrix,
    *,
    parameter_weights: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0),
    region_size: int = 8,
    candidate_block_sizes: tuple[int, ...] = (4, 2),
    quantization_step: float = 0.05,
    error_weight: float = 1.0,
    target_compression: float = 1.1,
    max_relative_l1_error: float = 0.1,
    include_free_delta: bool = True,
) -> ReconstructionResult:
    """Search parameter weights and pick the best result meeting constraints."""

    start = perf_counter()
    search_results = []
    for parameter_weight in parameter_weights:
        result = routed_budget_reconstruction(
            matrix,
            region_size=region_size,
            candidate_block_sizes=candidate_block_sizes,
            quantization_step=quantization_step,
            error_weight=error_weight,
            parameter_weight=parameter_weight,
            target_compression=target_compression,
            include_free_delta=include_free_delta,
        )
        search_results.append(
            make_weight_search_result(
                matrix,
                parameter_weight=parameter_weight,
                result=result,
                target_compression=target_compression,
                max_relative_l1_error=max_relative_l1_error,
            )
        )
    selected = select_budget_search_result(
        matrix,
        search_results=search_results,
        target_compression=target_compression,
        max_relative_l1_error=max_relative_l1_error,
    )

    return ReconstructionResult(
        name="routed_budget_search",
        reconstructed=selected.result.reconstructed,
        parameter_count=selected.result.parameter_count,
        elapsed_s=perf_counter() - start,
        metadata={
            **selected.result.metadata,
            "selected_parameter_weight": selected.parameter_weight,
            "target_compression": target_compression,
            "max_relative_l1_error": max_relative_l1_error,
            "target_met": selected.target_met,
            "search": [
                {
                    "parameter_weight": entry.parameter_weight,
                    "relative_l1_error": entry.relative_l1_error,
                    "compression_ratio": entry.compression_ratio,
                    "target_met": entry.target_met,
                }
                for entry in search_results
            ],
        },
    )


def routed_sensitivity_budget_reconstruction(
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
) -> ReconstructionResult:
    """Routed reconstruction with freeze, sensitivity proxy, and hard budgets."""

    start = perf_counter()
    plan, reconstructed = route_matrix_regions_by_sensitivity_budget(
        matrix,
        region_size=region_size,
        candidate_block_sizes=candidate_block_sizes,
        quantization_step=quantization_step,
        error_weight=error_weight,
        parameter_weight=parameter_weight,
        method_budgets=method_budgets,
        target_compression=target_compression,
        include_freeze=include_freeze,
        include_free_delta=include_free_delta,
    )
    return ReconstructionResult(
        name="routed_sensitivity_budget",
        reconstructed=reconstructed,
        parameter_count=plan.total_parameter_count,
        elapsed_s=perf_counter() - start,
        metadata={
            **plan.metadata,
            "region_count": len(plan.regions),
            "method_counts": method_counts(plan.regions),
        },
    )
