"""Heuristic block router based on reconstruction error by region."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from saint.blocks import MatrixBlock
from saint.blocks.metrics import reconstruction_error
from saint.blocks.partition import Matrix
from saint.reconstruction.baselines import ReconstructionResult, block_codebook_reconstruction
from saint.reconstruction.matrix_ops import shape, zeros


@dataclass(frozen=True)
class RoutedRegion:
    row_start: int
    col_start: int
    height: int
    width: int
    method: str
    relative_l1_error: float
    parameter_count: int


@dataclass(frozen=True)
class RoutingPlan:
    regions: tuple[RoutedRegion, ...]
    total_parameter_count: int
    metadata: dict


@dataclass(frozen=True)
class WeightSearchResult:
    parameter_weight: float
    result: ReconstructionResult
    relative_l1_error: float
    compression_ratio: float
    target_met: bool


@dataclass(frozen=True)
class RegionCandidate:
    method: str
    reconstructed: list[list[float]]
    relative_l1_error: float
    parameter_count: int
    score: float


def _count_matrix_values(matrix: Matrix) -> int:
    rows, cols = shape(matrix)
    return rows * cols


def _slice_region(
    matrix: Matrix,
    row_start: int,
    col_start: int,
    height: int,
    width: int,
) -> list[list[float]]:
    rows, cols = shape(matrix)
    row_end = min(row_start + height, rows)
    col_end = min(col_start + width, cols)
    return [
        [float(matrix[row][col]) for col in range(col_start, col_end)]
        for row in range(row_start, row_end)
    ]


def _write_region(
    target: list[list[float]],
    row_start: int,
    col_start: int,
    values: Matrix,
) -> None:
    for r_offset, row in enumerate(values):
        for c_offset, value in enumerate(row):
            target[row_start + r_offset][col_start + c_offset] = float(value)


def _region_l1(matrix: Matrix) -> float:
    return sum(abs(float(value)) for row in matrix for value in row)


def _zero_like(matrix: Matrix) -> list[list[float]]:
    rows, cols = shape(matrix)
    return zeros(rows, cols)


def _best_region_candidate(
    region: Matrix,
    *,
    candidate_block_sizes: tuple[int, ...],
    error_threshold: float,
    quantization_step: float,
) -> tuple[str, list[list[float]], float, int]:
    candidates = []
    for block_size in candidate_block_sizes:
        result = block_codebook_reconstruction(
            region,
            block_size=block_size,
            signature_mode="quantized",
            quantization_step=quantization_step,
        )
        error = reconstruction_error(region, result.reconstructed).relative_l1_error
        candidates.append(
            (
                f"codebook_{block_size}",
                result.reconstructed,
                error,
                result.parameter_count,
            )
        )

    acceptable = [
        candidate for candidate in candidates if candidate[2] <= error_threshold
    ]
    if acceptable:
        return min(acceptable, key=lambda candidate: (candidate[3], candidate[2]))

    rows, cols = shape(region)
    return (
        "free_delta",
        [[float(value) for value in row] for row in region],
        0.0,
        rows * cols,
    )


def _budget_region_candidate(
    region: Matrix,
    *,
    candidate_block_sizes: tuple[int, ...],
    quantization_step: float,
    error_weight: float,
    parameter_weight: float,
    include_free_delta: bool,
) -> tuple[str, list[list[float]], float, int]:
    original_values = _count_matrix_values(region)
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


def _sensitivity_candidates(
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
    original_values = _count_matrix_values(region)
    candidates: list[RegionCandidate] = []

    if include_freeze:
        reconstructed = _zero_like(region)
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


def route_matrix_regions(
    matrix: Matrix,
    *,
    region_size: int = 8,
    candidate_block_sizes: tuple[int, ...] = (4, 2),
    error_threshold: float = 0.1,
    quantization_step: float = 0.05,
) -> tuple[RoutingPlan, list[list[float]]]:
    """Route each region to the cheapest candidate below an error threshold."""

    rows, cols = shape(matrix)
    reconstructed = zeros(rows, cols)
    regions: list[RoutedRegion] = []

    for row_start in range(0, rows, region_size):
        for col_start in range(0, cols, region_size):
            region = _slice_region(
                matrix,
                row_start,
                col_start,
                region_size,
                region_size,
            )
            height, width = shape(region)
            method, region_recon, error, params = _best_region_candidate(
                region,
                candidate_block_sizes=candidate_block_sizes,
                error_threshold=error_threshold,
                quantization_step=quantization_step,
            )
            _write_region(reconstructed, row_start, col_start, region_recon)
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

    plan = RoutingPlan(
        regions=tuple(regions),
        total_parameter_count=sum(region.parameter_count for region in regions),
        metadata={
            "region_size": region_size,
            "candidate_block_sizes": candidate_block_sizes,
            "error_threshold": error_threshold,
            "quantization_step": quantization_step,
        },
    )
    return plan, reconstructed


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
            region = _slice_region(
                matrix,
                row_start,
                col_start,
                region_size,
                region_size,
            )
            height, width = shape(region)
            method, region_recon, error, params = _budget_region_candidate(
                region,
                candidate_block_sizes=candidate_block_sizes,
                quantization_step=quantization_step,
                error_weight=error_weight,
                parameter_weight=parameter_weight,
                include_free_delta=include_free_delta,
            )
            _write_region(reconstructed, row_start, col_start, region_recon)
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
            region = _slice_region(matrix, row_start, col_start, region_size, region_size)
            sensitivity_raw = _region_l1(region)
            max_l1 = max(max_l1, sensitivity_raw)
            region_entries.append((row_start, col_start, region, sensitivity_raw))

    region_count = len(region_entries)
    budgets = method_budgets or {
        "free_delta": 0.05,
        "codebook_2": 0.20,
    }
    max_counts: dict[str, int] = {}
    for method, fraction in budgets.items():
        max_counts[method] = (
            max(1, int(region_count * fraction))
            if fraction > 0
            else 0
        )
    used_counts: dict[str, int] = {}
    reconstructed = zeros(rows, cols)
    routed_regions: list[RoutedRegion] = []

    for row_start, col_start, region, sensitivity_raw in sorted(
        region_entries,
        key=lambda entry: entry[3],
        reverse=True,
    ):
        sensitivity = sensitivity_raw / max_l1 if max_l1 > 0 else 0.0
        candidates = _sensitivity_candidates(
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
        _write_region(reconstructed, row_start, col_start, selected.reconstructed)
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
    method_counts: dict[str, int] = {}
    for region in plan.regions:
        method_counts[region.method] = method_counts.get(region.method, 0) + 1
    return ReconstructionResult(
        name="routed_quality_first",
        reconstructed=reconstructed,
        parameter_count=plan.total_parameter_count,
        elapsed_s=perf_counter() - start,
        metadata={
            **plan.metadata,
            "region_count": len(plan.regions),
            "method_counts": method_counts,
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
    method_counts: dict[str, int] = {}
    for region in plan.regions:
        method_counts[region.method] = method_counts.get(region.method, 0) + 1
    return ReconstructionResult(
        name="routed_budget_first",
        reconstructed=reconstructed,
        parameter_count=plan.total_parameter_count,
        elapsed_s=perf_counter() - start,
        metadata={
            **plan.metadata,
            "region_count": len(plan.regions),
            "method_counts": method_counts,
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
    rows, cols = shape(matrix)
    original_values = rows * cols
    search_results: list[WeightSearchResult] = []

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
        error = reconstruction_error(matrix, result.reconstructed).relative_l1_error
        compression = (
            original_values / result.parameter_count
            if result.parameter_count > 0
            else 0.0
        )
        search_results.append(
            WeightSearchResult(
                parameter_weight=parameter_weight,
                result=result,
                relative_l1_error=error,
                compression_ratio=compression,
                target_met=(
                    error <= max_relative_l1_error
                    and compression >= target_compression
                ),
            )
        )

    feasible = [entry for entry in search_results if entry.target_met]
    if feasible:
        selected = max(
            feasible,
            key=lambda entry: (entry.compression_ratio, -entry.relative_l1_error),
        )
    else:
        selected = min(
            search_results,
            key=lambda entry: (
                max(entry.relative_l1_error - max_relative_l1_error, 0.0)
                + max(target_compression - entry.compression_ratio, 0.0),
                entry.relative_l1_error,
                -entry.compression_ratio,
            ),
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
    method_counts: dict[str, int] = {}
    for region in plan.regions:
        method_counts[region.method] = method_counts.get(region.method, 0) + 1
    return ReconstructionResult(
        name="routed_sensitivity_budget",
        reconstructed=reconstructed,
        parameter_count=plan.total_parameter_count,
        elapsed_s=perf_counter() - start,
        metadata={
            **plan.metadata,
            "region_count": len(plan.regions),
            "method_counts": method_counts,
        },
    )
