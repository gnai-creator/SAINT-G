"""Quality-first region routing."""

from __future__ import annotations

from saint.blocks.metrics import reconstruction_error
from saint.blocks.partition import Matrix
from saint.reconstruction.baselines import block_codebook_reconstruction
from saint.reconstruction.matrix_ops import shape, zeros
from saint.routing.regions import slice_region, write_region
from saint.routing.types import RoutedRegion, RoutingPlan


def best_region_candidate(
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
            region = slice_region(
                matrix,
                row_start,
                col_start,
                region_size,
                region_size,
            )
            height, width = shape(region)
            method, region_recon, error, params = best_region_candidate(
                region,
                candidate_block_sizes=candidate_block_sizes,
                error_threshold=error_threshold,
                quantization_step=quantization_step,
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
