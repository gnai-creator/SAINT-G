"""Metrics for matrix reconstruction and block reuse."""

from __future__ import annotations

from dataclasses import dataclass

from .grouping import group_blocks_by_signature
from .partition import Matrix, MatrixBlock, Number, partition_matrix, reconstruct_matrix


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Error metrics between an original matrix and a reconstruction."""

    l1_error: float
    l2_error: float
    relative_l1_error: float
    max_abs_error: float


@dataclass(frozen=True)
class BlockReuseMetrics:
    """Reuse and compression metrics for grouped blocks."""

    block_count: int
    prototype_count: int
    repeated_block_count: int
    reuse_ratio: float
    estimated_original_values: int
    estimated_prototype_values: int
    estimated_compression_ratio: float


@dataclass(frozen=True)
class BlockAnalysis:
    """Combined result for a block partitioning analysis."""

    blocks: list[MatrixBlock]
    groups: dict[tuple, list[MatrixBlock]]
    reconstruction: list[list[Number]]
    reconstruction_metrics: ReconstructionMetrics
    reuse_metrics: BlockReuseMetrics


def reconstruction_error(
    original: Matrix,
    reconstructed: Matrix,
) -> ReconstructionMetrics:
    """Compute simple reconstruction error metrics."""

    if len(original) != len(reconstructed):
        raise ValueError("matrix row counts differ")
    if not original:
        raise ValueError("matrix must contain at least one row")

    l1_error = 0.0
    sq_error = 0.0
    original_l1 = 0.0
    max_abs_error = 0.0

    for row_index, (original_row, reconstructed_row) in enumerate(
        zip(original, reconstructed)
    ):
        if len(original_row) != len(reconstructed_row):
            raise ValueError(f"matrix column counts differ at row {row_index}")
        for original_value, reconstructed_value in zip(original_row, reconstructed_row):
            diff = abs(float(original_value) - float(reconstructed_value))
            l1_error += diff
            sq_error += diff * diff
            original_l1 += abs(float(original_value))
            max_abs_error = max(max_abs_error, diff)

    return ReconstructionMetrics(
        l1_error=l1_error,
        l2_error=sq_error**0.5,
        relative_l1_error=l1_error / original_l1 if original_l1 > 0 else 0.0,
        max_abs_error=max_abs_error,
    )


def block_reuse_metrics(
    blocks: list[MatrixBlock],
    groups: dict[tuple, list[MatrixBlock]],
) -> BlockReuseMetrics:
    """Compute reuse and rough compression metrics from grouped blocks."""

    block_count = len(blocks)
    prototype_count = len(groups)
    repeated_block_count = sum(max(len(group) - 1, 0) for group in groups.values())

    estimated_original_values = sum(block.height * block.width for block in blocks)
    prototype_values = 0
    for group in groups.values():
        if not group:
            continue
        block = group[0]
        prototype_values += block.height * block.width

    return BlockReuseMetrics(
        block_count=block_count,
        prototype_count=prototype_count,
        repeated_block_count=repeated_block_count,
        reuse_ratio=repeated_block_count / block_count if block_count else 0.0,
        estimated_original_values=estimated_original_values,
        estimated_prototype_values=prototype_values,
        estimated_compression_ratio=(
            estimated_original_values / prototype_values
            if prototype_values > 0
            else 0.0
        ),
    )


def analyze_block_reuse(
    matrix: Matrix,
    block_size: int | tuple[int, int],
    *,
    pad_value: Number = 0,
    signature_mode: str = "exact",
    quantization_step: float = 1.0,
) -> BlockAnalysis:
    """Partition, reconstruct, group, and report metrics for a matrix."""

    blocks = partition_matrix(matrix, block_size=block_size, pad_value=pad_value)
    reconstruction = reconstruct_matrix(
        blocks,
        original_shape=(len(matrix), len(matrix[0])),
    )
    groups = group_blocks_by_signature(
        blocks,
        mode=signature_mode,
        quantization_step=quantization_step,
    )
    return BlockAnalysis(
        blocks=blocks,
        groups=groups,
        reconstruction=reconstruction,
        reconstruction_metrics=reconstruction_error(matrix, reconstruction),
        reuse_metrics=block_reuse_metrics(blocks, groups),
    )
