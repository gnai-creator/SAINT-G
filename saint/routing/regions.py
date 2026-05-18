"""Shared region helpers for routing."""

from __future__ import annotations

from saint.blocks.partition import Matrix
from saint.reconstruction.matrix_ops import shape, zeros


def count_matrix_values(matrix: Matrix) -> int:
    rows, cols = shape(matrix)
    return rows * cols


def slice_region(
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


def write_region(
    target: list[list[float]],
    row_start: int,
    col_start: int,
    values: Matrix,
) -> None:
    for r_offset, row in enumerate(values):
        for c_offset, value in enumerate(row):
            target[row_start + r_offset][col_start + c_offset] = float(value)


def region_l1(matrix: Matrix) -> float:
    return sum(abs(float(value)) for row in matrix for value in row)


def zero_like(matrix: Matrix) -> list[list[float]]:
    rows, cols = shape(matrix)
    return zeros(rows, cols)


def method_counts(regions) -> dict[str, int]:
    counts: dict[str, int] = {}
    for region in regions:
        counts[region.method] = counts.get(region.method, 0) + 1
    return counts
