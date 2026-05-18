"""Partition and reconstruct 2D matrices using fixed-size blocks.

The first implementation deliberately uses plain Python sequences. That keeps
the phase-1 contract independent of NumPy/PyTorch while the SAINT block model is
still being validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


Number = int | float
Matrix = Sequence[Sequence[Number]]
BlockSize = int | tuple[int, int]


@dataclass(frozen=True)
class MatrixBlock:
    """A rectangular block extracted from a matrix.

    `row` and `col` are block-grid coordinates, not raw matrix indices.
    `row_start` and `col_start` are raw matrix indices.
    """

    row: int
    col: int
    row_start: int
    col_start: int
    height: int
    width: int
    values: tuple[tuple[Number, ...], ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def _normalize_block_size(block_size: BlockSize) -> tuple[int, int]:
    if isinstance(block_size, int):
        height = width = block_size
    else:
        if len(block_size) != 2:
            raise ValueError("block_size must be an int or a (height, width) tuple")
        height, width = block_size

    if height <= 0 or width <= 0:
        raise ValueError("block dimensions must be positive")

    return int(height), int(width)


def _validate_matrix(matrix: Matrix) -> tuple[int, int]:
    if not matrix:
        raise ValueError("matrix must contain at least one row")

    row_count = len(matrix)
    col_count = len(matrix[0])
    if col_count == 0:
        raise ValueError("matrix must contain at least one column")

    for index, row in enumerate(matrix):
        if len(row) != col_count:
            raise ValueError(
                f"matrix must be rectangular; row 0 has {col_count} columns "
                f"but row {index} has {len(row)}"
            )

    return row_count, col_count


def partition_matrix(
    matrix: Matrix,
    block_size: BlockSize,
    *,
    pad_value: Number = 0,
) -> list[MatrixBlock]:
    """Split a 2D matrix into fixed-size blocks.

    Edge blocks are padded to the requested block size. The original matrix
    shape should be passed to `reconstruct_matrix` to remove padding.
    """

    row_count, col_count = _validate_matrix(matrix)
    block_height, block_width = _normalize_block_size(block_size)

    blocks: list[MatrixBlock] = []
    block_row = 0
    for row_start in range(0, row_count, block_height):
        block_col = 0
        for col_start in range(0, col_count, block_width):
            rows: list[tuple[Number, ...]] = []
            for r_offset in range(block_height):
                source_row = row_start + r_offset
                values: list[Number] = []
                for c_offset in range(block_width):
                    source_col = col_start + c_offset
                    if source_row < row_count and source_col < col_count:
                        values.append(matrix[source_row][source_col])
                    else:
                        values.append(pad_value)
                rows.append(tuple(values))

            blocks.append(
                MatrixBlock(
                    row=block_row,
                    col=block_col,
                    row_start=row_start,
                    col_start=col_start,
                    height=block_height,
                    width=block_width,
                    values=tuple(rows),
                )
            )
            block_col += 1
        block_row += 1

    return blocks


def reconstruct_matrix(
    blocks: Iterable[MatrixBlock],
    original_shape: tuple[int, int],
    *,
    fill_value: Number = 0,
) -> list[list[Number]]:
    """Reconstruct a matrix from blocks and crop padding to `original_shape`."""

    row_count, col_count = original_shape
    if row_count <= 0 or col_count <= 0:
        raise ValueError("original_shape dimensions must be positive")

    matrix: list[list[Number]] = [
        [fill_value for _ in range(col_count)]
        for _ in range(row_count)
    ]

    for block in blocks:
        for r_offset, row_values in enumerate(block.values):
            target_row = block.row_start + r_offset
            if target_row >= row_count:
                continue
            for c_offset, value in enumerate(row_values):
                target_col = block.col_start + c_offset
                if target_col >= col_count:
                    continue
                matrix[target_row][target_col] = value

    return matrix
