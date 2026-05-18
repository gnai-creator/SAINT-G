"""Grouping utilities for blocks that share a signature."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

from .partition import MatrixBlock
from .signatures import compute_block_signature


def group_blocks_by_signature(
    blocks: list[MatrixBlock],
    *,
    mode: str = "exact",
    quantization_step: float = 1.0,
) -> dict[tuple, list[MatrixBlock]]:
    """Group blocks by exact, quantized, or statistical signature."""

    groups: DefaultDict[tuple, list[MatrixBlock]] = defaultdict(list)
    for block in blocks:
        signature = compute_block_signature(
            block,
            mode=mode,
            quantization_step=quantization_step,
        )
        groups[signature].append(block)
    return dict(groups)
