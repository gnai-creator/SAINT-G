"""Initial fixed codebook utilities for grouped matrix blocks."""

from __future__ import annotations

from dataclasses import dataclass

from .grouping import group_blocks_by_signature
from .partition import MatrixBlock


@dataclass(frozen=True)
class FixedCodebook:
    """A fixed codebook built from grouped block prototypes."""

    prototypes: tuple[MatrixBlock, ...]
    assignments: dict[tuple[int, int], int]

    def prototype_for(self, block: MatrixBlock) -> MatrixBlock:
        return self.prototypes[self.assignments[(block.row, block.col)]]


def build_fixed_codebook(
    blocks: list[MatrixBlock],
    *,
    mode: str = "exact",
    quantization_step: float = 1.0,
) -> FixedCodebook:
    """Build a fixed codebook by taking the first block in each group."""

    groups = group_blocks_by_signature(
        blocks,
        mode=mode,
        quantization_step=quantization_step,
    )

    prototypes: list[MatrixBlock] = []
    assignments: dict[tuple[int, int], int] = {}
    for group in groups.values():
        prototype_id = len(prototypes)
        prototypes.append(group[0])
        for block in group:
            assignments[(block.row, block.col)] = prototype_id

    return FixedCodebook(
        prototypes=tuple(prototypes),
        assignments=assignments,
    )
