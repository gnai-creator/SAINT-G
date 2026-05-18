"""Matrix block utilities for SAINT phase 1."""

from .partition import MatrixBlock, partition_matrix, reconstruct_matrix
from .signatures import compute_block_signature, compute_block_signatures
from .grouping import group_blocks_by_signature
from .metrics import (
    BlockAnalysis,
    BlockReuseMetrics,
    ReconstructionMetrics,
    analyze_block_reuse,
    block_reuse_metrics,
    reconstruction_error,
)
from .codebook import FixedCodebook, build_fixed_codebook

__all__ = [
    "BlockAnalysis",
    "BlockReuseMetrics",
    "FixedCodebook",
    "MatrixBlock",
    "ReconstructionMetrics",
    "analyze_block_reuse",
    "block_reuse_metrics",
    "build_fixed_codebook",
    "partition_matrix",
    "reconstruct_matrix",
    "reconstruction_error",
    "compute_block_signature",
    "compute_block_signatures",
    "group_blocks_by_signature",
]
