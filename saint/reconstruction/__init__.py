"""Reconstruction benchmarks for SAINT phase 2."""

from .baselines import (
    ReconstructionResult,
    block_codebook_reconstruction,
    hierarchical_codebook_reconstruction,
    low_rank_reconstruction,
    multi_scale_codebook_reconstruction,
    original_reconstruction,
    residual_codebook_reconstruction,
    scaled_block_codebook_reconstruction,
    uniform_quantization_reconstruction,
)
from .benchmark import BenchmarkCase, BenchmarkResult, run_reconstruction_benchmark
from .criteria import CriteriaDecision, evaluate_method_against_thresholds
from .generators import (
    gaussian_matrix,
    low_rank_matrix,
    repeated_block_matrix,
    sparse_matrix,
)
from saint.routing import (
    routed_budget_reconstruction,
    routed_codebook_reconstruction,
    routed_sensitivity_budget_reconstruction,
    search_routed_budget_reconstruction,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkResult",
    "CriteriaDecision",
    "ReconstructionResult",
    "block_codebook_reconstruction",
    "gaussian_matrix",
    "hierarchical_codebook_reconstruction",
    "low_rank_matrix",
    "low_rank_reconstruction",
    "multi_scale_codebook_reconstruction",
    "original_reconstruction",
    "repeated_block_matrix",
    "residual_codebook_reconstruction",
    "run_reconstruction_benchmark",
    "routed_codebook_reconstruction",
    "routed_budget_reconstruction",
    "routed_sensitivity_budget_reconstruction",
    "search_routed_budget_reconstruction",
    "scaled_block_codebook_reconstruction",
    "sparse_matrix",
    "uniform_quantization_reconstruction",
    "evaluate_method_against_thresholds",
]
