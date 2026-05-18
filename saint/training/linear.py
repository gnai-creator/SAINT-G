"""Compatibility exports for phase-4 linear training experiments."""

from saint.training.criteria import (
    Phase4Decision,
    evaluate_phase4_regime_success,
    evaluate_phase4_success,
)
from saint.training.data import LinearTask, make_linear_delta_task
from saint.training.methods import (
    train_block_scalar_delta,
    train_budgeted_full_delta,
    train_codebook_delta,
    train_full_delta,
    train_lora_delta,
    train_saint_routed_delta,
    train_sparse_sensitivity_delta,
)
from saint.training.ops import TrainingResult, frozen_base_test_loss
from saint.training.sweeps import (
    run_linear_phase4_benchmark,
    run_linear_phase4_regime_sweep,
    run_linear_phase4_sweep,
    summarize_phase4_rows,
)

__all__ = [
    "LinearTask",
    "Phase4Decision",
    "TrainingResult",
    "evaluate_phase4_success",
    "evaluate_phase4_regime_success",
    "frozen_base_test_loss",
    "make_linear_delta_task",
    "run_linear_phase4_benchmark",
    "run_linear_phase4_regime_sweep",
    "run_linear_phase4_sweep",
    "summarize_phase4_rows",
    "train_block_scalar_delta",
    "train_budgeted_full_delta",
    "train_codebook_delta",
    "train_full_delta",
    "train_lora_delta",
    "train_saint_routed_delta",
    "train_sparse_sensitivity_delta",
]
