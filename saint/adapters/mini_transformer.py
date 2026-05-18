"""Mini-transformer adapter for the SAINT runtime."""

from __future__ import annotations

from saint.config import RuntimeConfig
from saint.sensitivity import train_mini_sensitivity_delta
from saint.transformer import (
    make_mini_transformer_task,
    train_mini_block_budgeted_delta,
    train_mini_budgeted_delta,
    train_mini_lora_delta,
    train_mini_saint_delta,
)


def make_task(config: RuntimeConfig):
    return make_mini_transformer_task(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        seq_len=config.seq_len,
        train_samples=config.train_samples,
        test_samples=config.test_samples,
        seed=config.seed,
        delta_mode=config.delta_mode,
        delta_scale=config.delta_scale,
    )


def run_method(config: RuntimeConfig):
    task = make_task(config)
    common = {
        "parameter_budget": config.parameter_budget,
        "steps": config.steps,
    }
    if config.method == "mini_saint_dynamic_delta":
        return train_mini_saint_delta(
            task,
            **common,
            sensitivity_method=config.sensitivity_method,
        )
    if config.method == "mini_budgeted_delta":
        return train_mini_budgeted_delta(task, **common)
    if config.method == "mini_block_budgeted_delta":
        return train_mini_block_budgeted_delta(task, **common)
    if config.method == "mini_lora_rank_1":
        return train_mini_lora_delta(task, rank=1, steps=config.steps)
    if config.method == "sensitivity_gradient_norm":
        return train_mini_sensitivity_delta(task, method="gradient_norm", **common)
    raise ValueError(f"unknown mini-transformer method: {config.method}")


def inspect_model(config: RuntimeConfig) -> dict:
    task = make_task(config)
    matrices = {
        name: {
            "rows": len(matrix),
            "cols": len(matrix[0]) if matrix else 0,
            "parameters": sum(len(row) for row in matrix),
        }
        for name, matrix in task.base_weights.items()
    }
    return {
        "task": config.task,
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "matrices": matrices,
        "total_parameters": sum(item["parameters"] for item in matrices.values()),
    }


__all__ = ["inspect_model", "make_task", "run_method"]
