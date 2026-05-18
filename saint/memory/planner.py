"""Simple memory planner for SAINT runtime experiments."""

from __future__ import annotations

from dataclasses import dataclass

from saint.config import RuntimeConfig


@dataclass(frozen=True)
class MemoryPlan:
    vram_gb: float
    trainable_parameters: int
    optimizer_state_values: int
    estimated_bytes: int
    fits_budget: bool
    notes: tuple[str, ...]


def estimate_runtime_memory(config: RuntimeConfig) -> MemoryPlan:
    trainable = max(1, int(config.parameter_budget))
    optimizer_values = trainable * 2
    # Python prototype estimate: fp32 param + two fp32 optimizer states.
    estimated_bytes = (trainable + optimizer_values) * 4
    budget_bytes = int(config.vram_gb * 1024 * 1024 * 1024)
    notes = (
        "prototype_estimate",
        "base_model_is_frozen",
        "activation_memory_not_modeled",
    )
    return MemoryPlan(
        vram_gb=config.vram_gb,
        trainable_parameters=trainable,
        optimizer_state_values=optimizer_values,
        estimated_bytes=estimated_bytes,
        fits_budget=estimated_bytes <= budget_bytes,
        notes=notes,
    )


__all__ = ["MemoryPlan", "estimate_runtime_memory"]
