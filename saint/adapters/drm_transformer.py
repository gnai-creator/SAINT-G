"""Checkpoint adapter for drm_transformer matrices.

The core SAINT package stays dependency-free. This adapter imports PyTorch only
when a DRM checkpoint is actually inspected or used as a merge base.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from saint.config import RuntimeConfig


DEFAULT_KEYWORDS = (
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
    "attn.out_proj.weight",
    "ffn.up_proj.weight",
    "ffn.down_proj.weight",
    "dim_gate.gate_net.0.weight",
)


@dataclass(frozen=True)
class DRMCheckpointTask:
    base_weights: dict[str, list[list[float]]]


def _metadata(config: RuntimeConfig) -> dict[str, Any]:
    return dict(config.metadata or {})


def _load_state_dict(checkpoint: str | Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to load drm_transformer checkpoints."
        ) from exc

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(state.get(key), dict):
                return state[key]
        return state
    raise ValueError("drm_transformer checkpoint did not contain a state dict")


def _keywords(metadata: dict[str, Any]) -> tuple[str, ...]:
    values = metadata.get("keywords", DEFAULT_KEYWORDS)
    if isinstance(values, list):
        return tuple(str(item) for item in values)
    return DEFAULT_KEYWORDS


def _matrix_payload(config: RuntimeConfig) -> dict[str, list[list[float]]]:
    metadata = _metadata(config)
    checkpoint = metadata.get("checkpoint")
    if not checkpoint:
        raise ValueError("drm_transformer adapter requires metadata.checkpoint")

    max_dim = int(metadata.get("max_dim", 64))
    max_matrices = int(metadata.get("max_matrices", 16))
    keywords = _keywords(metadata)
    state_dict = _load_state_dict(checkpoint)
    matrices: dict[str, list[list[float]]] = {}

    for name, tensor in state_dict.items():
        if not any(keyword in name for keyword in keywords):
            continue
        if not hasattr(tensor, "ndim") or int(tensor.ndim) != 2:
            continue
        rows = min(int(tensor.shape[0]), max_dim)
        cols = min(int(tensor.shape[1]), max_dim)
        matrices[name] = tensor[:rows, :cols].float().tolist()
        if len(matrices) >= max_matrices:
            break

    if not matrices:
        raise ValueError("no matching 2D drm_transformer matrices found")
    return matrices


def make_task(config: RuntimeConfig) -> DRMCheckpointTask:
    return DRMCheckpointTask(base_weights=_matrix_payload(config))


def run_method(config: RuntimeConfig):
    raise NotImplementedError(
        "drm_transformer training is planned for Phase 9; Phase 8 supports "
        "checkpoint inspection and reconstruction bases."
    )


def inspect_model(config: RuntimeConfig) -> dict:
    matrices = {
        name: {
            "rows": len(matrix),
            "cols": len(matrix[0]) if matrix else 0,
            "parameters": sum(len(row) for row in matrix),
        }
        for name, matrix in _matrix_payload(config).items()
    }
    return {
        "task": config.task,
        "adapter": "drm_transformer_checkpoint",
        "matrices": matrices,
        "total_parameters": sum(item["parameters"] for item in matrices.values()),
    }


__all__ = ["DRMCheckpointTask", "inspect_model", "make_task", "run_method"]
