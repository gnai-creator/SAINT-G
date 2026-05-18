"""Runtime configuration schema for SAINT experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    experiment_name: str = "saint_runtime_smoke"
    output_dir: str = "runs/runtime_smoke"
    task: str = "mini_transformer"
    method: str = "mini_saint_dynamic_delta"
    seed: int = 31
    delta_mode: str = "repeated"
    delta_scale: float = 3.0
    steps: int = 4
    parameter_budget: int = 48
    train_samples: int = 8
    test_samples: int = 4
    seq_len: int = 4
    vocab_size: int = 8
    d_model: int = 4
    sensitivity_method: str = "gradient_norm"
    vram_gb: float = 12.0
    metadata: dict[str, Any] = field(default_factory=dict)


def config_from_dict(data: dict[str, Any]) -> RuntimeConfig:
    allowed = RuntimeConfig.__dataclass_fields__.keys()
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValueError(f"unknown config fields: {unknown}")
    return RuntimeConfig(**data)


def load_config(path: str | Path) -> RuntimeConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return config_from_dict(data)


def save_config(config: RuntimeConfig, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(config.__dict__, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = ["RuntimeConfig", "config_from_dict", "load_config", "save_config"]
