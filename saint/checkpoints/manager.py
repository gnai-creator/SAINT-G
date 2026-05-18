"""Checkpoint helpers for SAINT runtime experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("checkpoint payload must be a JSON object")
    return data


def write_jsonl(path: str | Path, events: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(event, sort_keys=True) for event in events]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def checkpoint_payload(result, config, memory_plan) -> dict[str, Any]:
    metadata = dict(result.metadata)
    delta_payload = metadata.pop("delta_payload", None)
    payload = {
        "experiment_name": config.experiment_name,
        "method": result.name,
        "train_loss": result.train_loss,
        "test_loss": result.test_loss,
        "parameter_count": result.parameter_count,
        "optimizer_state_values": result.optimizer_state_values,
        "memory_plan": memory_plan.__dict__,
        "metadata": result.metadata,
        "has_delta_payload": delta_payload is not None,
    }
    if delta_payload is not None:
        payload["delta_payload"] = delta_payload
        payload["metadata"] = metadata
    return payload


def require_delta_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    payload = checkpoint.get("delta_payload")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint does not contain a delta_payload")
    return payload


__all__ = [
    "checkpoint_payload",
    "read_json",
    "require_delta_payload",
    "write_json",
    "write_jsonl",
]
