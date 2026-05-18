"""Sparse delta payload helpers for Hugging Face SAINT experiments."""

from __future__ import annotations

from typing import Any


def delta_payload(deltas, coordinates, shapes: dict[str, list[int]]) -> dict[str, Any]:
    sparse = {}
    for name, delta in deltas.items():
        if name not in shapes:
            continue
        rows, cols = int(shapes[name][0]), int(shapes[name][1])
        entries = []
        row_indices, col_indices, *extra = coordinates[name]
        values = delta
        if extra:
            prototype, scale_ids = extra
            values = (delta[scale_ids] * prototype).sum(dim=-1)
        for row, col, value in zip(
            row_indices.detach().cpu().tolist(),
            col_indices.detach().cpu().tolist(),
            values.detach().cpu().tolist(),
        ):
            if row < rows and col < cols and abs(float(value)) > 0.0:
                entries.append([int(row), int(col), float(value)])
        if entries:
            sparse[name] = entries
    return {"format": "saint_sparse_delta", "shapes": shapes, "values": sparse}


def delta_structure_metadata(deltas, coordinates) -> dict[str, Any]:
    structured = {}
    for name, delta in deltas.items():
        rows, _cols, *extra = coordinates[name]
        entry = {"values": int(rows.numel()), "parameters": int(delta.numel())}
        if extra:
            prototype, scale_ids = extra
            entry.update(
                {
                    "prototype_count": int(scale_ids.shape[1])
                    if getattr(scale_ids, "ndim", 0) > 1
                    else 1,
                    "prototype_shape": [int(item) for item in prototype.shape],
                    "scale_id_shape": [int(item) for item in scale_ids.shape],
                    "scale_count": int(delta.numel()),
                    "max_scale_id": int(scale_ids.max().item()),
                }
            )
        structured[name] = entry
    return structured


__all__ = ["delta_payload", "delta_structure_metadata"]
