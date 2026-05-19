"""Permanent state merge helpers for DRM-G graft payloads."""

from __future__ import annotations

from typing import Any

from saint.adapters.drm_grafting_decision import graft_equivalent_matrix


def _payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("format") == "drm_graft_sequence_payload":
        return list(payload.get("grafts", []))
    return [payload]


def merge_linear_grafts_into_state(torch, model, payload: dict[str, Any]) -> dict[str, Any]:
    modules = dict(model.named_modules())
    applied = []
    skipped = []
    with torch.no_grad():
        for graft in _payloads(payload):
            name = str(graft.get("target_module", ""))
            target = modules.get(name)
            if target is None or not hasattr(target, "weight"):
                skipped.append({"target_module": name, "reason": "target is not linear"})
                continue
            matrix = graft_equivalent_matrix(torch, graft).to(target.weight.device)
            weight = target.weight.detach()
            if weight.ndim != 2 or weight.shape[0] != matrix.shape[0]:
                skipped.append({"target_module": name, "reason": "shape mismatch"})
                continue
            target.weight.add_(matrix.transpose(0, 1).to(weight.dtype).matmul(weight))
            if getattr(target, "bias", None) is not None:
                bias = target.bias.detach()
                target.bias.add_(bias.matmul(matrix.to(bias.dtype)))
            applied.append({"target_module": name, "merge_kind": "linear_state_dict"})
    return {
        "format": "drm_graft_linear_state_merge",
        "applied": applied,
        "skipped": skipped,
        "state_dict_merge_supported": bool(applied) and not skipped,
    }


def evaluate_merged_payload(
    torch,
    model_cls,
    drm_config,
    payload: dict[str, Any],
    device: str,
    eval_inputs,
    eval_targets,
    seed: int,
):
    torch.manual_seed(seed)
    model = model_cls(drm_config).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    summary = merge_linear_grafts_into_state(torch, model, payload)
    _logits, loss = model(eval_inputs, eval_targets)
    if loss is None:
        raise ValueError("DRM model did not return loss")
    return float(loss.detach().cpu().item()), summary


__all__ = ["evaluate_merged_payload", "merge_linear_grafts_into_state"]
