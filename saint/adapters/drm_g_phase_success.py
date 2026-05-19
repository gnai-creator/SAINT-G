"""Automatic Marco 5 decision criteria for DRM-G."""

from __future__ import annotations

from typing import Any


def _best_phi(rows: list[dict[str, Any]]) -> dict[str, Any]:
    phi_rows = [row for row in rows if str(row.get("method", "")).startswith("phi_")]
    return max(phi_rows, key=lambda row: float(row.get("validation_gain", 0.0)))


def _best_full(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_rows = [row for row in rows if row.get("method") == "full_module_linear"]
    return max(full_rows, key=lambda row: float(row.get("validation_gain", 0.0)))


def _method_summary(summary: dict[str, Any], method: str) -> dict[str, Any] | None:
    for row in summary.get("method_summary", []):
        if row.get("method") == method:
            return row
    return None


def _best_phi_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    rows = [
        row for row in summary.get("method_summary", [])
        if str(row.get("method", "")).startswith("phi_")
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("mean_gain", 0.0)))


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for method in sorted({row.get("method") for row in rows}):
        subset = [row for row in rows if row.get("method") == method]
        summaries.append({
            "method": method,
            "mean_gain": sum(float(row.get("validation_gain", 0.0)) for row in subset) / len(subset),
            "mean_gain_per_parameter": sum(
                float(row.get("gain_per_parameter", 0.0)) for row in subset
            ) / len(subset),
            "positive_runs": sum(1 for row in subset if float(row.get("validation_gain", 0.0)) > 0.0),
            "run_count": len(subset),
            "params": int(subset[0].get("trainable_parameters", 0)),
        })
    return summaries


def evaluate_drm_g_phase_success(
    *,
    marco5a: dict[str, Any],
    marco5b: dict[str, Any],
    marco5c_rows: list[dict[str, Any]],
    marco5c_summary: dict[str, Any],
    marco5d_summary: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate Marco 5 with multi-axis criteria."""

    saved_diff = float(marco5a.get("saved_loss_abs_diff", float("inf")))
    artifact_bytes = int(marco5a.get("artifact_bytes", 0))
    retention_passed = bool(marco5b.get("phase_5b_passed", False))
    retention_runs = int(marco5b.get("retention_passed_runs", 0))
    best_phi_5c = _best_phi(marco5c_rows)
    best_full_5c = _best_full(marco5c_rows)
    if "method_summary" not in marco5c_summary:
        marco5c_summary = {**marco5c_summary, "method_summary": _summarize_rows(marco5c_rows)}
    phi_summary_5c = _best_phi_summary(marco5c_summary) or {}
    full_summary_5c = _method_summary(marco5c_summary, "full_module_linear") or {}
    phi_summary_5d = _best_phi_summary(marco5d_summary) or {}
    full_summary_5d = _method_summary(marco5d_summary, "full_module_linear") or {}
    best_case_win = float(best_phi_5c["validation_gain"]) > float(best_full_5c["validation_gain"])
    mean_multiseed_win = (
        float(phi_summary_5c.get("mean_gain", 0.0)) >= float(full_summary_5c.get("mean_gain", 0.0))
        or float(phi_summary_5d.get("mean_gain", 0.0)) >= float(full_summary_5d.get("mean_gain", 0.0))
    )
    stability_win = (
        int(phi_summary_5c.get("positive_runs", 0)) >= int(full_summary_5c.get("positive_runs", 0))
        and int(phi_summary_5d.get("positive_runs", 0)) >= int(full_summary_5d.get("positive_runs", 0))
    )
    checkpoint_size_win = artifact_bytes > 0 and artifact_bytes < 20_000_000
    memory_win = saved_diff <= 1e-6
    compression_win = int(phi_summary_5c.get("params", 0)) <= int(full_summary_5c.get("params", 0))
    axes = {
        "artifact_reproducible": saved_diff <= 1e-6,
        "retention_win": retention_passed and retention_runs >= 1,
        "best_case_win": best_case_win,
        "mean_multiseed_win": mean_multiseed_win,
        "stability_win": stability_win,
        "checkpoint_size_win": checkpoint_size_win,
        "memory_win": memory_win,
        "compression_win": compression_win,
    }
    passed_axes = sum(1 for value in axes.values() if value)
    passed = (
        axes["artifact_reproducible"]
        and axes["retention_win"]
        and axes["mean_multiseed_win"]
        and axes["stability_win"]
        and passed_axes >= 6
    )
    return {
        "passed": passed,
        "status": "partial_pass" if passed else "hold",
        "passed_axes": passed_axes,
        "required_axes": 6,
        "axes": axes,
        "best_phi_5c": best_phi_5c,
        "best_full_5c": best_full_5c,
        "best_phi_summary_5c": phi_summary_5c,
        "full_summary_5c": full_summary_5c,
        "best_phi_summary_5d": phi_summary_5d,
        "full_summary_5d": full_summary_5d,
        "reason": (
            "Marco 5 passes as partial/supportive evidence"
            if passed
            else "Marco 5 should hold; required multi-axis criteria not met"
        ),
    }


__all__ = ["evaluate_drm_g_phase_success"]
