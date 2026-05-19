"""DRM-G Marco 5E automatic phase decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from saint.adapters.drm_g_phase_success import evaluate_drm_g_phase_success


DEFAULT_INPUTS = {
    "marco5a": "runs/drm_g_marco5a_consolidate_linear/metrics.json",
    "marco5b": "runs/drm_g_marco5b_retention/summary.json",
    "marco5c_rows": "runs/drm_g_marco5c_phi_variants/results.json",
    "marco5c_summary": "runs/drm_g_marco5c_phi_variants/summary.json",
    "marco5d_summary": "runs/drm_g_marco5d_second_size/summary.json",
}


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _marco5a_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = manifest.get("metadata", {})
    state_merge = metadata.get("state_merge", {})
    return {
        "artifact_bytes": int(metadata.get("artifact_bytes", 0)),
        "saved_loss_abs_diff": float(metadata.get("saved_loss_abs_diff", float("inf"))),
        "validation_gain": float(metadata.get("validation_gain", 0.0)),
        "state_dict_merge_supported": bool(state_merge.get("state_dict_merge_supported", False)),
    }


def _markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# DRM-G Marco 5E Phase Decision",
        "",
        f"- passed: {decision['passed']}",
        f"- status: {decision['status']}",
        f"- passed_axes: {decision['passed_axes']} / {decision['required_axes']}",
        f"- reason: {decision['reason']}",
        "",
        "| axis | passed |",
        "|---|---:|",
    ]
    for axis, passed in decision["axes"].items():
        lines.append(f"| `{axis}` | {passed} |")
    lines.extend([
        "",
        "## Key Comparisons",
        "",
        "| item | method | gain | gain/param | positives |",
        "|---|---|---:|---:|---:|",
    ])
    for key in ("best_phi_summary_5c", "full_summary_5c", "best_phi_summary_5d", "full_summary_5d"):
        row = decision.get(key, {})
        if not row:
            lines.append(f"| {key} | missing | 0.000000 | 0.000000e+00 | 0/0 |")
            continue
        lines.append(
            "| {key} | {method} | {mean_gain:.6f} | {mean_gain_per_parameter:.6e} | "
            "{positive_runs}/{run_count} |".format(key=key, **row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/drm_g_marco5e_phase_decision")
    parser.add_argument("--marco5a", default=DEFAULT_INPUTS["marco5a"])
    parser.add_argument("--marco5b", default=DEFAULT_INPUTS["marco5b"])
    parser.add_argument("--marco5c-rows", default=DEFAULT_INPUTS["marco5c_rows"])
    parser.add_argument("--marco5c-summary", default=DEFAULT_INPUTS["marco5c_summary"])
    parser.add_argument("--marco5d-summary", default=DEFAULT_INPUTS["marco5d_summary"])
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decision = evaluate_drm_g_phase_success(
        marco5a=_marco5a_metrics(_read_json(args.marco5a)),
        marco5b=_read_json(args.marco5b),
        marco5c_rows=_read_json(args.marco5c_rows),
        marco5c_summary=_read_json(args.marco5c_summary),
        marco5d_summary=_read_json(args.marco5d_summary),
    )
    (out_dir / "phase_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (out_dir / "phase_decision.md").write_text(_markdown(decision), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
