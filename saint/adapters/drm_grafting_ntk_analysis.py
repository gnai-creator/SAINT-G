"""Offline NTK residual/saturation analysis for Phase 16 Marco 4N-A.

The Marco 4M probe records raw activation-gate sensitivity per candidate target.
This module joins those rows with stage/candidate outcomes and derives features
that are useful before trying any NTK-guided router:

- how many grafts a target already had before the stage;
- saturation-adjusted NTK score;
- NTK delta from the previous stage;
- candidate gains observed for each target.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


REQUIRED_ARTIFACTS = (
    "summary.json",
    "stage_metrics.json",
    "candidate_metrics.json",
    "ntk_activation_probe_metrics.json",
)


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"required 4N-A input artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_run_artifacts(
    run_dir: Path,
    summary: JsonDict,
    stage_metrics: list[JsonDict],
    candidate_metrics: list[JsonDict],
    ntk_rows: list[JsonDict],
) -> None:
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"run directory {run_dir} is missing required 4N-A artifact(s): {', '.join(missing)}"
        )
    if not summary or "composed_loss" not in summary or "accepted_grafts" not in summary:
        raise ValueError(f"run directory {run_dir} has empty or invalid summary.json")
    if not stage_metrics:
        raise ValueError(f"run directory {run_dir} has empty or invalid stage_metrics.json")
    if not candidate_metrics:
        raise ValueError(f"run directory {run_dir} has empty or invalid candidate_metrics.json")
    if not ntk_rows:
        raise ValueError(f"run directory {run_dir} has empty or invalid ntk_activation_probe_metrics.json")


def _stage_by_number(stage_metrics: Iterable[JsonDict]) -> dict[int, JsonDict]:
    return {int(row.get("stage", 0)): row for row in stage_metrics}


def _target_counts_before_stage(stage_metrics: list[JsonDict]) -> dict[int, Counter[str]]:
    counts_by_stage: dict[int, Counter[str]] = {}
    previous_target_by_graft: dict[str, str] = {}
    for stage in sorted(stage_metrics, key=lambda row: int(row.get("stage", 0))):
        stage_id = int(stage.get("stage", 0))
        counts_by_stage[stage_id] = Counter(str(target) for target in previous_target_by_graft.values())
        target_by_graft = stage.get("target_by_graft") or {}
        if target_by_graft:
            previous_target_by_graft = {str(graft_id): str(target) for graft_id, target in target_by_graft.items()}
    return counts_by_stage


def _best_candidate_by_stage_target(candidate_metrics: Iterable[JsonDict]) -> dict[tuple[int, str], JsonDict]:
    best: dict[tuple[int, str], JsonDict] = {}
    for row in candidate_metrics:
        if "stage" not in row or "candidate_target" not in row:
            continue
        key = (int(row["stage"]), str(row["candidate_target"]))
        current_gain = float(row.get("candidate_composed_gain", row.get("gain", 0.0)) or 0.0)
        best_gain = float(best.get(key, {}).get("candidate_composed_gain", best.get(key, {}).get("gain", float("-inf"))) or 0.0)
        if key not in best or current_gain > best_gain:
            best[key] = row
    return best


def build_joined_rows(
    *,
    seed: str | int,
    summary: JsonDict,
    stage_metrics: list[JsonDict],
    ntk_rows: list[JsonDict],
    candidate_metrics: list[JsonDict],
) -> list[JsonDict]:
    """Join NTK rows with stage/candidate outcomes and derived 4N-A features."""
    stages = _stage_by_number(stage_metrics)
    counts_before = _target_counts_before_stage(stage_metrics)
    best_candidate = _best_candidate_by_stage_target(candidate_metrics)
    ntk_by_stage_target = {
        (int(row.get("stage", 0)), str(row.get("target"))): row for row in ntk_rows
    }
    rows: list[JsonDict] = []
    for row in sorted(
        ntk_rows,
        key=lambda item: (
            int(item.get("stage", 0)),
            int(item.get("ntk_rank", 999)),
            str(item.get("target")),
        ),
    ):
        stage_id = int(row.get("stage", 0))
        target = str(row.get("target"))
        stage = stages.get(stage_id, {})
        previous_row = ntk_by_stage_target.get((stage_id - 1, target))
        ntk_score = float(row.get("ntk_activation_score", 0.0) or 0.0)
        previous_ntk_score = None
        ntk_delta = None
        ntk_delta_pct = None
        if previous_row is not None:
            previous_ntk_score = float(previous_row.get("ntk_activation_score", 0.0) or 0.0)
            ntk_delta = ntk_score - previous_ntk_score
            if previous_ntk_score:
                ntk_delta_pct = ntk_delta / previous_ntk_score
        accepted_before = int(counts_before.get(stage_id, Counter()).get(target, 0))
        candidate = best_candidate.get((stage_id, target), {})
        candidate_gain = float(candidate.get("candidate_composed_gain", candidate.get("gain", 0.0)) or 0.0)
        selected_target = target == str(stage.get("selected_target"))
        joined = {
            "seed": str(seed),
            "stage": stage_id,
            "target": target,
            "selected_target": selected_target,
            "stage_decision": stage.get("decision"),
            "stage_gain": float(stage.get("stage_gain", 0.0) or 0.0),
            "stage_selected_target": stage.get("selected_target"),
            "ntk_activation_score": ntk_score,
            "mean_ntk_activation_score": row.get("mean_ntk_activation_score"),
            "ntk_rank": int(row.get("ntk_rank", 999)),
            "top_channel": row.get("top_channel"),
            "top_channel_score": row.get("top_channel_score"),
            "previous_stage_ntk_activation_score": previous_ntk_score,
            "ntk_delta_from_previous_stage": ntk_delta,
            "ntk_delta_pct_from_previous_stage": ntk_delta_pct,
            "accepted_grafts_on_target_before_stage": accepted_before,
            "saturation_adjusted_ntk": ntk_score / (1 + accepted_before),
            "anti_saturation_penalty": accepted_before,
            "best_candidate_composed_gain": candidate_gain,
            "best_candidate_score": candidate.get("candidate_score"),
            "best_candidate_pass": candidate.get("pass"),
            "best_candidate_tag": candidate.get("candidate_tag"),
            "run_composed_loss": summary.get("composed_loss"),
            "run_accepted_grafts": summary.get("accepted_grafts"),
        }
        rows.append(joined)
    return rows


def summarize_routing_signal(rows: list[JsonDict]) -> JsonDict:
    selected = [row for row in rows if row.get("selected_target")]
    raw_top1_selected = [row for row in selected if int(row.get("ntk_rank", 999)) == 1]
    selected_not_top1 = [row for row in selected if int(row.get("ntk_rank", 999)) != 1]
    recommendations: list[str] = []
    if selected_not_top1:
        recommendations.append("reject_raw_ntk_prefilter")
        recommendations.append("test_saturation_adjusted_ntk")
        recommendations.append("test_residual_delta_ntk")
    else:
        recommendations.append("raw_ntk_prefilter_still_candidate")
    if any(int(row.get("accepted_grafts_on_target_before_stage", 0) or 0) > 0 for row in rows):
        recommendations.append("include_target_saturation_features")
    return {
        "selected_target_count": len(selected),
        "raw_ntk_top1_selected_count": len(raw_top1_selected),
        "selected_targets_not_top1_count": len(selected_not_top1),
        "raw_ntk_top1_selected_rate": (len(raw_top1_selected) / len(selected)) if selected else None,
        "recommendations": recommendations,
    }


def analyze_run(run_dir: str | Path, *, seed: str | int | None = None) -> tuple[list[JsonDict], JsonDict]:
    root = Path(run_dir)
    summary = _load_json(root / "summary.json")
    stage_metrics = _load_json(root / "stage_metrics.json")
    candidate_metrics = _load_json(root / "candidate_metrics.json")
    ntk_rows = _load_json(root / "ntk_activation_probe_metrics.json")
    _validate_run_artifacts(root, summary, stage_metrics, candidate_metrics, ntk_rows)
    if seed is None:
        seed = root.name.rsplit("seed", 1)[-1] if "seed" in root.name else "unknown"
    rows = build_joined_rows(
        seed=seed,
        summary=summary,
        stage_metrics=stage_metrics,
        ntk_rows=ntk_rows,
        candidate_metrics=candidate_metrics,
    )
    summary_row = {
        "seed": str(seed),
        "run_dir": str(root),
        "base_loss": summary.get("base_loss"),
        "composed_loss": summary.get("composed_loss"),
        "accumulated_gain": summary.get("accumulated_gain"),
        "accepted_groups": summary.get("accepted_groups"),
        "accepted_grafts": summary.get("accepted_grafts"),
        "recompose_abs_diff": summary.get("recompose_abs_diff"),
        **summarize_routing_signal(rows),
    }
    return rows, summary_row


def write_csv(path: str | Path, rows: list[JsonDict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown_report(rows: list[JsonDict], run_summaries: list[JsonDict]) -> str:
    lines = [
        "# Phase 16 Marco 4N-A - NTK Residual/Saturation Analysis",
        "",
        "Status: generated offline from Marco 4M artifacts.",
        "",
        "## Run Summary",
        "",
        "| seed | composed_loss | accepted_grafts | selected top-1 rate | recommendation |",
        "|---|---:|---:|---:|---|",
    ]
    for summary in run_summaries:
        rate = summary.get("raw_ntk_top1_selected_rate")
        rate_text = "n/a" if rate is None else f"{rate:.3f}"
        rec = ", ".join(summary.get("recommendations", []))
        lines.append(
            f"| {summary.get('seed')} | {summary.get('composed_loss')} | {summary.get('accepted_grafts')} | {rate_text} | {rec} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Raw NTK should remain diagnostic-only when selected stage targets are not the raw NTK top-1 target.",
        "The next routing candidate should use residual, novelty, or saturation-aware features instead of a raw top-1 prefilter.",
        "",
        "## Joined Stage/Target Rows",
        "",
        "| seed | stage | target | ntk_rank | selected | decision | accepted_before | raw_ntk | saturation_adjusted_ntk | ntk_delta | best_candidate_gain |",
        "|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            "| {seed} | {stage} | {target} | {ntk_rank} | {selected} | {decision} | {accepted_before} | {raw:.6g} | {sat:.6g} | {delta} | {gain:.6g} |".format(
                seed=row.get("seed"),
                stage=row.get("stage"),
                target=row.get("target"),
                ntk_rank=row.get("ntk_rank"),
                selected=str(bool(row.get("selected_target"))).lower(),
                decision=row.get("stage_decision"),
                accepted_before=row.get("accepted_grafts_on_target_before_stage"),
                raw=float(row.get("ntk_activation_score", 0.0) or 0.0),
                sat=float(row.get("saturation_adjusted_ntk", 0.0) or 0.0),
                delta="n/a" if row.get("ntk_delta_from_previous_stage") is None else f"{float(row['ntk_delta_from_previous_stage']):.6g}",
                gain=float(row.get("best_candidate_composed_gain", 0.0) or 0.0),
            )
        )
    lines.append("")
    return "\n".join(lines)
