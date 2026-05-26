"""Phase 16 Marco 4O-lite graft SVD anatomy.

The 4O-lite diagnostic inspects completed composed graft checkpoints without
training. It estimates singular spectra and effective ranks for trained graft
matrices so we can tell whether accepted graft blocks use broad capacity or are
compressible to low-rank structure.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

JsonDict = dict[str, Any]

ENERGY_THRESHOLDS = (0.90, 0.95, 0.99, 0.999, 0.9999)


def _load_torch():
    import torch

    return torch


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"required 4O-lite input artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_from_dir(run_dir: Path) -> str:
    return run_dir.name.rsplit("seed", 1)[-1] if "seed" in run_dir.name else "unknown"


def _rank_for_energy(cumulative_energy: list[float], threshold: float) -> int:
    for idx, value in enumerate(cumulative_energy, start=1):
        if value >= threshold:
            return idx
    return len(cumulative_energy)


def _spectral_metrics(matrix: Any, *, name: str, top_k: int = 12) -> JsonDict:
    """Compute SVD-derived metrics using the smaller Gram matrix.

    For SAINT-G graft matrices one side is usually d_model=96 and the other is a
    large hidden dimension. Eigen-decomposing the 96x96 Gram matrix is much
    cheaper than asking for the full rectangular SVD while yielding the same
    non-zero singular values.
    """
    torch = _load_torch()
    x = matrix.detach().cpu().double()
    shape = list(x.shape)
    if x.ndim != 2:
        raise ValueError(f"expected 2D matrix for {name}, got shape={shape}")
    rows, cols = int(x.shape[0]), int(x.shape[1])
    gram = x @ x.T if rows <= cols else x.T @ x
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    singular_values = eigenvalues.sqrt().flip(0)
    raw_sv = [float(value) for value in singular_values.tolist()]
    max_sv = raw_sv[0] if raw_sv else 0.0
    cutoff = max(max_sv * 1e-10, 1e-12)
    sv = [value for value in raw_sv if value > cutoff]
    if not sv:
        return {
            "matrix_name": name,
            "shape": shape,
            "rank_full": 0,
            "frobenius_norm": 0.0,
            "spectral_norm": 0.0,
            "stable_rank": 0.0,
            "condition_number": None,
            "top_singular_values": [],
            "energy_rank_90": 0,
            "energy_rank_95": 0,
            "energy_rank_99": 0,
            "energy_rank_999": 0,
            "energy_rank_9999": 0,
            "energy_top1": 0.0,
            "energy_top4": 0.0,
            "energy_top8": 0.0,
            "energy_top16": 0.0,
            "energy_top32": 0.0,
        }
    energy_values = [value * value for value in sv]
    total_energy = sum(energy_values)
    cumulative = []
    running = 0.0
    for value in energy_values:
        running += value
        cumulative.append(running / total_energy if total_energy else 0.0)
    spectral_norm = sv[0]
    smallest = sv[-1]
    return {
        "matrix_name": name,
        "shape": shape,
        "rank_full": len(sv),
        "frobenius_norm": math.sqrt(total_energy),
        "spectral_norm": spectral_norm,
        "stable_rank": total_energy / (spectral_norm * spectral_norm) if spectral_norm else 0.0,
        "condition_number": spectral_norm / smallest if smallest else None,
        "top_singular_values": sv[:top_k],
        "energy_rank_90": _rank_for_energy(cumulative, 0.90),
        "energy_rank_95": _rank_for_energy(cumulative, 0.95),
        "energy_rank_99": _rank_for_energy(cumulative, 0.99),
        "energy_rank_999": _rank_for_energy(cumulative, 0.999),
        "energy_rank_9999": _rank_for_energy(cumulative, 0.9999),
        "energy_top1": cumulative[min(0, len(cumulative) - 1)],
        "energy_top4": cumulative[min(3, len(cumulative) - 1)],
        "energy_top8": cumulative[min(7, len(cumulative) - 1)],
        "energy_top16": cumulative[min(15, len(cumulative) - 1)],
        "energy_top32": cumulative[min(31, len(cumulative) - 1)],
    }


def _selected_graft_ids(summary: JsonDict, *, include_unused_sample: int) -> list[int]:
    accepted = [int(value) for value in summary.get("accepted_graft_ids", [])]
    accepted_set = set(accepted)
    unused = []
    if include_unused_sample > 0:
        total = int(summary.get("graft_count", 0) or 0)
        if not total:
            # Composed checkpoint metadata does not always expose graft_count.
            total = 24
        unused = [idx for idx in range(total) if idx not in accepted_set][:include_unused_sample]
    return accepted + unused


def _status_for_graft(graft_id: int, accepted_graft_ids: set[int]) -> str:
    return "accepted" if graft_id in accepted_graft_ids else "unused_sample"


def analyze_run(
    run_dir: str | Path,
    *,
    include_unused_sample: int = 2,
    include_effective_linear: bool = False,
    top_k: int = 12,
) -> tuple[list[JsonDict], JsonDict]:
    torch = _load_torch()
    root = Path(run_dir)
    summary = _load_json(root / "summary.json")
    checkpoint_path = root / "composed_graft_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing composed graft checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    grafts = checkpoint.get("grafts") if isinstance(checkpoint, dict) else None
    if not isinstance(grafts, list) or not grafts:
        raise ValueError(f"checkpoint {checkpoint_path} does not contain a non-empty graft list")

    seed = _seed_from_dir(root)
    accepted_ids = {int(value) for value in summary.get("accepted_graft_ids", [])}
    target_by_graft = {str(k): str(v) for k, v in (summary.get("target_by_graft") or {}).items()}
    rows: list[JsonDict] = []
    for graft_id in _selected_graft_ids(summary, include_unused_sample=include_unused_sample):
        if graft_id >= len(grafts):
            continue
        graft = grafts[graft_id]
        status = _status_for_graft(graft_id, accepted_ids)
        for matrix_name in ("up", "down"):
            if matrix_name not in graft:
                continue
            metrics = _spectral_metrics(graft[matrix_name], name=matrix_name, top_k=top_k)
            rows.append({
                "seed": seed,
                "run_dir": str(root),
                "checkpoint": str(checkpoint_path),
                "graft_id": graft_id,
                "graft_status": status,
                "target": target_by_graft.get(str(graft_id)),
                "scale": float(graft.get("scale", 0.0)) if hasattr(graft.get("scale"), "item") else graft.get("scale"),
                "activation": graft.get("activation"),
                **metrics,
            })
        if include_effective_linear and "up" in graft and "down" in graft:
            # Linearized adapter without activation: x @ up @ down. This is only
            # diagnostic because the trained graft uses an activation in between.
            effective = graft["up"].detach().cpu().double() @ graft["down"].detach().cpu().double()
            metrics = _spectral_metrics(effective, name="effective_up_down", top_k=top_k)
            rows.append({
                "seed": seed,
                "run_dir": str(root),
                "checkpoint": str(checkpoint_path),
                "graft_id": graft_id,
                "graft_status": status,
                "target": target_by_graft.get(str(graft_id)),
                "scale": float(graft.get("scale", 0.0)) if hasattr(graft.get("scale"), "item") else graft.get("scale"),
                "activation": graft.get("activation"),
                **metrics,
            })
    run_summary = {
        "seed": seed,
        "run_dir": str(root),
        "checkpoint": str(checkpoint_path),
        "base_loss": summary.get("base_loss"),
        "composed_loss": summary.get("composed_loss"),
        "accumulated_gain": summary.get("accumulated_gain"),
        "accepted_grafts": summary.get("accepted_grafts"),
        "accepted_graft_ids": sorted(accepted_ids),
        "target_by_graft": target_by_graft,
        "recompose_abs_diff": summary.get("recompose_abs_diff"),
        "analyzed_grafts": sorted({int(row["graft_id"]) for row in rows}),
        "include_unused_sample": include_unused_sample,
        "include_effective_linear": include_effective_linear,
    }
    return rows, run_summary


def summarize(rows: list[JsonDict], run_summaries: list[JsonDict]) -> JsonDict:
    grouped: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("matrix_name")), str(row.get("graft_status")))].append(row)
    matrix_summaries = []
    for (matrix_name, status), group in sorted(grouped.items()):
        matrix_summaries.append({
            "matrix_name": matrix_name,
            "graft_status": status,
            "count": len(group),
            "mean_energy_rank_99": mean([float(row.get("energy_rank_99", 0) or 0) for row in group]),
            "mean_energy_rank_999": mean([float(row.get("energy_rank_999", 0) or 0) for row in group]),
            "mean_stable_rank": mean([float(row.get("stable_rank", 0.0) or 0.0) for row in group]),
            "mean_energy_top8": mean([float(row.get("energy_top8", 0.0) or 0.0) for row in group]),
            "mean_energy_top16": mean([float(row.get("energy_top16", 0.0) or 0.0) for row in group]),
            "mean_energy_top32": mean([float(row.get("energy_top32", 0.0) or 0.0) for row in group]),
        })
    accepted = [row for row in rows if row.get("graft_status") == "accepted"]
    high_compression = [
        row for row in accepted
        if int(row.get("energy_rank_99", 999) or 999) <= 16
        and str(row.get("matrix_name")) in {"up", "down"}
    ]
    return {
        "phase": "16",
        "marco": "4o_lite_graft_svd_anatomy",
        "run_count": len(run_summaries),
        "seeds": [summary.get("seed") for summary in run_summaries],
        "row_count": len(rows),
        "accepted_matrix_rows": len(accepted),
        "accepted_low_rank_99_le16_rows": len(high_compression),
        "matrix_summaries": matrix_summaries,
        "run_summaries": run_summaries,
        "recommendations": _recommendations(matrix_summaries),
    }


def _recommendations(matrix_summaries: list[JsonDict]) -> list[str]:
    accepted = [row for row in matrix_summaries if row.get("graft_status") == "accepted"]
    max_rank99 = max([float(row.get("mean_energy_rank_99", 0.0) or 0.0) for row in accepted], default=0.0)
    mean_top16 = mean([float(row.get("mean_energy_top16", 0.0) or 0.0) for row in accepted]) if accepted else 0.0
    recs = ["use_svd_anatomy_as_diagnostic_before_more_routing_sweeps"]
    if max_rank99 <= 16 or mean_top16 >= 0.99:
        recs.append("test_low_rank_adapter_or_truncated_graft_baseline")
    else:
        recs.append("do_not_assume_strong_low_rank_compressibility_yet")
    recs.append("compare_accepted_vs_unused_spectra_before_removing_capacity")
    return recs


def analyze_runs(
    run_dirs: list[str | Path],
    *,
    include_unused_sample: int = 2,
    include_effective_linear: bool = False,
    top_k: int = 12,
) -> tuple[list[JsonDict], JsonDict]:
    rows: list[JsonDict] = []
    run_summaries: list[JsonDict] = []
    for run_dir in run_dirs:
        run_rows, run_summary = analyze_run(
            run_dir,
            include_unused_sample=include_unused_sample,
            include_effective_linear=include_effective_linear,
            top_k=top_k,
        )
        rows.extend(run_rows)
        run_summaries.append(run_summary)
    return rows, summarize(rows, run_summaries)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[JsonDict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row if key != "top_singular_values"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def render_markdown_report(summary: JsonDict, rows: list[JsonDict]) -> str:
    lines = [
        "# Phase 16 Marco 4O-lite - Graft SVD Anatomy",
        "",
        "Status: completed offline from composed graft checkpoints; no training performed.",
        "",
        "## Run Summary",
        "",
        "| seed | composed_loss | gain | accepted_grafts | analyzed_grafts | recompose_abs_diff |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for run in summary.get("run_summaries", []):
        lines.append(
            "| {seed} | {loss} | {gain} | {grafts} | {analyzed} | {recompose} |".format(
                seed=run.get("seed"),
                loss=run.get("composed_loss"),
                gain=run.get("accumulated_gain"),
                grafts=run.get("accepted_grafts"),
                analyzed=run.get("analyzed_grafts"),
                recompose=run.get("recompose_abs_diff"),
            )
        )
    lines.extend([
        "",
        "## Matrix Summary",
        "",
        "| matrix | status | count | mean rank@99% | mean rank@99.9% | mean stable rank | mean energy top-8 | mean energy top-16 | mean energy top-32 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary.get("matrix_summaries", []):
        lines.append(
            "| {matrix} | {status} | {count} | {r99:.3f} | {r999:.3f} | {sr:.3f} | {e8:.6f} | {e16:.6f} | {e32:.6f} |".format(
                matrix=row.get("matrix_name"),
                status=row.get("graft_status"),
                count=row.get("count"),
                r99=float(row.get("mean_energy_rank_99", 0.0) or 0.0),
                r999=float(row.get("mean_energy_rank_999", 0.0) or 0.0),
                sr=float(row.get("mean_stable_rank", 0.0) or 0.0),
                e8=float(row.get("mean_energy_top8", 0.0) or 0.0),
                e16=float(row.get("mean_energy_top16", 0.0) or 0.0),
                e32=float(row.get("mean_energy_top32", 0.0) or 0.0),
            )
        )
    lines.extend([
        "",
        "## Accepted Graft Rows",
        "",
        "| seed | graft | target | matrix | rank@99% | rank@99.9% | stable_rank | energy top-8 | energy top-16 | energy top-32 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted([r for r in rows if r.get("graft_status") == "accepted"], key=lambda item: (str(item.get("seed")), int(item.get("graft_id", 0)), str(item.get("matrix_name")))):
        lines.append(
            "| {seed} | {graft} | {target} | {matrix} | {r99} | {r999} | {sr:.3f} | {e8:.6f} | {e16:.6f} | {e32:.6f} |".format(
                seed=row.get("seed"),
                graft=row.get("graft_id"),
                target=row.get("target"),
                matrix=row.get("matrix_name"),
                r99=row.get("energy_rank_99"),
                r999=row.get("energy_rank_999"),
                sr=float(row.get("stable_rank", 0.0) or 0.0),
                e8=float(row.get("energy_top8", 0.0) or 0.0),
                e16=float(row.get("energy_top16", 0.0) or 0.0),
                e32=float(row.get("energy_top32", 0.0) or 0.0),
            )
        )
    lines.extend([
        "",
        "## Recommendations",
        "",
    ])
    for rec in summary.get("recommendations", []):
        lines.append(f"- {rec}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "ENERGY_THRESHOLDS",
    "analyze_run",
    "analyze_runs",
    "render_markdown_report",
    "summarize",
    "write_csv",
    "write_json",
]
