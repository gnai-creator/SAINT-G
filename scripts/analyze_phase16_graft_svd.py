#!/usr/bin/env python
"""Run Phase 16 Marco 4O-lite graft SVD anatomy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from saint.adapters.drm_grafting_svd_anatomy import (
    analyze_runs,
    render_markdown_report,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Completed run directory containing summary.json and composed_graft_checkpoint.pt. Repeat for multiple seeds.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where 4O-lite SVD anatomy artifacts will be written.",
    )
    parser.add_argument(
        "--include-unused-sample",
        type=int,
        default=2,
        help="Number of unused grafts per checkpoint to include as a comparison sample. Default: 2.",
    )
    parser.add_argument(
        "--include-effective-linear",
        action="store_true",
        help="Also analyze the linearized up@down matrix. This ignores activation and is diagnostic only.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of singular values to store in JSON per matrix. Default: 12.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows, summary = analyze_runs(
        args.run_dir,
        include_unused_sample=args.include_unused_sample,
        include_effective_linear=args.include_effective_linear,
        top_k=args.top_k,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "graft_svd_rows.json", rows)
    write_json(output_dir / "graft_svd_summary.json", summary)
    write_csv(output_dir / "graft_svd_table.csv", rows)
    (output_dir / "graft_svd_anatomy.md").write_text(
        render_markdown_report(summary, rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} SVD anatomy rows to {output_dir}")
    print(
        "runs={runs} accepted_rows={accepted} low_rank99_le16={low_rank} recommendations={recs}".format(
            runs=summary.get("run_count"),
            accepted=summary.get("accepted_matrix_rows"),
            low_rank=summary.get("accepted_low_rank_99_le16_rows"),
            recs=",".join(summary.get("recommendations", [])),
        )
    )


if __name__ == "__main__":
    main()
