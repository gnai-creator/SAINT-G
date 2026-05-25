#!/usr/bin/env python
"""Analyze Phase 16 Marco 4M NTK probe artifacts for Marco 4N-A.

This is an offline analysis script. It does not train or mutate checkpoints.
It joins per-target NTK rows with stage/candidate outcomes and writes:

- ntk_candidate_joined_metrics.json
- ntk_stage_feature_table.csv
- ntk_routing_analysis.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from saint.adapters.drm_grafting_ntk_analysis import (
    analyze_run,
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
        help="Marco 4M run directory. Repeat for multiple seeds.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where 4N-A offline analysis artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    all_rows = []
    run_summaries = []
    for run_dir in args.run_dir:
        rows, summary = analyze_run(run_dir)
        all_rows.extend(rows)
        run_summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "ntk_candidate_joined_metrics.json", all_rows)
    write_json(output_dir / "ntk_run_summaries.json", run_summaries)
    write_csv(output_dir / "ntk_stage_feature_table.csv", all_rows)
    (output_dir / "ntk_routing_analysis.md").write_text(
        render_markdown_report(all_rows, run_summaries),
        encoding="utf-8",
    )
    print(f"wrote {len(all_rows)} joined rows to {output_dir}")
    for summary in run_summaries:
        print(
            "seed={seed} composed_loss={loss} accepted_grafts={grafts} recommendations={recs}".format(
                seed=summary.get("seed"),
                loss=summary.get("composed_loss"),
                grafts=summary.get("accepted_grafts"),
                recs=",".join(summary.get("recommendations", [])),
            )
        )


if __name__ == "__main__":
    main()
