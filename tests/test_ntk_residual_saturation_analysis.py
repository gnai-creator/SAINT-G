import unittest


class NTKResidualSaturationAnalysisTests(unittest.TestCase):
    def test_join_rows_add_saturation_and_delta_features(self):
        from saint.adapters.drm_grafting_ntk_analysis import build_joined_rows

        summary = {"accepted_grafts": 5, "composed_loss": 10.0}
        stage_metrics = [
            {
                "stage": 1,
                "selected_target": "blocks.4",
                "decision": "approved",
                "stage_gain": 0.5,
                "target_by_graft": {"0": "blocks.4", "1": "blocks.4", "2": "blocks.4", "3": "blocks.4"},
            },
            {
                "stage": 2,
                "selected_target": "blocks.2",
                "decision": "approved",
                "stage_gain": 0.1,
                "target_by_graft": {"0": "blocks.4", "1": "blocks.4", "2": "blocks.4", "3": "blocks.4", "4": "blocks.2"},
            },
        ]
        ntk_rows = [
            {"stage": 1, "target": "blocks.4", "ntk_activation_score": 4.0, "ntk_rank": 1},
            {"stage": 1, "target": "blocks.2", "ntk_activation_score": 2.0, "ntk_rank": 2},
            {"stage": 2, "target": "blocks.4", "ntk_activation_score": 5.0, "ntk_rank": 1},
            {"stage": 2, "target": "blocks.2", "ntk_activation_score": 3.0, "ntk_rank": 2},
        ]
        candidate_metrics = [
            {"stage": 2, "candidate_target": "blocks.2", "pass": "deep", "candidate_composed_gain": 0.1, "candidate_score": 0.1},
            {"stage": 2, "candidate_target": "blocks.4", "pass": "deep", "candidate_composed_gain": 0.0, "candidate_score": 0.0},
        ]

        rows = build_joined_rows(
            seed="42",
            summary=summary,
            stage_metrics=stage_metrics,
            ntk_rows=ntk_rows,
            candidate_metrics=candidate_metrics,
        )

        stage2_blocks4 = next(row for row in rows if row["stage"] == 2 and row["target"] == "blocks.4")
        stage2_blocks2 = next(row for row in rows if row["stage"] == 2 and row["target"] == "blocks.2")

        self.assertEqual(stage2_blocks4["accepted_grafts_on_target_before_stage"], 4)
        self.assertEqual(stage2_blocks2["accepted_grafts_on_target_before_stage"], 0)
        self.assertAlmostEqual(stage2_blocks4["saturation_adjusted_ntk"], 1.0)
        self.assertAlmostEqual(stage2_blocks2["saturation_adjusted_ntk"], 3.0)
        self.assertAlmostEqual(stage2_blocks4["ntk_delta_from_previous_stage"], 1.0)
        self.assertAlmostEqual(stage2_blocks2["ntk_delta_from_previous_stage"], 1.0)
        self.assertEqual(stage2_blocks2["selected_target"], True)
        self.assertEqual(stage2_blocks2["stage_decision"], "approved")
        self.assertAlmostEqual(stage2_blocks2["best_candidate_composed_gain"], 0.1)

    def test_recommendation_rejects_raw_ntk_when_selected_target_is_ranked_last(self):
        from saint.adapters.drm_grafting_ntk_analysis import summarize_routing_signal

        rows = [
            {"stage": 1, "target": "blocks.4", "ntk_rank": 1, "selected_target": True, "stage_decision": "approved"},
            {"stage": 1, "target": "blocks.2", "ntk_rank": 3, "selected_target": False, "stage_decision": "approved"},
            {"stage": 2, "target": "blocks.4", "ntk_rank": 1, "selected_target": False, "stage_decision": "approved"},
            {"stage": 2, "target": "blocks.2", "ntk_rank": 3, "selected_target": True, "stage_decision": "approved"},
        ]

        summary = summarize_routing_signal(rows)

        self.assertEqual(summary["selected_target_count"], 2)
        self.assertEqual(summary["raw_ntk_top1_selected_count"], 1)
        self.assertEqual(summary["selected_targets_not_top1_count"], 1)
        self.assertIn("reject_raw_ntk_prefilter", summary["recommendations"])


if __name__ == "__main__":
    unittest.main()
