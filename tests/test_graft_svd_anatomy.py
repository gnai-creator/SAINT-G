import tempfile
import unittest
from pathlib import Path


class GraftSVDAnatomyTests(unittest.TestCase):
    def test_spectral_metrics_detect_low_rank_matrix(self):
        import torch
        from saint.adapters.drm_grafting_svd_anatomy import _spectral_metrics

        left = torch.randn(8, 2)
        right = torch.randn(2, 6)
        matrix = left @ right

        metrics = _spectral_metrics(matrix, name="toy")

        self.assertLessEqual(metrics["energy_rank_99"], 2)
        self.assertLessEqual(metrics["energy_rank_9999"], 2)
        self.assertGreater(metrics["energy_top1"], 0.0)

    def test_analyze_run_reads_composed_graft_checkpoint(self):
        import json
        import torch
        from saint.adapters.drm_grafting_svd_anatomy import analyze_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = {
                "base_loss": 3.0,
                "composed_loss": 2.9,
                "accumulated_gain": 0.1,
                "accepted_grafts": 1,
                "accepted_graft_ids": [0],
                "target_by_graft": {"0": "blocks.4"},
                "recompose_abs_diff": 0.0,
                "graft_count": 2,
            }
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            checkpoint = {
                "format": "test",
                "grafts": [
                    {
                        "up": torch.randn(4, 6),
                        "down": torch.randn(6, 4),
                        "scale": 1.0,
                        "activation": "silu",
                    },
                    {
                        "up": torch.randn(4, 6),
                        "down": torch.randn(6, 4),
                        "scale": 1.0,
                        "activation": "silu",
                    },
                ],
            }
            torch.save(checkpoint, root / "composed_graft_checkpoint.pt")

            rows, run_summary = analyze_run(root, include_unused_sample=1)

        self.assertEqual(run_summary["accepted_grafts"], 1)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["graft_status"] for row in rows}, {"accepted", "unused_sample"})


if __name__ == "__main__":
    unittest.main()
