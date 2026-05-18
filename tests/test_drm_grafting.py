from pathlib import Path
import unittest

from saint.adapters.drm_grafting import _read_simple_yaml
from saint.config import RuntimeConfig


class DRMGraftingTests(unittest.TestCase):
    def test_reads_baseline_3_5m_config(self):
        root = Path(__file__).resolve().parents[2]
        config_path = root / "drm_transformer" / "configs" / "baselines" / "small_3.5M.yaml"
        if not config_path.exists():
            self.skipTest("external drm_transformer baseline config not available")

        data = _read_simple_yaml(config_path)

        self.assertEqual(data["vocab_size"], 50000)
        self.assertEqual(data["d_model"], 64)
        self.assertEqual(data["n_layers"], 4)
        self.assertEqual(data["d_ff"], 256)

    def test_inspects_drm_graft_baseline_when_available(self):
        try:
            from saint.adapters.drm_grafting import inspect_graft_model
        except RuntimeError as exc:
            self.skipTest(str(exc))

        root = Path(__file__).resolve().parents[2]
        drm_root = root / "drm_transformer"
        if not drm_root.exists():
            self.skipTest("external drm_transformer repo not available")

        config = RuntimeConfig(
            task="drm_transformer",
            method="drm_g_saint_phi_graft",
            metadata={"drm_root": str(drm_root), "phi_rank": 8},
        )

        summary = inspect_graft_model(config)

        self.assertEqual(summary["adapter"], "drm_grafting")
        self.assertEqual(summary["d_model"], 64)
        self.assertEqual(summary["n_layers"], 4)
        self.assertEqual(summary["graft_parameters"], 64)
        self.assertGreater(summary["base_parameters"], 3_000_000)


if __name__ == "__main__":
    unittest.main()
