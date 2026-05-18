import tempfile
from pathlib import Path
import unittest

from saint.cli import main as cli_main
from saint.config import RuntimeConfig, config_from_dict, load_config, save_config
from saint.memory import estimate_runtime_memory
from saint.runtime import (
    estimate_runtime,
    inspect_runtime,
    merge_runtime,
    resume_runtime,
    train_runtime,
)


class RuntimePhase7Tests(unittest.TestCase):
    def test_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config = RuntimeConfig(output_dir=str(Path(tmp) / "run"), steps=1)

            save_config(config, path)
            loaded = load_config(path)

            self.assertEqual(loaded.output_dir, config.output_dir)
            self.assertEqual(loaded.steps, 1)

    def test_config_rejects_unknown_fields(self):
        with self.assertRaises(ValueError):
            config_from_dict({"missing": True})

    def test_memory_plan_fits_small_runtime(self):
        plan = estimate_runtime_memory(RuntimeConfig(parameter_budget=24))

        self.assertTrue(plan.fits_budget)
        self.assertEqual(plan.trainable_parameters, 24)

    def test_inspect_and_estimate_runtime(self):
        config = RuntimeConfig(parameter_budget=24)

        inspected = inspect_runtime(config)
        estimated = estimate_runtime(config)

        self.assertIn("matrices", inspected)
        self.assertTrue(estimated["fits_budget"])

    def test_train_resume_and_merge_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = RuntimeConfig(output_dir=str(Path(tmp) / "run"), steps=1, parameter_budget=24)

            result = train_runtime(config)
            resumed = resume_runtime(config.output_dir)
            merged = merge_runtime(config.output_dir)

            self.assertEqual(result["experiment_name"], config.experiment_name)
            self.assertTrue((Path(config.output_dir) / "checkpoint.json").exists())
            self.assertTrue(resumed["resumed"])
            self.assertTrue(merged["merged"])

    def test_cli_commands_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            run_dir = Path(tmp) / "run"
            save_config(RuntimeConfig(output_dir=str(run_dir), steps=1, parameter_budget=24), config_path)

            self.assertEqual(cli_main(["inspect", "--config", str(config_path)]), 0)
            self.assertEqual(cli_main(["estimate", "--config", str(config_path)]), 0)
            self.assertEqual(cli_main(["train", "--config", str(config_path)]), 0)
            self.assertEqual(cli_main(["resume", "--run", str(run_dir)]), 0)
            self.assertEqual(cli_main(["merge", "--run", str(run_dir)]), 0)


if __name__ == "__main__":
    unittest.main()
