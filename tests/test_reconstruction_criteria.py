import unittest

from saint.reconstruction import (
    BenchmarkResult,
    evaluate_method_against_thresholds,
)


class ReconstructionCriteriaTests(unittest.TestCase):
    def test_evaluate_method_passes_when_thresholds_are_met(self):
        results = [
            BenchmarkResult("a", "method", 0.0, 0.0, 0.05, 0.0, 10, 1.5, 0.0, {}),
            BenchmarkResult("b", "method", 0.0, 0.0, 0.10, 0.0, 10, 1.3, 0.0, {}),
        ]

        decision = evaluate_method_against_thresholds(
            results,
            method_name="method",
            max_avg_relative_l1_error=0.1,
            min_avg_compression_ratio=1.2,
        )

        self.assertTrue(decision.passed)

    def test_evaluate_method_fails_when_thresholds_are_not_met(self):
        results = [
            BenchmarkResult("a", "method", 0.0, 0.0, 0.2, 0.0, 10, 1.0, 0.0, {}),
        ]

        decision = evaluate_method_against_thresholds(
            results,
            method_name="method",
            max_avg_relative_l1_error=0.1,
            min_avg_compression_ratio=1.2,
        )

        self.assertFalse(decision.passed)
        self.assertIn("failed thresholds", decision.reason)


if __name__ == "__main__":
    unittest.main()
