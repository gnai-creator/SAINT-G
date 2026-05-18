import unittest

from saint.blocks import reconstruction_error
from saint.reconstruction import repeated_block_matrix
from saint.routing import (
    route_matrix_regions,
    route_matrix_regions_by_budget,
    route_matrix_regions_by_sensitivity_budget,
    routed_budget_reconstruction,
    routed_codebook_reconstruction,
    routed_sensitivity_budget_reconstruction,
    search_routed_budget_reconstruction,
)


class BlockRouterTests(unittest.TestCase):
    def test_router_uses_codebook_when_region_error_is_acceptable(self):
        matrix = repeated_block_matrix(8, 8, block_size=2, prototypes=2, seed=4)

        plan, reconstructed = route_matrix_regions(
            matrix,
            region_size=4,
            candidate_block_sizes=(2,),
            error_threshold=0.0,
            quantization_step=0.05,
        )

        methods = {region.method for region in plan.regions}
        self.assertEqual(methods, {"codebook_2"})
        self.assertEqual(reconstruction_error(matrix, reconstructed).l1_error, 0.0)

    def test_router_falls_back_to_free_delta_when_error_is_too_high(self):
        matrix = [
            [0.01 * (row * 8 + col) for col in range(8)]
            for row in range(8)
        ]

        plan, reconstructed = route_matrix_regions(
            matrix,
            region_size=4,
            candidate_block_sizes=(2,),
            error_threshold=0.0,
            quantization_step=1.0,
        )

        methods = {region.method for region in plan.regions}
        self.assertEqual(methods, {"free_delta"})
        self.assertEqual(reconstruction_error(matrix, reconstructed).l1_error, 0.0)

    def test_routed_codebook_reports_method_counts(self):
        matrix = repeated_block_matrix(8, 8, block_size=2, prototypes=2, seed=4)

        result = routed_codebook_reconstruction(
            matrix,
            region_size=4,
            candidate_block_sizes=(2,),
            error_threshold=0.0,
            quantization_step=0.05,
        )

        self.assertEqual(result.name, "routed_quality_first")
        self.assertIn("method_counts", result.metadata)
        self.assertEqual(result.metadata["region_count"], 4)

    def test_budget_router_reports_target_compression(self):
        matrix = repeated_block_matrix(8, 8, block_size=2, prototypes=2, seed=4)

        plan, reconstructed = route_matrix_regions_by_budget(
            matrix,
            region_size=4,
            candidate_block_sizes=(4, 2),
            error_weight=1.0,
            parameter_weight=1.0,
            target_compression=1.0,
            quantization_step=0.05,
        )

        self.assertTrue(plan.metadata["target_met"])
        self.assertEqual(reconstruction_error(matrix, reconstructed).l1_error, 0.0)

    def test_routed_budget_reconstruction_exposes_method_counts(self):
        matrix = repeated_block_matrix(8, 8, block_size=2, prototypes=2, seed=4)

        result = routed_budget_reconstruction(
            matrix,
            region_size=4,
            candidate_block_sizes=(4, 2),
            error_weight=1.0,
            parameter_weight=1.0,
            target_compression=1.0,
            quantization_step=0.05,
        )

        self.assertEqual(result.name, "routed_budget_first")
        self.assertIn("method_counts", result.metadata)
        self.assertIn("target_met", result.metadata)

    def test_budget_router_can_disable_free_delta(self):
        matrix = [
            [0.01 * (row * 8 + col) for col in range(8)]
            for row in range(8)
        ]

        result = routed_budget_reconstruction(
            matrix,
            region_size=4,
            candidate_block_sizes=(2,),
            error_weight=1.0,
            parameter_weight=1.0,
            include_free_delta=False,
            quantization_step=1.0,
        )

        self.assertNotIn("free_delta", result.metadata["method_counts"])

    def test_search_routed_budget_reconstruction_records_search(self):
        matrix = repeated_block_matrix(8, 8, block_size=2, prototypes=2, seed=4)

        result = search_routed_budget_reconstruction(
            matrix,
            parameter_weights=(0.1, 1.0),
            region_size=4,
            candidate_block_sizes=(4, 2),
            target_compression=1.0,
            max_relative_l1_error=0.1,
            quantization_step=0.05,
        )

        self.assertEqual(result.name, "routed_budget_search")
        self.assertIn("selected_parameter_weight", result.metadata)
        self.assertEqual(len(result.metadata["search"]), 2)

    def test_sensitivity_budget_router_can_freeze_regions(self):
        matrix = [[0.0 for _ in range(8)] for _ in range(8)]

        plan, reconstructed = route_matrix_regions_by_sensitivity_budget(
            matrix,
            region_size=4,
            candidate_block_sizes=(4, 2),
            include_freeze=True,
            include_free_delta=False,
        )

        self.assertEqual({region.method for region in plan.regions}, {"freeze"})
        self.assertEqual(reconstruction_error(matrix, reconstructed).l1_error, 0.0)

    def test_routed_sensitivity_budget_reconstruction_runs(self):
        matrix = repeated_block_matrix(8, 8, block_size=2, prototypes=2, seed=4)

        result = routed_sensitivity_budget_reconstruction(
            matrix,
            region_size=4,
            candidate_block_sizes=(4, 2),
            method_budgets={"free_delta": 0.0, "codebook_2": 0.25},
        )

        self.assertEqual(result.name, "routed_sensitivity_budget")
        self.assertIn("method_counts", result.metadata)


if __name__ == "__main__":
    unittest.main()
