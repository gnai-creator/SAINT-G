import unittest

from saint.blocks import (
    analyze_block_reuse,
    block_reuse_metrics,
    group_blocks_by_signature,
    partition_matrix,
    reconstruction_error,
    reconstruct_matrix,
)


class BlockMetricsTests(unittest.TestCase):
    def test_reconstruction_error_is_zero_for_lossless_partition(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        blocks = partition_matrix(matrix, block_size=(2, 2))
        reconstructed = reconstruct_matrix(blocks, original_shape=(3, 3))

        metrics = reconstruction_error(matrix, reconstructed)

        self.assertEqual(metrics.l1_error, 0.0)
        self.assertEqual(metrics.l2_error, 0.0)
        self.assertEqual(metrics.relative_l1_error, 0.0)
        self.assertEqual(metrics.max_abs_error, 0.0)

    def test_reconstruction_error_detects_difference(self):
        metrics = reconstruction_error([[1, 2]], [[2, 4]])

        self.assertEqual(metrics.l1_error, 3.0)
        self.assertAlmostEqual(metrics.l2_error, 5**0.5)
        self.assertEqual(metrics.relative_l1_error, 1.0)
        self.assertEqual(metrics.max_abs_error, 2.0)

    def test_block_reuse_metrics_counts_repeated_blocks(self):
        matrix = [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
        ]
        blocks = partition_matrix(matrix, block_size=(2, 2))
        groups = group_blocks_by_signature(blocks)

        metrics = block_reuse_metrics(blocks, groups)

        self.assertEqual(metrics.block_count, 2)
        self.assertEqual(metrics.prototype_count, 1)
        self.assertEqual(metrics.repeated_block_count, 1)
        self.assertEqual(metrics.reuse_ratio, 0.5)
        self.assertEqual(metrics.estimated_compression_ratio, 2.0)

    def test_analyze_block_reuse_returns_combined_report(self):
        matrix = [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
        ]

        analysis = analyze_block_reuse(matrix, block_size=(2, 2))

        self.assertEqual(analysis.reconstruction, matrix)
        self.assertEqual(analysis.reconstruction_metrics.l1_error, 0.0)
        self.assertEqual(analysis.reuse_metrics.prototype_count, 1)


if __name__ == "__main__":
    unittest.main()
