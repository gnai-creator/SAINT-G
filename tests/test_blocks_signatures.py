import unittest

from saint.blocks import (
    compute_block_signature,
    group_blocks_by_signature,
    partition_matrix,
)


class BlockSignatureTests(unittest.TestCase):
    def test_exact_grouping_finds_equal_blocks(self):
        matrix = [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [5, 6, 1, 2],
            [7, 8, 3, 4],
        ]

        blocks = partition_matrix(matrix, block_size=(2, 2))
        groups = group_blocks_by_signature(blocks, mode="exact")

        repeated_groups = [group for group in groups.values() if len(group) == 3]
        self.assertEqual(len(repeated_groups), 1)
        self.assertEqual(
            sorted((block.row, block.col) for block in repeated_groups[0]),
            [(0, 0), (0, 1), (1, 1)],
        )

    def test_quantized_grouping_finds_similar_blocks(self):
        matrix = [
            [1.01, 1.99, 1.04, 2.02],
            [3.02, 3.98, 2.97, 4.01],
        ]

        blocks = partition_matrix(matrix, block_size=(2, 2))
        groups = group_blocks_by_signature(
            blocks,
            mode="quantized",
            quantization_step=0.1,
        )

        self.assertEqual(len(groups), 1)

    def test_stats_signature_includes_2x2_determinant(self):
        block = partition_matrix([[1, 2], [3, 4]], block_size=2)[0]
        signature = compute_block_signature(block, mode="stats")

        self.assertEqual(signature[0], (2, 2))
        self.assertEqual(signature[-1], -2.0)


if __name__ == "__main__":
    unittest.main()
