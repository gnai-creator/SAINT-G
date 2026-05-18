import unittest

from saint.blocks import build_fixed_codebook, partition_matrix


class FixedCodebookTests(unittest.TestCase):
    def test_fixed_codebook_assigns_repeated_blocks_to_same_prototype(self):
        matrix = [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [5, 6, 1, 2],
            [7, 8, 3, 4],
        ]
        blocks = partition_matrix(matrix, block_size=(2, 2))

        codebook = build_fixed_codebook(blocks)

        self.assertEqual(len(codebook.prototypes), 2)
        self.assertEqual(codebook.assignments[(0, 0)], codebook.assignments[(0, 1)])
        self.assertEqual(codebook.assignments[(0, 0)], codebook.assignments[(1, 1)])
        self.assertNotEqual(codebook.assignments[(0, 0)], codebook.assignments[(1, 0)])

    def test_fixed_codebook_supports_quantized_grouping(self):
        matrix = [
            [1.01, 1.99, 1.04, 2.02],
            [3.02, 3.98, 2.97, 4.01],
        ]
        blocks = partition_matrix(matrix, block_size=(2, 2))

        codebook = build_fixed_codebook(
            blocks,
            mode="quantized",
            quantization_step=0.1,
        )

        self.assertEqual(len(codebook.prototypes), 1)
        self.assertEqual(codebook.assignments[(0, 0)], codebook.assignments[(0, 1)])


if __name__ == "__main__":
    unittest.main()
