import unittest

from saint.blocks import partition_matrix, reconstruct_matrix


class PartitionMatrixTests(unittest.TestCase):
    def test_partition_and_reconstruct_even_shape(self):
        matrix = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]

        blocks = partition_matrix(matrix, block_size=(2, 2))
        reconstructed = reconstruct_matrix(blocks, original_shape=(4, 4))

        self.assertEqual(len(blocks), 4)
        self.assertEqual(blocks[0].values, ((1, 2), (5, 6)))
        self.assertEqual(blocks[3].values, ((11, 12), (15, 16)))
        self.assertEqual(reconstructed, matrix)

    def test_partition_and_reconstruct_with_padding(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        blocks = partition_matrix(matrix, block_size=(2, 2), pad_value=0)
        reconstructed = reconstruct_matrix(blocks, original_shape=(3, 3))

        self.assertEqual(len(blocks), 4)
        self.assertEqual(blocks[-1].values, ((9, 0), (0, 0)))
        self.assertEqual(reconstructed, matrix)

    def test_rejects_ragged_matrix(self):
        matrix = [[1, 2], [3]]

        with self.assertRaises(ValueError):
            partition_matrix(matrix, block_size=2)

    def test_rejects_invalid_block_size(self):
        with self.assertRaises(ValueError):
            partition_matrix([[1]], block_size=0)

    def test_multiple_block_sizes_round_trip(self):
        matrix = [[row * 17 + col for col in range(17)] for row in range(17)]

        for block_size in (2, 3, 4, 5, 6, 8, 16):
            with self.subTest(block_size=block_size):
                blocks = partition_matrix(matrix, block_size=block_size)
                reconstructed = reconstruct_matrix(blocks, original_shape=(17, 17))
                self.assertEqual(reconstructed, matrix)

    def test_rectangular_matrices_round_trip(self):
        shapes = [(3, 5), (5, 3), (7, 11), (11, 7)]

        for rows, cols in shapes:
            matrix = [[row * cols + col for col in range(cols)] for row in range(rows)]
            with self.subTest(shape=(rows, cols)):
                blocks = partition_matrix(matrix, block_size=(4, 3))
                reconstructed = reconstruct_matrix(blocks, original_shape=(rows, cols))
                self.assertEqual(reconstructed, matrix)


if __name__ == "__main__":
    unittest.main()
