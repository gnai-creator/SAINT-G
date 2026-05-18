import tempfile
from pathlib import Path
import unittest

from saint.checkpoints.robust import read_matrix_payload_entry, write_matrix_payload
from saint.checkpoints.scale import (
    benchmark_dtype_io,
    benchmark_dtype_quality,
    benchmark_large_shards,
    benchmark_partial_shard_read,
    synthetic_delta_payload,
)


class CheckpointScalePhase12Tests(unittest.TestCase):
    def test_large_single_matrix_splits_by_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = synthetic_delta_payload(matrix_count=1, rows=32, cols=16)
            entry = write_matrix_payload(
                Path(tmp) / "delta.saintbin",
                payload,
                dtype="float32",
                shard_bytes=128,
            )
            restored = read_matrix_payload_entry(tmp, entry)

            self.assertEqual(entry["format"], "saint_matrix_shards")
            self.assertGreater(entry["shard_count"], 1)
            self.assertGreater(
                sum(len(shard.get("matrix_parts", [])) for shard in entry["shards"]),
                1,
            )
            max_error = 0.0
            for row_index, row in enumerate(payload["matrix_000"]):
                for col_index, value in enumerate(row):
                    max_error = max(
                        max_error,
                        abs(value - restored["matrix_000"][row_index][col_index]),
                    )
            self.assertLess(max_error, 1e-7)

    def test_large_shard_benchmark_validates_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_large_shards(
                tmp,
                matrix_count=2,
                rows=32,
                cols=32,
                dtype="float16",
                shard_bytes=256,
            )

            self.assertEqual(result["format"], "saint_matrix_shards")
            self.assertGreater(result["shard_count"], 1)
            self.assertTrue(result["checksum_validated"])
            self.assertLess(result["max_abs_error"], 0.001)
            self.assertGreater(result["payload_bytes"], 0)
            self.assertGreaterEqual(result["read_peak_bytes"], 0)

    def test_large_shard_checksum_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = synthetic_delta_payload(matrix_count=1, rows=16, cols=16)
            entry = write_matrix_payload(
                Path(tmp) / "delta.saintbin",
                payload,
                dtype="float32",
                shard_bytes=128,
            )
            first_shard = Path(tmp) / entry["shards"][0]["path"]
            with first_shard.open("ab") as handle:
                handle.write(b"corrupt")

            with self.assertRaises(ValueError):
                read_matrix_payload_entry(tmp, entry)

    def test_partial_read_skips_unselected_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = synthetic_delta_payload(matrix_count=3, rows=16, cols=16)
            entry = write_matrix_payload(
                Path(tmp) / "delta.saintbin",
                payload,
                dtype="float32",
                shard_bytes=1024,
            )
            selected = {"matrix_001"}
            unused = [
                shard
                for shard in entry["shards"]
                if all(part["name"] not in selected for part in shard["matrix_parts"])
            ]
            Path(tmp, unused[0]["path"]).unlink()

            restored = read_matrix_payload_entry(tmp, entry, matrix_names=selected)

            self.assertEqual(set(restored), selected)
            self.assertEqual(len(restored["matrix_001"]), 16)

    def test_partial_read_benchmark_reports_selected_matrices(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_partial_shard_read(
                tmp,
                matrix_count=4,
                rows=16,
                cols=16,
                selected_count=1,
                dtype="float16",
                shard_bytes=256,
            )

            self.assertEqual(result["full_matrix_count"], 4)
            self.assertEqual(result["partial_matrix_count"], 1)
            self.assertEqual(result["partial_keys"], ["matrix_000"])
            self.assertLess(result["max_abs_error"], 0.001)

    def test_dtype_io_benchmark_reports_all_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_dtype_io(
                tmp,
                matrix_count=2,
                rows=16,
                cols=16,
                shard_bytes=256,
            )
            by_dtype = {item["dtype"]: item for item in result["results"]}

            self.assertEqual(set(by_dtype), {"float32", "float16", "bfloat16", "int8"})
            self.assertEqual(by_dtype["float32"]["size_ratio_vs_float32"], 1.0)
            self.assertLess(by_dtype["float16"]["size_ratio_vs_float32"], 1.0)
            self.assertLess(by_dtype["int8"]["size_ratio_vs_float32"], 1.0)
            self.assertLess(by_dtype["float16"]["max_abs_error"], 0.001)

    def test_dtype_quality_benchmark_reports_merged_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_dtype_quality(
                tmp,
                dtypes=("float32", "float16", "int8"),
            )
            by_dtype = {item["dtype"]: item for item in result["results"]}

            self.assertEqual(set(by_dtype), {"float32", "float16", "int8"})
            self.assertEqual(by_dtype["float32"]["loss_delta_vs_float32"], 0.0)
            self.assertLess(abs(by_dtype["float16"]["loss_delta_vs_float32"]), 1e-6)
            self.assertIn("merged_loss", by_dtype["int8"])


if __name__ == "__main__":
    unittest.main()
