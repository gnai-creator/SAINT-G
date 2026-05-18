from pathlib import Path
import unittest


class FileLineLimitTests(unittest.TestCase):
    def test_python_files_stay_under_500_lines(self):
        root = Path(__file__).resolve().parents[1]
        checked_roots = (root / "saint", root / "scripts", root / "tests")
        offenders = []
        for checked_root in checked_roots:
            for path in checked_root.rglob("*.py"):
                line_count = len(path.read_text(encoding="utf-8").splitlines())
                if line_count > 500:
                    offenders.append(f"{path.relative_to(root)}: {line_count}")

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
