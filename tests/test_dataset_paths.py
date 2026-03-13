import tempfile
import unittest
from pathlib import Path

from utils.dataset_paths import (
    get_dataset_split_path,
    get_reference_file_path,
    resolve_data_asset_path,
)
from utils.result_paths import resolve_gt_image_path


class DatasetPathsTest(unittest.TestCase):
    def assertPathsEquivalent(self, actual: Path, expected: Path):
        self.assertEqual(actual.resolve(strict=False), expected.resolve(strict=False))

    def test_dataset_split_and_reference_paths_follow_dataset_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            alias_root = root / ".." / root.name

            self.assertPathsEquivalent(
                get_dataset_split_path("CustomBench", "plot", "dev", work_dir=alias_root),
                root / "data" / "CustomBench" / "plot" / "dev.json",
            )
            self.assertPathsEquivalent(
                get_reference_file_path("CustomBench", "diagram", work_dir=alias_root),
                root / "data" / "CustomBench" / "diagram" / "ref.json",
            )

    def test_resolve_data_asset_path_prefers_explicit_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            alias_root = root / ".." / root.name
            expected = root / "data" / "CustomBench" / "diagram" / "images" / "sample.png"
            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.write_bytes(b"fake")

            resolved = resolve_data_asset_path(
                "images/sample.png",
                "diagram",
                dataset_name="CustomBench",
                work_dir=alias_root,
            )

            self.assertPathsEquivalent(resolved, expected)

    def test_resolve_gt_image_path_supports_results_bundle_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            alias_root = root / ".." / root.name
            results_dir = root / "results_bundle"
            results_dir.mkdir(parents=True, exist_ok=True)
            bundled_image = results_dir / "images" / "copy.png"
            bundled_image.parent.mkdir(parents=True, exist_ok=True)
            bundled_image.write_bytes(b"copy")

            resolved = resolve_gt_image_path(
                "images/copy.png",
                "diagram",
                results_path=str(results_dir / "run.json"),
                work_dir=alias_root,
                dataset_name="MissingBench",
            )

            self.assertPathsEquivalent(resolved, bundled_image)


if __name__ == "__main__":
    unittest.main()
