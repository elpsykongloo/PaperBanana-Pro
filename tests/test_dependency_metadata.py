import re
import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _normalize_dependency_name(spec: str) -> str:
    return re.split(r"[<>=!~;\s\[]", spec, maxsplit=1)[0].strip().lower().replace("_", "-")


class DependencyMetadataTest(unittest.TestCase):
    def test_requirements_file_delegates_to_project_metadata(self):
        requirements_lines = [
            line.strip()
            for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        self.assertEqual(requirements_lines, ["-e ."])

    def test_uv_lock_covers_declared_project_dependencies(self):
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        lock_data = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))

        declared_dependencies = {
            _normalize_dependency_name(spec)
            for spec in pyproject["project"]["dependencies"]
        }
        locked_packages = {
            package["name"].strip().lower().replace("_", "-")
            for package in lock_data.get("package", [])
        }

        missing_dependencies = declared_dependencies - locked_packages
        self.assertFalse(
            missing_dependencies,
            f"uv.lock is missing declared dependencies: {sorted(missing_dependencies)}",
        )

    def test_readme_documents_locked_sync_workflow(self):
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("uv sync --locked", readme_text)

    def test_readme_documents_repo_first_editable_contract(self):
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("uv tool install --editable . --force", readme_text)
        self.assertIn("paperbanana", readme_text)
        self.assertIn("uv tool install paperbanana-pro", readme_text)
        self.assertNotIn("standalone tool install", readme_text)
        self.assertNotIn("tool install / wheel 优先", readme_text)
        self.assertNotIn("当前主 CLI 命令是 `paperbanana-pro`", readme_text)


if __name__ == "__main__":
    unittest.main()
