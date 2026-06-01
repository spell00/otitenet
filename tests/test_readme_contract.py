import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


class ReadmeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme_text = README.read_text(encoding="utf-8")

    def test_local_markdown_links_exist(self):
        links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", self.readme_text)
        local_links = [link for link in links if not re.match(r"^[a-z]+://", link)]
        self.assertGreater(local_links, [])

        for link in local_links:
            path = link.split("#", 1)[0]
            if not path:
                continue
            with self.subTest(link=link):
                self.assertTrue((ROOT / path).exists(), f"Missing README link target: {link}")

    def test_documented_files_and_directories_exist(self):
        expected_paths = [
            "requirements.txt",
            "requirements-desktop.txt",
            "requirements-export.txt",
            "setup.py",
            "configs/preprocessing_config.json",
            "scripts/utils/init_db.py",
            "scripts/preprocessing/build_dataset.py",
            "scripts/create_mobile_deployment.py",
            "scripts/export_offline_onnx_model.py",
            "scripts/analysis/generate_paper_analysis.py",
            "launch.sh",
            "launch_optimize.sh",
            "app.py",
            "app_offline.py",
            "desktop/package.json",
            "desktop/src-tauri/tauri.conf.json",
        ]

        for rel_path in expected_paths:
            with self.subTest(path=rel_path):
                self.assertTrue((ROOT / rel_path).exists(), f"README references missing path: {rel_path}")

    def test_root_npm_scripts_used_by_readme_exist(self):
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
        scripts = package.get("scripts", {})
        for script in [
            "desktop:prepare:compact",
            "desktop:sidecar:compact",
            "desktop:tauri:compact",
            "desktop:prepare:exact",
            "desktop:sidecar:exact",
            "desktop:tauri:exact",
        ]:
            with self.subTest(script=script):
                self.assertIn(script, scripts)
                self.assertIn(f"npm run {script}", self.readme_text)

    def test_desktop_output_version_matches_desktop_package(self):
        package = json.loads((ROOT / "desktop/package.json").read_text(encoding="utf-8"))
        version = package["version"]
        documented_versions = set(re.findall(r"Otitenet_(\d+\.\d+\.\d+)_amd64", self.readme_text))
        self.assertEqual(documented_versions, {version})

    def test_quick_start_order_keeps_deployment_before_desktop_build(self):
        deployment_idx = self.readme_text.index("### 6. Export the Current Deployment")
        desktop_idx = self.readme_text.index("### 7. Build the Offline Desktop App")
        self.assertLess(deployment_idx, desktop_idx)

    def test_readme_uses_task_not_label_scheme(self):
        self.assertIn("--task=otite_four_class", self.readme_text)
        self.assertIn("--task=notNormal", self.readme_text)
        self.assertNotIn("--label_scheme", self.readme_text)
        self.assertNotIn("--label-scheme", self.readme_text)

    def test_github_actions_runs_tests_inside_dockerfile_image(self):
        workflow = (ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("docker build -t otitenet-test .", workflow)
        self.assertIn("docker run --rm otitenet-test bash scripts/test/unit.sh", workflow)
        self.assertIn("docker run --rm otitenet-test bash scripts/test/smoketest.sh", workflow)
        self.assertIn("COPY . .", dockerfile)


if __name__ == "__main__":
    unittest.main()
