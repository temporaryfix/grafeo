"""Real wheel metadata controls with inert payloads, not native ABI evidence."""

import email.parser
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
PLATFORMS = (
    "manylinux_2_17_x86_64.manylinux2014_x86_64",
    "manylinux_2_17_aarch64.manylinux2014_aarch64",
    "macosx_11_0_x86_64",
    "macosx_11_0_arm64",
    "win_amd64",
)


class WheelTests(unittest.TestCase):
    def stage(self):
        temp = tempfile.TemporaryDirectory(prefix="grafeo-cli-wheel-")
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name in ("pyproject.toml", "README.md", "hatch_build.py"):
            shutil.copy2(PACKAGE / name, root / name)
        shutil.copy2(ROOT / "LICENSE", root / "LICENSE")
        shutil.copytree(PACKAGE / "grafeo_cli", root / "grafeo_cli")
        return root

    def build(self, root, platform, code=0):
        env = dict(os.environ, GRAFEO_CLI_WHEEL_PLATFORM=platform, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--no-isolation"],
            cwd=root, env=env, text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return result

    def test_platform_wheels_have_native_metadata_and_license(self):
        for platform in PLATFORMS:
            with self.subTest(platform=platform):
                root = self.stage()
                binary = "grafeo.exe" if platform == "win_amd64" else "grafeo"
                (root / "grafeo_cli" / binary).write_bytes(b"inert test payload")
                self.build(root, platform)
                wheel = root / "dist" / ("grafeo_cli-0.0.1-py3-none-" + platform + ".whl")
                with zipfile.ZipFile(wheel) as archive:
                    prefix = "grafeo_cli-0.0.1.dist-info/"
                    parser = email.parser.BytesParser()
                    metadata = parser.parsebytes(archive.read(prefix + "METADATA"))
                    tags = parser.parsebytes(archive.read(prefix + "WHEEL"))
                    self.assertEqual(tags.get_all("Root-Is-Purelib"), ["false"])
                    self.assertEqual(set(tags.get_all("Tag")), {"py3-none-" + p for p in platform.split(".")})
                    self.assertEqual(metadata.get_all("License-Expression"), ["Apache-2.0"])
                    self.assertEqual(metadata.get_all("License-File"), ["LICENSE"])
                    self.assertEqual(archive.read(prefix + "licenses/LICENSE"), (ROOT / "LICENSE").read_bytes())
                    self.assertEqual(archive.read("grafeo_cli/" + binary), b"inert test payload")

    def test_invalid_platforms_fail_before_emitting_a_wheel(self):
        for platform in ("", "any", "linux_x86_64"):
            with self.subTest(platform=platform):
                root = self.stage()
                (root / "grafeo_cli/grafeo").write_bytes(b"inert test payload")
                result = self.build(root, platform, code=1)
                self.assertIn("must name a supported native platform", result.stderr)
                self.assertFalse(list((root / "dist").glob("*.whl")))

    def test_missing_empty_linked_or_conflicting_bundles_fail(self):
        for case in ("missing", "empty", "linked", "conflicting"):
            with self.subTest(case=case):
                root = self.stage()
                binary = root / "grafeo_cli/grafeo"
                if case == "empty":
                    binary.touch()
                elif case == "linked":
                    binary.symlink_to(root / "README.md")
                elif case == "conflicting":
                    binary.write_bytes(b"inert test payload")
                    (binary.parent / "grafeo.exe").write_bytes(b"wrong target")
                self.build(root, PLATFORMS[0], code=1)
                self.assertFalse(list((root / "dist").glob("*.whl")))

    def test_source_distribution_rejects_launcher_only_sources(self):
        root = self.stage()
        result = subprocess.run(
            [sys.executable, "-m", "build", "--sdist", "--no-isolation"],
            cwd=root, text=True, capture_output=True, timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires its committed Rust workspace", result.stderr)
        self.assertFalse(list((root / "dist").glob("*.tar.gz")))

    def test_missing_compiler_cleans_temporary_native_output(self):
        root = self.stage()
        rust = root / "rust"
        rust.mkdir()
        (rust / "Cargo.toml").write_text('[workspace]\nmembers=[]\n')
        temporary = root / "temporary"
        temporary.mkdir()
        with mock.patch.dict(os.environ, {"PATH": "", "TMPDIR": str(temporary)}):
            result = self.build(root, PLATFORMS[0], code=1)
        self.assertIn("cargo", result.stderr)
        self.assertFalse(list(temporary.glob("grafeo-cli-native-*")))
        self.assertFalse(list((root / "dist").glob("*.whl")))


if __name__ == "__main__":
    unittest.main()
