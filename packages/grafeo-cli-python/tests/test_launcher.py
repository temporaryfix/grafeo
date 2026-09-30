"""Launcher ownership controls; real native execution is checked from a wheel."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "grafeo_cli", Path(__file__).resolve().parents[1] / "grafeo_cli" / "__init__.py"
)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="grafeo-python-launcher-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.package = self.root / "grafeo_cli"
        self.package.mkdir()
        patch = mock.patch.object(CLI, "__file__", str(self.package / "__init__.py"))
        patch.start()
        self.addCleanup(patch.stop)

    def binary(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / CLI._binary_name()
        path.write_bytes(b"test stand-in; never executed")
        return path

    def test_never_selects_path_binary(self):
        external = self.binary(self.root / "external")
        with mock.patch("shutil.which", return_value=str(external)) as lookup:
            self.assertIsNone(CLI._find_binary())
            lookup.assert_not_called()

    def test_never_selects_adjacent_binary(self):
        for name in ("bin", "Scripts"):
            with self.subTest(directory=name):
                external = self.binary(self.root / name)
                with mock.patch("shutil.which", return_value=None):
                    self.assertIsNone(CLI._find_binary())
                external.unlink()

    def test_missing_bundle_fails_before_launch(self):
        external = self.binary(self.root / "external")
        stderr = io.StringIO()
        with mock.patch("shutil.which", return_value=str(external)), mock.patch.object(
            CLI.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
        ) as run, contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            CLI.main()
        self.assertEqual(error.exception.code, 1)
        self.assertIn("bundled grafeo binary is missing", stderr.getvalue())
        run.assert_not_called()

    def test_bundle_forwards_literal_arguments_and_exit_status(self):
        binary = self.binary(self.package)
        args = ["grafeo", "a space", "$(echo injected)", ";echo injected"]
        with mock.patch.object(CLI.sys, "argv", args), mock.patch.object(
            CLI.subprocess, "run", return_value=subprocess.CompletedProcess([], 23)
        ) as run, self.assertRaises(SystemExit) as error:
            CLI.main()
        self.assertEqual(error.exception.code, 23)
        run.assert_called_once_with([str(binary), *args[1:]], check=False)

    def test_removed_bundle_reports_execution_error(self):
        self.binary(self.package)
        stderr = io.StringIO()
        with mock.patch.object(CLI.subprocess, "run", side_effect=FileNotFoundError), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            CLI.main()
        self.assertEqual(error.exception.code, 1)
        self.assertIn("failed to execute", stderr.getvalue())

    def test_executable_bundle_does_not_require_write_permission(self):
        self.binary(self.package)
        with mock.patch.object(
            CLI.os, "access", return_value=True
        ), mock.patch.object(Path, "chmod", side_effect=PermissionError("read-only installation")) as chmod, mock.patch.object(
            CLI.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
        ), self.assertRaises(SystemExit) as error:
            CLI.main()
        self.assertEqual(error.exception.code, 0)
        chmod.assert_not_called()

    def test_keyboard_interrupt_retains_exit_130(self):
        self.binary(self.package)
        with mock.patch.object(CLI.subprocess, "run", side_effect=KeyboardInterrupt), self.assertRaises(SystemExit) as error:
            CLI.main()
        self.assertEqual(error.exception.code, 130)


if __name__ == "__main__":
    unittest.main()
