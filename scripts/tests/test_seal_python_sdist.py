"""A source package may prune the workspace lock, never change its selections."""
import importlib.util
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("seal", Path(__file__).resolve().parents[1] / "seal-python-sdist.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
OLD = b'''version = 4
[[package]]
name = "a"
version = "1.0.0"
source = "registry+https://example.invalid"
checksum = "abc"
dependencies = ["b"]
[[package]]
name = "b"
version = "2.0.0"
'''
NEW = OLD.split(b'[[package]]\nname = "b"')[0].replace(b'dependencies = ["b"]', b'dependencies = []')


class SourceLockTests(unittest.TestCase):
    def test_pruning_retains_exact_package_identity_and_checksum(self):
        M.validate_pruning(OLD, NEW)
        for before, after in ((b'1.0.0', b'1.0.1'), (b'checksum = "abc"', b'checksum = "bad"'),
                              (b'example.invalid', b'changed.invalid'), (b'dependencies = []', b'dependencies = ["c"]'),
                              (b'version = 4', b'version = 3')):
            with self.subTest(change=after), self.assertRaises(ValueError):
                M.validate_pruning(OLD, NEW.replace(before, after))

    def fixture(self, path):
        with tarfile.open(path, "w:gz") as archive:
            for name, data in (("Cargo.lock", OLD), ("src/lib.rs", b"source")):
                info = tarfile.TarInfo("grafeo-0.0.1/" + name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))

    def test_success_checks_locked_graph_and_only_replaces_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grafeo-0.0.1.tar.gz"
            self.fixture(path)
            calls = []
            def cargo(command, **kwargs):
                self.assertIn("--offline", command)
                self.assertEqual(kwargs["env"]["CARGO_NET_OFFLINE"], "true")
                calls.append(command)
                if "--locked" not in command:
                    (kwargs["cwd"] / "Cargo.lock").write_bytes(NEW)
            with patch.object(M.subprocess, "check_output", return_value="cargo 1.97.1 (fixture)"), patch.object(M.subprocess, "run", side_effect=cargo):
                M.seal_sdist(path)
            self.assertEqual(len(calls), 2)
            self.assertIn("--locked", calls[-1])
            with tarfile.open(path) as archive:
                self.assertEqual(archive.extractfile("grafeo-0.0.1/Cargo.lock").read(), NEW)
                self.assertEqual(archive.extractfile("grafeo-0.0.1/src/lib.rs").read(), b"source")
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_failure_preserves_archive_and_removes_temporary_files(self):
        for failure in ("cargo", "upgrade", "source"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "grafeo-0.0.1.tar.gz"
                self.fixture(path)
                original = path.read_bytes()
                def cargo(command, **kwargs):
                    if failure == "cargo":
                        raise subprocess.CalledProcessError(1, command)
                    (kwargs["cwd"] / "Cargo.lock").write_bytes(NEW.replace(b"1.0.0", b"1.0.1") if failure == "upgrade" else NEW)
                    if failure == "source":
                        (kwargs["cwd"] / "src/lib.rs").write_bytes(b"changed")
                with patch.object(M.subprocess, "check_output", return_value="cargo 1.97.1 (fixture)"), patch.object(M.subprocess, "run", side_effect=cargo), self.assertRaises((ValueError, subprocess.CalledProcessError)):
                    M.seal_sdist(path)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(list(Path(temporary).iterdir()), [path])
