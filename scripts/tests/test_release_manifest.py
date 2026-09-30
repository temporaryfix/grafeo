"""Real temporary artifact fixtures; never Cargo, extraction or publication."""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("release_manifest", ROOT / "scripts/release_manifest.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
SHA = "a" * 40
VERSION = "0.0.1"
# Source-derived obligations, not discovered files or proof of built products.
# Keep these literals independent of M.retained_catalog and its platform constants.
EXPECTED_RETAINED_ROLES = (
    ("grafeo", "native-archive", "x86_64-unknown-linux-gnu"),
    ("grafeo", "native-archive", "aarch64-unknown-linux-gnu"),
    ("grafeo", "native-archive", "x86_64-pc-windows-msvc"),
    ("grafeo", "native-archive", "x86_64-apple-darwin"),
    ("grafeo", "native-archive", "aarch64-apple-darwin"),
    ("grafeo-cli", "cli-native", "x86_64-unknown-linux-gnu"),
    ("grafeo-cli", "cli-native", "aarch64-unknown-linux-gnu"),
    ("grafeo-cli", "cli-native", "x86_64-pc-windows-msvc"),
    ("grafeo-cli", "cli-native", "x86_64-apple-darwin"),
    ("grafeo-cli", "cli-native", "aarch64-apple-darwin"),
    ("grafeo-c", "c-native", "x86_64-unknown-linux-gnu"),
    ("grafeo-c", "c-native", "aarch64-unknown-linux-gnu"),
    ("grafeo-c", "c-native", "x86_64-pc-windows-msvc"),
    ("grafeo-c", "c-native", "x86_64-apple-darwin"),
    ("grafeo-c", "c-native", "aarch64-apple-darwin"),
    ("@grafeo-db/js", "node-native", "x86_64-apple-darwin"),
    ("@grafeo-db/js", "node-native", "aarch64-apple-darwin"),
    ("@grafeo-db/js", "node-native", "x86_64-pc-windows-msvc"),
    ("@grafeo-db/js", "node-native", "x86_64-unknown-linux-gnu"),
    ("@grafeo-db/js", "node-native", "aarch64-unknown-linux-gnu"),
    ("@grafeo-db/js", "node-native", "aarch64-unknown-linux-musl"),
    ("@grafeo-db/js-darwin-x64", "node-package", "darwin-x64"),
    ("@grafeo-db/js-darwin-arm64", "node-package", "darwin-arm64"),
    ("@grafeo-db/js-win32-x64-msvc", "node-package", "win32-x64-msvc"),
    ("@grafeo-db/js-linux-x64-gnu", "node-package", "linux-x64-gnu"),
    ("@grafeo-db/js-linux-arm64-gnu", "node-package", "linux-arm64-gnu"),
    ("@grafeo-db/js-linux-arm64-musl", "node-package", "linux-arm64-musl"),
    ("@grafeo-db/js", "node-package", "any"),
    ("@grafeo-db/cli-linux-x64", "cli-npm", "linux-x64"),
    ("@grafeo-db/cli-linux-arm64", "cli-npm", "linux-arm64"),
    ("@grafeo-db/cli-win32-x64", "cli-npm", "win32-x64"),
    ("@grafeo-db/cli-darwin-x64", "cli-npm", "darwin-x64"),
    ("@grafeo-db/cli-darwin-arm64", "cli-npm", "darwin-arm64"),
    ("@grafeo-db/cli", "cli-npm", "any"),
    ("@grafeo-db/wasm", "wasm-package", "wasm32-unknown-unknown"),
    ("@grafeo-db/wasm-lite", "wasm-package", "wasm32-unknown-unknown"),
    ("grafeo", "python-wheel", "linux-x86_64"),
    ("grafeo", "python-wheel", "linux-aarch64"),
    ("grafeo", "python-wheel", "musllinux-x86_64"),
    ("grafeo", "python-wheel", "musllinux-aarch64"),
    ("grafeo", "python-wheel", "windows-x64"),
    ("grafeo", "python-wheel", "windows-x86"),
    ("grafeo", "python-wheel", "macos-x86_64"),
    ("grafeo", "python-wheel", "macos-aarch64"),
    ("grafeo", "python-sdist", "source"),
    ("grafeo-cli", "cli-wheel", "manylinux_2_17_x86_64.manylinux2014_x86_64"),
    ("grafeo-cli", "cli-wheel", "manylinux_2_17_aarch64.manylinux2014_aarch64"),
    ("grafeo-cli", "cli-wheel", "win_amd64"),
    ("grafeo-cli", "cli-wheel", "macosx_11_0_x86_64"),
    ("grafeo-cli", "cli-wheel", "macosx_11_0_arm64"),
    ("grafeo-cli", "cli-sdist", "source"),
    ("Grafeo", "nuget", "net8.0"),
    ("grafeo", "archive-checksums", "any"),
    ("grafeo", "rust-crate", "source"),
    ("grafeo-common", "rust-crate", "source"),
    ("grafeo-core", "rust-crate", "source"),
    ("grafeo-adapters", "rust-crate", "source"),
    ("grafeo-storage", "rust-crate", "source"),
    ("grafeo-engine", "rust-crate", "source"),
    ("grafeo-cli", "rust-crate", "source"),
    ("grafeo-python", "rust-crate", "source"),
    ("grafeo-node", "rust-crate", "source"),
    ("grafeo-c", "rust-crate", "source"),
    ("grafeo-wasm", "rust-crate", "source"),
    ("grafeo-bindings-common", "rust-crate", "source"),
    ("grafeo", "dart-source", "source"),
)




class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="grafeo-manifest-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "artifacts"
        self.root.mkdir()
        self.inventory = {"schema_version": 1, "candidate_sha": SHA, "version": VERSION,
                          "toolchains": {"fixture-producer": "test-only-not-build-proof"}, "artifacts": []}

    def artifact(self, path, fmt, package, target, platform, members):
        row = {"path": path, "format": fmt, "package": package, "target": target,
               "platform": platform, "required_members": sorted(members)}
        self.inventory["artifacts"].append(row)
        self.inventory["artifacts"].sort(key=lambda r: (r["package"], r["target"], r["platform"], r["path"]))
        return self.root / path

    def tar(self, path, entries):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w:gz") as archive:
            for name, value in entries.items():
                info = tarfile.TarInfo(name)
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))

    def zip(self, path, entries):
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in entries.items():
                archive.writestr(name, value)

    def crate(self):
        prefix = "grafeo-common-0.0.1"
        entries = {prefix + "/Cargo.toml": b'[package]\nname="grafeo-common"\nversion="0.0.1"\n',
                   prefix + "/src/lib.rs": b"pub fn fixture() {}\n"}
        path = self.artifact("rust/grafeo-common-0.0.1.crate", "crate", "grafeo-common", "rust-crate", "source", entries)
        self.tar(path, entries)
        return path

    def native_zip(self):
        platform = "x86_64-pc-windows-msvc"
        prefix = f"grafeo-v{VERSION}-{SHA}-{platform}"
        entries = {prefix + "/" + name: b"fixture bytes, not an executable or ABI proof\n"
                   for name in ("grafeo.exe", "grafeo_c.dll", "LICENSE", "README.md", "grafeo.h")}
        path = self.artifact(prefix + ".zip", "zip", "grafeo", "native-archive", platform, entries)
        self.zip(path, entries)
        return path

    def test_native_archives_require_nonempty_header_without_inventory_hint(self):
        for native in M.NATIVE:
            with self.subTest(platform=native[0]):
                path = self.fixture(dict(package="grafeo", target="native-archive", platform=native[0]))
                row = next(r for r in self.inventory["artifacts"] if r["path"] == str(path.relative_to(self.root)))
                row["required_members"] = [f"grafeo-v{VERSION}-{SHA}-{native[0]}/LICENSE"]
                entries = self.entries(path)
                header = f"grafeo-v{VERSION}-{SHA}-{native[0]}/grafeo.h"
                entries[header] = b"/* fixture header, not ABI proof */\n"
                write = self.zip if path.suffix == ".zip" else self.tar
                write(path, entries)
                M.build_manifest(self.root, self.inventory)
                for payload in (None, b""):
                    with self.subTest(header=payload):
                        broken = dict(entries)
                        if payload is None:
                            del broken[header]
                        else:
                            broken[header] = payload
                        write(path, broken)
                        self.rejects("required nonempty regular payload missing")
                write(path, entries)

    def entries(self, path):
        if path.suffix in (".zip", ".whl", ".nupkg"):
            with zipfile.ZipFile(path) as archive:
                return {name: archive.read(name) for name in archive.namelist()}
        with tarfile.open(path) as archive:
            return {entry.name: archive.extractfile(entry).read() for entry in archive if entry.isfile()}

    def test_cli_npm_license_and_notice_are_intrinsic_obligations(self):
        for platform in [r[4] for r in M.NATIVE] + ["any"]:
            with self.subTest(platform=platform):
                package = "@grafeo-db/cli" + ("-" + platform if platform != "any" else "")
                path = self.fixture(dict(package=package, target="cli-npm", platform=platform))
                row = next(r for r in self.inventory["artifacts"] if r["path"] == str(path.relative_to(self.root)))
                row["required_members"] = ["package/package.json"]
                entries = self.entries(path)
                M.build_manifest(self.root, self.inventory)
                for problem in ("missing-notice", "wrong-license"):
                    with self.subTest(problem=problem):
                        broken = dict(entries)
                        if problem == "missing-notice":
                            del broken["package/LICENSE"]
                        else:
                            meta = json.loads(broken["package/package.json"])
                            meta["license"] = "AGPL-3.0-or-later"
                            broken["package/package.json"] = json.dumps(meta).encode()
                        self.tar(path, broken)
                        self.rejects()
                self.tar(path, entries)

    def fixture(self, row):
        """Small format-real bytes, never a claimed build or platform executable."""
        package, target, platform = row["package"], row["target"], row["platform"]
        directory = f"{target}/{package.replace('/', '_').replace('@', '')}/{platform}/"
        payload = b"fixture payload\n"
        if target in ("cli-native", "c-native", "node-native", "provenance", "sbom"):
            path = self.artifact(directory + "bytes", "file", package, target, platform, [])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return path
        if target == "rust-crate":
            prefix = package + "-0.0.1"
            fmt, name = "crate", prefix + ".crate"
            entries = {prefix + "/Cargo.toml": f'[package]\nname="{package}"\nversion="0.0.1"\n'.encode(), prefix + "/src/lib.rs": payload}
        elif target == "native-archive":
            native = next(r for r in M.NATIVE if r[0] == platform)
            prefix = f"grafeo-v0.0.1-{SHA}-{platform}"
            fmt, name = native[3], prefix + "." + native[3]
            entries = {prefix + "/" + leaf: payload for leaf in ("grafeo.exe" if native[5] == "win32" else "grafeo", native[2], "LICENSE", "README.md", "grafeo.h")}
        elif target in ("node-package", "wasm-package", "cli-npm"):
            fmt, name = "npm-tgz", "package.tgz"
            meta = {"name": package, "version": VERSION}
            if target == "node-package" and platform != "any":
                _, suffix, system, cpu, libc = next(r for r in M.NODE if r[1] == platform)
                meta.update(os=[system], cpu=[cpu], main="grafeo." + suffix + ".node")
                if libc:
                    meta["libc"] = [libc]
                entries = {"package/" + meta["main"]: payload}
            elif target == "node-package":
                meta.update(main="index.js", types="index.d.ts", optionalDependencies={"@grafeo-db/js-" + r[1]: VERSION for r in M.NODE})
                entries = {"package/index.js": payload, "package/index.d.ts": payload}
            elif target == "wasm-package":
                meta.update(main="grafeo_wasm.js", types="grafeo_wasm.d.ts")
                entries = {"package/" + leaf: payload for leaf in ("grafeo_wasm.js", "grafeo_wasm.d.ts", "grafeo_wasm_bg.js", "grafeo_wasm_bg.wasm", "grafeo_wasm_bg.wasm.d.ts")}
            elif platform == "any":
                meta.update(bin={"grafeo": "bin/grafeo.js"}, optionalDependencies={"@grafeo-db/cli-" + r[4]: VERSION for r in M.NATIVE})
                entries = {"package/bin/grafeo.js": payload}
            else:
                native = next(r for r in M.NATIVE if r[4] == platform)
                meta.update(os=[native[5]], cpu=[native[6]])
                entries = {"package/" + ("grafeo.exe" if native[5] == "win32" else "grafeo"): payload}
            if target in ("cli-npm", "node-package"):
                meta["license"] = "Apache-2.0"
                entries["package/LICENSE"] = payload
            if target == "node-package":
                meta["engines"] = {"node": ">=20.3.0"}
            entries["package/package.json"] = json.dumps(meta).encode()
        elif target in ("python-wheel", "cli-wheel"):
            fmt = "wheel"
            if target == "python-wheel":
                tags = {"linux-x86_64": "manylinux_2_28_x86_64", "linux-aarch64": "manylinux_2_28_aarch64", "musllinux-x86_64": "musllinux_1_2_x86_64", "musllinux-aarch64": "musllinux_1_2_aarch64", "windows-x64": "win_amd64", "windows-x86": "win32", "macos-x86_64": "macosx_10_12_x86_64", "macos-aarch64": "macosx_11_0_arm64"}[platform]
                py, abi = "cp312", "abi3"
                entries = {"grafeo/__init__.py": payload, "grafeo/grafeo" + (".pyd" if platform.startswith("windows") else ".abi3.so"): payload}
            else:
                tags, py, abi = platform, "py3", "none"
                entries = {"grafeo_cli/__init__.py": payload, "grafeo_cli/" + ("grafeo.exe" if platform == "win_amd64" else "grafeo"): payload}
            normalized = package.replace("-", "_")
            name = f"{normalized}-0.0.1-{py}-{abi}-{tags}.whl"
            prefix = normalized + "-0.0.1.dist-info/"
            requires = ">=3.12" if target == "python-wheel" else ">=3.9"
            entries[prefix + "METADATA"] = f"Metadata-Version: 2.1\nName: {package}\nVersion: 0.0.1\nRequires-Python: {requires}\n\n".encode()
            entries[prefix + "WHEEL"] = ("Wheel-Version: 1.0\n" + "".join(f"Tag: {py}-{abi}-{tag}\n" for tag in tags.split(".")) + "\n").encode()
            if target in ("python-wheel", "cli-wheel"):
                entries[prefix + "METADATA"] = entries[prefix + "METADATA"].replace(
                    b"\n\n", b"\nLicense-Expression: Apache-2.0\nLicense-File: LICENSE\n\n")
                entries[prefix + "WHEEL"] = entries[prefix + "WHEEL"].replace(
                    b"\n\n", b"\nRoot-Is-Purelib: false\n\n")
                entries[prefix + "licenses/LICENSE"] = payload
        elif target in ("python-sdist", "cli-sdist"):
            fmt = "python-sdist"
            prefix = package.replace("-", "_") + "-0.0.1"
            name = prefix + ".tar.gz"
            requires = ">=3.12" if target == "python-sdist" else ">=3.9"
            entries = {prefix + "/PKG-INFO": f"Metadata-Version: 2.1\nName: {package}\nVersion: 0.0.1\nRequires-Python: {requires}\n\n".encode(), prefix + "/pyproject.toml": f'[project]\nname="{package}"\nversion="0.0.1"\nrequires-python="{requires}"\n'.encode(), prefix + ("/python/grafeo/__init__.py" if target == "python-sdist" else "/grafeo_cli/__init__.py"): payload}
            if target == "python-sdist":
                entries[prefix + "/pyproject.toml"] += b'[tool.maturin]\nmodule-name="grafeo"\npython-source="python"\n'
                entries[prefix + "/Cargo.toml"] = b'[package]\nname="grafeo-python"\nversion="0.0.1"\n'
                entries[prefix + "/src/lib.rs"] = payload
            else:
                entries[prefix + "/pyproject.toml"] += b'[build-system]\nrequires=["hatchling==1.27.0"]\nbuild-backend="hatchling.build"\n[tool.hatch.build.hooks.custom]\npath="hatch_build.py"\n'
                entries[prefix + "/hatch_build.py"] = payload
                entries[prefix + "/rust/Cargo.toml"] = b'[workspace]\nmembers=["crates/grafeo-cli"]\n[workspace.package]\nversion="0.0.1"\n'
                entries[prefix + "/rust/Cargo.lock"] = b'version = 4\n'
                entries[prefix + "/rust/crates/grafeo-cli/Cargo.toml"] = b'[package]\nname="grafeo-cli"\nversion.workspace=true\n'
                entries[prefix + "/rust/crates/grafeo-cli/src/main.rs"] = payload
        elif target == "nuget":
            fmt, name = "nuget", "Grafeo.0.0.1.nupkg"
            entries = {"Grafeo.nuspec": b"<package><metadata><id>Grafeo</id><version>0.0.1</version></metadata></package>", "lib/net8.0/Grafeo.dll": payload}
            entries.update({"runtimes/" + r[1] + "/native/" + r[2]: payload for r in M.NATIVE})
        elif target == "dart-source":
            fmt, name = "dart-source", "local-dart-audit.tar.gz"
            entries = {"pubspec.yaml": b"name: grafeo\nversion: 0.0.1\n", "lib/grafeo.dart": payload}
        elif target == "go-source":
            fmt, name = "go-source", "local-go-audit.tar.gz"
            entries = {"go.mod": ("module " + M.GO_MODULE + "\n\ngo 1.21\n").encode(), "grafeo.go": payload, "grafeo.h": payload}
        elif target == "archive-checksums":
            path = self.artifact(directory + "SHA256SUMS", "file", package, target, platform, [])
            path.parent.mkdir(parents=True, exist_ok=True)
            native = [r for r in self.inventory["artifacts"] if r["target"] == "native-archive"]
            path.write_text("".join(hashlib.sha256((self.root / r["path"]).read_bytes()).hexdigest() + "  " + Path(r["path"]).name + "\n" for r in sorted(native, key=lambda r: r["path"])))
            return path
        else:
            self.fail("missing fixture " + target)
        path = self.artifact(directory + name, fmt, package, target, platform, entries)
        (self.zip if fmt in ("zip", "wheel", "nuget") else self.tar)(path, entries)
        return path

    def rewrite(self, path, entries):
        (self.zip if path.suffix in (".zip", ".whl", ".nupkg") else self.tar)(path, entries)

    def rejects(self, reason=None):
        with self.assertRaises(M.ManifestError) as raised:
            M.build_manifest(self.root, self.inventory)
        if reason:
            self.assertIn(reason, str(raised.exception))
        self.assertTrue(raised.exception.location)
        self.assertTrue(raised.exception.reason)
        return raised.exception

    def test_round_trip_real_crate_and_zip(self):
        self.crate()
        self.native_zip()
        manifest = M.build_manifest(self.root, self.inventory)
        self.assertEqual(manifest["candidate_sha"], SHA)
        self.assertEqual(len(manifest["artifacts"]), 2)
        self.assertIsNone(M.verify_manifest(self.root, manifest, self.inventory))

    def test_changed_actual_bytes_reject(self):
        path = self.crate()
        manifest = M.build_manifest(self.root, self.inventory)
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaises(M.ManifestError):
            M.verify_manifest(self.root, manifest, self.inventory)

    def test_every_retained_role_has_real_format_fixture_round_trip(self):
        catalog = M.retained_catalog()
        for row in catalog:
            if row["target"] != "archive-checksums":
                self.fixture(row)
        self.fixture(next(row for row in catalog if row["target"] == "archive-checksums"))
        M.compare_catalog(self.inventory)
        manifest = M.build_manifest(self.root, self.inventory)
        self.assertEqual(len(manifest["artifacts"]), 66)
        self.assertIsNone(M.verify_manifest(self.root, manifest, self.inventory))
        for row in manifest["artifacts"]:
            raw = (self.root / row["path"]).read_bytes()
            self.assertEqual(row["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(row["blake3"], M.hash_factory()(raw).hexdigest())

    def test_go_local_audit_convention_round_trip(self):
        self.fixture({"package": M.GO_MODULE, "target": "go-source", "platform": "source"})
        M.verify_manifest(self.root, M.build_manifest(self.root, self.inventory), self.inventory)

    def test_intrinsic_metadata_cannot_be_omitted_from_required_members(self):
        path = self.crate()
        entries = self.entries(path)
        entries.pop("grafeo-common-0.0.1/Cargo.toml")
        self.inventory["artifacts"][0]["required_members"] = ["grafeo-common-0.0.1/src/lib.rs"]
        self.tar(path, entries)
        self.rejects("Cargo.toml")

    def test_missing_and_undeclared_artifact(self):
        path = self.crate()
        saved = path.read_bytes()
        path.unlink()
        self.rejects("missing artifacts")
        path.write_bytes(saved)
        (self.root / "extra").write_bytes(b"undeclared")
        self.rejects("undeclared artifacts")

    def test_root_symlink_and_hardlink_rejected(self):
        path = self.crate()
        link = self.root / "linked"
        link.symlink_to(path)
        self.rejects("non-linked")
        link.unlink()
        os.link(path, link)
        self.rejects("non-linked")

    def test_root_case_and_prefix_collision(self):
        self.crate()
        # This volume may fold case; exercise Names independently instead of
        # assuming that distinct case-only physical siblings can be created.
        names = M.Names()
        names.add("rust/x", False, "test")
        with self.assertRaises(M.ManifestError):
            names.add("RUST/y", False, "test")
        names = M.Names()
        names.add("a", False, "test")
        with self.assertRaises(M.ManifestError):
            names.add("a/b", False, "test")

    def test_invalid_descriptor_paths(self):
        self.crate()
        for name in ("/absolute", "../up", "a/../b", "a//b", "a/./b", "C:/drive", "a\\b", "", "a\x00b", "a\nb", "//server/share"):
            with self.subTest(name=name):
                inventory = copy.deepcopy(self.inventory)
                inventory["artifacts"][0]["path"] = name
                with self.assertRaises(M.ManifestError):
                    M.build_manifest(self.root, inventory)

    def test_tar_unsafe_effective_names_and_special_types(self):
        path = self.crate()
        for name in ("/absolute", "../outside", "C:/drive", "back\\slash", "a//b", "a/./b"):
            with self.subTest(name=name):
                self.tar(path, {name: b"not extracted"})
                self.rejects("path")
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE, tarfile.GNUTYPE_SPARSE):
            with self.subTest(kind=kind):
                with tarfile.open(path, "w:gz") as archive:
                    info = tarfile.TarInfo("unsafe")
                    info.type, info.linkname = kind, "../outside"
                    archive.addfile(info)
                self.rejects("forbidden")
        self.assertFalse((self.root.parent / "outside").exists())

    def test_tar_pax_and_gnu_long_name_effective_path(self):
        path = self.crate()
        for fmt in (tarfile.PAX_FORMAT, tarfile.GNU_FORMAT):
            with self.subTest(fmt=fmt):
                with tarfile.open(path, "w:gz", format=fmt) as archive:
                    info = tarfile.TarInfo("../" + "x" * 120)
                    info.size = 1
                    archive.addfile(info, io.BytesIO(b"x"))
                self.rejects("path")

    def test_archive_duplicate_case_prefix_and_directory_entries(self):
        path = self.native_zip()
        original = self.entries(path)
        for pair in (("a", "a"), ("A/x", "a/y"), ("a", "a/b"), ("dir/", "dir/")):
            with self.subTest(pair=pair), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(pair[0], b"" if pair[0].endswith("/") else b"x")
                    archive.writestr(pair[1], b"" if pair[1].endswith("/") else b"x")
                self.rejects()
        self.zip(path, original | {"empty/": b""})
        self.rejects("outside intrinsic")

    def test_zip_links_special_modes_encryption_and_directory_forms(self):
        path = self.native_zip()
        for mode, name, payload in ((stat.S_IFLNK, "link", b"target"), (stat.S_IFIFO, "fifo", b""), (stat.S_IFDIR, "dir", b""), (stat.S_IFREG, "dir/", b""), (stat.S_IFDIR, "dir/", b"data")):
            with self.subTest(mode=mode, name=name):
                info = zipfile.ZipInfo(name)
                info.create_system, info.external_attr = 3, (mode | 0o644) << 16
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(info, payload)
                self.rejects()
        self.native_zip_bytes(path)
        raw = bytearray(path.read_bytes())
        central = raw.index(b"PK\x01\x02")
        struct.pack_into("<H", raw, central + 8, 1)
        path.write_bytes(raw)
        self.rejects("encrypted")

    def native_zip_bytes(self, path):
        prefix = f"grafeo-v0.0.1-{SHA}-x86_64-pc-windows-msvc"
        self.zip(path, {prefix + "/" + name: b"fixture" for name in ("grafeo.exe", "grafeo_c.dll", "LICENSE", "README.md", "grafeo.h")})

    def test_zip_65537_byte_inflater_drain_round_trip(self):
        path = self.native_zip()
        entries = self.entries(path)
        entries[next(iter(entries))] = b"a" * 65537
        self.zip(path, entries)
        manifest = M.build_manifest(self.root, self.inventory)
        self.assertIn(65537, [member.get("size") for member in manifest["artifacts"][0]["members"]])

    def test_gzip_large_chunk_boundary_round_trip(self):
        path = self.crate()
        entries = self.entries(path)
        entries["grafeo-common-0.0.1/src/lib.rs"] = b"a" * 65537
        self.tar(path, entries)
        M.build_manifest(self.root, self.inventory)

    def test_archive_crc_truncation_and_trailing_bytes(self):
        for make in (self.crate, self.native_zip):
            path = make()
            raw = path.read_bytes()
            for broken in (raw[:-1], raw + b"trailing"):
                with self.subTest(path=path.name, length=len(broken)):
                    path.write_bytes(broken)
                    self.rejects()
            path.write_bytes(raw)
        path = self.root / self.inventory["artifacts"][0]["path"]
        raw = bytearray(path.read_bytes())
        if path.suffix == ".zip":
            central = raw.index(b"PK\x01\x02")
            local = struct.unpack_from("<I", raw, central+42)[0]
            crc = struct.unpack_from("<I", raw, central+16)[0] ^ 1
            struct.pack_into("<I", raw, central+16, crc)
            struct.pack_into("<I", raw, local+14, crc)
            path.write_bytes(raw)
            self.rejects("CRC")

    def test_concatenated_gzip_header_rejected_before_parsing(self):
        path = self.crate()
        raw = path.read_bytes()
        # A second stream's FNAME exceeds a small budget; never let a second
        # header parser see it, even when its decompressed content is zeroes.
        second = bytearray(gzip.compress(b"\x00" * 512))
        second[3] |= 8
        second[10:10] = b"n" * 4097 + b"\x00"
        path.write_bytes(raw + second)
        with mock.patch.object(M, "MAX_METADATA_BYTES", 4096), mock.patch.object(M, "gzip_header", wraps=M.gzip_header) as header:
            self.rejects("concatenated gzip")
            self.assertEqual(header.call_count, 1)

    def test_json_nonfinite_duplicate_depth_and_budget_before_parser(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'{"x":-1e999}'):
            with self.subTest(raw=raw), self.assertRaises(M.ManifestError):
                M.parse_json(raw, "fixture")
        with mock.patch.object(M, "MAX_METADATA_BYTES", 8), mock.patch.object(M.json, "loads", side_effect=AssertionError("must not parse")):
            with self.assertRaisesRegex(M.ManifestError, "before JSON"):
                M.parse_json(b'{"oversized":1}', "fixture")
        with mock.patch.object(M, "MAX_DEPTH", 2), self.assertRaisesRegex(M.ManifestError, "before parsing"):
            M.parse_json(b"[[[]]]", "fixture")

    def test_zip_central_budget_before_member_materialization(self):
        path = self.native_zip()
        self.inventory["artifacts"][0]["required_members"] = [next(iter(self.entries(path)))]
        with mock.patch.object(M, "MAX_METADATA_BYTES", 600), mock.patch.object(M.Members, "begin", side_effect=AssertionError("must not materialize")):
            self.rejects("central budget")

    def test_tar_extended_metadata_budget_before_pax_parser(self):
        path = self.crate()
        with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            info = tarfile.TarInfo("x")
            info.pax_headers = {"path": "n" * 4096}
            archive.addfile(info)
        with mock.patch.object(M, "MAX_METADATA_BYTES", 2048), mock.patch.object(M, "pax_records", side_effect=AssertionError("must not parse")):
            self.rejects("extended-header budget")

    def test_member_count_and_actual_expansion_budgets(self):
        path = self.native_zip()
        with mock.patch.object(M, "MAX_MEMBERS", 1):
            self.rejects()
        entries = self.entries(path)
        entries[next(iter(entries))] = b"x" * 8192
        self.zip(path, entries)
        with mock.patch.object(M, "MAX_MEMBER_BYTES", 4096):
            self.rejects("per-member")
        # Forge only size declarations to be small, retaining actual compressed
        # stream/CRC. The streaming limit must trip before final size comparison.
        raw = bytearray(path.read_bytes())
        central = raw.index(b"PK\x01\x02")
        local = struct.unpack_from("<I", raw, central+42)[0]
        struct.pack_into("<I", raw, central+24, 100)
        struct.pack_into("<I", raw, local+22, 100)
        path.write_bytes(raw)
        with mock.patch.object(M, "MAX_MEMBER_BYTES", 4096):
            self.rejects("actual expanded member")
        with mock.patch.object(M, "MAX_EXPANDED_BYTES", 4096):
            self.rejects("expanded-byte")

    def test_both_hashes_and_independent_context_are_verified(self):
        self.crate()
        manifest = M.build_manifest(self.root, self.inventory)
        for key, value in (("candidate_sha", "b" * 40), ("toolchains", {"fixture-producer": "different"})):
            changed = copy.deepcopy(manifest)
            changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(M.ManifestError, "context"):
                M.verify_manifest(self.root, changed, self.inventory)
        for key in ("sha256", "blake3", "size"):
            for member in (False, True):
                changed = copy.deepcopy(manifest)
                row = changed["artifacts"][0]
                if member:
                    row = row["members"][0]
                row[key] = 1 if key == "size" else "0" * 64
                with self.subTest(key=key, member=member), self.assertRaises(M.ManifestError):
                    M.verify_manifest(self.root, changed, self.inventory)

    def test_manifest_context_does_not_alias_independent_inventory(self):
        self.crate()
        manifest = M.build_manifest(self.root, self.inventory)
        original = dict(manifest["toolchains"])
        self.inventory["toolchains"]["fixture-producer"] = "changed independent expectation"
        self.assertEqual(manifest["toolchains"], original)
        self.assertIsNot(manifest["toolchains"], self.inventory["toolchains"])
        with self.assertRaisesRegex(M.ManifestError, "context mismatch"):
            M.verify_manifest(self.root, manifest, self.inventory)

    def test_blake3_version_import_vectors_and_runtime_failure_no_fallback(self):
        self.crate()
        self.assertEqual(M.hash_factory()(b"").hexdigest(), "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262")
        with mock.patch.object(M.importlib.metadata, "version", return_value="1.0.8"):
            self.rejects("1.0.9")
        with mock.patch.object(M.importlib, "import_module", side_effect=ImportError("distinct missing hash library")):
            self.rejects("distinct missing")
        fake = mock.Mock()
        fake.blake3.return_value.hexdigest.return_value = "0" * 64
        with mock.patch.object(M.importlib, "import_module", return_value=fake):
            self.rejects("vector mismatch")

    def test_detectable_mutation_and_replacement_during_inspection(self):
        path = self.crate()
        original = M.inspect_package
        def mutate(members, inventory):
            original(members, inventory)
            with path.open("ab") as file:
                file.write(b"mutation")
        with mock.patch.object(M, "inspect_package", side_effect=mutate):
            self.rejects("changed during")
        self.tar(path, {"grafeo-common-0.0.1/Cargo.toml": b'[package]\nname="grafeo-common"\nversion="0.0.1"\n', "grafeo-common-0.0.1/src/lib.rs": b"x"})
        def replace(members, inventory):
            original(members, inventory)
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        with mock.patch.object(M, "inspect_package", side_effect=replace):
            self.rejects("replacement")

    def test_schema_unknown_boolean_digest_and_order_rejection(self):
        self.crate()
        self.native_zip()
        manifest = M.build_manifest(self.root, self.inventory)
        mutations = [lambda m: m.update(extra=1), lambda m: m.update(schema_version=True), lambda m: m["artifacts"].reverse(), lambda m: m["artifacts"][0].update(size=True), lambda m: m["artifacts"][0].update(sha256="A" * 64), lambda m: m["artifacts"][0]["members"].reverse(), lambda m: m["artifacts"].append(m["artifacts"][0])]
        for mutation in mutations:
            changed = copy.deepcopy(manifest)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(M.ManifestError):
                M.verify_manifest(self.root, changed, self.inventory)

    def test_every_package_intrinsic_version_or_identity_mutation_rejected(self):
        cases = [("grafeo-common", "rust-crate", "source"), ("@grafeo-db/js-linux-x64-gnu", "node-package", "linux-x64-gnu"), ("grafeo", "python-wheel", "linux-x86_64"), ("grafeo-cli", "cli-sdist", "source"), ("Grafeo", "nuget", "net8.0"), ("grafeo", "dart-source", "source"), (M.GO_MODULE, "go-source", "source")]
        for package, target, platform in cases:
            with self.subTest(target=target):
                path = self.fixture(dict(package=package, target=target, platform=platform))
                entries = self.entries(path)
                changed = {name: value.replace(b"0.0.1", b"9.9.9") if target != "go-source" else value.replace(M.GO_MODULE.encode(), b"example.invalid/wrong") for name, value in entries.items()}
                self.rewrite(path, changed)
                self.rejects("mismatch")
                self.rewrite(path, entries)

    def test_empty_required_payload_and_wrong_platform_rejected(self):
        path = self.fixture(dict(package="@grafeo-db/js-linux-x64-gnu", target="node-package", platform="linux-x64-gnu"))
        entries = self.entries(path)
        binary = "package/grafeo.linux-x64-gnu.node"
        self.tar(path, entries | {binary: b""})
        self.rejects("nonempty")
        meta = json.loads(entries["package/package.json"])
        meta["cpu"] = ["arm64"]
        self.tar(path, entries | {"package/package.json": json.dumps(meta).encode()})
        self.rejects("cpu")

    def test_checksums_exact_five_archive_bytes_no_independent_authority(self):
        for row in M.retained_catalog():
            if row["target"] == "native-archive":
                self.fixture(row)
        path = self.fixture(dict(package="grafeo", target="archive-checksums", platform="any"))
        M.build_manifest(self.root, self.inventory)
        raw = path.read_bytes()
        for value in (b"0" * 64 + raw[64:], raw.splitlines(keepends=True)[0], raw + raw.splitlines(keepends=True)[0]):
            path.write_bytes(value)
            self.rejects("checksum")

    def test_python_sdist_declared_workspace_layout_and_escaping_paths(self):
        path = self.fixture(dict(package="grafeo", target="python-sdist", platform="source"))
        entries = self.entries(path)
        prefix = "grafeo-0.0.1/"
        cargo_dir = "crates/bindings/python/"
        for suffix in ("Cargo.toml", "src/lib.rs", "python/grafeo/__init__.py"):
            entries[prefix + cargo_dir + suffix] = entries.pop(prefix + suffix)
        project = entries[prefix + "pyproject.toml"].replace(b'python-source="python"', b'python-source="crates/bindings/python/python"\nmanifest-path="crates/bindings/python/Cargo.toml"')
        entries[prefix + "pyproject.toml"] = project
        self.inventory["artifacts"][0]["required_members"] = [prefix + "PKG-INFO"]
        self.tar(path, entries)
        M.build_manifest(self.root, self.inventory)
        inherited = dict(entries)
        inherited[prefix + cargo_dir + "Cargo.toml"] = b'[package]\nname="grafeo-python"\nversion.workspace=true\n'
        inherited[prefix + "Cargo.toml"] = b'[workspace]\nmembers=["crates/bindings/python"]\n[workspace.package]\nversion="0.0.1"\n'
        self.tar(path, inherited)
        M.build_manifest(self.root, self.inventory)
        for workspace in (None, b'[workspace]\nmembers=["crates/bindings/other"]\n[workspace.package]\nversion="0.0.1"\n', b'[workspace]\nmembers=["../outside"]\n[workspace.package]\nversion="0.0.1"\n', b'[workspace]\nmembers=["crates/bindings/python"]\n[workspace.package]\nversion="9.9.9"\n'):
            broken = dict(inherited)
            if workspace is None:
                broken.pop(prefix + "Cargo.toml")
            else:
                broken[prefix + "Cargo.toml"] = workspace
            self.tar(path, broken)
            self.rejects()
        for before, after in ((b'crates/bindings/python/Cargo.toml', b'../Cargo.toml'), (b'crates/bindings/python/python', b'/outside/python')):
            with self.subTest(after=after):
                self.tar(path, entries | {prefix + "pyproject.toml": project.replace(before, after)})
                self.rejects("path")
        broken = dict(entries)
        broken.pop(prefix + cargo_dir + "src/lib.rs")
        broken[prefix + "src/lib.rs"] = b"misleading unrelated source"
        self.tar(path, broken)
        self.rejects("crates/bindings/python/src/lib.rs")

    def test_node_package_requires_license_and_abi_runtime_intrinsically(self):
        for role in M.retained_catalog():
            if role["target"] != "node-package":
                continue
            path = self.fixture(role)
            entries = self.entries(path)
            row = next(r for r in self.inventory["artifacts"] if r["path"] == path.relative_to(self.root).as_posix())
            row["required_members"] = ["package/package.json"]
            for change in ("missing-license", "empty-license", "wrong-license", "missing-runtime", "malformed-runtime", "wrong-runtime"):
                with self.subTest(platform=role["platform"], change=change):
                    broken = dict(entries)
                    meta = json.loads(broken["package/package.json"])
                    if change == "missing-license":
                        del broken["package/LICENSE"]
                    elif change == "empty-license":
                        broken["package/LICENSE"] = b""
                    elif change == "wrong-license":
                        meta["license"] = "MIT"
                    elif change == "missing-runtime":
                        del meta["engines"]
                    elif change == "malformed-runtime":
                        meta["engines"] = None
                    else:
                        meta["engines"]["node"] = ">=20"
                    broken["package/package.json"] = json.dumps(meta).encode()
                    self.tar(path, broken)
                    self.rejects()
            self.tar(path, entries)

    def test_python_wheel_requires_native_metadata_and_license_intrinsically(self):
        for role in M.retained_catalog():
            if role["target"] != "python-wheel":
                continue
            path = self.fixture(role)
            entries = self.entries(path)
            prefix = "grafeo-0.0.1.dist-info/"
            row = next(r for r in self.inventory["artifacts"] if r["path"] == path.relative_to(self.root).as_posix())
            row["required_members"] = [prefix + "METADATA"]
            for change in ("pure", "missing-license", "empty-license", "wrong-expression", "missing-file-header"):
                with self.subTest(platform=role["platform"], change=change):
                    broken = dict(entries)
                    if change == "pure":
                        broken[prefix + "WHEEL"] = broken[prefix + "WHEEL"].replace(b"false", b"true")
                    elif change == "missing-license":
                        del broken[prefix + "licenses/LICENSE"]
                    elif change == "empty-license":
                        broken[prefix + "licenses/LICENSE"] = b""
                    elif change == "wrong-expression":
                        broken[prefix + "METADATA"] = broken[prefix + "METADATA"].replace(b"Apache-2.0", b"MIT")
                    else:
                        broken[prefix + "METADATA"] = broken[prefix + "METADATA"].replace(b"License-File: LICENSE\n", b"")
                    self.zip(path, broken)
                    self.rejects()
            self.zip(path, entries)

    def test_cli_sdist_requires_native_sources_and_builder_intrinsically(self):
        path = self.fixture(dict(package="grafeo-cli", target="cli-sdist", platform="source"))
        entries = self.entries(path)
        prefix = "grafeo_cli-0.0.1/"
        self.inventory["artifacts"][0]["required_members"] = [prefix + "PKG-INFO"]
        for missing in ("hatch_build.py", "rust/Cargo.toml", "rust/Cargo.lock",
                        "rust/crates/grafeo-cli/Cargo.toml", "rust/crates/grafeo-cli/src/main.rs"):
            with self.subTest(missing=missing):
                broken = dict(entries)
                del broken[prefix + missing]
                self.tar(path, broken)
                self.rejects()
        for before, after in ((b'hatchling.build', b'other.build'), (b'hatch_build.py', b'other.py')):
            with self.subTest(after=after):
                broken = dict(entries)
                broken[prefix + "pyproject.toml"] = broken[prefix + "pyproject.toml"].replace(before, after)
                self.tar(path, broken)
                self.rejects()

    def test_cli_wheel_requires_native_metadata_and_license_intrinsically(self):
        for role in M.retained_catalog():
            if role["target"] != "cli-wheel":
                continue
            path = self.fixture(role)
            entries = self.entries(path)
            prefix = "grafeo_cli-0.0.1.dist-info/"
            row = next(r for r in self.inventory["artifacts"] if r["path"] == path.relative_to(self.root).as_posix())
            row["required_members"] = [prefix + "METADATA"]
            for change in ("pure", "missing-pure", "duplicate-pure", "missing-license",
                           "empty-license", "wrong-expression", "missing-file-header", "duplicate-file-header"):
                with self.subTest(platform=role["platform"], change=change):
                    broken = dict(entries)
                    if change == "pure":
                        broken[prefix + "WHEEL"] = broken[prefix + "WHEEL"].replace(b"false", b"true")
                    elif change == "missing-pure":
                        broken[prefix + "WHEEL"] = broken[prefix + "WHEEL"].replace(b"Root-Is-Purelib: false\n", b"")
                    elif change == "duplicate-pure":
                        broken[prefix + "WHEEL"] = b"Root-Is-Purelib: false\n" + broken[prefix + "WHEEL"]
                    elif change == "missing-license":
                        del broken[prefix + "licenses/LICENSE"]
                    elif change == "empty-license":
                        broken[prefix + "licenses/LICENSE"] = b""
                    elif change == "wrong-expression":
                        broken[prefix + "METADATA"] = broken[prefix + "METADATA"].replace(b"Apache-2.0", b"MIT")
                    elif change == "missing-file-header":
                        broken[prefix + "METADATA"] = broken[prefix + "METADATA"].replace(b"License-File: LICENSE\n", b"")
                    else:
                        broken[prefix + "METADATA"] = b"License-File: LICENSE\n" + broken[prefix + "METADATA"]
                    self.zip(path, broken)
                    self.rejects()
            self.zip(path, entries)

    def test_python_metadata_requires_python_wheel_tags_and_sdist_project(self):
        for package, target, platform in (("grafeo", "python-wheel", "linux-x86_64"), ("grafeo-cli", "cli-wheel", "win_amd64"), ("grafeo", "python-sdist", "source"), ("grafeo-cli", "cli-sdist", "source")):
            path = self.fixture(dict(package=package, target=target, platform=platform))
            entries = self.entries(path)
            meta_path = next(name for name in entries if name.endswith(("/METADATA", "/PKG-INFO")))
            raw = entries[meta_path]
            self.rewrite(path, entries | {meta_path: raw.replace(b"Requires-Python: >=3.12", b"Requires-Python: >=3.10").replace(b"Requires-Python: >=3.9", b"Requires-Python: >=3.8")})
            with self.subTest(target=target):
                self.rejects("Requires-Python")
            self.rewrite(path, entries)
            if target.endswith("wheel"):
                wheel_path = next(name for name in entries if name.endswith("/WHEEL"))
                self.rewrite(path, entries | {wheel_path: entries[wheel_path] + b"Tag: cp310-abi3-any\n"})
                self.rejects("tag mismatch")
                self.rewrite(path, entries)

    def test_nuget_all_runtimes_and_xml_entity_rejection(self):
        path = self.fixture(dict(package="Grafeo", target="nuget", platform="net8.0"))
        entries = self.entries(path)
        self.inventory["artifacts"][0]["required_members"] = ["Grafeo.nuspec"]
        for name in entries:
            if name != "Grafeo.nuspec":
                with self.subTest(name=name):
                    self.zip(path, {key: value for key, value in entries.items() if key != name})
                    self.rejects(name)
        xml = '<!DOCTYPE package [<!ENTITY id "Grafeo">]><package><metadata><id>&id;</id><version>0.0.1</version></metadata></package>'
        for encoding in ("utf-8", "utf-16", "utf-16-be"):
            with self.subTest(encoding=encoding):
                self.zip(path, entries | {"Grafeo.nuspec": xml.encode(encoding)})
                self.rejects()

    def test_dart_duplicate_alias_and_nesting_yaml_rejected(self):
        path = self.fixture(dict(package="grafeo", target="dart-source", platform="source"))
        entries = self.entries(path)
        for raw in (b"name: grafeo\nname: grafeo\nversion: 0.0.1\n", b"name: &name grafeo\nversion: 0.0.1\nx: *name\n", b"name: grafeo\nversion: 0.0.1\nx: !custom value\n"):
            with self.subTest(raw=raw):
                self.tar(path, entries | {"pubspec.yaml": raw})
                self.rejects("YAML")

    def test_tar_duplicate_members_and_positive_pax_long_names(self):
        path = self.crate()
        entries = self.entries(path)
        with tarfile.open(path, "w:gz") as archive:
            for _ in range(2):
                info = tarfile.TarInfo("duplicate")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
        self.rejects("duplicate")
        long_name = "grafeo-common-0.0.1/" + "long" * 30 + "/payload"
        for fmt in (tarfile.PAX_FORMAT, tarfile.GNU_FORMAT):
            with tarfile.open(path, "w:gz", format=fmt) as archive:
                for name, value in (entries | {long_name: b"effective payload"}).items():
                    info = tarfile.TarInfo(name)
                    info.size = len(value)
                    archive.addfile(info, io.BytesIO(value))
            manifest = M.build_manifest(self.root, self.inventory)
            self.assertIn(long_name, [m["path"] for m in manifest["artifacts"][0]["members"]])

    def test_zip_unsafe_paths_unsupported_method_and_extra_metadata(self):
        path = self.native_zip()
        entries = self.entries(path)
        for name in ("/absolute", "../outside", "C:/drive", "back\\slash", "a//b", "a/./b", "a\x85b"):
            with self.subTest(name=name):
                self.zip(path, {name: b"x"})
                self.rejects()
        self.zip(path, entries)
        raw = bytearray(path.read_bytes())
        central = raw.index(b"PK\x01\x02")
        struct.pack_into("<H", raw, central+10, 99)
        path.write_bytes(raw)
        self.rejects("unsupported")
        info = zipfile.ZipInfo("payload")
        info.extra = struct.pack("<HH", 0x7075, 4) + b"name"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(info, b"x")
        self.rejects("extra record")

    def test_zip64_local_header_and_data_descriptor_round_trip(self):
        path = self.native_zip()
        entries = self.entries(path)
        # force_zip64 emits a genuine ZIP64 local extra without a large file.
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in entries.items():
                with archive.open(name, "w", force_zip64=True) as stream:
                    stream.write(value)
        M.build_manifest(self.root, self.inventory)
        class NonSeeking(io.BytesIO):
            def seekable(self):
                return False
            def seek(self, *args):
                raise io.UnsupportedOperation("unseekable fixture")
        buffer = NonSeeking()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in entries.items():
                with archive.open(name, "w", force_zip64=True) as stream:
                    stream.write(value)
        path.write_bytes(buffer.getvalue())
        M.build_manifest(self.root, self.inventory)

    def test_zip64_central_end_and_budget_before_records(self):
        path = self.native_zip()
        entries = self.entries(path)
        with mock.patch.object(zipfile, "ZIP_FILECOUNT_LIMIT", 1):
            self.zip(path, entries)
        M.build_manifest(self.root, self.inventory)
        raw = bytearray(path.read_bytes())
        record = raw.index(b"PK\x06\x06")
        struct.pack_into("<Q", raw, record+4, 17 * 1024**2)
        path.write_bytes(raw)
        self.rejects("metadata budget")

    def test_empty_directory_records_canonical_and_file_digests_exact(self):
        path = self.native_zip()
        entries = self.entries(path)
        prefix = next(iter(entries)).rsplit("/", 1)[0]
        entries[prefix + "/empty/"] = b""
        self.zip(path, entries)
        manifest = M.build_manifest(self.root, self.inventory)
        records = manifest["artifacts"][0]["members"]
        self.assertIn({"path": prefix + "/empty", "kind": "directory"}, records)
        for record in records:
            if record["kind"] == "file":
                value = entries[record["path"]]
                self.assertEqual(record["sha256"], hashlib.sha256(value).hexdigest())
                self.assertEqual(record["blake3"], M.hash_factory()(value).hexdigest())

    def test_gzip_optional_header_budget_before_allocation(self):
        path = self.crate()
        raw = bytearray(path.read_bytes())
        # Preserve the old optional filename after an additional FEXTRA block.
        raw[3] |= 4
        raw[10:10] = struct.pack("<H", 4096) + b"x" * 4096
        path.write_bytes(raw)
        with mock.patch.object(M, "MAX_METADATA_BYTES", 2048):
            self.rejects("metadata budget")

    def test_manifest_role_cannot_relabel_wheel_as_opaque(self):
        self.fixture(dict(package="grafeo", target="python-wheel", platform="linux-x86_64"))
        self.inventory["artifacts"][0].update(format="file", required_members=[])
        self.rejects("format does not agree")

    def test_blake3_nonempty_actual_update_failure_preserves_cause(self):
        self.crate()
        factory = M.hash_factory()
        class FailedHash:
            def update(self, data):
                raise RuntimeError("distinctive actual byte hash failure")
        def controlled(*args):
            return factory(*args) if args else FailedHash()
        with mock.patch.object(M, "hash_factory", return_value=controlled):
            self.rejects("distinctive actual byte hash failure")


    def catalog_inventory(self):
        """Descriptor-only catalog fixture: does not claim any package was built."""
        formats = {
            "native-archive": "tar.gz", "cli-native": "file", "c-native": "file",
            "node-native": "file", "node-package": "npm-tgz", "cli-npm": "npm-tgz",
            "wasm-package": "npm-tgz", "python-wheel": "wheel", "cli-wheel": "wheel",
            "python-sdist": "python-sdist", "cli-sdist": "python-sdist",
            "nuget": "nuget", "archive-checksums": "file", "rust-crate": "crate",
            "dart-source": "dart-source",
        }
        inventory = copy.deepcopy(self.inventory)
        inventory["artifacts"] = []
        for index, (package, target, platform) in enumerate(sorted(EXPECTED_RETAINED_ROLES)):
            fmt = "zip" if target == "native-archive" and platform == "x86_64-pc-windows-msvc" else formats[target]
            inventory["artifacts"].append({
                "path": f"catalog-fixture/{index:02d}.fixture",
                "format": fmt, "package": package, "target": target, "platform": platform,
                "required_members": [] if fmt == "file" else ["fixture/payload"],
            })
        return inventory

    def test_retained_catalog_exact_source_derived_rows(self):
        expected = [
            {"package": package, "target": target, "platform": platform}
            for package, target, platform in sorted(EXPECTED_RETAINED_ROLES)
        ]
        self.assertEqual(len(set(EXPECTED_RETAINED_ROLES)), 66)
        self.assertEqual(M.retained_catalog(), expected)

    def test_compare_catalog_accepts_exact_descriptor_roles_only(self):
        self.assertIsNone(M.compare_catalog(self.catalog_inventory()))
        # This check is deliberately independent of artifact existence/provenance.
        self.assertEqual(list(self.root.iterdir()), [])

    def test_compare_catalog_reports_each_missing_expected_role(self):
        complete = self.catalog_inventory()
        for missing in sorted(EXPECTED_RETAINED_ROLES):
            with self.subTest(missing=missing):
                inventory = copy.deepcopy(complete)
                inventory["artifacts"] = [
                    row for row in inventory["artifacts"]
                    if (row["package"], row["target"], row["platform"]) != missing
                ]
                with self.assertRaises(M.ManifestError) as raised:
                    M.compare_catalog(inventory)
                self.assertEqual(raised.exception.location, "catalog")
                self.assertIn(repr(missing), raised.exception.reason)
                self.assertIn("missing roles=", raised.exception.reason)
                self.assertIn("extra roles=[]", raised.exception.reason)

    def test_compare_catalog_rejects_valid_but_undeclared_extra_role(self):
        inventory = self.catalog_inventory()
        extra = ("grafeo", "provenance", "source")
        inventory["artifacts"].append({
            "path": "provenance/source.json", "format": "file",
            "package": extra[0], "target": extra[1], "platform": extra[2],
            "required_members": [],
        })
        inventory["artifacts"].sort(
            key=lambda row: (row["package"], row["target"], row["platform"], row["path"]))
        with self.assertRaises(M.ManifestError) as raised:
            M.compare_catalog(inventory)
        self.assertEqual(raised.exception.location, "catalog")
        self.assertIn("missing roles=[]", raised.exception.reason)
        self.assertIn("extra roles=", raised.exception.reason)
        self.assertIn(repr(extra), raised.exception.reason)

    def test_compare_catalog_additional_coverage_must_be_explicit_and_present(self):
        inventory = self.catalog_inventory()
        additional = [{"package": "grafeo", "target": "provenance", "platform": "source"}]
        with self.assertRaises(M.ManifestError) as raised:
            M.compare_catalog(inventory, additional=additional)
        self.assertEqual(raised.exception.location, "catalog")
        self.assertIn(repr(("grafeo", "provenance", "source")), raised.exception.reason)
        inventory["artifacts"].append({
            **additional[0], "path": "provenance/source.json", "format": "file",
            "required_members": [],
        })
        inventory["artifacts"].sort(
            key=lambda row: (row["package"], row["target"], row["platform"], row["path"]))
        self.assertIsNone(M.compare_catalog(inventory, additional=additional))

    @staticmethod
    def cli_json_bytes(value):
        return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")) + "\n").encode("utf-8")

    def write_cli_document(self, name, value):
        path = Path(self.temp.name).resolve() / name
        path.write_bytes(self.cli_json_bytes(value))
        return path

    def run_manifest_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/release_manifest.py"),
             *map(str, arguments)],
            cwd=self.temp.name, capture_output=True, check=False, timeout=30,
        )

    def assert_cli_failure(self, completed, status=1):
        self.assertEqual(completed.returncode, status, completed.stderr.decode("utf-8", "replace"))
        self.assertEqual(completed.stdout, b"")
        self.assertTrue(completed.stderr)
        if status == 2:
            self.assertIn(b"usage:", completed.stderr.lower())
            return None
        diagnostic = json.loads(completed.stderr)
        self.assertEqual(set(diagnostic), {"location", "reason"})
        for value in diagnostic.values():
            self.assertIsInstance(value, str)
            self.assertTrue(value)
        self.assertEqual(completed.stderr, self.cli_json_bytes(diagnostic))
        return diagnostic

    def cli_inputs(self):
        self.crate()
        manifest = M.build_manifest(self.root, self.inventory)
        return (self.write_cli_document("inventory.json", self.inventory),
                self.write_cli_document("manifest.json", manifest))

    def test_cli_build_and_verify_real_archives_from_an_unrelated_cwd(self):
        self.crate()
        self.native_zip()
        inventory = self.write_cli_document("inventory.json", self.inventory)
        built = self.run_manifest_cli("build", "--root", self.root, "--inventory", inventory)
        self.assertEqual(built.returncode, 0, built.stderr.decode("utf-8", "replace"))
        self.assertEqual(built.stderr, b"")
        manifest = json.loads(built.stdout)
        self.assertEqual(built.stdout, self.cli_json_bytes(manifest))
        self.assertEqual(manifest["candidate_sha"], SHA)
        self.assertEqual(manifest["version"], VERSION)
        self.assertEqual(manifest["toolchains"], self.inventory["toolchains"])
        self.assertEqual(
            [row["path"] for row in manifest["artifacts"]],
            [row["path"] for row in self.inventory["artifacts"]],
        )
        manifest_path = self.write_cli_document("manifest.json", manifest)
        verified = self.run_manifest_cli(
            "verify", "--root", self.root, "--inventory", inventory, "--manifest", manifest_path)
        self.assertEqual(verified.returncode, 0, verified.stderr.decode("utf-8", "replace"))
        self.assertEqual(verified.stderr, b"")
        self.assertEqual(verified.stdout, b'{"byte_agreement":true}\n')

    def test_cli_rejects_malformed_inventory_and_manifest_json_without_stdout(self):
        inventory, manifest = self.cli_inputs()
        malformed = (b"{", b"\xff", b'{"schema_version":1,"schema_version":1}',
                     b'{"value":NaN}', b'{"value":1e999}')
        for kind, path in (("inventory", inventory), ("manifest", manifest)):
            original = path.read_bytes()
            for data in malformed:
                with self.subTest(kind=kind, data=data):
                    path.write_bytes(data)
                    diagnostic = self.assert_cli_failure(self.run_manifest_cli(
                        "verify", "--root", self.root, "--inventory", inventory,
                        "--manifest", manifest))
                    self.assertIn("json", diagnostic["reason"].lower())
            path.write_bytes(original)

    def test_cli_rejects_oversized_json_before_parsing_without_large_fixture_buffers(self):
        inventory, manifest = self.cli_inputs()
        # Sparse files exceed the fixed 16 MiB input budget without allocating
        # a corresponding Python buffer or physically writing 16 MiB of JSON.
        for kind, path in (("inventory", inventory), ("manifest", manifest)):
            with self.subTest(kind=kind):
                original = path.read_bytes()
                with path.open("wb") as stream:
                    stream.truncate(16 * 1024**2 + 1)
                diagnostic = self.assert_cli_failure(self.run_manifest_cli(
                    "verify", "--root", self.root, "--inventory", inventory,
                    "--manifest", manifest))
                self.assertIn("budget", diagnostic["reason"].lower())
                path.write_bytes(original)

    def test_cli_rejects_inventory_inside_artifact_root(self):
        self.crate()
        inventory = self.root / "inventory.json"
        inventory.write_bytes(self.cli_json_bytes(self.inventory))
        diagnostic = self.assert_cli_failure(self.run_manifest_cli(
            "build", "--root", self.root, "--inventory", inventory))
        self.assertIn("outside", diagnostic["reason"].lower())
        self.assertIn("root", diagnostic["reason"].lower())

    def test_cli_rejects_manifest_inside_artifact_root(self):
        inventory, manifest = self.cli_inputs()
        inside = self.root / "manifest.json"
        inside.write_bytes(manifest.read_bytes())
        diagnostic = self.assert_cli_failure(self.run_manifest_cli(
            "verify", "--root", self.root, "--inventory", inventory, "--manifest", inside))
        self.assertIn("outside", diagnostic["reason"].lower())
        self.assertIn("root", diagnostic["reason"].lower())

    def test_cli_argument_errors_exit_two_with_empty_stdout(self):
        inventory, manifest = self.cli_inputs()
        cases = (
            (),
            ("unknown-command",),
            ("build",),
            ("build", "--root", self.root),
            ("verify", "--root", self.root, "--inventory", inventory),
            ("verify", "--root", self.root, "--inventory", inventory,
             "--manifest", manifest, "--unknown-option"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.assert_cli_failure(self.run_manifest_cli(*arguments), status=2)

    def test_cli_verify_context_disagreement_fails_with_empty_stdout(self):
        inventory, manifest = self.cli_inputs()
        independent = copy.deepcopy(self.inventory)
        independent["candidate_sha"] = "b" * 40
        inventory.write_bytes(self.cli_json_bytes(independent))
        self.assert_cli_failure(self.run_manifest_cli(
            "verify", "--root", self.root, "--inventory", inventory, "--manifest", manifest))

    def test_cli_missing_input_is_structured_exit_one_with_empty_stdout(self):
        self.assert_cli_failure(self.run_manifest_cli(
            "build", "--root", self.root,
            "--inventory", Path(self.temp.name) / "missing-inventory.json"))

    def test_early_path_component_budget_before_prefix_materialization(self):
        class NoSplit(str):
            def split(self, *args, **kwargs):
                raise AssertionError("path split/prefix materialization reached")
        deep = NoSplit("/".join(["a"] * 1000))
        names = M.Names()
        for call in (lambda: M.canonical_path(deep, "fixture"),
                     lambda: names.add(deep, False, "fixture"),
                     lambda: M.walk_root(self.root, {deep})):
            with self.subTest(call=call), self.assertRaisesRegex(M.ManifestError, "component budget"):
                call()
        self.assertEqual(names.explicit, set())
        self.assertEqual(names.spelling, {})
        self.assertEqual(names.files, set())
        with mock.patch.object(M, "MAX_DEPTH", 3):
            self.assertEqual(M.canonical_path("a/b/c", "fixture"), "a/b/c")
            with self.assertRaisesRegex(M.ManifestError, "component budget"):
                M.canonical_path("a/b/c/d", "fixture")

    def test_early_json_record_budget_before_parser_materialization(self):
        # Count each container opening, string (including object keys) and bare
        # scalar token once. Array [0,0,0] costs four records including its root.
        for raw in (b"[0,0,0,0,0]", b'{"a":0,"b":0}', b"[[],[],[],[]]"):
            with self.subTest(raw=raw), mock.patch.object(M, "MAX_MEMBERS", 4), \
                    mock.patch.object(M.json, "loads", side_effect=AssertionError("JSON parser materialization reached")) as parser:
                with self.assertRaisesRegex(M.ManifestError, "record budget"):
                    M.parse_json(raw, "fixture")
                parser.assert_not_called()
        with mock.patch.object(M, "MAX_MEMBERS", 4):
            self.assertEqual(M.parse_json(b"[0,0,0]", "fixture"), [0, 0, 0])
            self.assertEqual(M.parse_json(b'{"a":0}', "fixture"), {"a": 0})
        value = ',:{}[]\\"' * 20
        raw = json.dumps([value, True, None]).encode()
        with mock.patch.object(M, "MAX_MEMBERS", 4):
            self.assertEqual(M.parse_json(raw, "fixture"), [value, True, None])
        with mock.patch.object(M, "MAX_MEMBERS", 1):
            self.assertEqual(M.parse_json(b"1.25e-2", "fixture"), 0.0125)

    def test_early_toml_record_and_depth_budget_before_parser(self):
        cases = (("MAX_MEMBERS", 4, b"a=1\nb=2\nc=3\n", "record budget"),
                 ("MAX_DEPTH", 3, b"a=[[[[0]]]]\n", "nesting budget"),
                 ("MAX_DEPTH", 3, b"a.b.c.d=1\n", "component budget"),
                 ("MAX_DEPTH", 3, b"[a.b.c.d]\nx=1\n", "component budget"),
                 ("MAX_METADATA_BYTES", 4, b"long_key=1\n", "metadata budget"))
        for limit, maximum, raw, reason in cases:
            with self.subTest(raw=raw), mock.patch.object(M, limit, maximum), \
                    mock.patch.object(M.tomllib, "loads", side_effect=AssertionError("TOML parser materialization reached")) as parser:
                with self.assertRaisesRegex(M.ManifestError, reason):
                    M.toml_metadata(raw, "fixture")
                parser.assert_not_called()
        raw = b'''"quoted.dots" = "escaped \\" quote and \\\\ slash # [ { ." # ignored { [ .\nliteral = '[{.#]'\nmultiline = """one\n# { [ . \\" still a string\nthree"""\nmultiliteral = ''' + b"'''one\n# { [ .\nthree'''" + b'\nnumber=1.25e-2\ndate=1979-05-27T07:32:00.123Z\n'
        with mock.patch.object(M, "MAX_DEPTH", 3), mock.patch.object(M, "MAX_MEMBERS", 16):
            result = M.toml_metadata(raw, "fixture")
        self.assertIn("quoted.dots", result)
        self.assertEqual(result["number"], 0.0125)
        self.assertEqual(result["date"].year, 1979)
        self.assertIn("# { [ .", result["multiline"])
        self.assertEqual(result["multiliteral"], "one\n# { [ .\nthree")

    def test_early_email_header_budget_and_nonrecursive_body(self):
        raw = b"Name: grafeo\nVersion: 0.0.1\nRequires-Python: >=3.12\nExtra: value\n\n"
        for separator in (b"\n", b"\r\n", b"\r"):
            with self.subTest(separator=separator), mock.patch.object(M, "MAX_MEMBERS", 3), \
                    mock.patch.object(M.email.parser.BytesParser, "parsebytes", side_effect=AssertionError("email parser materialization reached")) as parser:
                with self.assertRaisesRegex(M.ManifestError, "header record budget"):
                    M.metadata_headers(raw.replace(b"\n", separator), "fixture")
                parser.assert_not_called()
        folded = b"Name: grafeo\nVersion: 0.0.1\nSummary: first\n second\n\nbody punctuation: [,{\n"
        with mock.patch.object(M, "MAX_MEMBERS", 4):
            result = M.metadata_headers(folded, "fixture")
        self.assertIn("second", result["Summary"])
        self.assertEqual(result.get_payload(), "body punctuation: [,{\n")
        # Body contains MIME-looking nested data; headers-only parsing leaves
        # the body uninterpreted, never recursively constructing MIME messages.
        mime = b"Name: grafeo\nVersion: 0.0.1\nContent-Type: multipart/mixed; boundary=BOUND\n\n--BOUND\nContent-Type: multipart/mixed; boundary=NEST\n\n--NEST\nContent-Type: text/plain\n\nbody\n--NEST--\n--BOUND--\n"
        with mock.patch.object(M, "MAX_MEMBERS", 3):
            result = M.metadata_headers(mime, "fixture")
        self.assertFalse(result.is_multipart())
        self.assertIsInstance(result.get_payload(), str)

    def test_early_wheel_header_budget_uses_shared_preflight(self):
        path = self.fixture(dict(package="grafeo", target="python-wheel", platform="linux-x86_64"))
        entries = self.entries(path)
        wheel_path = next(name for name in entries if name.endswith("/WHEEL"))
        entries[wheel_path] = b"Wheel-Version: 1.0\nTag: cp312-abi3-manylinux_2_28_x86_64\nExtra: a\nExtra: b\nExtra: c\nExtra: d\nExtra: e\n\n"
        self.zip(path, entries)
        # Isolate intrinsic inspection after real ZIP streaming so this small
        # header limit is not consumed first by the inventory JSON envelope.
        row = self.inventory["artifacts"][0]
        members = M.Members(row, M.hash_factory(), M.Budget(row["path"]))
        with path.open("rb") as file:
            M.inspect_zip(file, members)
        # METADATA now has six headers including its license declarations;
        # WHEEL alone exceeds this bound with seven.
        original_parser = M.email.parser.BytesParser.parsebytes
        with mock.patch.object(M, "MAX_MEMBERS", 6), \
                mock.patch.object(M.email.parser.BytesParser, "parsebytes", autospec=True, side_effect=original_parser) as parser:
            with self.assertRaisesRegex(M.ManifestError, "header record budget"):
                M.inspect_package(members, self.inventory)
            self.assertEqual(parser.call_count, 1)  # METADATA only, never WHEEL.

    def test_early_go_blank_line_budget_and_duplicate_module(self):
        path = self.fixture(dict(package=M.GO_MODULE, target="go-source", platform="source"))
        entries = self.entries(path)
        row = self.inventory["artifacts"][0]
        members = M.Members(row, M.hash_factory(), M.Budget(row["path"]))
        with path.open("rb") as file:
            M.inspect_tar(file, members)
        original = members.metadata["go.mod"]
        members.metadata["go.mod"] = b"\n" * 5 + original
        with mock.patch.object(M, "MAX_MEMBERS", 4), \
                mock.patch.object(M.re, "findall", side_effect=AssertionError("Go match collection reached")) as collector:
            with self.assertRaisesRegex(M.ManifestError, "line record budget"):
                M.inspect_package(members, self.inventory)
            collector.assert_not_called()
        members.metadata["go.mod"] = b"\n\n" + original
        with mock.patch.object(M, "MAX_MEMBERS", 5):
            M.inspect_package(members, self.inventory)
        members.metadata["go.mod"] = original + ("module " + M.GO_MODULE + "\n").encode()
        with self.assertRaises(M.ManifestError):
            M.inspect_package(members, self.inventory)

    def test_early_checksum_line_budget_before_line_collection(self):
        rows = [{"target": "native-archive", "platform": row[0], "path": str(index) + ".archive", "sha256": "0" * 64} for index, row in enumerate(M.NATIVE)]
        raw = "".join(row["sha256"] + "  " + row["path"] + "\n" for row in rows).encode()
        class NoSplitLines(str):
            def splitlines(self, *args, **kwargs):
                raise AssertionError("checksum line collection reached")
        class GuardedBytes(bytes):
            def decode(self, *args, **kwargs):
                return NoSplitLines(super().decode(*args, **kwargs))
        with mock.patch.object(M, "MAX_MEMBERS", 4):
            with self.assertRaisesRegex(M.ManifestError, "line record budget"):
                M.check_archive_checksums(GuardedBytes(raw), rows, "fixture")
        with mock.patch.object(M, "MAX_MEMBERS", 5):
            M.check_archive_checksums(raw, rows, "fixture")


if __name__ == "__main__":
    unittest.main()
