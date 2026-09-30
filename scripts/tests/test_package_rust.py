"""Rust12 producer/consumer boundary tests; command doubles never execute Cargo."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = (
    "grafeo-common", "grafeo-core", "grafeo-storage", "grafeo-adapters",
    "grafeo-engine", "grafeo", "grafeo-cli", "grafeo-bindings-common",
    "grafeo-python", "grafeo-node", "grafeo-c", "grafeo-wasm",
)
# A tiny, real Cargo-shaped graph. Unused packaged roles remain patched, while
# the executable consumer reaches the facade, engine, core and common crates.
DEPENDENCIES = {
    "grafeo-common": (),
    "grafeo-core": ("grafeo-common",),
    "grafeo-engine": ("grafeo-core",),
    "grafeo": ("grafeo-engine",),
}


def load_producer():
    spec = importlib.util.spec_from_file_location(
        "package_rust_test_subject", ROOT / "scripts/package_rust.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def success(stdout=""):
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


class Fixture:
    def __init__(self, base, producer):
        base = base.resolve()
        self.producer = producer
        self.root = base / "candidate"
        self.stage = base / "staged"
        self.target = base / "package-target"
        self.consumer = base / "consumer"
        self.consumer_target = base / "consumer-target"
        self.root.mkdir()
        self.calls = []
        workspace = ['[workspace]', 'members = ["members/*"]', 'resolver = "2"',
                     '[workspace.package]', 'version = "0.0.1"',
                     'edition = "2024"', '[workspace.dependencies]']
        workspace.extend(
            f'{name} = {{ path = "members/{name}", version = "0.0.1" }}'
            for name in PACKAGES
        )
        (self.root / "Cargo.toml").write_text("\n".join(workspace) + "\n")
        (self.root / "Cargo.lock").write_text("version = 4\n")
        for name in PACKAGES:
            member = self.root / "members" / name
            (member / "src").mkdir(parents=True)
            (member / "Cargo.toml").write_text(
                f'[package]\nname = "{name}"\nversion.workspace = true\n'
                'edition.workspace = true\n[dependencies]\n' + "".join(
                    f'{dep} = {{ workspace = true }}\n'
                    for dep in DEPENDENCIES.get(name, ())
                )
            )
            (member / "src/lib.rs").write_text("pub fn value() -> u8 { 7 }\n")
        self.inventory = {"artifacts": [
            {"path": f"rust/{name}-0.0.1.crate", "format": "crate",
             "package": name, "target": "rust-crate", "platform": "source",
             "required_members": ["Cargo.toml", "src/lib.rs"]}
            for name in PACKAGES
        ]}
        self.seal()

    def seal(self):
        self.context = {
            "candidate_sha": "a" * 40, "version": "0.0.1",
            "source_files": {
                path.relative_to(self.root).as_posix(): digest(path)
                for path in self.root.rglob("*") if path.is_file()
            },
        }

    def archive(self, package, *, duplicate=False, unsafe=False):
        directory = self.target / "package"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{package}-0.0.1.crate"
        manifest = (
            f'[package]\nname = "{package}"\nversion = "0.0.1"\n'
            'edition = "2024"\n[dependencies]\n' + "".join(
                f'{dep} = "=0.0.1"\n' for dep in DEPENDENCIES.get(package, ())
            )
        ).encode()
        prefix = f"{package}-0.0.1/"
        with tarfile.open(path, "w:gz") as archive:
            for name, payload in [("Cargo.toml", manifest),
                                  ("src/lib.rs", b"pub fn value() -> u8 { 7 }\n")]:
                info = tarfile.TarInfo(prefix + name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            if duplicate:
                info = tarfile.TarInfo(prefix + "src/lib.rs")
                info.size = 3
                archive.addfile(info, io.BytesIO(b"bad"))
            if unsafe:
                info = tarfile.TarInfo(prefix + "../escape")
                info.size = 3
                archive.addfile(info, io.BytesIO(b"bad"))
        return path

    def package_runner(self, argv, cwd):
        self.calls.append((list(argv), Path(cwd)))
        # Write only to the explicit output root. A default target/package
        # assumption in the producer therefore cannot make this fixture pass.
        assert argv[argv.index("--target-dir") + 1] == str(self.target)
        for package in PACKAGES:
            self.archive(package)
        return success()

    def stage_packages(self, runner=None):
        return self.producer.stage_rust_packages(
            self.root, self.stage, self.inventory, self.context,
            command_runner=self.package_runner if runner is None else runner,
            target_dir=self.target,
        )

    def make_consumer(self, receipt):
        self.consumer.mkdir()
        self.unpacked = {
            row["package"]: Path(row["unpacked_root"])
            for row in receipt["archives"]
        }
        manifest = ['[package]', 'name = "rust12-consumer"', 'version = "0.0.0"',
                    'edition = "2024"', '[dependencies]', 'grafeo = "=0.0.1"',
                    '[patch.crates-io]']
        manifest.extend(
            f'{name} = {{ path = {json.dumps(str(self.unpacked[name]))} }}'
            for name in PACKAGES
        )
        (self.consumer / "Cargo.toml").write_text("\n".join(manifest) + "\n")
        (self.consumer / "Cargo.lock").write_text("version = 4\n")
        (self.consumer / "src").mkdir()
        (self.consumer / "src/main.rs").write_text(
            'fn main() { assert_eq!(grafeo::value(), 7); }\n'
        )
        return self.metadata()

    def metadata(self):
        paths = {"rust12-consumer": self.consumer, **{
            name: self.unpacked[name] for name in DEPENDENCIES
        }}
        ids = {name: f"path+{path.as_uri()}#{name}@"
               + ("0.0.0" if name == "rust12-consumer" else "0.0.1")
               for name, path in paths.items()}
        edges = {"rust12-consumer": ("grafeo",), **DEPENDENCIES}
        packages = []
        nodes = []
        for name, path in paths.items():
            version = "0.0.0" if name == "rust12-consumer" else "0.0.1"
            packages.append({"name": name, "version": version, "id": ids[name],
                             "manifest_path": str(path / "Cargo.toml"),
                             "source": None, "dependencies": [], "targets": []})
            nodes.append({"id": ids[name], "dependencies": [ids[n] for n in edges[name]],
                          "deps": [{"name": n.replace("-", "_"), "pkg": ids[n],
                                    "dep_kinds": [{"kind": None, "target": None}]}
                                   for n in edges[name]], "features": []})
        return {"packages": packages, "workspace_members": [ids["rust12-consumer"]],
                "resolve": {"root": ids["rust12-consumer"], "nodes": nodes},
                "workspace_root": str(self.consumer),
                "target_directory": str(self.consumer_target), "version": 1}

    def verify(self, receipt, runner, **kwargs):
        return self.producer.verify_clean_consumer(
            self.consumer, runner, staged_receipt=receipt,
            target_dir=self.consumer_target, **kwargs,
        )


class RustPackageTests(unittest.TestCase):
    def setUp(self):
        self.producer = load_producer()

    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Fixture(Path(temporary.name), self.producer)

    def assert_not_staged(self, fixture):
        self.assertFalse(list(fixture.stage.glob("*.crate")))
        self.assertFalse((fixture.stage / "release_complete").exists())

    def test_binding_path_dependencies_keep_versions_and_effective_defaults(self):
        workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["dependencies"]
        for manifest, name, default in [
            ("crates/bindings/common/Cargo.toml", "grafeo-engine", False),
            ("crates/bindings/common/Cargo.toml", "grafeo-common", True),
            ("crates/bindings/wasm/Cargo.toml", "grafeo-bindings-common", False),
        ]:
            with self.subTest(manifest=manifest, dependency=name):
                member = tomllib.loads((ROOT / manifest).read_text())["dependencies"][name]
                inherited = workspace[name] if member.get("workspace") else {}
                effective = {**inherited, **member}
                self.assertEqual(effective.get("version"), "0.0.1")
                self.assertIn("path", effective)
                self.assertEqual(effective.get("default-features", True), default)

    def test_one_verified_command_stages_all_twelve_inherited_version_crates(self):
        fixture = self.fixture()
        receipt = fixture.stage_packages()
        self.assertEqual(receipt["status"], "staged")
        self.assertIs(receipt["release_complete"], False)
        self.assertEqual(receipt["candidate_sha"], fixture.context["candidate_sha"])
        self.assertEqual(len(fixture.calls), 1)
        argv, cwd = fixture.calls[0]
        self.assertEqual(argv[:3], ["cargo", "+1.97.1", "package"])
        self.assertEqual(cwd, fixture.root)
        self.assertIn("--locked", argv)
        self.assertIn("--offline", argv)
        self.assertNotIn("--no-verify", argv)
        self.assertEqual([argv[i + 1] for i, arg in enumerate(argv) if arg == "-p"], list(PACKAGES))
        self.assertEqual(argv[argv.index("--manifest-path") + 1], str(fixture.root / "Cargo.toml"))
        self.assertEqual(argv[argv.index("--target-dir") + 1], str(fixture.target))
        self.assertEqual({row["package"] for row in receipt["archives"]}, set(PACKAGES))
        self.assertEqual({p.name for p in fixture.stage.glob("*.crate")},
                         {f"{name}-0.0.1.crate" for name in PACKAGES})
        for row in receipt["archives"]:
            archive = Path(row["archive"])
            unpacked = Path(row["unpacked_root"])
            self.assertEqual(archive.read_bytes(), (fixture.target / "package" / archive.name).read_bytes())
            self.assertEqual(row["sha256"], digest(archive))
            self.assertEqual(row["files"], {
                name: digest(unpacked / name) for name in ("Cargo.toml", "src/lib.rs")
            })
        self.assertFalse((fixture.stage / "release_complete").exists())

    def test_missing_or_wrong_inherited_versions_fail_before_cargo(self):
        for change in ("workspace-version", "missing-path-version", "inherited-path-version"):
            with self.subTest(change=change):
                fixture = self.fixture()
                manifest = fixture.root / "Cargo.toml"
                text = manifest.read_text()
                if change == "workspace-version":
                    text = text.replace('version = "0.0.1"', 'version = "9.0.0"', 1)
                elif change == "inherited-path-version":
                    text = text.replace('version = "0.0.1" }', 'version = "9.0.0" }', 1)
                else:
                    member = fixture.root / "members/grafeo-core/Cargo.toml"
                    member.write_text(member.read_text().replace(
                        'grafeo-common = { workspace = true }',
                        'grafeo-common = { path = "../grafeo-common" }'))
                manifest.write_text(text)
                fixture.seal()
                with self.assertRaises(self.producer.PackageError):
                    fixture.stage_packages()
                self.assertEqual(fixture.calls, [])
                self.assert_not_staged(fixture)

    def test_exact_source_inventory_rejects_changed_extra_missing_and_directory_links(self):
        for change in ("changed", "extra", "missing", "directory-link", "file-link"):
            with self.subTest(change=change):
                fixture = self.fixture()
                source = fixture.root / "members/grafeo/src/lib.rs"
                if change == "changed":
                    source.write_text("pub fn substituted() {}\n")
                elif change == "extra":
                    (fixture.root / "unsealed.rs").write_text("// omitted\n")
                elif change == "missing":
                    source.unlink()
                elif change == "directory-link":
                    (fixture.root / "unsealed-dir").symlink_to(fixture.root / "members", target_is_directory=True)
                else:
                    (fixture.root / "unsealed-link.rs").symlink_to(source)
                with self.assertRaises(self.producer.PackageError):
                    fixture.stage_packages()
                self.assertEqual(fixture.calls, [])
                self.assert_not_staged(fixture)

    def test_role_duplicates_missing_roles_and_stale_outputs_never_invoke_cargo(self):
        for change in ("duplicate-role", "missing-role", "stale-stage", "stale-target"):
            with self.subTest(change=change):
                fixture = self.fixture()
                if change == "duplicate-role":
                    fixture.inventory["artifacts"].append(copy.deepcopy(fixture.inventory["artifacts"][0]))
                elif change == "missing-role":
                    fixture.inventory["artifacts"].pop()
                elif change == "stale-stage":
                    fixture.stage.mkdir()
                    (fixture.stage / "grafeo-common-0.0.1.crate").write_bytes(b"stale")
                else:
                    fixture.archive("grafeo-common")
                with self.assertRaises(self.producer.PackageError):
                    fixture.stage_packages()
                self.assertEqual(fixture.calls, [])
                if change == "stale-stage":
                    self.assertEqual((fixture.stage / "grafeo-common-0.0.1.crate").read_bytes(), b"stale")
                else:
                    self.assert_not_staged(fixture)

    def test_explicit_target_directories_cannot_overlap_source_stage_or_consumer(self):
        for location in ("candidate", ".git", "staged"):
            with self.subTest(package_target=location):
                fixture = self.fixture()
                if location == "candidate":
                    fixture.target = fixture.root / "source-output"
                elif location == ".git":
                    fixture.target = fixture.root / ".git" / "output"
                else:
                    fixture.target = fixture.stage / "target"
                with self.assertRaises(self.producer.PackageError):
                    fixture.stage_packages()
                self.assertEqual(fixture.calls, [])
        fixture = self.fixture()
        staged = fixture.stage_packages()
        fixture.make_consumer(staged)
        fixture.consumer_target = fixture.consumer / "target"
        calls = []
        with self.assertRaises(self.producer.PackageError):
            fixture.verify(staged, lambda argv, cwd: calls.append(argv) or success())
        self.assertEqual(calls, [])

    def test_reuses_excluded_target_cache_with_disjoint_staging_and_consumer(self):
        fixture = self.fixture()
        excluded = fixture.root / "target"
        fixture.target = excluded / "warm-cache"
        fixture.stage = excluded / "staged"
        fixture.consumer = excluded / "consumer"
        fixture.consumer_target = fixture.target
        cached = fixture.target / "debug/deps/existing.rlib"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"preserved build cache")
        source_seal = copy.deepcopy(fixture.context)
        staged = fixture.stage_packages()
        metadata = fixture.make_consumer(staged)
        calls = []
        def runner(argv, cwd):
            calls.append(list(argv))
            return success(json.dumps(metadata)) if argv[2] == "metadata" else success()
        receipt = fixture.verify(staged, runner)
        self.assertEqual(staged["status"], "staged")
        self.assertEqual(receipt["status"], "verified")
        self.assertEqual([argv[2] for argv in calls], ["metadata", "check", "run", "metadata"])
        self.assertEqual(len(fixture.calls), 1)
        self.assertEqual(cached.read_bytes(), b"preserved build cache")
        self.assertEqual(fixture.context, source_seal)
        for relative, expected in source_seal["source_files"].items():
            self.assertEqual(digest(fixture.root / relative), expected)
        self.assertEqual(Path(staged["target_dir"]), fixture.target)
        self.assertEqual(Path(staged["stage_root"]), fixture.stage)

    def test_failed_partial_duplicate_and_unsafe_package_outputs_cannot_stage(self):
        for change in ("command-failure", "partial", "duplicate-member", "unsafe-member", "unexpected-archive"):
            with self.subTest(change=change):
                fixture = self.fixture()
                calls = []
                def runner(argv, cwd):
                    calls.append(list(argv))
                    for package in PACKAGES if change != "partial" else PACKAGES[:-1]:
                        fixture.archive(package,
                                        duplicate=change == "duplicate-member" and package == "grafeo",
                                        unsafe=change == "unsafe-member" and package == "grafeo")
                    if change == "unexpected-archive":
                        (fixture.target / "package/stale-0.0.1.crate").write_bytes(b"stale")
                    return SimpleNamespace(returncode=1 if change == "command-failure" else 0,
                                           stdout="", stderr="injected package failure")
                with self.assertRaises(self.producer.PackageError):
                    fixture.stage_packages(runner)
                self.assertEqual(len(calls), 1)
                self.assert_not_staged(fixture)

    def test_source_mutation_during_package_cannot_receive_a_staged_receipt(self):
        fixture = self.fixture()
        def runner(argv, cwd):
            result = fixture.package_runner(argv, cwd)
            (fixture.root / "members/grafeo/src/lib.rs").write_text("// raced source\n")
            return result
        with self.assertRaises(self.producer.PackageError):
            fixture.stage_packages(runner)
        self.assert_not_staged(fixture)

    def test_clean_consumer_compiles_and_runs_exact_staged_graph(self):
        fixture = self.fixture()
        staged = fixture.stage_packages()
        metadata = fixture.make_consumer(staged)
        calls = []
        def runner(argv, cwd):
            calls.append((list(argv), Path(cwd)))
            return success(json.dumps(metadata)) if argv[2] == "metadata" else success()
        database_path = fixture.consumer.parent / "consumer-database.grafeo"
        receipt = fixture.verify(staged, runner, caller_args=(str(database_path),))
        self.assertEqual(receipt["status"], "verified")
        self.assertEqual(receipt["commands"], [argv for argv, _ in calls])
        run = next(argv for argv, _ in calls if argv[2] == "run")
        self.assertEqual(run[run.index("--") + 1:], [str(database_path)])
        self.assertFalse(database_path.exists(), "the command double never executes the caller")
        self.assertEqual([argv[2] for argv, _ in calls], ["metadata", "check", "run", "metadata"])
        for argv, cwd in calls:
            self.assertEqual(cwd, fixture.consumer)
            self.assertIn("--locked", argv)
            self.assertIn("--offline", argv)
            self.assertEqual(argv[argv.index("--manifest-path") + 1], str(fixture.consumer / "Cargo.toml"))
            if argv[2] != "metadata":
                self.assertEqual(argv[argv.index("--target-dir") + 1], str(fixture.consumer_target))
            else:
                self.assertIn(str(fixture.consumer_target), argv[argv.index("--config") + 1])

    def test_consumer_rejects_checkout_patch_fake_graph_and_wrong_package_identity(self):
        for change in ("checkout-patch", "checkout-metadata", "unreachable-engine", "wrong-version", "missing-node"):
            with self.subTest(change=change):
                fixture = self.fixture()
                staged = fixture.stage_packages()
                metadata = fixture.make_consumer(staged)
                if change == "checkout-patch":
                    manifest = fixture.consumer / "Cargo.toml"
                    manifest.write_text(manifest.read_text().replace(
                        str(fixture.unpacked["grafeo-engine"]),
                        str(fixture.root / "members/grafeo-engine")))
                elif change == "checkout-metadata":
                    next(p for p in metadata["packages"] if p["name"] == "grafeo-engine")["manifest_path"] = str(fixture.root / "members/grafeo-engine/Cargo.toml")
                elif change == "wrong-version":
                    next(p for p in metadata["packages"] if p["name"] == "grafeo-engine")["version"] = "9.0.0"
                elif change == "missing-node":
                    metadata["resolve"]["nodes"].pop()
                else:
                    root_node = next(n for n in metadata["resolve"]["nodes"] if n["id"] == metadata["resolve"]["root"])
                    root_node["dependencies"] = []
                    root_node["deps"] = []
                calls = []
                def runner(argv, cwd):
                    calls.append(argv[2])
                    return success(json.dumps(metadata))
                with self.assertRaises(self.producer.PackageError):
                    fixture.verify(staged, runner)
                self.assertNotIn("check", calls)
                self.assertNotIn("run", calls)

    def test_consumer_rejects_archive_unpacked_and_caller_tampering(self):
        for change in ("archive-before", "unpacked-before", "archive-during", "unpacked-during", "caller-during", "graph-during"):
            with self.subTest(change=change):
                fixture = self.fixture()
                staged = fixture.stage_packages()
                metadata = fixture.make_consumer(staged)
                archive = Path(next(r for r in staged["archives"] if r["package"] == "grafeo")["archive"])
                unpacked = fixture.unpacked["grafeo"] / "src/lib.rs"
                if change == "archive-before":
                    archive.write_bytes(archive.read_bytes() + b"tampered")
                elif change == "unpacked-before":
                    unpacked.write_text("pub fn value() -> u8 { 99 }\n")
                calls = []
                def runner(argv, cwd):
                    calls.append(argv[2])
                    if argv[2] == "check":
                        if change == "archive-during":
                            archive.write_bytes(archive.read_bytes() + b"tampered")
                        elif change == "unpacked-during":
                            unpacked.write_text("pub fn value() -> u8 { 99 }\n")
                        elif change == "caller-during":
                            (fixture.consumer / "src/main.rs").write_text("fn main() {}\n")
                        elif change == "graph-during":
                            metadata["resolve"]["nodes"][0]["dependencies"] = []
                            metadata["resolve"]["nodes"][0]["deps"] = []
                    return success(json.dumps(metadata)) if argv[2] == "metadata" else success()
                with self.assertRaises(self.producer.PackageError):
                    fixture.verify(staged, runner)
                if change.endswith("before"):
                    self.assertEqual(calls, [])

    def test_consumer_stops_before_run_after_compile_failure(self):
        fixture = self.fixture()
        staged = fixture.stage_packages()
        metadata = fixture.make_consumer(staged)
        calls = []
        def runner(argv, cwd):
            calls.append(argv[2])
            if argv[2] == "metadata":
                return success(json.dumps(metadata))
            return SimpleNamespace(returncode=1, stdout="", stderr="injected compile failure")
        with self.assertRaises(self.producer.PackageError):
            fixture.verify(staged, runner)
        self.assertEqual(calls, ["metadata", "check"])


if __name__ == "__main__":
    unittest.main()
