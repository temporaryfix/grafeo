#!/usr/bin/env python3
"""Local Rust12 staging and archive-bound consumer checks; never publishes.

The caller owns immutable source/staging roots and supplies the Cargo runner.
Byte agreement is not proof of Git provenance, runner execution or publication
authority. candidate_sha is a caller label; final-index binding is external.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_manifest", ROOT / "scripts/release_manifest.py")
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)
PACKAGE_ORDER = (
    "grafeo-common", "grafeo-core", "grafeo-storage", "grafeo-adapters",
    "grafeo-engine", "grafeo", "grafeo-cli", "grafeo-bindings-common",
    "grafeo-python", "grafeo-node", "grafeo-c", "grafeo-wasm",
)
REQUIRED_CONSUMER_PACKAGES = ("grafeo", "grafeo-engine", "grafeo-core", "grafeo-common")


class PackageError(ValueError):
    """Local qualification failure, never a publication receipt."""


def _path(value) -> Path:
    path = Path(os.path.abspath(value))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise PackageError(f"link in path: {part}")
    return path


def _hash_file(path: Path) -> str:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PackageError(f"only non-linked regular files admitted: {path}")
    if info.st_size > MANIFEST.MAX_MEMBER_BYTES:
        raise PackageError(f"file exceeds byte bound: {path}")
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
        before = os.fstat(stream.fileno())
        if MANIFEST.stat_identity(before) != MANIFEST.stat_identity(info):
            raise PackageError(f"file changed before reading: {path}")
        digest, count = hashlib.sha256(), 0
        while data := stream.read(MANIFEST.CHUNK):
            count += len(data)
            if count > MANIFEST.MAX_MEMBER_BYTES:
                raise PackageError(f"file exceeds byte bound: {path}")
            digest.update(data)
        if (count != before.st_size or
                MANIFEST.stat_identity(os.fstat(stream.fileno())) != MANIFEST.stat_identity(before) or
                MANIFEST.stat_identity(path.lstat()) != MANIFEST.stat_identity(before)):
            raise PackageError(f"file changed while reading: {path}")
    return digest.hexdigest()


def _tree(root: Path, *, source=False) -> dict[str, str]:
    root = _path(root)
    if not root.is_dir():
        raise PackageError(f"directory required: {root}")
    found, pending, names = {}, [root], MANIFEST.Names()
    count = total = 0
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    relative = (directory / entry.name).relative_to(root).as_posix()
                    info = entry.stat(follow_symlinks=False)
                    is_dir = stat.S_ISDIR(info.st_mode)
                    if not is_dir and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
                        raise PackageError(f"source/tree rejects link or special file: {relative}")
                    # Reject links before pruning these root-only exclusions.
                    if source and directory == root and entry.name in {"target", ".git"}:
                        continue
                    count += 1
                    if count > MANIFEST.MAX_MEMBERS:
                        raise PackageError("tree member bound exceeded")
                    names.add(relative, is_dir, relative)
                    if is_dir:
                        pending.append(Path(entry.path))
                    else:
                        total += info.st_size
                        if total > MANIFEST.MAX_EXPANDED_BYTES:
                            raise PackageError("tree byte bound exceeded")
                        found[relative] = _hash_file(Path(entry.path))
    except MANIFEST.ManifestError as error:
        raise PackageError(str(error)) from error
    return found


def tables_for_target(target: dict):
    return (target.get(key, {}) for key in ("dependencies", "dev-dependencies", "build-dependencies"))


def missing_path_dependency_versions(manifest_text: str, workspace_text: str | None = None) -> list[str]:
    data = tomllib.loads(manifest_text)
    workspace = tomllib.loads(workspace_text).get("workspace", {}) if workspace_text else {}
    inherited = workspace.get("dependencies", {})
    tables = list(tables_for_target(data))
    for target in data.get("target", {}).values():
        if isinstance(target, dict):
            tables.extend(tables_for_target(target))
    missing = []
    for table in tables:
        for name, dependency in table.items():
            if not isinstance(dependency, dict):
                continue
            if dependency.get("workspace") is True:
                if name not in inherited:
                    missing.append(name)
                    continue
                dependency = inherited[name]
            if isinstance(dependency, dict) and "path" in dependency and dependency.get("version") != "0.0.1":
                missing.append(name)
    return missing


def _require_source_context(context: dict, root: Path) -> None:
    if not isinstance(context, dict) or not re.fullmatch("[0-9a-f]{40}", str(context.get("candidate_sha", ""))):
        raise PackageError("candidate_sha must be a lowercase 40-character hex digest")
    if context.get("version") != "0.0.1":
        raise PackageError("version must be 0.0.1")
    files = context.get("source_files")
    if not isinstance(files, dict) or not {"Cargo.toml", "Cargo.lock"} <= files.keys():
        raise PackageError("source_files inventory with Cargo.toml and Cargo.lock required")
    if _tree(root, source=True) != files:
        raise PackageError("source inventory does not exactly match candidate_root")


def _require_candidate(candidate_root) -> Path:
    root = _path(candidate_root)
    source = _tree(root, source=True)
    workspace_text = (root / "Cargo.toml").read_text()
    workspace = tomllib.loads(workspace_text).get("workspace", {})
    found = set()
    for name in source:
        if Path(name).name != "Cargo.toml":
            continue
        text = (root / name).read_text()
        package = tomllib.loads(text).get("package")
        if not isinstance(package, dict) or package.get("name") not in PACKAGE_ORDER:
            continue
        package_name = package["name"]
        if package_name in found:
            raise PackageError(f"duplicate package manifest: {package_name}")
        version = package.get("version")
        if isinstance(version, dict) and version == {"workspace": True}:
            version = workspace.get("package", {}).get("version")
        if version != "0.0.1":
            raise PackageError(f"package {package_name} version must resolve to 0.0.1")
        missing = missing_path_dependency_versions(text, workspace_text)
        if missing:
            raise PackageError(f"path dependencies missing version in {name}: {missing}")
        found.add(package_name)
    if found != set(PACKAGE_ORDER):
        raise PackageError(f"rust package manifests missing: {sorted(set(PACKAGE_ORDER) - found)}")
    return root


def _rust_crate_roles(inventory: dict) -> set[str]:
    rows = inventory.get("artifacts") if isinstance(inventory, dict) else None
    if not isinstance(rows, list) or len(rows) > MANIFEST.MAX_MEMBERS:
        raise PackageError("bounded inventory artifacts required")
    names = []
    for row in rows:
        if not isinstance(row, dict):
            raise PackageError("inventory artifact rows must be objects")
        if row.get("target") == "rust-crate":
            if row.get("format") != "crate" or row.get("platform") != "source":
                raise PackageError("rust-crate must have crate/source identity")
            names.append(row.get("package"))
    if len(names) != 12 or set(names) != set(MANIFEST.RUST_PACKAGES) or set(names) != set(PACKAGE_ORDER):
        raise PackageError("rust-crate roles must be exactly the twelve distinct packages")
    return set(names)


def package_command(packages=PACKAGE_ORDER, candidate_root=None, *, target_dir) -> list[str]:
    if isinstance(packages, str):
        packages = (packages,)
    if len(packages) != 12 or set(packages) != set(PACKAGE_ORDER):
        raise PackageError("one package command must select exactly all twelve packages")
    if candidate_root is None:
        raise PackageError("candidate_root required")
    argv = ["cargo", "+1.97.1", "package", "--locked", "--offline", "--target-dir", str(_path(target_dir)),
            "--manifest-path", str(_path(candidate_root) / "Cargo.toml")]
    for package in PACKAGE_ORDER:
        argv.extend(["-p", package])
    return argv


class _CrateMembers:
    """Reuse the bounded release TAR grammar, paths and collision controls."""
    def __init__(self, path: Path, package: str, destination: Path | None = None):
        self.row = {"path": str(path)}
        self.budget = MANIFEST.Budget(str(path))
        self.names = MANIFEST.Names()
        self.prefix = package + "-0.0.1"
        self.files = {}
        self.manifest = bytearray()
        self.destination = destination
        self.output = None

    def begin(self, raw_name, directory, size):
        name = MANIFEST.canonical_path(raw_name, self.row["path"], directory)
        self.names.add(name, directory, name)
        if not name.startswith(self.prefix + "/") and not (directory and name == self.prefix):
            raise PackageError("crate member outside exact package root")
        if size > MANIFEST.MAX_MEMBER_BYTES or (directory and size):
            raise PackageError("crate member byte bound exceeded")
        relative = name[len(self.prefix) + 1:]
        if relative == "Cargo.toml" and size > MANIFEST.MAX_METADATA_BYTES:
            raise PackageError("crate manifest exceeds metadata bound")
        if self.destination is not None:
            output = self.destination / name
            if directory:
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                self.output = output.open("xb")
        return relative, directory, hashlib.sha256(), [0]

    def feed(self, state, data, charge=True):
        relative, directory, digest, count = state
        if directory and data:
            raise PackageError("directory payload forbidden")
        count[0] += len(data)
        digest.update(data)
        if relative == "Cargo.toml":
            self.manifest.extend(data)
        if self.output is not None:
            self.output.write(data)

    def finish(self, state, expected_size):
        relative, directory, digest, count = state
        if count[0] != expected_size:
            raise PackageError("crate member length mismatch")
        if self.output is not None:
            self.output.close()
            self.output = None
        if not directory:
            self.files[relative] = digest.hexdigest()


def _verify_crate(path: Path, package: str, *, destination=None) -> dict:
    path = _path(path)
    if package not in PACKAGE_ORDER or path.name != f"{package}-0.0.1.crate":
        raise PackageError("unexpected Cargo archive identity")
    before = _hash_file(path)
    members = _CrateMembers(path, package, destination)
    try:
        with path.open("rb") as stream:
            MANIFEST.inspect_tar(stream, members)
        data = MANIFEST.toml_metadata(bytes(members.manifest), str(path))
        metadata = data.get("package", {})
        if metadata.get("name") != package or metadata.get("version") != "0.0.1":
            raise PackageError("crate metadata identity mismatch")
        empty_digest = hashlib.sha256(b"").hexdigest()
        if not any(members.files.get(name) not in (None, empty_digest)
                   for name in ("src/lib.rs", "src/main.rs")):
            raise PackageError("crate requires a nonempty Rust source member")
        if _hash_file(path) != before:
            raise PackageError("archive bytes changed during inspection")
    except (MANIFEST.ManifestError, OSError, ValueError) as error:
        raise PackageError(f"invalid Cargo archive {package}: {error}") from error
    finally:
        if members.output is not None:
            members.output.close()
    return {"package": package, "archive": str(path), "sha256": before, "files": members.files}


def _disjoint(*roots: Path) -> None:
    for index, left in enumerate(roots):
        for right in roots[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise PackageError("source, staging, consumer and target roots must be disjoint")


def _output_roots(source: Path, *outputs: Path) -> None:
    """Permit the existing root target subtree, excluded from source seals."""
    _disjoint(*outputs)
    excluded = source / "target"
    for output in outputs:
        if output == source or output in source.parents:
            raise PackageError("an output cannot contain candidate source")
        if source in output.parents and output != excluded and excluded not in output.parents:
            raise PackageError("outputs inside source must be within the root target subtree")


def _run(runner, argv, root):
    result = runner(argv, root)
    if getattr(result, "returncode", None) != 0:
        raise PackageError(f"Cargo command failed: {argv[2]}: {getattr(result, 'stderr', '')}")
    return result


def stage_rust_packages(candidate_root, stage_root, inventory, source_context,
                        command_runner=None, *, target_dir) -> dict:
    """Plan/run one normally verified Cargo package batch, then seal/unpack it."""
    _rust_crate_roles(inventory)
    root, stage, target = map(_path, (candidate_root, stage_root, target_dir))
    _output_roots(root, stage, target)
    _require_source_context(source_context, root)
    _require_candidate(root)
    if stage.exists() and any(stage.iterdir()):
        raise PackageError("stage must be absent or empty")
    package_dir = target / "package"
    for package in PACKAGE_ORDER:
        archive = package_dir / f"{package}-0.0.1.crate"
        if archive.exists() or archive.is_symlink():
            raise PackageError("stale Cargo archive already exists")
    argv = package_command(candidate_root=root, target_dir=target)
    receipt = {"status": "planned", "candidate_sha": source_context["candidate_sha"],
               "version": "0.0.1", "candidate_root": str(root), "stage_root": str(stage),
               "target_dir": str(target), "command": argv, "release_complete": False}
    if command_runner is None:
        return receipt
    try:
        _run(command_runner, argv, root)
    finally:
        _require_source_context(source_context, root)
    expected = {f"{package}-0.0.1.crate" for package in PACKAGE_ORDER}
    if {path.name for path in package_dir.glob("*.crate")} != expected:
        raise PackageError("Cargo outputs must be exactly twelve archives")
    inspected = [_verify_crate(package_dir / f"{name}-0.0.1.crate", name) for name in PACKAGE_ORDER]
    stage.mkdir(parents=True, exist_ok=True)
    unpacked = stage / "unpacked"
    unpacked.mkdir()
    archives = []
    for item in inspected:
        origin = Path(item["archive"])
        archive = stage / origin.name
        with origin.open("rb") as source, archive.open("xb") as output:
            while data := source.read(MANIFEST.CHUNK):
                output.write(data)
        if _hash_file(archive) != item["sha256"]:
            raise PackageError("archive changed during staging")
        checked = _verify_crate(archive, item["package"], destination=unpacked)
        destination = unpacked / (item["package"] + "-0.0.1")
        if checked["files"] != item["files"] or _tree(destination) != item["files"]:
            raise PackageError("unpacked archive differs from inspected bytes")
        archives.append(checked | {"unpacked_root": str(destination)})
    _require_source_context(source_context, root)
    return receipt | {"status": "staged", "archives": archives}


def _verify_staged(receipt: dict) -> dict[str, Path]:
    if receipt.get("status") != "staged" or receipt.get("version") != "0.0.1":
        raise PackageError("staged Rust12 receipt required")
    stage = _path(receipt["stage_root"])
    rows = receipt.get("archives", [])
    if len(rows) != 12 or {r.get("package") for r in rows} != set(PACKAGE_ORDER):
        raise PackageError("staged receipt must bind exactly twelve archives")
    roots, expected_files = {}, {}
    for row in rows:
        name = row["package"]
        archive = stage / f"{name}-0.0.1.crate"
        unpacked = stage / "unpacked" / f"{name}-0.0.1"
        if _path(row["archive"]) != archive or _path(row["unpacked_root"]) != unpacked:
            raise PackageError("archive/unpacked path differs from staged identity")
        verified = _verify_crate(archive, name)
        if verified["sha256"] != row["sha256"] or verified["files"] != row["files"] or _tree(unpacked) != row["files"]:
            raise PackageError("staged archive or unpacked bytes changed")
        expected_files[archive.name] = row["sha256"]
        expected_files.update({f"unpacked/{name}-0.0.1/{path}": digest
                               for path, digest in row["files"].items()})
        roots[name] = unpacked
    if _tree(stage) != expected_files:
        raise PackageError("staging tree differs from the exact twelve archived packages")
    return roots


def _consumer_patches(root: Path, roots: dict[str, Path]) -> None:
    data = tomllib.loads((root / "Cargo.toml").read_text())
    patches = data.get("patch", {})
    if set(patches) != {"crates-io"} or set(patches["crates-io"]) != set(PACKAGE_ORDER) or "replace" in data:
        raise PackageError("consumer requires exactly twelve crates-io patches and no replacement")
    for name, expected in roots.items():
        patch = patches["crates-io"][name]
        if not isinstance(patch, dict) or set(patch) != {"path"} or _path(root / patch["path"]) != expected:
            raise PackageError(f"consumer patch does not bind staged archive: {name}")


def _consumer_graph(result, root: Path, roots: dict[str, Path], required: set[str], target: Path) -> dict:
    raw = getattr(result, "stdout", None)
    if not isinstance(raw, (str, bytes)):
        raise PackageError("Cargo metadata JSON stdout required")
    try:
        graph = MANIFEST.parse_json(raw.encode() if isinstance(raw, str) else raw, "cargo metadata")
        packages = graph["packages"]
        records = {p["id"]: p for p in packages}
        resolve = graph["resolve"]
        nodes = {n["id"]: n for n in resolve["nodes"]}
        root_id = resolve["root"]
        if (graph.get("version") != 1 or _path(graph["workspace_root"]) != root or
                _path(graph["target_directory"]) != target):
            raise PackageError("metadata workspace or build target differs from the clean consumer")
        if len(records) != len(packages) or len(nodes) != len(resolve["nodes"]):
            raise PackageError("duplicate Cargo metadata identities")
        if root_id not in records or _path(records[root_id]["manifest_path"]) != root / "Cargo.toml":
            raise PackageError("metadata root must be the clean consumer")
        reached, todo, internal = set(), [root_id], set()
        while todo:
            identity = todo.pop()
            if identity in reached:
                continue
            reached.add(identity)
            package = records[identity]
            node = nodes[identity]
            name = package["name"]
            manifest = _path(package["manifest_path"])
            if name in PACKAGE_ORDER:
                if name in internal or package["version"] != "0.0.1" or package.get("source") is not None or manifest != roots[name] / "Cargo.toml":
                    raise PackageError(f"metadata selects wrong staged package: {name}")
                internal.add(name)
            elif identity != root_id and package.get("source") is None:
                raise PackageError(f"consumer resolves an undeclared local package: {name}")
            if name in roots or identity == root_id:
                package_root = roots[name] if name in roots else root
                for build_target in package.get("targets", []):
                    source_path = _path(build_target["src_path"])
                    if package_root not in source_path.parents or not source_path.is_file():
                        raise PackageError(f"metadata target escapes its sealed package: {name}")
            dependencies = node.get("dependencies")
            if not isinstance(dependencies, list):
                raise PackageError("resolved dependency graph required")
            todo.extend(dependencies)
        if not required <= internal:
            raise PackageError(f"consumer graph missing required packages: {sorted(required - internal)}")
        return {"root": root_id, "packages": sorted(internal),
                "metadata_sha256": hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()}
    except (KeyError, TypeError, ValueError, MANIFEST.ManifestError) as error:
        raise PackageError(f"invalid clean consumer graph: {error}") from error


def verify_clean_consumer(consumer_root, command_runner, *, staged_receipt, target_dir,
                          required_packages=REQUIRED_CONSUMER_PACKAGES, caller_args=()) -> dict:
    """Check/run a locked consumer of the exact unpacked archives, then recheck."""
    root, target = _path(consumer_root), _path(target_dir)
    stage = _path(staged_receipt["stage_root"])
    source = _path(staged_receipt["candidate_root"])
    _output_roots(source, root, target, stage)
    if not callable(command_runner):
        raise PackageError("clean consumer command runner required")
    required = set(required_packages)
    if not set(REQUIRED_CONSUMER_PACKAGES) <= required <= set(PACKAGE_ORDER):
        raise PackageError("consumer must cover facade, engine, core and common")
    if isinstance(caller_args, (str, bytes)) or any(not isinstance(arg, str) or "\0" in arg for arg in caller_args):
        raise PackageError("caller_args must be a sequence of argument strings")
    caller_args = list(caller_args)
    roots = _verify_staged(staged_receipt)
    _consumer_patches(root, roots)
    baseline = _tree(root)
    if not {"Cargo.toml", "Cargo.lock"} <= baseline.keys():
        raise PackageError("locked clean consumer manifest and Cargo.lock required")
    common = ["--locked", "--offline", "--manifest-path", str(root / "Cargo.toml")]
    metadata = ["cargo", "+1.97.1", "metadata", "--format-version", "1", *common,
                "--config", "build.target-dir=" + json.dumps(str(target))]
    check = ["cargo", "+1.97.1", "check", *common, "--target-dir", str(target)]
    run = ["cargo", "+1.97.1", "run", *common, "--target-dir", str(target)]
    if caller_args:
        run.extend(["--", *caller_args])
    commands, graph = [metadata, check, run, metadata], None
    for command in commands:
        try:
            result = _run(command_runner, command, root)
        finally:
            if _tree(root) != baseline:
                raise PackageError("clean consumer source or lock bytes changed")
            _verify_staged(staged_receipt)
        if command[2] == "metadata":
            observed = _consumer_graph(result, root, roots, required, target)
            if graph is not None and observed != graph:
                raise PackageError("clean consumer dependency graph changed")
            graph = observed
    return {"status": "verified", "cwd": str(root), "commands": commands,
            "candidate_sha": staged_receipt["candidate_sha"], "graph": graph,
            "release_complete": False}
