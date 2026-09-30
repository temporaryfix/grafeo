#!/usr/bin/env python3
"""Prune Maturin's copied workspace lock to its packaged dependency graph.

Run after the pinned Maturin sdist producer. Cargo resolves offline using the
existing lock; reject every new package, version, source, checksum or dependency.
Only the lockfile may change. Replace the archive atomically after validation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import tomllib


def validate_pruning(before: bytes, after: bytes) -> None:
    old, new = (tomllib.loads(data.decode()) for data in (before, after))
    if {k: v for k, v in old.items() if k != "package"} != {k: v for k, v in new.items() if k != "package"}:
        raise ValueError("lockfile format/metadata changed")
    def packages(lock):
        result = {}
        for package in lock["package"]:
            key = (package["name"], package["version"], package.get("source"))
            if key in result:
                raise ValueError("duplicate locked package")
            result[key] = package
        if not result:
            raise ValueError("empty locked graph")
        return result
    original = packages(old)
    for key, package in packages(new).items():
        previous = original.get(key)
        if previous is None or {k: v for k, v in package.items() if k != "dependencies"} != {k: v for k, v in previous.items() if k != "dependencies"}:
            raise ValueError("locked package identity/checksum changed")
        if not set(package.get("dependencies", [])).issubset(previous.get("dependencies", [])):
            raise ValueError("new locked dependency")


def seal_sdist(archive: Path) -> None:
    if not archive.is_file() or archive.is_symlink() or archive.stat().st_size > 32 * 1024**2:
        raise ValueError("expected a regular bounded source archive")
    if not subprocess.check_output(["cargo", "-V"], text=True).startswith("cargo 1.97.1 "):
        raise ValueError("Cargo 1.97.1 required")
    with tempfile.TemporaryDirectory(prefix="grafeo-python-sdist-", dir=archive.parent) as temporary:
        stage = Path(temporary)
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            if len(members) > 10_000 or sum(m.size for m in members) > 128 * 1024**2:
                raise ValueError("source archive budget exceeded")
            names = set()
            for member in members:
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or member.size > 32 * 1024**2 or not parts or parts[0] != "grafeo-0.0.1" or ".." in parts or member.name in names:
                    raise ValueError("unexpected source member")
                names.add(member.name)
            source.extractall(stage, members=members, filter="data")
        root = stage / "grafeo-0.0.1"
        lock = root / "Cargo.lock"
        before = lock.read_bytes()
        hashes = {p: hashlib.sha256(p.read_bytes()).digest() for p in root.rglob("*") if p.is_file() and p != lock}
        env = dict(os.environ, CARGO_NET_OFFLINE="true", CARGO_TARGET_DIR=str(stage / "target"))
        command = ["cargo", "metadata", "--offline", "--format-version", "1", "--manifest-path", "crates/bindings/python/Cargo.toml", "--features", "pyo3/extension-module,pyo3/abi3-py312,full"]
        subprocess.run(command, cwd=root, env=env, stdout=subprocess.DEVNULL, check=True)
        validate_pruning(before, lock.read_bytes())
        subprocess.run(command + ["--locked"], cwd=root, env=env, stdout=subprocess.DEVNULL, check=True)
        current = {p: hashlib.sha256(p.read_bytes()).digest() for p in root.rglob("*") if p.is_file() and p != lock}
        if hashes != current:
            raise ValueError("Cargo changed source files outside the lockfile")
        output = stage / "sealed.tar.gz"
        with output.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed, tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target:
            for member in members:
                file = stage / member.name
                info = tarfile.TarInfo(member.name)
                info.size = file.stat().st_size
                info.mode = member.mode & 0o777
                with file.open("rb") as payload:
                    target.addfile(info, payload)
        os.replace(output, archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    seal_sdist(Path(os.path.abspath(parser.parse_args().archive)))


if __name__ == "__main__":
    main()
