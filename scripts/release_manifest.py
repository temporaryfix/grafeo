#!/usr/bin/env python3
"""Bounded local artifact-byte agreement, not provenance or publication authority.

The caller supplies an independent inventory and an immutable, exclusively owned
artifact root. Hash agreement is not proof of source origin, toolchain execution,
authorized destinations or full release coverage. Task 3 must compare the retained
catalog AND discharge Go/header/static/mobile/provenance obligations. An external
trusted manifest digest is required before any later publication-policy decision.
No builds, extraction, package hooks, network, subprocesses or filesystem writes.
"""
from __future__ import annotations

import argparse
import email.parser
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import tomllib
import unicodedata
import xml.etree.ElementTree as ET
import zlib

MAX_MEMBERS = 100_000
MAX_MEMBER_BYTES = 2 * 1024**3
MAX_EXPANDED_BYTES = 16 * 1024**3
MAX_METADATA_BYTES = 16 * 1024**2
MAX_DEPTH = 64
CHUNK = 64 * 1024
VERSION = "0.0.1"
CONTEXT_FIELDS = {"schema_version", "candidate_sha", "version", "toolchains"}
ROW_FIELDS = {"path", "format", "package", "target", "platform", "required_members"}
ARTIFACT_FIELDS = {"path", "size", "sha256", "blake3", "package", "target", "platform", "members"}
FORMATS = {
    "native-archive": {"tar.gz", "zip"}, "cli-native": {"file"}, "c-native": {"file"},
    "node-native": {"file"}, "node-package": {"npm-tgz"}, "wasm-package": {"npm-tgz"},
    "cli-npm": {"npm-tgz"}, "python-wheel": {"wheel"}, "cli-wheel": {"wheel"},
    "python-sdist": {"python-sdist"}, "cli-sdist": {"python-sdist"}, "nuget": {"nuget"},
    "rust-crate": {"crate"}, "dart-source": {"dart-source"}, "go-source": {"go-source"},
    "archive-checksums": {"file"}, "provenance": {"file"}, "sbom": {"file"},
}
# Exact source-derived rows, not output filenames or proof of existing products.
NATIVE = (
    ("x86_64-unknown-linux-gnu", "linux-x64", "libgrafeo_c.so", "tar.gz", "linux-x64", "linux", "x64", "manylinux_2_17_x86_64.manylinux2014_x86_64"),
    ("aarch64-unknown-linux-gnu", "linux-arm64", "libgrafeo_c.so", "tar.gz", "linux-arm64", "linux", "arm64", "manylinux_2_17_aarch64.manylinux2014_aarch64"),
    ("x86_64-pc-windows-msvc", "win-x64", "grafeo_c.dll", "zip", "win32-x64", "win32", "x64", "win_amd64"),
    ("x86_64-apple-darwin", "osx-x64", "libgrafeo_c.dylib", "tar.gz", "darwin-x64", "darwin", "x64", "macosx_11_0_x86_64"),
    ("aarch64-apple-darwin", "osx-arm64", "libgrafeo_c.dylib", "tar.gz", "darwin-arm64", "darwin", "arm64", "macosx_11_0_arm64"),
)
NODE = (
    ("x86_64-apple-darwin", "darwin-x64", "darwin", "x64", None),
    ("aarch64-apple-darwin", "darwin-arm64", "darwin", "arm64", None),
    ("x86_64-pc-windows-msvc", "win32-x64-msvc", "win32", "x64", None),
    ("x86_64-unknown-linux-gnu", "linux-x64-gnu", "linux", "x64", "glibc"),
    ("aarch64-unknown-linux-gnu", "linux-arm64-gnu", "linux", "arm64", "glibc"),
    ("aarch64-unknown-linux-musl", "linux-arm64-musl", "linux", "arm64", "musl"),
)
PYTHON_PLATFORMS = ("linux-x86_64", "linux-aarch64", "musllinux-x86_64", "musllinux-aarch64",
                    "windows-x64", "windows-x86", "macos-x86_64", "macos-aarch64")
RUST_PACKAGES = ("grafeo", "grafeo-common", "grafeo-core", "grafeo-adapters", "grafeo-storage",
                 "grafeo-engine", "grafeo-cli", "grafeo-python", "grafeo-node", "grafeo-c",
                 "grafeo-wasm", "grafeo-bindings-common")
GO_MODULE = "github.com/GrafeoDB/grafeo/crates/bindings/go"


class ManifestError(ValueError):
    def __init__(self, location: str, reason: str):
        self.location = location
        self.reason = reason
        super().__init__(f"{location}: {reason}")


def fail(location, reason):
    raise ManifestError(str(location), str(reason))


def retained_catalog() -> list[dict]:
    """66 known obligations, not built products or complete Task 3 coverage."""
    result = []
    def add(package, target, platform):
        result.append({"package": package, "target": target, "platform": platform})
    for row in NATIVE:
        triple, _, _, _, npm, _, _, wheel = row
        add("grafeo", "native-archive", triple)
        add("grafeo-cli", "cli-native", triple)
        add("grafeo-c", "c-native", triple)
        add("@grafeo-db/cli-" + npm, "cli-npm", npm)
        add("grafeo-cli", "cli-wheel", wheel)
    for triple, suffix, _, _, _ in NODE:
        add("@grafeo-db/js", "node-native", triple)
        add("@grafeo-db/js-" + suffix, "node-package", suffix)
    add("@grafeo-db/js", "node-package", "any")
    add("@grafeo-db/cli", "cli-npm", "any")
    for package in ("@grafeo-db/wasm", "@grafeo-db/wasm-lite"):
        add(package, "wasm-package", "wasm32-unknown-unknown")
    for platform in PYTHON_PLATFORMS:
        add("grafeo", "python-wheel", platform)
    add("grafeo", "python-sdist", "source")
    add("grafeo-cli", "cli-sdist", "source")
    add("Grafeo", "nuget", "net8.0")
    add("grafeo", "archive-checksums", "any")
    for package in RUST_PACKAGES:
        add(package, "rust-crate", "source")
    add("grafeo", "dart-source", "source")
    return sorted(result, key=identity)


def identity(row):
    return row["package"], row["target"], row["platform"]


def row_order(row):
    return (*identity(row), row["path"])


def compare_catalog(inventory: dict, additional: list[dict] | None = None) -> None:
    """Compare retained roles plus explicitly supplied extra coverage identities.

    Success says nothing about fulfillment of unspecified Task 3 obligations.
    """
    validate_inventory(inventory)
    expected = {identity(row) for row in retained_catalog()}
    for row in additional or []:
        fields(row, {"package", "target", "platform"}, "catalog.additional")
        for key in row:
            nonempty(row[key], "catalog.additional." + key)
        if identity(row) in expected:
            fail("catalog.additional", "duplicate expected role")
        expected.add(identity(row))
    actual = {identity(row) for row in inventory["artifacts"]}
    if actual != expected:
        fail("catalog", f"missing roles={sorted(expected-actual)!r}; extra roles={sorted(actual-expected)!r}")


def fields(value, expected, location):
    if not isinstance(value, dict) or set(value) != expected:
        fail(location, "exact fields required: " + ",".join(sorted(expected)))


def nonempty(value, location):
    if not isinstance(value, str) or not value or any(unicodedata.category(ch) == "Cc" for ch in value):
        fail(location, "nonempty control-free string required")


def integer(value, location):
    if type(value) is not int or value < 0:
        fail(location, "nonnegative integer required, not boolean")


def canonical_path(value, location, directory=False):
    nonempty(value, location)
    if directory and value.endswith("/"):
        value = value[:-1]
    if value.count("/") + 1 > MAX_DEPTH:
        fail(location, "path component budget exceeded before prefix materialization")
    if (any(ch in value for ch in "\\:*?[]") or value.startswith("/") or
            unicodedata.normalize("NFC", value) != value or
            any(part in ("", ".", "..") for part in value.split("/"))):
        fail(location, "noncanonical or unsafe relative POSIX path")
    return value


class Names:
    """Also check implicit directories for prefix/case collisions."""
    def __init__(self):
        self.spelling = {}
        self.explicit = set()
        self.files = set()

    def add(self, name, directory, location):
        canonical_path(name, location)
        if name in self.explicit:
            fail(location, "duplicate effective path")
        self.explicit.add(name)
        parts = name.split("/")
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            folded = prefix.casefold()
            if folded in self.spelling and self.spelling[folded] != prefix:
                fail(location, "case/normalization path collision")
            if prefix in self.files and (index < len(parts) or directory):
                fail(location, "file/directory prefix collision")
            if index == len(parts) and not directory:
                if folded in self.spelling and prefix not in self.files:
                    fail(location, "file replaces directory prefix")
                self.files.add(prefix)
            self.spelling[folded] = prefix


def json_budget(data: bytes, location):
    if len(data) > MAX_METADATA_BYTES:
        fail(location, "metadata budget exceeded before JSON parsing")
    # Conservative lexical records: each container opening, quoted string
    # (including object keys), and bare scalar token counts once. Punctuation
    # inside strings never counts. This is not a replacement JSON grammar.
    depth, records, quoted, escaped, scalar = 0, 0, False, False, False
    for ch in data:
        record = False
        if quoted:
            if escaped:
                escaped = False
            elif ch == 92:
                escaped = True
            elif ch == 34:
                quoted = False
        elif ch == 34:
            quoted = True
            scalar = False
            record = True
        elif ch in (91, 123):
            scalar = False
            record = True
            depth += 1
            if depth > MAX_DEPTH:
                fail(location, "JSON nesting budget exceeded before parsing")
        elif ch in (93, 125):
            depth -= 1
            scalar = False
        elif ch in b" \t\r\n,:":
            scalar = False
        elif not scalar:
            scalar = True
            record = True
        if record:
            records += 1
            if records > MAX_MEMBERS:
                fail(location, "JSON record budget exceeded before parsing")


def parse_json(data: bytes, location):
    json_budget(data, location)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                fail(location, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def constant(value):
        fail(location, "nonfinite JSON number: " + value)
    def floating(value):
        result = float(value)
        if not math.isfinite(result):
            constant(value)
        return result
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=constant, parse_float=floating)
    except (ValueError, UnicodeError, RecursionError) as error:
        if isinstance(error, ManifestError):
            raise
        fail(location, "invalid JSON: " + str(error))


def canonical_json(value) -> bytes:
    """Bound encoding for already-materialized API values; CLI bounds raw input."""
    try:
        pieces, total = [], 0
        for text in json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).iterencode(value):
            piece = text.encode("utf-8")
            total += len(piece)
            if total + 1 > MAX_METADATA_BYTES:
                fail("JSON", "canonical metadata exceeds budget")
            pieces.append(piece)
        data = b"".join(pieces) + b"\n"
        json_budget(data, "JSON")
        return data
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        if isinstance(error, ManifestError):
            raise
        fail("JSON", "cannot encode canonical metadata: " + str(error))


def validate_context(value, location):
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        fail(location, "schema_version must be integer 1")
    if not isinstance(value["candidate_sha"], str) or not re.fullmatch("[0-9a-f]{40}", value["candidate_sha"]):
        fail(location, "candidate_sha must be lowercase full 40-hex")
    if value["version"] != VERSION:
        fail(location, "version must be literal 0.0.1")
    tools = value["toolchains"]
    if not isinstance(tools, dict) or not tools:
        fail(location, "independent declared toolchains required")
    for name, version in tools.items():
        nonempty(name, location + ".toolchains.name")
        nonempty(version, location + ".toolchains." + name)
    if list(tools) != sorted(tools):
        fail(location, "toolchain mapping order is not canonical")


def validate_inventory(inventory):
    canonical_json(inventory)
    fields(inventory, CONTEXT_FIELDS | {"artifacts"}, "inventory")
    validate_context(inventory, "inventory")
    rows = inventory["artifacts"]
    if not isinstance(rows, list) or not rows or len(rows) > MAX_MEMBERS:
        fail("inventory.artifacts", "nonempty bounded descriptor list required")
    names, ids = Names(), set()
    allowed = {identity(row) for row in retained_catalog()} | {(GO_MODULE, "go-source", "source")}
    for index, row in enumerate(rows):
        at = f"inventory.artifacts[{index}]"
        fields(row, ROW_FIELDS, at)
        for key in ("path", "format", "package", "target", "platform"):
            nonempty(row[key], at + "." + key)
        name = canonical_path(row["path"], at)
        names.add(name, False, at)
        if identity(row) in ids:
            fail(at, "duplicate artifact identity")
        ids.add(identity(row))
        if row["target"] not in FORMATS or row["format"] not in FORMATS[row["target"]]:
            fail(at, "format does not agree with artifact role")
        if row["target"] not in ("provenance", "sbom") and identity(row) not in allowed:
            fail(at, "unknown package/role/platform identity")
        if row["target"] == "native-archive" and row["format"] != next(r[3] for r in NATIVE if r[0] == row["platform"]):
            fail(at, "native archive format/platform mismatch")
        members = row["required_members"]
        if not isinstance(members, list) or len(members) > MAX_MEMBERS:
            fail(at, "bounded required member list required")
        if (not members) != (row["format"] == "file"):
            fail(at, "only opaque files have an empty required member list")
        for member in members:
            canonical_path(member, at + ".required_members")
        if members != sorted(set(members)):
            fail(at, "required members must be sorted and distinct")
    if rows != sorted(rows, key=row_order):
        fail("inventory.artifacts", "noncanonical artifact order")


def hash_factory():
    try:
        if importlib.metadata.version("blake3") != "1.0.9":
            fail("tools.blake3", "requires exactly blake3 1.0.9")
        factory = importlib.import_module("blake3").blake3
        vectors = ((b"", "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"),
                   (b"abc", "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"))
        for data, expected in vectors:
            if factory(data).hexdigest() != expected:
                fail("tools.blake3", "known digest vector mismatch")
        return factory
    except (ImportError, importlib.metadata.PackageNotFoundError, AttributeError, TypeError, ValueError) as error:
        if isinstance(error, ManifestError):
            raise
        fail("tools.blake3", "unavailable or unusable: " + str(error))


def digest_string(value, location):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        fail(location, "expected lowercase 64-hex digest")


class Hashes:
    def __init__(self, factory):
        self.sha = hashlib.sha256()
        self.blake = factory()
        self.size = 0

    def add(self, data):
        self.sha.update(data)
        self.blake.update(data)
        self.size += len(data)

    def result(self):
        sha, blake = self.sha.hexdigest(), self.blake.hexdigest()
        digest_string(sha, "sha256")
        digest_string(blake, "blake3")
        return {"size": self.size, "sha256": sha, "blake3": blake}


def exact(stream, size, location):
    data = stream.read(size)
    if len(data) != size:
        fail(location, "truncated input")
    return data


class Budget:
    def __init__(self, location):
        self.location = location
        self.expanded = self.metadata = self.records = 0

    def charge(self, size, metadata=False):
        if size < 0 or self.expanded + size > MAX_EXPANDED_BYTES:
            fail(self.location, "expanded-byte budget exceeded")
        if metadata and (size > MAX_METADATA_BYTES or self.metadata + size > MAX_METADATA_BYTES):
            fail(self.location, "metadata budget exceeded before materialization")
        self.expanded += size
        if metadata:
            self.metadata += size

    def record(self):
        if self.records >= MAX_MEMBERS:
            fail(self.location, "metadata/member record count exceeded before materialization")
        self.records += 1


class Members:
    def __init__(self, row, factory, budget):
        self.row, self.factory, self.budget = row, factory, budget
        self.names = Names()
        self.records = []
        self.metadata = {}
        self.captured = 0

    def metadata_name(self, name):
        fmt = self.row["format"]
        if fmt == "crate":
            return name == f"{self.row['package']}-{VERSION}/Cargo.toml"
        if fmt == "npm-tgz":
            return name == "package/package.json"
        if fmt == "wheel":
            return name.endswith((".dist-info/METADATA", ".dist-info/WHEEL"))
        if fmt == "nuget":
            return name.endswith(".nuspec")
        if fmt == "python-sdist":
            return name.endswith(("/PKG-INFO", "/pyproject.toml", "/Cargo.toml"))
        return (fmt == "dart-source" and name == "pubspec.yaml") or (fmt == "go-source" and name == "go.mod")

    def begin(self, raw_name, directory, size):
        at = self.row["path"] + ":" + raw_name
        name = canonical_path(raw_name, at, directory)
        self.names.add(name, directory, at)
        if size > MAX_MEMBER_BYTES:
            fail(at, "per-member size budget exceeded")
        capture = self.metadata_name(name)
        if capture and (size > MAX_METADATA_BYTES or self.captured + size > MAX_METADATA_BYTES):
            fail(at, "intrinsic metadata budget exceeded before reading")
        if directory:
            if size:
                fail(at, "directory has payload")
            self.records.append({"path": name, "kind": "directory"})
            return name, None, None
        return name, Hashes(self.factory), bytearray() if capture else None

    def feed(self, state, data, charge=True):
        name, hashes, captured = state
        if hashes is None:
            if data:
                fail(name, "directory has data")
            return
        if hashes.size + len(data) > MAX_MEMBER_BYTES:
            fail(name, "actual expanded member exceeds budget")
        if charge:
            self.budget.charge(len(data))
        hashes.add(data)
        if captured is not None:
            if self.captured + len(data) > MAX_METADATA_BYTES:
                fail(name, "intrinsic metadata budget exceeded")
            self.captured += len(data)
            captured.extend(data)

    def finish(self, state, expected_size):
        name, hashes, captured = state
        if hashes is None:
            return
        if hashes.size != expected_size:
            fail(name, "actual size differs from archive metadata")
        self.records.append({"path": name, "kind": "file"} | hashes.result())
        if captured is not None:
            self.metadata[name] = bytes(captured)


def tar_number(data, location):
    if data[0] & 0x80:
        value = int.from_bytes(bytes([data[0] & 0x7f]) + data[1:], "big")
    else:
        raw = data.strip(b" \x00")
        if raw and not re.fullmatch(b"[0-7]+", raw):
            fail(location, "invalid TAR number")
        value = int(raw or b"0", 8)
    return value


def tar_text(data, location):
    prefix, separator, rest = data.partition(b"\x00")
    if separator and any(rest):
        fail(location, "nonzero bytes after TAR string terminator")
    try:
        return prefix.decode("utf-8")
    except UnicodeError as error:
        fail(location, "invalid TAR UTF-8: " + str(error))


def gzip_header(file, budget):
    """Consume exactly one bounded header, leaving the raw deflate stream."""
    file.seek(0)
    budget.charge(10, metadata=True)
    header = exact(file, 10, budget.location)
    if header[:3] != b"\x1f\x8b\x08" or header[3] & 0xe0:
        fail(budget.location, "unsupported gzip header")
    flags = header[3]
    crc = zlib.crc32(header)
    if flags & 4:
        budget.charge(2, metadata=True)
        length = exact(file, 2, budget.location)
        size = struct.unpack("<H", length)[0]
        budget.charge(size, metadata=True)
        crc = zlib.crc32(exact(file, size, budget.location), zlib.crc32(length, crc))
    for flag in (8, 16):
        if flags & flag:
            while True:
                budget.charge(1, metadata=True)
                byte = exact(file, 1, budget.location)
                crc = zlib.crc32(byte, crc)
                if byte == b"\x00":
                    break
    if flags & 2:
        budget.charge(2, metadata=True)
        if struct.unpack("<H", exact(file, 2, budget.location))[0] != crc & 0xffff:
            fail(budget.location, "gzip header CRC mismatch")


class SingleGzip:
    """Bounded raw inflater; never lets a parser see a second gzip header."""
    def __init__(self, file, budget):
        gzip_header(file, budget)
        self.file, self.at = file, budget.location
        self.decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        self.pending = b""
        self.crc = self.size = 0
        self.done = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        output = bytearray()
        while len(output) < size and not self.done:
            data = self.decoder.decompress(self.pending, min(CHUNK, size-len(output)))
            self.pending = self.decoder.unconsumed_tail
            output.extend(data)
            self.crc = zlib.crc32(data, self.crc)
            self.size += len(data)
            if self.decoder.eof:
                trailer = self.decoder.unused_data
                if len(trailer) < 8:
                    trailer += exact(self.file, 8-len(trailer), self.at)
                crc, length = struct.unpack("<II", trailer[:8])
                if crc != self.crc or length != self.size & 0xffffffff:
                    fail(self.at, "gzip CRC/size mismatch")
                # Any concatenated stream is rejected before parsing its header.
                if len(trailer) != 8 or self.file.read(1):
                    fail(self.at, "concatenated gzip/trailing compressed bytes forbidden")
                self.done = True
            elif not data and not self.pending:
                # zlib may retain output after consuming every input byte.
                # Drain empty-input output before fetching more compressed data.
                self.pending = self.file.read(CHUNK)
                if not self.pending:
                    fail(self.at, "truncated gzip deflate stream")
        return bytes(output)


def pax_records(data, budget):
    result, offset = {}, 0
    allowed = {"path", "size", "mtime", "atime", "ctime", "uid", "gid", "uname", "gname"}
    while offset < len(data):
        budget.record()
        space = data.find(b" ", offset)
        if space < 0 or space-offset > 10 or not data[offset:space].isdigit():
            fail(budget.location, "invalid PAX record length")
        length = int(data[offset:space])
        end = offset + length
        if end > len(data) or end <= space + 1 or data[end-1:end] != b"\n":
            fail(budget.location, "invalid PAX record boundary")
        try:
            key, value = data[space+1:end-1].decode("utf-8").split("=", 1)
        except (UnicodeError, ValueError) as error:
            fail(budget.location, "invalid PAX key/value: " + str(error))
        if key not in allowed or key in result:
            fail(budget.location, "duplicate/unsupported PAX key (links/sparse forbidden): " + key)
        result[key] = value
        offset = end
    return result


def inspect_tar(file, members):
    budget, at = members.budget, members.row["path"]
    pending, global_pax, long_name = {}, {}, None
    with SingleGzip(file, budget) as stream:
        def read(size, metadata=False):
            budget.charge(size, metadata)
            return exact(stream, size, at)
        while True:
            header = read(512, metadata=True)
            if header == b"\x00" * 512:
                if read(512, metadata=True) != b"\x00" * 512 or pending or long_name is not None:
                    fail(at, "invalid TAR end or dangling extended header")
                while True:
                    # Request at most one byte beyond the remaining budget.
                    tail = stream.read(min(CHUNK, MAX_EXPANDED_BYTES-budget.expanded+1))
                    if not tail:
                        return
                    budget.charge(len(tail))
                    if any(tail):
                        fail(at, "nonzero trailing TAR data")
            budget.record()
            checksum = tar_number(header[148:156], at)
            if checksum != sum(header[:148]) + 8*32 + sum(header[156:]):
                fail(at, "TAR header checksum mismatch")
            if header[257:263] not in (b"ustar\x00", b"ustar ", b"\x00"*6):
                fail(at, "unsupported TAR header format")
            size = tar_number(header[124:136], at)
            kind = header[156:157]
            if kind in (b"x", b"g", b"L"):
                if size > MAX_METADATA_BYTES:
                    fail(at, "TAR extended-header budget exceeded before materialization")
                data = read(size, metadata=True)
                read((-size) % 512)
                if kind == b"L":
                    if long_name is not None:
                        fail(at, "duplicate GNU long-name header")
                    long_name = tar_text(data, at)
                elif kind == b"g":
                    global_pax.update(pax_records(data, budget))
                else:
                    if pending:
                        fail(at, "duplicate per-file PAX header")
                    pending = pax_records(data, budget)
                continue
            if kind not in (b"0", b"\x00", b"5") or header[157:257].strip(b"\x00"):
                fail(at, "TAR links/special/sparse entries forbidden")
            name = tar_text(header[:100], at)
            prefix = tar_text(header[345:500], at) if header[257:263] == b"ustar\x00" else ""
            if prefix:
                name = prefix + "/" + name
            effective = global_pax | pending
            if long_name is not None:
                name = long_name
            name = effective.get("path", name)
            if "size" in effective:
                if not re.fullmatch("[0-9]{1,20}", effective["size"]):
                    fail(at, "invalid PAX size")
                size = int(effective["size"])
            pending, long_name = {}, None
            state = members.begin(name, kind == b"5", size)
            remaining = size
            while remaining:
                data = read(min(CHUNK, remaining))
                members.feed(state, data, charge=False)
                remaining -= len(data)
            members.finish(state, size)
            read((-size) % 512)


def zip_extra(data, budget):
    result, offset = {}, 0
    while offset < len(data):
        budget.record()
        if offset + 4 > len(data):
            fail(budget.location, "truncated ZIP extra header")
        tag, size = struct.unpack_from("<HH", data, offset)
        offset += 4
        if offset + size > len(data) or tag in result:
            fail(budget.location, "duplicate/truncated ZIP extra record")
        # In particular, do not allow an alternate Unicode effective pathname.
        if tag not in (1, 0x5455, 0x7875, 0x000a):
            fail(budget.location, f"unsupported ZIP extra record {tag:#x}")
        result[tag] = data[offset:offset+size]
        offset += size
    return result


def zip64_values(extra, values, sentinels, location):
    data, offset = extra.get(1, b""), 0
    result = []
    for value, sentinel in zip(values, sentinels):
        if value == sentinel:
            width = 4 if sentinel == 0xffff else 8
            if offset + width > len(data):
                fail(location, "missing ZIP64 override")
            value = int.from_bytes(data[offset:offset+width], "little")
            offset += width
        result.append(value)
    if offset != len(data):
        fail(location, "unexpected ZIP64 override bytes")
    return result


def inspect_zip(file, members):
    """Parse bounded central records, then hash actual independently inflated data."""
    budget, at = members.budget, members.row["path"]
    file.seek(0, os.SEEK_END)
    length = file.tell()
    file.seek(max(0, length-65557))
    tail = file.read(65557)  # Fixed maximum, independent of untrusted fields.
    end = tail.rfind(b"PK\x05\x06")
    if end < 0 or end + 22 > len(tail):
        fail(at, "missing ZIP end record")
    _, disk, cd_disk, disk_count, count, cd_size, cd_start, comment = struct.unpack_from("<4s4H2IH", tail, end)
    eocd = length-len(tail)+end
    if end+22+comment != len(tail) or disk or cd_disk or disk_count != count:
        fail(at, "ZIP trailing bytes/multidisk/inconsistent counts")
    budget.charge(22+comment, metadata=True)
    cd_end = eocd
    file.seek(max(0, eocd-20))
    has_zip64 = eocd >= 20 and file.read(4) == b"PK\x06\x07"
    if has_zip64 or count == 0xffff or cd_size == 0xffffffff or cd_start == 0xffffffff:
        if eocd < 20:
            fail(at, "missing ZIP64 locator")
        file.seek(eocd-20)
        locator = exact(file, 20, at)
        sig, disk64, offset64, disks = struct.unpack("<4sIQI", locator)
        if sig != b"PK\x06\x07" or disk64 or disks != 1 or offset64+56 > eocd-20:
            fail(at, "invalid ZIP64 locator")
        file.seek(offset64)
        fixed = exact(file, 56, at)
        sig, size64, _, _, disk64, cd_disk64, disk_count64, count64, size, start = struct.unpack("<4sQ2H2I4Q", fixed)
        budget.charge(size64+32, metadata=True)
        if sig != b"PK\x06\x06" or size64 < 44 or offset64+size64+12 != eocd-20:
            fail(at, "invalid ZIP64 end record")
        if disk64 or cd_disk64 or disk_count64 != count64:
            fail(at, "multidisk ZIP64 forbidden")
        for old, new, sentinel in ((count, count64, 0xffff), (cd_size, size, 0xffffffff),
                                    (cd_start, start, 0xffffffff)):
            if old != sentinel and old != new:
                fail(at, "inconsistent ZIP64 end record")
        count, cd_size, cd_start, cd_end = count64, size, start, offset64
    if count > MAX_MEMBERS or cd_size > MAX_METADATA_BYTES:
        fail(at, "ZIP central budget exceeded before materialization")
    if cd_start+cd_size != cd_end:
        fail(at, "invalid ZIP central range")
    budget.charge(cd_size, metadata=True)
    file.seek(cd_start)
    entries = []
    for _ in range(count):
        budget.record()
        if file.tell()+46 > cd_end:
            fail(at, "central entry exceeds ZIP directory")
        fixed = exact(file, 46, at)
        (sig, made, needed, flags, method, _, _, crc, compressed, size,
         name_len, extra_len, comment_len, entry_disk, _, mode, offset) = struct.unpack("<4s6H3I5H2I", fixed)
        if sig != b"PK\x01\x02" or file.tell()+name_len+extra_len+comment_len > cd_end:
            fail(at, "invalid ZIP central entry range")
        if flags & ~0x080e or method not in (0, 8) or (method == 0 and flags & 6) or needed > 45:
            fail(at, "encrypted or unsupported ZIP entry")
        raw_name = exact(file, name_len, at)
        extra = zip_extra(exact(file, extra_len, at), budget)
        exact(file, comment_len, at)
        size, compressed, offset, entry_disk = zip64_values(
            extra, (size, compressed, offset, entry_disk), (0xffffffff, 0xffffffff, 0xffffffff, 0xffff), at)
        if entry_disk:
            fail(at, "multidisk ZIP entry")
        name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
        directory = name.endswith("/")
        kind = stat.S_IFMT(mode >> 16) if made >> 8 == 3 else 0
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            fail(at, "ZIP link/special member forbidden")
        if (kind == stat.S_IFDIR and not directory) or (kind == stat.S_IFREG and directory) or (mode & 16 and not directory):
            fail(at, "inconsistent ZIP directory representation")
        state = members.begin(name, directory, size)
        entries.append((offset, compressed, size, crc, flags, method, raw_name, state))
    if file.tell() != cd_end:
        fail(at, "unaccounted ZIP central records")
    # Physical order need not be logical order. Require no overlaps, preambles or
    # unaccounted local records, as well as exact central/local agreement.
    previous = 0
    for offset, compressed, size, crc, flags, method, raw_name, state in sorted(entries, key=lambda r: r[0]):
        if offset != previous or offset+30 > cd_start:
            fail(at, "ZIP overlapping/unaccounted local range")
        file.seek(offset)
        sig, needed, local_flags, local_method, _, _, local_crc, local_compressed, local_size, nl, el = struct.unpack("<4s5H3I2H", exact(file, 30, at))
        budget.charge(30+nl+el, metadata=True)
        if file.tell()+nl+el+compressed > cd_start or sig != b"PK\x03\x04":
            fail(at, "invalid ZIP local range")
        local_name = exact(file, nl, at)
        extra = zip_extra(exact(file, el, at), budget)
        local_size, local_compressed = zip64_values(extra, (local_size, local_compressed), (0xffffffff, 0xffffffff), at)
        if (needed > 45 or local_name != raw_name or local_flags != flags or local_method != method or
                (not flags & 8 and (local_size, local_compressed, local_crc) != (size, compressed, crc))):
            fail(at, "ZIP central/local disagreement")
        decoder = zlib.decompressobj(-zlib.MAX_WBITS) if method == 8 else None
        remaining, actual_crc = compressed, 0
        while remaining:
            data = exact(file, min(CHUNK, remaining), at)
            remaining -= len(data)
            pending = data
            while True:
                output = decoder.decompress(pending, CHUNK) if decoder else pending
                pending = decoder.unconsumed_tail if decoder else b""
                members.feed(state, output)
                actual_crc = zlib.crc32(output, actual_crc)
                if decoder and decoder.eof and (decoder.unused_data or remaining):
                    fail(at, "ZIP compressed stream has trailing data")
                if not decoder or decoder.eof or (not pending and not output):
                    break
        if decoder and not decoder.eof:
            fail(at, "truncated ZIP deflate stream")
        members.finish(state, size)
        if actual_crc != crc:
            fail(at, "ZIP CRC mismatch")
        if flags & 8:
            first = exact(file, 4, at)
            descriptor_crc = struct.unpack("<I", exact(file, 4, at) if first == b"PK\x07\x08" else first)[0]
            wide = size >= 0xffffffff or compressed >= 0xffffffff or needed >= 45
            descriptor = exact(file, 16 if wide else 8, at)
            budget.charge(24 if wide else 16, metadata=True)
            csize, usize = struct.unpack("<QQ" if wide else "<II", descriptor)
            if (descriptor_crc, csize, usize) != (crc, compressed, size):
                fail(at, "ZIP data descriptor mismatch")
        previous = file.tell()
        if previous > cd_start:
            fail(at, "ZIP local entry overlaps central directory")
    if previous != cd_start:
        fail(at, "unaccounted ZIP local bytes")


def package_name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def bounded_lines(data, location, label="line"):
    """Yield bounded physical lines without allocating a splitlines list."""
    if len(data) > MAX_METADATA_BYTES:
        fail(location, "metadata budget exceeded before line parsing")
    offset = count = 0
    while offset < len(data):
        # Email's parser recognizes LF, CRLF and bare CR. Match that boundary
        # with one linear scan, not repeated searches of the remaining suffix.
        end = offset
        while end < len(data) and data[end] not in (10, 13):
            end += 1
        # The header terminator is not a header record. Its caller stops here.
        if label == "header" and end == offset:
            yield b""
            return
        count += 1
        if count > MAX_MEMBERS:
            fail(location, label + " record budget exceeded before line materialization")
        yield data[offset:end]
        offset = end+2 if data[end:end+2] == b"\r\n" else end+1


def metadata_headers(data, location, required=("Name", "Version")):
    # Physical header lines (including folds) are bounded before parsing.
    # Description bodies remain opaque strings, never recursive MIME trees.
    for line in bounded_lines(data, location, "header"):
        if not line:
            break
    message = email.parser.BytesHeaderParser().parsebytes(data)
    if message.defects:
        fail(location, "malformed package metadata headers")
    for key in required:
        if len(message.get_all(key, [])) != 1:
            fail(location, "exactly one " + key + " header required")
    return message


def python_requires(meta, target, location):
    expected = ">=3.12" if target.startswith("python-") else ">=3.9"
    if meta.get_all("Requires-Python", []) != [expected]:
        fail(location, "Requires-Python mismatch; expected " + expected)


def toml_metadata(data, location):
    toml_budget(data, location)
    try:
        return tomllib.loads(data.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError) as error:
        fail(location, "invalid TOML: " + str(error))


def toml_budget(data, location):
    """Conservative lexical guard, not a TOML parser or exact syntax-node count.

    Count strings, bare key/scalar tokens and container openings. Bound both
    bracket/brace nesting and dot-linked components outside strings/comments.
    Basic/literal and multiline strings are skipped without interpreting their
    content; tomllib still owns all grammar and value validation.
    """
    if len(data) > MAX_METADATA_BYTES:
        fail(location, "TOML metadata budget exceeded before parsing")
    index = depth = records = components = 0
    dotted = False
    separators = b" \t\r\n=,.[]{}#\"'"
    while index < len(data):
        ch = data[index]
        if ch in b" \t\r\n":
            if ch in b"\r\n":
                components, dotted = 0, False
            index += 1
            continue
        if ch == 35:
            end = data.find(b"\n", index)
            index = len(data) if end < 0 else end
            continue
        if ch == 46:
            dotted = True
            index += 1
            continue
        if ch in b"=,]}":
            if ch in b"]}":
                depth -= 1
            components, dotted = 0, False
            index += 1
            continue
        records += 1
        if records > MAX_MEMBERS:
            fail(location, "TOML record budget exceeded before parsing")
        if ch in b"[{":
            depth += 1
            if depth > MAX_DEPTH:
                fail(location, "TOML nesting budget exceeded before parsing")
            components, dotted = 0, False
            index += 1
            continue
        components = components+1 if dotted else 1
        dotted = False
        if components > MAX_DEPTH:
            fail(location, "TOML dotted component budget exceeded before parsing")
        if ch in (34, 39):
            triple = data[index:index+3] == bytes([ch]) * 3
            index += 3 if triple else 1
            while index < len(data):
                if ch == 34 and data[index] == 92:
                    index += 2  # Escaped quote/backslash/newline, grammar later.
                elif data[index] == ch:
                    end = index
                    while end < len(data) and data[end] == ch:
                        end += 1
                    if not triple:
                        index += 1
                        break
                    if end-index >= 3:
                        index = end  # Three delimiters plus up to two literals.
                        break
                    index = end
                else:
                    index += 1
        else:
            while index < len(data) and data[index] not in separators:
                index += 1


def yaml_metadata(data, location):
    """Bound events before constructing a plain scalar mapping, without aliases."""
    try:
        yaml = importlib.import_module("yaml")
        # No YAML object constructors or alias expansion. The event pass is
        # streaming and prevents recursive container materialization first.
        depth = count = 0
        for event in yaml.parse(data):
            count += 1
            if count > MAX_MEMBERS:
                fail(location, "YAML record budget exceeded")
            if isinstance(event, yaml.events.AliasEvent) or getattr(event, "anchor", None) or getattr(event, "tag", None):
                fail(location, "YAML aliases/anchors/tags forbidden")
            if isinstance(event, (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent)):
                depth += 1
                if depth > MAX_DEPTH:
                    fail(location, "YAML nesting budget exceeded")
            elif isinstance(event, (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent)):
                depth -= 1
        node = yaml.compose(data, Loader=yaml.BaseLoader)
        def convert(node):
            if isinstance(node, yaml.nodes.ScalarNode):
                return node.value
            if isinstance(node, yaml.nodes.SequenceNode):
                return [convert(child) for child in node.value]
            if not isinstance(node, yaml.nodes.MappingNode):
                fail(location, "invalid YAML node")
            result = {}
            for key, value in node.value:
                if not isinstance(key, yaml.nodes.ScalarNode) or key.value in result:
                    fail(location, "duplicate/complex YAML key")
                result[key.value] = convert(value)
            return result
        return convert(node)
    except (ImportError, ValueError, UnicodeError, RecursionError) as error:
        if isinstance(error, ManifestError):
            raise
        fail(location, "invalid/unavailable YAML parser: " + str(error))
    except Exception as error:
        # PyYAML exceptions are not ValueError; preserve a structured failure.
        fail(location, "YAML parsing failed: " + str(error))


def wheel_platform(platform, tag):
    if platform.startswith("linux-"):
        arch = platform.removeprefix("linux-")
        return bool(re.fullmatch(r"manylinux(?:_[0-9]+_[0-9]+|1|2010|2014)_" + arch, tag))
    if platform.startswith("musllinux-"):
        return tag == "musllinux_1_2_" + platform.removeprefix("musllinux-")
    if platform.startswith("windows-"):
        return tag == {"windows-x64": "win_amd64", "windows-x86": "win32"}[platform]
    arch = {"macos-x86_64": "x86_64", "macos-aarch64": "arm64"}.get(platform)
    return bool(arch and re.fullmatch(r"macosx_[0-9]+_[0-9]+_" + arch, tag))


def sdist_cargo_version(cargo, cargo_path, prefix, metadata, location):
    """Resolve only explicit archive-contained ancestor workspace inheritance."""
    version = cargo.get("version")
    if version == VERSION:
        return
    if version != {"workspace": True} or type(version.get("workspace")) is not bool:
        fail(location, "sdist referenced Cargo version mismatch")
    package_dir = cargo_path.rpartition("/")[0]
    candidates = []
    components = package_dir.split("/") if package_dir else []
    for depth in range(len(components)):
        ancestor = "/".join(components[:depth])
        path = prefix + "/" + (ancestor + "/" if ancestor else "") + "Cargo.toml"
        if path not in metadata:
            continue
        document = toml_metadata(metadata[path], location)
        workspace = document.get("workspace")
        if workspace is None:
            continue
        if not isinstance(workspace, dict) or not isinstance(workspace.get("members"), list):
            fail(location, "explicit sdist workspace members required")
        members = workspace["members"]
        for member in members:
            canonical_path(member, location)
        relative = "/".join(components[depth:])
        exclusions = workspace.get("exclude", [])
        if not isinstance(exclusions, list):
            fail(location, "invalid sdist workspace exclusions")
        for excluded in exclusions:
            canonical_path(excluded, location)
        if (members.count(relative) != 1 or relative in exclusions or
                not isinstance(workspace.get("package"), dict) or workspace["package"].get("version") != VERSION):
            fail(location, "sdist workspace membership/version mismatch")
        candidates.append(ancestor)
    if len(candidates) != 1 or "workspace" in cargo:
        fail(location, "missing/conflicting/explicitly redirected sdist workspace root")


def inspect_package(members, inventory):
    row, at = members.row, members.row["path"]
    fmt, target, package, platform = (row[k] for k in ("format", "target", "package", "platform"))
    files = {r["path"]: r for r in members.records if r["kind"] == "file"}
    def need(name):
        if name not in files or not files[name]["size"]:
            fail(at + ":" + name, "required nonempty regular payload missing")
        return name
    def data(name):
        need(name)
        if name not in members.metadata:
            fail(at + ":" + name, "intrinsic metadata not captured")
        return members.metadata[name]
    def inside(prefix):
        if any(r["path"] != prefix and not r["path"].startswith(prefix + "/") for r in members.records):
            fail(at, "member outside intrinsic package root " + prefix)
    def named(meta):
        if not isinstance(meta, dict) or meta.get("name") != package or meta.get("version") != VERSION:
            fail(at, "intrinsic package name/version mismatch")
    for name in row["required_members"]:
        need(name)
    if target == "native-archive":
        native = next(r for r in NATIVE if r[0] == platform)
        prefix = f"grafeo-v{VERSION}-{inventory['candidate_sha']}-{platform}"
        inside(prefix)
        for name in ("grafeo.exe" if native[1].startswith("win") else "grafeo", native[2], "LICENSE", "README.md", "grafeo.h"):
            need(prefix + "/" + name)
    elif fmt == "crate":
        prefix = package + "-" + VERSION
        inside(prefix)
        meta = toml_metadata(data(prefix + "/Cargo.toml"), at)
        named(meta.get("package"))
        if not any(files.get(prefix + "/src/" + leaf, {}).get("size") for leaf in ("lib.rs", "main.rs")):
            fail(at, "crate requires nonempty Rust source payload")
    elif fmt == "npm-tgz":
        inside("package")
        meta = parse_json(data("package/package.json"), at)
        named(meta)
        if target in ("cli-npm", "node-package"):
            if meta.get("license") != "Apache-2.0":
                fail(at, "native npm license differs from release declaration")
            need("package/LICENSE")
        if target == "node-package" and meta.get("engines") != {"node": ">=20.3.0"}:
            fail(at, "Node runtime must support the declared Node-API9 ABI")
        if target == "node-package" and platform != "any":
            native = next(r for r in NODE if r[1] == platform)
            _, suffix, os_name, cpu, libc = native
            binary = "grafeo." + suffix + ".node"
            if meta.get("os") != [os_name] or meta.get("cpu") != [cpu] or meta.get("main") != binary:
                fail(at, "Node platform/os/cpu/main mismatch")
            if (libc is None and "libc" in meta) or (libc is not None and meta.get("libc") != [libc]):
                fail(at, "Node libc mismatch")
            need("package/" + binary)
        elif target == "node-package":
            if meta.get("main") != "index.js" or meta.get("types") != "index.d.ts":
                fail(at, "Node launcher entrypoints mismatch")
            expected = {"@grafeo-db/js-" + r[1]: VERSION for r in NODE}
            if meta.get("optionalDependencies") != expected:
                fail(at, "Node retained platform dependency map mismatch")
            need("package/index.js")
            need("package/index.d.ts")
        elif target == "wasm-package":
            if meta.get("main") != "grafeo_wasm.js" or meta.get("types") != "grafeo_wasm.d.ts":
                fail(at, "WASM entrypoints mismatch")
            for name in ("grafeo_wasm.js", "grafeo_wasm.d.ts", "grafeo_wasm_bg.js",
                         "grafeo_wasm_bg.wasm", "grafeo_wasm_bg.wasm.d.ts"):
                need("package/" + name)
        elif platform == "any":
            if meta.get("bin") != {"grafeo": "bin/grafeo.js"}:
                fail(at, "CLI launcher bin mismatch")
            if meta.get("optionalDependencies") != {"@grafeo-db/cli-" + r[4]: VERSION for r in NATIVE}:
                fail(at, "CLI retained platform dependency map mismatch")
            need("package/bin/grafeo.js")
        else:
            native = next(r for r in NATIVE if r[4] == platform)
            if meta.get("os") != [native[5]] or meta.get("cpu") != [native[6]]:
                fail(at, "CLI npm platform/os/cpu mismatch")
            need("package/" + ("grafeo.exe" if native[5] == "win32" else "grafeo"))
    elif fmt == "wheel":
        parts = Path(at).name.removesuffix(".whl").split("-")
        if not at.endswith(".whl") or len(parts) not in (5, 6):
            fail(at, "invalid wheel filename")
        dist, version, *_, py, abi, platforms = parts
        if package_name(dist) != package_name(package) or version != VERSION:
            fail(at, "wheel filename identity/version mismatch")
        if len(parts) == 6 and not re.fullmatch(r"[0-9][A-Za-z0-9_]*", parts[2]):
            fail(at, "invalid wheel build tag")
        expected_py, expected_abi = ("cp312", "abi3") if target == "python-wheel" else ("py3", "none")
        tags = platforms.split(".")
        if py != expected_py or abi != expected_abi or len(set(tags)) != len(tags):
            fail(at, "wheel Python/ABI tags mismatch")
        if target == "cli-wheel":
            if platforms != platform:
                fail(at, "CLI wheel platform mismatch")
        elif not all(wheel_platform(platform, tag) for tag in tags):
            fail(at, "Python wheel platform mismatch")
        metas = [name for name in members.metadata if name.endswith(".dist-info/METADATA")]
        wheels = [name for name in members.metadata if name.endswith(".dist-info/WHEEL")]
        prefix = f"{dist}-{VERSION}.dist-info"
        if metas != [prefix + "/METADATA"] or wheels != [prefix + "/WHEEL"]:
            fail(at, "wheel dist-info identity/count mismatch")
        meta = metadata_headers(data(metas[0]), at)
        if package_name(meta["Name"]) != package_name(package) or meta["Version"] != VERSION:
            fail(at, "wheel METADATA identity/version mismatch")
        python_requires(meta, target, at)
        wheel = metadata_headers(data(wheels[0]), at, required=("Wheel-Version",))
        actual_tags = wheel.get_all("Tag", [])
        expected_tags = {f"{py}-{abi}-{tag}" for tag in tags}
        if wheel.defects or wheel.get_payload().strip() or len(actual_tags) != len(set(actual_tags)) or set(actual_tags) != expected_tags or wheel.get_all("Wheel-Version") != ["1.0"]:
            fail(at, "WHEEL/filename tag mismatch")
        if wheel.get_all("Root-Is-Purelib") != ["false"]:
            fail(at, "native wheel must declare non-pure payload")
        if meta.get_all("License-Expression") != ["Apache-2.0"] or meta.get_all("License-File") != ["LICENSE"]:
            fail(at, "native wheel license metadata mismatch")
        need(prefix + "/licenses/LICENSE")
        if target == "python-wheel":
            need("grafeo/__init__.py")
            suffix = ".pyd" if platform.startswith("windows-") else ".so"
            binaries = [name for name in files if re.fullmatch(r"grafeo/grafeo(?:\.[A-Za-z0-9_-]+)*" + re.escape(suffix), name)]
            if len(binaries) != 1:
                fail(at, "wheel requires one native grafeo extension payload")
            need(binaries[0])
        else:
            need("grafeo_cli/__init__.py")
            need("grafeo_cli/" + ("grafeo.exe" if platform == "win_amd64" else "grafeo"))
    elif fmt == "python-sdist":
        basename = Path(at).name
        if not basename.endswith(".tar.gz"):
            fail(at, "sdist filename must end .tar.gz")
        prefix = basename[:-7]
        if not prefix.endswith("-" + VERSION) or package_name(prefix[:-len(VERSION)-1]) != package_name(package):
            fail(at, "sdist filename identity/version mismatch")
        inside(prefix)
        meta = metadata_headers(data(prefix + "/PKG-INFO"), at)
        if package_name(meta["Name"]) != package_name(package) or meta["Version"] != VERSION:
            fail(at, "sdist PKG-INFO identity/version mismatch")
        python_requires(meta, target, at)
        pyproject = toml_metadata(data(prefix + "/pyproject.toml"), at)
        project = pyproject.get("project")
        if not isinstance(project, dict) or package_name(project.get("name", "")) != package_name(package) or project.get("version") != VERSION:
            fail(at, "sdist project identity/version mismatch")
        if project.get("requires-python") != (">=3.12" if target == "python-sdist" else ">=3.9"):
            fail(at, "sdist project requires-python mismatch")
        if target == "python-sdist":
            maturin = pyproject.get("tool", {}).get("maturin", {})
            if not isinstance(maturin, dict) or maturin.get("module-name") != "grafeo":
                fail(at, "sdist Maturin module-name mismatch")
            cargo_path = canonical_path(maturin.get("manifest-path", "Cargo.toml"), at)
            python_path = canonical_path(maturin.get("python-source"), at)
            if Path(cargo_path).name != "Cargo.toml":
                fail(at, "sdist manifest-path must name Cargo.toml")
            cargo = toml_metadata(data(prefix + "/" + cargo_path), at).get("package", {})
            if not isinstance(cargo, dict) or cargo.get("name") != "grafeo-python":
                fail(at, "sdist referenced Cargo identity/version mismatch")
            sdist_cargo_version(cargo, cargo_path, prefix, members.metadata, at)
            cargo_parent = cargo_path.rpartition("/")[0]
            need(prefix + "/" + (cargo_parent + "/" if cargo_parent else "") + "src/lib.rs")
            need(prefix + "/" + python_path + "/grafeo/__init__.py")
        else:
            need(prefix + "/grafeo_cli/__init__.py")
            backend = pyproject.get("build-system", {})
            hook = pyproject
            for key in ("tool", "hatch", "build", "hooks", "custom"):
                if not isinstance(hook, dict):
                    fail(at, "CLI sdist native build configuration must be a table")
                hook = hook.get(key, {})
            if not isinstance(backend, dict) or backend.get("build-backend") != "hatchling.build" or not isinstance(hook, dict) or hook.get("path") != "hatch_build.py":
                fail(at, "CLI sdist native build backend/hook mismatch")
            need(prefix + "/hatch_build.py")
            need(prefix + "/rust/Cargo.lock")
            cargo_path = "rust/crates/grafeo-cli/Cargo.toml"
            cargo = toml_metadata(data(prefix + "/" + cargo_path), at).get("package", {})
            if not isinstance(cargo, dict) or cargo.get("name") != "grafeo-cli":
                fail(at, "CLI sdist Cargo identity mismatch")
            sdist_cargo_version(cargo, cargo_path, prefix, members.metadata, at)
            need(prefix + "/rust/Cargo.toml")
            need(prefix + "/rust/crates/grafeo-cli/src/main.rs")
    elif fmt == "nuget":
        names = [name for name in members.metadata if name.endswith(".nuspec")]
        if len(names) != 1 or "/" in names[0]:
            fail(at, "one root NuGet nuspec required")
        raw = data(names[0])
        text = raw.decode("utf-8")
        if "\x00" in text or re.search(r"<!\s*(?:DOCTYPE|ENTITY)", text, re.IGNORECASE):
            fail(at, "XML DTD/entities forbidden")
        parser = ET.XMLPullParser(events=("start", "end"))
        depth = count = 0
        # Bounded chunks, inspect depth/count before feeding the next chunk.
        for offset in range(0, len(raw), 1024):
            parser.feed(raw[offset:offset+1024])
            for event, element in parser.read_events():
                if event == "start":
                    depth += 1
                    count += 1
                    if depth > MAX_DEPTH or count > MAX_MEMBERS:
                        fail(at, "XML nesting/record budget exceeded")
                else:
                    depth -= 1
        parser.close()
        tree = ET.fromstring(raw)
        values = lambda key: [e.text for e in tree.iter() if e.tag.rsplit("}", 1)[-1] == key]
        if values("id") != [package] or values("version") != [VERSION]:
            fail(at, "NuGet nuspec identity/version mismatch")
        need("lib/net8.0/Grafeo.dll")
        for _, rid, lib, *_ in NATIVE:
            need("runtimes/" + rid + "/native/" + lib)
    elif fmt == "dart-source":
        named(yaml_metadata(data("pubspec.yaml"), at))
        need("lib/grafeo.dart")
    elif fmt == "go-source":
        module = None
        for line in bounded_lines(data("go.mod"), at):
            match = re.fullmatch(r"[ \t]*module[ \t]+([^ \t\r\n]+)[ \t]*(?://[^\r\n]*)?", line.decode("utf-8"))
            if match:
                if module is not None:
                    fail(at, "duplicate Go module declaration")
                module = match[1]
        if module != GO_MODULE:
            fail(at, "Go module identity mismatch")
        need("grafeo.go")
        need("grafeo.h")
        # go.mod has no module-version field; context is declaration, not proof.


def stat_identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def walk_root(root, expected):
    if not isinstance(root, Path):
        fail("root", "pathlib.Path required")
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode):
        fail(root, "artifact root must be a real directory, not a link")
    names, found, pending, count = Names(), {}, [(root, "")], 0
    expected_directories = set()
    for path in expected:
        canonical_path(path, "root.expected")
        parts = path.split("/")
        expected_directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
    while pending:
        parent, prefix = pending.pop()
        with os.scandir(parent) as entries:
            # Do not materialize an unbounded directory iterator before counting.
            for entry in entries:
                count += 1
                if count > MAX_MEMBERS:
                    fail(root, "artifact-tree entry budget exceeded")
                name = canonical_path(prefix + entry.name, "root")
                info = entry.stat(follow_symlinks=False)
                directory = stat.S_ISDIR(info.st_mode)
                names.add(name, directory, name)
                if directory:
                    if name not in expected_directories:
                        fail(name, "undeclared artifact directory")
                    pending.append((Path(entry.path), name + "/"))
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    found[name] = info
                else:
                    fail(name, "only non-linked regular files/directories admitted")
    if set(found) != expected:
        fail("root", f"missing artifacts={sorted(expected-set(found))!r}; undeclared artifacts={sorted(set(found)-expected)!r}")
    return found


def hash_file(file, factory):
    file.seek(0)
    hashes = Hashes(factory)
    while data := file.read(CHUNK):
        hashes.add(data)
    return hashes.result()


def inspect_artifact(root, row, inventory, initial, factory):
    path, at = root / row["path"], row["path"]
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as file:
        opened = os.fstat(file.fileno())
        if not stat.S_ISREG(opened.st_mode) or stat_identity(opened) != stat_identity(initial):
            fail(at, "artifact identity changed before inspection")
        if row["format"] == "file" and (not opened.st_size or opened.st_size > MAX_MEMBER_BYTES):
            fail(at, "opaque artifact must be nonempty and within member budget")
        raw = hash_file(file, factory)
        budget = Budget(at)
        members = Members(row, factory, budget)
        checksum = None
        if row["format"] == "file":
            if not raw["size"] or raw["size"] > MAX_MEMBER_BYTES:
                fail(at, "opaque artifact must be nonempty and within member budget")
            if row["target"] == "archive-checksums":
                if raw["size"] > MAX_METADATA_BYTES:
                    fail(at, "checksum document exceeds metadata budget")
                file.seek(0)
                checksum = exact(file, raw["size"], at)
        else:
            if row["format"] in ("zip", "wheel", "nuget"):
                inspect_zip(file, members)
            else:
                inspect_tar(file, members)
            members.records.sort(key=lambda value: value["path"])
            inspect_package(members, inventory)
        # One descriptor, both complete passes. Stat alone does not detect a
        # rewritten stream with restored timestamps; byte agreement adds a check,
        # not an adversarial filesystem lease. Immutable root remains required.
        if hash_file(file, factory) != raw:
            fail(at, "artifact bytes changed during inspection")
        if stat_identity(os.fstat(file.fileno())) != stat_identity(opened) or stat_identity(path.lstat()) != stat_identity(opened):
            fail(at, "artifact mutation/replacement detected")
        result = {key: row[key] for key in ("path", "package", "target", "platform")}
        return result | raw | {"members": members.records}, checksum


def check_archive_checksums(data, artifacts, location):
    native = [r for r in artifacts if r["target"] == "native-archive"]
    if {r["platform"] for r in native} != {r[0] for r in NATIVE}:
        fail(location, "checksums require exactly all five retained native archives")
    expected = {Path(r["path"]).name: r["sha256"] for r in native}
    if len(expected) != 5:
        fail(location, "native archive basenames are ambiguous")
    actual = {}
    for raw_line in bounded_lines(data, location):
        line = raw_line.decode("utf-8")
        match = re.fullmatch(r"([0-9a-f]{64}) [ *]([^\r\n]+)", line)
        if not match or match[2] in actual:
            fail(location, "malformed/duplicate native archive checksum")
        canonical_path(match[2], location)
        actual[match[2]] = match[1]
    if actual != expected:
        fail(location, "native archive checksum byte/set mismatch")


def build_manifest(root: Path, inventory: dict) -> dict:
    """Inspect the exact independent inventory. No provenance/completeness claim."""
    try:
        validate_inventory(inventory)
        factory = hash_factory()
        initial = walk_root(root, {row["path"] for row in inventory["artifacts"]})
        artifacts, checksums = [], []
        for row in inventory["artifacts"]:
            try:
                artifact, checksum = inspect_artifact(root, row, inventory, initial[row["path"]], factory)
            except ManifestError:
                raise
            except Exception as error:
                fail(row["path"], f"inspection failed ({type(error).__name__}): {error}")
            artifacts.append(artifact)
            if checksum is not None:
                checksums.append((row["path"], checksum))
        for location, checksum in checksums:
            check_archive_checksums(checksum, artifacts, location)
        # Detect additions/removals/identity changes after the first tree walk.
        final = walk_root(root, set(initial))
        if any(stat_identity(final[name]) != stat_identity(info) for name, info in initial.items()):
            fail("root", "artifact tree changed during inspection")
        result = {key: inventory[key] for key in CONTEXT_FIELDS} | {"artifacts": artifacts}
        result["toolchains"] = dict(inventory["toolchains"])
        canonical_json(result)
        return result
    except ManifestError:
        raise
    except Exception as error:
        fail("build_manifest", f"inspection failed ({type(error).__name__}): {error}")


def validate_manifest(manifest):
    canonical_json(manifest)
    fields(manifest, CONTEXT_FIELDS | {"artifacts"}, "manifest")
    validate_context(manifest, "manifest")
    rows = manifest["artifacts"]
    if not isinstance(rows, list) or not rows or len(rows) > MAX_MEMBERS:
        fail("manifest.artifacts", "nonempty bounded artifact list required")
    names, identities = Names(), set()
    for index, row in enumerate(rows):
        at = f"manifest.artifacts[{index}]"
        fields(row, ARTIFACT_FIELDS, at)
        for key in ("path", "package", "target", "platform"):
            nonempty(row[key], at + "." + key)
        names.add(canonical_path(row["path"], at), False, at)
        if identity(row) in identities:
            fail(at, "duplicate artifact identity")
        identities.add(identity(row))
        integer(row["size"], at)
        for key in ("sha256", "blake3"):
            digest_string(row[key], at + "." + key)
        records = row["members"]
        if not isinstance(records, list) or len(records) > MAX_MEMBERS:
            fail(at, "bounded member list required")
        member_names = Names()
        for record in records:
            if not isinstance(record, dict) or record.get("kind") not in ("file", "directory"):
                fail(at, "file/directory member record required")
            directory = record["kind"] == "directory"
            fields(record, {"path", "kind"} if directory else {"path", "kind", "size", "sha256", "blake3"}, at)
            member_names.add(canonical_path(record["path"], at), directory, at)
            if not directory:
                integer(record["size"], at)
                for key in ("sha256", "blake3"):
                    digest_string(record[key], at)
        if records != sorted(records, key=lambda record: record["path"]):
            fail(at, "noncanonical member order")
    if rows != sorted(rows, key=row_order):
        fail("manifest", "noncanonical artifact order")


def verify_manifest(root: Path, manifest: dict, inventory: dict) -> None:
    """Recompute actual bytes against an independent expected context/inventory."""
    validate_inventory(inventory)
    validate_manifest(manifest)
    for key in CONTEXT_FIELDS:
        if inventory[key] != manifest[key]:
            fail("manifest." + key, "independent expected context mismatch")
    actual = build_manifest(root, inventory)
    if actual != manifest:
        fail("manifest.artifacts", "actual bytes/member records differ from manifest")


def read_json_file(path, root):
    if path.resolve().is_relative_to(root.resolve()):
        fail(path, "inventory/manifest must be outside the artifact root")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_METADATA_BYTES:
        fail(path, "regular JSON input within metadata budget required before parsing")
    with path.open("rb") as file:
        data = file.read(MAX_METADATA_BYTES+1)
    return parse_json(data, str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--inventory", type=Path, required=True)
        if name == "verify":
            command.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inventory = read_json_file(args.inventory, args.root)
        if args.command == "build":
            result = build_manifest(args.root, inventory)
        else:
            manifest = read_json_file(args.manifest, args.root)
            verify_manifest(args.root, manifest, inventory)
            result = {"byte_agreement": True}
        output = canonical_json(result)
    except Exception as error:
        if not isinstance(error, ManifestError):
            error = ManifestError(args.command, f"{type(error).__name__}: {error}")
        # Keep the failure channel bounded even when the failure is the metadata
        # limit itself. JSON only appears on stdout after complete success.
        sys.stderr.write(json.dumps({"location": error.location, "reason": error.reason},
                                    sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
        return 1
    sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
