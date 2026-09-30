#!/usr/bin/env python3
"""Verify that every shipping package surface has one release version."""

from __future__ import annotations

import json
import re
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"(?<![A-Za-z0-9_])\d+\.\d+\.\d+(?![A-Za-z0-9_])")


class VersionAudit:
    """Collect every mismatch so one run exposes the complete repair set."""

    def __init__(self, expected: str) -> None:
        self.expected = expected
        self.mismatches: list[str] = []

    def mismatch(self, path: Path, field: str, actual: object) -> None:
        relative = path.relative_to(ROOT)
        self.mismatches.append(
            f"{relative}: {field}: expected {self.expected!r}, found {actual!r}"
        )

    def read_text(self, relative: str) -> tuple[Path, str | None]:
        path = ROOT / relative
        try:
            return path, path.read_text(encoding="utf-8")
        except OSError as error:
            self.mismatches.append(f"{relative}: cannot read file: {error}")
            return path, None

    def read_toml(self, relative: str) -> tuple[Path, dict[str, Any] | None]:
        path, text = self.read_text(relative)
        if text is None:
            return path, None
        try:
            return path, tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            self.mismatches.append(f"{relative}: invalid TOML: {error}")
            return path, None

    def check_value(self, path: Path, field: str, actual: object) -> None:
        if actual != self.expected:
            self.mismatch(path, field, actual)

    def check_cargo_manifest(self) -> None:
        path, document = self.read_toml("Cargo.toml")
        if document is None:
            return
        workspace = document.get("workspace", {})
        package = workspace.get("package", {})
        self.check_value(path, "workspace.package.version", package.get("version"))

        dependencies = workspace.get("dependencies", {})
        path_dependencies = 0
        for name, dependency in sorted(dependencies.items()):
            if isinstance(dependency, dict) and "path" in dependency:
                path_dependencies += 1
                self.check_value(
                    path,
                    f"workspace.dependencies.{name}.version",
                    dependency.get("version"),
                )
        if path_dependencies == 0:
            self.mismatches.append("Cargo.toml: no internal path dependencies found")

    def check_cargo_lock(self) -> None:
        path, document = self.read_toml("Cargo.lock")
        if document is None:
            return
        packages = [
            package
            for package in document.get("package", [])
            if isinstance(package, dict)
            and str(package.get("name", "")).startswith("grafeo")
        ]
        if not packages:
            self.mismatches.append("Cargo.lock: no grafeo workspace packages found")
            return
        for package in packages:
            name = package.get("name", "<unnamed>")
            self.check_value(path, f"package[{name}].version", package.get("version"))

    def check_json_package(self, relative: str) -> None:
        path, text = self.read_text(relative)
        if text is None:
            return
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            self.mismatches.append(f"{relative}: invalid JSON: {error}")
            return
        self.check_value(path, "version", document.get("version"))
        for section, dependencies in sorted(document.items()):
            if not section.lower().endswith("dependencies") or not isinstance(dependencies, dict):
                continue
            for name, version in sorted(dependencies.items()):
                if name.startswith("@grafeo-db/"):
                    self.check_value(path, f"{section}.{name}", version)

    def check_python_projects(self) -> None:
        binding_path, binding = self.read_toml("crates/bindings/python/pyproject.toml")
        if binding is not None:
            project = binding.get("project", {})
            self.check_value(binding_path, "project.version", project.get("version"))
            cli_dependencies = project.get("optional-dependencies", {}).get("cli", [])
            expected_pin = f"grafeo-cli=={self.expected}"
            actual_pins = [
                dependency
                for dependency in cli_dependencies
                if isinstance(dependency, str) and dependency.startswith("grafeo-cli")
            ]
            if actual_pins != [expected_pin]:
                self.mismatch(binding_path, "project.optional-dependencies.cli", actual_pins)

        cli_path, cli = self.read_toml("packages/grafeo-cli-python/pyproject.toml")
        if cli is not None:
            self.check_value(cli_path, "project.version", cli.get("project", {}).get("version"))

        init_path, init_text = self.read_text(
            "packages/grafeo-cli-python/grafeo_cli/__init__.py"
        )
        if init_text is not None:
            match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.MULTILINE)
            self.check_value(init_path, "__version__", match.group(1) if match else None)

    def check_dart(self) -> None:
        path, text = self.read_text("crates/bindings/dart/pubspec.yaml")
        if text is not None:
            match = re.search(r"^version:\s*([^\s#]+)", text, re.MULTILINE)
            self.check_value(path, "version", match.group(1) if match else None)

        # The product front door can describe an unpublished source candidate
        # without advertising a registry install. Audit every example if one
        # is present; the actual Dart package guide still requires its example.
        for relative, required in (
            ("README.md", False),
            ("crates/bindings/dart/README.md", True),
        ):
            readme_path, readme = self.read_text(relative)
            if readme is None:
                continue
            declarations = re.findall(r"^[ \t]*grafeo[ \t]*:[ \t]*([^\r\n]*)", readme, re.MULTILINE)
            if not declarations and required:
                self.mismatch(readme_path, "Dart dependency example", None)
            for index, declaration in enumerate(declarations, start=1):
                # In a plain YAML scalar an attached '#' is part of the value;
                # only a start/whitespace-delimited '#' begins a comment.
                value = re.split(r"(?<!\S)#", declaration, maxsplit=1)[0].strip()
                version = value.removeprefix("^")
                self.check_value(readme_path, f"Dart dependency example {index}", version)

    def check_csharp(self) -> None:
        relative = "crates/bindings/csharp/src/Grafeo/Grafeo.csproj"
        path, text = self.read_text(relative)
        if text is None:
            return
        try:
            root = ET.fromstring(text)
        except ET.ParseError as error:
            self.mismatches.append(f"{relative}: invalid XML: {error}")
            return
        versions = [element.text for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "Version"]
        if not versions:
            self.mismatch(path, "Version", None)
        for index, version in enumerate(versions, start=1):
            self.check_value(path, f"Version {index}", version)

    def check_generated_node_loader(self) -> None:
        path, text = self.read_text("crates/bindings/node/index.js")
        if text is None:
            return
        versions = SEMVER.findall(text)
        if not versions:
            self.mismatch(path, "generated version literals", None)
        for index, version in enumerate(versions, start=1):
            self.check_value(path, f"generated version literal {index}", version)

    def run(self) -> int:
        self.check_cargo_manifest()
        self.check_cargo_lock()

        json_packages = [
            "crates/bindings/node/package.json",
            "crates/bindings/wasm/package.json",
            "crates/bindings/wasm/package-lite.json",
            "packages/grafeo-cli-npm/package.json",
        ]
        json_packages.extend(
            str(path.relative_to(ROOT))
            for path in sorted((ROOT / "crates/bindings/node/npm").glob("*/package.json"))
        )
        for relative in json_packages:
            self.check_json_package(relative)

        self.check_python_projects()
        self.check_dart()
        self.check_csharp()
        self.check_generated_node_loader()

        if self.mismatches:
            print("release version audit failed:")
            for mismatch in self.mismatches:
                print(f"- {mismatch}")
            return 1
        print(f"release version audit passed: all shipping surfaces are {self.expected}")
        return 0


def main() -> int:
    if len(sys.argv) != 2 or SEMVER.fullmatch(sys.argv[1]) is None:
        print(f"usage: {Path(sys.argv[0]).name} MAJOR.MINOR.PATCH", file=sys.stderr)
        return 2
    return VersionAudit(sys.argv[1]).run()


if __name__ == "__main__":
    raise SystemExit(main())
