#!/usr/bin/env python3
"""Audit production library restriction lints; emit one JSON evidence record.

Defaults: Rust 1.97.1, common/core/storage/engine, all features. For development:
  python3 scripts/check-production-panics.py -p grafeo-core \
      --no-default-features --features lpg,text-index

Cargo JSON protocol: https://doc.rust-lang.org/cargo/reference/external-tools.html
Human Cargo/rustc diagnostics go to stderr; stdout is exclusively the result.
This runner never installs a toolchain or repairs findings.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tomllib


TOOLCHAIN = "1.97.1"
PACKAGES = ["grafeo-common", "grafeo-core", "grafeo-storage", "grafeo-engine"]
LINTS = ["clippy::unwrap_used", "clippy::expect_used", "clippy::panic", "clippy::unreachable"]
LIMITATIONS = ("Restriction-lint audit only: this does not prove arithmetic, indexing, "
               "allocation or semantic totality, nor complete Task 0 or release qualification.")


class Arguments(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number: {value}")
    return number


def source_identity(repo):
    def git(*args):
        process = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True)
        return process.stdout

    if Path(git("rev-parse", "--show-toplevel").strip()).resolve() != repo:
        raise ValueError("script directory is not the repository root")
    commit = git("rev-parse", "--verify", "HEAD").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {"repo": str(repo), "commit": commit, "dirty": bool(status), "status": status}


def expected_libraries(repo, packages):
    """Only the four known production packages and their default lib layouts."""
    workspace = tomllib.loads((repo / "Cargo.toml").read_text(encoding="utf-8"))
    expected = {}
    for package in packages:
        directory = repo / "crates" / package
        manifest = directory / "Cargo.toml"
        document = tomllib.loads(manifest.read_text(encoding="utf-8"))
        metadata = document["package"]
        library = document.get("lib", {})
        name = package.replace("-", "_")
        source = directory / "src/lib.rs"
        if (metadata["name"] != package or library.get("name", name) != name
                or library.get("path", "src/lib.rs") != "src/lib.rs"
                or library.get("crate-type", ["lib"]) != ["lib"] or not source.is_file()):
            raise ValueError(f"unsupported production library layout: {package}")
        version = metadata["version"]
        if version == {"workspace": True}:
            version = workspace["workspace"]["package"]["version"]
        if not isinstance(version, str) or not version:
            raise ValueError(f"missing package version: {package}")
        prefix = f"path+{directory.as_uri()}#"
        expected[package] = {"package_ids": [prefix + version, prefix + package + "@" + version],
                             "manifest_path": str(manifest), "name": name, "src_path": str(source)}
    return expected


def absolute_path(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("artifact paths must be absolute")
    return str(Path(value).resolve())


def record_library_coverage(message, result):
    manifest = absolute_path(message.get("manifest_path"))
    if type(message.get("fresh")) is not bool:
        raise ValueError("artifact fresh must be a boolean")
    target = message["target"]
    for package, expected in result["expected_libraries"].items():
        same_package = message["package_id"] in expected["package_ids"]
        if same_package != (manifest == expected["manifest_path"]):
            raise ValueError(f"artifact package/manifest identity mismatch: {package}")
        # Dependencies and build scripts can appear, but do not prove that the
        # requested production library target was checked.
        if not same_package or target["kind"] != ["lib"]:
            continue
        if (target.get("crate_types") != ["lib"] or target.get("name") != expected["name"]
                or absolute_path(target.get("src_path")) != expected["src_path"]):
            raise ValueError(f"artifact library target identity mismatch: {package}")
        if package not in result["library_coverage"]["observed"]:
            result["library_coverage"]["observed"].append(package)


def validate_target(message):
    if not isinstance(message.get("package_id"), str) or not message["package_id"]:
        raise ValueError("missing package identity")
    target = message.get("target")
    if not isinstance(target, dict) or not isinstance(target.get("kind"), list) or not target["kind"]:
        raise ValueError("missing target kind")
    if any(kind not in ("lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro", "custom-build")
           for kind in target["kind"]):
        raise ValueError("unexpected non-production target")


def inspect_diagnostic(message, counts):
    if not isinstance(message, dict) or not isinstance(message.get("message"), str):
        raise ValueError("malformed diagnostic message")
    level = message.get("level")
    if level not in ("error", "warning", "note", "help", "failure-note", "error: internal compiler error"):
        raise ValueError("unknown diagnostic level")
    code = message.get("code")
    if code is not None:
        if not isinstance(code, dict) or not isinstance(code.get("code"), str):
            raise ValueError("malformed diagnostic code")
        code = code["code"]
    spans = message.get("spans")
    if not isinstance(spans, list):
        raise ValueError("malformed diagnostic spans")
    for span in spans:
        if not isinstance(span, dict) or not isinstance(span.get("file_name"), str):
            raise ValueError("malformed diagnostic location")
        for field in ("line_start", "line_end", "column_start", "column_end"):
            if type(span.get(field)) is not int or span[field] < 1:
                raise ValueError(f"malformed diagnostic {field}")
    children = message.get("children")
    if not isinstance(children, list):
        raise ValueError("malformed diagnostic children")
    rendered = message.get("rendered")
    if rendered is not None and not isinstance(rendered, str):
        raise ValueError("malformed rendered diagnostic")
    counts["errors"] += int(level.startswith("error"))
    counts["warnings"] += int(level == "warning")
    if code in counts["restrictions"]:
        counts["restrictions"][code] += 1
        counts["restriction_total"] += 1
    for child in children:
        inspect_diagnostic(child, counts)


def inspect_message(message, result):
    if not isinstance(message, dict):
        raise ValueError("Cargo message must be an object")
    if result["build_finished"]["seen"]:
        raise ValueError("Cargo message after build-finished (or duplicate completion)")
    reason = message.get("reason")
    if reason == "build-finished":
        if type(message.get("success")) is not bool:
            raise ValueError("build-finished.success must be a boolean")
        result["build_finished"] = {"seen": True, "success": message["success"]}
    elif reason == "compiler-message":
        validate_target(message)
        result["diagnostics"].append(message)
        inspect_diagnostic(message.get("message"), result["counts"])
        diagnostic = message["message"]
        print(diagnostic.get("rendered") or diagnostic["message"], file=sys.stderr)
    elif reason == "compiler-artifact":
        validate_target(message)
        profile = message.get("profile")
        features = message.get("features")
        if not isinstance(profile, dict) or profile.get("test") is not False:
            raise ValueError("missing or non-production artifact profile")
        if not isinstance(features, list) or any(not isinstance(feature, str) for feature in features):
            raise ValueError("malformed artifact features")
        record_library_coverage(message, result)
        result["artifacts"].append({"package_id": message["package_id"], "target": message["target"],
                                    "features": features, "profile": profile,
                                    "manifest_path": message["manifest_path"], "fresh": message["fresh"]})
    elif reason == "build-script-executed":
        if not isinstance(message.get("package_id"), str):
            raise ValueError("missing build-script package identity")
    else:
        raise ValueError(f"unknown Cargo message reason: {reason!r}")


def run_cargo(result, repo):
    # Missing pinned toolchains must fail, not trigger an implicit rustup install.
    env = {**os.environ, "RUSTUP_AUTO_INSTALL": "0"}
    with subprocess.Popen(result["cargo_argv"], cwd=repo, stdout=subprocess.PIPE, env=env) as process:
        for line_number, line in enumerate(process.stdout, 1):
            try:
                message = json.loads(line.decode("utf-8"), object_pairs_hook=unique_object,
                                     parse_constant=reject_constant, parse_float=finite_float)
                inspect_message(message, result)
            except (ValueError, UnicodeError, RecursionError) as error:
                result["protocol_errors"].append(f"Cargo stdout line {line_number}: {error}")
        result["cargo_exit_status"] = process.wait()
    if not result["build_finished"]["seen"]:
        result["protocol_errors"].append("missing build-finished message")
    coverage = result["library_coverage"]
    coverage["observed"] = [package for package in coverage["required"] if package in coverage["observed"]]
    coverage["missing"] = [package for package in coverage["required"] if package not in coverage["observed"]]
    if result["cargo_exit_status"] == 0 and result["build_finished"]["success"] is True and coverage["missing"]:
        result["protocol_errors"].append("missing requested production library artifacts: " + ", ".join(coverage["missing"]))


def main():
    result = {"schema_version": 1, "source": None, "toolchain": TOOLCHAIN, "cargo_argv": [],
              "packages": [], "features": {}, "cargo_exit_status": None,
              "build_finished": {"seen": False, "success": None}, "diagnostics": [], "artifacts": [],
              "expected_libraries": {}, "library_coverage": {"required": [], "observed": [], "missing": []},
              "counts": {"restrictions": dict.fromkeys(LINTS, 0), "restriction_total": 0,
                         "errors": 0, "warnings": 0},
              "protocol_errors": [], "tool_errors": [], "audit_passed": False, "limitations": LIMITATIONS}
    try:
        parser = Arguments(description=__doc__)
        parser.add_argument("-p", "--package", action="append")
        parser.add_argument("--features", action="append", default=[])
        parser.add_argument("--no-default-features", action="store_true")
        parser.add_argument("--all-features", action="store_true")
        args = parser.parse_args()
        if args.all_features and (args.features or args.no_default_features):
            raise ValueError("--all-features conflicts with focused feature selection")
        packages = args.package or PACKAGES
        if any(package not in PACKAGES for package in packages):
            raise ValueError("supported production packages: " + ", ".join(PACKAGES))
        if any(not features or features.startswith("-") for features in args.features):
            raise ValueError("features must be nonempty and not options")
        all_features = not args.features and not args.no_default_features
        result["packages"] = packages
        result["library_coverage"] = {"required": packages, "observed": [], "missing": list(packages)}
        result["features"] = {"all": all_features, "no_default": args.no_default_features,
                              "requested": args.features}
        argv = ["cargo", f"+{TOOLCHAIN}", "clippy", "--locked", "--lib", "--no-deps", "--message-format=json"]
        for package in packages:
            argv += ["-p", package]
        if all_features:
            argv.append("--all-features")
        if args.no_default_features:
            argv.append("--no-default-features")
        for features in args.features:
            argv += ["--features", features]
        argv += ["--", "-D", "warnings"]
        for lint in LINTS:
            argv += ["-D", lint]
        result["cargo_argv"] = argv
        repo = Path(__file__).resolve().parents[1]
        result["source"] = source_identity(repo)
        result["expected_libraries"] = expected_libraries(repo, packages)
        run_cargo(result, repo)
        result["audit_passed"] = (result["cargo_exit_status"] == 0
            and result["build_finished"] == {"seen": True, "success": True}
            and not result["library_coverage"]["missing"]
            and not result["protocol_errors"] and not result["counts"]["restriction_total"]
            and not result["counts"]["errors"] and not result["counts"]["warnings"])
    except Exception as error:
        # An unforeseen runner/tool error must still produce a failing envelope.
        result["tool_errors"].append(f"{type(error).__name__}: {error}")
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
