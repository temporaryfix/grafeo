#!/usr/bin/env python3
"""Validate benchmark policy and compare complete critcmp JSON exports.

Usage:
    python scripts/parse-thresholds.py bench-thresholds.toml
    python scripts/parse-thresholds.py CONFIG --compare BASE CANDIDATE \
        BASE_JSON CANDIDATE_JSON CRITERION_DIR

Output (TSV):
    benchmark_pattern\tthreshold_pct\tfail_ci
    epoch_arena_*\t8\ttrue
    query_*\t12\ttrue
    ...
    __default__\t15\tfalse
"""

from __future__ import annotations

import fnmatch
import json
import sys
import tomllib
from decimal import Decimal
from pathlib import Path


# A completed comparison exceeding policy is distinct from interpreter failure
# (1) and invalid evidence (2). bench-compare.sh maps this result to CI failure.
BLOCKING_COMPARISON = 10


def main() -> int:
    if len(sys.argv) < 2:
        print(
            f"Usage: {sys.argv[0]} <thresholds.toml> [benchmark_name]", file=sys.stderr
        )
        return 2

    config = load_config(Path(sys.argv[1]))

    if len(sys.argv) >= 3 and sys.argv[2] == "--compare":
        if len(sys.argv) != 8:
            raise ValueError("--compare requires BASE CANDIDATE BASE_JSON CANDIDATE_JSON CRITERION_DIR")
        return compare(config, *sys.argv[3:])
    if len(sys.argv) > 3:
        raise ValueError("expected CONFIG and an optional benchmark name")

    defaults = config.get("defaults", {})
    default_threshold = defaults.get("threshold_pct", 15)
    default_fail_ci = defaults.get("fail_ci", False)

    categories = config.get("categories", {})

    # If a benchmark name is given, resolve its threshold and exit.
    if len(sys.argv) >= 3:
        bench_name = sys.argv[2]
        threshold, fail_ci = resolve(
            bench_name, categories, default_threshold, default_fail_ci
        )
        print(f"{bench_name}\t{threshold}\t{str(fail_ci).lower()}")
        return 0

    # Otherwise, dump every pattern as TSV.
    print("benchmark_pattern\tthreshold_pct\tfail_ci")
    for _name, cat in categories.items():
        threshold = cat.get("threshold_pct", default_threshold)
        fail_ci = cat.get("fail_ci", default_fail_ci)
        for pattern in cat.get("benchmarks", []):
            print(f"{pattern}\t{threshold}\t{str(fail_ci).lower()}")
    print(f"__default__\t{default_threshold}\t{str(default_fail_ci).lower()}")
    return 0


def resolve(
    bench_name: str,
    categories: dict,
    default_threshold: int,
    default_fail_ci: bool,
) -> tuple[int, bool]:
    """Return (threshold_pct, fail_ci) for a specific benchmark name."""
    for _name, cat in categories.items():
        for pattern in cat.get("benchmarks", []):
            if fnmatch.fnmatchcase(bench_name, pattern):
                return cat.get("threshold_pct", default_threshold), cat.get(
                    "fail_ci", default_fail_ci
                )
    return default_threshold, default_fail_ci


def number(value: object, label: str, *, positive: bool = False) -> Decimal:
    """Reject missing, boolean, nonnumeric, negative, and non-finite evidence."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{label}: expected a number")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise ValueError(f"{label}: expected a finite {'positive' if positive else 'nonnegative'} number")
    return result


def load_config(path: Path) -> dict:
    with path.open("rb") as source:
        config = tomllib.load(source)
    defaults = config.get("defaults")
    if not isinstance(defaults, dict) or not {"threshold_pct", "fail_ci"} <= defaults.keys():
        raise ValueError("configuration requires defaults.threshold_pct and defaults.fail_ci")
    categories = config.get("categories", {})
    if not isinstance(categories, dict):
        raise ValueError("categories must be a table")
    for name, category in [("defaults", defaults), *categories.items()]:
        if not isinstance(category, dict):
            raise ValueError(f"{name}: expected a threshold table")
        number(category.get("threshold_pct", defaults["threshold_pct"]), f"{name}.threshold_pct")
        if type(category.get("fail_ci", defaults["fail_ci"])) is not bool:
            raise ValueError(f"{name}.fail_ci must be a boolean")
    for name, category in categories.items():
        patterns = category.get("benchmarks")
        if not isinstance(patterns, list) or not patterns or any(
            not isinstance(pattern, str) or not pattern for pattern in patterns
        ):
            raise ValueError(f"categories.{name}.benchmarks must be nonempty patterns")
    memory = config.get("memory", {})
    if not isinstance(memory, dict) or type(memory.get("fail_ci", False)) is not bool:
        raise ValueError("memory.fail_ci must be a boolean")
    bounds = memory.get("bounds", {})
    if not isinstance(bounds, dict):
        raise ValueError("memory.bounds must be a table")
    for name, bound in bounds.items():
        if type(bound) is not int or bound <= 0:
            raise ValueError(f"memory.bounds.{name} must be a positive byte count")
    if memory.get("fail_ci", False) and not bounds:
        raise ValueError("blocking memory policy requires bounds")
    return config


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        document = json.load(source, parse_float=Decimal,
                             parse_constant=invalid_constant, object_pairs_hook=unique_object)
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return document


def baseline(path: Path, expected: str) -> dict:
    """Read critcmp --export NAME (upstream src/data.rs BaseBenchmarks).

    critcmp uses criterion_estimates_v1.mean.point_estimate for nanoseconds.
    https://github.com/BurntSushi/critcmp/blob/master/src/data.rs
    """
    document = read_json(path)
    if document.get("name") != expected:
        raise ValueError(f"{path}: expected baseline {expected!r}")
    rows = document.get("benchmarks")
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f"{expected}: no benchmark evidence")
    result = {}
    for name, row in rows.items():
        if not name or not isinstance(row, dict):
            raise ValueError(f"{expected}/{name}: malformed benchmark")
        info = row.get("criterion_benchmark_v1")
        estimates = row.get("criterion_estimates_v1")
        if (
            row.get("baseline") != expected
            or row.get("fullname") != f"{expected}/{name}"
            or not isinstance(info, dict) or info.get("full_id") != name
            or not isinstance(estimates, dict)
            or not isinstance(estimates.get("mean"), dict)
        ):
            raise ValueError(f"{expected}/{name}: invalid identity or estimate schema")
        mean = number(estimates["mean"].get("point_estimate"), f"{expected}/{name}.mean", positive=True)
        result[name] = (mean, info)
    return result


def memory_report(policy: dict, criterion_dir: Path) -> tuple[list[str], int]:
    base_path = criterion_dir / "memory_snapshot_base.json"
    candidate_path = criterion_dir / "memory_snapshot.json"
    if not base_path.exists() and not candidate_path.exists():
        if policy.get("fail_ci", False):
            raise ValueError("blocking memory policy has no memory snapshots")
        return ["Memory: not measured (no snapshots)."], 0
    base = read_json(base_path)
    candidate = read_json(candidate_path)
    bounds = policy.get("bounds", {})
    if not base or base.keys() != candidate.keys() or not bounds.keys() <= candidate.keys():
        raise ValueError("memory snapshots must contain matching keys and every configured bound")
    lines = ["| Memory | Base bytes | Candidate bytes | Bound bytes | Status |",
             "| --- | ---: | ---: | ---: | --- |"]
    failures = 0
    for name in sorted(candidate):
        if type(base[name]) is not int or type(candidate[name]) is not int:
            raise ValueError(f"{name}: memory bytes must be integers")
        number(base[name], f"{name} base bytes")
        number(candidate[name], f"{name} candidate bytes")
        bound = bounds.get(name)
        exceeded = bound is not None and candidate[name] > bound
        failures += int(exceeded and policy.get("fail_ci", False))
        status = "EXCEEDED" if exceeded else "OK" if bound is not None else "UNBOUNDED"
        lines.append(f"| {name} | {base[name]} | {candidate[name]} | {bound or '—'} | {status} |")
    return lines, failures


def compare(config: dict, base_name: str, candidate_name: str, base_file: str,
            candidate_file: str, criterion_dir: str) -> int:
    if not base_name or not candidate_name or base_name == candidate_name:
        raise ValueError("two distinct baseline names are required")
    base = baseline(Path(base_file), base_name)
    candidate = baseline(Path(candidate_file), candidate_name)
    if base.keys() != candidate.keys():
        missing_base = sorted(candidate.keys() - base.keys())
        missing_candidate = sorted(base.keys() - candidate.keys())
        raise ValueError(f"incomplete benchmark sets; missing baseline: {missing_base}; "
                         f"missing candidate: {missing_candidate}")
    lines = [f"## Benchmark comparison: {base_name} → {candidate_name}", "",
             "Mean time change = (candidate / baseline − 1) × 100. "
             "Runner, features, and noise are not qualified by this report.", "",
             "| Benchmark | Base mean ns | Candidate mean ns | Change | Limit | Status |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    blocking = advisory = 0
    for name in sorted(base):
        base_mean, base_info = base[name]
        candidate_mean, candidate_info = candidate[name]
        for key in ("group_id", "function_id", "value_str", "throughput"):
            if key not in base_info or key not in candidate_info or base_info[key] != candidate_info[key]:
                raise ValueError(f"{name}: mismatched or missing fixture field {key}")
        percent = (candidate_mean / base_mean - 1) * 100
        threshold, fail_ci = resolve(name, config.get("categories", {}),
                                     config["defaults"]["threshold_pct"], config["defaults"]["fail_ci"])
        exceeded = percent > Decimal(str(threshold))
        blocking += int(exceeded and fail_ci)
        advisory += int(exceeded and not fail_ci)
        status = "BLOCKING" if exceeded and fail_ci else "ADVISORY" if exceeded else "OK"
        lines.append(f"| {name} | {base_mean} | {candidate_mean} | {percent:+.1f}% | {threshold}% | {status} |")
    memory, memory_failures = memory_report(config.get("memory", {}), Path(criterion_dir))
    lines += ["", *memory, "", f"Compared {len(base)} benchmark(s): {blocking} blocking, "
              f"{advisory} advisory; {memory_failures} blocking memory bound failure(s).",
              "", "<!-- grafeo-bench-comparison -->"]
    # Only emit a comparison once all evidence has validated.
    print("\n".join(lines))
    return BLOCKING_COMPARISON if blocking + memory_failures > 0 else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ArithmeticError) as error:
        print(f"Benchmark evidence error: {error}", file=sys.stderr)
        raise SystemExit(2)
