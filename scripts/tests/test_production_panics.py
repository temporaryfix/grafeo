"""Run the real production audit against controlled Cargo protocol fixtures."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ["grafeo-common", "grafeo-core", "grafeo-storage", "grafeo-engine"]
LINTS = ["clippy::unwrap_used", "clippy::expect_used", "clippy::panic", "clippy::unreachable"]
FINISHED = {"reason": "build-finished", "success": True}
TARGET = {"kind": ["lib"], "crate_types": ["lib"], "name": "grafeo_core",
          "src_path": "/fixture/src/lib.rs", "edition": "2024",
          "doc": True, "doctest": True, "test": True}


def diagnostic(code, level="error"):
    return {"reason": "compiler-message", "package_id": "path+file:///fixture#grafeo-core@0.0.1",
            "manifest_path": "/fixture/Cargo.toml", "target": TARGET,
            "message": {"message": "fixture diagnostic", "code": {"code": code, "explanation": None},
                        "level": level, "spans": [{"file_name": "src/lib.rs", "byte_start": 0,
                        "byte_end": 1, "line_start": 7, "line_end": 7, "column_start": 3,
                        "column_end": 4, "is_primary": True, "text": [], "label": None,
                        "suggested_replacement": None, "suggestion_applicability": None,
                        "expansion": None}], "children": [], "rendered": "fixture diagnostic at src/lib.rs:7:3\n"}}


def library_artifact(root, package, *, fresh=False):
    directory = (root / "crates" / package).resolve()
    return {"reason": "compiler-artifact", "package_id": f"path+{directory.as_uri()}#0.0.1",
            "manifest_path": str(directory / "Cargo.toml"),
            "target": {**TARGET, "name": package.replace("-", "_"),
                       "src_path": str(directory / "src/lib.rs")},
            "profile": {"opt_level": "0", "debuginfo": 2, "debug_assertions": True,
                        "overflow_checks": True, "test": False},
            "features": [], "filenames": [str(root / "target/debug/libfixture.rmeta")],
            "executable": None, "fresh": fresh}


class ProductionPanicTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "scripts/check-production-panics.py").is_file(),
                        "The production audit runner must exist")
        self.tmp = tempfile.TemporaryDirectory(prefix="grafeo-panic-gate-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repository with spaces"
        (self.root / "scripts").mkdir(parents=True)
        self.script = self.root / "scripts/check-production-panics.py"
        shutil.copy2(ROOT / "scripts/check-production-panics.py", self.script)
        (self.root / "Cargo.toml").write_text('[workspace.package]\nversion = "0.0.1"\n')
        for package in PACKAGES:
            directory = self.root / "crates" / package
            (directory / "src").mkdir(parents=True)
            (directory / "Cargo.toml").write_text(f'[package]\nname = "{package}"\nversion.workspace = true\n')
            (directory / "src/lib.rs").write_text("// fixture library\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "fixture")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.commands = Path(self.tmp.name) / "commands"
        self.commands.mkdir()
        self.log = Path(self.tmp.name) / "cargo-call.json"
        self.output = Path(self.tmp.name) / "cargo-output.jsonl"
        cargo = self.commands / "cargo"
        cargo.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
pathlib.Path(os.environ["PANIC_CALL"]).write_text(json.dumps([sys.argv, os.getcwd()]))
sys.stdout.write(pathlib.Path(os.environ["PANIC_OUTPUT"]).read_text())
sys.stderr.write("cargo fixture stderr\\n")
sys.exit(int(os.environ.get("PANIC_EXIT", "0")))
''', encoding="utf-8")
        cargo.chmod(0o755)
        self.env = {**os.environ, "PATH": str(self.commands) + os.pathsep + os.environ["PATH"],
                    "PANIC_CALL": str(self.log), "PANIC_OUTPUT": str(self.output)}
        self.messages(*(library_artifact(self.root, package) for package in PACKAGES), FINISHED)

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, text=True,
                                       stderr=subprocess.PIPE)

    def run_audit(self, *args):
        result = subprocess.run([sys.executable, str(self.script), *args], cwd=self.commands,
                                env=self.env, text=True, capture_output=True, timeout=15)
        self.assertTrue(result.stdout.strip(), result.stderr)
        document = json.loads(result.stdout)
        self.assertIs(type(document["audit_passed"]), bool)
        return result, document

    def messages(self, *messages):
        self.output.write_text("".join(json.dumps(item) + "\n" for item in messages), encoding="utf-8")

    def complete_messages(self, *messages):
        self.messages(*(library_artifact(self.root, package) for package in PACKAGES), *messages)

    def assert_failed(self, *args):
        result, document = self.run_audit(*args)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(document["audit_passed"])
        return result, document

    def test_review_completion_only_cannot_satisfy_requested_library_coverage(self):
        self.messages(FINISHED)
        _, document = self.assert_failed("-p", "grafeo-core")
        self.assertEqual(document["library_coverage"]["missing"], ["grafeo-core"])

    def test_review_partial_coverage_cannot_satisfy_default_four_package_audit(self):
        self.messages(library_artifact(self.root, "grafeo-core"), FINISHED)
        _, document = self.assert_failed()
        self.assertEqual(document["library_coverage"]["missing"],
                         ["grafeo-common", "grafeo-storage", "grafeo-engine"])

    def test_review_dependency_or_build_script_artifacts_cannot_replace_requested_library(self):
        for package, kind in (("grafeo-common", "lib"), ("grafeo-common", "bin"),
                              ("grafeo-core", "custom-build")):
            with self.subTest(package=package, kind=kind):
                artifact = library_artifact(self.root, package)
                artifact["target"]["kind"] = [kind]
                if kind != "lib":
                    artifact["target"]["crate_types"] = ["bin"]
                    artifact["target"]["name"] = "build_script_build" if kind == "custom-build" else "helper"
                    artifact["target"]["src_path"] = str(self.root / "build.rs")
                self.messages(artifact, FINISHED)
                self.assert_failed("-p", "grafeo-core")

    def test_review_unknown_target_kind_fails_even_beside_complete_libraries(self):
        artifact = library_artifact(self.root, "grafeo-common")
        artifact["target"]["kind"] = ["unknown-kind"]
        self.messages(artifact, library_artifact(self.root, "grafeo-core"), FINISHED)
        self.assert_failed("-p", "grafeo-core")

    def test_review_wrong_requested_library_identity_fails(self):
        for field, value in (("manifest_path", str(self.root / "wrong/Cargo.toml")),
                             ("package_id", "registry+https://example.invalid#index#grafeo-core@0.0.1"),
                             ("name", "wrong_name"), ("src_path", str(self.root / "wrong/lib.rs")),
                             ("crate_types", ["bin"]), ("fresh", "true")):
            with self.subTest(field=field):
                artifact = library_artifact(self.root, "grafeo-core")
                target = artifact if field in ("manifest_path", "package_id", "fresh") else artifact["target"]
                target[field] = value
                self.messages(artifact, FINISHED)
                self.assert_failed("-p", "grafeo-core")

    def test_review_unsupported_package_selector_fails_before_cargo(self):
        self.assert_failed("-p", "grafeo-unrelated")
        self.assertFalse(self.log.exists())

    def test_review_fresh_cached_libraries_still_satisfy_every_requested_package(self):
        self.messages(*(library_artifact(self.root, package, fresh=True) for package in PACKAGES), FINISHED)
        result, document = self.run_audit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(document.get("library_coverage"),
                         {"required": PACKAGES, "observed": PACKAGES, "missing": []})

    def test_clean_completed_build_records_real_git_and_strict_library_argv_from_foreign_cwd(self):
        result, document = self.run_audit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(document["audit_passed"])
        self.assertEqual(document["source"]["commit"], self.commit)
        self.assertFalse(document["source"]["dirty"])
        self.assertEqual(document["cargo_exit_status"], 0)
        self.assertEqual(document["build_finished"], {"seen": True, "success": True})
        expected = ["cargo", "+1.97.1", "clippy", "--locked", "--lib", "--no-deps",
                    "--message-format=json", "-p", "grafeo-common", "-p", "grafeo-core",
                    "-p", "grafeo-storage", "-p", "grafeo-engine", "--all-features", "--",
                    "-D", "warnings", "-D", "clippy::unwrap_used", "-D", "clippy::expect_used",
                    "-D", "clippy::panic", "-D", "clippy::unreachable"]
        self.assertEqual(document["cargo_argv"], expected)
        argv, cwd = json.loads(self.log.read_text())
        self.assertEqual(argv[1:], expected[1:])
        self.assertEqual(Path(cwd).resolve(), self.root.resolve())
        self.assertIn("cargo fixture stderr", result.stderr)
        self.assertIn("semantic totality", document["limitations"])

    def test_dirty_source_is_identified_not_presented_as_clean_commit(self):
        (self.root / "untracked.txt").write_text("dirty")
        _, document = self.run_audit()
        self.assertTrue(document["source"]["dirty"])
        self.assertIn("untracked.txt", document["source"]["status"])

    def test_each_restriction_is_counted_with_locations_and_human_detail(self):
        self.messages(*(diagnostic(code) for code in LINTS), {"reason": "build-finished", "success": False})
        self.env["PANIC_EXIT"] = "101"
        result, document = self.assert_failed()
        self.assertEqual(document["counts"]["restrictions"], dict.fromkeys(LINTS, 1))
        self.assertEqual(document["counts"]["restriction_total"], 4)
        self.assertEqual(document["counts"]["errors"], 4)
        self.assertEqual(document["diagnostics"][0]["message"]["spans"][0]["line_start"], 7)
        self.assertIn("fixture diagnostic at src/lib.rs:7:3", result.stderr)

    def test_unrelated_compile_failure_is_not_a_zero_restriction_pass(self):
        self.messages(diagnostic("E0308"), {"reason": "build-finished", "success": False})
        self.env["PANIC_EXIT"] = "101"
        _, document = self.assert_failed()
        self.assertEqual(document["counts"]["restriction_total"], 0)
        self.assertEqual(document["cargo_exit_status"], 101)

    def test_exit_zero_error_or_warning_diagnostics_still_fail(self):
        for level in ("error", "warning", "error: internal compiler error"):
            with self.subTest(level=level):
                self.complete_messages(diagnostic("E0308", level), FINISHED)
                self.assert_failed()

    def test_restriction_diagnostic_cannot_pass_at_a_nondeny_level(self):
        self.complete_messages(diagnostic("clippy::panic", "note"), FINISHED)
        self.assert_failed()

    def test_successful_build_message_does_not_override_nonzero_tool_exit(self):
        self.env["PANIC_EXIT"] = "7"
        self.assert_failed()

    def test_missing_malformed_duplicate_or_nonterminal_build_evidence_fails(self):
        for payload in ("", "{}\n", "not json\n", "[]\n", '{"reason":',
                        '{"reason":"build-finished","success":1}\n',
                        '{"reason":"build-finished","success":true,"success":false}\n',
                        json.dumps(FINISHED) + "\n" + json.dumps(FINISHED) + "\n",
                        json.dumps(FINISHED) + "\n" + json.dumps(diagnostic("E0308")) + "\n"):
            with self.subTest(payload=payload):
                prefix = "".join(json.dumps(library_artifact(self.root, package)) + "\n" for package in PACKAGES)
                self.output.write_text(prefix + payload)
                _, document = self.assert_failed()
                self.assertTrue(document["protocol_errors"])

    def test_malformed_diagnostics_cannot_be_silently_ignored(self):
        for field, value in (("code", 7), ("spans", "invalid"), ("level", None)):
            with self.subTest(field=field):
                message = diagnostic("clippy::panic")
                message["message"][field] = value
                self.complete_messages(message, FINISHED)
                _, document = self.assert_failed()
                self.assertTrue(document["protocol_errors"])

    def test_overflowing_json_number_still_emits_a_failing_json_record(self):
        payload = json.dumps(diagnostic("E0308", "note"))
        payload = payload.replace('"explanation": null', '"explanation": 1e999')
        prefix = "".join(json.dumps(library_artifact(self.root, package)) + "\n" for package in PACKAGES)
        self.output.write_text(prefix + payload + "\n" + json.dumps(FINISHED) + "\n")
        _, document = self.assert_failed()
        self.assertTrue(document["protocol_errors"])

    def test_test_profile_artifacts_are_not_production_evidence(self):
        artifacts = [library_artifact(self.root, package) for package in PACKAGES]
        artifacts[1]["profile"]["test"] = True
        self.messages(*artifacts, FINISHED)
        self.assert_failed()

    def test_focused_package_features_and_no_defaults_are_forwarded(self):
        result, document = self.run_audit("-p", "grafeo-core", "--no-default-features",
                                          "--features", "text-index,lpg")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = document["cargo_argv"]
        self.assertEqual(argv[7:13], ["-p", "grafeo-core", "--no-default-features",
                                     "--features", "text-index,lpg", "--"])
        self.assertNotIn("--all-features", argv)
        self.assertEqual(document["features"], {"all": False, "no_default": True,
                                              "requested": ["text-index,lpg"]})

    def test_missing_cargo_is_a_json_tool_failure(self):
        (self.commands / "cargo").unlink()
        (self.commands / "git").symlink_to(shutil.which("git"))
        self.env["PATH"] = str(self.commands)
        _, document = self.assert_failed()
        self.assertIsNone(document["cargo_exit_status"])
        self.assertTrue(document["tool_errors"])

    def test_missing_source_identity_is_a_json_tool_failure(self):
        (self.root / ".git").rename(self.root / "git-hidden")
        _, document = self.assert_failed()
        self.assertTrue(document["tool_errors"])
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
