"""Package the qualified CLI, or rebuild it from the source distribution."""

import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


PLATFORMS = {
    "manylinux_2_17_x86_64.manylinux2014_x86_64": "grafeo",
    "manylinux_2_17_aarch64.manylinux2014_aarch64": "grafeo",
    "macosx_11_0_x86_64": "grafeo",
    "macosx_11_0_arm64": "grafeo",
    "win_amd64": "grafeo.exe",
}


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        rust = root / "rust"
        if self.target_name == "sdist":
            for relative in ("Cargo.toml", "Cargo.lock", "crates/grafeo-cli/src/main.rs"):
                if not (rust / relative).is_file():
                    raise ValueError("the source distribution requires its committed Rust workspace")
            return
        platform = os.environ.get("GRAFEO_CLI_WHEEL_PLATFORM")
        source_build = not (root / "grafeo_cli/grafeo").exists() and not (root / "grafeo_cli/grafeo.exe").exists() and (rust / "Cargo.toml").is_file()
        if platform is None and source_build:
            # A locally compiled wheel makes only the native platform claim.
            # Published prebuilt wheels still require the explicit release tag.
            platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        elif platform not in PLATFORMS:
            raise ValueError("GRAFEO_CLI_WHEEL_PLATFORM must name a supported native platform")
        name = "grafeo.exe" if platform == "win_amd64" else "grafeo"
        binary = root / "grafeo_cli" / name
        if source_build:
            temporary = tempfile.TemporaryDirectory(prefix="grafeo-cli-native-")
            self._native_temporary = temporary
            try:
                env = dict(os.environ)
                env.pop("CARGO_BUILD_TARGET", None)
                subprocess.run(
                    ["cargo", "build", "--locked", "--release", "--workspace",
                     "--exclude", "grafeo-python", "--target-dir", temporary.name],
                    cwd=rust, env=env, check=True,
                )
                binary = Path(temporary.name) / "release" / name
                if not binary.is_file() or binary.stat().st_size == 0:
                    raise ValueError("the native source build did not produce the CLI executable")
                build_data["force_include"][str(binary)] = "grafeo_cli/" + name
            except BaseException:
                temporary.cleanup()
                raise
        if binary.is_symlink() or not binary.is_file() or binary.stat().st_size == 0:
            raise ValueError("the platform wheel requires its bundled native executable")
        other = binary.with_name("grafeo" if binary.name == "grafeo.exe" else "grafeo.exe")
        if other.exists() or other.is_symlink():
            raise ValueError("the platform wheel must contain only its target executable")
        build_data["pure_python"] = False
        build_data["tag"] = "py3-none-" + platform

    def finalize(self, version, build_data, artifact_path):
        temporary = getattr(self, "_native_temporary", None)
        if temporary is not None:
            temporary.cleanup()
