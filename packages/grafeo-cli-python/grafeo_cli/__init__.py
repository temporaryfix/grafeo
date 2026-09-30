"""Grafeo CLI launcher.

Thin wrapper that finds and runs the grafeo binary bundled with this package.
Install with: pip install grafeo-cli
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

__version__ = "0.0.1"

def _binary_name() -> str:
    """Return the platform-specific binary name."""
    return "grafeo.exe" if platform.system() == "Windows" else "grafeo"


def _find_binary() -> Path | None:
    """Find only the executable owned by this installed package."""
    bundled = Path(__file__).parent / _binary_name()
    return bundled if bundled.is_file() else None


def main() -> None:
    """Run the grafeo CLI binary, forwarding all arguments."""
    binary = _find_binary()

    if binary is None:
        print(
            "error: bundled grafeo binary is missing.\n"
            "Reinstall a grafeo-cli wheel for your platform.\n"
            "This launcher requires the executable bundled with its package.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Make binary executable on Unix
    if os.name != "nt" and not os.access(binary, os.X_OK):
        binary.chmod(binary.stat().st_mode | 0o111)

    try:
        result = subprocess.run(
            [str(binary), *sys.argv[1:]],
            check=False,
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except FileNotFoundError:
        print(f"error: failed to execute {binary}", file=sys.stderr)
        sys.exit(1)
