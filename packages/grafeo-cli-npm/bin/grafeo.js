#!/usr/bin/env node

/**
 * Grafeo CLI launcher for npm.
 *
 * Resolves the platform-specific binary from optional dependencies
 * and spawns it with all arguments forwarded.
 */

"use strict";

const { execFileSync } = require("child_process");
const { existsSync } = require("fs");
const { join } = require("path");
const { platform, arch } = require("os");

/** Map of Node.js platform/arch to package names and binary paths. */
const PLATFORMS = {
  "linux-x64": { pkg: "@grafeo-db/cli-linux-x64", bin: "grafeo" },
  "linux-arm64": { pkg: "@grafeo-db/cli-linux-arm64", bin: "grafeo" },
  "darwin-x64": { pkg: "@grafeo-db/cli-darwin-x64", bin: "grafeo" },
  "darwin-arm64": { pkg: "@grafeo-db/cli-darwin-arm64", bin: "grafeo" },
  "win32-x64": { pkg: "@grafeo-db/cli-win32-x64", bin: "grafeo.exe" },
};

function findBinary() {
  const key = `${platform()}-${arch()}`;
  const entry = PLATFORMS[key];

  if (!entry) {
    console.error(
      `error: unsupported platform ${key}\n` +
        `Supported: ${Object.keys(PLATFORMS).join(", ")}\n` +
        "Install the binary manually: cargo install grafeo-cli"
    );
    process.exit(1);
  }

  // Only the matching package owns this launcher's native executable.
  try {
    const pkgDir = require.resolve(`${entry.pkg}/package.json`);
    const binPath = join(pkgDir, "..", entry.bin);
    if (existsSync(binPath)) {
      return binPath;
    }
  } catch {
    // Package not installed (optional dependency)
  }

  console.error(
    `error: native package ${entry.pkg} is missing or incomplete.\n` +
      "Reinstall @grafeo-db/cli with optional dependencies enabled."
  );
  process.exit(1);
}

const binary = findBinary();

try {
  execFileSync(binary, process.argv.slice(2), {
    stdio: "inherit",
    windowsHide: true,
  });
} catch (error) {
  if (error.status != null) {
    process.exit(error.status);
  }

  if (error.code === "ENOENT") {
    console.error(
      `error: grafeo binary not found.\n\n` +
        `Install via one of:\n` +
        `  npm install @grafeo-db/cli\n` +
        `  cargo install grafeo-cli\n` +
        `  Download from https://github.com/GrafeoDB/grafeo/releases\n`
    );
    process.exit(1);
  }

  console.error(`error: ${error.message}`);
  process.exit(1);
}
