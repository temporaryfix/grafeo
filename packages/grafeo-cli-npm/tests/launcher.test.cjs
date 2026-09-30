"use strict";

const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

// The stand-in runs Node, so a fallback can be observed without recursively
// launching this package. POSIX keeps Node at its installed shared-library path.
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "grafeo-cli-launcher-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const launcher = path.join(root, "launcher", "bin", "grafeo.js");
  fs.mkdirSync(path.dirname(launcher), { recursive: true });
  fs.copyFileSync(path.join(__dirname, "..", "bin", "grafeo.js"), launcher);
  const search = path.join(root, "search");
  fs.mkdirSync(search);
  const binary = os.platform() === "win32" ? "grafeo.exe" : "grafeo";
  function executable(file) {
    if (os.platform() === "win32") {
      fs.copyFileSync(process.execPath, file);
    } else {
      const node = "'" + process.execPath.replace(/'/g, "'\"'\"'") + "'";
      fs.writeFileSync(file, `#!/bin/sh\nexec ${node} "$@"\n`, { mode: 0o755 });
    }
  }
  executable(path.join(search, binary));
  const pkg = path.join(root, "launcher", "node_modules", "@grafeo-db", `cli-${os.platform()}-${os.arch()}`);
  function dependency(complete = false) {
    fs.mkdirSync(pkg, { recursive: true });
    fs.writeFileSync(path.join(pkg, "package.json"), JSON.stringify({ name: `@grafeo-db/cli-${os.platform()}-${os.arch()}`, version: "0.0.1" }));
    if (complete) executable(path.join(pkg, binary));
  }
  function run(args = ["-e", "console.log('UNRELATED-BINARY')"]) {
    const result = spawnSync(process.execPath, [launcher, ...args], {
      env: { ...process.env, PATH: search, NODE_PATH: "" },
      encoding: "utf8",
      timeout: 5000,
    });
    assert.equal(result.error, undefined);
    return result;
  }
  return { root, binary, dependency, executable, run };
}

for (const state of ["missing", "incomplete", "adjacent"]) {
  test(`rejects ${state} native dependency without selecting another binary`, (t) => {
    const f = fixture(t);
    if (state === "incomplete") f.dependency();
    if (state === "adjacent") {
      f.executable(path.join(f.root, "launcher", f.binary));
    }
    const result = f.run();
    assert.equal(result.status, 1);
    assert.equal(result.stdout, "");
    assert.match(result.stderr, /native package.*missing or incomplete/i);
    assert.match(result.stderr, /optional dependencies/i);
  });
}

test("forwards literal arguments and the packaged executable's exit status", (t) => {
  const f = fixture(t);
  f.dependency(true);
  const args = ["a space", "$(echo injected)", ";echo injected"];
  const result = f.run(["-e", "console.log(JSON.stringify(process.argv.slice(1))); process.exit(23)", "--", ...args]);
  assert.equal(result.status, 23, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), args);
  assert.equal(result.stderr, "");
});
