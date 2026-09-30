#!/usr/bin/env bash
# Generate the Linux addon, loader and declarations against GLIBC 2.28.
set -euo pipefail
target=${1:?usage: build-node-linux.sh TARGET}
case "$target" in
    x86_64-unknown-linux-gnu)
        arch=x86_64; platform=linux-x64-gnu
        image=quay.io/pypa/manylinux_2_28_x86_64@sha256:407f771c51a2c3e83ebe5a7970b4289ead3a6db21d9b9c089168775cad11d328
        ;;
    aarch64-unknown-linux-gnu)
        arch=aarch64; platform=linux-arm64-gnu
        image=quay.io/pypa/manylinux_2_28_aarch64@sha256:c22ffd129ac99a8a42d1f2c2f4e88a9089288dd9ee987a7092da1c7dc48f27a9
        ;;
    *) echo "unsupported native Linux target: $target" >&2; exit 2 ;;
esac
test "$(uname -m)" = "$arch"
test "$(node --version)" = v24.21.0
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
compiler=$(rustup which --toolchain 1.97.1 rustc)
toolchain=$(cd "$(dirname "$compiler")/.." && pwd)
node_root=$(dirname "$(dirname "$(node -p 'process.execPath')")")
test -d "$node_root/lib/node_modules/npm"
tools=${NAPI_BUILD_TOOLS:-$root/crates/bindings/node/tools/node_modules}
tools=$(cd "$tools" && pwd)
output=${CARGO_TARGET_DIR:-$root/target/node-release}
mkdir -p "$output"
output=$(cd "$output" && pwd)
mkdir -p "$output/cargo-home" "$output/package"
cp "$root/crates/bindings/node/package.json" "$root/crates/bindings/node/types-header.d.ts" "$output/package/"
network=bridge
if [[ ${CARGO_NET_OFFLINE:-false} == true ]]; then network=none; fi
# NAPI's post-build file lock needs an owned package namespace. Rust source and
# tool installations remain read-only; the copied package inputs are exact bytes.
docker run --rm --read-only --cap-drop ALL --security-opt no-new-privileges \
    --network "$network" --user "$(id -u):$(id -g)" \
    --tmpfs /tmp:rw,exec,nosuid,size=2g \
    --mount "type=bind,src=$root,dst=/source,readonly" \
    --mount "type=bind,src=$toolchain,dst=/rust,readonly" \
    --mount "type=bind,src=$node_root,dst=/node,readonly" \
    --mount "type=bind,src=$tools,dst=/tools/node_modules,readonly" \
    --mount "type=bind,src=$output,dst=/work" \
    --workdir /source/crates/bindings/node \
    --env HOME=/work --env CARGO_HOME=/work/cargo-home \
    --env CARGO_TARGET_DIR=/work/target --env CARGO_INCREMENTAL=0 \
    --env CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
    --env CARGO_NET_OFFLINE="${CARGO_NET_OFFLINE:-false}" \
    "$image" bash -euo pipefail -c '
        export PATH=/rust/bin:/node/bin:/tools/node_modules/.bin:$PATH
        test "$(rustc --version | cut -d " " -f 2)" = 1.97.1
        test "$(node --version)" = v24.21.0
        test "$(napi --version)" = 3.10.5
        test "$(getconf GNU_LIBC_VERSION)" = "glibc 2.28"
        napi build --platform --release --features full --package-json-path /work/package/package.json --output-dir /work/package/generated -- --locked
        cat /source/crates/bindings/node/loader-footer.js >> /work/package/generated/index.js
        binary="/work/package/generated/grafeo.$1.node"
        readelf --version-info "$binary" > /tmp/versions
        highest=$(grep -oE "GLIBC_[0-9]+(\.[0-9]+)+" /tmp/versions | sort -Vu | tail -1)
        test -n "$highest"
        test "$(printf "%s\n" "$highest" GLIBC_2.28 | sort -V | tail -1)" = GLIBC_2.28
        printf "%s: %s\n" "$binary" "$highest"
        test -s /work/package/generated/index.js
        test -s /work/package/generated/index.d.ts
    ' bash "$platform"
