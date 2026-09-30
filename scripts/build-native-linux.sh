#!/usr/bin/env bash
# Build the native Linux release products against the advertised GLIBC 2.17 floor.
set -euo pipefail

target=${1:?usage: build-native-linux.sh TARGET}
case "$target" in
    x86_64-unknown-linux-gnu)
        arch=x86_64
        image=quay.io/pypa/manylinux2014_x86_64@sha256:6f74cabeac2432570aa4bfdb29f7c1f30313d4d6654d764c44e574b6ffdd4ed5
        ;;
    aarch64-unknown-linux-gnu)
        arch=aarch64
        image=quay.io/pypa/manylinux2014_aarch64@sha256:d36e257b4f7b1130a1442a5cd28022307645a92b8093ab65543205426354cfd2
        ;;
    *) echo "unsupported native Linux target: $target" >&2; exit 2 ;;
esac
test "$(uname -m)" = "$arch"
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
compiler=$(rustup which --toolchain 1.97.1 rustc)
toolchain=$(cd "$(dirname "$compiler")/.." && pwd)
output=${CARGO_TARGET_DIR:-$root/target}
mkdir -p "$output"
output=$(cd "$output" && pwd)
mkdir -p "$output/.native-cargo"
network=bridge
if [[ ${CARGO_NET_OFFLINE:-false} == true ]]; then network=none; fi

# Only source, the pinned compiler and owned output are visible to the build.
# Host credentials, Cargo configuration and global caches are not mounted.
docker run --rm --read-only --cap-drop ALL --security-opt no-new-privileges \
    --network "$network" --user "$(id -u):$(id -g)" \
    --tmpfs /tmp:rw,exec,nosuid,size=2g \
    --mount "type=bind,src=$root,dst=/source,readonly" \
    --mount "type=bind,src=$toolchain,dst=/rust,readonly" \
    --mount "type=bind,src=$output,dst=/output" \
    --workdir /source \
    --env HOME=/output/.native-cargo --env CARGO_HOME=/output/.native-cargo \
    --env CARGO_TARGET_DIR=/output --env CARGO_INCREMENTAL=0 \
    --env CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
    --env CARGO_NET_OFFLINE="${CARGO_NET_OFFLINE:-false}" \
    "$image" bash -euo pipefail -c '
        export PATH=/rust/bin:$PATH
        test "$(rustc --version | cut -d " " -f 2)" = 1.97.1
        test "$(getconf GNU_LIBC_VERSION)" = "glibc 2.17"
        rustc -vV
        cargo -V
        cargo build --locked --release --workspace --target "$1" --exclude grafeo-python
        for binary in "/output/$1/release/grafeo" "/output/$1/release/libgrafeo_c.so"; do
            readelf --version-info "$binary" > /tmp/versions
            highest=$(grep -oE "GLIBC_[0-9]+(\.[0-9]+)+" /tmp/versions | sort -Vu | tail -1)
            test -n "$highest"
            test "$(printf "%s\n" "$highest" GLIBC_2.17 | sort -V | tail -1)" = GLIBC_2.17
            printf "%s: %s\n" "$binary" "$highest"
        done
        "/output/$1/release/grafeo" --version
    ' bash "$target"
