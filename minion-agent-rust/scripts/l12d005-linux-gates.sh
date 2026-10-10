#!/bin/sh
# rust:1.97.1-trixie, read-only rootfs; /tmp tmpfs; E:-backed mounts below.
set -eu
export TMP=/tmp TEMP=/tmp TMPDIR=/tmp
export CARGO_HOME=/cargo-home CARGO_TARGET_DIR=/target CARGO_BUILD_JOBS=2
export RUSTUP_HOME=/cargo-home/rustup RUSTUP_TOOLCHAIN=1.97.1
export RUST_ICU_MAJOR_VERSION_NUMBER=78
export RUSTFLAGS='-L native=/icu/icu/lib'
export RUSTDOCFLAGS='-D warnings -L native=/icu/icu/lib'
export MINION_AGENT_ICU_IDENTITY=/icu/icu-identity.txt
export LD_LIBRARY_PATH=/icu/icu/lib MINION_SEARCH_ENGINE_ARTIFACTS=/search-artifacts
export PYTHONPYCACHEPREFIX=/cargo-home/pycache XDG_CACHE_HOME=/cargo-home/xdg
export NODE_COMPILE_CACHE=/cargo-home/node-cache npm_config_cache=/cargo-home/npm
export PIP_CACHE_DIR=/cargo-home/pip UV_CACHE_DIR=/cargo-home/uv
mkdir -p /tmp/work /tmp/bin /logs
cp -r /source/. /tmp/work/
cp /node /tmp/bin/node
chmod +x /tmp/bin/node
export PATH=/tmp/bin:/cargo-home/rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin:$PATH
find /tmp/work -name '*.sh' -exec sed -i 's/\r$//' {} +
cd /tmp/work/minion-agent-rust
rustc --version
node --version
cargo fmt --all -- --check > /logs/fmt.log 2>&1
cargo clippy --workspace --all-targets --all-features -- -D warnings > /logs/clippy.log 2>&1
cargo test --workspace --all-features > /logs/test.log 2>&1
cargo doc --workspace --no-deps > /logs/doc.log 2>&1
cargo run -p xtask -- conformance verify > /logs/xtask.log 2>&1
# Run the permission-sensitive corpus as an unprivileged uid, not as root.
cargo test -p minion-agent --test fs_remove_readonly_conformance --no-run --message-format=json > /logs/corpus-build.json 2>/logs/corpus-build.log
binary=$(python3 -c 'import json; print(next(x["executable"] for x in map(json.loads, open("/logs/corpus-build.json")) if x.get("target",{}).get("name")=="fs_remove_readonly_conformance" and x.get("executable")))')
setpriv --reuid 1000 --regid 1000 --clear-groups "$binary" --nocapture > /logs/corpus-unprivileged.log 2>&1
# Controls run in a separate disposable copy, with the same single platform target.
mkdir -p /tmp/control-code
cp -r /tmp/work/. /tmp/control-code/
python3 scripts/readonly-remove-negative-controls.py --tree /tmp/control-code/minion-agent-rust --logs /logs/controls
echo 'L12-D005 LINUX G3, UNPRIVILEGED CORPUS AND CONTROLS PASS'
