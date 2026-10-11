#!/bin/sh
# Read-only container root, project-backed cache/target/log mounts, private /tmp.
set -eu
export TMP=/tmp TEMP=/tmp TMPDIR=/tmp
export CARGO_HOME=/cargo-home CARGO_TARGET_DIR=/target CARGO_BUILD_JOBS=2
export RUSTUP_HOME=/cargo-home/rustup RUSTUP_TOOLCHAIN=1.97.1
export RUST_ICU_MAJOR_VERSION_NUMBER=78
export RUSTFLAGS='-L native=/build/icu/lib'
export RUSTDOCFLAGS='-D warnings -L native=/build/icu/lib'
export MINION_AGENT_ICU_IDENTITY=/build/icu-identity.txt
export LD_LIBRARY_PATH=/build/icu/lib MINION_SEARCH_ENGINE_ARTIFACTS=/search-artifacts
export PYTHONPYCACHEPREFIX=/cargo-home/pycache XDG_CACHE_HOME=/cargo-home/xdg
export NODE_COMPILE_CACHE=/cargo-home/node-cache npm_config_cache=/cargo-home/npm
export PIP_CACHE_DIR=/cargo-home/pip UV_CACHE_DIR=/cargo-home/uv
mkdir -p /tmp/l12d007/work /tmp/bin /tmp/l12d007/fixtures
chmod 777 /tmp/l12d007/fixtures
export MINION_FIXTURE_ROOT=/tmp/l12d007/fixtures
cp -r /source/. /tmp/l12d007/work/
cp /node /tmp/bin/node
chmod +x /tmp/bin/node
export PATH=/tmp/bin:/cargo-home/rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin:$PATH
find /tmp/l12d007/work -name '*.sh' -exec sed -i 's/\r$//' {} +
cd /tmp/l12d007/work/minion-agent-rust
rustc --version
node --version
cargo fmt --all -- --check > /logs/fmt.log 2>&1
cargo clippy --workspace --all-targets --all-features -- -D warnings > /logs/clippy.log 2>&1
cargo test --workspace --all-features > /logs/test.log 2>&1
cargo doc --workspace --no-deps > /logs/doc.log 2>&1
cargo run -p xtask -- conformance verify > /logs/xtask.log 2>&1
cargo test -p minion-agent --lib l12d007_all_41_pi_routing_rows_match_and_injections_fire -- --nocapture > /logs/ce02.log 2>&1
# Permission-sensitive canonical corpora run as uid 1000, never credited as root.
cargo test -p minion-agent --test fs_path_domain_conformance --test fs_remove_readonly_conformance --no-run --message-format=json > /logs/corpus-build.json 2>/logs/corpus-build.log
python3 - <<'PY'
import json, subprocess
for row in map(json.loads, open('/logs/corpus-build.json')):
    if row.get('executable') and row.get('target', {}).get('name') in ('fs_path_domain_conformance', 'fs_remove_readonly_conformance'):
        with open('/logs/' + row['target']['name'] + '-uid1000.log', 'w') as log:
            subprocess.run(['setpriv', '--reuid', '1000', '--regid', '1000', '--clear-groups', row['executable'], '--nocapture'], stdout=log, stderr=subprocess.STDOUT, check=True)
PY
mkdir /tmp/l12d007/control-code
cp -r /tmp/l12d007/work/minion-agent-rust /tmp/l12d007/control-code/
cp -r /tmp/l12d007/work/conformance /tmp/l12d007/control-code/
mkdir -p /tmp/l12d007/control-code/minion-agent-python/tests/skills/data /tmp/l12d007/control-code/minion-agent-python/tests/execution/data/r002_ada_oracle
cp /tmp/l12d007/work/minion-agent-python/tests/skills/data/frontmatter-corpus.json /tmp/l12d007/work/minion-agent-python/tests/skills/data/ignore-corpus.json /tmp/l12d007/control-code/minion-agent-python/tests/skills/data/
cp /tmp/l12d007/work/minion-agent-python/tests/execution/data/r002_ada_oracle/systematic_ada292.txt /tmp/l12d007/control-code/minion-agent-python/tests/execution/data/r002_ada_oracle/
set +e
python3 scripts/error-codes-negative-controls.py --tree /tmp/l12d007/control-code/minion-agent-rust --logs /tmp/l12d007/controls > /logs/controls.log 2>&1
control_status=$?
set -e
# Preserve INVALID/failing evidence too; never rerun a vanished failure blind.
cp -r /tmp/l12d007/controls /logs/control-details
if [ "$control_status" -ne 0 ]; then exit "$control_status"; fi
echo 'L12-D007 LINUX G3, CE02, UNPRIVILEGED CORPORA AND CONTROLS PASS'
