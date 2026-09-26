# Pinned ICU for `ls` collation

`ls` sorts with ONE pinned collation tuple (spec/tools.md `TOOL-028` "Collation", `R006-C`,
manifest `TOOL-040`): PyICU 2.16.2 linked against ICU 78.3 built from the official
`icu4c-78.3-sources.tgz` (SHA-512 checked against the release's `SHASUM512.txt`). There is no
usable PyICU wheel, and `ls` refuses to sort with any other ICU: loading checks PyICU's version,
the ICU it was compiled against, and the ICU actually loaded at runtime (`u_getVersion`).

```bash
bash scripts/pinned-icu/build.sh ../.toolchain/icu-78.3        # once; prints the environment
eval "$(bash scripts/pinned-icu/build.sh ../.toolchain/icu-78.3 --env)"
uv sync                                                          # builds PyICU from the locked sdist
uv run pytest
```

**Build identity (`L13-WP131-FR003`).** A version check cannot tell this build from another ICU
that also reports 78.3, so the script records the build's identity in
`<prefix>/pinned-icu-identity.txt` (the source tarball's SHA-512 and the SHA-256 of the built
`icuuc`/`icui18n`/`icudata` libraries), and `--env` exports its path as
`MINION_AGENT_ICU_IDENTITY`. At load, `ls` hashes the ICU libraries actually mapped into the
process and fails closed unless all three match. For a build made before this check existed,
`bash scripts/pinned-icu/build.sh <prefix> --identity` records it (the source tarball must
still verify).

`uv sync` builds PyICU from the sdist whose hash `uv.lock` pins, using the `PYICU_*` variables the
script prints. `--env` also sets `UV_NO_CACHE`, so a PyICU wheel built against some other ICU is
never reused. Nothing is installed outside the prefix.

- **Windows** (Git Bash + Visual Studio 2022): the Windows SDK ships its own `icuuc.lib`/`icuin.Lib`
  (the system ICU) on the default library path, so the script links the pinned libraries by
  ABSOLUTE path. At runtime `MINION_AGENT_ICU_BIN` names the pinned DLL directory; the loader adds
  it before importing PyICU.
- **Linux**: the pinned library directory is passed with `-L` ahead of the system path and baked in
  with `-rpath`. A host `libicu-dev` of another version must not win the link; the runtime version
  check fails closed if it does.

`read` needs nothing extra: its image processing runs the vendored, SHA-256-checked
`photon_rs_bg.wasm` through `wasmtime` (`src/minion_agent/tools/builtin/photon/PROVENANCE.txt`).
