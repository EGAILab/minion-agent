# Node path-resolution observations (L12-D012)

These are frozen **Node**, not Python, observations. The shared 80-case corpus
remains in `conformance/agent/path-resolution/path-resolution.json`.

- `node_selected_value.json`: Node v22.15.1 / pinned Pi b7bb00b9;
  minion-agent-docs `assurance/layers/data/l12-d012/ce01/build_fixture.mjs` rebuilds
  it from the recorded entry, cwd and logical partitions. SHA-256:
  `634ceeb85ca7477e262ea33e295df772b7ce4ea725e178a82d878efdefa8ee84`.
- `node_win32_resolve.json`: 46 ordinary resolve rows, seven crafted-environment
  resolve rows and eight normalize rows. The generator is minion-agent-docs
  `assurance/layers/data/l12-d012/gen/gen_port_fixture.mjs`. The Rust copy equals
  the existing Node artifact after CRLF normalization. LF SHA-256:
  `c4b7c0ef0dcdf69c3f543136cb303c82caae9524da37d07c2a6a669ef76892eb`.

No case path in these files is an OS write, read, remove or cwd target. The native
child replay creates only a new guarded fixture root and the five plain empty
cwd-name fixtures; it keeps them. The observer receives inputs and preconditions
only, never expected results.
