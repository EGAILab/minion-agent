# Pinned Photon runtime artifact

`photon_rs_bg.wasm` is the exact WASM artifact from `@silvia-odwyer/photon-node` 0.3.4 used by pinned Pi. It was mechanically copied from the accepted Python WP-13.1 package; the bytes are not regenerated or substituted by another image engine.

SHA-256: `10468181565c56004c867f3a4af96f89a0ef5a63a72f2b5fb12c1f1992a3615c`.

The Rust host verifies this hash before compiling the module and uses `wasmtime` 49.0.0. The R005-A differential corpus and the Rust canonical cases compare the resulting image bytes to the pinned Pi/Photon outputs. Updating this artifact or runtime is a contract-affecting change, not routine dependency maintenance.
