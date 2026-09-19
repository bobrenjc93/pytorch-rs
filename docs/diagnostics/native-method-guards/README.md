# Native compile method identity batch

This candidate carries C's required R publication repair plus one stateless
native method-identity batch. Python retains the inventory and expected objects;
no ShapeGuards hash change or other runtime optimization is included. The
[guide](../../compile-pointwise-jit.md) describes the boundary and publication lifetime.

## Development validation

Fresh, separate release builds used Python 3.12.14 and 3.14.5, Rust 1.92,
PyO3 0.29.2/ABI3, PyTorch 2.13.0+cu130 and H100 GPU0
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`. The selected runtime and NVRTC
reported 13.0; `nvcc` was 12.6. Format, Cargo check, Clippy with warnings denied,
the unchanged native verifier and local interpreter/CUDA containment probes passed.

| Python | Complete pointwise suite | Compiler-owner suite |
|---|---|---|
| 3.12.14 | 516 passed, 9 skipped | 8 passed, 0 skipped |
| 3.14.5 | 516 passed, 9 skipped | 8 passed, 0 skipped |

Each pointwise inventory contains 525 ordered IDs: R's 519 tests plus six new
boundary tests, with one instrumentation test renamed. The eight imported
`tests.*` IDs are unchanged. Only L's exact nine two-device reservation skip
pairs were accepted; these runs do not establish two-device restoration.

[Raw development evidence](development.tar.gz) and its [manifest](manifest.json)
retain complete test IDs/results/logs, both native/default smoke histories,
build failures and successful receipts, source/wheel/installed/interpreter
inventories, recipes and the exact L-to-development recipe diff. The ordinary
144-line driver is unchanged and was not run. Full source, wheel, interpreter
and cache bundles remain in local `target/native-method-guard-trial/`; their
hashes/locations are recorded, but their bytes are omitted from the compact
archive and those local paths are not publicly durable.

## Qualification status

These are uncommitted development snapshots, not final correctness gates.
This worker leaves commits to Burner; no semantic T commit or later evidence-only
commit was made. The required six fresh C/R/T builds, frozen four-gate capture
and twelve ordinary legs remain unavailable. Zero of 28,944 ordinary samples
were collected. No T/R or T/C ratio, speedup, default-Inductor performance parity,
official score or qualification is claimed. A Burner-created T commit and a
fresh final capture are required before drawing those conclusions.
