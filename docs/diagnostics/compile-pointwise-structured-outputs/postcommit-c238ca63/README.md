# Clean structured-output capture: c238ca63

Measured code commit: `c238ca63736c65d218253fea01f5301c07755e6f`.
Base: `30a3b504ef4d43bf2958998cc39545996cc09970`.
The checkout was clean before and after every measured command. Full tracked-file
fingerprints and compiler/test source hashes matched throughout the capture.
Only evidence and documentation were added afterward.

This captures the committed signed-product use repair and the complete bounded
structured-output pointwise test matrix. It is a correctness/provenance capture,
not a fixed-corpus evaluation score, performance claim, or waiver of independent
review. Earlier development failures and clean captures remain unchanged.

## Results

- H100 pointwise suite: 304 passed; eight two-device cases explicitly skipped.
- Separate two-device/restoration checks: eight passed.
- Release native pointwise tests: 64 passed.
- Python conversion-failure ownership test: one passed.
- CPython 3.10–3.14: 149 portable tests passed per interpreter, with 163
  hardware cases skipped per interpreter. Skips are not GPU passes.
- Installed-wheel identity, native-extension validation, formatting and Clippy passed.

The signed-product regression has 1,050 comparisons: five coefficient spellings,
six tuple orders plus a Tensor-only control, three sizes, five IEEE input pairs,
and cold/warm calls. Existing strict tolerances and zero-sign assertions remain
unchanged. Its raw finite example returns 2384.185791015625 on both implementations;
the overflow example returns NaN on both. The full suite also covers retained
outputs, executor reuse across topology/order changes, realization history,
input aliases/current metadata, admission and failure atomicity.

## Build, runtime and retention

Wheel SHA256: `a21bc2524bdf415caa1726c4dadfdc905c24f68b1c3355b8030e2b3be0d8e287`.
All five interpreters' installed Python/native-extension members match the wheel.

The source-bound release wheel was built offline with locked dependencies in a
fresh worktree-local Cargo target directory. All installed Python/extension
members are checked against that wheel; Python sources are also checked against
the checkout. Native execution is exercised before importing PyTorch. The
reference is ordinary PyTorch 2.13.0+cu130 torch.compile, without compile overrides.
Runtime provenance records H100 UUIDs, driver 580.82.07, selected NVRTC 13.0,
CUDA runtime 13000, compiler options and loaded-library paths. Toolchain capture
records NVCC 12.6 and Rust 1.92.0. Normal hardware checks use device 0;
only the eight device/restoration checks expose devices 0 and 1.

[Measurements](measurements.json.gz) retain exact commands, timing, environment,
source hashes, commit, and clean-status observations. [Raw captures](raw-captures.tar.xz)
retain the original report files, wheel, committed source snapshot, numerical JSON,
CUDA/PTX, reference compiler caches, logs, and reconstructed plan listings. A plan
listing is a diagnostic reconstruction, not a dump of the launched GPU buffer.
Only rebuildable Cargo output is excluded. The initial packaging helper failed
on source-tar directory entries; its original source and terminal output are
retained beside the corrected packaging helper. Measurement commands all passed. [Raw manifest](raw-manifest.json.gz)
and [verification](verification.json) record byte-exact archive checks.

The raw archive is included in this worktree for delivery; no external observer
coverage is assumed. Working report root:
`target/default-compile-eval/structured-outputs-postcommit-c238ca63`.
These measurements apply to the recorded commit, not later implementation changes.
