# Signed-product use accounting repair

Development measurements after `cba212e3e32df1a2cfaac266f5fe67f7a20d9765`,
with the source changes retained in the archive. This is an uncommitted repair,
not a clean-commit capture or an evaluation score. Burner must commit the repair
before fresh clean-commit qualification. Earlier captures remain unchanged.

Sign extraction now retains relationships between sign-equivalent expressions
for FMA priority. Each basic block counts their current external uses, excluding
internal wrapper edges. Value CSE, realization boundaries, allocation slots and
stores are unchanged. This also handles negative coefficients other than -2.

The permanent differential regression uses ordinary `torch_rs.compile(fn)` and
ordinary PyTorch 2.13 `torch.compile(fn)`: five coefficient spellings, all six
return orders plus a Tensor-only control, sizes 1/13/257, five finite/overflow/
underflow/zero/nonfinite input pairs, and cold/warm calls (1,050 comparisons).
It retains prior outputs and uses the existing tolerances and zero-sign checks.
A native regression checks the actual contracted factors and duplicate output slots.

The initial installed-candidate reproduction failed 72 comparisons; the expanded
coefficient matrix failed 270. Both complete failure logs and all their raw
outputs are preserved separately from the successful repaired captures. For the
reported finite shape-13 input, repaired c is 2384.185791015625, matching the
reference; the overflow case returns NaN on both sides.

## Checks and provenance

- H100 pointwise suite: 304 passed, eight explicit two-device skips.
- Separate H100 two-device checks: eight passed.
- Release native pointwise checks: 64 passed.
- New differential matrix: 1,050 comparisons passed, both alone and in the full suite.
- CPython 3.10–3.14: 18 portable structured-output tests passed per interpreter;
  24 hardware cases skipped per interpreter. Installed wheel identities verified.
- Cargo formatting, Clippy, and native-extension verification passed.

Reference: PyTorch `2.13.0+cu130`; H100, driver `580.82.07`. The native
compiler selected NVRTC 13.0 and CUDA runtime 13000. Toolchain capture also
records the installed NVCC 12.6 and Rust 1.92.0. Release builds used locked,
offline dependencies and worktree-local build/cache paths.

The final source-bound release wheel and the earlier repair wheel have identical
Python/native-extension members. The intervening source edit added only the
native test. Both original wheel files are retained. Runtime verification checks
installed members against the final wheel and source, records actual loaded
CUDA/NVRTC libraries, and executes native compilation before importing PyTorch.
Normal CUDA checks use physical device 0; only the eight restoration checks use
0 and 1. Portable hardware skips are not CUDA passes.

[Measurements](measurements.json.gz) record commands, timestamps, environment,
source hashes, commit and actual dirty status. [Raw captures](raw-captures.tar.xz)
retain every report file, including original failures, wheel files, source
snapshot/diff, numeric JSON, native CUDA/PTX and plan diagnostics, reference
compiler caches and independent review conclusions. Plan files reconstruct the
validated plan for inspection; they are not dumps of the launched GPU buffer.
Only rebuildable Cargo output is excluded.

[Raw manifest](raw-manifest.json.gz) and [verification](verification.json) record
byte-exact archive checks. The archive is stored inside this worktree; no external
collector coverage or external retention writes are assumed. The working capture
root is `target/default-compile-eval/structured-outputs-signed-uses-repair`.
