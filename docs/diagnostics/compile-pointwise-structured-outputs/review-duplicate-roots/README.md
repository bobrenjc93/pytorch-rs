# Duplicate observable-root repair

These are development checks of the uncommitted repair after
`245e91f00aa6e41ed6262b5f40f92ca686ffea38`, captured on September 14, 2026.
They do not qualify a clean implementation commit. A new clean capture remains
pending Burner's commit; earlier clean measurements retain their original identities.

Distinct original computations still own distinct output allocations and stores.
Numerical use analysis now counts each canonical observable value once, including
aliases introduced by unit-product normalization. Arithmetic operand uses and
observable output order are unchanged.

The new regression failed against the previously installed, verified `523d51d7`
wheel: duplicate positive products biased contraction toward the negative product,
giving `-168.0928955078125` instead of `+168.0928955078125`. That first failure is
retained unchanged. The repaired release wheel passes 720 strict comparisons:
five product constructions, three return topologies, sizes 1/13/257, eight value
histories, and first/repeated calls. Cases include overflow, subnormal values,
signed zeros, infinities and NaNs, with the existing tolerances and zero-sign
assertions. Separate allocations, repeated aliases and retained prior outputs
are checked in the same matrix.

| Check | Result |
| --- | --- |
| Native pointwise release tests | 59 passed |
| Full pointwise suite, one H100 | 297 passed, 8 two-device skips |
| Two-device restoration suite | 8 passed on physical devices 0 and 1 |
| Portable suite, each CPython 3.10–3.14 | 145 passed, 160 hardware skips |
| Clippy, formatting, native-extension verification | Passed |
| Runtime and installed-wheel identity checks | Passed |

The source-bound release wheel SHA256 is
`52006ac266223009825451a2e3592b92c423b7707fd908e28230ca6b8eb3a922`.
Builds used Rust 1.92.0, eight Cargo jobs, and the existing in-worktree Cargo
build cache. Reference caches were new for this repair and shared between its
successive checks. Ordinary `torch_rs.compile` and stock `torch.compile` used
their defaults. The reference was PyTorch `2.13.0+cu130`; the loaded CUDA runtime
was 13.0 and native NVRTC was 13.0. The available `nvcc` was 12.6.85. Driver
580.82.07 and physical GPU UUIDs are recorded. Native execution was also checked
before importing PyTorch.

[Measurements](measurements.json.gz) retain exact commands, timestamps, dirty
status, source hashes and exit codes, including the pre-repair failure.
[Raw evidence](raw.tar.gz) contains logs, numerical results, native CUDA/PTX and
instruction plans, runtime provenance, and the focused source review.
[The manifest](raw-manifest.json.gz) verifies all 13,839 archived files;
[verification](verification.json) records archive hashes and confirms all 52
pre-existing structured-output evidence files remain byte-identical.

The complete 48,111-file capture, including the wheel, source snapshot, reference
compiler caches, and byte-exact original reviewer archive, remains at
`target/default-compile-eval/structured-outputs-duplicate-roots-repair/` in this
worktree, under the canonical archival observer's report root. Its
`retention-manifest.json.gz` records file hashes. No cleanup was performed and no
external archival copy is claimed. These checks make no benchmark score or
performance claim; evaluation definitions and historical measurements are unchanged.
