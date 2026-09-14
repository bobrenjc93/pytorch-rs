# Clean capture at 4554bbe8: numerical blocker retained

Measured implementation: `4554bbe80adc2042f75293ed82c30d8c9cee685e`, against
`main` at `30a3b504ef4d43bf2958998cc39545996cc09970`. This capture supplies the
clean-commit measurements deferred in the [sibling-product repair record](../review-sibling-products.md).
The **structured-output numerical milestone remains incomplete**: the known
nonlinear signed-zero failure reproduced on this clean commit.

## Results

The unchanged committed checks ran against a newly built release wheel:

| Check | Passed | Explicit skips |
| --- | ---: | ---: |
| Python pointwise suite on H100 | 285 | 8 device tests |
| Dedicated two-physical-device checks | 8 | 0 |
| Release native pointwise and ownership | 42 | 0 |
| Release Python-conversion failure ownership | 1 | 0 |
| Portable pointwise suite, each CPython version | 143 | 150 hardware tests |

Portable versions were 3.10.19, 3.11.15, 3.12.12, 3.13.13 and 3.14.5. Hardware
skips are not GPU passes. The native-extension verifier passed. The pointwise
suite includes the 600 sibling-product differential comparisons and the 91
returned-product combinations. In total, 757 materialized native/reference
comparisons and their dispatched CUDA/PTX were retained, each with one native
kernel entry. Existing tests cover identity, freshness, current metadata, cache
reuse, helper/loop/branch composition and failure atomicity. The isolated
no-PyTorch-import/no-original-or-helper-replay check passed.

The unchanged diagnostic from the committed repair archive separately checked
`p=x*y; q=-p; r=p.sin(); return (q,r)` with positive float32 `x=y=1e-38`.
At shape `(1,)`, both implementations return `q=-0`. At shapes `(13,)` and
`(257,)`, native returns `q=-0` while ordinary stock `torch.compile` returns
`q=+0`. Both first and repeated calls fail: **4 of 6 comparisons failed**.
The original failing log, all six raw outputs, native CUDA/PTX and reference
compiler caches are retained. This is an unresolved admitted-program correctness
issue, not an infrastructure failure or a passing qualification. No assertion,
tolerance, implementation or external producer was changed to suppress it.

## Provenance and retention

[Measurements](measurements.json.gz) record actual clean before/after status,
all tracked source hashes, timestamped commands, cache state, interpreter/import
paths, wheel members, loaded libraries, GPU snapshots, successful checks and the
failed diagnostic. All tracked files stayed byte-identical throughout capture.
Every interpreter's installed Python package and extension matched the wheel.
Both frameworks used ordinary `compile(fn)` defaults.

Wheel SHA256: `9cd50382153d0d80b4df64283db66d375c48f2c7d98f086702d726070ec298e0`.

The locked offline release build used Rust 1.92.0 and Maturin 1.15.0, with a fresh
Cargo target and initially absent reference compiler caches. Ordinary H100 runs
used `CUDA_VISIBLE_DEVICES=0`; device checks used only `0,1`. Driver 580.82.07,
PyTorch 2.13.0+cu130, native NVRTC 13.0 and CUDA runtime 13000 were recorded.
PATH `nvcc` reported 12.6.85; NVRTC compiled the native kernels. Runtime probes
reused the report-local caches populated by the suite. No performance score or
cold-build latency claim is made.

The [raw archive](raw-captures.tar.gz) retains logs, raw numerical outputs,
generated code and orchestration scripts for the existing build/tests. The
[manifest](raw-retention-manifest.json) identifies verified wheel, committed-source,
reference-cache and nested-diagnostic archives under
`target/default-compile-eval/structured-outputs-postcommit-4554bbe8/`.
All 18,163 earlier evidence files, including original failures and archives,
were verified unchanged. Earlier clean captures retain their original provenance.
No disposable checkout was cleaned up. All writes stayed in this worktree;
Burner's canonical report-root observer owns external archival.

This step changes only evidence and its guide link. No implementation, tests,
dependencies, benchmark harnesses, evaluation definitions or managed progress
artifacts changed. No official score or unrelated full suite was rerun. The
remaining numerical blocker requires implementation work and independent review;
this clean capture does not approve the branch or complete the milestone.
