# Operator-requested diagnostic repair

The [clean `bdcfb051` capture](postcommit-bdcfb051/report.md) now measures this
repair after commit. The development measurements and original failures below
remain pinned to their recorded sources.

This revision fixes reference-zero sign comparison, failure-side observation
retention, timeout stopping, and the public `compile()` scope documentation.
The approved zero-trip body admission repair remains intact. A separate narrow
frontend change removes redundant warm validation: the compiled wrapper checks
mutable signature containers once, reuses admission of the identical immutable
root code/constant pool, and still checks mutable namespaces, range/helper
bindings and tensor ABIs. Code replacement repeats full admission. Independent
private `resolve()` calls retain full validation.

## Capture contract and reproduction

The comparator requires the native sign to match whenever the reference is zero,
including native near-zero values within the unchanged `rtol=1e-5, atol=1e-6`.
It does not require native exact zero. The CPU reproductions establish a
comparator defect, not an observed incorrect CUDA result.

Before parity/dispatch assertions, each history saves its initial output and last
available returned output, input observations, timings accumulated so far, and
CUDA/PTX obtained from the receivers actually observed entering the native
executor. An execution failure saves available observations without replay;
an observation/export failure is recorded independently. This does not archive
every normal warmup/sample tensor return. A child timeout checkpoints available
reports and stops the entire campaign. Direct-child termination does not prove
compiler descendants exited; the existing execution owner must establish
quiescence before any further measurement. No signal/cleanup owner was added.

With the canonical GPU/CPU-heavy reservation, a fresh locally installed release
wheel and the worktree-local environment described in the original diagnostic:

```bash
.venv/bin/python -B docs/diagnostics/compile-pointwise-loops/compare.py \
  --output target/operator-recovery/comparison-NEW \
  --durable-raw-output "$APPROVED_RAW_PARENT/comparison-NEW" \
  --wheel target/operator-recovery/repaired-wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl \
  --devices 0,1
```

Both attempt directories must be new; the explicit durable parent must already
exist. Only raw observations/logs/reports/artifacts go there. Interpreters,
dependencies, wheels, builds, temporary files and CUDA/Inductor/Triton/XDG caches
stay under the checkout. The option contains no operator-specific path and adds
no evaluator/observer wiring. The matrix, persistent wrappers, ordinary default
Inductor, five warmups, 17 samples, synchronization, orders and tolerances are
unchanged. No favorable repeats were selected.

## Development evidence and limits

The repair was measured once on the reserved H100s 0 and 1 (UUIDs
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1` and
`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`) from dirty source on
`94c2ec053a173333b07c0d09089d6861723608aa`. The source manifest, installed-wheel
byte checks and build/extension hashes bind these observations to the repair;
they are **not clean-commit qualification**. Burner must commit the repair and
collect fresh clean evidence/canonical evaluations through the normal owner.
Historical captures at `6ef71d3a` do not measure this later frontend change.
The run used CPython 3.12.12, PyTorch `2.13.0+cu130`, driver 580.82.07,
NVRTC 13.0 targeting `compute_90`, CUDA runtime 13.0 synchronization and Rust
1.92.0 release builds. System `nvcc` 12.6.85 was recorded but did not compile
these generated kernels.

The complete focused pointwise Python suite passed 242 tests on the reserved
H100s, and all 34 native Rust pointwise regressions passed from the fresh local
build. Portable loop/helper/signature/diagnostic checks passed on CPython 3.10.19,
3.11.15, 3.13.13 and 3.14.5 (63 each, nine hardware skips); the main suite used
3.12.12. Initial portable attempts lacked the existing `typing_extensions`
dependency in the isolated wheel extraction. Those failures remain recorded;
copying the same installed dependency locally completed the setup, without
changing its version or any production dependency.

All eight independent native/reference legs and 60 paired histories passed,
including IEEE values, empty outputs, fresh values/offsets and shape histories;
native unequal-shape rejection remained intact. Ratios ranged from 0.591645 to
1.948725; native was slower in 36 of 60 histories. All samples, including slow
results, remain recorded. No original root/helper body or warm lowering was
observed. Executor-entry evidence is not a driver-level launch trace; empty
outputs do not prove a launch, and reference calls without attributable new
kernels remain explicitly unknown. Padding/prior-input storage-history coverage
is not exhaustive; this is a limitation, not an additional qualification blocker.

The hardware-free [warm probe](warm_profile.py) used the existing mocked native
test bridge, 5 warmup batches and 17 sample batches of 1,000 calls, followed by a
separate 100-call profile. Run it from the checkout with the matching wheel:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B \
  docs/diagnostics/compile-pointwise-loops/warm_profile.py \
  --output "$APPROVED_RAW_PARENT/NEW-ATTEMPT/warm-report.json"
```

The one before/after pair recorded signature checks decreasing from 200 to 100
per 100 warm calls and root-code scans from 100 to zero. Namespace/range checks
remained 100; emitted mocked graphs matched. Arithmetic medians were
21.535104/20.748845 microseconds per call (before/after), and literal-loop medians
21.906736/20.690917. These mocked host measurements neither establish GPU speedup
nor explain the earlier CUDA regression. Frontend SHA256 changed from
`1e5ca6378b7ddb737192b66ef4946d82853463b9d3c721f684a3b109224977c8` to
`2309690925aef68bcc2890202087190dfb83bb96e9d57431d5331fa2fc85302e`.

## Retained regression and raw-retention gap

The original canonical result remains unqualified: coverage 14.5 (14/112), CUDA
19.6 versus 20, polish 82 versus 83, and no full gate submitted. The retained
`run-20260914T013846Z-0d7ba2bb` report has exact CUDA score 19.613404424665234;
its SHA256 is `3b2a568cb2d79dbe6f6bf7c00a07ef2f77a0ef825e8a717eb0abbaeada279c80`.
Its affine variant-0 candidate/reference medians were 71.778/73.801 microseconds
in reference-first order and 92.800/69.455 in candidate-first order (ratios
1.028184 and 0.748438). That cell's capped geometric contribution accounts for
the arithmetic-category loss in this report. Other retained rounds vary too;
the accompanying extraction retains both orders and every sample for all
arithmetic/broadcasting cells across all four available candidate reports.
Neither a favorable earlier run nor the separate manual-main result replaces
the canonical result or its baseline.

The canonical report records Inductor generated-kernel counters and native
body-exclusion/import evidence, but no per-cell driver trace or native PTX
identity establishing a CUDA execution-time cause. For the *separate generic
loop diagnostic*, all 60 newly observed receiver CUDA and PTX hashes match the
decoded historical `6ef71d3a` capture. This is comparable kernel evidence for
those loop workloads only, not causal attribution for the canonical arithmetic
timings. No unchanged canonical candidate was resampled during this repair.

Every new corrected diagnostic run wrote directly beneath the operator-approved
durable parent:

`/tmp/burner-approved-default-compile.UA2M6I/static-loop-operator-recovery.iJS0Kv/diagnostic-outputs`

The distinct attempts are `warm-baseline-01`, `repaired-01` and `comparison-01`.
They retain build/test logs, observations, compressed reports and actual observed
CUDA/PTX; environments/caches remain worktree-local. The supplemental
[recovery archive](operator-recovery-captures.json.xz) and
[manifest](operator-recovery-manifest.json) preserve 169 available raw files,
including failures, with their actual compressed-byte hashes. Exported bytes were
verified against the durable originals. They do not replace those originals.

The earlier late-raw retention gap remains: original `comparison-v4`,
`postcommit-comparison`, `revision-comparison`, `postcommit-logs`, `revision-logs`
and `review-fix-logs` were unavailable after canonical cleanup. Checked-in decoded
exports and declared hashes are not proven original compressed bytes. No missing
historical raw file was recovered or relabeled. The unchanged `f0747ea1` observer
covers usual canonical default-evaluator reports, not these diagnostic or nested
development outputs; this explicit destination is the operator-authorized
retention exception, not expanded observer coverage.
