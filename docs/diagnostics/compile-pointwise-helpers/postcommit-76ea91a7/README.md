# Clean evidence after the warm-admission fix

These measurements used clean commit
`76ea91a7184005205b704c6b11f4ee68f140000d` on September 13, 2026, including the
review fix for ignored helper arguments on warm cache hits. All measurements
finished before this evidence-only change. They supersede the earlier candidate
captures at `a8211b07`; those reports and the development failures remain intact.

The existing [clean-main report](../postcommit-a8211b07/main-fixed.json) still
measures current main, `fe7537251a53b54e1064309d28db2631605fb720`. Its clean checkout,
source files, wheel, imports and raw worker reports were revalidated inside the
current worktree. Main and the frozen evaluator/corpus are unchanged, so its
original timings are reused without relabeling them as a new measurement.

## Measured results

| Check | Candidate `76ea91a7` | Main `fe753725` |
| --- | --- | --- |
| Fixed public-default-compile-v2 coverage | 12.5%; 12/112 cells passed | 11.0%; 10/112 cells passed |
| Fixed CUDA performance score | 20.0%; 12/56 cells passed | 20.0%; 10/56 cells passed |
| Common-success geometric reference/native latency ratio | 1.155967 | 1.283077 |
| Gate validity | Valid | Valid; existing capture revalidated |
| Two-H100 helper diagnostic | 8 successful legs; 64 paired histories passed | Helper domain unsupported |
| Focused helper tests, GPUs 0 and 1 visible | 30 passed | Not repeated |
| Physical GPU 1 hardware repeat | 4 passed; 1 explicit two-device reservation skip | Not repeated |

The only fixed-corpus pass/fail changes against main are the two CUDA
`python_helper` cells. Their reference/native median latency ratios were
**0.958638 and 0.943103**: both candidate variants were slower. Every failure,
unsupported cell and timing remains in the report. The full denominators and
14 category weights are unchanged. Unsupported cells keep the custom-functions
category's geometric performance contribution at zero, despite the helper's
coverage gain. These are fixed-corpus scores, not percentages of all Python
programs. Common-success ratios use different success sets and are not a paired
speedup claim.

The non-scoring helper diagnostic exercises multiple/composed calls, identity
and runtime-scalar forwarding, scalar-literal returns and the unchanged frozen
helper factory. Each wrapper persists through four shape/value/alias states.
Both framework orders run in separate fresh processes and framework caches on
each GPU, with five warmups, 17 synchronized samples and fixed `rtol=1e-5`,
`atol=1e-6`, including NaN/infinity/signed-zero checks. All 64 histories passed;
individual reference/native median ratios span **0.655829–1.297494**. Native
cold calls include body-execution policing, while steady timings are unprofiled.
No reference-only reset, compiler-limit change or favorable-repeat selection was
used. The focused tests additionally verify the warm global/closure argument
rejection, unchanged caches and valid ignored-value replacements.

## Source, build and hardware

Burner's canonical `gpu` and `cpu-heavy` locks remained owned by `idea_1f7d364a`.
GPUs 0 and 1 were explicitly reserved for helper checks; only GPU 0 was visible
for the fixed gate. Their before/after inventories and resource record are
archived:

- GPU 0: `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`
- GPU 1: `GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`

Both are H100s on driver 580.82.07. Python was 3.12.14 and reference PyTorch
`2.13.0+cu130`. Both sides used the worktree-local CUDA 13.0 runtime for device
synchronization. Native kernels used NVRTC 13.0, `compute_90`, `--fmad=true`,
`--ftz=false`, `--prec-div=true` and `--prec-sqrt=true`. System nvcc 12.6.85 was
recorded but did not generate these JIT kernels. Rust was 1.92.0.

The unchanged evaluator wrapper built and installed a fresh locked release
wheel from the clean source. Cargo/dependency caches were warm; this was not a
clean-target Rust rebuild. Setup timestamps and cache state are preserved in the
report. Checkout Python files, wheel contents and installed native/Python files
were byte-verified. Native diagnostic processes blocked importing PyTorch.

- Frontend SHA256: `74f9575e660b69e50c855bd342c826bc7b39a1e5c568094918a51b69c5574641`
- Evaluator source manifest SHA256: `926ebe43bbfc2b9acb3f405205558c11fbb2a2b9d85c048bb2569272ccef9dd6`
- Wheel SHA256: `1459562e7f43a844339a4100cf601c8f4cfbf8aafa87413340685b33bf95854a`
- Imported native extension SHA256: `0fdb7ccb205ece5477f928b4cab896041066c2ff5a26b1914eeb4d5ece9ea697`

From the clean checkout with resources held, using the local environment/cache
settings in archived `target/postcommit-76ea/env.sh`:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric both --output target/postcommit-76ea/candidate-fixed.json
CUDA_VISIBLE_DEVICES=0,1 python docs/diagnostics/compile-pointwise-helpers/compare.py \
  --output target/postcommit-76ea/helpers \
  --wheel target/default-compile-eval/wheels.srNdWW/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl \
  --devices 0,1
CUDA_VISIBLE_DEVICES=0,1 python -m unittest -v tests.test_compile_pointwise_helpers
CUDA_VISIBLE_DEVICES=1 python -m unittest -v tests.test_compile_pointwise_helpers.HelperHardware
```

The wheel directory above is the actual recorded build; use the new directory
produced by the wrapper for a reproduction. The archived command records include
UTC timestamps and clean status before/after every measurement. No implementation,
test, dependency definition or benchmark harness was changed in this step.

## Retained evidence and handoff

[candidate-fixed.json](candidate-fixed.json) is an unedited copy of the generated
report. [captures.json.xz](captures.json.xz) retains all helper reports and
CUDA/PTX, build/test/command logs, source manifest, resource record and audit
procedure. It contains 182 files in a `files` mapping with `text`,
`original_sha256` and `uncompressed_sha256`; gzip inputs are decoded losslessly.
[audit.json](audit.json) records source/wheel/import/raw-report checks and the
exact two-cell pass/fail delta. [SHA256SUMS](SHA256SUMS) covers this evidence set.

[raw-retention-manifest.json](raw-retention-manifest.json) identifies the original
six new candidate worker reports and six unchanged baseline worker reports,
with their logs, exact worktree-local paths and verified hashes. They remain in
their run directories. The earlier candidate's raw reports remain covered by
its original manifest.

**The external observer receipt remains unverified.** The existing separately
reviewed campaign-scoped observer must preserve detailed frozen-worker reports
before ordinary worktree removal. This worker did not alter or replace that
observer and does not claim its export succeeded. These captures do not replace
independent review, exact-head qualification or Burner's publication/merge gates.
