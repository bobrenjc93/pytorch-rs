# Default compile input admission experiment

The required clean-commit captures are now recorded in
[postcommit-32105a22](postcommit-32105a22/README.md): candidate `32105a22` and
accepted main `60202557`, unchanged full corpus and both CUDA orders. Coverage
is unchanged; the CUDA score increase is only 0.0708 points and does not establish
a reliable benefit. The development records below retain their original identities.

Author correctness evidence, 2026-09-17, based on accepted main
`60202557b4f110d07777f585e804ab5f55e1ff7b`. This is an uncommitted implementation
check, not a score or a performance result. No historical measurement is used
as candidate evidence. Burner must perform the unchanged paired default-compile
measurement and qualification before deciding whether this experiment helps.
No extra scoring run, private timing shortcut or parameter tuning was performed.

Production changes are limited to `_compile_pointwise.py`, `python_pointwise.rs`
and `python.rs`. `_pointwise_admit_inputs` borrows the exact complete input tuple,
projects copied metadata through tracing's existing six-field projection,
preserves the any-CPU diagnostic, and calls the canonical whole-input validator.
It returns only after successful admission. The old validation bridge still
returns a device ordinal; tracing still admits CPU/noncontiguous metadata.
Preparation and launch retain their independent current-input validation. The
single borrow scope does not make subsequent preparation atomic.

Portable fixtures now intercept the coalesced boundary and supply explicit
metadata readers to fake preparation. The scalar fixture's pre-existing swapped
requires-grad/dtype fields are corrected and compared with real native metadata.
The new tests compare direct old-API results, field values and exact types; they
check every flattened tensor occurrence on warm calls, unused inputs, repeated
inputs, scalar/tensor ABI transitions, private exports, CPU error precedence,
failed admission/metadata allocation, cache order/accounting and recovery/reset.
Mock call counts establish crossings only. Real GPU tests supply correctness
evidence separately. Float32 is the sole native dtype, and public CUDA gradient
creation is unsupported; the unchanged canonical validator's gradient, element
count, offset/storage and layout rejection is exercised by the Rust tests.

## Outcomes

| Command group | Complete outcome | Log |
| --- | --- | --- |
| Final release setup-only | Passed build, wheel install and native provenance | [setup-final.log](setup-final.log) |
| Focused Python (portable cases plus 3 direct GPU tests) | 106 passed | [focused-final.log](focused-final.log) |
| Rust pointwise | 72 passed, 177 filtered | [rust-pointwise.log](rust-pointwise.log) |
| Existing GPU correctness | 15 passed, 2 explicit two-device skips | [gpu-correctness.log](gpu-correctness.log) |
| Tracing regressions | 63 passed | [tracing-regression.log](tracing-regression.log) |
| Formatting and diff whitespace | Passed | [fmt-final.log](fmt-final.log) |

GPU correctness uses ordinary native CUDA tensors and existing default-Inductor
comparisons for one/two inputs, aliasing, fresh storage/changed values, offsets,
rank-zero/length-one/empty tensors, broadcast histories, original-graph admission,
prepared-shape mismatch, launch rejection, failure recovery and reset. Existing
no-body-replay checks remain active. No test assertions or tolerances were relaxed.

## Reproduction and identity

[commands.sh](commands.sh) records the actual setup and test commands;
[env.sh](env.sh) places interpreters, dependencies, wheels, temporary directories
and writable caches inside the worktree. Python 3.12.14 and pinned dependencies
were copied read-only from an existing checkout after the download failure.
The unchanged wrapper ran locked dependency synchronization and rebuilt/replaced
the copied native package with this source's release wheel. Nothing was installed
into the source checkout or home directory.

[identity.json](identity.json) records final production source, lockfile, wheel
and installed-extension hashes. Installed frontend bytes and wheel frontend
bytes were checked against this checkout. [environment.json](environment.json)
records the local interpreter/import paths, PyTorch `2.13.0+cu130`, CUDA runtime
13.0 and Rust 1.92.0. Generated kernels reported NVRTC 13.0; native Python storage
loaded the local wheel's CUDA 13 runtime. The installed `nvcc` reports 12.6.85;
this native release build does not invoke nvcc. Build configuration is the
repository's unchanged release profile, extension-module feature, four Cargo
jobs and one host thread for reference execution.

[resources.json](resources.json) is a read-only snapshot showing Burner holds
canonical `gpu` and `cpu-heavy` locks for `idea_7ba378d2`. Every GPU command used
`CUDA_VISIBLE_DEVICES=0`. H100 GPU0 is
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, capability 9.0, driver 580.82.07.
Before/after inventories are retained as CSVs; they are observations, not locks.
Only GPU0 was used, so two-device checks were explicitly skipped.

## First failures retained

- [setup.log](setup.log): Python download failed through the proxy after retries.
  [setup-retry.log](setup-retry.log) records successful local-copy recovery.
- [fmt-first.log](fmt-first.log): rustfmt requested wrapping the new CPU predicate;
  formatting was applied and the final wheel rebuilt.
- [focused-first.log](focused-first.log): the new tests initially assumed native
  Tensor subclassing/weakrefs and public CUDA gradient mutation existed, and
  expected two lowerings despite a distinct repeated-input ABI. Corrections use
  exact-type rejection, refcount non-retention, existing gradient coverage and
  the actual three ABI lowerings. Existing regression tests passed.
- [focused-before-install.log](focused-before-install.log): a test invocation
  overlapped setup's package removal/reinstall and failed imports. Final tests
  ran only after successful installation.
- [focused-second.log](focused-second.log): another new fixture tried the absent
  `float64` API. The dtype enum has only Float32. The test now exercises supported
  CPU-gradient and CUDA-layout invalid inputs instead of inventing a dtype.

Original local artifacts remain under `target/admission`; the copies here
preserve complete logs. Evaluator scripts, denominators, references and benchmark
evidence are unchanged. Burner owns final evaluation artifacts exported through
`BURNER_EVALUATION_ARTIFACT_DIR`, collection, scoring and qualification; those
phases have not run in this implementation turn.
