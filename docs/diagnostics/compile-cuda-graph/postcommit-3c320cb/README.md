# Clean composite evidence: `3c320cb2`

Measured code commit: `3c320cb2e36ae1f165437e095ecf69cbcd1f59f4` (base `fbb0aa0`). This is a new empty-target,
clean-commit build and complete diagnostic sequence for PRs #1959/#1960/#1961.
No implementation, test, dependency, workload, evaluator or managed progress
artifact changed during this evidence step. Source, test and harness hashes
match the committed tree at capture start/end; production identity and clean
status are checked before and after every recorded command.

The path remains native `backend="eager"` capture through one execution bridge,
reusing native kernels/cuBLAS. It is not general Inductor, training or hardware
parity. These reports do not establish repeatable native acceleration.

## Complete declared sequence

[Initial inspection/declaration](inspection.json) predates all measurements.
The committed diagnostic kept all 12 cells, both orders, 10 warmups per order,
31 five-call samples per order, synchronization, first-call observations,
untimed correctness/body/forwarding checks and zero-credit failure accounting.
No slow cell was removed and no favorable-only rerun was performed.

| Run | Seed | Passed cells | Capped diagnostic parity | Raw calls |
| --- | --- | --- | --- | --- |
| primary | 798431 | 12/12 | 80.0976% | 7,440 |
| held-481723 | 481723 | 12/12 | 76.2867% | 7,440 |
| held-926051 | 926051 | 12/12 | 75.7168% | 7,440 |
| primary-repeat | 798431 | 12/12 | 78.8315% | 7,440 |

[Primary](primary.json), [held-out 481723](held-481723.json),
[held-out 926051](held-926051.json), and [primary repeat](primary-repeat.json)
retain all raw samples and failures. [Verification](verification.json) checks
all 29,760 individual calls, 192
first-call observations, block/pooled medians, caps, input hashes, and the
unchanged per-seed cells/policies against the historical source reports.
The table is diagnostic output, not a Burner score or evidence of a gain.

**Timing limitation:** an unrelated workload occupied GPU 0 during this
capture window. The before/after GPU inventories retain the observed processes
(5 nonempty snapshots). No claim of isolated performance
non-regression, CPU-binding-induced CUDA improvement, or repeatable acceleration
is made. These results are retained without replacement. Burner admission
state was not modified; this evidence step cannot guarantee external GPU
exclusivity. Controlled performance gates remain the coordinator's responsibility.

## Focused verification

- [Installed wheel](wheel-verifier.log): isolated `-I` verification passed;
  the audit also checked every wheel Python source and both native copies.
- [Clippy](clippy.log) and [Python-binding Clippy](clippy-bindings.log): exact
  all-targets `-D warnings` commands passed. The source PR #1960 job
  `34511502296 / 102986492020` remains a historical failure; these are new runs.
- [Combined CPU differentials](focused-cpu.log): 67 tests passed, including
  division/L1 weighting, either/both/shared operands, views, IEEE cases,
  accumulation and graph release.
- [H100 CUDA regressions](cuda-regressions.log): 33 tests, 30 passed and three
  two-device skips. [Separate GPUs 0,1](two-gpu.log): all seven tests passed,
  covering restoration and mixed-device rejection as well as graph guards.
- [Fixed compiler corpus](frozen-compiler.log): 38/38 eligible cases.
- [Fixed CUDA math](fixed-math.json): 6/6 cases,
  all 18 prescribed-seed trials passed.
- [Fixed four-shape workload](fixed-scoring.json):
  4/4 eligible shapes,
  capped aggregate 100.0000%.
  This unchanged private workload is separate from the graph diagnostic;
  its measured output supplies no acceleration claim for the CPU changes.

Unrelated full Rust/Python suites already run by the author were not repeated.

## Build, provenance and cache state

[Build receipt](build/build-record.json): clean status, absent target,
locked/offline release wheel, then installation into this worktree's `.venv`.
The measured implementation source SHA-256 is `5349cb16843b1f26de92c30516af7502f0a7ce52054377955ad4fbe9499c3aed`;
native extension SHA-256 is `1e271f0d0e6c52ad77179c6909514192e8bb655727779c8b6703ba9acc2fb6bf`.
[Source/test/harness manifest](source-manifest.json) and
[implementation diff](implementation.diff.txt) bind the complete included code.

Worktree-local CPython 3.12.14, NumPy 2.5.1 and
PyTorch 2.13.0+cu130 were reused. H100, driver 580.82.07; GPU 0 except the
separate two-GPU test. Native runtime reports CUDA 13.0 from local nvidia/cu13
packages. Rust 1.92.0, release, thin LTO, one codegen unit, extension-module/abi3.
Native matmul uses cuBLAS and pointwise kernels use driver-JIT PTX. nvcc
12.6.85 is recorded; the separate scoring lane uses nvcc-built private kernels.
The verification includes actual current interpreter, wheel, extension,
CUDA library, PyTorch native-library and reference Triton/PTX assembler hashes.

[Launch/setup record](launch-command.txt), [environment](environment.json),
and [orchestration source](command-sources/capture.py.txt) describe the new
build/temporary/CUDA/Triton/Inductor directories and reused local package/Cargo
caches. Correctness checks warmed CUDA before timing. Diagnostic compiler
caches start empty and are reused across the declared sequence; each report
records its initial counts. Scoring uses separate initially empty compiler
caches and existing local private-kernel binaries, whose source checksums,
actual hashes and reuse flags are preserved in verification.json.

[Commands](commands.json) and individual receipts record actual argv,
timestamps, exit status, clean status, source identities and log hashes.
[Capture completion](capture-complete.json) records unchanged sources.
[Artifact hashes](artifact-hashes.json) bind published bytes to local originals;
recorded paths retain their actual current-composite identities. No provenance
was rewritten to the publication location.

## Historical evidence and remaining gates

All 81 checked-in files in
[the historical source publication](../postcommit-786c1b2/README.md), including
the original `fbb0aa0` baseline captures, remain byte-for-byte unchanged:
[historical preservation audit](historical-preservation.json).
Only published blobs/current copies were checked; deleted historical runtime
binaries were not claimed to have been freshly rehashed.

Historical source primary/repeat parity was 81.4367% / 80.0108%; composed native
ratios versus actual main were 1.0609x / 0.9826x, and held-outs 1.0054x / 1.0036x.
Those verified pairs do not establish repeatable native acceleration. No fresh
baseline was needed or manufactured for this evidence-only, no-gain claim.
The earlier dirty composite runs are not substituted for this clean capture.

This publication does not approve the branch. Independent exact-head review,
all ten current-definition non-regressing evaluation gates and exact-head CI
remain required before Burner's managed merge/cleanup. Source associations
remain intact. No branch, commit, push, PR, admission setting or external tool
was changed by this evidence step.
