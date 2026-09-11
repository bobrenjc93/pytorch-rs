# CUDA compilation observer campaign v1

This is an evaluator-only campaign. Main production remains byte-identical to
`563d55960332a9a09c5825da22e4212c053ce374`. The only change to the existing
evaluator is `native_hook` mapping `sum` to `_compile_trace_reduction`, the real
native reduction entry point. There is no substitute unary execution, fallback,
new acceptance path, workload change, or implementation credit on this branch.
The evaluator, common helper and entire hardware matrix match historical
candidate publication `b3659e76011239388da710d4de2f30c52018016f` byte for byte.

The [versioned manifest](../scripts/campaigns/cuda-compilation-observer-v1.json)
pins those three SHA-256 hashes, the complete six-case definition, compile
options, tolerances, two seeds, GPU0 UUID, baseline and candidate production
file digests, ownership and reproduction commands. Candidate production is
`e9adfdca71626190a17e36544af97b1066499bbe`; `b3659e7` is its evidence publication.
Neither historical commit is modified by this campaign. The broader frozen
38-program compiler corpus, private four-workload timing suite, hardware
weights/backends and Burner-managed progress are unchanged.

The seeds `8454740352132840342` and `4006623364652518084` were independently
drawn with `secrets.randbits(63)` at `2026-09-10T23:29:20.835412+00:00`, before
running either revision. They were not selected from candidate results. Both
revisions use the same unmodified `random.Random(seed)` float32 uniform input
distribution, fixed shapes, and changed-input seed XOR mask. Now published,
these seeds establish repeatability, not continuing secrecy or broad held-out
shape coverage. Every slot remains in the denominator, including unsupported
baseline row sum. Both initial and changed inputs must pass through the same
compiled wrapper, with real native returns, no Python body execution, and no
re-lowering on the changed input.

## Reproduction and clean-commit handoff

The required clean-commit refresh is now captured at
[`postcommit-10b4cc76`](diagnostics/cuda-compilation-observer-v1/postcommit-10b4cc76/campaign.json).
It measures evaluator-only commit `10b4cc76f1bf211714e7588b2b92790895a1d6a6`
against the pinned historical candidate, with both checkouts clean before and
after execution. Development records below remain unchanged historical evidence.

Portable checks (no GPU or installed torch required):

```bash
python3 -B -m unittest discover -s tests -p test_cuda_compilation_evaluator.py
python3 -B -m unittest discover -s tests -p test_cuda_observer_campaign.py
python3 -B scripts/cuda_compilation_campaign.py validate \
  --evidence docs/diagnostics/cuda-compilation-observer-v1/postcommit-10b4cc76
```

To reproduce from the committed evaluator-only code, use a new output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -B scripts/cuda_compilation_campaign.py run \
  --baseline-commit 10b4cc76f1bf211714e7588b2b92790895a1d6a6 \
  --output target/cuda-observer-final
python3 -B scripts/cuda_compilation_campaign.py validate \
  --evidence target/cuda-observer-final
```

The runner requires a new output directory and creates separately owned detached
Git checkouts inside it. It never uses `git worktree`, creates commits or branches,
modifies the parent object store, or materializes `.burner`. It fetches objects
read-only from the current repository. It does not access `agent_468d83fa`.
Final mode requires the campaign files committed in the baseline checkout,
unchanged main production, matching observer hashes and clean status. Candidate
files are never overlaid. A mismatch stops the run. Baseline development mode
is explicit: it copies only campaign-owned files into a main checkout, records
that dirty status and cannot emit a clean-commit label.

On this host the runner needs `/usr/bin/python3.12`, `uv`, installed Rust 1.92.0,
CUDA driver tools and access to the dependencies pinned by the unchanged
`uv.lock` and `Cargo.lock`. It creates each `.venv` with `venv --copies`, installs
locked dependencies locally with copy mode, and builds a fresh release wheel in
each checkout with fresh Cargo caches. Python executable files resolve inside
their own `.venv`; the system standard library/base prefix is `/usr`. The two
checkouts own distinct dependency trees and wheels. The wheel's Python files
must match source; its extension is copied into that checkout's ignored source
package location because the pinned observer explicitly requires that location.
No editable-install pointer is used.

The selected native runtime and cuBLAS come from each local CUDA 13 dependency
installation. Raw worker reports retain the mapped runtime paths and queried
versions. The native pointwise/reduction path uses embedded PTX and driver JIT;
matmul uses native cuBLAS. `nvcc --version` records the installed CUDA 12.6 tool,
which is not used to compile these evaluator workloads. Both builds use Rust
release configuration (thin LTO, one codegen unit), identical lockfiles and one
OMP/MKL/OpenBLAS thread. GPU0 UUID, memory usage and utilization are recorded
before/after each build-and-execution sequence. Snapshots are observations, not
exclusive GPU reservations. No other user's jobs are interrupted.

Retain each run's `baseline/`, `candidate/`, `selection.json`, `campaign.json`
and `validation.json` in a new publication directory. These contain all worker
results, full raw input/output values, rejection-control logs, build logs,
commands, dependency/runtime identity and before/after status. `campaign.json`
binds the retained files by hash; portable validation recomputes six-slot
accounting and numeric comparisons rather than trusting the reported total.
Publish the complete directory without replacing prior attempts. Builds and
virtual environments remain ignored under `target/`; wheel and extension
hashes, fresh build commands and logs remain in the publication.

The post-commit refresh rebuilt and re-compared both revisions using the
committed runner without `--development`. The runner records the measured
commit explicitly in its receipt, avoiding a self-referential hash in the
manifest. The subsequent publication changes only evidence and its
documentation; implementation, dependencies, tests and harnesses remain at
`10b4cc76`. Development results were not relabelled or overwritten. Independent
review and normal gates remain separate requirements.

## Review and measurement boundaries

The author inspected the historical root findings and candidate raw-evidence
audit supplied for context. That audit and the historical source review applied
to the earlier implementation publication, not this new campaign. No independent
review of this campaign and no human campaign approval have occurred during this
task. Author-run tests and raw-value verification are automated validation, not
independent or human review.

[BENCHMARKING.md](../BENCHMARKING.md) requires: “Benchmark changes are separate,
human-reviewed campaign changes and never earn implementation impact in the
same comparison.” Human campaign approval remains required, together with
independent review and normal gates. The clean-commit refresh is complete; it
does not approve the branch. This document does
not fabricate approval or amend that policy.

This correctness-only campaign does not run a performance benchmark. Latency
samples, medians, dispersion, warmups, timing speedups and performance credit
are not applicable. Command elapsed seconds describe installation/build/test
processes only; they are not workload latency samples. Candidate success is
historical comparison evidence and does not establish baseline row-sum support
or hardware implementation credit for this evaluator branch. Timing wording
and guide/navigation repairs in the historical implementation are separate
integration work and remain outside this branch.

## Observed development results

The [complete paired capture](diagnostics/cuda-compilation-observer-v1/development/campaign.json)
completed on GPU0 with driver 580.82.07, local PyTorch `2.13.0+cu130`, queried
CUDA runtime version `13000`, Rust/Cargo 1.92.0 and installed nvcc 12.6.85.
The [portable audit](diagnostics/cuda-compilation-observer-v1/checks/portable-validation.json)
recomputed these results from the raw records:

| Fixed slot | Main production | Historical row-sum candidate |
| --- | ---: | ---: |
| Same-shape add | 1 | 1 |
| Trailing-vector add | 1 | 1 |
| Negation | 1 | 1 |
| Literal multiply | 1 | 1 |
| Row sum | 0 | 1 |
| Matrix product | 1 | 1 |
| Total | **5/6** | **6/6** |

There are 12 trials per revision, each retaining its separate reference and
native worker result. All 24 reference workers passed. Main's two sum workers
failed with `AttributeError` for the absent `_compile_trace_reduction`, before
execution; the other 10 native workers passed. All 12 historical-candidate
native workers passed. The audit compared 1,596 baseline and 1,624 candidate
native output values, including changed-input execution. All passing native
workers showed the required genuine hook returns, no original-body execution,
no changed-input re-lowering, no blocked torch imports and no loaded torch
modules. All raw inputs, outputs, errors and compilation observations remain
in the reports; none of the failed sum slots was skipped.

Both source/evaluator snapshots stayed unchanged during capture. Baseline
status explicitly records the uncommitted campaign overlay; candidate status
is clean before and after. The local interpreter binaries, dependency versions,
native CUDA runtime and cuBLAS library hashes match across revisions. Native
wheels were independently built and retain distinct source-bound hashes.
The [host context](diagnostics/cuda-compilation-observer-v1/checks/host.json)
records OS and CPU information during the comparison.

All 17 existing accounting/isolation rejection controls passed with each local
interpreter, including malformed records, forwarded imports, fake/old hooks,
original-body calls and re-lowering. The portable CUDA evaluator suite passed
47 tests with four explicit hardware-only skips; the new campaign suite passed
nine tests without skips. Hardware correctness comes from the real paired run,
not those portable skips. Ruff and `git diff --check` passed.

Controller snapshots in each raw capture preserve the harness/test versions
actually used. Subsequent portable hardening adds explicit local uv-environment
selection, excludes inherited Rust cache wrappers, and cross-checks interpreter
and CUDA library hashes; it also validates the publication and rejected-attempt
inventories. The final validator passed against the unchanged raw capture.
These changes do not alter any pinned observer bytes or workload. They are
included in the committed runner used for the separate post-commit refresh.
No external Burner producer or installation was edited for that handoff.

## Clean-commit results

The [new capture](diagnostics/cuda-compilation-observer-v1/postcommit-10b4cc76/campaign.json)
uses baseline code `10b4cc76f1bf211714e7588b2b92790895a1d6a6` (unchanged main
production) and historical candidate `b3659e76011239388da710d4de2f30c52018016f`
(production `e9adfdca71626190a17e36544af97b1066499bbe`). Both were fetched into
new, separately owned clean checkouts under this worktree's
`target/cuda-observer-postcommit-10b4cc76/`, with new local interpreters,
dependency trees, build caches and release wheels. No candidate files were
overlaid, and all controller snapshots match the committed campaign files.

Using the same two manifest seeds, options, inputs, tolerances and H100 GPU0,
the fixed slots were `[1, 1, 1, 1, 0, 1]` for the baseline (**5/6**) and
`[1, 1, 1, 1, 1, 1]` for the historical candidate (**6/6**). All 24 reference
workers passed. The baseline retained both missing-reduction failures with
their exact errors and zero credit. Its other 10 native workers and all 12
historical-candidate native workers passed, including changed-input execution
through the same wrapper without forwarding, original-body execution or
re-lowering. All 17 rejection controls passed in each environment.

The [raw audit](diagnostics/cuda-compilation-observer-v1/postcommit-10b4cc76/checks/raw-audit.json)
checked all 48 distinct worker results, reconstructed the declared seeded
float32 inputs, and compared 3,220 native output values against the reference.
The observed maximum absolute difference was zero for every passing case;
the unchanged tolerances remained enforced. It also verified actual local
interpreter paths, wheel/source/extension hashes, pinned observer and production
hashes, and clean before/after status for both checkouts. The new
[portable validation](diagnostics/cuda-compilation-observer-v1/postcommit-10b4cc76/checks/portable-validation.json)
passed against the published files. Full command output, compiler/runtime
identity, GPU utilization/memory snapshots and all unsupported outcomes are
retained. The environment again used driver 580.82.07, PyTorch `2.13.0+cu130`,
CUDA runtime `13000`, Rust/Cargo 1.92.0 and installed, unused nvcc 12.6.85.

These are author-run measurements and audits. No independent campaign review
or human approval occurred in this evidence-refresh step. The historical
candidate's success does not add row-sum capability or implementation credit
to this evaluator-only branch. No performance workloads or latency samples
were measured, and no timing speedup is claimed.
