# Clean input-admission evidence

This post-commit evidence measures candidate
`32105a2277965dbf15a84cc464aaf3e5be0006d2` and accepted main
`60202557b4f110d07777f585e804ab5f55e1ff7b` on 2026-09-17. Both captures started
and finished with clean checkouts. All implementation, tests, dependency
specifications and evaluator files remained unchanged throughout this step.
Evidence was copied into this directory only after the captures and source audit.
The earlier development records remain unchanged and are not clean-run evidence.

## Outcome: no reliable end-to-end benefit established

| Frozen public-default-compile-v2 result | Main `60202557` | Candidate `32105a22` |
| --- | ---: | ---: |
| Coverage score | 19.5% | 19.5% |
| Coverage cells passed | 20/112 | 20/112 |
| CUDA performance score | 33.3678445484 | 33.4386405165 |
| CUDA cells passed | 20/56 | 20/56 |
| Common-success reference/native geometric latency ratio | 1.1143679075 | 1.1473995780 |
| Valid, non-diagnostic gate result | Yes | Yes |

There are **no pass/fail changes**. All 92 unsupported/failed coverage cells and
36 unsupported/failed CUDA cells stay in their denominators. All reference
programs passed in all six reference workers across the two captures. The score
increase is **0.0707959681 points**. It is too small to establish a reliable
performance improvement from this single paired capture; no repeat was selected
to improve the result and no tolerance, reference or workload was changed.

The geometric main/candidate native-median ratio over the same 20 successful
CUDA cells is 1.076289, while the reference-median ratio also shifted by 1.045304.
The reference-normalized change is therefore about 1.029642. These are descriptive
comparisons of two sequential build captures, not confidence intervals or proof
that admission coalescing caused the difference. Two native cell medians became
slower, and several reference-normalized ratios regressed. Every common cell,
including those regressions, is listed in [common-cells.md](common-cells.md).
The unchanged frozen evaluator remains the score owner; these derived ratios
neither replace its aggregation nor award unsupported cells credit.

The candidate's clean-commit focused capture passed **24 tests**, with **two
explicit two-GPU skips**. It includes direct metadata/old-API parity, private
exports, complete warm input admission and ABI transitions, failure-atomic
caches, and existing real GPU tests for one/two inputs, aliases, fresh storage,
offsets, scalar/length-one/empty tensors, broadcasting and native preparation/
launch guards. See [clean-gpu-checks.log](clean-gpu-checks.log). This is a focused
refresh of the requested clean correctness evidence; unrelated full test suites
were not repeated. Its elapsed duration is not a benchmark result.

## Commands, setup and hardware

The measured commands were the repository-supported wrapper, once per revision:

```bash
# Candidate checkout root:
source docs/diagnostics/compile-input-admission/env.sh
export BURNER_EVALUATION_ARTIFACT_DIR="$PWD/target/postcommit-32105a22/candidate-export"
bash scripts/evaluate_torch_compile_default.sh --metric both \
  --output target/postcommit-32105a22/candidate-report.json

# Clean detached main checkout at target/postcommit-32105a22/main:
# Set the same environment variables to paths under this nested checkout.
export BURNER_EVALUATION_ARTIFACT_DIR="$PWD/target/postcommit/export"
bash scripts/evaluate_torch_compile_default.sh --metric both \
  --output target/postcommit/main-report.json
```

The command records contain exact argv, cwd, UTC start/end times, clean status,
return codes, resource environment and GPU inventories. The focused unittest
command in `clean-gpu-checks.json` additionally used `PYTHONPATH="$PWD/tests"`;
all other environment settings are in `env.sh`. `record.py.txt` records
commands without altering them; `prepare-main.sh.txt` records the detached
checkout and local dependency-copy procedure. No branch was created. Both
interpreters, installations, wheels, writable caches and exports belong to the
current worktree; main is a nested checkout inside it. Main's interpreter and
locked dependencies were copied from the candidate's local environment, then the
unchanged wrapper rebuilt and replaced the native package for main. No external
checkout, tool, installation or global configuration was modified.

The wrapper used locked dependency synchronization and
`maturin build --release --locked`. Candidate Cargo/dependency caches were warm;
main had a fresh Cargo target and copied dependency cache. The reports retain
actual separate setup timestamps and durations; these unequal setup cache states
are not attributed to compiler performance. Every evaluator worker used fresh
Inductor/Triton caches, one host thread, five warmups and 17 samples. Both CUDA
implementation orders ran, and the two revision captures ran sequentially.
Cold factory/compile costs remain separate from steady samples.

Burner's canonical `gpu` and `cpu-heavy` locks belonged to `idea_7ba378d2`;
[resources.json](resources.json) records the read-only observation. Only GPU0 was
reserved and visible (`CUDA_VISIBLE_DEVICES=0`): H100,
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, capability 9.0, driver 580.82.07.
Before/after UUID, utilization, memory and clock inventories are in each command
record and evaluator report. They are observations, not substitutes for locks.

Both builds used local Python 3.12.14, PyTorch `2.13.0+cu130`, and identical CUDA
13.0 runtime bytes for the evaluator's synchronization. The source-bound
`candidate-environment-command.log` and `main-environment-command.log` probes
reported actual native NVRTC **13.0**, local CUDA 13 runtime paths, Rust 1.92.0,
and installed nvcc 12.6.85. nvcc did not compile these native JIT kernels. The
probes execute a public default-compiled negation only to establish compiler and
runtime identity; they contain no timing shortcut or performance score.

## Verification and retained artifacts

[candidate-report.json](candidate-report.json) and [main-report.json](main-report.json)
are byte-for-byte copies of the complete frozen reports. They include source
manifests, measured commits, wheel hashes, imports, runtime identity, setup data,
all 112 cells, cold/steady observations and complete worker summaries.
[audit.json](audit.json) records successful checks of source/wheel/import bytes,
clean provenance, interpreter/runtime hashes, all workers and denominators,
reference success, unchanged evaluator/dependency definitions, the original
aggregation and exact exported copies. The audit code is preserved as
[audit.py.txt](audit.py.txt); it performs no new workload measurement.

The frozen evaluator exported every direct regular run artifact through
`BURNER_EVALUATION_ARTIFACT_DIR` to separate worktree-local sinks. Original gzip
files, logs and reports remain untouched in their run directories, and export
copies were verified against their hashes. [raw-retention-manifest.json](raw-retention-manifest.json)
identifies every original and exported file. The large raw payload archive is
retained locally for Burner collection; its location, hash and lossless payload
manifest are recorded in `archive-receipt.json` and `payload-manifest.json`.
Gzip payloads are decoded without JSON reserialization for that archive, then
round-trip verified. Original compressed files remain unchanged. No external
collector receipt is claimed; Burner owns collection and worktree cleanup.

There were no failed post-commit builds, measurements or correctness checks,
and neither measurement was repeated. The first evidence-packaging attempt used
system Python, which lacks `hashlib.file_digest`; it failed before publication
and was reissued with the local Python 3.12 interpreter. The observed exception
and recovery are retained in `publication-first-failure.log`. Measured artifacts
were untouched. The original development failures remain in the parent directory.
`SHA256SUMS` covers the checked-in evidence copies. This evidence step does not
approve the branch or replace independent review, exact-head evaluation, full
qualification or merge gates. No managed README/history/SVG progress artifact
was changed.
