# Broadcast trigonometry: clean implementation evidence

Measured on 2026-09-16 UTC at clean implementation commit
`6255306165615d9502a4aa491187bcf626e9d585`, based on
`9a36b4a4642528d5af7533b309a8562b74447084`. The source identity was
`7c507bde8112dbb33e3a08d1a49371956759da3891d1f462bf0bd6a34185fcf2`.
The setup, focused checks, and full evaluator recorded no working-tree changes.
Only evidence and this documentation were added afterward. These captures do
not replace independent review or Burner's subsequent qualification gates.

The implementation admits unequal-shape returned roots with at most one
arithmetic stage, including live sin/cos. The tensor-leaf multiply-add exception
requires no live trig in any returned root. The numerical planner and generic
native CUDA instruction executor are unchanged. In particular,
`p = x * y; return (p + x, p.sin())` remains excluded in both output orders:
the returned multiply-add cannot use the depth-two exception when another
returned root has live trig.

## Measurements

| Check | Result | Evidence |
| --- | --- | --- |
| Rust original-IR indexing tests | 11 passed | [Log](rust-indexing.log) |
| Focused Python compiler regressions on H100 | 21 passed | [Log](focused-h100.log) |
| Portable broadcast-trig tests, CUDA hidden | 2 passed, 5 explicit hardware skips | [Log](portable-trig.log) |
| Fixed default-compile coverage | **19.5%**, 20/112 cells passed | [Full report](full-evaluation/report.json) |
| Fixed default-compile CUDA performance | **33.7010604577856**, 20/56 cells successful | [Full report](full-evaluation/report.json) |

The full report has `valid: true`, `diagnostic: false`. All reference programs
passed in the CPU round and both CUDA rounds. The unchanged
`public-default-compile-v2` contract retains all unsupported outcomes, slow
results, raw observations, and both implementation orders. Its common-success
geometric mean reference/candidate latency ratio is `1.1495595242006327`;
this separate statistic excludes failed cells and is not the aggregate score.
Both CUDA variants of the existing `nested_inputs` program passed in both
orders; its CPU variants remain unsupported. No new baseline was measured here,
so this record makes no claim about score movement.

The focused checks cover one-stage trig before/after arithmetic, original
rejection replacements, both sin/cos, scalar kinds and histories, unused
arguments, offset/rank-zero/empty/broadcast shapes, exact IEEE witnesses,
subnormal amplification, nested runtime trig, cached execution without eager
replay, output freshness/aliases, unchanged inputs, mixed/deeper exclusions,
and the independent cached-address/prepared-shape guards. Exact zero signs and
the existing general tolerances are preserved; NaN payload equality is not
required. The broader development test runs are not relabeled as clean-commit
measurements here.

## Commands and build identity

[Check receipts](check-receipts.json) retain the actual argument vectors,
timestamps, exit codes, source identity, and log hashes. The Rust command was:

```bash
cargo test --locked pointwise_ir::indexing -- --test-threads=1
```

The Python commands use the freshly installed evaluator environment and the
existing `tests/test_compile_pointwise_*.py` tests. The complete focused test
selection is recorded in the receipts. Ordinary `torch_rs.compile(fn)` and
default `torch.compile(fn)` were used, without backend or compiler-limit
overrides.

The repository-supported setup and full measurement commands were:

```bash
bash scripts/evaluate_torch_compile_default.sh --setup-only
bash scripts/evaluate_torch_compile_default.sh --metric both \
  --output target/postcommit-6255306/full-evaluation.json
```

[Setup receipt](setup-receipt.json), [setup log](setup.log),
[evaluation receipt](fixed-evaluation-receipt.json), and
[evaluation log](fixed-evaluation.log) preserve the executions.
[env.sh](env.sh) records the worktree-local environment and cache configuration;
the full evaluation additionally set `BURNER_EVALUATION_ARTIFACT_DIR` to this
directory's `full-evaluation/`, as recorded in its receipt. The initial setup
bootstrapped with the existing worktree-local `.venv/bin/python` before creating
the fresh evaluator virtualenv. Heavy commands ran serially, with
`CUDA_VISIBLE_DEVICES=0` except the explicitly portable test. Dependency and
native compiler caches could be warm; each setup built and installed a new
release wheel. The evaluator gave each worker fresh Inductor/Triton directories
and used one host thread, five warmups, and 17 samples per cell.

[Provenance](provenance.json) records imported source/native hashes, Python and
toolchain identities, runtime paths, NVRTC options, and device metadata:

- Python 3.12.12; reference PyTorch `2.13.0+cu130`, CUDA 13.0.
- Native NVRTC 13.0, CUDA runtime `13000`; installed nvcc 12.6 was not used for JIT.
- NVIDIA H100, GPU 0 UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`,
  driver `580.82.07`.
- Imported native extension SHA256
  `98e94bba85f5d0311c3cc72ae510cbcb434fed7fcd9ae07fd650999e13b407b7`.
- Focused-check setup wheel SHA256
  `5186c2095b84b42b6f5ad4c910522626927880b769c8c2422a9d6d6c1f5c45de`.
- Full-evaluation setup wheel SHA256
  `3d4d42b56ca867b7ca73482eea93449a0bc99115844ed9af7d85a2ad27636453`.

Both setup wheels contain the same native extension. The full evaluator's
worker provenance independently records that extension and verifies installed
Python sources against the measured checkout. All recorded worktree, build,
import, and executable paths belong to this worktree. GPU snapshots accompany
the checks and full report.

## Retention and limits

The existing [evaluator export contract](../../../torch-compile-default-evaluator.md#evidence-retention)
copied every direct regular file from
`target/default-compile-eval/run-20260916T001411Z-b5914370/` into
[full-evaluation/](full-evaluation/), byte-for-byte, with the report last.
Worker paths in the report identify those original worktree-local locations;
their basenames and hashes identify the retained copies here. The six raw gzip
files and six worker logs are intentionally retained, including failures.
Wheels and compiler caches remain in ignored worktree-local `target/` paths.
[Artifact verification](artifact-verification.json) records the export, source,
build, and log integrity checks. No evaluator, corpus, threshold, dependency,
implementation, test, or managed progress artifact changed in this evidence step.

These are finite numerical and execution observations, not general Inductor
coverage or performance parity. The compiler metadata probe is not an
independent device instruction trace; neither generic VM PTX nor a reconstructed
plan would establish one. Empty outputs execute no trig. Constant-only trig
does not prove runtime-input behavior, and a scalar `-0.0` call may reuse the
`+0.0` specialization rather than prove a separate specialization. The original
frozen diagnostic and earlier development captures remain separate and
unchanged; none supplies current-candidate performance credit here.
