# Native default pointwise JIT validation

**Refresh required after review fixes.** The fixes to Rust feature gating,
contraction/negation and globals admission change implementation beyond the
measured commit below. These existing reports and generated-code provenance are
preserved verbatim, but do not measure the repaired candidate. Burner must
repeat both clean-commit gates and `capture.py` after committing the fixes;
dirty development checks cannot substitute for that capture. See the
[review-fix validation](review-fixes.md) for the new regression checks.

Base: `76738b39fd6884ffd43b4dff2f5292257a1c6e6b`. The baseline ran from that
clean committed checkout before any source edits. Fresh post-commit coverage,
CUDA-performance and generated-code captures measured clean implementation
commit `5b93c983f6b4300d804113b514bf9da3fe45de93` on 2026-09-12. Both scoring
commands report `valid: true`, `diagnostic: false`, with no infrastructure error.
No evaluator, corpus, tolerance, denominator or managed progress artifact was
changed. See the [implementation contract](../../compile-pointwise-jit.md).

| Unchanged public-default-compile-v2 gate | Clean baseline | Candidate at `5b93c983` (refresh pending) |
| --- | ---: | ---: |
| Weighted coverage | 0% (0/112 cells) | 6% (4/112 cells) |
| Weighted CUDA performance | 0% (0/56 cells) | 12% (4/56 cells) |
| CUDA-performance common-success geometric mean, reference/candidate | null | 2.2146089076321167 |

Only the two arithmetic programs, in both CUDA variants, passed. All other
cells remain zero. Each successful cell's capped performance ratio is one;
this accounts for the 12% category weight. The common-success ratio describes
only those four cells, not general PyTorch performance. Every reference program
passed in all three coverage and both performance reference workers. Both CUDA implementation orders, five
warmups, 17 samples, one host compute thread, cold timings, changed-input
checks and all 112 coverage/56 performance cells were retained.

The original development diagnostic remains unchanged: 6% coverage, 12% CUDA
performance and a 2.12810760647372 common-success ratio. It intentionally records
`valid: false`, `diagnostic: true` without an infrastructure error. The fresh
explicit commands above supply the previously deferred committed measurements;
the coverage command's additional CUDA ratio (2.2158146403685066) is also retained
in its complete report. The table uses the separate performance command.
These measurements do not approve the branch or replace independent review.

## Evidence

- [Clean candidate coverage](candidate-coverage.json.gz) and
  [clean candidate CUDA performance](candidate-cuda-perf.json.gz) are
  byte-preserving gzip copies of the complete evaluator reports. They retain
  every cell, both orders, raw timing samples, cold costs, correctness observation
  hashes, source/lock/wheel identities, worker statuses and hardware snapshots.
  Full raw output observations remain at the worktree-local paths in the reports;
  they are not duplicated here. The [baseline](baseline.json.gz) and
  [development diagnostic](candidate-diagnostic.json.gz) are preserved unchanged.
- [Post-commit receipt](postcommit.json) records commands, environment settings,
  setup timestamps, report hashes and verification results. [Post-commit logs](postcommit-logs.json.gz)
  retain both build/gate outputs, all ten workers and the fresh codegen/JIT checks.
  The baseline's 123 source hashes were verified against its commit, the direct
  parent of the candidate. Each new report's 128 source hashes, wheel/native
  identities, interpreter, logs and raw-observation hashes were verified locally.
- [Generated CUDA source](kernel.cu), [PTX](kernel.ptx.gz),
  [provenance](provenance.json) and [source manifest](source-manifest.json.gz)
  were recaptured from the clean committed build using [capture.py](capture.py).
  CUDA, PTX and source-manifest bytes are unchanged; provenance now records the
  clean commit and the evaluator's freshly built installed wheel. The capture
  records an independent ordinary public function, two shapes, changed values,
  fresh outputs, two graph entries and one code module. They are code-generation
  evidence, not a timing benchmark. The capture blocks production dependence on
  installed PyTorch by checking that it was never imported. The existing
  provenance field named `base_commit` identifies the measured commit, now `5b93c983`.
- The original author [validation inventory](validation.json) records all 67 compiler test files
  (797 unittest cases), original exit statuses and passing targeted repairs.
  [Logs](logs.json.gz) preserve every file's output, initial failures, builds,
  final checks and evaluator worker logs. [Initial IEEE probes](initial-ieee-probes.json.gz)
  preserve the original observations that motivated zero-sign and FMA fixes.

The native generated kernel selected NVRTC **13.0**, CUDA runtime **13000**,
`compute_90`, explicit FMA contraction and gradual underflow, without fast math.
The independently queried `nvcc` is **12.6** and was not used to generate the
JIT kernel. The GPU was H100 index 0,
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver **580.82.07**. Python was
3.12.14, reference PyTorch 2.13.0+cu130, Rust 1.92.0, release native builds.
The provenance/report files contain the actual library paths, hashes and
before/after GPU snapshots. Snapshots do not establish a scheduler reservation.

## Checks and retained failures

The fresh committed wheel passed the focused JIT module: 13 tests, with the
explicit two-device case skipped under `CUDA_VISIBLE_DEVICES=0`. This includes
the generated trees, IEEE/FMA cases, guards, cache/reset/concurrency, lifetimes
and isolated no-PyTorch/no-body checks described below. Source status remained
clean before and after these checks. The earlier exhaustive compiler, Rust,
documentation and two-device checks below remain the unchanged author record;
unrelated full suites were not repeated for this evidence-only step.

The full compiler selection ran each file in a fresh process without omitting
files after a failure. The installed-wheel matmul test initially required
`.venv` instead of an editable or evaluator-specific environment; it passed in
that required local environment. The default wrapper initially raised
`TypeError` for a non-tensor argument; restoring its existing
`NotImplementedError` contract made all 49 entrypoint tests pass. The combined
final entrypoint/JIT run passed 62 cases with one explicit two-device skip;
that two-device case passed separately with `CUDA_VISIBLE_DEVICES=0,1`.

Independent JIT tests include 12 generated expression trees over six shapes
(including multiple blocks, rank 3, scalar and empty), held-out constants and
fresh values, repeated intermediates, offsets, IEEE values, FMA cancellation,
no original-body/eager-evaluator/native-eager-chain execution, blocked PyTorch
imports, guards, failed compilation/retry, concurrent cache/reset and lifetimes.
Python 3.14.7 ran all five hardware-free cases and clearly skipped eight CUDA
cases without the reference CUDA environment.

Rust checks passed: seven focused pointwise tests, 20 CUDA unit tests and
20 tests across all eight CUDA integration executables. All-target Clippy with
Python bindings and warnings denied passed. Documentation checks passed
(30 Python cases, one hardware-only skip; rustdoc completed with zero doctests),
as did formatting and diff whitespace checks.

Initial build/type/fixture failures remain in the log bundle. The first IEEE
probe exposed eager/Inductor zero-sign differences; subsequent cancellation
probes exposed missing FMA contraction. A compiler flag alone was insufficient
because toolkit strength reduction could precede contraction. The final local
SSA contraction rule and corresponding semantic tests resolve that failure.
A final large-integer rounding probe also matched native output to default
Inductor in all four cases. Three differ by one float32 ULP from eager's direct
integer conversion; its logged three-way `match False` records that reference
distinction, not a native-versus-Inductor discrepancy.
No failed measurement was overwritten or promoted as a score.

## Reproduce

Run from the checkout root, using locked local environments. Keep
`CARGO_HOME`, `CARGO_TARGET_DIR`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`,
`TMPDIR`, `XDG_CACHE_HOME` and `CUDA_CACHE_PATH` inside this worktree. The
enterprise download settings used were `UV_SYSTEM_CERTS=true` and
`HTTPS_PROXY=http://fwdproxy:8080`. Build/install a release wheel into `.venv`;
clear an inherited `CONDA_PREFIX` when supplying `VIRTUAL_ENV` to maturin.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest tests.test_compile_pointwise_jit -v
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest \
  tests.test_compile_pointwise_jit.Hardware.test_two_device_restoration_and_module_ownership -v
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  docs/diagnostics/compile-pointwise-jit/capture.py target/pointwise-reproduction
```

The exhaustive compiler selection was `sorted(Path('tests').glob('test*compile*.py'))`,
with one `python -m unittest tests.<file_stem>` process per file. Rust commands:

```bash
cargo test --locked --features python-bindings --lib pointwise -- --test-threads=1
cargo test --locked --features python-bindings --lib cuda:: -- --test-threads=1
cargo test --locked --features python-bindings \
  --test cuda_add --test cuda_contiguous --test cuda_matmul --test cuda_mul_scalar \
  --test cuda_native_boundaries --test cuda_relu --test cuda_same_device_copy \
  --test cuda_sum_rows -- --test-threads=1
cargo clippy --locked --all-targets --features python-bindings -- -D warnings
cargo test --locked --features python-bindings --doc
```

The unchanged full baseline command was:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric both --output target/default-compile-baseline.json
```

The development diagnostic added `--diagnostic` and `CUDA_CACHE_DISABLE=1`.
The fresh clean candidate used the two explicit commands from
[the gate documentation](../../torch-compile-default-evaluator.md):

```bash
export TMPDIR="$PWD/target/tmp"
export XDG_CACHE_HOME="$PWD/target/xdg-cache"
export CUDA_CACHE_PATH="$PWD/target/default-compile-eval/cuda-cache"
export CUDA_CACHE_DISABLE=1 UV_SYSTEM_CERTS=true HTTPS_PROXY=http://fwdproxy:8080
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric coverage --output target/postcommit-5b93/coverage.json
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric cuda-perf --output target/postcommit-5b93/cuda-perf.json
```

Run from a clean checkout and choose new output paths when reproducing: the
evaluator refuses to overwrite evidence. The wrapper creates its local build
environment; create the local temporary/cache directories before running.
The post-commit codegen and focused tests used
`target/default-compile-eval/venv/bin/python`, the exact environment installed by
the gates; complete commands and dedicated test-cache paths are in the receipt.

Driver disk-code caching was disabled for both implementations in the new
candidate gates, while the historical baseline did not set that flag. Neither
result is a cold-cache speed comparison between builds. Inductor/Triton worker
caches were separately fresh in every run. Dependency/native build caches may
be warm, as recorded in each setup receipt. The review fixes now require a new
committed capture, as described above.
