# Native default pointwise JIT validation

Base: `76738b39fd6884ffd43b4dff2f5292257a1c6e6b`. The baseline ran from that
clean committed checkout before any source edits. The candidate is an
**uncommitted development diagnostic**, not an adopted Burner score.
No evaluator, corpus, tolerance, denominator or managed progress artifact was
changed. See the [implementation contract](../../compile-pointwise-jit.md).

| Unchanged public-default-compile-v2 gate | Clean baseline | Candidate diagnostic |
| --- | ---: | ---: |
| Weighted coverage | 0% (0/112 cells) | 6% (4/112 cells) |
| Weighted CUDA performance | 0% (0/56 cells) | 12% (4/56 cells) |
| Common-success geometric mean, reference/candidate | null | 2.12810760647372 |

Only the two arithmetic programs, in both CUDA variants, passed. All other
cells remain zero. Each successful cell's capped performance ratio is one;
this accounts for the 12% category weight. The common-success ratio describes
only those four cells, not general PyTorch performance. Every reference program
passed in all three reference workers. Both CUDA implementation orders, five
warmups, 17 samples, one host compute thread, cold timings, changed-input
checks and all 112 coverage/56 performance cells were retained.

The diagnostic intentionally records `valid: false`, `diagnostic: true` and
has no infrastructure `error`: that is the unchanged evaluator's explicit
unscored mode. Burner must make a fresh clean committed capture under its
`gpu` and `cpu-heavy` reservations during the post-commit evidence phase. This
worker was instructed not to commit or modify Burner's delivery state.

## Evidence

- [Baseline report](baseline.json.gz) and [candidate diagnostic report](candidate-diagnostic.json.gz)
  are byte-preserving gzip copies of the complete evaluator reports. They retain
  every cell, both orders, raw timing samples, cold costs, correctness observation
  hashes, source/lock/wheel identities, worker statuses and hardware snapshots.
  Full raw output observations remain at the worktree-local paths in the reports;
  they are not duplicated here (about 132 MiB per run).
- [Generated CUDA source](kernel.cu), [PTX](kernel.ptx.gz),
  [provenance](provenance.json) and [source manifest](source-manifest.json.gz)
  record an independent ordinary public function, two shapes, changed values,
  fresh outputs, two graph entries and one code module. They are code-generation
  evidence, not a timing benchmark. The capture blocks production dependence on
  installed PyTorch by checking that it was never imported.
- [Validation inventory](validation.json) records all 67 compiler test files
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

The candidate used the same command with `--diagnostic` and a new output path,
plus `CUDA_CACHE_DISABLE=1` to prevent driver disk-code reuse between workers.
This affects cold compilation provenance; neither result is a cold-cache speed
comparison between builds. Inductor/Triton worker caches were separately fresh
in both runs. For final committed scoring, Burner must run the unchanged
explicit `--metric coverage` and `--metric cuda-perf` commands from
[the gate documentation](../../torch-compile-default-evaluator.md), preserving
all cells and failures. The diagnostic percentages above must not be substituted
for that post-commit capture.
