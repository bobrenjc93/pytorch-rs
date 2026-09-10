# Native CUDA graph / CPU division and mean L1 integration

This integration retains the native CUDA graph bridge from `786c1b2`, CPU
scalar-division backward from `0480e2c`, and mean L1 backward from `792a01d`.
The combined implementation at `8df8ad19ba6811bd1dbf9e8e8e4a8c739d51ffc2`
needed no additional production changes. The integration patch fixes the
reported PR #1960 Clippy failure with an exact float32 `to_bits()` assertion
and adds differential tests for division before/after mean L1, either/both
operands, shared operands and views, accumulation, graph release, empty and
offset/strided inputs, and IEEE loss weights. Mean forward alone uses the
existing one-ULP reduction allowance; division and gradients remain bit-exact.
Unsupported forms and eager behavior remain unchanged.

These are **precommit integration checks**, not clean committed candidate
performance evidence. The author was prohibited from committing or operating
Burner delivery. Full local command receipts, source/test/harness hashes,
logs, build outputs, raw calls and cache identities are retained under
`target/integration/`; they are intentionally not published as clean evidence.
The final build is `target/integration/build-final/build-record.json`, built
with the repository's `capture_depth_concat_build.py --allow-dirty` from an
empty build target. The installed wheel and source extension were verified.

Final local results (command receipts use the corresponding names below):

| Check | Result | Receipt/log stem |
| --- | --- | --- |
| `cargo fmt --check` | Passed | `fmt-final` |
| `cargo clippy --all-targets -- -D warnings` | Passed | `local-clippy` |
| Same Clippy command with `--features python-bindings` | Passed | `local-clippy-bindings` |
| `cargo test --all-targets`, without / with Python bindings | 385 / 397 passed | `rust-tests`, `rust-bindings-final` |
| Installed wheel provenance / focused Python tests | Passed / 67 passed | `wheel-final`, `focused-final` |
| H100 CUDA regression suite / final-wheel subset | 75 tests, 7 skips / 33 tests, 3 skips | `cuda-regressions`, `cuda-final` |
| Separate GPUs 0,1 restoration tests | 7 passed | `two-gpu-final` |
| CUDA hidden | 1 portable test passed, 7 hardware tests skipped | `portable-skips` |
| Fixed CUDA math / compiler corpus | 18/18 trials / 38/38 eligible cases | `fixed-math`, `compiler-final` |
| Fixed four-shape workload | 4/4 eligible; existing capped aggregate 100%, no gain attributed | `fixed-scoring` |
| Declared primary 798431, held-outs 481723/926051, repeat 798431 | 48/48 cells passed; 29,760 raw timed calls | `primary`, `held-481723`, `held-926051`, `primary-repeat` |
| Source/build/raw-data audit | Passed | `evidence-audit`, `validation-summary.json` |

The final runtime is worktree-local CPython 3.12.14, PyTorch 2.13.0+cu130 and
CUDA runtime 13.0 on H100 (driver 580.82.07), Rust 1.92.0 release with thin LTO,
one codegen unit and extension-module/abi3. Native operations use cuBLAS and
driver-JIT PTX; nvcc 12.6.85 is recorded and used by the separate fixed private
scoring kernels. Local Cargo downloads and locked dependencies were populated;
final diagnostic CUDA/Triton/Inductor caches began empty, with correctness
checks warming CUDA before the sequence. The scoring compiler caches began
empty separately. Reports retain first calls and cache inventories.

The original failed attempts remain in the local logs: the first system
interpreter lacked the static Python library for Rust binding tests, and the
first new mean-forward assertion incorrectly required bit equality for an
existing one-ULP reduction-order difference. Final checks use the local
interpreter and the corrected assertion. The source CI job
`34511502296 / 102986492020` remains a historical failure; local successes do
not change its status.

Historical publication `42c933e2d38d26b5ffa18cdf936fced3829bc4ba` and all 81
files in its `postcommit-786c1b2` directory were checked byte-for-byte against
Git, including the original `fbb0aa0` baseline captures. None were changed.
Its primary/repeat capped parity was 81.4367% / 80.0108%; composed native
ratios versus actual main were 1.0609x / 0.9826x and held-out ratios were
1.0054x / 1.0036x. This establishes no repeatable native acceleration.

GPU 0 contention from an unrelated workload was observed during the new
sequence and recorded in `capture-contention.receipt.json` and the per-run
GPU inventories. All measurements, slow cells and failures are retained;
these timings establish neither a performance non-regression nor a gain.
No evaluator, workload, matrix, denominator, weight, threshold, or managed
progress artifact was changed.

Independent read-only integration review found no correctness issue, including
the final mean-forward assertion adjustment. Burner must still commit the
implementation/tests, pause admissions and secure GPU resources, capture
fresh clean committed evidence into a new empty target, publish evidence-only
docs, obtain independent exact-head review, pass all ten current-definition
non-regressing gates and exact-head CI, and use managed merge/cleanup with
source associations preserved. No branch, commit, push or PR was created here.

A setup boundary error is disclosed: `uv python install` created
`/home/bobren/.local/bin/python3.12`, a symlink to the worktree-local interpreter,
because its separate executable-link directory was not initially overridden.
It was left untouched under the restriction against further external writes.
Subsequent commands also direct uv executable/tool directories into the worktree.
