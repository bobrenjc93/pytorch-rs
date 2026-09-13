# Tensor-leaf multiply-add broadcast diagnostic

This increment admits only original `Add(Mul(Input(a), Input(b)), Input(c))`
and `Add(Input(c), Mul(Input(a), Input(b)))` for unequal input shapes. All leaves
are tensor inputs; IDs may repeat in the existing one/two-tensor ABI. The old
equal-shape language and one-stage/no-live-sin-or-cos broadcasting remain intact.
Scalar leaves, identity wrappers, competing products, signed expressions,
additional arithmetic and outer ReLU/sin/cos do not enter the exception.

The sole production change is the capability predicate in
[`Graph::indexing`](../../../src/pointwise_indexing.rs). Compilation and direct
cached-kernel execution both use this original-IR owner and its checked shapes,
sizes and addresses. Numerical lowering, NVRTC execution and cache ownership
are unchanged. The [focused tests](../../../tests/test_compile_pointwise_tensor_madd.py)
exercise acceptance/rejection, all ordered input-ID triples, genuine CUDA direct
execution, persistent wrapper histories, offsets, unchanged inputs, fresh outputs,
cache failures, reset and current-device restoration on both visible ordinals.

## Reproduction

Use a worktree-local interpreter, dependencies and freshly built release wheel.
Hold Burner's canonical `gpu` and `cpu-heavy` resources; these commands do not
allocate resources. Run ordinary checks with `CUDA_VISIBLE_DEVICES=0`, and the
two-device test with an explicit `CUDA_VISIBLE_DEVICES=0,1` reservation.
Keep Cargo, uv, temporary, Python, CUDA, Inductor and Triton caches inside the
worktree. The default nvcc on this host is not the pointwise JIT compiler;
the dispatched kernel records the NVRTC version and options actually used.

```bash
cargo fmt --all -- --check
cargo clippy --locked --offline --all-targets --features python-bindings -- -D warnings
cargo test --locked --offline --lib --features python-bindings -- --test-threads=1
.venv/bin/maturin build --release --locked --offline --out target/tensor-madd/wheels-final
uv pip install --python .venv/bin/python --no-deps --force-reinstall target/tensor-madd/wheels-final/*.whl
.venv/bin/python .github/scripts/verify_native_extension.py
.venv/bin/python -m unittest discover -s tests -p 'test_compile*.py' -v
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest discover -s tests -p test_compile_pointwise_tensor_madd.py -v
```

[`capture.py`](capture.py) runs an unscored fresh-process diagnostic with untouched
`framework.compile(fn)` defaults. The `--build` command creates a source/build manifest containing
`source` (relative source file paths to SHA256), `wheel` and `extension` (each
with absolute `path` and `sha256`). Record the source hashes immediately around
the release build and verify the wheel's installed Python and extension bytes.
The capture rejects mismatched source, wheel or installed package bytes.
The build command archives its source, uses a fresh Cargo target, verifies that source
hashes remain unchanged, installs the resulting wheel and verifies installed bytes.
It preserves command logs and failures. The development manifest below records
the concrete build used.

```bash
# With worktree-local cache variables set as above:
.venv/bin/python docs/diagnostics/compile-pointwise-tensor-madd/capture.py --build target/madd-build
# Each output directory must be new. Run serially, in both measurement orders.
.venv/bin/python docs/diagnostics/compile-pointwise-tensor-madd/capture.py native target/madd-native-first target/madd-build/build.json
.venv/bin/python docs/diagnostics/compile-pointwise-tensor-madd/capture.py reference target/madd-reference-second target/madd-build/build.json
.venv/bin/python docs/diagnostics/compile-pointwise-tensor-madd/capture.py reference target/madd-reference-first target/madd-build/build.json
.venv/bin/python docs/diagnostics/compile-pointwise-tensor-madd/capture.py native target/madd-native-second target/madd-build/build.json
.venv/bin/python docs/diagnostics/compile-pointwise-tensor-madd/capture.py --compare target/madd-native-first/report.json.gz target/madd-reference-second/report.json.gz target/madd-order-one.json
.venv/bin/python docs/diagnostics/compile-pointwise-tensor-madd/capture.py --compare target/madd-reference-first/report.json.gz target/madd-native-second/report.json.gz target/madd-order-two.json
```

Every case keeps its wrapper through shape/value/identity changes. Each state
records one initial call, five warmups and 17 samples, with identical real CUDA
runtime synchronization and output materialization outside timing. Raw reports
retain full inputs/outputs, exact hexadecimal IEEE values (including signed
zeros; NaN payloads coalesce), every timing and failure. Each native call must
invoke the module identified by the executor LRU; its actual CUDA and PTX are
captured. The profiler and patched eager bridges reject original-body or per-node
replay. These timings are diagnostic observations, not a performance score.

## Development evidence

Validation passed on an uncommitted worktree based on
`77aa16fc2d0cbd258f0fa309140dc75708cf2ee3` (merged PR #1992).
The [validation summary](development/validation.json) and
[command receipts](development/final-commands.json) retain results and GPU snapshots.

| Check | Result |
| --- | --- |
| Existing compiler selection | 892 unique tests across all 85 modules: 867 passed, 25 explicit two-device reservation skips; all processes exited zero and test-source hashes remained unchanged |
| Focused multiply-add suite, GPUs 0 and 1 | All 10 tests passed, including real NVRTC/direct execution, both device ordinals and exact overflowing-product cancellation |
| Existing pointwise/scalar device guards | All four tests passed under the same two-device reservation |
| CUDA hidden | Three metadata tests passed; seven hardware tests explicitly skipped |
| Rust library | 210 default-feature tests and 237 Python-binding tests passed; the 28-test pointwise IR subset also passed |
| Formatting, Clippy and docs | Both default and binding all-target Clippy checks passed with warnings denied; formatting, local documentation links and Python syntax passed; executable Python AST unchanged; Rust doc-tests completed with zero examples |
| Separate-process ordinary-default comparisons | Both native/reference and reference/native orders passed exact comparisons for all 42 states per order |

The five pointwise reservation skips in the broad selection are covered by the
focused and existing two-device runs. The other 20 reservation skips belong to
unchanged compiler owners. The complete
[module receipt](development/module-results.json),
[module logs](development/compiler-module-logs.tar.gz) and
[serial module runner](development/run-modules.py) preserve the full selection.

Each of the four comparison legs executes six programs and 42 shape/value/identity
states, with one initial call, five warmups and 17 samples per state: 966 calls per
leg. Both orders preserve exact finite results, signed zeros and nonfinite classes;
NaN payload equality is not required. Wrappers persist throughout each case.
No numerical mismatch was observed. These are diagnostic observations, not scores.

| Order | Raw reports and dispatched CUDA/PTX | Exact comparison |
| --- | --- | --- |
| Native, reference | [Native report](development/paired-0-native.json.gz), [native code archive](development/paired-0-native.tar.gz), [reference report](development/paired-1-reference.json.gz) | [42 states passed](development/comparison-0-native-1-reference.json) |
| Reference, native | [Reference report](development/paired-2-reference.json.gz), [native report](development/paired-3-native.json.gz), [native code archive](development/paired-3-native.tar.gz) | [42 states passed](development/comparison-2-reference-3-native.json) |

The [fresh build manifest](development/verified-build/build.json),
[build log](development/verified-build/build.log),
[source archive](development/verified-build/source.tar.gz) and
[execution driver](development/run-final.py) bind these captures to the installed
wheel and extension. The wheel SHA256 is
`a1572c62a14ed6698c804f58234379a4a3834059cda7017e04a9f64f0e2a5b36`;
the extension SHA256 is
`932a5ae9cd1759d987141ceb9d3fa2c50ddc6fd52b44329861c4a6c0f49a7d84`.
The build uses Rust 1.92.0, Python 3.12.12, release/thin LTO/codegen-units=1 and
ABI3 Python bindings. Reference PyTorch is `2.13.0+cu130`. Kernels actually
compiled with NVRTC 13.0, `compute_90`, FMA enabled and FTZ disabled; both legs
used CUDA runtime 13000 from the worktree-local CUDA 13 package. The host nvcc
reports 12.6 and was not the JIT compiler.

The H100s report driver 580.82.07. The paired runs use GPU 0
(`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`); the explicit two-device tests also
use GPU 1 (`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`). Burner assigned this run
both canonical `gpu` and `cpu-heavy` resources. Interpreters, dependencies, builds
and writable caches stayed in this worktree. See the retained
[environment setup](development/environment.sh).

The [exploratory status record](development/exploratory-status.json) preserves
setup and fixture failures, the interrupted all-in-one compiler attempt, and the
[earlier smoke capture](development/smoke.tar.gz). That smoke used an earlier
extension before verified build-manifest enforcement and is excluded from the
paired evidence. Its raw report and script are retained unchanged. No fixed
tolerance, compiler limit, reference setting, scoring corpus or historical
operator expectation was changed to obtain the passing results.

## Burner-owned follow-up

Clean candidate/main captures, exact-head qualification and frozen-corpus
measurement belong to Burner's later post-commit phase. This worker does not
commit, run a replacement evaluator, alter denominators or expectation catalogs,
or install the operator's scoped pre-cleanup observer. The operator must retain
frozen-command detailed reports through that separately reviewed observer.

The historical competing-product default-autotune failure remains an unsupported
boundary, not a numerical defect repaired here. See the immutable
[broadcast review record](../compile-pointwise-broadcast/review-autotune-blocker.md).
Neither these bounded tests nor the fixed corpus establish broad Inductor
equivalence or an all-program percentage.
