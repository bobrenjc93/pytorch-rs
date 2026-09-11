# Compiled CUDA contiguous

The bounded eager compiler accepts default parameterless `Tensor.contiguous()`
on native CUDA float32 tensors without gradients:

```python
import torch_rs as torch

def packed_negation(x):
    return -x.contiguous()

x = torch.tensor([[1., 2., 3.], [4., 5., 6.]]).to("cuda:0").t()
f = torch.compile(packed_negation, backend="eager", fullgraph=True)
assert f(x).cpu().tolist() == [[-1., -4.], [-2., -5.], [-3., -6.]]
```

Already-contiguous results retain the same Python object, storage, shape,
strides and offset, including scalar, empty, singleton and higher-rank aliases.
Positive-stride rank-1/rank-2 views materialize through the existing native
packer into independent same-device storage, with canonical strides, offset
zero and bit-preserved values (including signed zeros and NaN payloads).
The original storage is unchanged; later mutations respect alias/copy ownership.

The existing one/two-input, global/helper, output-pytree, cache and shape-policy
contracts apply, including no-break `fullgraph=False`. Dynamic shapes retain
exact stride/offset guards and may change whether the result aliases or packs.
Both frontend and native planning validate all nodes before executing the first
one. Arithmetic still requires contiguous operands. Execution uses Rust directly,
without Tensor-method redispatch, Python-body replay or reference-PyTorch imports.

Arguments (including explicit default memory formats), CPU capture of this
method, noncontiguous rank >2, zero-stride packing, other dtypes and gradients
remain unsupported. [Bounded `Tensor.t()` capture](compile-cuda-t.md) can now
create rank-0/1/2 views inside a graph before packing. General reshape/view
grammar, new backends, training, fusion and performance parity remain excluded. Graphlets below are
non-scoring diagnostics; the frozen 38-case denominator and PR1970/PR1971
campaigns remain unchanged.

## Validation

Prepare locked dev/reference dependencies and an exact-source release build
using the [contributor setup](../CONTRIBUTING.md). Run:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_compile_cuda_contiguous
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v tests.test_compile_cuda_contiguous.CompileContiguousDeviceTests
```

Hardware tests skip clearly without the required devices. Test-only Rust
accounting verifies that malformed late native nodes execute zero operations;
it is absent from release builds and does not change evaluator observers.

## Clean-commit validation

The deferred capture, including the three review fixes, is complete at
`a3c5870bc56c97560061ffa68da6d6f253a3c644`. Git status was empty before and
after the release build and every measurement. This evidence and documentation
were added afterward; implementation, dependencies, tests and harnesses are unchanged.

The [release receipt](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/release/build-record.json),
[measured inputs](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/measured-inputs.json), [audit](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/audit.json)
and [inventory](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/inventory.json) bind the commit, source, build,
commands, installed package and native binary.

| Artifact | SHA-256 |
| --- | --- |
| Production source | `02647f45847695e0f386956541be69aca2a8c76f7ce05d0e5fa6734c46b56488` |
| Production diff against measured commit (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Native extension, installed and source package | `0aa957dad68e63d207d8945d25d51d4c4bb931dae303d10d53e063180725f3ce` |

A fresh canonical worktree-local `.venv` used locked dev/reference dependencies;
its predecessor was preserved inside `target`. The unchanged capture tool ran
`maturin build --release --locked --offline` without `--allow-dirty`, in a fresh
Cargo target. Local dependency caches were reused; CUDA JIT, Python and compiler
caches started fresh. See [setup](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/setup.receipt.json),
[environment](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/environment.sh.txt) and [preflight](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/preflight.log).
The H100 driver was 580.82.07; native runtime was explicitly the local `.venv`'s
CUDA 13.0 runtime (13000), with reference PyTorch 2.13.0+cu130. Python was 3.12.13,
NumPy 2.5.1 and Rust/Cargo 1.92.0. nvcc 12.6.85 was installed but unused;
native kernels use driver-JIT PTX. Receipts include physical UUID, utilization
and memory snapshots. Ordinary GPU checks used mask `0`; restoration used `0,1`.

| Clean-commit check | Result | Log |
| --- | --- | --- |
| Compiled contiguous differentials and guards | 12 passed; 1 hardware skip | [log](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/compiled-contiguous.log) |
| Existing CPU/CUDA compiler regressions | 189 passed; 7 hardware skips | [log](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/compiler-regressions.log) |
| Device/context restoration | 3 passed | [log](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/two-device.log) |
| Rust graph planning and zero-execution negatives | 3 passed | [log](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/rust-graph.log) |
| Rust CUDA packing integration | 1 passed | [log](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/rust-packing.log) |
| Eager CUDA packing | 6 passed; 1 hardware skip | [log](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/cuda-packing.log) |
| CPU layout/reference regressions | 101 passed | [log](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/cpu-layout.log) |
| No-device portability | 1 passed; 12 hardware skips | [log](diagnostics/compile-cuda-contiguous/postcommit-a3c5870b/no-device.log) |

The compiled checks include seeded packing and alias differentials, raw float
bits, mutation semantics, cold/warm caches, blocked-PyTorch imports and all three
review regressions: large empty aliases, mixed-device unused captures and strict
contiguous declarations. Invalid cache hits execute zero native operations.
Every capture, wheel verification and provenance audit passed. Unrelated full
Rust suites, Clippy and performance workloads were not repeated. These are
non-scoring diagnostics; the frozen38 denominator and independent review gates
are unchanged. No required clean capture remains deferred.

The earlier [review-fix development receipt](diagnostics/compile-cuda-contiguous/review-fixes/release-final/build-record.json)
and [audit](diagnostics/compile-cuda-contiguous/review-fixes/audit.json) remain pinned
to uncommitted repairs based on `d148fcb78560756ba6bac38d098ffe40ddc05aca`.
Their [failing reproduction](diagnostics/compile-cuda-contiguous/review-fixes/review-red.log),
original regression source and formatting failure are preserved unchanged.
The initial reproduction also attempted a transpose rejected by the existing
eager constructor; the final regression uses supported empty aliases at
`2**32` and `2**32 + 1`. The clean capture above supersedes development validation.

## Prior clean-commit validation (before review fixes)

Implementation commit `68766b43882081ab62c54925e94562297ecd5999` was
captured clean before independent review. Git status was empty before and after
the release build and every measurement. The subsequent evidence-only commit
`d148fcb78560756ba6bac38d098ffe40ddc05aca` added these records and
documentation. This capture predates the repairs; their clean recapture is recorded above.

The [release receipt](diagnostics/compile-cuda-contiguous/postcommit-68766b43/release/build-record.json),
[measured input hashes](diagnostics/compile-cuda-contiguous/postcommit-68766b43/measured-inputs.json),
[audit](diagnostics/compile-cuda-contiguous/postcommit-68766b43/audit.json) and [inventory](diagnostics/compile-cuda-contiguous/postcommit-68766b43/inventory.json)
bind the commit, source, commands, installed package and native binary.

| Artifact | SHA-256 |
| --- | --- |
| Production source | `a2a2235d1b2f6b3db84f8ef15082b16c71cb699e38678e1ca194f85e9945c8b7` |
| Production diff against measured commit (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Native extension, installed and source package | `8ea77b560d033412929e3e29e5b585a1e152896b51509fea651e7e7b9eae70ea` |

A fresh canonical worktree-local `.venv` was installed using locked dev/reference
dependencies; its predecessor was preserved inside `target`. The unchanged
repository build tool ran `maturin build --release --locked --offline` without
`--allow-dirty`, using a fresh Cargo target. Local dependency caches were reused;
CUDA JIT, Python and compiler diagnostic caches started fresh. All generated
files stayed in this worktree. See [setup](diagnostics/compile-cuda-contiguous/postcommit-68766b43/setup.receipt.json) and
[environment](diagnostics/compile-cuda-contiguous/postcommit-68766b43/environment.sh.txt) for exact commands and paths.

[Preflight](diagnostics/compile-cuda-contiguous/postcommit-68766b43/preflight.log) records Python 3.12.13, NumPy 2.5.1,
PyTorch 2.13.0+cu130, Rust/Cargo 1.92.0, NVIDIA H100, compute capability 9.0 and
driver 580.82.07. Native runtime was explicitly selected from the new `.venv`'s
`nvidia/cu13/lib/libcudart.so.13` (13000), matching reference CUDA 13.0.
nvcc 12.6.85 was installed but unused: native kernels use driver-JIT PTX.
Command receipts include physical UUID/index/utilization/memory snapshots.
Ordinary GPU checks used mask `0`; only restoration used `0,1`.

| Clean-commit check | Result | Raw log |
| --- | --- | --- |
| Rust graph planning and zero-native-execution negatives | 2 passed | [log](diagnostics/compile-cuda-contiguous/postcommit-68766b43/rust-graph.log) |
| Rust CUDA packing integration | 1 passed | [log](diagnostics/compile-cuda-contiguous/postcommit-68766b43/rust-packing.log) |
| Compiled layout differentials and guards | 9 passed; 1 expected skips | [log](diagnostics/compile-cuda-contiguous/postcommit-68766b43/compiled-contiguous.log) |
| Two-device/context restoration | 3 passed | [log](diagnostics/compile-cuda-contiguous/postcommit-68766b43/two-device.log) |
| Eager CUDA packing regressions | 6 passed; 1 expected skips | [log](diagnostics/compile-cuda-contiguous/postcommit-68766b43/cuda-packing.log) |
| Existing CPU/CUDA compiler regressions | 189 passed; 7 expected skips | [log](diagnostics/compile-cuda-contiguous/postcommit-68766b43/compiler-regressions.log) |
| CPU layout/reference regressions | 101 passed | [log](diagnostics/compile-cuda-contiguous/postcommit-68766b43/cpu-layout.log) |
| Portability with CUDA hidden | 1 passed; 9 expected skips | [log](diagnostics/compile-cuda-contiguous/postcommit-68766b43/no-device.log) |

The unchanged graphlets cover seeded transposes and offset/sliced views, exact
alias metadata, scalar/empty/singleton/higher-rank aliases, raw signed-zero/NaN
bits, nonmutation and copy/alias mutations, cold/warm execution, cache and
callable/global guards, malformed IR, blocked PyTorch imports, and native
composition. Skips receive no hardware correctness credit. Every capture and
the provenance audit passed; earlier failed development attempts remain intact.
Unrelated full Rust suites, Clippy and performance workloads were not repeated.
These are non-scoring diagnostics, with no added frozen38 coverage or general
compiler/performance claim. Independent review and normal Burner merge gates
remain separate; PR1970/PR1971 and all evaluator/observer contracts are unchanged.

## Preserved development validation

Started from verified clean main `22c4e1c4d32126b91ee9f5417842d245c0183fce`.
The [release receipt](diagnostics/compile-cuda-contiguous/development/release-hardened/build-record.json)
binds the original implementation's uncommitted source and release wheel. **This is development
evidence, not a clean implementation-commit capture.** The implementation task
prohibited creating commits and deferred the clean capture now reported above.
Its original receipts, paths, results and failed attempts remain unchanged.

| Final artifact | SHA-256 |
| --- | --- |
| Production source | `a2a2235d1b2f6b3db84f8ef15082b16c71cb699e38678e1ca194f85e9945c8b7` |
| Production diff against base | `7587bcf9d55fa0df27dddda026be5c3ef7488fb2e0739f8d64f115f8593a1c44` |
| Native extension (installed and source package) | `8ea77b560d033412929e3e29e5b585a1e152896b51509fea651e7e7b9eae70ea` |

The initially absent canonical `.venv` was installed with locked dev/reference
dependencies. The unchanged build tool used `maturin build --release --locked
--offline` in a fresh worktree-local Cargo target, with thin LTO and one codegen
unit. Dependency and CUDA JIT caches stayed local; the final capture reused the
CUDA cache from earlier diagnostics. All generated files stayed in this worktree.
[Preflight](diagnostics/compile-cuda-contiguous/development/preflight.log) records
Python 3.12.13, NumPy 2.5.1, PyTorch 2.13.0+cu130, Rust/Cargo 1.92.0, NVIDIA H100,
driver 580.82.07 and compute capability 9.0. Native runtime was explicitly the
local `.venv`'s `nvidia/cu13/lib/libcudart.so.13` (13000), matching reference CUDA
13.0. nvcc 12.6.85 was installed but unused; native kernels use driver-JIT PTX.
Per-command receipts retain physical UUID/index/utilization/memory snapshots.
Ordinary GPU work used `CUDA_VISIBLE_DEVICES=0`; only restoration used `0,1`.

| Check | Result | Log |
| --- | --- | --- |
| Rust default, CUDA hidden | 392 passed; hardware cases return early | [log](diagnostics/compile-cuda-contiguous/development/rust-default-hardened.log) |
| Rust Python bindings, GPU 0 | 406 passed, including zero-native-execution negatives | [log](diagnostics/compile-cuda-contiguous/development/rust-bindings-hardened.log) |
| New compiled layout graphlets, GPU 0 | 9 passed; 1 two-device skip | [log](diagnostics/compile-cuda-contiguous/development/contiguous-hardened.log) |
| Device/context restoration, GPUs 0/1 | 3 passed | [log](diagnostics/compile-cuda-contiguous/development/two-device-hardened.log) |
| Existing CPU/CUDA compiler regressions | 189 passed; 7 two-device skips | [log](diagnostics/compile-cuda-contiguous/development/compiler-regressions-hardened.log) |
| Existing eager CUDA regressions | 55 passed; 10 two-device skips | [log](diagnostics/compile-cuda-contiguous/development/cuda-eager-regressions.log) |
| CPU layout/reference and documentation checks | 113 passed | [log](diagnostics/compile-cuda-contiguous/development/cpu-layout-hardened.log) |
| Hardware-test portability, CUDA hidden | 1 metadata test passed; 9 hardware skips | [log](diagnostics/compile-cuda-contiguous/development/no-device-final.log) |

Formatting, Clippy with warnings denied (default and Python bindings), and
installed-wheel provenance passed. The [audit](diagnostics/compile-cuda-contiguous/development/audit.json)
checks final source/test/native hashes and installed Python source identity.
Eager CUDA checks preceded the last frontend hardening and used the identical
native binary; compiler checks were repeated afterward. The
[inventory](diagnostics/compile-cuda-contiguous/development/inventory.json),
[source manifest](diagnostics/compile-cuda-contiguous/development/source-files.json),
[environment](diagnostics/compile-cuda-contiguous/development/environment.sh.txt)
and per-command receipts provide exact reproduction inputs and commands.

Earlier attempts are preserved in the same bundle: Rust type inference and
Clippy length failures, missing method-guard registration, a dtype rejected
earlier than the first test expected, an incorrectly named CPU test module,
and an existing negation test's changed guard-error ordering. The separately
reproduced dynamic-declaration negative initially reached a blocked native
hook; the final check rejects it before native execution. No failed result
was converted into a pass or removed. Evaluators, fixed cases/weights,
benchmark evidence and Burner-managed progress remain unchanged.
