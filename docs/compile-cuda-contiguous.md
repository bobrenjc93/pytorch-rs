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
remain unsupported. This adds no compiled reshape or view-creation grammar,
backend, training, fusion or performance-parity claim. Graphlets below are
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

## Worktree validation evidence

Started from verified clean main `22c4e1c4d32126b91ee9f5417842d245c0183fce`.
The [release receipt](diagnostics/compile-cuda-contiguous/development/release-hardened/build-record.json)
binds the final uncommitted source and release wheel. **This is development
evidence, not a clean implementation-commit capture.** Creating commits is
prohibited in this task; Burner must perform the clean-commit recapture during
its normal delivery workflow. No replacement PR or push was attempted.

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
