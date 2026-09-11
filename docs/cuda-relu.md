# Native CUDA ReLU and eager graph capture

`Tensor.relu()` accepts native contiguous CUDA float32 tensors without gradients,
including scalars, empty tensors, higher ranks and contiguous offset views.
Noncanonical singleton strides are accepted when the tensor is contiguous.
Results own independent storage with canonical strides and storage offset zero.
The existing eager `torch.relu` and out-of-place `torch.nn.functional.relu`
aliases share this behavior; both have real H100 differential coverage.

```python
import torch_rs as torch

def activate(x):
    return (x * 0.5).relu().sum(1)

compiled = torch.compile(activate, backend="eager", fullgraph=True)
x = torch.tensor([[-2., 4.], [6., -8.]]).to("cuda:0")
assert compiled(x).cpu().tolist() == [2., 3.]
```

Only zero-argument `Tensor.relu()` is captured. It composes with negation,
scalar multiplication, addition/bias, matmul, row sums and the existing bounded
transpose/contiguous/reshape paths. Packing and view capture keep their existing
rank limits; ReLU itself has no additional rank limit. CPU behavior and ReLU
backward stay unchanged. Functional capture, in-place ReLU, CUDA gradients,
other dtypes, fusion, new backends and general accelerator/compiler parity are
outside this change. These checks establish semantics, not timing parity.

The embedded driver-JIT PTX uses integer compare/select: negative non-NaN bits
map to positive zero, including negative zero, subnormals and infinity. Positive
values, positive infinity and all positive/negative quiet/signaling NaN payloads
retain their exact bits. Tests upload and download raw buffers without Python
float conversions. No `max` or flush-to-zero instruction is used.

Storage reuses `unary_pointwise`/`unary_output` bounds, device guards, owner
lifetimes and legacy-stream synchronization. Native graph planning validates all
nodes before any operation, including single-node ReLU. Metadata inference,
exact method identity, same-module helpers, captured globals, CPU/CUDA cache
separation, dynamic rank/stride/device/offset guards and repeated-output
metadata-pair checks retain the existing [compiler contract](compile-cuda-add.md).

## Development validation

The baseline was reproduced before production edits on exact main
`9407a208d12a6a650ced579e943154849973e037` with a fresh release extension:
`x.relu()`, `(x*0.5).relu().sum(1)` and `(x@w).relu()` passed reference eager and
compiled execution, but failed native eager/capture. The pinned reference also
confirmed all IEEE cases above. Development captures live in
[the development evidence index](diagnostics/cuda-relu/development/index.json).
The [replayable audit](diagnostics/cuda-relu/development/audit-evidence.py) checks
artifact/build-log hashes, the final source/test manifest and baseline/acceptance
results. The evidence index records source, native, build, input, command and artifact
hashes, retaining unsuccessful checks. It does not represent a clean candidate
commit or a scoring run.

The environment uses worktree-local uv-managed CPython 3.12.14, locked dev and
PyTorch 2.13.0 reference dependencies, Rust 1.92.0 and release thin LTO with one
codegen unit. Ordinary H100 tests use `CUDA_VISIBLE_DEVICES=0`; restoration tests
use only `0,1`. Receipts record UUIDs, runtime library/version, driver, compiler,
and utilization/memory snapshots. A snapshot does not reserve hardware.
ReLU uses embedded PTX compiled by the driver; the extension build does not
invoke nvcc. Broader compiler regression tests retain their existing reference
and private-kernel compilation paths. Final standalone Rust checks explicitly
select the worktree wheel's CUDA 13 runtime through `TORCH_RS_CUDART`.

| Check | Result |
| --- | --- |
| Fresh release identity | Native bytes and all 59 Python files match wheel/checkout |
| ReLU H100 eager/compiled differentials and boundaries | 13 passed; two-device cases run separately |
| Two-device restoration | 2 passed on visible devices `0,1` |
| CUDA hidden | All 15 hardware-only Python tests skipped clearly |
| Existing CUDA regression suite | 163 passed, 15 device-specific skips |
| CPU ReLU/autograd/layout regressions | 196 passed |
| Broad compiler regression run | 601 passed, 12 skips; one new fixture error corrected and rechecked in the ReLU suite |
| Rust default / Python bindings | 395 / 417 passed, including storage and zero-operation graph prevalidation |
| Clippy / formatting | Default and all-feature Clippy with warnings denied; formatting and whitespace checks passed |

The broad compiler suite's sole error was a new test that omitted the native
bridge's `RuntimeError` for an invalid node index. Its corrected rejection test
passes; unrelated assertions were retained. Earlier attempts also preserve the
single-node Python-dispatch failure (fixed by native whole-graph routing), two
Rust fixture compilation errors, a Clippy documentation warning, and a test
that incorrectly rejected the bridge's existing list-of-indices form. Every
attempt remains in the command receipts; none is presented as a passing run.

Focused commands (after building the exact-source release wheel):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest tests.test_cuda_relu tests.test_compile_cuda_relu -v
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest tests.test_cuda_relu.CudaReluDeviceTests tests.test_compile_cuda_relu.CompileCudaReluDeviceTests -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest tests.test_cuda_relu tests.test_compile_cuda_relu -v
CUDA_VISIBLE_DEVICES=0 cargo test --locked --no-default-features
CUDA_VISIBLE_DEVICES=0 cargo test --locked --features python-bindings
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo fmt --all --check
```

Old rejection tests that used CUDA ReLU solely as an unsupported placeholder now
use CUDA `absolute`/`abs`. Their rejection, cache and prevalidation assertions
remain intact. Evaluator definitions, frozen38, performance workloads, hardware
matrix and historical evidence remain unchanged; PR1970/PR1971 remain separate
unadopted review campaigns.

After Burner commits the implementation, a separate fresh release build and
clean-commit capture must be published without overwriting this development
bundle. Commit creation, review, delivery and full merge gates belong to Burner;
they cannot be claimed by this uncommitted worktree run.
