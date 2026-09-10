# Captured native CUDA matrix products

`torch_rs.compile(fn, backend="eager", fullgraph=True)` records and executes
`@`, positional `Tensor.matmul`/`Tensor.__matmul__`, and positional
`torch_rs.matmul` (including imported aliases). Both operands must be exact
native, same-device, contiguous rank-2 CUDA float32 tensors without gradients.
`fullgraph=False` supports the same straight-line programs with `dynamic=None`;
`fullgraph=True` also accepts `dynamic=False/True` under the existing rank and
exact-stride guards. Unsupported programs raise, with no eager fallback.

```python
import torch_rs as torch

def product(left, right):
    return -(left * 0.5) @ (right * -2)

compiled = torch.compile(product, backend="eager", fullgraph=True)
a = torch.ones((2, 3)).to("cuda:0")
b = torch.ones((3, 4)).to("cuda:0")
assert compiled(a, b).cpu().tolist() == [[3.] * 4] * 2
assert compiled(a * 2, b).cpu().tolist() == [[6.] * 4] * 2
```

The bytecode frontend emits an actual `matmul` graph node. Whole-graph metadata
validation checks the inner dimension, output size, layout and device before
execution, including dynamic cache hits. The native bridge independently calls
`Tensor::matmul` and its existing cuBLAS SGEMM implementation. Float32
accumulation, disabled TF32, independent offsets, overlapping inputs, contiguous
singleton layouts, empty/zero-inner products, fresh contiguous output storage,
input preservation, device restoration and completion/lifetimes have the same
contract as [eager CUDA matmul](cuda-matmul.md). No Python body, PyTorch or
libTorch operator runs on the candidate execution path.

Negation, scalar multiplication, equal-shape addition and trailing-vector
addition compose before/after products, including repeated products and
list/tuple tensor outputs with aliases. Existing same-module helpers and guarded
global tensors are supported. A bias can be a global capture; three positional
tensor inputs remain unsupported. Cache guards include shape, stride, offset,
device, dtype, gradients, code/callable identity, method descriptors and global
captures. Rejected captures do not consume cache entries; reset and recompilation
limits retain their existing semantics.

CPU matmul capture, CUDA gradients, other dtypes, vector/batched/noncontiguous
products, mixed devices, kwargs/out, `torch.mm` (still CPU-only eagerly), new
views inside the graph, reductions and other unary operations remain explicit
rejections. General default-backend/Inductor capture and fusion are not added.

## Reproduction and delivery

Use the worktree-local locked environment in [CONTRIBUTING](../CONTRIBUTING.md).
For measurements, start from a clean checkout with an empty build target and
new output paths.
Set local `TMPDIR`, `XDG_CACHE_HOME`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`,
`CARGO_HOME`, `CARGO_TARGET_DIR`, `CUDA_CACHE_PATH`, `TRITON_CACHE_DIR` and
`TORCHINDUCTOR_CACHE_DIR` before setup. Install a release wheel (not a copied
editable installation); verify `sys.executable`, `sys.base_prefix`, NumPy,
PyTorch, every installed `torch_rs` Python source, and the native extension
resolve inside this worktree. Select local `TORCH_RS_CUDART` and
`TORCH_RS_CUBLAS` libraries from the reference environment.

```bash
export CUDA_VISIBLE_DEVICES=0
export PYO3_PYTHON="$PWD/.venv/bin/python"
.venv/bin/maturin build --release --locked --out target/wheels
uv pip install --python .venv/bin/python --force-reinstall target/wheels/*.whl
.venv/bin/python -B -m unittest tests.test_compile_cuda_matmul tests.test_compile_cuda_matmul_diagnostic
cargo test --locked --features python-bindings --lib compiled_matmul_bridge
cargo test --locked --test cuda_matmul
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -B -m unittest tests.test_compile_cuda_matmul.CompileCudaMatmulDeviceTests
.venv/bin/python -B scripts/diagnose_compile_cuda_matmul.py \
  --build-record target/capture/build-record.json --output target/compiled-matmul.json
```

Obtain the source/binary-bound build record from the unchanged
[CUDA math capture procedure](cuda-matmul.md#reproduction). Rerun all six fixed
math cases at its three predeclared seeds, the unchanged 38-case compiler corpus,
and existing compatibility checks. The installed-interpreter test uses `-I`,
blocks PyTorch imports and Python-body execution, and checks resolved imports.
Permanent numerical tests include generated rectangles/squares, offsets, overlap,
aliases, changed data, empty/singleton/zero-inner cases, K through one million,
and overflow counterexamples; two-GPU checks run separately.

The separate diagnostic compares public native capture (`backend="eager"`)
with stock PyTorch 2.13 `torch.compile` (`backend="inductor"`, default mode), both
`fullgraph=True, dynamic=False`. Twelve predeclared equal-weight cells combine
fixed/generated held-out squares/rectangles with pure and composed programs.
It records wrapper/first-call costs separately, both execution orders, ten
warmups and 31 five-call synchronized samples per order, raw samples and
median/p10/p90/min/max, materialized outputs, exact input preservation and
changed-data checks. Compiler disk caches are retained across orders and their
initial state is recorded: reversed-order first calls are not claimed to be
cold-cache compiles. Each cell contributes `min(1, reference/native median)`;
the fixed geometric mean is zero if any planned cell fails or is missing.
No slow or failed cells are dropped. Failed output checks retain preceding
raw calls and complete samples, including the unfinished block and the count
of calls whose outputs passed validation. Attempts cannot overwrite earlier reports.
`--allow-dirty` labels development evidence and cannot qualify a clean commit.

The diagnostic records source/binary hashes, command, local imports/libraries,
Python/PyTorch/CUDA/GPU/driver and compiler identities, including the selected
local Triton/PTX assembler path, version and hash. Native matmul uses cuBLAS,
not nvcc; pointwise kernels use driver-JIT PTX. This unfused eager-backend graph
diagnostic is separate from the fixed four-shape CUDA scoring workload and does
not establish general Inductor parity.

[Current measured evidence](diagnostics/composite-matmul-unflatten-l1/postcommit-cbcbd84/README.md)
contains results, source/native/runtime identities, command receipts and raw samples.
