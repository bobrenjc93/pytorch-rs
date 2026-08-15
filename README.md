# pytorch-rs

`pytorch-rs` is a native Rust tensor and deep-learning engine exposed through a PyTorch-compatible Python API. It pursues PyTorch semantics, broad feature coverage, and competitive performance. It is an early experimental implementation, not yet a PyTorch replacement.

The project is improved through [Burner](https://github.com/bobrenjc93/burner): each increment is developed in an isolated branch, independently reviewed, and measured against the same base revision before it can merge.

## Current baseline

Python package names may contain a hyphen, but Python identifiers may not. The package is therefore installed as `torch-rs` and imported as `torch_rs`, conventionally aliased to `torch` for drop-in-style code:

```python
import torch_rs as torch

x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
y = torch.ones([2, 2])
result = torch.relu(x + y)
assert result.tolist() == [[0.0, 3.0], [4.0, 0.0]]
functional_result = torch.nn.functional.relu(x + y)
assert functional_result.tolist() == result.tolist()
assert torch.is_signed(input=x)
assert torch.get_device(input=x) == -1

# Transposes are native shared-storage views. Tensor.T reverses every dimension,
# Tensor.mT and the real-valued Tensor.mH swap the final matrix dimensions,
# Tensor.swapdims()/torch.swapdims() and Tensor.swapaxes()/torch.swapaxes() swap
# any two, while Tensor.t()/torch.t() accept rank <= 2.
assert x.T.shape == (2, 2)
assert x.t().shape == (2, 2)
assert torch.t(input=x).shape == (2, 2)
batched_matrices = torch.zeros((4, 2, 3))
assert batched_matrices.mT.shape == (4, 3, 2)
assert batched_matrices.mH.shape == (4, 3, 2)
assert torch.swapdims(input=batched_matrices, dim0=0, dim1=-1).shape == (3, 2, 4)
assert torch.swapaxes(input=batched_matrices, axis0=0, axis1=-1).shape == (3, 2, 4)
assert batched_matrices.permute(-1, 0, 1).shape == (3, 4, 2)
assert torch.permute(input=batched_matrices, dims=[-1, 0, 1]).shape == (3, 4, 2)

# PyTorch-compatible view calls use the same conventional alias.
batched = torch.zeros((1, 2, 1, 3))
matrix = torch.squeeze(batched, dim=(0, 2))
assert matrix.shape == (2, 3)

# Flatten compatible ranges as views and materialize non-contiguous ranges.
features = torch.flatten(matrix, start_dim=0, end_dim=1)
assert features.shape == (6,)

# Materialize a view in native row-major or channel-last storage.
view = torch.transpose(torch.zeros((2, 3, 4, 5)), 0, 3)
packed = view.contiguous()
nhwc_storage = packed.contiguous(memory_format=torch.channels_last)
assert packed.is_contiguous()
assert nhwc_storage.is_contiguous(memory_format=torch.channels_last)

# Default and preserve-format CPU calls retain the exact existing CPU object;
# an explicit different layout can materialize new storage.
assert view.cpu() is view
cpu_nhwc = view.cpu(memory_format=torch.channels_last)
assert cpu_nhwc.is_contiguous(memory_format=torch.channels_last)
```

The CPU core provides `float32` tensors, checked construction including copied one-dimensional numeric PEP 3118 buffers, constant-filled creation, layout queries, read-only `Tensor.is_sparse` and `Tensor.is_sparse_csr` strided-layout introspection, read-only `Tensor.is_cpu` and `Tensor.is_cuda` device introspection, PyTorch-compatible `Tensor.cpu()` identity and memory-format conversion for the supported CPU device, read-only `Tensor.is_quantized` dtype introspection, dtype-backed `Tensor.is_signed()` and `torch.is_signed()` queries, stride-aware indexing, arbitrary metadata-only `Tensor.permute()` and `torch.permute()` views, metadata-only transpose, `Tensor.swapdims()`/`torch.swapdims()` and `Tensor.swapaxes()`/`torch.swapaxes()`, and squeeze views, compatible `Tensor.reshape()` and `Tensor.reshape_as()` view-or-copy transforms, PyTorch-compatible read-only `Tensor.T`, `Tensor.mT`, and real-valued `Tensor.mH` views, rank-limited `Tensor.t()` and `torch.t()`, `Tensor.flatten()`, `Tensor.ravel()`, and `torch.flatten()`, native `Tensor.contiguous()` materialization for row-major, channels-last, and channels-last-3d storage, no-op identity `Tensor.type_as()` conversion and read-only `Tensor.real` identity for the supported type, independent deep cloning, exact `Tensor.equal()` and `torch.equal()` comparison, identity `Tensor.positive()`/`torch.positive()` and unary `+`, unary `-`, `Tensor.neg()`, and its `Tensor.negative()` alias, broadcast tensor and real-scalar addition, subtraction, multiplication (including `Tensor.mul()` and its `Tensor.multiply()` alias), and true division, `Tensor.relu()`, `torch.relu()`, and out-of-place `torch.nn.functional.relu(input, inplace=False)`, plus sum and rank-2 matrix multiplication. The functional ReLU delegates to the same native kernel; `inplace=True` is rejected before the input can be mutated. `Tensor.permute()` accepts variadic dimensions, a tuple or list, and the `dims=` form; top-level `torch.permute(input, dims)` accepts a tuple or list. Both normalize negative axes and delegate all shape, stride, offset, storage, and autograd behavior to the native permutation engine. `Tensor.T` reverses the complete shape and stride tables; `Tensor.mT` and, because the supported dtype is real, `Tensor.mH` swap only the final two dimensions through the same native transpose path. Both swap aliases use the transpose engine for arbitrary dimension pairs, while `Tensor.t()` and `torch.t()` are unwarned alias views for scalars, vectors, and matrices and reject higher ranks. All of these permutation and transpose-family calls retain shared storage, offsets, dtype, and device. `Tensor.reshape_as(other)` passes `other.shape` through the same reshape engine, so storage sharing, materialization, strides, and autograd behavior match `Tensor.reshape(other.shape)`. Flatten preserves shared storage for stride-compatible ranges and eagerly creates an independent contiguous copy otherwise; ravel always returns a new one-dimensional Tensor wrapper, aliasing row-contiguous storage and materializing other layouts. Already-matching contiguous, `cpu()`, `type_as()`, `real`, `Tensor.positive()`, `torch.positive()`, and unary `+` calls preserve Python object identity and shared storage; CPU channel-last requests that need a different layout use the same checked materialization and autograd path as `contiguous()`. `Tensor.squeeze()`, `Tensor.squeeze(dim)`, and `torch.squeeze(input, dim)` retain shared storage, strides, and offsets just like PyTorch. This intentionally small surface gives the campaign an honest starting point. The compatibility contract is the observable Python API; the Rust library is its implementation engine.

## Non-negotiable evaluation rules

- Correctness gates performance. A workload contributes no performance credit unless its outputs, shapes, dtypes, errors, aliasing, and edge cases pass differential checks.
- Benchmarks compare equivalent Python calls through `torch_rs` and `torch`, with inputs created outside the timed region. Warmup, synchronization, sampling, and thread counts must be symmetric.
- The suite covers multiple sizes, ranks, dtypes, layouts, broadcast patterns, and thread counts. Results use per-workload ratios and geometric aggregation so one favorable kernel cannot hide broad regressions.
- Missing or unsupported capabilities score zero; they are never removed from the denominator.
- Fixed seeds make failures reproducible, while generated and held-out shapes prevent implementations from specializing for a short public list.
- Checksums or materialized outputs prevent dead-code elimination and lazy-work deferral.
- Compile time and dependency installation are reported separately from steady-state execution.
- Burner-authored feature branches may not weaken, delete, skip, special-case, or rewrite evaluation infrastructure. Benchmark changes are separate, human-reviewed campaign changes and never earn implementation impact in the same comparison.
- Native platform libraries and explicit hardware backends are valid implementation techniques. Forwarding production tensor operations to Python or PyTorch is not.

See [BENCHMARKING.md](BENCHMARKING.md) and [FEATURES.md](FEATURES.md) for the full campaign contract.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --doc
uv venv --clear --python 3.12
uv sync --locked --no-install-project
uv pip install --python .venv/bin/python 'maturin>=1.14,<2'
PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --all-targets --features python-bindings -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --all-targets --features python-bindings
env -u CONDA_PREFIX VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" .venv/bin/maturin develop --release --uv
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

To validate the Python package from one freshly built, exact-HEAD release wheel,
run:

```bash
./scripts/test-python-exact-head.sh
```

This recreates `.venv` with Python 3.12, installs the locked reference group,
clears inherited environment and import markers, force-installs the new wheel,
and verifies its native-extension provenance before checking for PyTorch 2.13.0
and running the full unittest suite. It preserves `CUDA_VISIBLE_DEVICES`, so the
existing hardware-aware tests use available CUDA hardware and skip their CUDA
cases when PyTorch reports none.

The checked-in tests are only the public floor. Burner also uses independent generated workloads and side-by-side `torch_rs`/`torch` differential runs.

## License

MIT

<!-- burner-progress:start -->
## Burner evaluation progress

![Burner evaluation progress](docs/burner-evaluation-progress.svg)

Burner updates this graph atomically after each successful merge. It validates a complete finite 0–100 score map for every enabled evaluation, then upserts the canonical baseline-commit or `pr:<number>` key; retrying a merge replaces the existing point instead of duplicating it. Missing or malformed scores abort artifact generation before any file is written. The [raw versioned history](docs/burner-evaluation-history.json) records this merge-coupled policy.
<!-- burner-progress:end -->
