# pytorch-rs

`pytorch-rs` is a native Rust tensor and deep-learning engine exposed through a PyTorch-compatible Python API. It pursues PyTorch semantics, broad feature coverage, and competitive performance. It is an early experimental implementation, not yet a PyTorch replacement.

The project is improved through [Burner](https://github.com/bobrenjc93/burner): each increment is developed in an isolated branch, independently reviewed, and measured against the same base revision before it can merge.

## Current baseline

Python package names may contain a hyphen, but Python identifiers may not. The package is therefore installed as `torch-rs` and imported as `torch_rs`, conventionally aliased to `torch` for drop-in-style code:

```python
import torch_rs as torch

x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
y = torch.ones([2, 2])
result = (x + y).relu()
assert result.tolist() == [[0.0, 3.0], [4.0, 0.0]]

# Transposes are native shared-storage views. Tensor.T reverses every dimension,
# Tensor.mT swaps the final matrix dimensions, and Tensor.t() accepts rank <= 2.
assert x.T.shape == (2, 2)
assert x.t().shape == (2, 2)
batched_matrices = torch.zeros((4, 2, 3))
assert batched_matrices.mT.shape == (4, 3, 2)

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
```

The CPU core provides `float32` tensors, checked construction including copied one-dimensional numeric PEP 3118 buffers, constant-filled creation, layout queries, stride-aware indexing, metadata-only transpose and squeeze views, compatible reshape views, PyTorch-compatible read-only `Tensor.T` and `Tensor.mT` views, rank-limited `Tensor.t()`, `Tensor.flatten()`, `Tensor.ravel()`, and `torch.flatten()`, native `Tensor.contiguous()` materialization for row-major, channels-last, and channels-last-3d storage, independent deep cloning, exact `Tensor.equal()` and `torch.equal()` comparison, unary negation, broadcast tensor and real-scalar addition, subtraction, multiplication, and true division, ReLU, sum, and rank-2 matrix multiplication. `Tensor.T` reverses the complete shape and stride tables, `Tensor.mT` swaps only the final two dimensions, and `Tensor.t()` is an unwarned alias view for scalars, vectors, and matrices while rejecting higher ranks; all retain shared storage, offsets, dtype, and device. Flatten preserves shared storage for stride-compatible ranges and eagerly creates an independent contiguous copy otherwise; ravel always returns a new one-dimensional Tensor wrapper, aliasing row-contiguous storage and materializing other layouts. Already-matching contiguous calls preserve Python object identity and shared storage; materializing calls copy logical values into independent storage with offset zero. `Tensor.squeeze()`, `Tensor.squeeze(dim)`, and `torch.squeeze(input, dim)` retain shared storage, strides, and offsets just like PyTorch. This intentionally small surface gives the campaign an honest starting point. The compatibility contract is the observable Python API; the Rust library is its implementation engine.

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
uv venv --clear
uv sync --locked --no-install-project
uv pip install --python .venv/bin/python 'maturin>=1.14,<2'
.venv/bin/maturin develop --release --uv
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

The checked-in tests are only the public floor. Burner also uses independent generated workloads and side-by-side `torch_rs`/`torch` differential runs.

## License

MIT

<!-- burner-progress:start -->
## Burner evaluation progress

![Burner evaluation progress](docs/burner-evaluation-progress.svg)

Burner updates this graph atomically after each successful merge. It validates a complete finite 0–100 score map for every enabled evaluation, then upserts the canonical baseline-commit or `pr:<number>` key; retrying a merge replaces the existing point instead of duplicating it. Missing or malformed scores abort artifact generation before any file is written. The [raw versioned history](docs/burner-evaluation-history.json) records this merge-coupled policy.
<!-- burner-progress:end -->
