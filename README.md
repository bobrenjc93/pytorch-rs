# pytorch-rs

`pytorch-rs` is a native Rust tensor and deep-learning engine exposed through a PyTorch-compatible Python API. It pursues PyTorch semantics, broad feature coverage, and competitive performance. It is an early experimental implementation, not yet a PyTorch replacement.

The project is improved through [Burner](https://github.com/bobrenjc93/burner): each increment is developed in an isolated branch, independently reviewed, and measured against the same base revision before it can merge.

## Current baseline

Python package names may contain a hyphen, but Python identifiers may not. The package is therefore installed as `torch-rs` and imported as `torch_rs`, conventionally aliased to `torch` for drop-in-style code:

```python
import mmap

import torch_rs as torch

x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
y = torch.ones([2, 2])
result = torch.relu(x + y)
assert result.tolist() == [[0.0, 3.0], [4.0, 0.0]]
product = torch.matmul(input=x, other=y)
assert product.tolist() == [[1.0, 1.0], [-1.0, -1.0]]
assert torch.get_float32_matmul_precision() == "highest"
scaled = torch.multiply(input=2.0, other=x)
assert scaled.tolist() == [[-2.0, 4.0], [6.0, -8.0]]
exponential = torch.exp(input=x)
assert exponential.shape == x.shape
functional_result = torch.nn.functional.relu(x + y)
assert functional_result.tolist() == result.tolist()
assert torch.nn.functional.dropout(x, training=False) is x
assert torch.nn.functional.dropout(x, p=0, training=True, inplace=True) is x
fully_dropped = torch.nn.functional.dropout(x, p=1, training=True)
assert fully_dropped.tolist() == [[-0.0, 0.0], [0.0, -0.0]]
assert not fully_dropped.is_set_to(x)
signals = torch.zeros((1, 2, 3))
assert torch.nn.functional.dropout1d(signals, p=0, training=True) is signals
feature_maps = torch.zeros((1, 2, 3, 4))
assert torch.nn.functional.dropout2d(feature_maps, training=False) is feature_maps
volumes = torch.zeros((1, 2, 3, 4, 5))
assert torch.nn.functional.dropout3d(volumes, p=0, training=True) is volumes
assert torch.is_signed(input=x)
assert torch.get_device(input=x) == -1
assert torch.cpu.is_available() is True
assert torch.cpu.current_device() == "cpu"
assert torch.cpu.device_count() == 1
assert torch.cpu.synchronize() is None
assert torch.get_num_threads() == 1
assert torch.get_num_interop_threads() == 1
assert torch.compiler.is_compiling() is False
assert torch.compiler.is_dynamo_compiling() is False
assert torch.compiler.is_exporting() is False
assert torch.serialization.get_crc32_options() is True
assert torch.serialization.get_default_mmap_options() == getattr(
    mmap, "MAP_PRIVATE", None
)
assert torch.distributed.is_available() is False
assert torch.distributed.is_initialized() is False
assert torch.distributed.is_mpi_available() is False
assert torch.distributed.is_nccl_available() is False
assert torch.float32.to_real() is torch.float32
limits = torch.finfo()
assert limits == torch.finfo(torch.float)
assert limits.dtype == "float32" and limits.bits == 32
assert torch.can_cast(from_=torch.float, to=torch.float32) is True
assert torch.promote_types(type1=torch.float, type2=torch.float32) is torch.float32
assert torch.broadcast_shapes((2,), [3, 1]) == torch.Size([3, 2])
assert x.dense_dim() == x.ndim
assert x.sparse_dim() == 0
assert x.is_pinned() is False
assert x.output_nr == 0

# Transposes are native shared-storage views. Tensor.T reverses every dimension,
# real-valued Tensor.H transposes matrices, Tensor.mT and real-valued Tensor.mH
# swap the final matrix dimensions, Tensor.swapdims()/torch.swapdims() and
# Tensor.swapaxes()/torch.swapaxes() swap any two, while Tensor.t()/torch.t()
# accept rank <= 2.
assert x.T.shape == (2, 2)
assert x.H.shape == (2, 2)
assert x.t().shape == (2, 2)
assert torch.t(input=x).shape == (2, 2)
batched_matrices = torch.zeros((4, 2, 3))
assert batched_matrices.mT.shape == (4, 3, 2)
assert batched_matrices.mH.shape == (4, 3, 2)
assert torch.adjoint(input=batched_matrices).shape == (4, 3, 2)
assert torch.swapdims(input=batched_matrices, dim0=0, dim1=-1).shape == (3, 2, 4)
assert torch.swapaxes(input=batched_matrices, axis0=0, axis1=-1).shape == (3, 2, 4)
assert batched_matrices.permute(-1, 0, 1).shape == (3, 4, 2)
assert torch.permute(input=batched_matrices, dims=[-1, 0, 1]).shape == (3, 4, 2)
assert batched_matrices.movedim(0, -1).shape == (2, 3, 4)
assert batched_matrices.moveaxis(0, -1).shape == (2, 3, 4)
assert torch.movedim(batched_matrices, source=0, destination=-1).shape == (2, 3, 4)
assert torch.moveaxis(batched_matrices, source=0, destination=-1).shape == (2, 3, 4)
matrix_view = batched_matrices.select(dim=-3, index=1)
assert matrix_view.shape == (2, 3)
assert torch.select(batched_matrices, dim=-3, index=1).is_set_to(matrix_view)

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

The CPU core provides `float32` tensors, checked construction including copied one-dimensional numeric PEP 3118 buffers, constant-filled creation, layout queries, `Tensor.dense_dim()` and `Tensor.sparse_dim()` strided-layout dimension metadata, no-argument `Tensor.is_pinned()` metadata for the exclusively pageable CPU storage model, no-argument `Tensor.is_distributed()` metadata for supported local tensors, leaf-only `Tensor.retain_grad()` as a no-op for tensors with `requires_grad=True`, and read-only `Tensor.retains_grad` metadata that remains `False` because retained non-leaf gradients are unsupported, dimension-zero `Tensor.select()`/`torch.select()` single first-axis views and `Tensor.unbind()`/`torch.unbind()` first-axis views, read-only `Tensor.output_nr` metadata (`0` for single-output operations, with PyTorch-compatible multi-output indices for grad-tracked `Tensor.unbind()`, `torch.unbind()`, and tensor iteration; `Tensor.chunk` and `torch.chunk` are not exposed), read-only `Tensor.is_sparse` and `Tensor.is_sparse_csr` strided-layout introspection, read-only `Tensor.is_cpu` and `Tensor.is_cuda` device introspection, PyTorch-compatible `Tensor.cpu()` identity and memory-format conversion for the supported CPU device, read-only `Tensor.is_quantized` dtype introspection, dtype-backed `Tensor.is_signed()` and `torch.is_signed()` queries, canonical `torch.float32.to_real()` identity, float32-only `torch.finfo` metadata, stride-aware indexing, `Tensor.view()`/`Tensor.view_as()` shared-storage views, arbitrary metadata-only `Tensor.permute()` and `torch.permute()` views, integer-axis `Tensor.movedim()`/`Tensor.moveaxis()`, `torch.movedim()`, and top-level `torch.moveaxis()` views, metadata-only transpose, `Tensor.swapdims()`/`torch.swapdims()` and `Tensor.swapaxes()`/`torch.swapaxes()`, and squeeze views, compatible `Tensor.reshape()` and `Tensor.reshape_as()` view-or-copy transforms, PyTorch-compatible read-only `Tensor.T`, `Tensor.mT`, and real-valued `Tensor.H`/`Tensor.mH` views, `Tensor.adjoint()`/`torch.adjoint()` matrix-adjoint views, rank-limited `Tensor.t()` and `torch.t()`, `Tensor.flatten()`, `Tensor.ravel()`, and `torch.flatten()`, native `Tensor.contiguous()` materialization for row-major, channels-last, and channels-last-3d storage, no-op identity `Tensor.type_as()` conversion and read-only `Tensor.real` identity for the supported type, independent deep cloning, exact `Tensor.equal()` and `torch.equal()` comparison, identity `Tensor.positive()`/`torch.positive()` and unary `+`, unary `-`, `Tensor.neg()`, and its `Tensor.negative()` alias, broadcast tensor and real-scalar addition, subtraction, multiplication through `*`, `Tensor.mul()`, `Tensor.multiply()`, `torch.mul()`, and the distinct top-level `torch.multiply()` builtin, and true division, `Tensor.relu()`, `torch.relu()`, out-of-place `torch.nn.functional.relu(input, inplace=False)`, deterministic `torch.nn.functional.dropout(input, p=1, training=True, inplace=False)` plus its identity cases, rank-3 identity-only `torch.nn.functional.dropout1d(input, p, training, inplace)`, rank-4 identity-only `torch.nn.functional.dropout2d(input, p, training, inplace)`, and rank-5 identity-only `torch.nn.functional.dropout3d(input, p, training, inplace)`, plus sum and rank-2 matrix multiplication through `@`, `Tensor.matmul()`, and `torch.matmul()`. Top-level `torch.mul()` and `torch.multiply()` accept tensor/tensor or tensor/real-scalar operands in either order and reuse the same broadcast and autograd kernels; their `out` forms and scalar-only multiplication remain unsupported. The named matrix-multiplication calls intentionally reuse the existing rank-2, no-`out` path rather than exposing PyTorch's broader rank and output-tensor variants. The functional ReLU delegates to the same native kernel; `inplace=True` is rejected before the input can be mutated. Functional dropout, rank-3 dropout1d, rank-4 dropout2d, and rank-5 dropout3d return the exact input object, including its storage, layout, and autograd history, when `training=False`, `p=0`, or the input is empty. For nonempty standard dropout, out-of-place `training=True, p=1` delegates to native scalar multiplication by positive zero, producing independent layout-preserving storage with signed-zero and autograd behavior matching PyTorch. Probabilities strictly between zero and one, nonidentity inplace calls, and all nonidentity dropout1d, dropout2d, and dropout3d calls remain explicitly unsupported; the feature-dropout variants also reject all non-rank-3, non-rank-4, and non-rank-5 inputs, respectively, instead of exposing PyTorch's optional-batch and lower-rank behavior. No RNG state or top-level dropout API is added. `dtype.to_real()` returns the exact canonical singleton for every supported `float32` descriptor; complex dtypes and their complex-to-real mapping remain unsupported. `torch.finfo()`, `torch.finfo(torch.float32)`, `torch.finfo(torch.float)`, and the corresponding `type=` forms create fresh immutable objects whose limits come from the native dtype metadata; Python's `float` shorthand and all non-float32 dtypes remain unsupported. `Tensor.select(dim, index)` and `torch.select(input, dim, index)` normalize the negative first dimension and delegate values, strides, offsets, aliasing, empty views, and autograd to the native leading integer-index engine; other dimensions remain unsupported. `Tensor.view(shape)` accepts exactly one positional integer or `__index__` value, or one tuple, list, or `torch.Size` (including the sequence `size=` form) and delegates shape inference, strides, offsets, aliasing, empty tensors, compatible noncontiguous layouts, errors, and autograd to the native view engine; integer `size=` keywords, multiple integer arguments, and dtype reinterpretation overloads remain unsupported. `Tensor.permute()` accepts variadic dimensions, a tuple or list, and the `dims=` form; top-level `torch.permute(input, dims)` accepts a tuple or list. Both normalize negative axes and delegate all shape, stride, offset, storage, and autograd behavior to the native permutation engine. `Tensor.movedim(source, destination)`, `Tensor.moveaxis(source, destination)`, `torch.movedim(input, source, destination)`, and `torch.moveaxis(input, source, destination)` reuse that engine for integer axes, including negative dimensions, scalars, empty tensors, offset views, and noncontiguous views; sequence dimensions remain unsupported. `Tensor.T` reverses the complete shape and stride tables; because the supported dtype is real, `Tensor.H` uses that same transpose path for matrices while `Tensor.mT`, `Tensor.mH`, `Tensor.adjoint()`, and `torch.adjoint()` swap the final two dimensions for matrices or batches. Both swap aliases use the transpose engine for arbitrary dimension pairs, while `Tensor.t()` and `torch.t()` are unwarned alias views for scalars, vectors, and matrices and reject higher ranks. All of these permutation and transpose-family calls retain shared storage, offsets, dtype, and device. `Tensor.reshape_as(other)` passes `other.shape` through the same reshape engine, so storage sharing, materialization, strides, and autograd behavior match `Tensor.reshape(other.shape)`. Flatten preserves shared storage for stride-compatible ranges and eagerly creates an independent contiguous copy otherwise; ravel always returns a new one-dimensional Tensor wrapper, aliasing row-contiguous storage and materializing other layouts. Already-matching contiguous, `cpu()`, `type_as()`, `real`, `Tensor.positive()`, `torch.positive()`, and unary `+` calls preserve Python object identity and shared storage; CPU channel-last requests that need a different layout use the same checked materialization and autograd path as `contiguous()`. `Tensor.squeeze()`, `Tensor.squeeze(dim)`, and `torch.squeeze(input, dim)` retain shared storage, strides, and offsets just like PyTorch. This intentionally small surface gives the campaign an honest starting point. The compatibility contract is the observable Python API; the Rust library is its implementation engine.

`Tensor.ravel()` and top-level `torch.ravel()` share the native view-or-copy path: row-contiguous inputs alias storage through a new one-dimensional wrapper, while other layouts are materialized.

`Tensor.exp()` and inference-only `torch.exp()` share the native float32 CPU kernel. Top-level calls accept `out=None`; concrete output tensors and calls that would record an autograd edge remain explicitly unsupported.

`torch.can_cast` and `torch.promote_types` accept the existing `torch.float32`/`torch.float` singleton aliases in positional or canonical keyword forms. Casting between the supported aliases returns `True`, while promotion returns the same canonical singleton. No additional dtype, casting pair, or promotion pair is exposed.

`torch.functional.broadcast_shapes` and its identical top-level `torch.broadcast_shapes` alias compute canonical `torch.Size` results directly from nonnegative Python integer, tuple, list, and `torch.Size` inputs without creating tensors. Symbolic dimensions, tracing, and `torch.broadcast_tensors` remain unsupported.

`torch.cpu.is_available()` is the canonical device-agnostic CPU availability query and returns the exact `True` singleton. `torch.cpu.is_initialized()` likewise returns exact `True`, reflecting that the eager CPU backend is always initialized. `torch.cpu.current_device()` returns the invariant string `"cpu"`, and `torch.cpu.device_count()` reports the single logical CPU device as the exact integer `1`. Because native CPU execution is eager, `torch.cpu.synchronize(device=None)` ignores any device value and returns the exact `None` singleton. These APIs do not probe hardware, environment variables, or PyTorch. CPU streams, events, device mutation, capabilities, AMP, and the rest of the `torch.cpu` namespace remain unsupported.

`torch.get_num_threads()` reports the native engine's fixed single intra-op worker as the exact integer `1`. `torch.get_num_interop_threads()` likewise returns the exact integer `1`, reflecting the absence of a separate inter-op executor. Neither query probes hardware, environment variables, or PyTorch; both thread setters and parallel execution remain unsupported.

`torch.get_float32_matmul_precision()` reports the invariant string `"highest"`, reflecting that the native CPU float32 matrix-multiplication engine has no reduced-precision modes. The setter and its `"high"` and `"medium"` states remain unsupported, and the query does not change native matmul behavior.

`torch.compiler.is_compiling()`, `torch.compiler.is_dynamo_compiling()`, and `torch.compiler.is_exporting()` are eager-state compatibility queries that return the exact `False` singleton without importing PyTorch. `torch.compile`, `torch.export`, and the rest of the compiler namespace remain unsupported.

`torch.serialization.get_crc32_options()` and `torch.serialization.set_crc32_options(compute_crc32)` expose mutable process-global archive-record checksum state without importing PyTorch. The state starts as the exact `True` singleton, the setter returns `None`, and the getter returns the most recently supplied value. `torch.serialization.get_default_mmap_options()` reports the process-global default used by PyTorch for memory-mapped loads: `mmap.MAP_PRIVATE` initially on supported POSIX platforms. `torch.serialization.set_default_mmap_options(flags)` immediately selects `mmap.MAP_PRIVATE` or `mmap.MAP_SHARED` and also acts as a context manager that restores the prior setting on exit. The setter is unavailable on Windows, while `torch.save`, `torch.load`, and the rest of the serialization namespace remain unsupported.

`Tensor.is_distributed()` returns the exact `False` singleton for every supported local CPU tensor without inspecting or changing storage, layout, or autograd state. `torch.distributed.is_available()`, `torch.distributed.is_mpi_available()`, and `torch.distributed.is_nccl_available()` are honest package, MPI, and NCCL backend-capability queries, while `torch.distributed.is_initialized()` exposes the stable default process-group state. All four package queries return the exact `False` singleton without probing hardware, environment variables, or PyTorch. Distributed tensor types, MPI initialization, process-group creation, backend execution, collectives, and every other distributed API remain unsupported.

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
PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --all-targets --features python-bindings -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --all-targets --features python-bindings
./scripts/test-python.sh
```

The developer Python test command builds a release wheel from the current
worktree, force-installs it into `.venv`, verifies the installed native
extension's provenance, and then runs the full unittest suite. If the suite
fails, it reports the resolved interpreter, package, and extension paths.

To validate the Python package from one freshly built, exact-HEAD release wheel,
run:

```bash
./scripts/test-python-exact-head.sh
```

This exports the exact `HEAD` commit to a temporary directory under `target/`,
creates a Python 3.12 environment there, and installs the locked development and
reference dependencies. It clears inherited environment, import, and Python
optimization markers; builds with the locked Maturin and Cargo dependencies;
force-installs the new wheel; and verifies its native-extension provenance
before checking for PyTorch 2.13.0 and running the full unittest suite. Dirty
worktree files are therefore excluded. To keep every artifact inside the
worktree without inheriting Cargo settings, the command uses a fresh Cargo home
and rejects `.cargo/config` files above the archived checkout. It also ignores
external uv configuration and explicitly installs both locked dependency
groups. Git and tar settings are cleared, and every extracted file is checked
against `HEAD` before testing. The committed Rust channel is explicitly selected
and verified, while ambient Cargo, PyO3, and Python runtime settings, including
warning policy, are cleared. The command rejects a symlinked `target/` before
creating artifacts and uses its verified physical path throughout. It preserves
`CUDA_VISIBLE_DEVICES`, so the existing hardware-aware tests use available CUDA
hardware and skip their CUDA cases when PyTorch reports none.

The checked-in tests are only the public floor. Burner also uses independent generated workloads and side-by-side `torch_rs`/`torch` differential runs.

## License

MIT

<!-- burner-progress:start -->
## Burner evaluation progress

![Burner evaluation progress](docs/burner-evaluation-progress.svg)

Burner updates this graph atomically after each successful merge. It validates a complete finite 0–100 score map for every enabled evaluation, then upserts the canonical baseline-commit or `pr:<number>` key; retrying a merge replaces the existing point instead of duplicating it. Missing or malformed scores abort artifact generation before any file is written. The [raw versioned history](docs/burner-evaluation-history.json) records this merge-coupled policy.
<!-- burner-progress:end -->
