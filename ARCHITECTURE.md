# Architecture

`pytorch-rs` exposes a PyTorch-shaped Python package backed by a native Rust
tensor core. The native implementation is intentionally small today: tensors
carry strided CPU `float32` storage and optional native CUDA `float32` storage,
Python-facing metadata objects, selected CPU operators, and limited eager
reverse-mode autograd. Native CUDA storage, transfers, bounded view packing,
contiguous addition, negation and scalar multiplication, matrix-plus-trailing-vector
addition, row sums, and [rank-2 matmul](docs/cuda-matmul.md) execute without Python.

## Source Map

| Area | Source files | Notes |
| --- | --- | --- |
| Crate entry | [src/lib.rs](src/lib.rs) | Declares the Rust modules and re-exports `Tensor`, `TensorError`, `DType`, `Device`, and `MemoryFormat`. Python-only modules are gated behind `python-bindings`. |
| Storage | [src/storage.rs](src/storage.rs) | Owns `Storage`, native CPU/CUDA payload dispatch, the CPU `f32` payload, inline scalar storage, owned vectors, and mutex-backed leaf-gradient buffers. |
| CUDA backend | [src/cuda.rs](src/cuda.rs), [src/cuda/pointwise.rs](src/cuda/pointwise.rs), [src/cuda/pool.rs](src/cuda/pool.rs), [src/cuda/blas.rs](src/cuda/blas.rs), [src/cuda/add.ptx](src/cuda/add.ptx), [src/cuda/add_trailing_vector.ptx](src/cuda/add_trailing_vector.ptx), [src/cuda/neg.ptx](src/cuda/neg.ptx), [src/cuda/mul_scalar.ptx](src/cuda/mul_scalar.ptx), [src/cuda/sum_rows.rs](src/cuda/sum_rows.rs), [src/cuda/sum_rows.ptx](src/cuda/sum_rows.ptx) | Loads the optional CUDA runtime, owns device allocations, restores the calling thread's device, and performs synchronous host-to-device and device-to-host transfers from/to Rust buffers, plus bounded same-device vector copies. A front cache retains at most 32 buffers / 64 MiB across devices; optional private pools budget another 256 MiB of unused backing per device, as detailed below. The optional driver loads embedded contiguous float32 addition, negation, scalar multiplication and matrix row reduction kernels. Matmul lazily loads native cuBLAS SGEMM with float32 accumulation and context-owned handles. Python only discovers optional wheel library paths. |
| CUDA view packing | [src/cuda/contiguous.ptx](src/cuda/contiguous.ptx), [src/cuda.rs](src/cuda.rs) | Copies float32 bits in logical order for positive-stride rank-1/rank-2 views. The checked layout planner and shared unary storage helper own bounds, allocation, device restoration and completion; copy-requiring reshape reuses this path. |
| Dimension reduction kernels | [src/reduction.rs](src/reduction.rs) | Uses layout-aware slices and four-level float32 accumulation for CPU rank-2 single-axis sums and means. CUDA contiguous row sums use the [native reduction geometry and PTX](docs/cuda-sum-rows.md). [src/parallel.rs](src/parallel.rs) manages an explicit worker budget; large reductions split independent outputs without changing their accumulation order. |
| Tensor layout | [src/tensor.rs](src/tensor.rs) | `Tensor` stores shared storage plus shape, strides, storage offset, element count, output number, view grad state, and optional autograd metadata. It also implements contiguity, view, stride, indexing, and materialization helpers. Integer-size and list/tuple-section split and chunk reuse `partition_dimension` slice views and one shared multi-output backward node per call; explicit sections use `SplitWithSizes` metadata. |
| Metadata types | [src/dtype.rs](src/dtype.rs), [src/device.rs](src/device.rs), [src/memory_format.rs](src/memory_format.rs) | Define the currently compiled native dtype/device/memory-format enums and query behavior. |
| Tensor operations | [src/tensor.rs](src/tensor.rs) | Constructors, unary and binary kernels, reductions, matrix multiply, layout transforms, and backward kernels live with the core tensor representation. |
| Autograd | [src/tensor.rs](src/tensor.rs), [src/autograd_node.rs](src/autograd_node.rs), [src/grad_mode.rs](src/grad_mode.rs) | `AutogradMeta`, `GradFn`, `SavedTensor`, backward traversal, VJP kernels, Python-visible node names, and thread-local grad-mode context state. |
| PyO3 module | [src/python.rs](src/python.rs) | Defines `torch_rs.torch_rs`, `PyTensorBase`, `PyTensor`, Python argument parsing, Tensor methods, top-level native functions, module constants, and module initialization. |
| Top-level built-ins | [src/python_variable_functions.rs](src/python_variable_functions.rs) | Builds the immutable `_VariableFunctionsClass` and registers PyTorch-style top-level descriptors such as `torch.sqrt`. The callbacks call implementations in `src/python.rs`. |
| Unflatten bindings | [src/python_unflatten.rs](src/python_unflatten.rs), [src/python_tensor_shape.rs](src/python_tensor_shape.rs) | Top-level schema and override dispatch precede a direct call to shared Rust view construction and error translation. The Tensor method retains its own binding checks; neither path duplicates native view/autograd logic. |
| Error translation | [src/tensor_error.rs](src/tensor_error.rs), [src/python_tensor_errors.rs](src/python_tensor_errors.rs) | `TensorError` is the native error vocabulary; Python bindings translate it to the closest Python exception class. |
| Python package shell | [python/torch_rs/__init__.py](python/torch_rs/__init__.py) | Imports the native extension, exposes `_C`, patches package-level compatibility helpers, and binds Python submodules. |
| Compile bytecode frontend | [python/torch_rs/_compile_bytecode.py](python/torch_rs/_compile_bytecode.py) | Normalizes and validates the narrow CPython 3.10-3.14 straight-line bytecode subset used by public `torch.compile(..., backend="eager", fullgraph=True)` and no-break `fullgraph=False`, then emits operations through `CompileTraceRecorder`. It owns opcode compatibility only. |
| Compile trace IR | [python/torch_rs/_compile_trace.py](python/torch_rs/_compile_trace.py) | Defines the private immutable `CompileTraceGraph`, Tensor proxy recording helpers, centralized layout and broadcast metadata planning, and native graph execution dispatch. It stays independent of CPython bytecode opcodes. |
| Private CUDA benchmark lane | [python/torch_rs/_cuda_buffer.py](python/torch_rs/_cuda_buffer.py), [python/torch_rs/_cuda_benchmark_tensor.py](python/torch_rs/_cuda_benchmark_tensor.py), [python/torch_rs/_cuda_pointwise_kernel.py](python/torch_rs/_cuda_pointwise_kernel.py), [python/torch_rs/_cuda_pointwise_reduce_workload.py](python/torch_rs/_cuda_pointwise_reduce_workload.py) | Owns benchmark-only CUDA runtime probes, private buffers, synchronized metadata wrappers, H100 kernels, and the exact prepared-executor evidence path for the release benchmark. This lane is intentionally outside the native Rust tensor/device model and must not be expanded into general tensor semantics without first adding a real Rust-side backend/device abstraction. |
| Python wrappers | [python/torch_rs/_tensor.py](python/torch_rs/_tensor.py), [python/torch_rs/functional.py](python/torch_rs/functional.py), [python/torch_rs/nn/functional.py](python/torch_rs/nn/functional.py), [python/torch_rs/autograd/__init__.py](python/torch_rs/autograd/__init__.py), [python/torch_rs/overrides.py](python/torch_rs/overrides.py) | Add Python-owned methods and functions when Python-level validation, dispatch, or namespace compatibility is better expressed outside Rust. |
| Public scope docs | [README.md](README.md), [FEATURES.md](FEATURES.md), [docs/supported-surface.md](docs/supported-surface.md), [BENCHMARKING.md](BENCHMARKING.md) | README gives setup and scope; the other documents record API coverage and benchmark policy. |
| Tests | [tests/](tests) and Rust `#[cfg(test)]` modules | Python tests compare public behavior against PyTorch references where available; Rust unit tests exercise core layout, storage, and autograd internals. |

## Operation Trace: `torch.sqrt(x)`

1. Public import: [python/torch_rs/__init__.py](python/torch_rs/__init__.py)
   imports every native export from `torch_rs.torch_rs`, so `torch.sqrt` is the
   native descriptor registered for the package.
2. Descriptor registration: [src/python_variable_functions.rs](src/python_variable_functions.rs)
   creates `_VariableFunctionsClass`, installs a `sqrt` method definition, and
   adds that descriptor to the extension module.
3. Top-level entry: [src/python.rs](src/python.rs) handles the callback in
   `sqrt_variable_function`, forwarding to the shared unary-with-optional-`out`
   path.
4. Python argument validation: [src/python.rs](src/python.rs) selects the
   legacy single `input` argument, accepts an exact native `Tensor` or a
   `__torch_function__` override candidate, validates duplicate and unexpected
   keywords, and records whether `out` was provided.
5. Dispatch: [src/python.rs](src/python.rs) first offers the call to the active
   `TorchFunctionMode` and operand overrides. The native path rejects `out=`
   for `sqrt`, borrows the `PyTensor`, and calls `CoreTensor::sqrt`.
6. Tensor method entry: `x.sqrt()` goes through `PyTensorBase::sqrt` in
   [src/python.rs](src/python.rs). After method-mode dispatch, it also borrows
   the same `CoreTensor` and calls `CoreTensor::sqrt`.
7. Native layout and kernel: `Tensor::sqrt` in [src/tensor.rs](src/tensor.rs)
   calls `unary_map(sqrt_value)`. `unary_map` clones the result shape, chooses
   output strides from the input layout, materializes logical values into fresh
   storage, and applies `sqrt_value` element by element.
8. Error translation: native failures return `TensorError`; the Python boundary
   maps them through [src/python_tensor_errors.rs](src/python_tensor_errors.rs).
   Shape, allocation, layout, and autograd errors become `RuntimeError`; index
   and dimension errors become `IndexError`. The unsupported `out=` case is
   raised directly as a Python `RuntimeError` before entering the kernel.
9. VJP recording: after the forward kernel, `Tensor::sqrt` calls
   `finish_saved_input_unary_vjp` with `AutogradNode::Sqrt` and
   `apply_sqrt_vjp`. If the input records gradients and grad mode is enabled,
   the output receives a `GradFn::SavedInputUnary` edge with a saved input
   snapshot.
10. Backward execution: `Tensor.backward()` is a Python wrapper in
    [python/torch_rs/_tensor.py](python/torch_rs/_tensor.py) that validates
    unsupported options and calls the native no-argument method. The native
    method checks for a one-element gradient root, runs `run_backward`, applies
    the saved-input unary VJP, and accumulates leaf gradients through
    `Storage::from_shared_gradient`.

## Change Checklist

- `src/tensor.rs`: add or adjust native semantics first, including layout,
  allocation, aliasing, autograd recording, and Rust unit coverage near related
  tests.
- `src/tensor_error.rs`: add native error variants and messages only when the
  core needs a new failure mode.
- `src/python_tensor_errors.rs`: map any new `TensorError` variant to the
  Python exception class that matches the observed public behavior.
- `src/python.rs`: update PyO3 parsing, method bindings, top-level native
  helpers, docstrings, and module initialization for Python-visible changes.
- `src/python_variable_functions.rs`: register PyTorch-style top-level
  descriptors when a public `torch.*` callable should behave like a built-in.
- `python/torch_rs/*.py`: use Python wrappers for namespace wiring,
  Python-level validation, `__torch_function__` dispatch helpers, or APIs that
  do not need a native tensor kernel.
- `tests/test_*.py`: add focused Python behavior and reference tests for public
  API changes, including unsupported argument and error cases.
- `FEATURES.md` and [docs/supported-surface.md](docs/supported-surface.md):
  update supported-surface documentation when the user-visible API changes.
- Private CUDA benchmark helpers: keep exact H100 workload gates, evidence
  schemas, runtime allocation, and kernel-launch code isolated in
  `python/torch_rs/_cuda_*`. Do not route ordinary tensors through this lane or
  add more benchmark shapes there as a substitute for a Rust-side backend/device
  abstraction.

## Focused Validation Commands

Run the narrowest checks that cover the files you changed, then broaden if the
operation touches shared parsing, layout, or autograd paths:

```bash
cargo fmt --check
cargo test --all-targets
PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --all-targets --features python-bindings
VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
.venv/bin/python -m unittest tests.test_sqrt tests.test_top_level_sqrt tests.test_autograd_backward
```

## Native CUDA storage boundary

`Tensor::cuda_zeros_float32`, `Tensor::try_copy_cpu_to_cuda`, and
`Tensor::try_copy_cuda_to_cpu` are available without `python-bindings`.
Host uploads accept CPU float32 tensors with `requires_grad=False` and explicit
indexed CUDA destinations, including scalar, empty, and multidimensional
tensors. They preserve dense strides and pack sparse views in preserve-format
dimension order, allocating and copying only logical elements with output
storage offset zero. Shared `Arc<Storage>` ownership keeps allocations
alive through narrow/select/reshape/transpose views. CPU copies preserve dense
strides and pack non-dense views; empty views perform no device transfer.
Sparse transfers allocate the packed output plus layout metadata. Logical
contiguous runs copy directly into that output; regularly spaced runs batch
through `cudaMemcpy2D`. Tall selections with short, closely spaced runs use
at most 256 KiB of staging and at most eight times the logical transfer size
to avoid the copy engine's per-row cost. Large gaps are always skipped; no
allocation or transfer grows with an arbitrary backing span. Packed outputs
retain CPU clone's dimension ordering, including transposed and selected views.
CUDA addition accepts same-shape contiguous float32 inputs on one device,
including offsets, scalars and empty tensors. Eager and compiled addition accept
contiguous `(M, N)` and `(N,)` on that device in either operand order. The
64-bit grid-stride trailing-vector kernel preserves independent bounds checks,
device restoration, fresh storage, and completion before return; empty outputs
launch no kernel. Other broadcasts, noncontiguous layouts, other dtypes,
nonunit alpha, concrete out, and autograd remain excluded. The compiler metadata
planner admits the same bounded relation, derives singleton/empty output strides
with its elementwise planner, and guards inputs before execution. Its private
native bridge delegates independent validation to `Tensor::add`, sharing the
eager kernel boundary. Compiled add syntax remains operator/positional method
only; compiler alpha/out forms remain unsupported.
Native float32 CUDA negation
also accepts contiguous inputs, offsets, scalars, and empties. The
[src/cuda/neg.ptx](src/cuda/neg.ptx) sign-bit kernel uses the shared 64-bit
grid-stride launcher, fresh device storage, device guards, and synchronized
completion. The shared unary graph operation also executes negation under
bounded marker-free eager capture, composing with CUDA addition without fusion.
Native metadata guards device, layout, dtype, autograd and offsets; value outputs
have canonical contiguous strides and fresh CUDA storage at offset zero. The
executor preflights the whole graph before launching any operation, including
on dynamic cache hits. This is not a general Inductor compiler or a performance
parity claim. Noncontiguous CUDA negation and CUDA autograd remain unsupported;
see [capture scope](docs/compile-cuda-add.md) and [kernel validation](docs/cuda-neg-validation.md).
The compiler's `matmul` binary node plans rank-2 `(M,K)@(K,N)` output
metadata and validates inner dimensions and output size before any node runs.
`_compile_trace_binary` calls native `Tensor::matmul`, which independently checks
CUDA storage/layout/dtype/gradients and submits cuBLAS SGEMM. Cache hits execute
recorded nodes on current inputs; they never call the original Python program.
Matmul result strides are canonical, including singleton/empty cases. Existing
method/callable, capture, offset, and device guards apply. Supported pointwise
nodes can precede/follow matmul without fusion; a third tensor must be a guarded
global capture because graphs still accept at most two positional inputs.
See [scope and reproduction](docs/compile-cuda-matmul.md).

Compiled parameterless `Tensor.contiguous()` routes through
`src/python_compile_cuda_graph.rs` to native `Tensor::try_contiguous`, reusing
PR1974's positive-stride rank-1/rank-2 packer. The frontend admits strided CUDA
inputs but validates each arithmetic operand's contiguous layout separately.
Both frontend and Rust planning validate the whole graph before execution;
Rust layouts track offsets as well as shape/strides. Contiguous no-ops preserve
metadata, storage and the original Python owner, including intermediate aliases,
scalars, empties, singleton strides and higher ranks. Packs own fresh storage.
Callable/global/device/gradient/cache guards remain live. See
[compiled layout scope and validation](docs/compile-cuda-contiguous.md).

CUDA `Tensor.t()` and constant-axis `Tensor.transpose(dim0, dim1)` add bounded
rank-0/1/2 view nodes to the same executor. The bytecode binder and proxy record
transpose axes separately from arithmetic payloads. `src/tensor_cuda_graph.rs`
reverses or swaps shape/strides during planning, reusing native axis normalization;
execution calls the existing checked `Tensor::t`/`Tensor::transpose` primitives. Storage
is shared without storage allocation, CPU staging or a kernel launch. Unlike
contiguous no-ops, each view node receives a new Python owner even when metadata
is unchanged; output references reuse that owner. Packing and arithmetic consume
the resulting layout normally. Frontend declarations and native shape/stride
fields reject bool/float substitutions before equality or execution. See the
[compiled transpose guide](docs/compile-cuda-t.md) for scope and non-scoring diagnostics.

Eager scalar multiplication routes `BinaryOperation::Multiply.apply_scalar` to
`Tensor::mul_scalar`, shared scalar output-stride planning, and the native
[src/cuda/mul_scalar.ptx](src/cuda/mul_scalar.ptx) 64-bit grid-stride kernel.
It multiplies contiguous device float32 storage directly, with round-to-nearest
and no flush-to-zero. The shared unary storage helper owns allocation, bounds,
device guards, completion, and failure cleanup for both multiplication and negation.
Results are fresh, offset zero, and preserve the reference scalar operation
stride ordering, including singleton dimensions. No compiler capture, tensor-tensor
CUDA multiplication, out variants, or autograd support is added. See
[validation](docs/cuda-mul-scalar-validation.md). Other CUDA arithmetic rejects
at the operation boundary.

CUDA `contiguous()` preserves identity and metadata for already-contiguous
float32 tensors without gradients, including scalar, empty and higher-rank
layouts. Noncontiguous positive-stride rank-1/rank-2 views pack directly into
fresh same-device contiguous storage at offset zero, preserving float32 bits
without CPU staging. Copy-requiring `reshape()` reuses that packing path;
view-only reshape remains an alias. Other dtypes, gradient tracking, higher-rank
packing and channels-last materialization remain unsupported. See the
[layout contract](docs/supported-surface.md) and
[clean-commit H100 checks](docs/cuda-contiguous-validation.md).

Direct public CUDA factory allocation supports rank-1 and rank-2 float32 zeros
without autograd on explicit indexed devices. Contiguous rank-1 CUDA float32
tensors with `requires_grad=False` support `clone()` and same-device
`to(copy=True)` with preserve format, including empty and offset views. Native
device-to-device copies own independent storage at offset zero, complete before
return, and restore the calling thread's device. Ordinary same-device `to()`
returns the original object. Cross-device copies remain unsupported, as do
noncontiguous, scalar and rank-2-or-higher `clone()` and `to(copy=True)` copies.
Transfers do not add dtype conversions,
autograd (even under `no_grad` for grad-requiring inputs), asynchronous copies,
or other CUDA math. See the [exact transfer contract](docs/supported-surface.md)
for Python argument forms and unsupported boundaries.

Native `cat`, `stack`, backward, and gradient mutation reject unsupported devices
before accessing storage or modifying graph state. Use `try_sum`, `try_equal`,
and `try_with_requires_grad` for fallible native dispatch; they return
`TensorError::UnsupportedDevice` for unsupported CUDA operations. Existing CPU
convenience APIs (`sum`, `PartialEq`, and `with_requires_grad`) retain their
signatures and panic at the operation boundary on unsupported devices, as
`Clone` does when `try_clone` fails. CUDA `Debug` displays metadata without
reading device storage. None of these operations implicitly transfers to CPU.

CUDA runtime ABI calls live in `src/cuda.rs`; driver kernel ABI calls live in
`src/cuda/pointwise.rs`; native cuBLAS ABI calls live in `src/cuda/blas.rs`. All document their safety invariants. The runtime
library remains loaded for the process lifetime. Zero-fill uses the legacy default
stream. Host uploads synchronize that stream after `cudaMemcpy` so even
pageable H2D copies complete before storage is published; blocking D2H copies
establish host visibility without a separate whole-device synchronization.
Allocation reuse never occurs while an alias is
alive. The cache retains at most 32 allocations and 64 MiB across devices. It
reuses the smallest compatible allocation with at most 25% excess capacity,
tracks allocation capacity separately from logical tensor bounds, and evicts
the oldest returned entries when full. Eviction frees each pointer under its
own device guard after releasing the cache lock. This prevents mixed-size
workloads from permanently blocking reuse for later sizes. It does not expose
a public allocator API.

On devices with CUDA memory-pool support, `src/cuda/pool.rs` allocates from
a private pool and enqueues releases on the same legacy stream as all native
uses. Each pool tracks original allocation capacities, including front-cache
entries, under a mutex. Its release threshold is live bytes plus 256 MiB, so
large live inputs do not evict every reusable output at synchronization. CUDA
reclaims unused backing above that threshold at synchronization, subject to
allocation granularity; pending frees may persist until the next wait. The
front cache is globally bounded, and each device has this separate unused-pool
budget. Other libraries' default pools are untouched. Missing pool symbols or
unsupported devices retain guarded `cudaMalloc`/`cudaFree`. A launch/completion
or pooled-release failure disables reuse; uncertain pool allocations are
quarantined rather than recycled.
See [optional runtime setup](docs/troubleshooting.md#optional-native-cuda-runtime).

## Native CUDA row reduction

Before selecting launch geometry, Rust mirrors TensorIterator's signed 32-bit
byte-offset splits: recursively halve contiguous regions, rows before columns,
and visit lower halves first. Later pieces of a split row accumulate into its
existing output in float32. Each piece selects its own geometry; stream ordering
allows one scratch allocation to be reused until final synchronization.

Contiguous rank-2 float32 `sum(dim=1/-1)` uses four float32 accumulators,
aligned vector loads with scalar heads/tails, descending shuffle-down, and
geometry-dependent shared-memory x/y reductions. Rust selects block geometry
from row count and width and queries the current device's multiprocessor and
thread capacity for CTA splitting. Wide-row partials use fresh device scratch;
a second legacy-stream kernel reduces them in y-then-x order. Scratch and
output survive both launches and synchronization, including failure paths.
There is no float64 promotion, host value reduction, or PyTorch dependency.
Empty axes, offsets, fresh output, guards, and unsupported dimensions are
specified in the [row-sum contract](docs/cuda-sum-rows.md).

## Native CUDA addition

All three public forms (`x + y`, `x.add(y)`, `torch.add(x, y)`) reach
`Tensor::add`. CUDA dispatch validates same device, float32 dtype, exact shape,
contiguity and no gradient tracking before allocation. CPU broadcasting and
VJP paths retain their existing implementation. CUDA results normalize singleton
and empty strides to contiguous layout, with independent storage and offset zero.

`Storage::cuda_add_float32` borrows both allocation owners and validates the
logical offset ranges. `CudaFloat32Storage::add` allocates through the existing
cache under a device guard. The embedded PTX kernel uses a 64-bit grid-stride
loop over the element count, aligned four-float loads/stores with a scalar tail
(or the scalar kernel for unaligned offsets), and round-to-nearest float32 addition without
flush-to-zero. NVIDIA's driver JIT compiles it; no nvcc, NVRTC, Python, or CPU
computation is used. Modules/functions are cached by runtime context and kept
alive with the driver library for process lifetime. A fresh host thread that
reuses cached storage initializes its guarded runtime context before driver
module lookup; it cannot assume an earlier `cudaMalloc` ran on that thread.
External context destruction
or device reset while native state exists is unsupported.

Launch uses the explicit legacy default stream, ordered after native zero-fill
and transfers. Repeated exact argument sets may use the bounded graph replay
cache in [src/cuda/replay.rs](src/cuda/replay.rs). It holds at most 64 executable
entries and 64 recent signatures globally. A hit requires matching current live
input/output pointers, element count, and context-specific function; metadata
never owns tensor storage. Callers retain the executable through the stream
wait, and per-executable host launch locks protect concurrent graph access.
Eviction destroys metadata in its owning context and restores the caller's
context. Optional graph setup failures fall back to direct kernel launch.
The caller synchronizes that stream even after a launch failure,
while all three allocations are still live. Thus chained additions, immediate
drops, and cache reuse need no deferred ownership or event bookkeeping. A launch
or completion error disables allocation reuse; later ordinary allocations use
guarded `cudaFree`, while uncertain pooled allocations are quarantined. Inputs never alias the fresh output, while shared or overlapping
input views are permitted. Empty tensors skip pointer arithmetic and launch.
Current-device selection is restored on success, error, and uncached frees.
Public stream selection and external asynchronous writes remain unsupported.

See `tests/cuda_add.rs` and `tests/test_cuda_add.py` for native and H100
reference coverage, including held-out nonzero shapes, subnormals/signed zeros,
view ownership, thread/cache reuse, independent-stream reads, and two-device
restoration. Addition deliberately synchronizes per call; device-resident
compute diagnostics must report this overhead with symmetric reference timing.

CUDA-add latency, sustained throughput, cache saturation and large-output
diagnostics are documented in [docs/cuda-add-diagnostics.md](docs/cuda-add-diagnostics.md).
