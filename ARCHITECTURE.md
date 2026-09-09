# Architecture

`pytorch-rs` exposes a PyTorch-shaped Python package backed by a native Rust
tensor core. The native implementation is intentionally small today: tensors
carry strided CPU `float32` storage and optional native CUDA `float32` storage,
Python-facing metadata objects, selected CPU operators, and limited eager
reverse-mode autograd. CUDA storage, transfers, and same-shape contiguous addition work without Python.

## Source Map

| Area | Source files | Notes |
| --- | --- | --- |
| Crate entry | [src/lib.rs](src/lib.rs) | Declares the Rust modules and re-exports `Tensor`, `TensorError`, `DType`, `Device`, and `MemoryFormat`. Python-only modules are gated behind `python-bindings`. |
| Storage | [src/storage.rs](src/storage.rs) | Owns `Storage`, native CPU/CUDA payload dispatch, the CPU `f32` payload, inline scalar storage, owned vectors, and mutex-backed leaf-gradient buffers. |
| CUDA backend | [src/cuda.rs](src/cuda.rs), [src/cuda/pointwise.rs](src/cuda/pointwise.rs), [src/cuda/pool.rs](src/cuda/pool.rs), [src/cuda/add.ptx](src/cuda/add.ptx) | Loads the optional CUDA runtime, owns device allocations, restores the calling thread's device, and performs synchronous host-to-device and device-to-host transfers from/to Rust buffers. A front cache retains at most 32 buffers / 64 MiB across devices; optional private pools budget another 256 MiB of unused backing per device, as detailed below. The optional driver loads an embedded general float32 addition kernel. Python only discovers optional wheel library paths. |
| Dimension reduction kernels | [src/reduction.rs](src/reduction.rs) | Uses layout-aware slices and four-level float32 accumulation for rank-2 single-axis sums and means. [src/parallel.rs](src/parallel.rs) manages an explicit worker budget; large reductions split independent outputs without changing their accumulation order. |
| Tensor layout | [src/tensor.rs](src/tensor.rs) | `Tensor` stores shared storage plus shape, strides, storage offset, element count, output number, view grad state, and optional autograd metadata. It also implements contiguity, view, stride, indexing, and materialization helpers. Integer-size and list/tuple-section split and chunk reuse `partition_dimension` slice views and one shared multi-output backward node per call; explicit sections use `SplitWithSizes` metadata. |
| Metadata types | [src/dtype.rs](src/dtype.rs), [src/device.rs](src/device.rs), [src/memory_format.rs](src/memory_format.rs) | Define the currently compiled native dtype/device/memory-format enums and query behavior. |
| Tensor operations | [src/tensor.rs](src/tensor.rs) | Constructors, unary and binary kernels, reductions, matrix multiply, layout transforms, and backward kernels live with the core tensor representation. |
| Autograd | [src/tensor.rs](src/tensor.rs), [src/autograd_node.rs](src/autograd_node.rs), [src/grad_mode.rs](src/grad_mode.rs) | `AutogradMeta`, `GradFn`, `SavedTensor`, backward traversal, VJP kernels, Python-visible node names, and thread-local grad-mode context state. |
| PyO3 module | [src/python.rs](src/python.rs) | Defines `torch_rs.torch_rs`, `PyTensorBase`, `PyTensor`, Python argument parsing, Tensor methods, top-level native functions, module constants, and module initialization. |
| Top-level built-ins | [src/python_variable_functions.rs](src/python_variable_functions.rs) | Builds the immutable `_VariableFunctionsClass` and registers PyTorch-style top-level descriptors such as `torch.sqrt`. The callbacks call implementations in `src/python.rs`. |
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
including offsets, scalars and empty tensors. Native float32 CUDA negation
also accepts contiguous inputs, offsets, scalars, and empties. The
[src/cuda/neg.ptx](src/cuda/neg.ptx) sign-bit kernel uses the shared 64-bit
grid-stride launcher, fresh device storage, device guards, and synchronized
completion. Noncontiguous CUDA negation, CUDA autograd, and compiled CUDA
negation remain unsupported; see [validation](docs/cuda-neg-validation.md).
Other CUDA materialization and
arithmetic reject at the operation boundary.
Direct public CUDA factory allocation remains rank-1 float32 zeros without
autograd. Transfers do not add dtype conversions, autograd (even under
`no_grad` for grad-requiring inputs), asynchronous copies, CUDA-to-CUDA copies,
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
`src/cuda/pointwise.rs`. Both document their safety invariants. The runtime
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
