# Architecture

`pytorch-rs` exposes a PyTorch-shaped Python package backed by a native Rust
tensor core. The native implementation is intentionally small today: tensors
carry strided CPU `float32` storage, Python-facing metadata objects, selected
operators, and limited eager reverse-mode autograd.

## Source Map

| Area | Source files | Notes |
| --- | --- | --- |
| Crate entry | [src/lib.rs](src/lib.rs) | Declares the Rust modules and re-exports `Tensor`, `TensorError`, `DType`, `Device`, and `MemoryFormat`. Python-only modules are gated behind `python-bindings`. |
| Storage | [src/storage.rs](src/storage.rs) | Owns `Storage`, the CPU `f32` payload, inline scalar storage, owned vectors, and mutex-backed leaf-gradient buffers. |
| Tensor layout | [src/tensor.rs](src/tensor.rs) | `Tensor` stores shared storage plus shape, strides, storage offset, element count, output number, view grad state, and optional autograd metadata. It also implements contiguity, view, stride, indexing, and materialization helpers. |
| Metadata types | [src/dtype.rs](src/dtype.rs), [src/device.rs](src/device.rs), [src/memory_format.rs](src/memory_format.rs) | Define the currently compiled native dtype/device/memory-format enums and query behavior. |
| Tensor operations | [src/tensor.rs](src/tensor.rs) | Constructors, unary and binary kernels, reductions, matrix multiply, layout transforms, and backward kernels live with the core tensor representation. |
| Autograd | [src/tensor.rs](src/tensor.rs), [src/autograd_node.rs](src/autograd_node.rs), [src/grad_mode.rs](src/grad_mode.rs) | `AutogradMeta`, `GradFn`, `SavedTensor`, backward traversal, VJP kernels, Python-visible node names, and thread-local `no_grad` state. |
| PyO3 module | [src/python.rs](src/python.rs) | Defines `torch_rs.torch_rs`, `PyTensorBase`, `PyTensor`, Python argument parsing, Tensor methods, top-level native functions, module constants, and module initialization. |
| Top-level built-ins | [src/python_variable_functions.rs](src/python_variable_functions.rs) | Builds the immutable `_VariableFunctionsClass` and registers PyTorch-style top-level descriptors such as `torch.sqrt`. The callbacks call implementations in `src/python.rs`. |
| Error translation | [src/tensor_error.rs](src/tensor_error.rs), [src/python_tensor_errors.rs](src/python_tensor_errors.rs) | `TensorError` is the native error vocabulary; Python bindings translate it to the closest Python exception class. |
| Python package shell | [python/torch_rs/__init__.py](python/torch_rs/__init__.py) | Imports the native extension, exposes `_C`, patches package-level compatibility helpers, and binds Python submodules. |
| Python wrappers | [python/torch_rs/_tensor.py](python/torch_rs/_tensor.py), [python/torch_rs/functional.py](python/torch_rs/functional.py), [python/torch_rs/backends/](python/torch_rs/backends/), [python/torch_rs/nn/functional.py](python/torch_rs/nn/functional.py), [python/torch_rs/autograd/__init__.py](python/torch_rs/autograd/__init__.py), [python/torch_rs/overrides.py](python/torch_rs/overrides.py) | Add Python-owned methods and functions when Python-level validation, dispatch, stateful backend preference proxies, or namespace compatibility is better expressed outside Rust. |
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
