# Architecture

`pytorch-rs` has a Rust tensor engine, an optional PyO3 extension, and a Python
package that assembles a PyTorch-shaped public API. This is a source map, not a
support matrix: consult [FEATURES.md](FEATURES.md) before assuming a dtype,
device, backend, compiler feature, or operation is implemented.

```text
Python caller
  -> package exports and compatibility wrappers
  -> PyO3 argument binding, override dispatch, and exception conversion
  -> Rust Tensor operations and autograd recording
  -> shared Storage plus shape/stride/offset metadata
```

The public Rust API enters at [`src/lib.rs`](src/lib.rs) and can call the tensor
core directly, without the Python layers.

## Source map

| Area | Primary files | Responsibility |
| --- | --- | --- |
| Build and assembly | [`Cargo.toml`](Cargo.toml), [`pyproject.toml`](pyproject.toml), [`src/lib.rs`](src/lib.rs) | Rust crate exports and feature gates, Maturin configuration, and the `torch_rs.torch_rs` extension/package boundary. |
| Storage and layout | [`src/storage.rs`](src/storage.rs), [`src/tensor.rs`](src/tensor.rs), [`src/dtype.rs`](src/dtype.rs), [`src/device.rs`](src/device.rs), [`src/memory_format.rs`](src/memory_format.rs) | `Storage` owns values; `Tensor` owns shared-storage identity, shape, strides, offset, and element count. View operations change metadata, while materializing operations allocate storage. |
| Tensor operations | [`src/tensor.rs`](src/tensor.rs), [`src/tensor_error.rs`](src/tensor_error.rs) | Native constructors, indexing, views, elementwise kernels, reductions, matmul, layout planning, and typed failures. This is the numerical semantics boundary. |
| Autograd | [`src/tensor.rs`](src/tensor.rs), [`src/autograd_node.rs`](src/autograd_node.rs), [`src/grad_mode.rs`](src/grad_mode.rs) | Graph metadata, saved tensors, VJP kernels, topology traversal, leaf accumulation, public node names, and thread-local `no_grad` state. |
| PyO3 bindings | [`src/python.rs`](src/python.rs), [`src/python_variable_functions.rs`](src/python_variable_functions.rs), [`src/python_argument_schema.rs`](src/python_argument_schema.rs), [`src/python_tensor_errors.rs`](src/python_tensor_errors.rs) | Native `Tensor` methods and module initialization, immutable top-level callables, argument parsing, `__torch_function__` dispatch, object conversion, and Python exception mapping. Focused `src/python_*.rs` modules split out reusable types and method groups. |
| Python wrappers | [`python/torch_rs/__init__.py`](python/torch_rs/__init__.py), [`python/torch_rs/_tensor.py`](python/torch_rs/_tensor.py), [`python/torch_rs/functional.py`](python/torch_rs/functional.py), [`python/torch_rs/nn/functional.py`](python/torch_rs/nn/functional.py), [`python/torch_rs/autograd/__init__.py`](python/torch_rs/autograd/__init__.py), [`python/torch_rs/overrides.py`](python/torch_rs/overrides.py) | Package assembly, compatibility-only policy, composed operations, namespace APIs, Python-owned methods, and override helpers. Native exports are imported first; selected Python wrappers then add or replace public behavior. |
| Tests | [`tests/tensor_baseline.rs`](tests/tensor_baseline.rs), [`tests/autograd.rs`](tests/autograd.rs), [`tests/`](tests/) | Rust invariants live in Rust integration tests. Python contract tests use `test_<feature>.py`; matching `test_<feature>_reference.py` files compare against the pinned PyTorch reference when needed. |

## Representative path: `torch.exp(x)`

1. **Public export.** [`src/python_variable_functions.rs`](src/python_variable_functions.rs)
   defines and installs the immutable `exp` built-in on the native extension.
   [`python/torch_rs/__init__.py`](python/torch_rs/__init__.py) imports that export
   into the public package.
2. **Validation and dispatch.** In [`src/python.rs`](src/python.rs),
   `exp_variable_function` selects `UnaryOutOperation::EXP`.
   `bind_unary_out_arguments` validates arity, input aliases, tensor or
   `__torch_function__` operands, and the optional `out` keyword. Dispatch runs
   overrides first; the native path rejects a non-`None` `out` before execution.
3. **Kernel execution.** `UnaryOutOperation::EXP` points to `CoreTensor::exp`.
   [`Tensor::exp`](src/tensor.rs) calls `unary_map(f32::exp)`, which clones the
   shape, selects output strides from the input layout, allocates fallibly, and
   materializes logical values into fresh storage before constructing the result.
4. **Error translation.** Layout, indexing, and allocation failures originate as
   [`TensorError`](src/tensor_error.rs). The binding maps them with
   [`tensor_error`](src/python_tensor_errors.rs); call-shape/type validation has
   already produced Python exceptions at the binding boundary.
5. **VJP recording.** After the forward result exists, `Tensor::exp` calls
   `finish_saved_output_unary_vjp`. When the input requires gradients and grad
   mode is enabled, it stores the input graph edge, saved output values,
   `AutogradNode::Exp`, and `apply_exp_vjp`; otherwise it returns an untracked
   result. `AutogradNode::Exp` supplies the Python name `ExpBackward0`.
6. **Backward use.** `run_backward` topologically visits the recorded node,
   `apply_saved_output_unary` invokes `apply_exp_vjp`, and the VJP multiplies the
   upstream gradient by the saved exponential output before leaf accumulation.
   [`python/torch_rs/_tensor.py`](python/torch_rs/_tensor.py) validates the
   Python-owned `Tensor.backward` options before calling the native engine.

## File-oriented change checklist

- Put storage representation or ownership changes in `src/storage.rs`; keep
  shape/stride/offset invariants and view/materialization decisions in
  `src/tensor.rs`.
- Add numerical semantics to `src/tensor.rs`. Add or update `TensorError` and
  its `Display` text in `src/tensor_error.rs`, then map it deliberately in
  `src/python_tensor_errors.rs`.
- For differentiable operations, add the node identity in
  `src/autograd_node.rs` and the recording plus VJP path in `src/tensor.rs`.
  Check behavior with `no_grad`, views, empty tensors, graph reuse, and gradient
  accumulation as applicable.
- Add Tensor methods and shared binding logic in `src/python.rs`; add top-level
  built-ins in `src/python_variable_functions.rs`. Put reusable argument/type
  machinery in the focused `src/python_*.rs` modules and register new modules in
  `src/lib.rs`.
- Use `python/torch_rs/` for Python-only compatibility, namespace assembly, and
  compositions. Update public imports and `__all__` where the public API changes.
- Cover native invariants in Rust tests and Python-visible behavior in a focused
  `tests/test_<feature>.py`; add a reference counterpart for differential
  semantics. Update `FEATURES.md` only when the supported contract changes.

## Focused validation

Use the environment setup in [README.md](README.md). Start with the narrowest
checks that cover the edited layer, then run the full Python script for public
API changes:

```bash
cargo fmt --check
cargo test --test tensor_baseline
cargo test --test autograd
cargo test exponential
PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --all-targets --features python-bindings
.venv/bin/python -m unittest discover -s tests -p 'test_exp*.py'
./scripts/test-python.sh
```

The single-file Python command assumes the current worktree's extension is
already installed; `scripts/test-python.sh` rebuilds it and verifies provenance.
