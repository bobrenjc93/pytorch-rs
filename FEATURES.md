# Feature coverage contract

Feature completeness is measured against stable public PyTorch Python concepts, with supported claims backed by executable side-by-side semantic tests. Documentation, Rust-only capabilities, and stubs do not count.

Fixed top-level weights prevent easy APIs from overwhelming core gaps:

| Area | Weight | Baseline |
| --- | ---: | --- |
| tensor storage, shapes, strides, views, indexing | 15% | CPU `f32` with dimension-zero `Tensor.select`/`torch.select` first-axis views, single-Tensor `torch.atleast_1d`, `torch.atleast_2d`, and `torch.atleast_3d`, one-sequence `Tensor.view` plus `Tensor.view_as`, arbitrary `Tensor.permute`/`torch.permute` views, integer-axis `Tensor.movedim`/`Tensor.moveaxis`, `torch.movedim`, and top-level `torch.moveaxis` views, strided transpose including read-only `Tensor.T`, `Tensor.mT`, real-valued `Tensor.H`/`Tensor.mH`, and `Tensor.adjoint`/`torch.adjoint`, `Tensor.swapdims`/`Tensor.swapaxes`, and top-level `torch.swapdims`/`torch.swapaxes`, plus squeeze, view-or-copy flatten/reshape, indexing views, and native row-major/channel-last contiguous materialization |
| dtypes, promotion, devices, dispatch | 10% | Pageable CPU `f32` only, with no-argument `Tensor.is_pinned` metadata, `Tensor.cpu` identity/channel-last conversion, `Tensor.get_device`/`torch.get_device` ordinal metadata, dtype-backed `torch.is_signed` introspection, canonical `torch.float32.to_real()` identity, native float32-only `torch.finfo`, and float32-only `torch.can_cast` and `torch.promote_types` |
| creation, elementwise, reductions | 15% | Python-integer `torch.broadcast_shapes` shape inference, basic arithmetic, inference-only top-level `torch.exp`, reductions, and exact tensor equality |
| linear algebra and signal operations | 10% | rank-2 matmul only |
| autograd and higher-order differentiation | 15% | unsupported |
| neural-network functional API and modules | 15% | out-of-place `torch.nn.functional.relu` backed by the native ReLU kernel, plus exact-identity `torch.nn.functional.dropout`, rank-3 `torch.nn.functional.dropout1d`, rank-4 `torch.nn.functional.dropout2d`, rank-5 `torch.nn.functional.dropout3d`, `torch.nn.functional.alpha_dropout`, and `torch.nn.functional.feature_alpha_dropout` for evaluation mode, zero probability, or empty inputs |
| optimizers, initialization, data utilities | 5% | unsupported |
| serialization, state dictionaries, model interchange | 5% | unsupported |
| compilation, parallelism, distributed execution | 5% | Eager-state `torch.compiler.is_compiling()` and backend-capability `torch.distributed.is_available()` queries; compilation, export, parallel execution, process groups, and collectives remain unsupported |
| ergonomics, diagnostics, documentation, ecosystem integration | 5% | minimal |

Within an area, coverage includes ordinary use plus error behavior, empty tensors, numerical edge cases, non-contiguous layouts, and interactions with autograd and device/dtype dispatch. Newly supported cells are added to permanent regression and performance matrices; existing cells are not retired to improve a score.
