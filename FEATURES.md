# Feature coverage contract

Feature completeness is measured against stable public PyTorch Python concepts, with supported claims backed by executable side-by-side semantic tests. Documentation, Rust-only capabilities, and stubs do not count.

Fixed top-level weights prevent easy APIs from overwhelming core gaps:

| Area | Weight | Baseline |
| --- | ---: | --- |
| tensor storage, shapes, strides, views, indexing | 15% | CPU `f32` with arbitrary `Tensor.permute`/`torch.permute` views, strided transpose including read-only `Tensor.T`, `Tensor.mT`, real-valued `Tensor.H`/`Tensor.mH`, and `Tensor.adjoint`/`torch.adjoint`, `Tensor.swapdims`/`Tensor.swapaxes`, and top-level `torch.swapdims`/`torch.swapaxes`, plus squeeze, view-or-copy flatten/reshape, indexing views, and native row-major/channel-last contiguous materialization |
| dtypes, promotion, devices, dispatch | 10% | CPU `f32` only, with `Tensor.cpu` identity/channel-last conversion, `Tensor.get_device`/`torch.get_device` ordinal metadata, dtype-backed `torch.is_signed` introspection, and canonical `torch.float32.to_real()` identity |
| creation, elementwise, reductions | 15% | basic arithmetic, reductions, and exact tensor equality |
| linear algebra and signal operations | 10% | rank-2 matmul only |
| autograd and higher-order differentiation | 15% | unsupported |
| neural-network functional API and modules | 15% | out-of-place `torch.nn.functional.relu` backed by the native ReLU kernel |
| optimizers, initialization, data utilities | 5% | unsupported |
| serialization, state dictionaries, model interchange | 5% | unsupported |
| compilation, parallelism, distributed execution | 5% | unsupported |
| ergonomics, diagnostics, documentation, ecosystem integration | 5% | minimal |

Within an area, coverage includes ordinary use plus error behavior, empty tensors, numerical edge cases, non-contiguous layouts, and interactions with autograd and device/dtype dispatch. Newly supported cells are added to permanent regression and performance matrices; existing cells are not retired to improve a score.
