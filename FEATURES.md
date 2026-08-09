# Feature coverage contract

Feature completeness is measured against stable public PyTorch Python concepts, with supported claims backed by executable side-by-side semantic tests. Documentation, Rust-only capabilities, and stubs do not count.

Fixed top-level weights prevent easy APIs from overwhelming core gaps:

| Area | Weight | Baseline |
| --- | ---: | --- |
| tensor storage, shapes, strides, views, indexing | 15% | CPU `f32` with strided transpose, squeeze, view-or-copy flatten/reshape, indexing views, and native row-major/channel-last contiguous materialization |
| dtypes, promotion, devices, dispatch | 10% | CPU `f32` only |
| creation, elementwise, reductions | 15% | six basic operations |
| linear algebra and signal operations | 10% | broadcast batched matmul for rank-two-and-higher operands; rank-1 matmul is unsupported |
| autograd and higher-order differentiation | 15% | unsupported |
| neural-network functional API and modules | 15% | unsupported |
| optimizers, initialization, data utilities | 5% | unsupported |
| serialization, state dictionaries, model interchange | 5% | unsupported |
| compilation, parallelism, distributed execution | 5% | unsupported |
| ergonomics, diagnostics, documentation, ecosystem integration | 5% | minimal |

Within an area, coverage includes ordinary use plus error behavior, empty tensors, numerical edge cases, non-contiguous layouts, and interactions with autograd and device/dtype dispatch. Newly supported cells are added to permanent regression and performance matrices; existing cells are not retired to improve a score.
