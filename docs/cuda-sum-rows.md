# Native CUDA matrix row sums

`torch_rs.sum(x, dim=1)` and `x.sum(dim=-1)` now reduce contiguous native CUDA
float32 matrices, with either `keepdim` value and no gradients. The shared sum
binding also accepts the existing one-item dimension sequences, such as `(1,)`.
Rust uses `Tensor::sum_rank_two_dimension(1, keepdim)` (a normalized dimension).

One embedded PTX kernel handles arbitrary rectangular dimensions with 64-bit
indexing. Each warp accumulates a row in float32 and reduces it with a shuffle
tree; a bounded grid strides over rows. The installed NVIDIA driver JITs PTX;
there is no nvcc/NVRTC dependency, CPU value computation, PyTorch forwarding,
shape-specific benchmark path, or compiler integration. Floating-point addition
order can differ from PyTorch, so finite sums are compared with tolerances;
NaNs and infinities follow float32 arithmetic, and subnormals are not flushed.

Contiguous offset views and contiguous singleton layouts are accepted. Outputs
have fresh contiguous storage on the input device, offset zero, and shape
`(rows,)` or `(rows, 1)`. Zero rows produce empty outputs without a kernel launch.
Zero-width rows produce zeros on device without forming or reading an input
pointer, even for empty views with offsets beyond storage. Input values remain
unchanged. Zero-element allocations use the existing null-pointer convention.

The storage path shares checked bounds, checked allocation sizes, device guards,
and legacy-stream completion with native unary pointwise operations. The guard
spans allocation, launch, and completion and restores the caller's current device.
Launch/completion errors do not publish output and disable allocation reuse.
Completed output can be consumed by another stream after return; arbitrary
external concurrent writes or public stream selection are outside this boundary.

Full sums, dim=0/-2, multiple reduction dimensions, other ranks, noncontiguous
inputs, other dtypes, autograd, and compiled reductions remain unsupported.
Other CUDA reductions (including mean) retain their existing rejection paths.
Existing CPU sum bindings, override dispatch, and CPU autograd are unchanged.

## Validation

[Python differentials](../tests/test_cuda_sum_rows.py) cover generated rectangular
shapes, irregular widths through 65,539, a row count beyond the grid stride,
empty outputs, zero-width rows, offsets, singleton layouts, cancellation,
nonfinite values, signed zeros/subnormals, metadata, source preservation, fresh
storage, thread/lifetime reuse, stream completion, and unsupported boundaries.
A subprocess blocks all PyTorch imports. A separate two-GPU test checks device
guard restoration on success, rejection, and destruction.

[Rust integration tests](../tests/cuda_sum_rows.rs) exercise the public native API
without Python. Internal CUDA storage tests check overflowing products and bounds,
empty offsets, and injected launch failure cleanup. Hardware-only tests clearly
skip when their required CUDA devices are unavailable.

## H100 results and reproduction

The [unchanged math evaluator result](diagnostics/cuda-sum-rows/cuda-math.json)
credits `cuda_f32_sum_axis` at both evaluator-selected seeds
`7763153567161607008` and `2618969910755569448`, with its existing `rtol=1e-5`,
`atol=1e-6`. All six cases remain in the denominator: five pass, and unsupported
CUDA matrix multiplication receives zero. Candidate workers blocked PyTorch
imports, independently inspected device pointers, materialized outputs with the
CUDA driver, and verified unchanged inputs. This is correctness evidence only.

The [build receipt](diagnostics/cuda-sum-rows/build-record.json) binds the dirty
worktree's complete production-source fingerprint to a fresh release extension;
`source_unchanged_during_run` is true. The extension SHA-256 is
`c8cb388f6f57a37e0b2641a7025e893b50d65e429deac1fe1cebe1f5ab7475e3`.
The receipt records exact build/install/evaluator commands, Rust/Cargo 1.92.0,
Python 3.12.12, release/thin-LTO configuration, evaluator and matrix hashes.
The release target was absent before building. All build outputs and caches
were worktree-local; the installed reference Python environment was read-only.

Hardware was NVIDIA H100 (97,871 MiB, compute capability 9.0), driver 580.82.07.
Reference PyTorch was `2.13.0+cu130`; both evaluator workers actually loaded
`nvidia/cu13/lib/libcudart.so.13`, runtime version 13000. The available nvcc was
12.6.85, but no CUDA compiler was invoked: the driver compiled embedded PTX.
Ordinary GPU checks used `CUDA_VISIBLE_DEVICES=0`; only the device-restoration
check used `0,1`.

Passed checks (commands use the reference interpreter recorded in the receipt,
`PYTHONPATH=$PWD/python`, and worktree-local temporary/cache paths):

- `cargo test --offline --lib --test cuda_sum_rows --test cuda_native_boundaries
  --test cuda_mul_scalar --test cuda_add --test cuda_same_device_copy`: 180 passed.
  The final run explicitly selected the same CUDA 13 runtime with
  `TORCH_RS_CUDART`; see [Rust log](diagnostics/cuda-sum-rows/rust-tests.log).
- `python -B -m unittest tests.test_cuda_sum_rows tests.test_cuda_add
  tests.test_cuda_add_trailing_vector tests.test_cuda_neg tests.test_cuda_mul_scalar
  tests.test_tensor_sum tests.test_top_level_sum tests.test_rank2_sum_numerics -v`:
  73 passed, 8 mask-specific two-device cases skipped;
  [Python log](diagnostics/cuda-sum-rows/python-tests.log).
- `CUDA_VISIBLE_DEVICES=0,1 python -B -m unittest
  tests.test_cuda_sum_rows.CudaSumRowsDeviceTests -v`: 1 passed;
  [two-device log](diagnostics/cuda-sum-rows/two-device.log).
- With no visible devices, all seven new Python hardware tests skip clearly;
  [skip log](diagnostics/cuda-sum-rows/no-gpu.log).
- `cargo clippy --offline --all-targets --features python-bindings -- -D warnings`,
  `cargo fmt --all -- --check`, and `git diff --check` passed.

Evaluator definitions, scoring corpora, existing benchmark evidence, and
Burner-managed progress artifacts were not changed.
