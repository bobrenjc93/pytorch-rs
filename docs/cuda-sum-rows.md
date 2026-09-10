# Native CUDA matrix row sums

`torch_rs.sum(x, dim=1)` and `x.sum(dim=-1)` now reduce contiguous native CUDA
float32 matrices, with either `keepdim` value and no gradients. The shared sum
binding also accepts the existing one-item dimension sequences, such as `(1,)`.
Rust uses `Tensor::sum_rank_two_dimension(1, keepdim)` (a normalized dimension).

One embedded PTX kernel handles arbitrary rectangular dimensions with 64-bit
indexing. Each warp widens its float32 inputs to float64, accumulates and
reduces them with a float64 shuffle tree, then rounds once to float32 for the
output. This prevents serial float32 rounding error from growing with row width.
A bounded grid strides over rows. The installed NVIDIA driver JITs PTX;
there is no nvcc/NVRTC dependency, CPU value computation, PyTorch forwarding,
shape-specific benchmark path, or compiler integration. Floating-point addition
order can differ from PyTorch, so finite sums are compared with tolerances;
Input NaNs and infinities retain their arithmetic classifications, and subnormals
are not flushed. Accumulation precision and intermediate overflow can differ
from PyTorch's float32 reduction; this does not add a float64 tensor API.

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
shapes, same-sign decimal rows through width 1,000,003, irregular widths,
a row count beyond the grid stride,
empty outputs, zero-width rows, offsets, singleton layouts, cancellation,
nonfinite values, signed zeros/subnormals, metadata, source preservation, fresh
storage, thread/lifetime reuse, stream completion, and unsupported boundaries.
A subprocess blocks all PyTorch imports. A separate two-GPU test checks device
guard restoration on success, rejection, and destruction.

[Rust integration tests](../tests/cuda_sum_rows.rs) exercise the public native API
without Python. Internal CUDA storage tests check overflowing products and bounds,
empty offsets, and injected launch failure cleanup. Hardware-only tests clearly
skip when their required CUDA devices are unavailable.

## Review regression and validation

The previous kernel added an entire strided row into one float32 accumulator
per lane. For two rows of 1,000,000 copies of float32 `0.1`, it returned
`100022.3515625` per row on H100, versus PyTorch's `100000.0`. The revised kernel
returns `100000.0`. At width 65,539, it returns `6553.89990234375` versus
PyTorch's `6553.8984375`, within the existing `rtol=1e-5`, `atol=1e-4`.

New Rust and H100 regressions cover widths 65,539, 1,000,000, and 1,000,003;
positive and negative decimal inputs; both `keepdim` forms; contiguous offsets;
source preservation; and fresh storage. The Python regression fails against
the previous extension and passes against the revised one without changing
any tolerance. The Rust oracle uses the input's exact float32 value multiplied
by the row width in float64, independently of either tensor reduction.

Development checks use a fresh source-matched release wheel built with
`scripts/capture_depth_concat_build.py --allow-dirty`, Rust/Cargo 1.92.0,
Python 3.12.14+meta, and PyTorch `2.13.0+cu130`. H100 driver 580.82.07 loads
CUDA runtime 13.0 from the worktree-local `.venv`. The available nvcc is 12.6.85;
embedded PTX is compiled by the driver, without invoking nvcc. Raw build,
test, and evaluator diagnostics are under `target/sum-row-review/` and are
explicitly uncommitted-source diagnostics, not clean-commit evaluation evidence.
The seven single-GPU Python tests, one two-device test, three Rust integration
tests, and eleven internal CUDA tests passed. With no visible GPU, all eight
Python hardware tests skipped clearly. Clippy with warnings denied, formatting,
and diff checks passed. The unchanged math evaluator passed row sums at both
existing evaluator-selected seeds in this diagnostic run (five of six cases;
CUDA matrix multiplication remains unsupported).

## Required clean-commit evidence refresh

**The checked-in artifacts under `docs/diagnostics/cuda-sum-rows/` are stale for
this revision.** They measure commit `6d240ceef6ee719a041852c904506622995ef4c7`,
which contains the inaccurate float32 accumulator. Their recorded provenance
and measurements are preserved unchanged; they must not establish credit for
the revised kernel. This is an outstanding capture requirement, not a historical
reclassification or a waiver of the required clean measurement.

After Burner commits the fix, rerun the existing
`scripts/capture_depth_concat_build.py` without `--allow-dirty`, followed by
`scripts/evaluate_cuda_math.py` with seeds `7763153567161607008` and
`2618969910755569448` and that fresh build receipt. Keep all six cases, the
original tolerances, and unsupported matrix multiplication at zero. Refresh the
focused Rust/H100 test logs and replace the stale artifacts only after their
clean source/build provenance is verified. Commit creation is owned by Burner,
so a clean capture of this uncommitted fix cannot be produced in this turn.
