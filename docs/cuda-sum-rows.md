# Native CUDA matrix row sums

`torch_rs.sum(x, dim=1)` and `x.sum(dim=-1)` now reduce contiguous native CUDA
float32 matrices, with either `keepdim` value and no gradients. The shared sum
binding also accepts the existing one-item dimension sequences, such as `(1,)`.
Rust uses `Tensor::sum_rank_two_dimension(1, keepdim)` (a normalized dimension).

Embedded PTX uses four float32 accumulators per thread, aligned vector loads
with scalar heads and tails, descending shuffle-down reductions, and shared
memory when reduction geometry spans warps. Rust chooses geometry from row
count and width; wide rows can use CTA partials according to the current
device's multiprocessor and thread capacity. A second legacy-stream kernel
reduces those partials in a fixed order. Indexing is 64-bit and bounded grids
stride across rows and partials.

Large inputs also follow TensorIterator's signed 32-bit byte-offset partitioning.
A contiguous nonempty float32 region fits when `1 + 4 * (numel - 1)` is at most
`INT32_MAX`. Larger regions are recursively halved, processing the lower half
first: rows split before columns, and columns split only within one row. Each
piece gets its own device-dependent reduction geometry. Later column pieces add
their float32 partial to the preceding output on device, in launch order. This
preserves PyTorch's reduction ordering across the indexing boundary without
restricting matrix sizes. Scratch is reused only through ordered stream launches
and remains live through final synchronization, including failure paths.

This follows the contiguous float32 reduction ordering in PyTorch 2.13's
`ATen/native/cuda/Reduce.cuh`, including float32 intermediate overflow. It does
not widen to float64 or flush subnormals. Finite comparisons use the existing
tolerances; NaN and infinity classifications are checked explicitly. The
installed driver JITs PTX; production requires neither nvcc/NVRTC nor PyTorch,
and performs no CPU value computation. Compiled reductions remain unsupported.

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

The [combined integration report](composite-row-sum-glu-unflatten-validation.md)
records the [clean eb2af97 capture](diagnostics/composite-row-sum-glu-unflatten/postcommit-eb2af97/README.md),
including the indexing-boundary repair, earlier full-suite checks, and the
remaining independent review and CI gates.

[Python differentials](../tests/test_cuda_sum_rows.py) cover generated rectangular
shapes, same-sign decimal rows through width 1,000,003, irregular widths,
differing row counts, empty outputs, zero-width rows, offsets, singleton layouts, cancellation,
nonfinite values, signed zeros/subnormals, metadata, source preservation, fresh
storage, thread/lifetime reuse, stream completion, and unsupported boundaries.
A subprocess blocks all PyTorch imports. A separate two-GPU test checks device
guard restoration on success, rejection, and destruction.

[Indexing-boundary differentials](../tests/test_cuda_sum_rows_indexing.py) use
sparse device fixtures around 536,870,912 columns, multiple row partitions,
all four offset alignments, nested odd column splits, and nonfinite values.
They require enough GPU memory for the actual boundary; no reduced limit is
substituted in production or tests. A two-GPU case exercises split launches.

[Rust integration tests](../tests/cuda_sum_rows.rs) exercise the public native API
without Python. Internal CUDA storage tests check overflowing products and bounds,
empty offsets, and injected launch failure cleanup. Hardware-only tests clearly
skip when their required CUDA devices are unavailable.

## Historical source H100 evidence

The artifacts below belong to the original row-sum source, before the composite
float32-tree repair. They are retained under their original identities and paths,
not as proof of the combined candidate. Their float64 kernel failed the later
cancellation and overflow regressions; passing the six-case corpus did not make
that source merge-qualified. The earlier composite has a separate
[clean capture at 02535d5](diagnostics/composite-row-sum-glu-unflatten/postcommit-02535d5/README.md);
that capture predates the indexing repair, and these historical measurements
remain pinned to their original source.

The source capture measures implementation commit
`f1040cdf723cf9b172f27d50d3580bbb21f3c430`, including the wide-row accuracy fix.
Every build, evaluation, and test command began and ended with an empty
`git status --porcelain`; artifacts were copied into `docs` only after the
captures and provenance checks completed. These results replace the stale
candidate records for the earlier float32-accumulator kernel and fulfill the
previously deferred clean-commit capture.

The existing `scripts/capture_depth_concat_build.py` built a release wheel
without `--allow-dirty`, using a fresh Cargo target and locked offline
dependencies. The [build receipt](diagnostics/cuda-sum-rows/build-record.json)
records `clean_checkout=true`, unchanged source fingerprints, exact commands,
timestamps, cache state, compiler versions, dependencies, and binary hashes.
The wheel's extension, installed extension, and checkout extension have
identical bytes. All wheel Python sources were checked against the checkout.
The raw install log also records removal of the prior development wheel; the
installed and measured extension hashes match the new committed build.

Extension SHA-256:
`36dfe5a572392d0602423ffe5cdbb141b88561ce74f8b0be95aa34ffe211da1a`.

The [unchanged math evaluator](diagnostics/cuda-sum-rows/cuda-math.json)
passes `cuda_f32_sum_axis` at the original evaluator-selected seeds
`7763153567161607008` and `2618969910755569448`, using the unchanged
`rtol=1e-5` and `atol=1e-6`. The full six-case denominator remains intact:
five cases pass and unsupported CUDA matrix multiplication receives zero.
The report records `source_unchanged_during_run=true`. Candidate workers
blocked PyTorch imports, inspected native device pointers, copied results
through the driver, and checked input preservation. This is correctness
evidence, not a performance score.

## Historical wide-row regression

The reviewer identified serial float32 accumulation drift in the previous
kernel. The committed regressions cover widths 65,539, 1,000,000, and
1,000,003; positive and negative decimal values; contiguous offsets; both
`keepdim` forms; metadata; source preservation; and fresh storage. All pass
with the existing `rtol=1e-5`, `atol=1e-4`. The Rust oracle multiplies the exact
float32 input value by the width in float64, independently of tensor reduction.

The [source reproduction](diagnostics/cuda-sum-rows/wide-row-check.log) records
these per-row results for a `(2, width)` matrix filled with float32 `0.1`:

| Width | Native CUDA sum | PyTorch CUDA sum |
| --- | --- | --- |
| 65,539 | 6553.89990234375 | 6553.8984375 |
| 1,000,000 | 100000.0 | 100000.0 |

Both cases satisfy the unchanged regression tolerances.

## Environment and checks

Hardware was NVIDIA H100 (97,871 MiB, compute capability 9.0), driver
580.82.07. Reference PyTorch was `2.13.0+cu130` on Python 3.12.14+meta.
Both source evaluator workers loaded their original worktree's
`.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13`, runtime
version 13000 (CUDA 13.0). Rust tests explicitly selected that library with
`TORCH_RS_CUDART`. Rust/Cargo were 1.92.0; the build used release optimization,
thin LTO, one codegen unit, and `extension-module`. The available nvcc was
12.6.85 but was not invoked: the driver JIT-compiled embedded PTX.

Candidate/reference packages, extensions, build outputs, and caches were
worktree-local. Shared system Python, Rust, and CUDA tools were used read-only.
Ordinary GPU checks used `CUDA_VISIBLE_DEVICES=0`; only the device-restoration
test used `0,1`. [Command receipts](diagnostics/cuda-sum-rows/commands.json)
retain exact commands, environments, timestamps, clean status checks, and log
hashes. The [verification record](diagnostics/cuda-sum-rows/verification.json)
binds the preserved artifacts to that original source and build.

Passed focused checks:

- Three [Rust integration tests](diagnostics/cuda-sum-rows/rust-tests.log),
  including the wide decimal regression, plus one
  [bounds/failure-cleanup test](diagnostics/cuda-sum-rows/rust-bounds.log).
- Seven [single-GPU Python differentials](diagnostics/cuda-sum-rows/python-tests.log)
  and one [two-device guard test](diagnostics/cuda-sum-rows/two-device.log).
- All eight Python hardware tests [skip clearly with no visible GPU](diagnostics/cuda-sum-rows/no-gpu.log).
- Clippy with warnings denied, formatting, and diff checks.

These historical records remain unchanged. They do not supply current-composite
correctness or performance credit, independent review, or merge qualification.
