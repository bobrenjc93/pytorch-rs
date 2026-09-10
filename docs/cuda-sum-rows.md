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

## Clean-commit H100 results and reproduction

The post-commit capture measures implementation commit
`6d240ceef6ee719a041852c904506622995ef4c7` from this worktree with an empty
`git status --porcelain` before and after every build, evaluation, and test
command. Reports were copied into `docs` only after all captures completed.
These records replace the candidate's pre-commit measurements.

The [unchanged math evaluator result](diagnostics/cuda-sum-rows/cuda-math.json)
credits `cuda_f32_sum_axis` at both evaluator-selected seeds
`7763153567161607008` and `2618969910755569448`, retaining the original seeds,
six-case matrix, `rtol=1e-5`, and `atol=1e-6`. Five cases pass; unsupported CUDA
matrix multiplication still receives zero. Candidate workers block PyTorch
imports, independently inspect device pointers, materialize outputs with the
CUDA driver, and verify unchanged inputs. This is correctness evidence only.

The repository's existing `scripts/capture_depth_concat_build.py` built a fresh
release wheel offline from the clean checkout into an initially absent Cargo
target, then installed it into this worktree's real `.venv`. The
[build receipt](diagnostics/cuda-sum-rows/build-record.json) records
`clean_checkout=true`, unchanged source hashes, the wheel and extension hashes,
compiler versions, dependency versions, commands, timestamps, and cache state.
The installed wheel and checkout's source package contain identical extension
bytes; the unchanged evaluator imports the source package. Its
`source_unchanged_during_run` is true and the production diff is empty.

The fresh extension SHA-256 is
`ce56a8cb238e20c3bbb60639a4c7f7a043e6c8002a51b62ea6ec7b73af2da7b9`.
Rust/Cargo were 1.92.0, with release optimization, thin LTO, one codegen unit,
and the `extension-module` feature. The reference interpreter was Python
3.12.14+meta, with stable PyTorch `2.13.0+cu130` installed from the unchanged
lockfile. Candidate/reference packages, extension paths, build targets, and
caches are inside this worktree; the Python interpreter and Rust/CUDA tools are shared
system installations used read-only.

Hardware was NVIDIA H100 (97,871 MiB, compute capability 9.0), driver 580.82.07.
Both evaluator workers loaded the worktree-local
`.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13`; their actual
runtime version was 13000 (CUDA 13.0). Rust tests explicitly selected that library with `TORCH_RS_CUDART`. The available nvcc was 12.6.85,
but no CUDA compiler was invoked: the driver JIT-compiled embedded PTX.
Ordinary GPU checks used `CUDA_VISIBLE_DEVICES=0`; only the device-restoration
check used `0,1`.

[Command receipts](diagnostics/cuda-sum-rows/commands.json) preserve the exact
commands, working directories, environments, timestamps, exit statuses, clean
status checks, and log hashes. The [verification record](diagnostics/cuda-sum-rows/verification.json)
also binds the preserved artifacts and checks all 59 wheel Python sources
against the checkout. The post-commit refresh reran the focused row-sum checks:

- `cargo test --locked --offline --test cuda_sum_rows -- --nocapture`:
  two public Rust tests passed; [log](diagnostics/cuda-sum-rows/rust-tests.log).
- `cargo test --locked --offline --lib
  cuda::tests::row_sum_bounds_empty_inputs_and_failure_cleanup -- --exact --nocapture`:
  one storage-bounds/failure-cleanup test passed;
  [log](diagnostics/cuda-sum-rows/rust-bounds.log).
- `.venv/bin/python -B -m unittest tests.test_cuda_sum_rows.CudaSumRowsTests -v`:
  six single-GPU differential tests passed;
  [log](diagnostics/cuda-sum-rows/python-tests.log).
- With `CUDA_VISIBLE_DEVICES=0,1`,
  `.venv/bin/python -B -m unittest tests.test_cuda_sum_rows.CudaSumRowsDeviceTests -v`:
  one device-restoration test passed;
  [log](diagnostics/cuda-sum-rows/two-device.log).
- With no visible devices, all seven new Python hardware tests skip clearly;
  [log](diagnostics/cuda-sum-rows/no-gpu.log).
- `cargo clippy --locked --offline --all-targets --features python-bindings -- -D warnings`,
  `cargo fmt --all -- --check`, and `git diff --check` passed.

Evaluator definitions, scoring corpora, implementation, tests, dependencies,
historical benchmark evidence, and Burner-managed progress artifacts were not
changed by this evidence refresh. This capture does not replace independent
review or merge gates.
