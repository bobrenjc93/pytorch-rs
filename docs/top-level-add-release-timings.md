# `torch.add` Release Timings

Date: 2026-09-02

Candidate provenance: source snapshot based on
`c54198202b09ffcb58457cd1de3b826cf10895e4`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/top-level-add-release-timings*.json`. No Conda
environment was active in the shell (`CONDA_PREFIX=`), so setup used a
worktree-local `.venv`. Cargo registry data was copied read-only from the
existing user cache into `target/cargo-home`, then Cargo ran offline so build
artifacts and dependency state stayed inside this worktree.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
mkdir -p target/cargo-home/registry && \
  cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/ && \
  wheel_dir="$(mktemp -d "$PWD/target/top-level-add-wheels.XXXXXX")" && \
  printf '%s\n' "$wheel_dir" > target/top-level-add-wheel-dir.txt && \
  env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
    CARGO_HOME="$PWD/target/cargo-home" \
    CARGO_TARGET_DIR="$PWD/target" \
    TMPDIR="$PWD/target" \
    VIRTUAL_ENV="$PWD/.venv" \
    PYO3_PYTHON="$PWD/.venv/bin/python" \
    .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
wheel_dir="$(cat target/top-level-add-wheel-dir.txt)" && \
  env UV_CACHE_DIR="$PWD/target/uv-cache" \
    UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
    uv pip install --python "$PWD/.venv/bin/python" \
    --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_tensor_add tests.test_tensor_add_reference \
  tests.test_top_level_add tests.test_top_level_add_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/top_level_add_release_timings.py \
  > target/top-level-add-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  TOP_LEVEL_ADD_TIMING_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python target/top_level_add_release_timings.py \
  > target/top-level-add-release-timings-pass2.json
```

Checks run for this evidence:

```bash
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets add
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets binary_arithmetic
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets scalar_arithmetic
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_tensor_add tests.test_tensor_add_reference \
  tests.test_top_level_add tests.test_top_level_add_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 23 tests. The focused Rust `add` filter passed 1 test, the Rust binary
arithmetic filter passed 4 tests, the Rust scalar arithmetic filter passed 1
test, `cargo fmt --check` passed, the README/docs smoke test passed, and
`git diff --check` passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0, from
  `.venv/lib/python3.12/site-packages/torch`
- `torch_rs`: 0.1.0 from the wheel-installed
  `.venv/lib/python3.12/site-packages/torch_rs`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Release install mode: `maturin build --release --locked --offline`, followed
  by `uv pip install --force-reinstall --no-deps` of the generated wheel
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing runs
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 27 ms, prepared
  packages in 16.37s, and installed in 3.22s
- Build time: successful offline release extension build completed in 37.25s;
  the release wheel reinstall resolved in 3 ms, prepared in 57 ms, and
  installed in 15 ms

Inputs were created outside the timed region with NumPy seed `20260902`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
The first pass measured `torch_rs` before PyTorch; the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks.
A block repeated the operation according to the table's `Repeats` column;
times below are median microseconds per operation. Reported medians are
medians of the two per-process medians. MAD and variance are the medians of the
per-process MAD and sample variance values.

Before timing each supported cell, the driver bit-compared `torch_rs` output
values with PyTorch and checked shape, stride, storage offset, contiguity,
dtype, device, `requires_grad`, and leaf status. For forward+backward cells,
the driver checked the materialized leaf gradients after
`torch.add(...).sum().backward()`. Backward timings used fresh leaf tensors
created before each timed block so the measured region did not include input
construction and did not reuse a freed graph. After every warmup and measured
block, the driver materialized the last output or gradient tuple as a 64-bit
BLAKE2b rolling checksum over output metadata and logical bytes. The checksum
column shows the final rolling sink from one pass as `torch_rs`/PyTorch; both
process passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 0.63x uncapped, 0.63x capped
- Tensor/tensor cells: 0.76x uncapped, 0.76x capped
- Tensor/scalar cells: 0.56x uncapped, 0.56x capped
- Scalar/tensor cells: 0.55x uncapped, 0.55x capped
- Eager value cells: 0.76x uncapped, 0.76x capped
- `no_grad` cells: 0.70x uncapped, 0.70x capped
- Autograd forward cells: 0.64x uncapped, 0.64x capped
- Autograd forward+backward cells: 0.20x uncapped, 0.20x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 1.15x.

| Workload | Category | API | Operand path | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar_tensor_tensor` | scalar tensor | `torch.add` | tensor/tensor | left/right scalar tensors, shape (), stride (), offset 0 | (), stride (), offset 0, requires_grad=False | 10000 | 0.298 us +/- 0.003 us, var 0.000 | 1.259 us +/- 0.015 us, var 0.005 | 0.24x | `5882066803946197440`/`5882066803946197440` |
| `scalar_tensor_scalar` | scalar tensor | `torch.add` | tensor/scalar | left scalar tensor, shape (), stride (); scalar 2.25 | (), stride (), offset 0, requires_grad=False | 10000 | 1.124 us +/- 0.036 us, var 0.002 | 2.666 us +/- 0.088 us, var 0.011 | 0.42x | `5882066803946197440`/`5882066803946197440` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.add` | tensor/tensor | left/right (257, 263), stride (263, 1) | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 9.243 us +/- 0.255 us, var 0.170 | 10.705 us +/- 0.273 us, var 0.284 | 0.86x | `7931044512434601408`/`7931044512434601408` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.add` | tensor/scalar | left (640, 768), stride (768, 1); scalar -2.25 | (640, 768), stride (768, 1), offset 0, requires_grad=False | 10 | 55.782 us +/- 1.553 us, var 69.043 | 49.381 us +/- 0.954 us, var 11.231 | 1.13x | `2872984427249845216`/`2872984427249845216` |
| `scalar_tensor_640x768` | scalar/tensor contiguous | `torch.add` | scalar/tensor | scalar -2.25; right (640, 768), stride (768, 1) | (640, 768), stride (768, 1), offset 0, requires_grad=False | 10 | 60.215 us +/- 3.514 us, var 196.413 | 53.710 us +/- 2.857 us, var 122.944 | 1.12x | `17218274053451415488`/`17218274053451415488` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.add` | tensor/tensor | left (640, 768), stride (768, 1); right (768,), stride (1,) | (640, 768), stride (768, 1), offset 0, requires_grad=False | 16 | 55.034 us +/- 1.166 us, var 70.466 | 48.755 us +/- 1.236 us, var 11.428 | 1.13x | `15224267358505639648`/`15224267358505639648` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.add` | tensor/tensor | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 0.401 us +/- 0.004 us, var 0.002 | 1.271 us +/- 0.008 us, var 0.012 | 0.32x | `14630084997776431072`/`14630084997776431072` |
| `empty_tensor_scalar_3x0x2` | empty tensor/scalar | `torch.add` | tensor/scalar | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); scalar 4.0 | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 1.184 us +/- 0.006 us, var 0.007 | 2.698 us +/- 0.026 us, var 0.140 | 0.44x | `14630084997776431072`/`14630084997776431072` |
| `empty_scalar_tensor_3x0x2` | empty scalar/tensor | `torch.add` | scalar/tensor | scalar 4.0; right zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2) | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 1.253 us +/- 0.008 us, var 0.006 | 2.694 us +/- 0.026 us, var 0.135 | 0.47x | `14630084997776431072`/`14630084997776431072` |
| `offset_transposed_521x509` | offset tensor/tensor | `torch.add` | tensor/tensor | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 134.113 us +/- 3.994 us, var 298.017 | 38.489 us +/- 1.339 us, var 8.585 | 3.48x | `1306895864262985696`/`1306895864262985696` |
| `offset_tensor_scalar_521x509` | offset tensor/scalar | `torch.add` | tensor/scalar | left tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189; scalar 1.75 | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 34.041 us +/- 2.136 us, var 15.103 | 29.502 us +/- 1.004 us, var 40.163 | 1.15x | `1018198425642332384`/`1018198425642332384` |
| `offset_scalar_tensor_521x509` | offset scalar/tensor | `torch.add` | scalar/tensor | scalar 1.75; right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 39.190 us +/- 3.176 us, var 22.889 | 30.758 us +/- 1.188 us, var 37.725 | 1.27x | `2911571552740817920`/`2911571552740817920` |
| `noncontig_transpose_512x1024` | noncontiguous tensor/tensor | `torch.add` | tensor/tensor | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 277.768 us +/- 6.243 us, var 152.938 | 78.923 us +/- 2.583 us, var 36.740 | 3.52x | `9466784407040071424`/`9466784407040071424` |
| `signed_zero_nan_inf` | signed-zero NaN/inf | `torch.add` | tensor/tensor | special float32 bit patterns [0, -0, inf, -inf, qnan] plus zeros | (5,), stride (1,), offset 0, requires_grad=False | 10000 | 0.325 us +/- 0.002 us, var 0.000 | 1.261 us +/- 0.008 us, var 0.020 | 0.26x | `8534731495320647552`/`8534731495320647552` |
| `signed_zero_nan_inf_tensor_scalar` | signed-zero NaN/inf tensor/scalar | `torch.add` | tensor/scalar | special float32 bit patterns [0, -0, inf, -inf, qnan] plus scalar -0.0 | (5,), stride (1,), offset 0, requires_grad=False | 10000 | 2.589 us +/- 0.018 us, var 0.005 | 3.305 us +/- 0.017 us, var 0.155 | 0.78x | `12374855441278413024`/`12374855441278413024` |
| `signed_zero_nan_inf_scalar_tensor` | signed-zero NaN/inf scalar/tensor | `torch.add` | scalar/tensor | scalar inf plus special float32 bit patterns [0, -0, inf, -inf, qnan] | (5,), stride (1,), offset 0, requires_grad=False | 10000 | 1.201 us +/- 0.007 us, var 0.003 | 2.683 us +/- 0.017 us, var 0.134 | 0.45x | `4310213078302272224`/`4310213078302272224` |
| `no_grad_tensor_tensor_257x263` | no_grad tensor/tensor | `torch.add` | tensor/tensor | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 9.649 us +/- 0.241 us, var 0.208 | 11.059 us +/- 0.302 us, var 0.696 | 0.87x | `7931044512434601408`/`7931044512434601408` |
| `no_grad_tensor_scalar_127x131` | no_grad tensor/scalar | `torch.add` | tensor/scalar | left leaf (127, 131), requires_grad=True; scalar 4.0; operation inside no_grad | (127, 131), stride (131, 1), offset 0, requires_grad=False | 20 | 2.668 us +/- 0.027 us, var 0.071 | 4.229 us +/- 0.028 us, var 0.040 | 0.63x | `7186506708902554496`/`7186506708902554496` |
| `no_grad_scalar_tensor_127x131` | no_grad scalar/tensor | `torch.add` | scalar/tensor | scalar 4.0; right leaf (127, 131), requires_grad=True; operation inside no_grad | (127, 131), stride (131, 1), offset 0, requires_grad=False | 20 | 2.653 us +/- 0.032 us, var 0.031 | 4.227 us +/- 0.030 us, var 0.133 | 0.63x | `14743380753610412928`/`14743380753610412928` |
| `autograd_forward_tensor_tensor_127x131` | autograd forward tensor/tensor | `torch.add` | tensor/tensor | left/right leaves (127, 131), requires_grad=True; forward construction only | (127, 131), stride (131, 1), offset 0, requires_grad=True | 20 | 2.555 us +/- 0.031 us, var 0.052 | 3.403 us +/- 0.020 us, var 0.057 | 0.75x | `136929574794581440`/`136929574794581440` |
| `autograd_forward_tensor_scalar_127x131` | autograd forward tensor/scalar | `torch.add` | tensor/scalar | left leaf (127, 131), requires_grad=True; scalar 4.0; forward construction only | (127, 131), stride (131, 1), offset 0, requires_grad=True | 20 | 2.759 us +/- 0.028 us, var 0.030 | 4.638 us +/- 0.039 us, var 0.071 | 0.59x | `9069246164551090912`/`9069246164551090912` |
| `autograd_forward_scalar_tensor_127x131` | autograd forward scalar/tensor | `torch.add` | scalar/tensor | scalar 4.0; right leaf (127, 131), requires_grad=True; forward construction only | (127, 131), stride (131, 1), offset 0, requires_grad=True | 20 | 2.733 us +/- 0.031 us, var 0.062 | 4.603 us +/- 0.040 us, var 0.072 | 0.59x | `8862510904597864448`/`8862510904597864448` |
| `autograd_forward_backward_tensor_tensor_32x33` | autograd forward+backward tensor/tensor | `torch.add` | tensor/tensor | left/right leaves (32, 33), requires_grad=True; timed `torch.add(...).sum().backward()` | (32, 33), stride (33, 1), offset 0, requires_grad=False | 5 | 16.126 us +/- 0.194 us, var 4.251 | 35.451 us +/- 2.165 us, var 29.451 | 0.45x | `18337651485950501760`/`18337651485950501760` |
| `autograd_forward_backward_tensor_scalar_32x33` | autograd forward+backward tensor/scalar | `torch.add` | tensor/scalar | left leaf (32, 33), requires_grad=True; scalar 4.0; timed `torch.add(...).sum().backward()` | (32, 33), stride (33, 1), offset 0, requires_grad=False | 5 | 4.507 us +/- 0.073 us, var 0.572 | 34.601 us +/- 2.569 us, var 37.729 | 0.13x | `7535769365259487680`/`7535769365259487680` |
| `autograd_forward_backward_scalar_tensor_32x33` | autograd forward+backward scalar/tensor | `torch.add` | scalar/tensor | scalar 4.0; right leaf (32, 33), requires_grad=True; timed `torch.add(...).sum().backward()` | (32, 33), stride (33, 1), offset 0, requires_grad=False | 5 | 4.559 us +/- 0.067 us, var 0.323 | 32.501 us +/- 1.121 us, var 8.524 | 0.14x | `7535769365259487680`/`7535769365259487680` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `top_level_torch_add_nondefault_alpha_tensor_tensor` | `NotImplementedError: add(): alpha values other than 1 are not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_add_nondefault_alpha_tensor_scalar` | `NotImplementedError: add(): alpha values other than 1 are not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_add_nondefault_alpha_scalar_tensor` | `NotImplementedError: add(): alpha values other than 1 are not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_add_out_tensor_tensor` | `RuntimeError: add(): the 'out' argument is not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_add_out_tensor_scalar` | `RuntimeError: add(): the 'out' argument is not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_add_out_scalar_tensor` | `RuntimeError: add(): the 'out' argument is not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_add_scalar_scalar` | `NotImplementedError: add(): only exact native CPU float32 Tensor/Tensor, Tensor/real-number, or real-number/Tensor operands are supported` | supported (), stride (), offset 0, requires_grad=False | zero |
