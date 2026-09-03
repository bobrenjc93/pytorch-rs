# Top-Level `torch.add` Release Timings

Date: 2026-09-03

Candidate provenance: source snapshot based on
`d0a916007edb2f9ee090f66c8ec2f4307df5cebe`. This branch adds timing evidence
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
  cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/
mkdir -p target/cargo-home && \
  wheel_dir="$(mktemp -d "$PWD/target/top-level-add-wheels.XXXXXX")" && \
  printf '%s\n' "$wheel_dir" > target/top-level-add-wheel-dir.txt
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --offline \
  --out "$(cat target/top-level-add-wheel-dir.txt)"
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps \
  "$(cat target/top-level-add-wheel-dir.txt)"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
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
  tests.test_top_level_add tests.test_top_level_add_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 12 tests. The focused Rust `add` filter passed 1 test, the Rust binary
arithmetic filter passed 4 tests, the Rust scalar arithmetic filter passed 1
test, the README/docs smoke test passed 7 tests, `cargo fmt --check` passed,
and `git diff --check` passed.

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
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing runs
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 27 ms, prepared
  packages in 14.91s, and installed in 791 ms
- Build time: successful offline release extension build completed in 36.47s;
  the release wheel reinstall resolved in 1 ms, prepared in 63 ms, and
  installed in 17 ms

Inputs were created outside the timed region with NumPy seed `20260903`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Timed supported calls used top-level `torch.add`
with default-equivalent `alpha=1` and omitted `out`, equivalent to `out=None`.
Every timing cell ran in two pinned process passes. The first pass measured
`torch_rs` before PyTorch; the second pass reversed that order. Each pass used
15 untimed warmup blocks and 81 measured blocks. A block repeated the operation
according to the table's `Repeats` column; times below are median microseconds
per operation. Reported medians are medians of the two per-process medians. MAD
and variance are the medians of the per-process MAD and sample variance values.

Before timing each supported cell, the driver bit-compared `torch_rs` output
values with PyTorch and checked shape, stride, storage offset, contiguity,
dtype, device, `requires_grad`, and leaf status. For autograd forward+backward
cells it also checked the backward leaf gradients. The backward cells use
small exactly representable float32 inputs so the scalar `sum()` loss is also
bit-exact across implementations. After every warmup and measured block, the
driver materialized the last output, or the backward scalar loss plus leaf
gradients, as a 64-bit BLAKE2b rolling checksum over output metadata and
logical bytes. The checksum column shows the final rolling sink from one pass
as `torch_rs`/PyTorch; both process passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 0.68x uncapped, 0.68x capped
- Tensor/tensor cells: 0.79x uncapped, 0.79x capped
- Tensor/scalar cells: 0.64x uncapped, 0.64x capped
- Scalar/tensor cells: 0.60x uncapped, 0.60x capped
- Scalar tensor cells: 0.33x uncapped, 0.33x capped
- Tensor/tensor contiguous cells: 0.90x uncapped, 0.90x capped
- Tensor/scalar contiguous cells: 4.41x uncapped, 4.41x capped
- Scalar/tensor contiguous cells: 3.21x uncapped, 3.21x capped
- Broadcasting cells: 1.38x uncapped, 1.38x capped
- Empty cells: 0.39x uncapped, 0.39x capped
- Offset tensor/tensor cells: 3.67x uncapped, 3.67x capped
- Offset tensor/scalar cells: 1.23x uncapped, 1.23x capped
- Offset scalar/tensor cells: 1.10x uncapped, 1.10x capped
- Noncontiguous cells: 3.78x uncapped, 3.78x capped
- Signed-zero/NaN/inf cells: 0.35x uncapped, 0.35x capped
- Autograd forward cells: 0.64x uncapped, 0.64x capped
- Autograd forward+backward cells: 0.23x uncapped, 0.23x capped
- `no_grad` tensor/tensor cells: 0.83x uncapped, 0.83x capped
- `no_grad` tensor/scalar cells: 0.74x uncapped, 0.74x capped
- `no_grad` scalar/tensor cells: 0.75x uncapped, 0.75x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 1.20x.

| Workload | Category | API | Operand path | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `rank0_tensor_tensor` | scalar tensor | `torch.add` | tensor/tensor | left/right scalar tensors, shape (), stride (), offset 0 | (), stride (), offset 0, requires_grad=False | 10000 | 0.261 us +/- 0.001 us, var 0.001 | 1.229 us +/- 0.019 us, var 0.002 | 0.21x | `3414690964265842695`/`3414690964265842695` |
| `rank0_tensor_scalar` | scalar tensor | `torch.add` | tensor/scalar | left scalar tensor, shape (), stride (); scalar -2.25 | (), stride (), offset 0, requires_grad=False | 10000 | 1.102 us +/- 0.007 us, var 0.001 | 2.666 us +/- 0.026 us, var 0.011 | 0.41x | `13505628535941244505`/`13505628535941244505` |
| `rank0_scalar_tensor` | scalar tensor | `torch.add` | scalar/tensor | scalar -2.25; right scalar tensor, shape (), stride () | (), stride (), offset 0, requires_grad=False | 10000 | 1.108 us +/- 0.010 us, var 0.001 | 2.668 us +/- 0.020 us, var 0.004 | 0.42x | `13505628535941244505`/`13505628535941244505` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.add` | tensor/tensor | left/right (257, 263), stride (263, 1) | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 9.426 us +/- 0.224 us, var 0.209 | 10.499 us +/- 0.100 us, var 0.097 | 0.90x | `17640684704199057311`/`17640684704199057311` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.add` | tensor/scalar | left (640, 768), stride (768, 1); scalar -2.25 | (640, 768), stride (768, 1), offset 0, requires_grad=False | 10 | 212.963 us +/- 18.513 us, var 3608.043 | 48.258 us +/- 0.970 us, var 395.103 | 4.41x | `12701156681644492316`/`12701156681644492316` |
| `scalar_tensor_640x768` | scalar/tensor contiguous | `torch.add` | scalar/tensor | scalar -2.25; right (640, 768), stride (768, 1) | (640, 768), stride (768, 1), offset 0, requires_grad=False | 10 | 156.675 us +/- 15.466 us, var 1249.502 | 48.826 us +/- 1.051 us, var 129.180 | 3.21x | `8088932759683535813`/`8088932759683535813` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.add` | tensor/tensor | left (640, 768), stride (768, 1); right (768,), stride (1,) | (640, 768), stride (768, 1), offset 0, requires_grad=False | 16 | 64.938 us +/- 0.883 us, var 54.689 | 47.202 us +/- 0.444 us, var 2.787 | 1.38x | `8393319226522702544`/`8393319226522702544` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.add` | tensor/tensor | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 0.381 us +/- 0.002 us, var 0.000 | 1.229 us +/- 0.007 us, var 0.001 | 0.31x | `18385694370852161198`/`18385694370852161198` |
| `empty_tensor_scalar_3x0x2` | empty | `torch.add` | tensor/scalar | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); scalar 4.0 | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 1.159 us +/- 0.006 us, var 0.001 | 2.687 us +/- 0.021 us, var 0.209 | 0.43x | `18385694370852161198`/`18385694370852161198` |
| `empty_scalar_tensor_3x0x2` | empty | `torch.add` | scalar/tensor | scalar 4.0; right zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2) | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 1.164 us +/- 0.005 us, var 0.002 | 2.702 us +/- 0.019 us, var 0.051 | 0.43x | `18385694370852161198`/`18385694370852161198` |
| `offset_transposed_521x509` | offset | `torch.add` | tensor/tensor | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (509, 1), offset 0, requires_grad=False | 5 | 141.067 us +/- 6.592 us, var 141.663 | 38.450 us +/- 0.715 us, var 5.440 | 3.67x | `17574006401166824078`/`17574006401166824078` |
| `offset_tensor_scalar_521x509` | offset tensor/scalar | `torch.add` | tensor/scalar | left tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189; scalar -2.25 | (521, 509), stride (509, 1), offset 0, requires_grad=False | 5 | 36.546 us +/- 1.778 us, var 10.303 | 29.720 us +/- 0.747 us, var 3.928 | 1.23x | `4847602430464324808`/`4847602430464324808` |
| `offset_scalar_tensor_521x509` | offset scalar/tensor | `torch.add` | scalar/tensor | scalar -2.25; right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (509, 1), offset 0, requires_grad=False | 5 | 32.932 us +/- 1.524 us, var 14.614 | 29.899 us +/- 0.714 us, var 3.801 | 1.10x | `4847602430464324808`/`4847602430464324808` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.add` | tensor/tensor | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1024, 1), offset 0, requires_grad=False | 5 | 295.040 us +/- 7.102 us, var 662.138 | 78.151 us +/- 3.501 us, var 251.749 | 3.78x | `13019278377776432875`/`13019278377776432875` |
| `signed_zero_nan_inf` | signed-zero NaN/inf | `torch.add` | tensor/tensor | special float32 bit patterns [0, -0, 1, -1, inf, -inf, qnan] plus [-0, 0, -1, 1, inf, -inf, 0] | (7,), stride (1,), offset 0, requires_grad=False | 10000 | 0.294 us +/- 0.002 us, var 0.000 | 1.219 us +/- 0.008 us, var 0.009 | 0.24x | `4097696838118711365`/`4097696838118711365` |
| `signed_zero_nan_inf_tensor_scalar` | signed-zero NaN/inf | `torch.add` | tensor/scalar | special float32 bit patterns [0, -0, 1, -1, inf, -inf, qnan] plus scalar -0.0 | (7,), stride (1,), offset 0, requires_grad=False | 10000 | 1.155 us +/- 0.006 us, var 0.001 | 2.706 us +/- 0.029 us, var 0.066 | 0.43x | `3447670289282280123`/`3447670289282280123` |
| `signed_zero_nan_inf_scalar_tensor` | signed-zero NaN/inf | `torch.add` | scalar/tensor | scalar inf plus special float32 bit patterns [0, -0, 1, -1, inf, -inf, qnan] | (7,), stride (1,), offset 0, requires_grad=False | 10000 | 1.213 us +/- 0.009 us, var 0.007 | 2.898 us +/- 0.074 us, var 0.084 | 0.42x | `2370142114559020502`/`2370142114559020502` |
| `autograd_forward_tensor_tensor_127x131` | autograd forward | `torch.add` | tensor/tensor | left/right leaves (127, 131), requires_grad=True; forward construction only | (127, 131), stride (131, 1), offset 0, requires_grad=True | 20 | 2.520 us +/- 0.042 us, var 0.087 | 3.405 us +/- 0.021 us, var 0.100 | 0.74x | `10759111294308643066`/`10759111294308643066` |
| `autograd_forward_tensor_scalar_127x131` | autograd forward | `torch.add` | tensor/scalar | left leaf (127, 131), requires_grad=True; scalar -2.25; forward construction only | (127, 131), stride (131, 1), offset 0, requires_grad=True | 20 | 2.686 us +/- 0.023 us, var 0.060 | 4.537 us +/- 0.043 us, var 0.146 | 0.59x | `9469296913120180631`/`9469296913120180631` |
| `autograd_forward_scalar_tensor_127x131` | autograd forward | `torch.add` | scalar/tensor | scalar -2.25; right leaf (127, 131), requires_grad=True; forward construction only | (127, 131), stride (131, 1), offset 0, requires_grad=True | 20 | 2.693 us +/- 0.028 us, var 0.177 | 4.596 us +/- 0.165 us, var 0.180 | 0.59x | `9469296913120180631`/`9469296913120180631` |
| `autograd_forward_backward_tensor_tensor_32x33` | autograd forward+backward | `torch.add` | tensor/tensor | left/right leaves (32, 33), requires_grad=True; timed add(...).sum().backward() | scalar loss plus left/right leaf gradients | 5 | 15.095 us +/- 0.076 us, var 0.682 | 26.752 us +/- 0.638 us, var 2.434 | 0.56x | `6794213895464230606`/`6794213895464230606` |
| `autograd_forward_backward_tensor_scalar_32x33` | autograd forward+backward | `torch.add` | tensor/scalar | left leaf (32, 33), requires_grad=True; scalar 2.0; timed add(...).sum().backward() | scalar loss plus leaf gradient | 5 | 4.092 us +/- 0.029 us, var 0.751 | 27.431 us +/- 0.463 us, var 2.785 | 0.15x | `8154245054797153225`/`8154245054797153225` |
| `autograd_forward_backward_scalar_tensor_32x33` | autograd forward+backward | `torch.add` | scalar/tensor | scalar 2.0; right leaf (32, 33), requires_grad=True; timed add(...).sum().backward() | scalar loss plus leaf gradient | 5 | 4.090 us +/- 0.039 us, var 0.231 | 27.811 us +/- 0.282 us, var 3.935 | 0.15x | `8154245054797153225`/`8154245054797153225` |
| `no_grad_tensor_tensor_257x263` | no_grad tensor/tensor | `torch.add` | tensor/tensor | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 10.596 us +/- 0.173 us, var 0.246 | 12.691 us +/- 0.198 us, var 0.239 | 0.83x | `12504381733898885835`/`12504381733898885835` |
| `no_grad_tensor_scalar_257x263` | no_grad tensor/scalar | `torch.add` | tensor/scalar | left leaf (257, 263), requires_grad=True; scalar 2.0; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 8.130 us +/- 0.155 us, var 0.456 | 10.944 us +/- 0.159 us, var 0.367 | 0.74x | `11971956308191282863`/`11971956308191282863` |
| `no_grad_scalar_tensor_257x263` | no_grad scalar/tensor | `torch.add` | scalar/tensor | scalar 2.0; right leaf (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 8.178 us +/- 0.167 us, var 0.387 | 10.939 us +/- 0.176 us, var 0.132 | 0.75x | `11971956308191282863`/`11971956308191282863` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `top_level_torch_add_nondefault_alpha_tensor_tensor` | `NotImplementedError: add(): alpha values other than 1 are not supported` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_add_nondefault_alpha_tensor_scalar` | `NotImplementedError: add(): alpha values other than 1 are not supported` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_add_nondefault_alpha_scalar_tensor` | `NotImplementedError: add(): alpha values other than 1 are not supported` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_add_out_tensor_tensor` | `RuntimeError: add(): the 'out' argument is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_add_out_tensor_scalar` | `RuntimeError: add(): the 'out' argument is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_add_out_scalar_tensor` | `RuntimeError: add(): the 'out' argument is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_add_scalar_scalar` | `NotImplementedError: add(): only exact native CPU float32 Tensor/Tensor, Tensor/real-number, or real-number/Tensor operands are supported` | supported `(), stride (), offset 0, requires_grad=False` | zero |
