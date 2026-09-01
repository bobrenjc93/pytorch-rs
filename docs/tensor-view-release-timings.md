# Tensor View Operation Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`46710a4ee8d14f3f305df16fe7851786d38a3e97`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/tensor-view-release-timings*.json`. No Conda
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
mkdir -p target/cargo-home/registry
cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/
wheel_dir="$(mktemp -d "$PWD/target/tensor-view-wheels.XXXXXX")"
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
printf '%s\n' "$wheel_dir" > target/tensor-view-wheel-dir.txt
wheel_dir="$(cat target/tensor-view-wheel-dir.txt)"
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_view tests.test_view_reference \
  tests.test_reshape tests.test_reshape_reference \
  tests.test_flatten tests.test_flatten_reference \
  tests.test_ravel tests.test_ravel_reference \
  tests.test_tensor_newaxis tests.test_tensor_newaxis_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets reshape
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets view
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets flatten
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets unsqueeze
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets ravel
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets sum
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_view_release_timings.py \
  > target/tensor-view-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  TENSOR_VIEW_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python target/tensor_view_release_timings.py \
  > target/tensor-view-release-timings-pass2.json
```

Checks run for this evidence:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_view tests.test_view_reference \
  tests.test_reshape tests.test_reshape_reference \
  tests.test_flatten tests.test_flatten_reference \
  tests.test_ravel tests.test_ravel_reference \
  tests.test_tensor_newaxis tests.test_tensor_newaxis_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets reshape
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets view
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets flatten
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets unsqueeze
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets ravel
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets sum
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 114 tests. `cargo fmt --check` passed. The filtered native Rust tests
passed 5 `reshape` tests, 37 `view` tests, 4 `flatten` tests, 2 `ravel`
tests, and 35 `sum` tests; the `unsqueeze` native filter matched no Rust
tests but completed successfully. The README/docs smoke test passed 7 tests,
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
- Dependency installation: locked `uv sync` resolved in 29 ms, prepared
  packages in 15.85s, and installed in 860 ms
- Build time: successful offline release extension build completed in 35.19s;
  the release wheel reinstall resolved in 2 ms, prepared in 45 ms, and
  installed in 11 ms

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
The first pass measured `torch_rs` before PyTorch; the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks.
A block repeated the operation according to the table's `Repeats` column;
times below are median microseconds per operation. Reported medians are
medians of the two per-process medians. MAD and variance are the medians of the
per-process MAD and sample variance values.

Before timing each supported forward cell, the driver bit-compared `torch_rs`
output values with PyTorch and checked shape, stride, storage offset,
contiguity, dtype, device, `requires_grad`, leaf status, and whether the output
aliased the input storage. Before timing each backward cell, it checked the
view/copy output metadata and values, checked the scalar full-`sum` loss with
`rtol=1e-5`, `atol=1e-4`, and `equal_nan=True`, then bit-compared the leaf
gradient and checked its metadata. Backward timings used pre-created fresh
leaf/view inputs for every measured invocation so the timed region did not
include input construction and did not reuse a freed graph.

After every warmup and measured block, the driver consumed the last forward
output, or the last backward view/copy output plus scalar loss plus leaf
gradient, as a 64-bit BLAKE2b rolling checksum over tensor metadata and
logical bytes. The checksum column shows the final rolling sink from one pass
as `torch_rs`/PyTorch; both process passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 0.35x uncapped, 0.35x capped
- Forward metadata/view-or-copy cells: 0.52x uncapped, 0.52x capped
- Alias/view forward cells: 0.34x uncapped, 0.34x capped
- Copy-producing forward cells: 1.78x uncapped, 1.78x capped
- Backward-through-full-`sum` cells: 0.13x uncapped, 0.13x capped
- `Tensor.view` cells: 0.27x uncapped, 0.27x capped
- `Tensor.reshape` cells: 0.34x uncapped, 0.34x capped
- `torch.reshape` cells: 0.88x uncapped, 0.88x capped
- Flatten cells: 0.39x uncapped, 0.39x capped
- Ravel cells: 0.51x uncapped, 0.51x capped
- Edge unsqueeze cells: 0.19x uncapped, 0.19x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 0.44x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `view_contiguous_257x263_to_263x257` | view alias | `Tensor.view(263, 257)` | contiguous rank-2; (257, 263), stride (263, 1), offset 0, requires_grad=False | (263, 257), stride (257, 1), offset 0, requires_grad=False, aliases_source=True | 10000 | 0.309 us +/- 0.004 us, var 0.000 | 0.716 us +/- 0.004 us, var 0.000 | 0.43x | `3684230127886259054`/`3684230127886259054` |
| `view_sequence_offset_257x263_to_263x257` | view alias | `Tensor.view((263, 257))` | offset rank-2; (257, 263), stride (263, 1), offset 67591, requires_grad=False | (263, 257), stride (257, 1), offset 67591, requires_grad=False, aliases_source=True | 10000 | 0.299 us +/- 0.002 us, var 0.000 | 1.344 us +/- 0.007 us, var 0.003 | 0.22x | `8797383534501012474`/`8797383534501012474` |
| `view_noncontig_compatible_split_3x2x2x2` | view alias | `Tensor.view(3, 2, 2, 2)` | compatible noncontiguous transpose; (3, 2, 4), stride (4, 12, 1), offset 0, requires_grad=False | (3, 2, 2, 2), stride (4, 12, 2, 1), offset 0, requires_grad=False, aliases_source=True | 10000 | 0.378 us +/- 0.002 us, var 0.000 | 0.987 us +/- 0.004 us, var 0.000 | 0.38x | `12833908657852249775`/`12833908657852249775` |
| `view_same_dtype_detached_alias` | dtype metadata alias | `Tensor.view(torch.float32)` | requires-grad offset transpose; (131, 127), stride (1, 131), offset 16637, requires_grad=True | (131, 127), stride (1, 131), offset 16637, requires_grad=False, aliases_source=True | 10000 | 0.187 us +/- 0.001 us, var 0.000 | 0.547 us +/- 0.003 us, var 0.000 | 0.34x | `4773619416381842949`/`4773619416381842949` |
| `reshape_contiguous_alias_257x263_to_263x257` | reshape alias | `Tensor.reshape((263, 257))` | contiguous rank-2; (257, 263), stride (263, 1), offset 0, requires_grad=False | (263, 257), stride (257, 1), offset 0, requires_grad=False, aliases_source=True | 10000 | 0.246 us +/- 0.002 us, var 0.000 | 1.175 us +/- 0.007 us, var 0.008 | 0.21x | `3684230127886259054`/`3684230127886259054` |
| `reshape_transpose_copy_263x257_to_257x263` | reshape copy | `Tensor.reshape((257, 263))` | incompatible transposed rank-2; (263, 257), stride (1, 263), offset 0, requires_grad=False | (257, 263), stride (263, 1), offset 0, requires_grad=False, aliases_source=False | 16 | 68.661 us +/- 0.825 us, var 31.006 | 39.891 us +/- 0.398 us, var 3.430 | 1.72x | `7777634252196889447`/`7777634252196889447` |
| `torch_reshape_sequence_tuple_alias_257x263` | torch.reshape alias | `torch.reshape(input, (263, 257))` | contiguous rank-2; (257, 263), stride (263, 1), offset 0, requires_grad=False | (263, 257), stride (257, 1), offset 0, requires_grad=False, aliases_source=True | 10000 | 1.789 us +/- 0.013 us, var 0.004 | 0.785 us +/- 0.004 us, var 0.000 | 2.28x | `3684230127886259054`/`3684230127886259054` |
| `torch_reshape_sequence_list_copy_263x257` | torch.reshape copy | `torch.reshape(input, [257, 263])` | incompatible transposed rank-2; (263, 257), stride (1, 263), offset 0, requires_grad=False | (257, 263), stride (263, 1), offset 0, requires_grad=False, aliases_source=False | 16 | 70.270 us +/- 0.652 us, var 7.298 | 39.338 us +/- 0.558 us, var 2.142 | 1.79x | `7777634252196889447`/`7777634252196889447` |
| `flatten_full_contiguous_view` | flatten alias | `Tensor.flatten()` | contiguous rank-2; (257, 263), stride (263, 1), offset 0, requires_grad=False | (67591,), stride (1,), offset 0, requires_grad=False, aliases_source=True | 10000 | 0.204 us +/- 0.002 us, var 0.000 | 0.668 us +/- 0.003 us, var 0.001 | 0.31x | `16457668958880272358`/`16457668958880272358` |
| `torch_flatten_full_contiguous_view` | torch.flatten alias | `torch.flatten(input)` | contiguous rank-2; (257, 263), stride (263, 1), offset 0, requires_grad=False | (67591,), stride (1,), offset 0, requires_grad=False, aliases_source=True | 10000 | 0.232 us +/- 0.001 us, var 0.000 | 0.729 us +/- 0.003 us, var 0.002 | 0.32x | `16457668958880272358`/`16457668958880272358` |
| `flatten_partial_offset_view` | flatten alias | `Tensor.flatten(1, -1)` | offset compatible range; (2, 64, 5), stride (960, 5, 1), offset 320, requires_grad=False | (2, 320), stride (960, 1), offset 320, requires_grad=False, aliases_source=True | 10000 | 0.271 us +/- 0.004 us, var 0.000 | 1.246 us +/- 0.007 us, var 0.003 | 0.22x | `12477147639547370654`/`12477147639547370654` |
| `flatten_transpose_copy` | flatten copy | `Tensor.flatten()` | incompatible transposed rank-2; (263, 257), stride (1, 263), offset 0, requires_grad=False | (67591,), stride (1,), offset 0, requires_grad=False, aliases_source=False | 16 | 68.733 us +/- 0.822 us, var 11.769 | 38.076 us +/- 0.626 us, var 37.205 | 1.81x | `9654457142591791356`/`9654457142591791356` |
| `torch_flatten_transpose_copy` | torch.flatten copy | `torch.flatten(input)` | incompatible transposed rank-2; (263, 257), stride (1, 263), offset 0, requires_grad=False | (67591,), stride (1,), offset 0, requires_grad=False, aliases_source=False | 16 | 68.753 us +/- 0.925 us, var 10.382 | 38.011 us +/- 0.500 us, var 2.366 | 1.81x | `9654457142591791356`/`9654457142591791356` |
| `ravel_contiguous_view` | ravel alias | `Tensor.ravel()` | contiguous rank-2; (257, 263), stride (263, 1), offset 0, requires_grad=False | (67591,), stride (1,), offset 0, requires_grad=False, aliases_source=True | 10000 | 0.232 us +/- 0.002 us, var 0.000 | 0.570 us +/- 0.007 us, var 0.001 | 0.41x | `16457668958880272358`/`16457668958880272358` |
| `torch_ravel_contiguous_view` | torch.ravel alias | `torch.ravel(input)` | contiguous rank-2; (257, 263), stride (263, 1), offset 0, requires_grad=False | (67591,), stride (1,), offset 0, requires_grad=False, aliases_source=True | 10000 | 0.444 us +/- 0.006 us, var 0.000 | 0.678 us +/- 0.006 us, var 0.000 | 0.65x | `16457668958880272358`/`16457668958880272358` |
| `torch_ravel_transpose_copy` | ravel copy | `torch.ravel(input)` | incompatible transposed rank-2; (263, 257), stride (1, 263), offset 0, requires_grad=False | (67591,), stride (1,), offset 0, requires_grad=False, aliases_source=False | 16 | 68.777 us +/- 0.765 us, var 3.648 | 38.177 us +/- 0.434 us, var 1.379 | 1.80x | `9654457142591791356`/`9654457142591791356` |
| `unsqueeze_front_contiguous` | edge unsqueeze alias | `Tensor.unsqueeze(0)` | contiguous rank-2; (257, 263), stride (263, 1), offset 0, requires_grad=False | (1, 257, 263), stride (67591, 263, 1), offset 0, requires_grad=False, aliases_source=True | 10000 | 0.210 us +/- 0.003 us, var 0.000 | 0.776 us +/- 0.005 us, var 0.000 | 0.27x | `12681282311643249535`/`12681282311643249535` |
| `unsqueeze_back_offset` | edge unsqueeze alias | `Tensor.unsqueeze(-1)` | offset rank-2; (257, 263), stride (263, 1), offset 67591, requires_grad=False | (257, 263, 1), stride (263, 1, 1), offset 67591, requires_grad=False, aliases_source=True | 10000 | 0.202 us +/- 0.002 us, var 0.000 | 0.938 us +/- 0.007 us, var 0.003 | 0.22x | `17417893726523753982`/`17417893726523753982` |
| `torch_unsqueeze_front_empty` | edge unsqueeze alias | `torch.unsqueeze(input, 0)` | empty offset; (0, 2), stride (3, 3), offset 1, requires_grad=False | (1, 0, 2), stride (0, 3, 3), offset 1, requires_grad=False, aliases_source=True | 10000 | 0.213 us +/- 0.003 us, var 0.000 | 0.907 us +/- 0.005 us, var 0.000 | 0.23x | `15115191713181669326`/`15115191713181669326` |
| `torch_unsqueeze_back_noncontig` | edge unsqueeze alias | `torch.unsqueeze(input, -1)` | offset noncontiguous transpose; (131, 127), stride (1, 131), offset 16637, requires_grad=False | (131, 127, 1), stride (1, 131, 1), offset 16637, requires_grad=False, aliases_source=True | 10000 | 0.212 us +/- 0.003 us, var 0.000 | 0.945 us +/- 0.005 us, var 0.000 | 0.22x | `3853641471572560148`/`3853641471572560148` |
| `view_full_sum_backward_contiguous` | backward full-sum | `Tensor.view(17, 16).sum().backward()` | fresh contiguous leaf; (16, 17), stride (17, 1), offset 0, requires_grad=True | view (17, 16), stride (16, 1), offset 0, requires_grad=True, aliases_source=True; loss (); leaf grad (16, 17), stride (17, 1) | 10 | 3.070 us +/- 0.062 us, var 0.089 | 26.081 us +/- 0.512 us, var 4.447 | 0.12x | `12811304792050903977`/`11626816504167513736` |
| `reshape_full_sum_backward_alias` | backward full-sum | `Tensor.reshape((17, 16)).sum().backward()` | fresh contiguous leaf; (16, 17), stride (17, 1), offset 0, requires_grad=True | view (17, 16), stride (16, 1), offset 0, requires_grad=True, aliases_source=True; loss (); leaf grad (16, 17), stride (17, 1) | 10 | 3.051 us +/- 0.068 us, var 0.192 | 27.791 us +/- 0.506 us, var 1.932 | 0.11x | `12811304792050903977`/`11626816504167513736` |
| `torch_reshape_full_sum_backward_copy` | backward full-sum | `torch.reshape(input, (272,)).sum().backward()` | fresh transposed leaf view; (17, 16), stride (1, 17), offset 0, requires_grad=True | view (272,), stride (1,), offset 0, requires_grad=True, aliases_source=False; loss (); leaf grad (16, 17), stride (17, 1) | 10 | 5.805 us +/- 0.288 us, var 0.599 | 34.413 us +/- 1.001 us, var 10.598 | 0.17x | `14335205718186998061`/`14335205718186998061` |
| `flatten_full_sum_backward_view` | backward full-sum | `Tensor.flatten().sum().backward()` | fresh contiguous leaf; (16, 17), stride (17, 1), offset 0, requires_grad=True | view (272,), stride (1,), offset 0, requires_grad=True, aliases_source=True; loss (); leaf grad (16, 17), stride (17, 1) | 10 | 3.000 us +/- 0.081 us, var 0.136 | 25.731 us +/- 0.389 us, var 2.836 | 0.12x | `17773958687869724300`/`11701204518769238766` |
| `torch_flatten_full_sum_backward_copy` | backward full-sum | `torch.flatten(input).sum().backward()` | fresh transposed leaf view; (17, 16), stride (1, 17), offset 0, requires_grad=True | view (272,), stride (1,), offset 0, requires_grad=True, aliases_source=False; loss (); leaf grad (16, 17), stride (17, 1) | 10 | 5.631 us +/- 0.654 us, var 1.028 | 33.581 us +/- 0.911 us, var 4.800 | 0.17x | `14335205718186998061`/`14335205718186998061` |
| `torch_ravel_full_sum_backward_copy` | backward full-sum | `torch.ravel(input).sum().backward()` | fresh transposed leaf view; (17, 16), stride (1, 17), offset 0, requires_grad=True | view (272,), stride (1,), offset 0, requires_grad=True, aliases_source=False; loss (); leaf grad (16, 17), stride (17, 1) | 10 | 5.553 us +/- 0.118 us, var 0.224 | 39.093 us +/- 3.760 us, var 42.168 | 0.14x | `14335205718186998061`/`14335205718186998061` |
| `unsqueeze_front_full_sum_backward` | backward full-sum | `Tensor.unsqueeze(0).sum().backward()` | fresh contiguous leaf; (16, 17), stride (17, 1), offset 0, requires_grad=True | view (1, 16, 17), stride (272, 17, 1), offset 0, requires_grad=True, aliases_source=True; loss (); leaf grad (16, 17), stride (17, 1) | 10 | 3.004 us +/- 0.079 us, var 0.185 | 25.763 us +/- 0.368 us, var 1.374 | 0.12x | `17386326298185700165`/`8775146350304270789` |
| `torch_unsqueeze_back_full_sum_backward` | backward full-sum | `torch.unsqueeze(input, -1).sum().backward()` | fresh contiguous leaf; (16, 17), stride (17, 1), offset 0, requires_grad=True | view (16, 17, 1), stride (17, 1, 1), offset 0, requires_grad=True, aliases_source=True; loss (); leaf grad (16, 17), stride (17, 1) | 10 | 3.051 us +/- 0.066 us, var 0.060 | 25.757 us +/- 0.590 us, var 5.303 | 0.12x | `8007196407971233848`/`8775988416951513829` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `tensor_unsqueeze_middle_dim` | `NotImplementedError: unsqueeze(): only leading and trailing dimensions are supported; got normalized dim 1 for input with 3 dimensions` | `supported (2, 1, 3, 4), stride (12, 12, 4, 1), offset 0, requires_grad=False` | zero |
| `torch_unsqueeze_middle_dim` | `NotImplementedError: unsqueeze(): only leading and trailing dimensions are supported; got normalized dim 1 for input with 3 dimensions` | `supported (2, 1, 3, 4), stride (12, 12, 4, 1), offset 0, requires_grad=False` | zero |
