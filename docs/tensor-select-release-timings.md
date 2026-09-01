# `Tensor.select` and `torch.select` Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`0a6fe63470fd7787b226c1a1384e185d7e7d00a4`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/tensor-select-release-timings*.json`. No Conda
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
wheel_dir="$(mktemp -d "$PWD/target/tensor-select-wheels.XXXXXX")"
printf '%s\n' "$wheel_dir" > target/tensor-select-wheel-dir.txt
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
wheel_dir="$(cat target/tensor-select-wheel-dir.txt)"
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_select tests.test_select_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets select
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_select_release_timings.py \
  > target/tensor-select-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  TENSOR_SELECT_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python target/tensor_select_release_timings.py \
  > target/tensor-select-release-timings-pass2.json
```

Checks run for this evidence:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_select tests.test_select_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets select
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check -- BENCHMARKING.md
awk '/[[:blank:]]$/ { print FILENAME ":" FNR ": trailing whitespace"; bad=1 } END { exit bad }' \
  docs/tensor-select-release-timings.md
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 25 tests. The focused Rust `select` filter passed 7 native tests across
the library and autograd test targets, `cargo fmt --check` passed, the
README/docs smoke test passed 7 tests, the tracked diff whitespace check
passed, and the new report had no trailing whitespace.

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
- Dependency installation: locked `uv sync` resolved in 31 ms, prepared
  packages in 15.74s, and installed in 993 ms
- Build time: successful offline release extension build completed in 34.75s;
  the release wheel reinstall resolved in 1 ms, prepared in 40 ms, and
  installed in 115 ms

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
matched an independently constructed direct-indexing alias via both
`data_ptr()` and `is_set_to()`. Before timing each backward cell, it checked
the view output metadata and values, checked the scalar full-`sum` loss with
`rtol=1e-5`, `atol=1e-4`, then bit-compared the leaf gradient and checked its
metadata. Backward timings used pre-created fresh leaf/view inputs for every
measured invocation so the timed region did not include input construction and
did not reuse a freed graph.

After every warmup and measured block, the driver consumed the last forward
output, or the last backward view output plus scalar loss plus leaf gradient,
as a 64-bit BLAKE2b rolling checksum over tensor metadata and logical bytes.
The checksum column shows the final rolling sink from one pass as `torch_rs`/
PyTorch; both process passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 0.20x uncapped, 0.21x capped
- Forward view cells: 0.24x uncapped, 0.24x capped
- Backward-through-full-`sum` cells: 0.12x uncapped, 0.14x capped
- `Tensor.select` cells: 0.19x uncapped, 0.20x capped
- `torch.select` cells: 0.20x uncapped, 0.21x capped
- Leading-dimension cells: 0.15x uncapped, 0.18x capped
- Middle-dimension cells: 0.21x uncapped, 0.21x capped
- Trailing-dimension cells: 0.24x uncapped, 0.24x capped
- Negative-index cells: 0.23x uncapped, 0.23x capped
- Offset cells: 0.22x uncapped, 0.22x capped
- Noncontiguous cells: 0.22x uncapped, 0.22x capped
- Empty cells: 0.12x uncapped, 0.16x capped

| Workload | Category | API | Input / mode | Output / alias checks | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `tensor_select_leading_contiguous_257x263x3` | leading | `Tensor.select(0, 128)` | contiguous rank-3; (257, 263, 3), stride (789, 3, 1), offset 0 | (263, 3), stride (3, 1), offset 100992, contiguous=True, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.252 us +/- 0.002 us, var 0.000 | 1.010 us +/- 0.006 us, var 0.001 | 0.25x | `10230721005490785545`/`10230721005490785545` |
| `torch_select_leading_contiguous_257x263x3` | leading | `torch.select(input, 0, 128)` | contiguous rank-3; (257, 263, 3), stride (789, 3, 1), offset 0 | (263, 3), stride (3, 1), offset 100992, contiguous=True, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.275 us +/- 0.002 us, var 0.000 | 1.072 us +/- 0.006 us, var 0.000 | 0.26x | `10230721005490785545`/`10230721005490785545` |
| `tensor_select_middle_contiguous_257x263x5` | middle | `Tensor.select(1, 17)` | contiguous rank-3; (257, 263, 5), stride (1315, 5, 1), offset 0 | (257, 5), stride (1315, 1), offset 85, contiguous=False, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.251 us +/- 0.002 us, var 0.000 | 1.028 us +/- 0.005 us, var 0.001 | 0.24x | `4366768101729113649`/`4366768101729113649` |
| `torch_select_middle_contiguous_257x263x5` | middle | `torch.select(input, 1, 17)` | contiguous rank-3; (257, 263, 5), stride (1315, 5, 1), offset 0 | (257, 5), stride (1315, 1), offset 85, contiguous=False, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.274 us +/- 0.002 us, var 0.000 | 1.090 us +/- 0.005 us, var 0.000 | 0.25x | `4366768101729113649`/`4366768101729113649` |
| `tensor_select_trailing_contiguous_257x263x5` | trailing | `Tensor.select(2, 3)` | contiguous rank-3; (257, 263, 5), stride (1315, 5, 1), offset 0 | (257, 263), stride (1315, 5), offset 3, contiguous=False, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.252 us +/- 0.002 us, var 0.000 | 1.026 us +/- 0.006 us, var 0.000 | 0.25x | `5697006037415978975`/`5697006037415978975` |
| `torch_select_trailing_contiguous_257x263x5` | trailing | `torch.select(input, 2, 3)` | contiguous rank-3; (257, 263, 5), stride (1315, 5, 1), offset 0 | (257, 263), stride (1315, 5), offset 3, contiguous=False, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.276 us +/- 0.002 us, var 0.000 | 1.086 us +/- 0.006 us, var 0.010 | 0.25x | `5697006037415978975`/`5697006037415978975` |
| `tensor_select_negative_dim_negative_index_37x41x43` | negative-index | `Tensor.select(-2, -3)` | contiguous rank-3; (37, 41, 43), stride (1763, 43, 1), offset 0; dim=-2, index=-3 | (37, 43), stride (1763, 1), offset 1634, contiguous=False, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.254 us +/- 0.001 us, var 0.000 | 1.041 us +/- 0.005 us, var 0.000 | 0.24x | `16924206001078716461`/`16924206001078716461` |
| `torch_select_negative_dim_negative_index_37x41x43` | negative-index | `torch.select(input, -2, -3)` | contiguous rank-3; (37, 41, 43), stride (1763, 43, 1), offset 0; dim=-2, index=-3 | (37, 43), stride (1763, 1), offset 1634, contiguous=False, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.277 us +/- 0.001 us, var 0.000 | 1.095 us +/- 0.005 us, var 0.001 | 0.25x | `16924206001078716461`/`16924206001078716461` |
| `tensor_select_offset_middle_521x509` | offset | `Tensor.select(1, 257)` | offset rank-2 from base[1]; (521, 509), stride (509, 1), offset 265189 | (521,), stride (509,), offset 265446, contiguous=False, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.246 us +/- 0.002 us, var 0.000 | 1.170 us +/- 0.010 us, var 0.007 | 0.21x | `2489465242036595262`/`2489465242036595262` |
| `torch_select_offset_middle_521x509` | offset | `torch.select(input, 1, 257)` | offset rank-2 from base[1]; (521, 509), stride (509, 1), offset 265189 | (521,), stride (509,), offset 265446, contiguous=False, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.266 us +/- 0.002 us, var 0.000 | 1.160 us +/- 0.004 us, var 0.000 | 0.23x | `2489465242036595262`/`2489465242036595262` |
| `tensor_select_noncontig_leading_512x1024` | noncontiguous | `Tensor.select(0, 17)` | transposed rank-2; (512, 1024), stride (1, 512), offset 0 | (1024,), stride (512,), offset 17, contiguous=False, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.247 us +/- 0.002 us, var 0.000 | 1.175 us +/- 0.006 us, var 0.000 | 0.21x | `15220672236965111517`/`15220672236965111517` |
| `torch_select_noncontig_leading_512x1024` | noncontiguous | `torch.select(input, 0, 17)` | transposed rank-2; (512, 1024), stride (1, 512), offset 0 | (1024,), stride (512,), offset 17, contiguous=False, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.268 us +/- 0.002 us, var 0.000 | 1.167 us +/- 0.007 us, var 0.014 | 0.23x | `15220672236965111517`/`15220672236965111517` |
| `tensor_select_empty_middle_2x3x0x4` | empty | `Tensor.select(1, 2)` | empty rank-4; (2, 3, 0, 4), stride (12, 4, 4, 1), offset 0 | (2, 0, 4), stride (12, 4, 1), offset 8, contiguous=True, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.245 us +/- 0.001 us, var 0.000 | 0.988 us +/- 0.004 us, var 0.001 | 0.25x | `3287042868037442687`/`3287042868037442687` |
| `torch_select_empty_middle_2x3x0x4` | empty | `torch.select(input, 1, 2)` | empty rank-4; (2, 3, 0, 4), stride (12, 4, 4, 1), offset 0 | (2, 0, 4), stride (12, 4, 1), offset 8, contiguous=True, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.270 us +/- 0.003 us, var 0.000 | 1.054 us +/- 0.006 us, var 0.001 | 0.26x | `3287042868037442687`/`3287042868037442687` |
| `tensor_select_vector_to_scalar_negative_index_65521` | negative-index | `Tensor.select(-1, -2)` | contiguous rank-1; (65521,), stride (1,), offset 0; dim=-1, index=-2 | (), stride (), offset 65519, contiguous=True, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.219 us +/- 0.002 us, var 0.000 | 0.981 us +/- 0.004 us, var 0.000 | 0.22x | `10072564414427776553`/`10072564414427776553` |
| `torch_select_vector_to_scalar_negative_index_65521` | negative-index | `torch.select(input, -1, -2)` | contiguous rank-1; (65521,), stride (1,), offset 0; dim=-1, index=-2 | (), stride (), offset 65519, contiguous=True, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10000 | 0.241 us +/- 0.002 us, var 0.000 | 1.030 us +/- 0.004 us, var 0.000 | 0.23x | `10072564414427776553`/`10072564414427776553` |
| `tensor_select_backward_middle_contiguous_16x17x5` | backward full-sum | `Tensor.select(1, 7).sum().backward()` | fresh contiguous leaf; (16, 17, 5), stride (85, 5, 1), offset 0, requires_grad=True | view (16, 5), stride (85, 1), offset 35, contiguous=False, requires_grad=True, is_leaf=False; loss (); leaf grad (16, 17, 5), stride (85, 5, 1), offset 0, contiguous=True, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10 | 3.557 us +/- 0.052 us, var 0.104 | 29.574 us +/- 0.837 us, var 2.225 | 0.12x | `451680469951457177`/`14956955296260641313` |
| `torch_select_backward_middle_contiguous_16x17x5` | backward full-sum | `torch.select(input, 1, 7).sum().backward()` | fresh contiguous leaf; (16, 17, 5), stride (85, 5, 1), offset 0, requires_grad=True | view (16, 5), stride (85, 1), offset 35, contiguous=False, requires_grad=True, is_leaf=False; loss (); leaf grad (16, 17, 5), stride (85, 5, 1), offset 0, contiguous=True, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10 | 3.592 us +/- 0.062 us, var 0.058 | 29.212 us +/- 0.649 us, var 2.419 | 0.12x | `451680469951457177`/`14956955296260641313` |
| `tensor_select_backward_offset_noncontig_67x65` | backward full-sum | `Tensor.select(1, -2).sum().backward()` | fresh leaf[1].transpose(0, 1); source (67, 65), stride (1, 67), offset 4355, requires_grad=True | view (67,), stride (1,), offset 8576, contiguous=True, requires_grad=True, is_leaf=False; loss (); leaf grad (3, 65, 67), stride (4355, 67, 1), offset 0, contiguous=True, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10 | 9.278 us +/- 0.189 us, var 0.377 | 43.868 us +/- 1.021 us, var 17.165 | 0.21x | `17478758900206680721`/`6350577293912791994` |
| `torch_select_backward_offset_noncontig_67x65` | backward full-sum | `torch.select(input, 1, -2).sum().backward()` | fresh leaf[1].transpose(0, 1); source (67, 65), stride (1, 67), offset 4355, requires_grad=True | view (67,), stride (1,), offset 8576, contiguous=True, requires_grad=True, is_leaf=False; loss (); leaf grad (3, 65, 67), stride (4355, 67, 1), offset 0, contiguous=True, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 10 | 9.417 us +/- 0.231 us, var 0.498 | 44.588 us +/- 1.922 us, var 19.035 | 0.21x | `17478758900206680721`/`6350577293912791994` |
| `tensor_select_backward_empty_leading_2x0x3` | backward full-sum | `Tensor.select(0, 1).sum().backward()` | fresh empty leaf; (2, 0, 3), stride (3, 3, 1), offset 0, requires_grad=True | view (0, 3), stride (3, 1), offset 3, contiguous=True, requires_grad=True, is_leaf=False; loss (); leaf grad (2, 0, 3), stride (3, 3, 1), offset 0, contiguous=True, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 100 | 1.550 us +/- 0.015 us, var 0.005 | 26.600 us +/- 0.696 us, var 1.601 | 0.06x | `17242077774611249423`/`17242077774611249423` |
| `torch_select_backward_empty_leading_2x0x3` | backward full-sum | `torch.select(input, 0, 1).sum().backward()` | fresh empty leaf; (2, 0, 3), stride (3, 3, 1), offset 0, requires_grad=True | view (0, 3), stride (3, 1), offset 3, contiguous=True, requires_grad=True, is_leaf=False; loss (); leaf grad (2, 0, 3), stride (3, 3, 1), offset 0, contiguous=True, requires_grad=False, is_leaf=True; direct_data_ptr_match=True, direct_is_set_to=True | 100 | 1.637 us +/- 0.086 us, var 0.031 | 24.863 us +/- 0.206 us, var 0.164 | 0.07x | `17242077774611249423`/`17242077774611249423` |
