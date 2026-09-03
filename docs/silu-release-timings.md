# `torch.nn.functional.silu` Release Timings

Date: 2026-09-03

Candidate provenance: source snapshot based on
`73add5da047ea44b54b10b5fa9843fd83cd61220`, plus the worktree changes that
add the native no-grad inference path for `torch.nn.functional.silu`.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/silu-release-timings*.json`. The active Conda
environment was used for Cargo Python-binding lint and tests. A worktree-local
`.venv` was used for the PyTorch 2.13 reference dependency because the active
Conda environment had a newer PyTorch build. Cargo registry data was copied
read-only from the existing user cache into `target/cargo-home`, then Cargo ran
offline so build artifacts and dependency state stayed inside this worktree.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
mkdir -p target/cargo-home/registry
cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/
env -u CONDA_PREFIX \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked --offline
env -u CONDA_PREFIX \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/silu_release_timings.py \
  > target/silu-release-timings.json
env -u CONDA_PREFIX \
  SILU_TIMING_IMPL_ORDER=pytorch,torch_rs_composition,torch_rs \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/silu_release_timings.py \
  > target/silu-release-timings-pass2.json
```

Checks run for this evidence:

```bash
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo clippy --locked --offline --all-targets -- -D warnings
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  PYO3_PYTHON="$CONDA_PREFIX/bin/python" \
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
  cargo clippy --locked --offline --all-targets --features python-bindings -- -D warnings
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  PYO3_PYTHON="$CONDA_PREFIX/bin/python" \
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
  cargo test --locked --offline --all-targets --features python-bindings
env -u CONDA_PREFIX \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_silu \
  tests.test_nn_functional_silu_reference
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 14 tests. The full native Rust suite passed 307 tests, the full
`python-bindings` Rust suite passed 318 tests, both Clippy configurations
passed, `cargo fmt --check` passed, and `git diff --check` passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0
- `torch_rs`: 0.1.0 from the release editable install
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
  packages in 16.05s, and installed in 4.68s for the initial local reference
  environment
- Build time: the final offline release extension rebuild completed in 30.21s

Inputs were created outside the timed region with NumPy seed `20260903`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. The timed implementations were the fused
`torch_rs.nn.functional.silu`, explicit `torch_rs` primitive composition
`input * input.sigmoid()`, and PyTorch 2.13 `torch.nn.functional.silu`.
Every timing cell ran in two pinned process passes. The first pass measured
`torch_rs`, the composition, then PyTorch; the second pass reversed that order.
Each pass used 15 untimed warmup blocks and 81 measured blocks. A block repeated
the operation according to the table's `Repeats` column; times below are median
microseconds per operation. Reported medians are medians of the two per-process
medians. MAD and variance are the medians of the per-process MAD and sample
variance values.

Before timing each supported cell, the driver checked shape, stride, storage
offset, contiguity, channels-last contiguity, dtype, device, `requires_grad`,
leaf status, fresh output storage, output values with `rtol=2e-6`,
`atol=nextafter(float32(0), float32(1))`, `equal_nan=True`, and input
nonmutation. After every warmup and measured block, the driver consumed the
last output as a 64-bit BLAKE2b rolling checksum over tensor metadata and
logical bytes. The checksum column shows one final rolling sink as
`torch_rs`/composition/PyTorch; both process passes produced the same sinks.
NaN payloads are implementation-defined across the explicit `torch_rs`
composition and PyTorch reference in some cells, so value validation used the
stated numerical criterion rather than cross-library bit equality. The focused
tests cover non-NaN edge results against the existing `torch_rs` composition
and cover native quieting of signaling and quiet NaN inputs directly.

`torch_rs / PyTorch` and `torch_rs / composition` are slowdown ratios, so lower
is better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean slowdown for the supported `silu(input, inplace=False)` cells:

- Fused `torch_rs` / PyTorch: 2.33x uncapped, 2.33x capped
- Explicit `torch_rs` composition / PyTorch: 2.19x uncapped, 2.19x capped
- Fused `torch_rs` / explicit `torch_rs` composition: 1.06x uncapped, 1.06x
  capped

The native path removes the intermediate tensor and public composition dispatch
for the supported no-grad/untracked surface, and it wins scalar and empty
latency cells in this run. The larger finite CPU cells remain dominated by the
current scalar `exp` element loop and measured slightly slower than the
explicit primitive composition.

| Workload | Category | Input | Output | Repeats | fused `torch_rs` median +/- MAD, variance | composition median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | `torch_rs` / composition | Materialized checksums |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `scalar` | scalar | (), stride (), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 10000 | 0.391 us +/- 0.002, var 0.001 | 0.402 us +/- 0.002, var 0.000 | 1.278 us +/- 0.013, var 0.018 | 0.31x | 0.97x | `8012886692243182955`/`8012886692243182955`/`8012886692243182955` |
| `empty` | empty | (0, 2), stride (3, 3), offset 1, requires_grad=False | (0, 2), stride (2, 1), offset 0, requires_grad=False | 5000 | 0.390 us +/- 0.004, var 0.002 | 0.426 us +/- 0.004, var 0.005 | 1.157 us +/- 0.006, var 0.013 | 0.34x | 0.92x | `15265362922237676529`/`15265362922237676529`/`15265362922237676529` |
| `contiguous_257x263` | contiguous | (257, 263), stride (263, 1), offset 0, requires_grad=False | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 196.387 us +/- 2.696, var 177.242 | 164.024 us +/- 1.529, var 27.456 | 32.106 us +/- 0.376, var 2.365 | 6.12x | 1.20x | `6663207557751864570`/`6663207557751864570`/`13719881266753521821` |
| `offset_257x263` | offset | (257, 263), stride (263, 1), offset 67591, requires_grad=False | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 205.998 us +/- 4.571, var 551.299 | 171.942 us +/- 2.359, var 37.268 | 32.265 us +/- 0.315, var 1.797 | 6.38x | 1.20x | `14697570763432741206`/`14697570763432741206`/`4751838546607490907` |
| `noncontig_transpose_512x1024` | noncontiguous | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 1571.182 us +/- 88.289, var 12673.622 | 1615.086 us +/- 69.621, var 22537.505 | 244.313 us +/- 2.731, var 117.805 | 6.43x | 0.97x | `8878766051455339280`/`8878766051455339280`/`14917614665542228041` |
| `channels_last_8x15x31x33` | channels_last | (8, 15, 31, 33), stride (15345, 1, 495, 15), offset 0, requires_grad=False | (8, 15, 31, 33), stride (15345, 1, 495, 15), offset 0, requires_grad=False | 8 | 371.379 us +/- 8.000, var 1705.471 | 339.438 us +/- 2.398, var 118.703 | 58.554 us +/- 1.104, var 6.709 | 6.34x | 1.09x | `16825153328568895560`/`16825153328568895560`/`13179330306797967477` |
| `channels_last_3d_2x5x11x13x17` | channels_last_3d | (2, 5, 11, 13, 17), stride (12155, 1, 1105, 85, 5), offset 0, requires_grad=False | (2, 5, 11, 13, 17), stride (12155, 1, 1105, 85, 5), offset 0, requires_grad=False | 32 | 74.656 us +/- 1.076, var 50.918 | 66.617 us +/- 0.327, var 2.952 | 12.368 us +/- 0.352, var 0.950 | 6.04x | 1.12x | `17332973070271646594`/`17332973070271646594`/`12264295647348829078` |
| `quiet_nan_inf_signed_zero` | edge_values | (25,), stride (1,), offset 0, requires_grad=False | (25,), stride (1,), offset 0, requires_grad=False | 10000 | 0.534 us +/- 0.004, var 0.006 | 0.567 us +/- 0.004, var 0.000 | 1.493 us +/- 0.036, var 0.009 | 0.36x | 0.94x | `7569459385417993172`/`11878134366299863279`/`16668170615565749775` |
| `no_grad_tracked_257x263` | no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=True | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 196.261 us +/- 0.823, var 101.764 | 163.419 us +/- 1.174, var 75.020 | 34.193 us +/- 0.203, var 5.687 | 5.74x | 1.20x | `12780442928955978301`/`12780442928955978301`/`11366402658142069887` |

## Boundary Notes

The optimized path is intentionally limited to exact CPU float32 tensors for
`inplace=False` when either grad mode is disabled or the input does not require
grad. Active autograd on a tracked input continues to use the existing
`input * input.sigmoid()` composition so the supported backward graph remains
unchanged. `inplace=True`, `Tensor.silu`, `torch.silu`, and `nn.SiLU` remain
outside this repository's supported surface.
