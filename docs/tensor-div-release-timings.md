# Tensor True-Division Release Timings

Date: 2026-08-31

Code revision under test: `bf3f255616331d5b83697392d99b9c9004260cd5`.
This branch did not change tensor division implementation code before the timing
run.

Build and timing commands:

```bash
UV_CACHE_DIR="$PWD/target/uv-cache" uv venv --clear --python 3.12
UV_CACHE_DIR="$PWD/target/uv-cache" uv sync --locked --no-install-project --group dev --group reference
cp -a /home/bobren/.cargo/registry target/cargo-home/
env PATH="/home/bobren/.cargo/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_NET_OFFLINE=true \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked \
  --out "$PWD/target/tensor-div-release-timings/wheels"
UV_CACHE_DIR="$PWD/target/uv-cache" uv pip install \
  --python .venv/bin/python --force-reinstall --no-deps \
  target/tensor-div-release-timings/wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
env PATH="/home/bobren/.cargo/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_NET_OFFLINE=true \
  CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  taskset -c 0 \
  .venv/bin/python target/tensor-div-release-timings/benchmark_tensor_div.py \
  --output target/tensor-div-release-timings/tensor-div-release-timings.json
```

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.12
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0 installed; `CUDA_VISIBLE_DEVICES=`
  made `torch.cuda.is_available()` report `False`, and all tensors were CPU
  tensors
- NumPy: 2.5.1
- `torch_rs`: 0.1.0
- Profile: release `maturin build --release --locked`, Cargo
  `[profile.release]` with thin LTO and one codegen unit
- Threads: `taskset -c 0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`
- Dependency installation: `uv sync` resolved 36 packages in 28 ms, prepared
  31 packages in 17.01 s, and installed 31 packages in 3.32 s; release wheel
  force-install resolved 1 package in 4 ms, prepared it in 50 ms, installed it
  in 23 ms, and took 0.24 s wall time
- Compile time: 35.98 s wall time; Cargo reported 35.71 s in the release
  profile

The timing driver constructed inputs outside the timed region, then ran 9
warmup blocks and 51 measured blocks for each `torch_rs` and PyTorch cell. Each
measured block repeated the eager operation shown in the table and reports
median microseconds per call. The driver read first, middle, and last scalar
values from every non-empty result inside the timed loop, and read zero-size
metadata for empty results. Before timing each cell, it fully materialized the
`torch_rs` and PyTorch outputs through `tolist()` and checked shape, stride,
dtype, numel, output storage offset, and exact float32 output bits. MAD is the
median absolute deviation in microseconds; variance is sample variance of the
51 per-call block timings in `us^2`. Lower ratios are faster for `torch_rs`.

Input layouts:

| Workload family | Input shapes and layouts |
| --- | --- |
| Contiguous tensor/tensor and tensor/scalar | `left` and `right`: shape `(192, 256)`, stride `(256, 1)`, offset `0`; scalar `2.75` |
| Reflected scalar contiguous | `scalar / left` with `left`: shape `(192, 256)`, stride `(256, 1)`, offset `0`; scalar `2.75` |
| Broadcasting | `left`: shape `(192, 256)`, stride `(256, 1)`, offset `0`; `right`: shape `(1, 256)`, stride `(256, 1)`, offset `0` |
| Empty tensor/tensor and tensor/scalar | `left` and `right`: shape `(0, 4096)`, stride `(4096, 1)`, offset `0`; scalar `2.75` |
| Noncontiguous offset tensor/tensor and tensor/scalar | `left`: shape `(128, 64)`, stride `(97, 12416)`, offset `1`; `right`: shape `(128, 64)`, stride `(97, 12416)`, offset `2`; scalar `2.75` |
| Reflected scalar noncontiguous offset | `scalar / left` with `left`: shape `(128, 64)`, stride `(97, 12416)`, offset `1`; scalar `2.75` |

Supported-cell geometric aggregate: `torch_rs` / PyTorch median ratio
`1.583x` across the timed cells. This aggregate covers only supported cells
with correctness gates passed; unsupported cells below remain zero-credit
denominator entries rather than being converted into timings.

| API | Workload | Expression | Output layout | Repeats | `torch_rs` median / MAD / variance | PyTorch median / MAD / variance | Ratio |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `/` | tensor/tensor contiguous | `left / right` | shape `(192, 256)`, stride `(256, 1)` | 64 | 13.410 us / 0.851 us / 3.169696 us^2 | 14.424 us / 0.578 us / 8.766568 us^2 | 0.93x |
| `Tensor.div` | tensor/tensor contiguous | `left.div(right)` | shape `(192, 256)`, stride `(256, 1)` | 64 | 13.535 us / 0.527 us / 1.011923 us^2 | 13.686 us / 0.426 us / 2.447729 us^2 | 0.99x |
| `Tensor.divide` | tensor/tensor contiguous | `left.divide(right)` | shape `(192, 256)`, stride `(256, 1)` | 64 | 13.501 us / 0.564 us / 0.964952 us^2 | 14.586 us / 1.401 us / 12.748122 us^2 | 0.93x |
| `/` | tensor/scalar contiguous | `left / scalar` | shape `(192, 256)`, stride `(256, 1)` | 64 | 12.999 us / 0.322 us / 0.303370 us^2 | 15.462 us / 1.080 us / 5.563050 us^2 | 0.84x |
| `Tensor.div` | tensor/scalar contiguous | `left.div(scalar)` | shape `(192, 256)`, stride `(256, 1)` | 64 | 13.827 us / 0.374 us / 0.499118 us^2 | 14.755 us / 0.356 us / 0.290626 us^2 | 0.94x |
| `Tensor.divide` | tensor/scalar contiguous | `left.divide(scalar)` | shape `(192, 256)`, stride `(256, 1)` | 64 | 13.907 us / 0.486 us / 0.634551 us^2 | 15.004 us / 0.592 us / 9.241237 us^2 | 0.93x |
| `/` | reflected scalar contiguous | `scalar / left` | shape `(192, 256)`, stride `(256, 1)` | 64 | 13.036 us / 0.528 us / 2.063796 us^2 | 24.280 us / 2.455 us / 23.999106 us^2 | 0.54x |
| `/` | tensor/tensor broadcasting | `left / right` | shape `(192, 256)`, stride `(256, 1)` | 64 | 415.373 us / 27.119 us / 8909.867056 us^2 | 13.893 us / 0.985 us / 19.197884 us^2 | 29.90x |
| `Tensor.div` | tensor/tensor broadcasting | `left.div(right)` | shape `(192, 256)`, stride `(256, 1)` | 64 | 436.441 us / 43.805 us / 8804.018758 us^2 | 13.962 us / 1.540 us / 12.490073 us^2 | 31.26x |
| `Tensor.divide` | tensor/tensor broadcasting | `left.divide(right)` | shape `(192, 256)`, stride `(256, 1)` | 64 | 444.243 us / 34.094 us / 4099.377916 us^2 | 13.970 us / 0.995 us / 3.558544 us^2 | 31.80x |
| `/` | empty tensor/tensor | `left / right` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 0.275 us / 0.022 us / 0.003236 us^2 | 1.919 us / 0.205 us / 0.089451 us^2 | 0.14x |
| `Tensor.div` | empty tensor/tensor | `left.div(right)` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 0.443 us / 0.069 us / 0.008389 us^2 | 1.522 us / 0.118 us / 0.092293 us^2 | 0.29x |
| `Tensor.divide` | empty tensor/tensor | `left.divide(right)` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 0.365 us / 0.047 us / 0.014643 us^2 | 1.524 us / 0.120 us / 0.120501 us^2 | 0.24x |
| `/` | empty tensor/scalar | `left / scalar` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 0.326 us / 0.036 us / 0.003553 us^2 | 3.351 us / 0.191 us / 0.302510 us^2 | 0.10x |
| `Tensor.div` | empty tensor/scalar | `left.div(scalar)` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 1.127 us / 0.051 us / 0.007625 us^2 | 4.558 us / 0.201 us / 0.567869 us^2 | 0.25x |
| `Tensor.divide` | empty tensor/scalar | `left.divide(scalar)` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 1.684 us / 0.044 us / 0.004447 us^2 | 4.949 us / 0.144 us / 0.356867 us^2 | 0.34x |
| `/` | empty reflected scalar | `scalar / left` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 0.423 us / 0.066 us / 0.011479 us^2 | 4.584 us / 0.120 us / 0.355372 us^2 | 0.09x |
| `/` | tensor/tensor noncontiguous offset | `left / right` | shape `(128, 64)`, stride `(1, 128)` | 64 | 347.052 us / 16.684 us / 870.680167 us^2 | 30.388 us / 1.366 us / 6.642362 us^2 | 11.42x |
| `Tensor.div` | tensor/tensor noncontiguous offset | `left.div(right)` | shape `(128, 64)`, stride `(1, 128)` | 64 | 375.693 us / 25.069 us / 2729.752400 us^2 | 33.527 us / 1.524 us / 12.902604 us^2 | 11.21x |
| `Tensor.divide` | tensor/tensor noncontiguous offset | `left.divide(right)` | shape `(128, 64)`, stride `(1, 128)` | 64 | 380.421 us / 16.663 us / 1436.320320 us^2 | 32.720 us / 2.228 us / 46.804591 us^2 | 11.63x |
| `/` | tensor/scalar noncontiguous offset | `left / scalar` | shape `(128, 64)`, stride `(1, 128)` | 64 | 139.486 us / 8.798 us / 159.323260 us^2 | 24.032 us / 0.788 us / 1.191923 us^2 | 5.80x |
| `Tensor.div` | tensor/scalar noncontiguous offset | `left.div(scalar)` | shape `(128, 64)`, stride `(1, 128)` | 64 | 136.872 us / 15.656 us / 478.377332 us^2 | 37.433 us / 3.717 us / 49.020864 us^2 | 3.66x |
| `Tensor.divide` | tensor/scalar noncontiguous offset | `left.divide(scalar)` | shape `(128, 64)`, stride `(1, 128)` | 64 | 131.414 us / 5.214 us / 122.793823 us^2 | 24.459 us / 0.914 us / 8.374081 us^2 | 5.37x |
| `/` | reflected scalar noncontiguous offset | `scalar / left` | shape `(128, 64)`, stride `(1, 128)` | 64 | 157.824 us / 7.957 us / 221.191899 us^2 | 32.718 us / 3.910 us / 29.716096 us^2 | 4.82x |

Zero-credit unsupported cells:

| Cell | Reason |
| --- | --- |
| `Tensor.div` reflected scalar | Python reflected scalar division is spelled `scalar / tensor`; a float left operand has no `Tensor.div` method. |
| `Tensor.divide` reflected scalar | Python reflected scalar division is spelled `scalar / tensor`; a float left operand has no `Tensor.divide` method. |
| Top-level `torch.div` and `torch.divide` | These top-level aliases are outside the current supported surface for this repository. |
| `Tensor.div`/`Tensor.divide` with non-`None` `rounding_mode` | The supported surface accepts only true division with `rounding_mode=None`. |
| In-place `div_`/`divide_` and concrete `out` variants | The supported surface excludes in-place division and output-tensor division variants. |
| Active-autograd division operands | Division currently rejects active autograd recording until a division VJP exists. |
