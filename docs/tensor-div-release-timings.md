# Tensor True-Division Release Timings

Date: 2026-08-31

Code revision under test: `3c5a290843bf9b86e1a9d87de69d6658abbd2420`
plus worktree changes in `src/tensor.rs` for the rank-2 owned same-shape,
leading-singleton broadcast, and rank-2 owned unary/scalar materialization fast
paths.

Build and timing commands:

```bash
env PATH="$PWD/.venv/bin:/home/bobren/.cargo/bin:$PATH" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_NET_OFFLINE=true \
  TMPDIR="$PWD/target" \
  .venv/bin/maturin build --release --locked \
  --out "$PWD/target/perf-wheels"
UV_CACHE_DIR="$PWD/target/uv-cache" uv pip install \
  --python .venv/bin/python --force-reinstall --no-deps \
  target/perf-wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
env PATH="$PWD/.venv/bin:/home/bobren/.cargo/bin:$PATH" \
  VIRTUAL_ENV="$PWD/.venv" \
  CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  taskset -c 0 \
  .venv/bin/python
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
- Dependency installation: reused the existing locked `.venv`; release wheel
  force-install resolved 1 package in 2 ms, prepared it in 45 ms, uninstalled
  the prior wheel in 4 ms, and installed it in 15 ms
- Compile time: Cargo reported 27.83 s in the release profile

The transient timing driver wrote
`target/perf-results/tensor-div-release-timings.json`. It constructed inputs
outside the timed region, then ran 9 warmup blocks and 51 measured blocks for
each `torch_rs` and PyTorch cell. Each measured block repeated the eager
operation shown in the table and reports median microseconds per call. The
driver read first, middle, and last scalar values from every non-empty result
inside the timed loop, and read zero-size metadata for empty results. Before
timing each cell, it fully materialized the `torch_rs` and PyTorch outputs
through `tolist()`/NumPy and checked shape, stride, dtype, numel, output storage
offset, and exact float32 output bits. MAD is the median absolute deviation in
microseconds; variance is sample variance of the 51 per-call block timings in
`us^2`. Lower ratios are faster for `torch_rs`.

Input layouts:

| Workload family | Input shapes and layouts |
| --- | --- |
| Contiguous tensor/tensor and tensor/scalar | `left` and `right`: shape `(192, 256)`, stride `(256, 1)`, offset `0`; scalar `2.75` |
| Reflected scalar contiguous | `scalar / left` with `left`: shape `(192, 256)`, stride `(256, 1)`, offset `0`; scalar `2.75` |
| Broadcasting | `left`: shape `(192, 256)`, stride `(256, 1)`, offset `0`; `right`: shape `(1, 256)`, stride `(256, 1)`, offset `0` |
| Empty tensor/tensor and tensor/scalar | `left` and `right`: shape `(0, 4096)`, stride `(4096, 1)`, offset `0`; scalar `2.75` |
| Noncontiguous offset tensor/tensor and tensor/scalar | `left`: `tensor(values).transpose(1, 2)[1]`, shape `(128, 64)`, stride `(1, 128)`, offset `8192`; `right`: same shape, stride, and offset with distinct values; scalar `2.75` |
| Reflected scalar noncontiguous offset | `scalar / left` with `left`: shape `(128, 64)`, stride `(1, 128)`, offset `8192`; scalar `2.75` |

Supported-cell geometric aggregate: `torch_rs` / PyTorch median ratio `0.601x`
across the timed cells. This aggregate covers only supported cells with
correctness gates passed; unsupported cells below remain zero-credit
denominator entries rather than being converted into timings.

| API | Workload | Expression | Output layout | Repeats | `torch_rs` median / MAD / variance | PyTorch median / MAD / variance | Ratio |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `/` | tensor/tensor contiguous | `left / right` | shape `(192, 256)`, stride `(256, 1)` | 64 | 12.903 us / 0.057 us / 0.036111 us^2 | 14.405 us / 0.061 us / 0.064777 us^2 | 0.90x |
| `Tensor.div` | tensor/tensor contiguous | `left.div(right)` | shape `(192, 256)`, stride `(256, 1)` | 64 | 12.788 us / 0.036 us / 0.050484 us^2 | 14.696 us / 0.120 us / 0.058721 us^2 | 0.87x |
| `Tensor.divide` | tensor/tensor contiguous | `left.divide(right)` | shape `(192, 256)`, stride `(256, 1)` | 64 | 12.684 us / 0.056 us / 0.069766 us^2 | 14.845 us / 0.031 us / 0.032558 us^2 | 0.85x |
| `/` | tensor/scalar contiguous | `left / scalar` | shape `(192, 256)`, stride `(256, 1)` | 64 | 12.506 us / 0.044 us / 0.072303 us^2 | 15.150 us / 0.045 us / 0.126172 us^2 | 0.83x |
| `Tensor.div` | tensor/scalar contiguous | `left.div(scalar)` | shape `(192, 256)`, stride `(256, 1)` | 64 | 13.303 us / 0.057 us / 0.037876 us^2 | 14.925 us / 0.033 us / 0.049245 us^2 | 0.89x |
| `Tensor.divide` | tensor/scalar contiguous | `left.divide(scalar)` | shape `(192, 256)`, stride `(256, 1)` | 64 | 13.196 us / 0.043 us / 0.034986 us^2 | 15.174 us / 0.038 us / 0.039970 us^2 | 0.87x |
| `/` | reflected scalar contiguous | `scalar / left` | shape `(192, 256)`, stride `(256, 1)` | 64 | 12.600 us / 0.044 us / 0.032817 us^2 | 21.383 us / 0.254 us / 0.189688 us^2 | 0.59x |
| `/` | tensor/tensor broadcasting | `left / right` | shape `(192, 256)`, stride `(256, 1)` | 64 | 12.790 us / 0.044 us / 0.070912 us^2 | 13.525 us / 0.096 us / 0.202531 us^2 | 0.95x |
| `Tensor.div` | tensor/tensor broadcasting | `left.div(right)` | shape `(192, 256)`, stride `(256, 1)` | 64 | 12.995 us / 0.049 us / 0.040664 us^2 | 13.505 us / 0.045 us / 0.057916 us^2 | 0.96x |
| `Tensor.divide` | tensor/tensor broadcasting | `left.divide(right)` | shape `(192, 256)`, stride `(256, 1)` | 64 | 12.875 us / 0.052 us / 0.037304 us^2 | 13.497 us / 0.045 us / 0.039569 us^2 | 0.95x |
| `/` | empty tensor/tensor | `left / right` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 0.781 us / 0.008 us / 0.003249 us^2 | 1.683 us / 0.019 us / 0.011323 us^2 | 0.46x |
| `Tensor.div` | empty tensor/tensor | `left.div(right)` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 0.893 us / 0.004 us / 0.000167 us^2 | 1.668 us / 0.008 us / 0.000743 us^2 | 0.54x |
| `Tensor.divide` | empty tensor/tensor | `left.divide(right)` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 0.890 us / 0.004 us / 0.000185 us^2 | 1.705 us / 0.007 us / 0.000345 us^2 | 0.52x |
| `/` | empty tensor/scalar | `left / scalar` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 0.826 us / 0.005 us / 0.000219 us^2 | 3.324 us / 0.026 us / 0.053176 us^2 | 0.25x |
| `Tensor.div` | empty tensor/scalar | `left.div(scalar)` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 1.573 us / 0.015 us / 0.001636 us^2 | 3.263 us / 0.026 us / 0.038698 us^2 | 0.48x |
| `Tensor.divide` | empty tensor/scalar | `left.divide(scalar)` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 1.567 us / 0.020 us / 0.001663 us^2 | 3.345 us / 0.024 us / 0.059253 us^2 | 0.47x |
| `/` | empty reflected scalar | `scalar / left` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 0.894 us / 0.003 us / 0.000301 us^2 | 4.519 us / 0.022 us / 0.046863 us^2 | 0.20x |
| `/` | tensor/tensor noncontiguous offset | `left / right` | shape `(128, 64)`, stride `(1, 128)` | 64 | 8.654 us / 0.175 us / 0.071790 us^2 | 9.148 us / 0.080 us / 0.055783 us^2 | 0.95x |
| `Tensor.div` | tensor/tensor noncontiguous offset | `left.div(right)` | shape `(128, 64)`, stride `(1, 128)` | 64 | 8.744 us / 0.113 us / 0.064306 us^2 | 9.021 us / 0.081 us / 0.061562 us^2 | 0.97x |
| `Tensor.divide` | tensor/tensor noncontiguous offset | `left.divide(right)` | shape `(128, 64)`, stride `(1, 128)` | 64 | 8.728 us / 0.087 us / 0.055328 us^2 | 9.169 us / 0.110 us / 0.077122 us^2 | 0.95x |
| `/` | tensor/scalar noncontiguous offset | `left / scalar` | shape `(128, 64)`, stride `(1, 128)` | 64 | 3.755 us / 0.018 us / 0.088539 us^2 | 10.766 us / 0.079 us / 0.122986 us^2 | 0.35x |
| `Tensor.div` | tensor/scalar noncontiguous offset | `left.div(scalar)` | shape `(128, 64)`, stride `(1, 128)` | 64 | 4.504 us / 0.015 us / 0.041502 us^2 | 10.727 us / 0.080 us / 0.071290 us^2 | 0.42x |
| `Tensor.divide` | tensor/scalar noncontiguous offset | `left.divide(scalar)` | shape `(128, 64)`, stride `(1, 128)` | 64 | 4.493 us / 0.019 us / 0.050365 us^2 | 10.857 us / 0.076 us / 0.074448 us^2 | 0.41x |
| `/` | reflected scalar noncontiguous offset | `scalar / left` | shape `(128, 64)`, stride `(1, 128)` | 64 | 3.883 us / 0.026 us / 0.067193 us^2 | 12.975 us / 0.039 us / 0.047141 us^2 | 0.30x |

Zero-credit unsupported cells:

| Cell | Reason |
| --- | --- |
| `Tensor.div` reflected scalar | Python reflected scalar division is spelled `scalar / tensor`; a float left operand has no `Tensor.div` method. |
| `Tensor.divide` reflected scalar | Python reflected scalar division is spelled `scalar / tensor`; a float left operand has no `Tensor.divide` method. |
| Top-level `torch.div` and `torch.divide` | These top-level aliases are outside the current supported surface for this repository. |
| `Tensor.div`/`Tensor.divide` with non-`None` `rounding_mode` | The supported surface accepts only true division with `rounding_mode=None`. |
| In-place `div_`/`divide_` and concrete `out` variants | The supported surface excludes in-place division and output-tensor division variants. |
| Active-autograd division operands | Division currently rejects active autograd recording until a division VJP exists. |
