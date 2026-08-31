# Tensor Sign Release Timings

Date: 2026-08-31

Code revision under test: `3c5a290843bf9b86e1a9d87de69d6658abbd2420`
plus worktree changes in `src/tensor.rs` for rank-2 owned unary/scalar
materialization. The top-level `torch.sign` CPU surface is included in the
timed cells.

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
`target/perf-results/sign-release-timings.json`. It constructed inputs outside
the timed region, then ran 9 warmup blocks and 51 measured blocks for each
`torch_rs` and PyTorch cell. Each measured block repeated the eager operation
shown in the table and reports median microseconds per call. The driver read
representative scalar values from every non-empty result inside the timed loop,
and read zero-size metadata for empty results. Before timing each cell, it fully
materialized the `torch_rs` and PyTorch outputs through `tolist()`/NumPy and
checked shape, stride, dtype, numel, output storage offset, and exact float32
output bits. MAD is the median absolute deviation in microseconds; variance is
sample variance of the 51 per-call block timings in `us^2`. Lower ratios are
faster for `torch_rs`.

Input layouts:

| Workload family | Input shape and layout |
| --- | --- |
| Scalar | shape `()`, stride `()`, offset `0` |
| Contiguous | shape `(192, 256)`, stride `(256, 1)`, offset `0` |
| Empty | shape `(0, 4096)`, stride `(4096, 1)`, offset `0` |
| Noncontiguous offset | `tensor(values).transpose(1, 2)[1]`, shape `(128, 64)`, stride `(1, 128)`, offset `8192` |

Supported-cell geometric aggregate: `torch_rs` / PyTorch median ratio `0.508x`
across the timed cells. This aggregate covers only supported cells with
correctness gates passed; unsupported cells below remain zero-credit
denominator entries rather than being converted into timings. Autograd sign
forward/backward behavior is covered by tests, but this timing snapshot measures
inference eager dispatch and kernel cost only.

| API | Workload | Expression | Output layout | Repeats | `torch_rs` median / MAD / variance | PyTorch median / MAD / variance | Ratio |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `Tensor.sign` | scalar | `input.sign()` | shape `()`, stride `()` | 4096 | 0.578 us / 0.004 us / 0.000035 us^2 | 1.591 us / 0.048 us / 0.095448 us^2 | 0.36x |
| `torch.sign` | scalar | `torch.sign(input)` | shape `()`, stride `()` | 4096 | 0.916 us / 0.049 us / 0.018894 us^2 | 1.598 us / 0.008 us / 0.000213 us^2 | 0.57x |
| `Tensor.sign` | contiguous | `input.sign()` | shape `(192, 256)`, stride `(256, 1)` | 64 | 9.465 us / 0.169 us / 0.146681 us^2 | 13.437 us / 0.074 us / 0.238661 us^2 | 0.70x |
| `torch.sign` | contiguous | `torch.sign(input)` | shape `(192, 256)`, stride `(256, 1)` | 64 | 11.972 us / 2.117 us / 20.336316 us^2 | 18.453 us / 4.082 us / 14.147033 us^2 | 0.65x |
| `Tensor.sign` | empty | `input.sign()` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 0.806 us / 0.009 us / 0.022754 us^2 | 1.495 us / 0.019 us / 0.000827 us^2 | 0.54x |
| `torch.sign` | empty | `torch.sign(input)` | shape `(0, 4096)`, stride `(4096, 1)` | 1024 | 0.871 us / 0.004 us / 0.000157 us^2 | 1.553 us / 0.015 us / 0.000441 us^2 | 0.56x |
| `Tensor.sign` | noncontiguous offset | `input.sign()` | shape `(128, 64)`, stride `(1, 128)` | 64 | 3.425 us / 0.014 us / 0.046312 us^2 | 8.749 us / 0.157 us / 0.053906 us^2 | 0.39x |
| `torch.sign` | noncontiguous offset | `torch.sign(input)` | shape `(128, 64)`, stride `(1, 128)` | 64 | 3.463 us / 0.009 us / 0.035203 us^2 | 8.787 us / 0.161 us / 0.067400 us^2 | 0.39x |

Zero-credit unsupported cells:

| Cell | Reason |
| --- | --- |
| Concrete `out` tensors | The current top-level surface accepts `out=None` but rejects writing into a provided output tensor. |
| In-place `sign_` | In-place mutation is outside the current supported surface. |
| Non-CPU or non-`float32` tensors | The current native tensor runtime is CPU `float32`; other devices and dtypes remain outside this implementation. |
