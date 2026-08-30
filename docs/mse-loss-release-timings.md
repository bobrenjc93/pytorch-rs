# `nn.functional.mse_loss(reduction="none")` Release Timings

Date: 2026-08-29

Measured revision: `280961a`
(`280961a1a470fe5fd2a5e9fa2fd3eaa91425dc16`)

Command shape: release `maturin develop --release --locked --offline` build of
the current worktree, followed by the focused native and reference MSE loss
tests, then an inline timing driver in the same worktree virtual environment.
The timing driver ran after imports and deterministic input construction, with
7 warmup blocks and 31 measured blocks. Each measured iteration called
`mse_loss(..., reduction="none")` and materialized the nonempty output with
`sum().item()`; empty outputs materialized their metadata with `numel() == 0`.
Broadcast warnings were suppressed symmetrically in measured regions. Before
timing each cell, the driver checked `torch_rs` and PyTorch outputs for matching
shape, stride, and bitwise float32 values.

Correctness gate run before timing:

```bash
PATH="/home/bobren/.cargo/bin:$PATH" \
RUSTUP_HOME="/home/bobren/.rustup" \
RUSTUP_NO_UPDATE_CHECK=1 \
CARGO_HOME="$PWD/target/cargo-home" \
UV_CACHE_DIR="$PWD/target/uv-cache" \
TMPDIR="$PWD/target/tmp" \
VIRTUAL_ENV="$PWD/.venv" \
PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_mse_loss \
  tests.test_nn_functional_mse_loss_reference
```

Result: 24 tests passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket, SMT2
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.12
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0 visible but CPU tensors only
- `torch_rs`: 0.1.0
- `torch_rs` native extension:
  `python/torch_rs/torch_rs.abi3.so`
- Rust: `rustc 1.92.0`, `cargo 1.92.0`
- Build profile: Cargo release, `[profile.release]` thin LTO and one codegen
  unit
- Device/dtype: CPU float32
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads() == 1` and
  `torch_rs.get_num_interop_threads() == 1`
- Input seed/procedure: deterministic `numpy.linspace` operands; no random
  timing inputs

Times are median microseconds per materialized call. MAD is median absolute
deviation in microseconds; variance is sample variance in microseconds squared.
`Current / PyTorch` is lower-is-better. The capped ratio applies a 10.0x cap
before geometric aggregation, per `BENCHMARKING.md`.

| Workload | Category | Shape | Output elements | Iterations/block | `torch_rs` median / MAD / var | PyTorch median / MAD / var | Current / PyTorch | Capped ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `same_contiguous_256x257` | same-shape contiguous | `(256, 257) vs (256, 257)` | 65,792 | 256 | 62.975629 / 0.170219 / 0.206184 | 18.837266 / 0.061539 / 0.024331 | 3.34x | 3.34x |
| `same_contiguous_1024x1024` | same-shape contiguous | `(1024, 1024) vs (1024, 1024)` | 1,048,576 | 16 | 1007.071750 / 3.543563 / 20.822988 | 209.635687 / 0.578938 / 1.357469 | 4.80x | 4.80x |
| `same_noncontiguous_transpose` | same-shape noncontiguous | `transpose((256, 257)) vs transpose((256, 257)) -> (257, 256)` | 65,792 | 128 | 98.316281 / 0.192172 / 0.154932 | 18.848961 / 0.047648 / 0.024604 | 5.22x | 5.22x |
| `same_offset_noncontiguous` | same-shape offset noncontiguous | `base[1].T/base[2].T from (3, 129, 67) -> (67, 129)` | 8,643 | 1,024 | 13.583512 / 0.040100 / 0.019763 | 6.754165 / 0.015697 / 0.006998 | 2.01x | 2.01x |
| `broadcast_scalar_target` | scalar broadcast | `(512, 513) vs scalar` | 262,656 | 64 | 246.110625 / 0.473062 / 2.212983 | 50.872375 / 1.271141 / 14.701014 | 4.84x | 4.84x |
| `broadcast_scalar_input` | scalar broadcast | `scalar vs (512, 513)` | 262,656 | 64 | 245.770891 / 0.749719 / 4.752803 | 52.685891 / 0.212516 / 2.815064 | 4.66x | 4.66x |
| `broadcast_vector_target` | vector broadcast | `(512, 257) vs (257,)` | 131,584 | 128 | 128.833734 / 0.349437 / 11.826274 | 32.651703 / 0.237469 / 0.461491 | 3.95x | 3.95x |
| `broadcast_column_target` | column broadcast | `(257, 512) vs (257, 1)` | 131,584 | 128 | 125.527883 / 0.472516 / 0.585688 | 31.243094 / 0.082852 / 0.939042 | 4.02x | 4.02x |
| `broadcast_noncontig_vector` | noncontiguous vector broadcast | `transpose((512, 257)) -> (257, 512) vs (512,)` | 131,584 | 8 | 1890.336375 / 11.357125 / 1118.019672 | 32.132375 / 0.175250 / 0.182115 | 58.83x | 10.00x |
| `empty_same_noncontiguous` | empty same-shape | `transpose((16, 0, 33)) -> (33, 0, 16) vs same` | 0 | 4,096 | 0.364971 / 0.001756 / 0.000246 | 3.183264 / 0.012558 / 0.002747 | 0.11x | 0.11x |
| `empty_scalar_broadcast` | empty scalar broadcast | `transpose((16, 0, 33)) -> (33, 0, 16) vs scalar` | 0 | 2,048 | 2.449861 / 0.008695 / 0.004396 | 6.587692 / 0.026706 / 0.040716 | 0.37x | 0.37x |

Geometric mean current/PyTorch ratio: `2.96x`.

Geometric mean capped current/PyTorch ratio: `2.52x` with a 10.0x cap.
