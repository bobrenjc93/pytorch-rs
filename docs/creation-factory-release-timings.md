# Creation Factory Release Timings

Date: 2026-09-04

This document records the benchmark coverage for the public eager CPU
`torch.empty`, `torch.zeros`, and `torch.ones` factory surface. The checked-in
driver is `scripts/benchmark_creation_factories.py`.

## Workloads

The benchmark matrix covers every public factory added to this surface across
four shape categories:

| Shape case | Shape | Purpose |
| --- | --- | --- |
| `scalar` | `()` | rank-0 allocation overhead |
| `empty` | `(2, 0, 3)` | zero-element metadata and stride handling |
| `small` | `(32, 32)` | ordinary small CPU allocation |
| `large` | `(1024, 1024)` | large CPU allocation cost |

Every `factory x shape` cell is timed for both `torch_rs` and PyTorch 2.13 in
two reversed implementation orders. Each timed call materializes tensor
metadata (`shape`, `stride`, `storage_offset`, `numel`, `dtype`, `device`,
`layout`, contiguity, leaf state, gradient state, and pointer nonzero status)
to keep both implementations doing equivalent observable work. `zeros` and
`ones` also validate the filled value sum outside the timed loop. `empty`
values are intentionally not read or compared because PyTorch leaves them
unspecified.

## Current `torch.empty` Storage Behavior

The public contract for `torch.empty` is unspecified element values. The
current `torch_rs` implementation still stores initialized CPU `float32` values
because the Rust tensor/storage model exposes safe initialized slices to all
read paths and the crate denies unsafe code. Today that means `torch.empty`
uses a zero-initialized backing allocation as an implementation detail. The
factory benchmark includes that cost directly instead of crediting it as a
PyTorch-style uninitialized allocation.

## Command

Run from a release-built wheel install in the worktree virtual environment:

```bash
env -u CONDA_PREFIX TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --out target/factory-wheel
uv pip install --python "$PWD/.venv/bin/python" --force-reinstall --no-deps \
  target/factory-wheel/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
.venv/bin/python .github/scripts/verify_native_extension.py
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/benchmark_creation_factories.py \
  --cpu 24 \
  --warmups 15 \
  --samples 81 \
  --output target/creation-factory-release-timings.json
```

The driver sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` before importing either
backend, then checks PyTorch and `torch_rs` thread counts. It records CPU,
Python, PyTorch, Rust, package, git, command, affinity, and CUDA visibility
provenance in the JSON output. `CUDA_VISIBLE_DEVICES` defaults to the empty
string, and the report records `private_cuda_roundtrip_included=false`; private
CUDA roundtrip evidence belongs only to the CUDA compile benchmark until public
CUDA tensor operations exist.

## Review Run

The 2026-09-04 review run used the command above with a release wheel installed
into the worktree `.venv`, PyTorch `2.13.0+cu130`, Python 3.12.12, one backend
thread, `CUDA_VISIBLE_DEVICES=`, and CPU affinity pinned to CPU 24. The JSON
artifact was written to `target/creation-factory-release-timings.json`, which
keeps generated benchmark output inside the worktree without committing a
host-specific timing artifact.

Times are median microseconds per call across two reversed implementation-order
passes. `torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x
is parity. The overall geometric mean was `0.635x`; by API, `empty` was
`1.403x`, `zeros` was `0.489x`, and `ones` was `0.373x`. The large `empty`
cell shows the expected cost of zero-initializing storage while PyTorch returns
uninitialized memory.

| Workload | Repeats | `torch_rs` median | PyTorch median | `torch_rs` / PyTorch |
| --- | ---: | ---: | ---: | ---: |
| `empty_scalar` | 4096 | 2.261 us | 4.408 us | 0.51x |
| `empty_empty` | 4096 | 2.893 us | 5.042 us | 0.57x |
| `empty_small` | 512 | 2.959 us | 5.202 us | 0.57x |
| `empty_large` | 8 | 114.402 us | 4.946 us | 23.13x |
| `zeros_scalar` | 4096 | 2.230 us | 4.594 us | 0.49x |
| `zeros_empty` | 4096 | 2.861 us | 5.315 us | 0.54x |
| `zeros_small` | 512 | 2.957 us | 5.288 us | 0.56x |
| `zeros_large` | 8 | 121.399 us | 310.707 us | 0.39x |
| `ones_scalar` | 4096 | 2.246 us | 4.611 us | 0.49x |
| `ones_empty` | 4096 | 2.867 us | 5.310 us | 0.54x |
| `ones_small` | 512 | 3.058 us | 5.596 us | 0.55x |
| `ones_large` | 8 | 79.379 us | 588.455 us | 0.13x |
