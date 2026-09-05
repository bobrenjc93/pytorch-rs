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
to keep both implementations doing equivalent observable work. For `zeros` and
`ones`, each timed block also computes one final-output sum checksum after its
allocation repeats and before the timer stops; this keeps filled-value
materialization symmetric without turning every repeated allocation into a full
reduction. `empty` values are intentionally not read or compared because
PyTorch leaves them unspecified.

## Current `torch.empty` Storage Behavior

The public contract for `torch.empty` is unspecified element values. The
current `torch_rs` implementation still stores initialized CPU `float32` values
because the Rust tensor/storage model exposes safe initialized slices to all
read paths and the crate denies unsafe code. Today that means `torch.empty`
uses a zero-initialized backing allocation as an implementation detail. The
factory benchmark includes that cost directly instead of crediting it as a
PyTorch-style uninitialized allocation.

## Benchmark Integrity

The fixed matrix above is a public repeatability benchmark. It is useful as
local release evidence, but it is not sufficient by itself for merge scoring
when the same candidate branch is also adding benchmark coverage. Before
creation-factory timing numbers are used for a merge decision, run the
independent generated-shape validator with an evaluator-supplied held-out seed:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python \
  scripts/validate_creation_factory_benchmark.py \
  --seed <held-out-seed> \
  --output target/creation-factory-generated-validator.json
```

The validator excludes the fixed public matrix shapes, generates additional
CPU float32 creation shapes from the supplied seed, and reuses the same
symmetric warmup, sampling, metadata, and timed checksum checks. Benchmark
campaign changes should land separately from optimization claims; this report
documents the campaign shape and local evidence without treating fixed public
creation timings as standalone merge-gating proof.

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
is parity. The overall geometric mean was `0.740x`; by API, `empty` was
`1.402x`, `zeros` was `0.526x`, and `ones` was `0.550x`. The large `empty`
cell shows the expected cost of zero-initializing storage while PyTorch returns
uninitialized memory. The `zeros` and `ones` timings include one full-output
checksum per timed block, so they are not metadata-only allocation timings.

| Workload | Repeats | `torch_rs` median | PyTorch median | `torch_rs` / PyTorch |
| --- | ---: | ---: | ---: | ---: |
| `empty_scalar` | 4096 | 2.269 us | 4.474 us | 0.51x |
| `empty_empty` | 4096 | 2.883 us | 4.956 us | 0.58x |
| `empty_small` | 512 | 2.968 us | 5.035 us | 0.59x |
| `empty_large` | 8 | 112.982 us | 5.085 us | 22.22x |
| `zeros_scalar` | 4096 | 2.224 us | 4.670 us | 0.48x |
| `zeros_empty` | 4096 | 2.846 us | 5.266 us | 0.54x |
| `zeros_small` | 512 | 2.917 us | 5.175 us | 0.56x |
| `zeros_large` | 8 | 281.562 us | 534.503 us | 0.53x |
| `ones_scalar` | 4096 | 2.222 us | 4.646 us | 0.48x |
| `ones_empty` | 4096 | 2.865 us | 5.338 us | 0.54x |
| `ones_small` | 512 | 3.877 us | 5.606 us | 0.69x |
| `ones_large` | 8 | 318.195 us | 615.416 us | 0.52x |

The generated-validator audit run used seed `20260905`, four generated shapes
(`(31, 7, 0)`, `(1, 0, 5, 4)`, `(0, 1)`, and `(257,)`), one backend thread,
CPU affinity pinned to CPU 24, five warmups, and 17 samples. It wrote
`target/creation-factory-generated-validator.json`, covered 12 held-out
`factory x shape` cells, and reported an overall geometric mean of `0.547x`
with all metadata and timed filled-value checksum checks passing.
