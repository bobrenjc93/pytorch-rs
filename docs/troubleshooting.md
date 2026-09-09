# Setup Troubleshooting

Use these fixes from the repository root after following the locked setup in
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Ambient Python Missing Pytest

If `pytest` or `python -m pytest` fails before it reaches repository code, the
command is using an ambient interpreter. The checked-in smoke tests use
`unittest`; run them through the repository environment instead:

```bash
. .venv/bin/activate
python -m unittest tests.test_readme_quickstart
```

For the full Python suite, prefer `./scripts/test-python.sh`; it rebuilds and
checks the installed native extension before running tests.

## `PYTHONPATH=python` Finds Python Files But Not the Native Extension

`PYTHONPATH=python` only exposes the pure-Python package files. It does not build
or install `torch_rs.torch_rs`, so imports can fail while loading
`python/torch_rs/__init__.py`.

Build the extension into the active repository environment, then import without
`PYTHONPATH`:

```bash
unset PYTHONPATH
VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
.venv/bin/python -c 'import torch_rs; print(torch_rs.__file__)'
```

## Missing Reference PyTorch 2.13

Reference and differential checks expect PyTorch 2.13.0. If `import torch`
fails, or the preflight reports another version, install the locked reference
dependency group:

```bash
uv sync --locked --no-install-project --group reference
.venv/bin/python -c 'import torch; print(torch.__version__)'
```

The version printed before any local suffix should be `2.13.0`.

## Stale Wheel Installs

If tests import an older `torch-rs` wheel, rebuild and reinstall from this
checkout instead of relying on the previous environment state:

```bash
VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
.venv/bin/python .github/scripts/verify_native_extension.py
```

`./scripts/test-python.sh` performs the stricter path: it builds one release
wheel from the current worktree, force-installs it into `.venv`, verifies native
extension provenance, and then runs the suite.

## Optional native CUDA runtime

CPU builds need neither the CUDA toolkit nor a CUDA runtime. The native backend
loads `libcudart` at first use; no CUDA compiler is used for public zero tensors.
Python discovers libraries from optional `nvidia.cuda_runtime` / `nvidia.cu13`
wheel packages without importing PyTorch. The supported reference environment
(`uv sync --locked --no-install-project --group dev --group reference`) includes
a runtime wheel. System CUDA 12/13 library names are fallback candidates.

For standalone Rust, or to select a particular installed runtime, set
`TORCH_RS_CUDART` to its absolute shared-library path before the first probe or
allocation. This override is authoritative: a bad path produces a runtime error
on allocation and availability probes return false. A runtime or driver error
never silently creates CPU storage. `cuda::configure_candidates` is also
available to Rust embedders before the first backend call.

Use `CUDA_VISIBLE_DEVICES=0` for single-GPU checks. Record the loaded libcudart
path (on Linux, `/proc/self/maps`), PyTorch version, driver and GPU model;
`nvcc --version` describes the compiler and need not match the runtime. Public
CUDA tensors support rank-1 float32 zeros and metadata views with synchronous
`.cpu()` / `.to("cpu")` transfers. CPU-to-CUDA copies, CUDA math/autograd,
nondefault streams, and general CUDA runtime management remain unsupported.

After building the current checkout with `./scripts/test-python.sh`, run this
smoke check through the repository environment (PyTorch is not required):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python - <<'PY'
import torch_rs as torch
assert torch.cuda.is_available(), "install a CUDA runtime and NVIDIA driver"
x = torch.zeros((12,), device="cuda:0").reshape(3, 4).t()
assert x.is_cuda
assert x.cpu().tolist() == [[0.0, 0.0, 0.0]] * 4
print(x.device, x.shape, x.stride(), "roundtrip passed")
PY
```

The optional release screening script
[`scripts/benchmark_rank2_sum_cuda.py`](../scripts/benchmark_rank2_sum_cuda.py)
compares rank-2 sums and public CUDA allocation, transfer, and roundtrip costs.
Run it with one visible GPU after verifying the current-worktree release build;
its JSON output records environment, source hashes, dispersion, and capped
per-cell parity. Dirty-tree output is local diagnostic data, not release evidence.
