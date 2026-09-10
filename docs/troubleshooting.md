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

For historical `nn.factory_kwargs` order-only failures, see the
[cache-state diagnosis and regression check](factory-kwargs-cache-parity.md).
Changing the hash seed or warming caches alone does not fix the assertions.

## Stale Wheel Installs

If tests import an older `torch-rs` wheel, rebuild and reinstall from this
checkout instead of relying on the previous environment state:

```bash
unset PYTHONPATH
./scripts/test-python.sh
```

The script builds one release wheel from the current worktree, force-installs
it into `.venv`, verifies native extension provenance, and then runs the Python
suite. Clearing `PYTHONPATH` prevents source files from shadowing that wheel.

`maturin develop --release --locked` remains valid for editable development:
it resolves `torch_rs` under `python/torch_rs` so Python edits take effect
immediately. The wheel-only `.github/scripts/verify_native_extension.py` check
intentionally rejects that layout; it requires both `torch_rs` and the native
`torch_rs.torch_rs` extension to resolve inside `.venv`. Use the workflow above
when recovering a stale wheel or preparing an install for wheel verification.

## Exact-HEAD validation

`./scripts/test-python-exact-head.sh` exports the exact `HEAD` commit to a
temporary directory under `target/`, creates a Python 3.12 environment there,
and installs both locked development and reference dependency groups. Local
edits are excluded. It builds with locked Maturin and Cargo dependencies,
force-installs the release wheel, verifies native-extension provenance, and
checks for PyTorch 2.13.0 before running the full unittest suite.

The script clears inherited environment, import, optimization, and warning
settings, including ambient Cargo, PyO3, and Python runtime settings. It selects
and verifies the committed Rust channel, uses a fresh Cargo home, rejects
`.cargo/config` files above the archived checkout, and ignores external uv
configuration. Git and tar settings are cleared, and each extracted file is
checked against `HEAD`. It rejects a symlinked `target/` and uses its verified
physical path to keep artifacts inside the worktree. `CUDA_VISIBLE_DEVICES` is
preserved so hardware-aware tests use available GPUs and skip CUDA cases when
PyTorch reports none.

## Optional native CUDA runtime

CPU builds need neither the CUDA toolkit nor a CUDA runtime. The native backend
loads `libcudart` at first use; no CUDA compiler is used for public zero tensors
or CPU/CUDA copies.
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
CUDA tensors support rank-1 and rank-2 float32 zeros on explicit devices
(including empty dimensions), metadata views, and synchronous
`.cpu()` / `.to("cpu")` transfers. CPU float32 tensors without autograd can be
copied using `.to("cuda:N")` or the equivalent indexed `torch.device` and
`device=` forms. Scalars, empty tensors, dense and sparse views are supported;
preserve-format packing copies only the logical payload. The checked Rust API
is `Tensor::try_copy_cpu_to_cuda(Device::Cuda(index))`. Same-shape contiguous
CUDA float32 tensor addition uses `+`, `Tensor.add`, or `torch.add` with
default-equivalent alpha, including offset views, scalars and empties. The Rust
API is `Tensor::add`. It loads embedded PTX through `libcuda.so.1` (`nvcuda.dll`
on Windows) and the driver JIT; nvcc and NVRTC are not used. Results complete
on the legacy default stream before return. Contiguous float32 CUDA negation
(`-x`, `neg`, and `negative` methods/functions) also supports scalars, empties,
and contiguous offset views; see [validation](cuda-neg-validation.md).
Bounded eager native neg/add capture also supports scalar multiplication on
contiguous float32 CUDA tensors; see the [capture guide](compile-cuda-add.md)
for its guards and scope.
Contiguous float32 CUDA scalar multiplication also uses the native driver kernel;
see its [scope and validation](cuda-mul-scalar-validation.md).
Noncontiguous CUDA negation/multiplication, tensor-tensor multiplication, other CUDA math, CUDA autograd,
asynchronous transfers, dtype changes, unindexed CUDA targets, nondefault
streams, and general CUDA runtime management remain unsupported.

After a current-worktree release build, run the focused hardware tests:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest \
  tests.test_cuda_mul_scalar tests.test_cuda_neg tests.test_cuda_add tests.test_cuda_host_transfer tests.test_cuda_native_views tests.test_cuda_zero_roundtrip
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/diagnose_compile_cuda_neg_add.py \
  --case-set neg_add_v1 --output target/compile-neg-add-diagnostic.json
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest \
  tests.test_cuda_mul_scalar.CudaMulScalarDeviceTests tests.test_cuda_neg.CudaNegDeviceTests tests.test_cuda_add.CudaAddDeviceTests tests.test_cuda_host_transfer.CudaHostTransferDeviceGuardTests
# Standalone Rust needs TORCH_RS_CUDART set when libcudart is not on the loader path.
CUDA_VISIBLE_DEVICES=0 cargo test --locked --test cuda_add
CUDA_VISIBLE_DEVICES=0 cargo test --locked cuda
```

The historical `neg_add_v1` diagnostic now reports four obsolete matrix/vector
rejection expectations (exit 1 on GPU 0). Preserve that result; use the
[current trailing-vector diagnostic](compile-cuda-trailing-vector-validation.md)
for the new supported relation.


Hardware-only tests skip clearly when the reference runtime or required devices
are unavailable. The two-device test checks current-device restoration after
copies, addition, invalid ordinals/mixed devices, cached drops and backing-allocation releases. GPU transfer diagnostics
must warm both implementations equally, synchronize timing boundaries,
materialize outputs, and use matching shapes/layouts, threads and sampling.

`tensor(cuda_tensor)` copy construction also remains unsupported, including
empty inputs; use `.cpu()` or `.to("cpu")` for an explicit host transfer.

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

For public CUDA-add latency, sustained-throughput and allocation-cache
diagnostics, see [CUDA-add diagnostics](cuda-add-diagnostics.md).

The optional release screening script
[`scripts/benchmark_rank2_sum_cuda.py`](../scripts/benchmark_rank2_sum_cuda.py)
compares rank-2 sums and public CUDA allocation, transfer, and roundtrip costs.
Run it with one visible GPU after verifying the current-worktree release build;
its JSON output records environment, source hashes, dispersion, and output
metadata. Only matched native/PyTorch thread counts contribute to capped parity;
unmatched counts are labeled `scaling_diagnostic` and excluded from parity
aggregates. Dtype, device, shape, strides, storage offset, layout, gradient
metadata, and values are checked before and after timing. Invalid cells receive
zero credit, remain in the matched-cell denominator, and cause a nonzero exit.
Missing CUDA hardware also retains requested CUDA cells with zero credit.

The default 115-cell public matrix in
[`scripts/campaigns/rank2_sum_cuda_diagnostic.json`](../scripts/campaigns/rank2_sum_cuda_diagnostic.json)
produces `diagnostic_parity` only. `--quick` selects its declared strict subset;
both modes use identical per-cell inputs derived from the seed and cell ID.
Dirty-tree output is local diagnostic data, not release evidence.

For independent scoring, the reviewer or evaluation runner owns a campaign JSON
outside the candidate worktree and supplies a held-out seed:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark_rank2_sum_cuda.py \
  --campaign /path/to/reviewer-owned/campaign.json --seed "$HELD_OUT_SEED" \
  > target/reviewer-screen.json
```

Set `HELD_OUT_SEED` to a fresh reviewer-selected nonnegative integer distinct
from the public diagnostic seed. The runner reads this contract without modifying it;
it records its SHA-256, seed, full/quick membership, and selected IDs. Ownership
and seed secrecy belong to the independent runner. Candidate-local campaign
paths (including symlinks resolving inside the worktree) cannot produce the
`parity` scoring section; use `--diagnostic-campaign PATH` for local fixtures.
Burner's evaluator definitions and scoring remain separate and unchanged.

Campaign schema version 1 has `id`, `full` (explicit cell objects), and `quick`
(unique IDs forming a nonempty strict subset of `full`). Each cell has unique
`id`, `kind`, and positive `pytorch_threads`. CPU `sum` and
`backward_accumulate` cells specify rank-2 `shape`, `axis` (0 or 1), and `layout`
(`contiguous`, `transposed`, `offset`, or `selected`; backward uses contiguous
leaves). CUDA `zeros`, `to_cpu`, and `roundtrip` cells specify `elements`.
The public diagnostic file illustrates the contract; it is not a held-out
scoring corpus. Workload changes require independent campaign review.
