# Repository agent guidance

## GPU-capable Burner host

The current Burner hill-climb host has real NVIDIA accelerators. The native
backend supports CPU tensors and bounded CUDA float32 storage, transfers,
operations and eager-backend graph capture; see docs/supported-surface.md.
This does not imply general accelerator, Inductor or training parity.

- 8 NVIDIA H100 GPUs, each reporting 97,871 MiB of memory and compute
  capability 9.0
- NVIDIA driver 580.82.07; `nvidia-smi` reports CUDA 13.0 compatibility
- The repository virtual environment contains PyTorch `2.13.0+cu130`, and
  `torch.cuda.is_available()` is true with all 8 devices visible
- The default `nvcc` is CUDA 12.6; multiple CUDA runtimes are installed, so
  record the compiler and runtime actually selected by a test or benchmark

When work touches devices, dispatch, CUDA kernels, transfers, distributed
execution, or accelerator performance, run real GPU tests on this host. Use
`CUDA_VISIBLE_DEVICES=0` for ordinary single-GPU work, and reserve only the
minimum number of devices needed for multi-GPU tests. Burner ideas that need an
accelerator should declare the shared `gpu` resource so concurrent agents do
not contend for it.

For paired runs of the public, non-scoring compile matmul diagnostic, an idle
nonzero GPU may be declared explicitly with `--gpu-uuid GPU-<full-uuid>` and
the identical full UUID in `CUDA_VISIBLE_DEVICES` before process startup.
Use the same selected physical UUID for both builds; all inputs remain logical
`cuda:0`. The diagnostic verifies the physical inventory against the native
driver/runtime and reference runtime and rejects mismatches or multiple visible
devices. Other evaluators retain their existing selection rules. Record UUID,
index, utilization and memory snapshots before/after each run. An idle snapshot
is not exclusive access; never interrupt another user's jobs. See
docs/compile-cuda-matmul.md for the diagnostic command and provenance contract.

Useful preflight:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Keep portable tests usable on machines without NVIDIA GPUs: detect accelerator
availability and skip hardware-only cases clearly instead of weakening their
assertions. Do not make the general GitHub Actions job depend on this particular
host. For GPU benchmarks, synchronize around timing, warm up both sides, use
equivalent work and fixed seeds, materialize outputs, and record the GPU,
driver, CUDA, PyTorch, and build configuration.
