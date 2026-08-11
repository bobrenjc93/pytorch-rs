# Repository agent guidance

## GPU-capable Burner host

The current Burner hill-climb host has real NVIDIA accelerators. Do not assume
that testing must be CPU-only merely because the repository's present native
backend is CPU-only.

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
