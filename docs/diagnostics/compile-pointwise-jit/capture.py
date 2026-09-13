"""Reproduce non-scoring generated-kernel provenance in a new output directory."""
import ctypes
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import torch_rs as torch
from torch_rs import torch_rs as native

root = Path.cwd()
out = Path(sys.argv[1]).resolve()
assert out.is_relative_to(root) and not out.exists()
out.mkdir(parents=True)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def command(*args):
    return subprocess.check_output(args, text=True).strip()

def program(x, y):
    wave = torch.sin(x * 0.71359)
    square = wave * wave
    return (square - y.cos() + wave * 0.03173).relu()

before = command('nvidia-smi', '--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used', '--format=csv')
compiled = torch.compile(program)
observations = []
for count, offset in [(41, 0.125), (41, -0.375), (17, 0.713)]:
    x = torch.tensor([i * 0.017 + offset for i in range(count)]).to('cuda:0')
    y = torch.tensor([offset - i * 0.13 for i in range(count)]).to('cuda:0')
    result = compiled(x, y)
    assert result.data_ptr() not in (x.data_ptr(), y.data_ptr())
    observations.append({'shape': list(result.shape), 'values': result.cpu().tolist()})
entries = list(compiled._torch_rs_pointwise_cache.graphs.values())
kernel = entries[0][1]
assert all(entry[1] is kernel for entry in entries)
(out / 'kernel.cu').write_text(kernel.source)
(out / 'kernel.ptx.gz').write_bytes(gzip.compress(kernel.ptx.encode(), mtime=0))
paths = command('rg', '--files', 'src', 'python', '-g', '*.rs', '-g', '*.py', '-g', '*.ptx').splitlines()
paths += ['Cargo.toml', 'Cargo.lock', 'pyproject.toml', 'uv.lock', 'rust-toolchain.toml']
manifest = {p: sha(Path(p).read_bytes()) for p in sorted(paths)}
(out / 'source-manifest.json.gz').write_bytes(gzip.compress(json.dumps(manifest, sort_keys=True).encode(), mtime=0))
runtimes = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcudart.so' in line})
versions = []
for path in runtimes:
    lib = ctypes.CDLL(path)
    version = ctypes.c_int()
    assert lib.cudaRuntimeGetVersion(ctypes.byref(version)) == 0
    versions.append({'library': path, 'runtime_version': version.value, 'sha256': sha(Path(path).read_bytes())})
record = {
    'kind': 'non-scoring generated-code regression evidence',
    'base_commit': command('git', 'rev-parse', 'HEAD'),
    'dirty': bool(command('git', 'status', '--porcelain')),
    'source_sha256': sha(json.dumps(manifest, sort_keys=True).encode()),
    'python': sys.version, 'executable': sys.executable,
    'native_extension': native.__file__, 'native_sha256': sha(Path(native.__file__).read_bytes()),
    'pytorch_distribution': importlib.metadata.version('torch'),
    'nvrtc_version': kernel.nvrtc_version, 'options': kernel.options,
    'nvcc_path': shutil.which('nvcc'), 'nvcc_version': command('nvcc', '--version'),
    'runtime': versions, 'device': kernel.device,
    'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'gpu_before': before,
    'gpu_after': command('nvidia-smi', '--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used', '--format=csv'),
    'graph_entries': len(entries), 'code_modules': len({id(entry[1]) for entry in entries}),
    'source_kernel_sha256': sha(kernel.source.encode()), 'ptx_sha256': sha(kernel.ptx.encode()),
    'observations': observations,
}
assert 'torch' not in sys.modules
(out / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
