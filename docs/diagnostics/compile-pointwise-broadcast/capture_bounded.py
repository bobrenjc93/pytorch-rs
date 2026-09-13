"""Capture a dispatched bounded-broadcast module; never score numerical parity."""
import ctypes
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch_rs as torch
from torch_rs import torch_rs as native

root = Path.cwd().resolve()
out = Path(sys.argv[1]).resolve()
assert out.is_relative_to(root) and not out.exists()
assert Path(__file__).resolve().is_relative_to(root)
assert Path(sys.executable).resolve().is_relative_to(root)
assert Path(native.__file__).resolve().is_relative_to(root)
out.mkdir(parents=True)
started = datetime.now(timezone.utc).isoformat()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def command(*args):
    return subprocess.check_output(args, text=True).strip()

def program(x, y):
    return (x.relu() - y.relu()).relu()

before = command('nvidia-smi', '--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used', '--format=csv')
compiled = torch.compile(program)
observations = []
dispatched = []
for rows, columns, offset in [(7, 11, 0.125), (7, 11, -0.375), (3, 17, 0.713)]:
    x = torch.tensor([i * 0.017 + offset for i in range(rows)]).reshape(rows, 1).to('cuda:0')
    y = torch.tensor([offset - i * 0.13 for i in range(columns)]).to('cuda:0')
    result = compiled(x, y)
    assert tuple(result.shape) == (rows, columns)
    assert result.data_ptr() not in (x.data_ptr(), y.data_ptr())
    # Match the actual guarded call, including its shapes and device. Cache
    # insertion order does not identify the last module dispatched on a hit.
    metadata = tuple(native._compile_trace_tensor_metadata(arg)[:5] for arg in (x, y))
    matching = [entry for key, entry in compiled._torch_rs_pointwise_cache.graphs.items()
                if key[2] == metadata and key[3] == (0, 1)]
    assert len(matching) == 1
    kernel = matching[0][1]
    dispatched.append(kernel)
    observations.append({
        'input_shapes': [list(x.shape), list(y.shape)],
        'shape': list(result.shape), 'values': result.cpu().tolist(),
        'source_kernel_sha256': sha(kernel.source.encode()),
        'ptx_sha256': sha(kernel.ptx.encode()),
    })
entries = list(compiled._torch_rs_pointwise_cache.graphs.values())
assert len(entries) == 2 and len({id(entry[1]) for entry in entries}) == 2
assert dispatched[0] is dispatched[1] and dispatched[1] is not dispatched[2]
# The files describe the last observed output's module, not the first cache entry.
kernel = dispatched[-1]
(out / 'kernel.cu').write_text(kernel.source)
(out / 'kernel.ptx.gz').write_bytes(gzip.compress(kernel.ptx.encode(), mtime=0))
paths = command('rg', '--files', 'src', 'python', '-g', '*.rs', '-g', '*.py', '-g', '*.ptx').splitlines()
paths += ['Cargo.toml', 'Cargo.lock', 'pyproject.toml', 'uv.lock', 'rust-toolchain.toml']
paths += [str(Path(__file__).resolve().relative_to(root))]
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
    'kind': 'non-scoring bounded-broadcast generated-code regression evidence',
    'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'worktree': str(root), 'capture_script': str(Path(__file__).resolve()),
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
    'captured_observation': len(observations) - 1,
    'observations': observations,
}
assert 'torch' not in sys.modules
(out / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
