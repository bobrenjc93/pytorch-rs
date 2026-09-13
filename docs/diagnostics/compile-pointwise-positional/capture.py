"""Capture the module actually dispatched by a non-corpus positional program.

Run with the freshly installed local wheel, from the checkout root:
  python docs/diagnostics/compile-pointwise-positional/capture.py OUTPUT WHEEL
This is development evidence, not a scoring or progress-artifact generator.
"""
import ctypes
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import torch_rs as torch
from torch_rs import _compile_pointwise as frontend
from torch_rs import torch_rs as native

root = Path.cwd().resolve()
out, wheel = (Path(value).resolve() for value in sys.argv[1:])
assert out.is_relative_to(root) and wheel.is_relative_to(root) and not out.exists()
assert Path(sys.executable).resolve().is_relative_to(root)
assert Path(native.__file__).resolve().is_relative_to(root)
out.mkdir(parents=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def inventory():
    return command('nvidia-smi', '--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used', '--format=csv')


def pointwise(scale, x, enabled, y):
    return (x * scale + y * enabled).relu()


started, before = datetime.now(timezone.utc).isoformat(), inventory()
compiled = torch.compile(pointwise)
observations, dispatched = [], []
for scale, enabled, shape, offset in ((0.375, False, (2, 17), 0),
                                       (-0.75, False, (2, 17), 1),
                                       (0.375, True, (2, 17), 0),
                                       (0.375, False, (2, 17), 1)):
    x = torch.tensor([i * 0.03125 - 0.5 for i in range(34+offset)]).to('cuda:0')[offset:].reshape(shape)
    y = torch.tensor([0.25-i * 0.015625 for i in range(34)]).reshape(shape).to('cuda:0')
    args = (scale, x, enabled, y)
    result = compiled(*args)
    assert result.shape == shape and result.data_ptr() not in (x.data_ptr(), y.data_ptr())
    # Resolve the same complete guard as this call. In particular the last
    # call returns to the initial scalar value but dispatches its runtime graph,
    # not the first (static) module or the last inserted Boolean specialization.
    tensors, parameters = frontend.bind_arguments(args)
    program = frontend.analyze(pointwise, len(args))
    keys, values = frontend.resolve(pointwise, program, parameters)
    cache = compiled._torch_rs_pointwise_cache.graphs
    keys, _, _ = frontend.runtime_bindings(program, keys, values, cache, promote=False)
    metadata = tuple(native._compile_trace_tensor_metadata(t)[:5] for t in tensors)
    kernel = cache[(program.code, keys, metadata, (0, 1))][1]
    dispatched.append(kernel)
    observations.append({'scale': scale, 'enabled': enabled, 'shape': shape, 'offset': offset,
                         'output': result.cpu().tolist(), 'source_sha256': sha(kernel.source.encode()),
                         'ptx_sha256': sha(kernel.ptx.encode())})
assert len(cache) == 3
assert dispatched[-1] is dispatched[1] and dispatched[-1] is not dispatched[0]
assert dispatched[-1] is not dispatched[2]
assert 'float s0' in kernel.source
(out/'kernel.cu').write_text(kernel.source)
(out/'kernel.ptx.gz').write_bytes(gzip.compress(kernel.ptx.encode(), mtime=0))
paths = command('rg', '--files', 'src', 'python', '-g', '*.rs', '-g', '*.py', '-g', '*.ptx').splitlines()
paths += ['Cargo.toml', 'Cargo.lock', 'pyproject.toml', 'uv.lock', 'rust-toolchain.toml',
          'tests/test_compile_pointwise_runtime_scalars.py', 'tests/test_compile_pointwise_scalar_admission.py',
          str(Path(__file__).resolve().relative_to(root))]
manifest = {p: sha(Path(p).read_bytes()) for p in sorted(paths)}
(out/'source-manifest.json.gz').write_bytes(gzip.compress(json.dumps(manifest, sort_keys=True).encode(), mtime=0))
with zipfile.ZipFile(wheel) as archive:
    assert archive.read('torch_rs/_compile_pointwise.py') == (root/'python/torch_rs/_compile_pointwise.py').read_bytes()
    assert archive.read('torch_rs/torch_rs.abi3.so') == Path(native.__file__).read_bytes()
runtimes = []
for path in sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcudart.so' in line}):
    library, version = ctypes.CDLL(path), ctypes.c_int()
    assert library.cudaRuntimeGetVersion(ctypes.byref(version)) == 0
    runtimes.append({'path': path, 'version': version.value, 'sha256': sha(Path(path).read_bytes())})
record = {'kind': 'non-scoring positional binding dispatched-module evidence',
          'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
          'commit': command('git', 'rev-parse', 'HEAD'), 'status': command('git', 'status', '--porcelain'),
          'source_manifest_sha256': sha(json.dumps(manifest, sort_keys=True).encode()),
          'wheel': str(wheel), 'wheel_sha256': sha(wheel.read_bytes()),
          'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
          'executable_sha256': sha(Path(sys.executable).read_bytes()),
          'native_extension': native.__file__, 'native_sha256': sha(Path(native.__file__).read_bytes()),
          'pytorch_distribution': importlib.metadata.version('torch'),
          'rustc': command('rustc', '-Vv'), 'nvcc': command('nvcc', '--version'),
          'nvrtc_version': kernel.nvrtc_version, 'options': kernel.options, 'runtime': runtimes,
          'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device': kernel.device,
          'gpu_before': before, 'gpu_after': inventory(), 'observations': observations,
          'captured_observation': len(observations)-1}
assert 'torch' not in sys.modules
(out/'provenance.json').write_text(json.dumps(record, indent=2)+'\n')
