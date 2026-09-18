#!/usr/bin/env python3
"""Non-scoring, single-public-call CUDA diagnostic; captures never imply parity.

Run each framework/build in a fresh process, with fresh cache directories, and
repeat with framework order reversed. Use --compare on the resulting JSON pair.
Input construction and complete readback are outside every timed invocation.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WARMUPS, SAMPLES = 5, 17
RTOL, ATOL = 1e-5, 1e-6


def affine(x):
    return x * 1.125 + 0.25


def polynomial(x, y):
    return (x * y + x) * (y - 0.125)


def scalar(x, scale):
    return x * scale + 0.375


def trigonometric(x, y):
    return x.sin() * y.cos() + x


def nested(x, y):
    z = x * y + x
    return {'first': z, 'others': (y - x, z, x)}


def broadcast(x, y):
    return x * y + x


def views(x):
    a = x.transpose(0, -1)
    return (a, a, x.transpose(0, -1), a.transpose(0, -1), x)


def mixed(x, y):
    z = x * y + x
    a = y.transpose(0, -1)
    return (z, a, z, x, a)


# Independent developer examples, deliberately separate from the scoring corpus.
COMMON = [(), (0,), (257,), (13, 29), (4097,)]
PROGRAMS = [(affine, COMMON + [(262147,)]), (polynomial, COMMON),
            (scalar, COMMON), (trigonometric, COMMON), (nested, COMMON),
            (broadcast, [(13, 29), (1, 257), (0, 29)]),
            (views, [(), (0,), (257,), (13, 29)]), (mixed, COMMON)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'path outside worktree: {path}')
    return path


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {'argv': args, 'status': result.returncode,
            'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}


def checked(lib, name, *args):
    status = getattr(lib, name)(*args)
    if status:
        raise RuntimeError(f'{name}: CUDA status {status}')


def device_info(runtime):
    driver = ctypes.CDLL('libcuda.so.1')
    checked(driver, 'cuInit', 0)
    count, device, version = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    identity = (ctypes.c_ubyte * 16)()
    checked(driver, 'cuDeviceGetCount', ctypes.byref(count))
    checked(driver, 'cuDeviceGet', ctypes.byref(device), 0)
    checked(driver, 'cuDeviceGetUuid_v2', identity, device)
    checked(driver, 'cuDriverGetVersion', ctypes.byref(version))
    if count.value != 1 or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('requires only physical GPU0 via CUDA_VISIBLE_DEVICES=0')
    runtime_version = ctypes.c_int()
    checked(runtime, 'cudaRuntimeGetVersion', ctypes.byref(runtime_version))
    physical = command('nvidia-smi', '-i', '0', '--query-gpu=index,uuid,name,driver_version,compute_cap,utilization.gpu,memory.used', '--format=csv,noheader')
    selected_uuid = 'GPU-' + str(uuid.UUID(bytes=bytes(identity)))
    if selected_uuid not in physical['stdout']:
        raise RuntimeError('driver UUID disagrees with physical GPU0')
    return {'uuid': selected_uuid, 'driver': version.value,
            'runtime': runtime_version.value, 'snapshot': physical}


def inputs(module, fn, shape, step):
    rng = np.random.default_rng(281 + step)
    shapes = [shape] if fn in (affine, scalar, views) else [shape, shape]
    if fn is broadcast:
        shapes[1] = (shape[-1],)
    result = []
    for dimensions in shapes:
        count = int(np.prod(dimensions))
        # All tensors have an offset dense subspan; sentinels stay outside it.
        array = rng.uniform(-0.75, 0.75, size=count + 3).astype(np.float32)
        value = module.tensor(array.tolist(), dtype=module.float32).to('cuda:0')
        result.append(value[3:].reshape(dimensions))
    if fn is scalar:
        result.append(0.625 + (step % 2) * 0.125)
    return tuple(result)


def snapshot(value, args, arrays):
    seen = {}
    def visit(item):
        if isinstance(item, dict):
            return {'dict': [[key, visit(child)] for key, child in item.items()]}
        if isinstance(item, (tuple, list)):
            return {type(item).__name__: [visit(child) for child in item]}
        if not hasattr(item, 'data_ptr'):
            return {'literal': item}
        wrapper = seen.setdefault(id(item), len(seen))
        data = np.asarray(item.detach().cpu().tolist(), dtype=np.float32).reshape(tuple(item.shape))
        key = hashlib.sha256(data.tobytes()).hexdigest()
        arrays.setdefault(key, data.reshape(-1))
        return {'array': key, 'shape': list(item.shape), 'stride': list(item.stride()),
                'offset': item.storage_offset(), 'dtype': str(item.dtype),
                'device': str(item.device), 'grad': item.requires_grad,
                'wrapper': wrapper, 'input_aliases': [i for i, a in enumerate(args)
                    if hasattr(a, 'data_ptr') and data.size and item.data_ptr() == a.data_ptr()]}
    return visit(value)


def compare(reference_path, candidate_path):
    paths = [Path(p) for p in (reference_path, candidate_path)]
    reference, candidate = [json.loads(p.read_text()) for p in paths]
    # Absolute capture paths remain provenance, never a live replay dependency.
    # Captures and extracted bundles both keep each NPZ beside its JSON receipt.
    array_paths = [path.parent / Path(report['arrays']).name
                   for path, report in zip(paths, (reference, candidate))]
    errors, unsupported, compared = [], [], 0
    expected = [(fn.__name__, tuple(shape), mode) for fn, shapes in PROGRAMS
                for mode in ('reused', 'fresh') for shape in shapes]
    for report, array_path in zip((reference, candidate), array_paths):
        if (report.get('samples'), report.get('warmups'), report.get('calls_per_sample')) != (SAMPLES, WARMUPS, 1):
            raise ValueError('capture sampling contract mismatch')
        if not array_path.is_file():
            raise ValueError(f'missing array archive beside report: {array_path}')
        if sha(array_path) != report['arrays_sha256']:
            raise ValueError('array archive hash mismatch')
        observed = [(c['program'], tuple(c['shape']), c['mode']) for c in report['cells']]
        if observed != expected:
            raise ValueError('incomplete or reordered workload matrix')
        for cell in report['cells']:
            if cell['status'] not in ('ok', 'unsupported', 'error'):
                raise ValueError('unknown cell status')
            if cell['status'] == 'ok' and (len(cell['samples_ns']) != SAMPLES or
                    len(cell['outputs']) != SAMPLES + 1 or not cell.get('no_body_replay')):
                raise ValueError('incomplete successful cell')
            if any(type(n) is not int or n <= 0 for n in cell['samples_ns']):
                raise ValueError('invalid timing sample')
    with np.load(array_paths[0], allow_pickle=False) as ra, np.load(array_paths[1], allow_pickle=False) as ca:
        for archive in (ra, ca):
            for key in archive.files:
                if hashlib.sha256(archive[key].tobytes()).hexdigest() != key:
                    raise ValueError('array content hash mismatch')
        def check(a, b):
            if isinstance(a, dict) and 'array' in a:
                np.testing.assert_allclose(ca[b['array']], ra[a['array']], rtol=RTOL, atol=ATOL, equal_nan=True)
                zeros = (ca[b['array']] == 0) & (ra[a['array']] == 0)
                np.testing.assert_array_equal(np.signbit(ca[b['array']][zeros]), np.signbit(ra[a['array']][zeros]))
                assert {k: v for k, v in a.items() if k != 'array'} == {k: v for k, v in b.items() if k != 'array'}
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for key in a:
                    check(a[key], b[key])
            elif isinstance(a, list):
                assert len(a) == len(b)
                for x, y in zip(a, b):
                    check(x, y)
            else:
                assert a == b
        assert len(reference['cells']) == len(candidate['cells'])
        for a, b in zip(reference['cells'], candidate['cells']):
            identity = (a['program'], a['shape'], a['mode'])
            assert identity == (b['program'], b['shape'], b['mode'])
            if a['status'] != 'ok':
                errors.append([identity, 'reference failed', a.get('error')])
            elif b['status'] == 'unsupported':
                unsupported.append([identity, b.get('error')])
            elif b['status'] != 'ok':
                errors.append([identity, 'candidate failed', b.get('error')])
            else:
                try:
                    check(a['outputs'], b['outputs'])
                    compared += 1
                except (AssertionError, KeyError) as error:
                    errors.append([identity, str(error)])
    print(json.dumps({'compared': compared, 'candidate_rejections': unsupported,
                      'errors': errors, 'rtol': RTOL, 'atol': ATOL}, indent=2))
    return bool(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare', nargs=2, metavar=('REFERENCE_JSON', 'CANDIDATE_JSON'))
    parser.add_argument('--framework', choices=('torch', 'torch_rs'))
    parser.add_argument('--import-dir', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--source-commit', default=None, help='archived source identity, independently verify hashes')
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--label')
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not all((args.framework, args.output, args.label)):
        parser.error('capture requires --framework, --output and --label')
    output, source = inside(args.output), inside(args.source_root)
    inside(sys.prefix)
    if args.import_dir:
        sys.path.insert(0, str(inside(args.import_dir)))
    if args.framework == 'torch_rs':
        class NoTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'torch' or fullname.startswith('torch.'):
                    raise ImportError('native diagnostic forbids PyTorch imports')
        sys.meta_path.insert(0, NoTorch())
    module = __import__(args.framework)
    package = inside(module.__file__).parent
    runtime_path = inside(os.environ['TORCH_RS_CUDART'])
    runtime = ctypes.CDLL(str(runtime_path))
    sync = lambda: checked(runtime, 'cudaDeviceSynchronize')
    source_hashes = {str(p.relative_to(source)): sha(p) for folder in ('src', 'python')
                     for p in sorted((source / folder).rglob('*')) if p.is_file() and p.suffix in ('.rs', '.py', '.ptx')}
    native = None
    if args.framework == 'torch_rs':
        from torch_rs import _compile_trace
        native = inside(_compile_trace._native.__file__)
        for p in (source / 'python/torch_rs').rglob('*.py'):
            if sha(p) != sha(package / p.relative_to(source / 'python/torch_rs')):
                raise RuntimeError(f'installed Python source mismatch: {p}')
    report = {'schema': 'public-full-call-developer-v1', 'purpose': 'developer diagnostic, no score',
              'label': args.label, 'framework': args.framework, 'argv': sys.argv,
              'numpy_version': np.__version__, 'installed_reference_version': importlib.metadata.version('torch'),
              'timestamp_ns': time.time_ns(), 'source_root': str(source),
              'declared_source_commit': args.source_commit, 'worktree_head': command('git', 'rev-parse', 'HEAD'),
              'worktree_status': command('git', 'status', '--porcelain'), 'source_sha256': source_hashes,
              'script_sha256': sha(__file__), 'python': sys.version, 'executable': sys.executable,
              'prefix': sys.prefix, 'platform': platform.platform(), 'package': str(package),
              'native_extension': str(native) if native else None, 'native_sha256': sha(native) if native else None,
              'wheel': str(inside(args.wheel)) if args.wheel else None,
              'wheel_sha256': sha(args.wheel) if args.wheel else None,
              'framework_version': getattr(module, '__version__', None), 'runtime_library': str(runtime_path),
              'device_before': device_info(runtime), 'nvcc_inventory': command('nvcc', '--version'),
              'rustc': command('rustc', '--version'), 'environment': {k: os.environ.get(k) for k in
                  ('CUDA_VISIBLE_DEVICES', 'CUDA_CACHE_PATH', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'OMP_NUM_THREADS')},
              'warmups': WARMUPS, 'samples': SAMPLES, 'calls_per_sample': 1,
              'boundary': 'sync; start; public compiled call; sync; stop; full readback',
              'replay_check': 'one additional profiled warm call after each cell, outside timing', 'cells': []}
    arrays = {}
    report['native_compilers'] = {}
    for fn, shapes in PROGRAMS:
        for mode in ('reused', 'fresh'):
            if args.framework == 'torch':
                module._dynamo.reset()
            start = time.perf_counter_ns()
            compiled = module.compile(fn)
            creation_ns = time.perf_counter_ns() - start
            for shape in shapes:
                cell = {'program': fn.__name__, 'shape': shape, 'mode': mode, 'wrapper_creation_ns': creation_ns, 'samples_ns': [], 'outputs': []}
                try:
                    reused = inputs(module, fn, shape, 0)
                    sync()
                    start = time.perf_counter_ns()
                    result = compiled(*reused)
                    sync()
                    cell['first_shape_call_ns'] = time.perf_counter_ns() - start
                    cell['outputs'].append(snapshot(result, reused, arrays))
                    for step in range(WARMUPS + SAMPLES):
                        actual = inputs(module, fn, shape, step + 1) if mode == 'fresh' else reused
                        sync()
                        start = time.perf_counter_ns()
                        result = compiled(*actual)
                        sync()
                        elapsed = time.perf_counter_ns() - start
                        if step >= WARMUPS:
                            cell['samples_ns'].append(elapsed)
                            cell['outputs'].append(snapshot(result, actual, arrays))
                    observed_body = []
                    def profile(frame, event, arg):
                        if event == 'call' and frame.f_code is fn.__code__:
                            observed_body.append(True)
                    previous_profile = sys.getprofile()
                    try:
                        sys.setprofile(profile)
                        probe = compiled(*actual)
                        sync()
                    finally:
                        sys.setprofile(previous_profile)
                    if observed_body:
                        raise RuntimeError('original Python body replayed on warm verification call')
                    cell['no_body_replay'] = True
                    del probe
                    cell['status'] = 'ok'
                except Exception as error:
                    status = 'unsupported' if args.framework == 'torch_rs' and isinstance(error, NotImplementedError) else 'error'
                    cell.update(status=status, error=f'{type(error).__name__}: {error}')
                report['cells'].append(cell)
                if args.framework == 'torch_rs':
                    # NVRTC may unload after compilation. Retained module
                    # receipts expose the actual compiler outside every timer.
                    for executor in compiled._torch_rs_pointwise_cache.executors.values():
                        ptx_hash = hashlib.sha256(executor.ptx.encode()).hexdigest()
                        report['native_compilers'][ptx_hash] = {
                            'nvrtc_version': executor.nvrtc_version, 'options': executor.options,
                            'device': executor.device,
                            'source_sha256': hashlib.sha256(executor.source.encode()).hexdigest()}
                print(json.dumps({k: cell[k] for k in ('program', 'shape', 'mode', 'status')}), flush=True)
    report['device_after'] = device_info(runtime)
    report['loaded_cuda_libraries'] = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                           if any(name in line for name in ('libnvrtc', 'libcudart', 'libcuda.so'))})
    report['nvrtc_versions'] = {}
    for library in report['loaded_cuda_libraries']:
        if 'libnvrtc.so' in library:
            major, minor = ctypes.c_int(), ctypes.c_int()
            checked(ctypes.CDLL(library), 'nvrtcVersion', ctypes.byref(major), ctypes.byref(minor))
            report['nvrtc_versions'][library] = [major.value, minor.value]
    report['torch_imported'] = 'torch' in sys.modules
    output.parent.mkdir(parents=True, exist_ok=True)
    array_path = output.with_suffix('.npz')
    np.savez_compressed(array_path, **arrays)
    report.update(arrays=str(array_path), arrays_sha256=sha(array_path))
    output.write_text(json.dumps(report, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
