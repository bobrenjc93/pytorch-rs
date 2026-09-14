"""Non-scoring, source-bound helper diagnostic in isolated native/reference legs.

Run from the checkout root with a freshly installed wheel, reserved GPUs, and
worktree-local writable caches. Every output/failure is retained; no evaluator
or corpus definition is changed. See README.md for reproduction and scope.
"""
import argparse
import ctypes
import gzip
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_bytes(gzip.compress(json.dumps(value, sort_keys=True, allow_nan=False).encode(), mtime=0))


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def inventory():
    return command('nvidia-smi', '--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used', '--format=csv')


class Synchronize:
    def __init__(self):
        self.path, = Path(sys.prefix).glob('lib/python*/site-packages/nvidia/cu13/lib/libcudart.so.13')
        self.runtime = ctypes.CDLL(str(self.path))
        version = ctypes.c_int()
        assert self.runtime.cudaRuntimeGetVersion(ctypes.byref(version)) == 0
        assert self.runtime.cudaSetDevice(0) == 0
        self.metadata = {'path': str(self.path), 'sha256': sha(self.path), 'version': version.value,
                         'api': 'cudaDeviceSynchronize', 'logical_device': 0}

    def __call__(self):
        assert self.runtime.cudaDeviceSynchronize() == 0


def encode(tensor):
    return {'shape': list(tensor.shape), 'stride': list(tensor.stride()), 'device': str(tensor.device),
            'dtype': str(tensor.dtype), 'requires_grad': tensor.requires_grad,
            'values': [float(v).hex() for v in tensor.cpu().reshape(-1).tolist()]}


def make(fw, name):
    if name == 'frozen_python_helper':
        sys.path.insert(0, str(ROOT / 'scripts'))
        from torch_compile_default_corpus import CASES, build
        case = next(c for c in CASES if c.name == 'python_helper')
        item = build(case, fw, 'cuda')
        return item.function, (item.function.__closure__[0].cell_contents,), item.inputs
    sources = {
        'multiple_composed': ('def h(a,b):\n return (a-b).relu()\ndef k(a):\n return a.sin()',
                              'k(h(x,y)) + h(y,x)'),
        'identity_runtime': ('def h(a):\n return a', 'x * h(scale) - y'),
        'literal_return': ('def h(a):\n return 0.375', '(x-y) * h(x)'),
    }
    helpers, expression = sources[name]
    scope = {}
    exec(helpers + '\ndef f(x,y,scale):\n return ' + expression, scope)
    fn = scope['f']
    bodies = tuple(v for k, v in scope.items() if k in ('h', 'k'))
    def inputs(variant, sample):
        import numpy as np
        rng = np.random.default_rng(1729 + 1009*variant + 1000003*sample)
        shape = (7, 11) if variant == 0 else (3, 17)
        left, right = [rng.normal(0, 0.2, shape).astype('float32').tolist() for _ in range(2)]
        x = fw.tensor(left, dtype=fw.float32).to('cuda:0')
        y = x if sample == 1 else fw.tensor(right, dtype=fw.float32).to('cuda:0')
        return x, y, (0.375 if sample % 2 == 0 else 0.5)
    return fn, bodies, inputs


def leg(args):
    out = args.output
    report = {'role': args.role, 'gpu_before': inventory(), 'visible': os.environ['CUDA_VISIBLE_DEVICES'],
              'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
              'executable_sha256': sha(sys.executable), 'wheel': str(args.wheel), 'wheel_sha256': sha(args.wheel),
              'commit': command('git', 'rev-parse', 'HEAD'), 'cases': [], 'passed': False}
    try:
        if args.role == 'native':
            class BlockTorch(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, *unused):
                    if fullname == 'torch' or fullname.startswith('torch.'):
                        raise AssertionError('reference import in candidate')
            sys.meta_path.insert(0, BlockTorch())
            import torch_rs as fw
            from torch_rs import torch_rs as bridge
            installed = Path(fw.__file__).parent
            with zipfile.ZipFile(args.wheel) as archive:
                for path in (ROOT / 'python/torch_rs').rglob('*.py'):
                    relative = path.relative_to(ROOT / 'python')
                    assert archive.read(str(relative)) == path.read_bytes() == (installed.parent / relative).read_bytes()
                assert archive.read('torch_rs/torch_rs.abi3.so') == Path(bridge.__file__).read_bytes()
            report['extension'] = {'path': bridge.__file__, 'sha256': sha(bridge.__file__)}
        else:
            import torch as fw
            fw.set_num_threads(1)
            report['reference'] = {'version': fw.__version__, 'cuda': fw.version.cuda,
                                   'path': fw.__file__, 'config': fw.__config__.show()}
        assert Path(fw.__file__).resolve().is_relative_to(ROOT)
        sync = Synchronize()
        report['synchronization'] = sync.metadata
        for name in ('multiple_composed', 'identity_runtime', 'literal_return', 'frozen_python_helper'):
            row = {'name': name, 'histories': [], 'passed': False}
            report['cases'].append(row)
            try:
                fn, helpers, inputs = make(fw, name)
                compiled = fw.compile(fn)
                codes = (fn.__code__, *(helper.__code__ for helper in helpers))
                def forbid(frame, event, arg):
                    if event == 'call' and any(frame.f_code is code for code in codes):
                        raise AssertionError('original body executed')
                for variant, sample in ((0, 0), (1, 0), (1, 1), (0, 2)):
                    values = inputs(variant, sample)
                    tensors = tuple(v for v in values if type(v) is fw.Tensor)
                    before = [encode(t) for t in tensors]
                    history = {'variant': variant, 'sample': sample, 'inputs': before, 'samples_ns': []}
                    row['histories'].append(history)
                    if args.role == 'native':
                        sys.setprofile(forbid)
                    try:
                        sync()
                        start = time.perf_counter_ns()
                        result = compiled(*values)
                        sync()
                        history['cold_ns'] = time.perf_counter_ns() - start
                        history['output'] = encode(result)
                        # Body policing covers cold admission; steady timings on both
                        # sides run without profiling instrumentation.
                        sys.setprofile(None)
                        for _ in range(5):
                            result = compiled(*values)
                            sync()
                        for _ in range(17):
                            sync()
                            start = time.perf_counter_ns()
                            result = compiled(*values)
                            sync()
                            history['samples_ns'].append(time.perf_counter_ns() - start)
                        if args.role == 'native':
                            sys.setprofile(forbid)
                            result = compiled(*values)
                            sync()
                        assert encode(result) == history['output']
                        assert [encode(t) for t in tensors] == before
                        assert all(result.data_ptr() != t.data_ptr() for t in tensors)
                        history['median_ns'] = statistics.median(history['samples_ns'])
                        if args.role == 'native':
                            kernel = next(reversed(compiled._torch_rs_pointwise_cache.executors.values()))
                            assert 'torch_rs_pointwise' in kernel.ptx
                            assert kernel.ptx.count('.visible .entry') == 1
                            tag = f'{name}-{variant}-{sample}'
                            (out / f'{tag}.cu').write_text(kernel.source)
                            (out / f'{tag}.ptx.gz').write_bytes(gzip.compress(kernel.ptx.encode(), mtime=0))
                            history['kernel'] = {'source_sha256': sha(out / f'{tag}.cu'),
                                                 'ptx_sha256': hashlib.sha256(kernel.ptx.encode()).hexdigest(),
                                                 'nvrtc': kernel.nvrtc_version, 'options': kernel.options,
                                                 'device': kernel.device}
                    finally:
                        sys.setprofile(None)
                row['passed'] = True
            except Exception:
                row['failure'] = traceback.format_exc()
            write(out / 'report.json.gz', report)
        report['passed'] = all(row['passed'] for row in report['cases'])
    except Exception:
        report['failure'] = traceback.format_exc()
    finally:
        report['gpu_after'] = inventory()
        report['torch_imported'] = 'torch' in sys.modules
        report['loaded_cuda_runtimes'] = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                                if 'libcudart.so' in line})
        write(out / 'report.json.gz', report)
    return 0 if report['passed'] else 1


def compare(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if key != 'values':
            assert left[key] == right[key], (key, left[key], right[key])
    assert len(left['values']) == len(right['values'])
    for a, b in zip(left['values'], right['values']):
        a, b = float.fromhex(a), float.fromhex(b)
        if math.isnan(b):
            assert math.isnan(a)
        elif math.isinf(b):
            assert a == b
        else:
            assert math.isfinite(a) and abs(a-b) <= 1e-6 + 1e-5*abs(b), (a, b)
            if a == b == 0:
                assert math.copysign(1, a) == math.copysign(1, b)


def run(args):
    manifest_paths = command('git', 'ls-files', 'src', 'python', 'Cargo.toml', 'Cargo.lock',
                             'pyproject.toml', 'uv.lock', 'rust-toolchain.toml',
                             'scripts/torch_compile_default_corpus.py').splitlines()
    manifest = {p: sha(ROOT / p) for p in manifest_paths}
    manifest[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    manifest['tests/test_compile_pointwise_helpers.py'] = sha(ROOT / 'tests/test_compile_pointwise_helpers.py')
    write(args.output / 'source-manifest.json.gz', manifest)
    report = {'kind': 'non-scoring helper diagnostic', 'rtol': 1e-5, 'atol': 1e-6,
              'warmups': 5, 'samples': 17, 'cold_native_body_profile': True, 'steady_profile': False, 'legs': [], 'comparisons': [],
              'rustc': command('rustc', '-Vv'), 'nvcc': command('nvcc', '--version')}
    for gpu in args.devices.split(','):
        for order in (('native', 'reference'), ('reference', 'native')):
            pair = {}
            for role in order:
                directory = args.output / f'gpu-{gpu}-{order[0]}-first-{role}'
                directory.mkdir()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                           TORCHINDUCTOR_CACHE_DIR=str(directory / 'inductor'),
                           TRITON_CACHE_DIR=str(directory / 'triton'), CUDA_CACHE_PATH=str(directory / 'cuda'))
                row = {'gpu': gpu, 'role': role, 'order': order, 'directory': directory.name}
                try:
                    with (directory / 'stdout.log').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
                        result = subprocess.run([sys.executable, '-B', __file__, '--role', role,
                                                 '--output', str(directory), '--wheel', str(args.wheel)],
                                                env=env, stdout=stdout, stderr=stderr, timeout=600)
                    row['returncode'] = result.returncode
                except subprocess.TimeoutExpired:
                    row.update(returncode=None, failure='600 second child timeout')
                report['legs'].append(row)
                path = directory / 'report.json.gz'
                if path.exists():
                    pair[role] = json.loads(gzip.decompress(path.read_bytes()))
                    row['passed'] = row['returncode'] == 0 and pair[role]['passed']
                else:
                    row['passed'] = False
                write(args.output / 'summary.json.gz', report)
            comparison = {'gpu': gpu, 'order': order, 'passed': False, 'histories': []}
            report['comparisons'].append(comparison)
            try:
                assert all(pair[role]['passed'] for role in ('native', 'reference'))
                for a, b in zip(pair['native']['cases'], pair['reference']['cases'], strict=True):
                    assert a['name'] == b['name']
                    for x, y in zip(a['histories'], b['histories'], strict=True):
                        assert x['inputs'] == y['inputs']
                        compare(x['output'], y['output'])
                        comparison['histories'].append({'name': a['name'], 'variant': x['variant'],
                            'sample': x['sample'], 'native_median_ns': x['median_ns'],
                            'reference_median_ns': y['median_ns'],
                            'reference_over_native': y['median_ns']/x['median_ns']})
                comparison['passed'] = True
            except Exception:
                comparison['failure'] = traceback.format_exc()
            write(args.output / 'summary.json.gz', report)
    report['passed'] = all(row['passed'] for row in report['legs'] + report['comparisons'])
    write(args.output / 'summary.json.gz', report)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--role', choices=('native', 'reference'))
    args = parser.parse_args()
    args.output, args.wheel = args.output.resolve(), args.wheel.resolve()
    assert args.output.is_relative_to(ROOT) and args.wheel.is_relative_to(ROOT)
    assert Path(sys.executable).resolve().is_relative_to(ROOT)
    if not args.role:
        devices = args.devices.split(',')
        assert len(devices) == 2 and len(set(devices)) == 2, 'select two explicitly reserved GPUs'
    args.output.mkdir(parents=True, exist_ok=bool(args.role))
    sys.exit(leg(args) if args.role else run(args))
