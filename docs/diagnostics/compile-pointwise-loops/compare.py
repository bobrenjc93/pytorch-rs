"""Non-scoring, source-bound literal-loop diagnostic in isolated native/reference legs.

Run from the checkout root with a freshly installed wheel, reserved GPUs, and
worktree-local writable caches. Retains initial and last available result/input
observations, observed native receiver artifacts, timings and failures; individual
normal warmup/sample tensor returns are not archived. No evaluator or corpus
definition is changed. See README.md for reproduction and scope.
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
    sources = {
        'recurrence': 'def f(x,y,scale):\n for i in range(3):\n  x=(x*scale+y).relu()\n return x',
        'signed_sequential': 'def h(a,b):\n return a+b\ndef f(x,y,scale):\n for i in range(3,-2,-2):\n  x=h(x,i)\n for j in range(1,3):\n  x=x*scale\n return x-y',
        'zero_trip': 'def f(x,y,scale):\n for i in range(0):\n  scale=7\n  x=-x\n return x*scale+y',
    }
    scope = {}
    exec(sources[name], scope)
    def inputs(variant, sample):
        import numpy as np
        rng = np.random.default_rng(1729 + 1009*variant + 1000003*sample)
        shape = ((7, 11), (3, 17), (0, 11))[variant]
        offset = sample + 1
        count = math.prod(shape)
        data = [rng.normal(0, 0.2, count+offset).astype('float32') for _ in range(2)]
        if sample == 2 and count:
            for row in data:
                row[offset:offset+7] = [float('nan'), float('inf'), -float('inf'), -0., 0., 0.25, -0.75]
        tensors = [fw.tensor(row.tolist(), dtype=fw.float32).to('cuda:0')[offset:].reshape(shape)
                   for row in data]
        return *tensors, (0.375 if sample % 2 == 0 else 0.5)
    return scope['f'], tuple(scope[k] for k in ('h',) if k in scope), inputs


def capture_observations(history, result, tensors, dispatches, out, tag):
    """Best-effort evidence capture, without invoking the workload again."""
    errors = history.setdefault('observation_errors', [])
    for key, observe in (
        ('observed_output', lambda: encode(result) if result is not None else None),
        ('observed_inputs', lambda: [encode(t) for t in tensors]),
    ):
        try:
            history[key] = observe()
        except Exception:
            errors.append({'observation': key, 'failure': traceback.format_exc()})
    receivers = []
    history['observed_kernels'] = []
    for phase, receiver in dispatches:
        if any(receiver is previous for previous in receivers):
            continue
        receivers.append(receiver)
        record = {'phases': [p for p, k in dispatches if k is receiver]}
        history['observed_kernels'].append(record)
        stem = f'{tag}-observed-{len(receivers)-1}'
        # Each artifact is saved independently: e.g. a PTX/property failure
        # must not discard a source already obtained from an observed receiver.
        for kind in ('source', 'ptx'):
            try:
                data = getattr(receiver, kind).encode()
                path = out / (stem + ('.cu' if kind == 'source' else '.ptx.gz'))
                with path.open('xb') as stream:
                    stream.write(data if kind == 'source' else gzip.compress(data, mtime=0))
                record[kind + '_path'] = path.name
                record[kind + '_sha256'] = hashlib.sha256(data).hexdigest()
            except Exception:
                errors.append({'observation': stem + '/' + kind, 'failure': traceback.format_exc()})
        for key, attribute in (('nvrtc', 'nvrtc_version'), ('options', 'options'), ('device', 'device')):
            try:
                record[key] = getattr(receiver, attribute)
            except Exception:
                errors.append({'observation': stem + '/' + key, 'failure': traceback.format_exc()})


def measure_history(compiled, values, sync, history, save, profile=None):
    """Keep the timing/profiling contract; checkpoint observations on every exit."""
    result = None
    try:
        history['phase'] = 'cold'
        if profile is not None:
            sys.setprofile(profile)
        sync()
        start = time.perf_counter_ns()
        result = compiled(*values)
        sync()
        history['cold_ns'] = time.perf_counter_ns() - start
        history['output'] = encode(result)
        sys.setprofile(None)
        for i in range(5):
            history.update(phase='warmup', phase_index=i)
            result = None  # A throwing call has no returned output to observe.
            sync()
            start = time.perf_counter_ns()
            result = compiled(*values)
            sync()
            history['warmup_ns'].append(time.perf_counter_ns() - start)
        for i in range(17):
            history.update(phase='sample', phase_index=i)
            result = None
            sync()
            start = time.perf_counter_ns()
            result = compiled(*values)
            sync()
            history['samples_ns'].append(time.perf_counter_ns() - start)
        if profile is not None:
            history.update(phase='warm_guard', phase_index=None)
            result = None
            sys.setprofile(profile)
            result = compiled(*values)
            sync()
    except Exception:
        history['failure'] = traceback.format_exc()
        raise
    finally:
        sys.setprofile(None)
        save(result)
    return result


def leg(args):
    out = args.output
    report = {'role': args.role, 'started_ns': time.time_ns(),
              'caches': {key: os.environ.get(key) for key in ('CUDA_CACHE_PATH', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')},
              'gpu_before': inventory(), 'visible': os.environ['CUDA_VISIBLE_DEVICES'],
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
            from torch_rs import torch_rs as bridge, _compile_pointwise as frontend
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
            from torch._inductor import metrics
            from torch._dynamo.utils import counters
            report['reference'] = {'version': fw.__version__, 'cuda': fw.version.cuda,
                                   'path': fw.__file__, 'config': fw.__config__.show()}
        assert Path(fw.__file__).resolve().is_relative_to(ROOT)
        sync = Synchronize()
        report['synchronization'] = sync.metadata
        for name in ('recurrence', 'signed_sequential', 'zero_trip'):
            row = {'name': name, 'histories': [], 'passed': False}
            report['cases'].append(row)
            try:
                fn, helpers, inputs = make(fw, name)
                compiled = fw.compile(fn)
                codes = (fn.__code__, *(helper.__code__ for helper in helpers))
                dispatches, lowering_calls, body_calls = [], [], []
                def forbid(frame, event, arg):
                    if event == 'c_call' and getattr(arg, '__name__', '') == 'run':
                        owner = getattr(arg, '__self__', None)
                        if type(owner) is bridge._PointwiseKernel:
                            dispatches.append((history['phase'], owner))
                    if event == 'call' and frame.f_code is frontend.lower.__code__:
                        lowering_calls.append(history['phase'])
                    if event == 'call' and any(frame.f_code is code for code in codes):
                        body_calls.append(history['phase'])
                        raise AssertionError('original body executed')
                for variant, sample in ((0, 0), (1, 0), (1, 1), (0, 2), (2, 0)):
                    dispatches.clear()
                    lowering_calls.clear()
                    body_calls.clear()
                    values = inputs(variant, sample)
                    tensors = tuple(v for v in values if type(v) is fw.Tensor)
                    before = [encode(t) for t in tensors]
                    history = {'variant': variant, 'sample': sample, 'inputs': before, 'samples_ns': [], 'warmup_ns': []}
                    row['histories'].append(history)
                    if args.role == 'reference':
                        history['reference_kernels_before'] = metrics.generated_kernel_count
                    def save(result):
                        capture_observations(history, result, tensors, dispatches, out,
                                             f'{name}-{variant}-{sample}')
                        if history['samples_ns']:
                            history['median_ns'] = statistics.median(history['samples_ns'])
                        if args.role == 'reference':
                            history['reference_kernels_after'] = metrics.generated_kernel_count
                            history['reference_counters'] = {key: dict(value) for key, value in counters.items()}
                            history['reference_attribution'] = ('generated Inductor kernel' if
                                history['reference_kernels_after'] > history['reference_kernels_before'] else
                                'unknown for this call; retained wrapper and aggregate counters only')
                        else:
                            history['native_dispatch'] = {
                                'api': '_PointwiseKernel.run', 'profiled_phases': [p for p, _ in dispatches],
                                'lowering_phases': lowering_calls[:], 'original_body_calls': len(body_calls),
                                'nonempty_output': bool(result is not None and math.prod(result.shape)),
                                'scope': 'C-extension executor entry; no driver-level kernel trace'}
                        # Persist available values/artifacts before any parity or
                        # attribution assertion; this also runs on execution errors.
                        write(out / 'report.json.gz', report)
                    result = measure_history(compiled, values, sync, history, save,
                                             forbid if args.role == 'native' else None)
                    assert not history['observation_errors'], history['observation_errors']
                    assert history['observed_output'] == history['output']
                    assert history['observed_inputs'] == before
                    assert not math.prod(result.shape) or all(result.data_ptr() != t.data_ptr() for t in tensors)
                    if args.role == 'native':
                        assert len(dispatches) == 2 and dispatches[0][1] is dispatches[1][1]
                        assert [p for p, _ in dispatches] == ['cold', 'warm_guard']
                        assert 'warm_guard' not in lowering_calls
                        kernel = dispatches[0][1]
                        history['kernel'] = history['observed_kernels'][0]
                        assert 'torch_rs_pointwise' in kernel.ptx
                        assert kernel.ptx.count('.visible .entry') == 1
                row['passed'] = True
            except Exception:
                row['failure'] = traceback.format_exc()
            write(out / 'report.json.gz', report)
        # Native admission must still reject original two-stage unequal shapes.
        if args.role == 'native':
            namespace = {}
            exec('def f(x,y):\n for i in range(2):\n  x=x+y\n return x', namespace)
            bounded = fw.compile(namespace['f'])
            x = fw.tensor([[1.,2.],[3.,4.]], dtype=fw.float32).to('cuda:0')
            y = fw.tensor([1.,2.], dtype=fw.float32).to('cuda:0')
            try:
                bounded(x,y)
                raise AssertionError('unequal-shape original IR admitted')
            except NotImplementedError as error:
                report['unequal_shape_rejection'] = str(error)
            assert not bounded._torch_rs_pointwise_cache.graphs
        report['passed'] = all(row['passed'] for row in report['cases'])
    except Exception:
        report['failure'] = traceback.format_exc()
    finally:
        report['finished_ns'] = time.time_ns()
        report['gpu_after'] = inventory()
        report['torch_imported'] = 'torch' in sys.modules
        report['loaded_cuda_runtimes'] = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                                if any(name in line for name in ('libcudart.so', 'libnvrtc.so', 'libcuda.so'))})
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
            if b == 0:
                assert math.copysign(1, a) == math.copysign(1, b)


def run(args):
    manifest_paths = command('git', 'ls-files', 'src', 'python', 'Cargo.toml', 'Cargo.lock',
                             'pyproject.toml', 'uv.lock', 'rust-toolchain.toml',
                             'scripts/torch_compile_default_corpus.py').splitlines()
    manifest = {p: sha(ROOT / p) for p in manifest_paths}
    manifest[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    manifest['tests/test_compile_pointwise_loops.py'] = sha(ROOT / 'tests/test_compile_pointwise_loops.py')
    write(args.output / 'source-manifest.json.gz', manifest)
    report = {'kind': 'non-scoring literal-loop diagnostic', 'rtol': 1e-5, 'atol': 1e-6,
              'warmups': 5, 'samples': 17, 'cold_native_body_profile': True, 'steady_profile': False, 'legs': [], 'comparisons': [],
              'rustc': command('rustc', '-Vv'), 'nvcc': command('nvcc', '--version')}
    for gpu in args.devices.split(','):
        for order in (('native', 'reference'), ('reference', 'native')):
            pair = {}
            for role in order:
                directory = args.output / f'gpu-{gpu}-{order[0]}-first-{role}'
                directory.mkdir(mode=0o700)
                workspace = args.workspace / 'caches' / directory.name
                workspace.mkdir(parents=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                           TORCHINDUCTOR_CACHE_DIR=str(workspace / 'inductor'),
                           TRITON_CACHE_DIR=str(workspace / 'triton'), CUDA_CACHE_PATH=str(workspace / 'cuda'),
                           TMPDIR=str(workspace), XDG_CACHE_HOME=str(workspace / 'xdg'))
                row = {'gpu': gpu, 'role': role, 'order': order, 'directory': directory.name}
                timed_out = False
                try:
                    with (directory / 'stdout.log').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
                        result = subprocess.run([sys.executable, '-B', __file__, '--role', role,
                                                 '--output', str(workspace), '--durable-raw-output', str(directory),
                                                 '--wheel', str(args.wheel)],
                                                env=env, stdout=stdout, stderr=stderr, timeout=600)
                    row['returncode'] = result.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                    row.update(returncode=None, failure='600 second child timeout',
                               descendants_quiescent=False)
                report['legs'].append(row)
                path = directory / 'report.json.gz'
                row['passed'] = False
                if path.exists():
                    try:
                        pair[role] = json.loads(gzip.decompress(path.read_bytes()))
                        row['passed'] = row['returncode'] == 0 and pair[role]['passed']
                    except Exception:
                        row['report_read_failure'] = traceback.format_exc()
                if timed_out:
                    report.update(passed=False, stopped='timeout: execution owner must establish descendant quiescence before further measurements')
                    write(args.output / 'summary.json.gz', report)
                    return 1
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


def prepare_paths(args):
    # --output always owns the disposable, worktree-local workspace. Only raw
    # observations may be redirected to an explicitly supplied durable location.
    args.workspace, args.wheel = args.output.resolve(), args.wheel.resolve()
    assert args.workspace.is_relative_to(ROOT) and args.wheel.is_relative_to(ROOT)
    assert Path(sys.executable).resolve().is_relative_to(ROOT)
    args.output = args.durable_raw_output.resolve() if args.durable_raw_output else args.workspace
    if not args.role:
        devices = args.devices.split(',')
        assert len(devices) == 2 and len(set(devices)) == 2, 'select two explicitly reserved GPUs'
        # Never append to another attempt, even after a failed run. Require an
        # existing durable parent; do not create an external directory hierarchy.
        args.workspace.mkdir(parents=True, exist_ok=False)
        if args.output != args.workspace:
            args.output.mkdir(mode=0o700, exist_ok=False)
    else:
        assert args.workspace.is_dir() and args.output.is_dir()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--durable-raw-output', type=Path,
                        help='new exclusive raw-output directory; caches remain under --output in the checkout')
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--role', choices=('native', 'reference'))
    args = parser.parse_args()
    prepare_paths(args)
    sys.exit(leg(args) if args.role else run(args))
