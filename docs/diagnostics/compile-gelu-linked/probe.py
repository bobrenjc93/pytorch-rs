"""Non-scoring private actual-planner prerequisite, not public compile evidence.

The historical 44 formulas/113 calls/335 leaves and numerical policy are pinned.
A reference history has one ordinary default torch.compile wrapper and no reset.
Native shape hints are private diagnostics, not a public cache certificate.
Cache snapshots are passive associations, not selected-executable attestations.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'docs/diagnostics/compile-gelu-stage1/probe.py'
OLD_SHA = 'd7b26d9ccad73571579f5342c008942366a5e11090487c70d22434d322853a26'
POLICY = ROOT / 'docs/diagnostics/compile-pointwise-positional/warm-dispatch/gpu-dispatch.py'
POLICY_SHA = 'de002c8bf7c2c1fc46936b0a7b03273822f362897bdc6f22ea1d4a7b7043c811'
CACHE_NAMES = ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'XDG_CACHE_HOME', 'TMPDIR')
ENV_NAMES = ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'TORCH_RS_NVRTC', 'TORCH_RS_CUDART', 'LD_LIBRARY_PATH')
SELECTED = {'erf', 'gelu', 'erf_affine', 'products_first', 'sum_first', 'threshold_32', 'threshold_102', 'scalar_history'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def canonical(value):
    # Python's JSON NaN spelling is stable here; direct NaN equality is false.
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def load(path, name, expected):
    assert sha(path) == expected, (path, 'pinned input changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = load(OLD, 'unchanged_stage1_fixtures', OLD_SHA)


def command(*args):
    return subprocess.check_output(args, text=True, cwd=ROOT).strip()


def gpu_snapshot():
    return command('nvidia-smi', '--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used', '--format=csv')


def declaration():
    cases = fixtures.cases()
    result = dict(cases=cases, calls=sum(len(c['history']) for c in cases),
                  leaves=sum(len(c['history']) * len(c['order']) for c in cases),
                  input_words=fixtures.fixture(), fixture_sha256=OLD_SHA, policy_sha256=POLICY_SHA)
    assert len(cases) == 44 and result['calls'] == 113 and result['leaves'] == 335
    assert {0x40700000, 0x407fffff, 0x40800000, 0x40800001, 0x40880000, 0x40a00000} <= set(result['input_words'])
    return result


def inside(path):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT) and '.burner' not in path.relative_to(ROOT).parts
    return path


class DeviceBytes:
    """Copy actual float32 storage bytes; no float conversion before recording.

    Observed NaN words are retained too, but NaN payload/sign equality is outside
    the unchanged numerical policy. Input construction and arithmetic may change
    NaN payloads; neither payload preservation nor portable equality is claimed.
    """
    def __init__(self):
        self.lib = ctypes.CDLL('libcuda.so.1')
        signatures = {
            'cuPointerGetAttribute': [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64],
            'cuCtxGetCurrent': [ctypes.POINTER(ctypes.c_void_p)],
            'cuCtxGetDevice': [ctypes.POINTER(ctypes.c_int)],
            'cuCtxSynchronize': [],
            'cuMemcpyDtoH_v2': [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
            'cuDriverGetVersion': [ctypes.POINTER(ctypes.c_int)],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args; fn.restype = ctypes.c_int

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f'{name}: CUDA driver status {status}')

    def record(self, tensor, path):
        shape = list(tensor.shape)
        assert len(shape) == 1 and list(tensor.stride()) == [1]
        assert str(tensor.dtype) == 'torch.float32' and str(tensor.device) == 'cuda:0'
        ptr = tensor.data_ptr()
        allocation_context = ctypes.c_void_p()
        self.call('cuPointerGetAttribute', ctypes.byref(allocation_context), 1, ptr)
        context = ctypes.c_void_p(); device = ctypes.c_int()
        pointer_device = ctypes.c_int(); memory_type = ctypes.c_int()
        self.call('cuCtxGetCurrent', ctypes.byref(context))
        assert context.value is not None, 'raw capture requires an active CUDA context'
        self.call('cuCtxGetDevice', ctypes.byref(device))
        self.call('cuPointerGetAttribute', ctypes.byref(pointer_device), 9, ptr)
        self.call('cuPointerGetAttribute', ctypes.byref(memory_type), 2, ptr)
        assert device.value == pointer_device.value == 0
        assert memory_type.value == 2, 'raw capture requires CUDA device storage'
        # Stream-ordered pool allocations have no allocation context. Use the
        # already-current, device-validated context without changing its stack.
        assert allocation_context.value in (None, context.value)
        self.call('cuCtxSynchronize')
        buffer = ctypes.create_string_buffer(shape[0] * 4)
        self.call('cuMemcpyDtoH_v2', buffer, ptr, len(buffer))
        raw = buffer.raw
        context_after = ctypes.c_void_p()
        self.call('cuCtxGetCurrent', ctypes.byref(context_after))
        assert context_after.value == context.value
        path.write_bytes(raw)
        words = struct.unpack('<' + 'I' * shape[0], raw)
        values = struct.unpack('<' + 'f' * shape[0], raw)
        return dict(shape=shape, stride=list(tensor.stride()), dtype=str(tensor.dtype),
                    device=str(tensor.device), requiresGrad=tensor.requires_grad,
                    offset=tensor.storage_offset(), pointer=ptr, context=context.value,
                    allocation_context=allocation_context.value,
                    pointer_device=pointer_device.value, memory_type=memory_type.value,
                    values=[float(v).hex() for v in values], words=[f'{w:08x}' for w in words],
                    sha256=hashlib.sha256(raw).hexdigest(), raw=path.name)


def source_identities():
    paths = list((ROOT / 'src').rglob('*.rs')) + list((ROOT / 'python/torch_rs').rglob('*.py'))
    paths += list((ROOT / 'src/cuda').glob('erf_provider.*'))
    paths += [ROOT / 'Cargo.toml', ROOT / 'Cargo.lock', Path(__file__), OLD, POLICY]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def mapped_libraries():
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if any(s in line for s in ('libcuda', 'libcudart', 'libnvrtc', 'torch_rs'))}
    return [identity(p) for p in sorted(paths) if Path(p).is_file()]


def runtime_versions(libraries):
    versions = []
    for item in libraries:
        if 'libcudart' not in Path(item['path']).name:
            continue
        library = ctypes.CDLL(item['path'])
        function = library.cudaRuntimeGetVersion
        function.argtypes = [ctypes.POINTER(ctypes.c_int)]
        function.restype = ctypes.c_int
        version = ctypes.c_int()
        status = function(ctypes.byref(version))
        versions.append(dict(path=item['path'], status=status, version=version.value))
        assert status == 0
    return versions


def observe_cache(out, seen, case, step):
    changed = []
    for name in CACHE_NAMES[:2]:
        for path in sorted((out / name).rglob('*')):
            if not path.is_file():
                continue
            relative = str(path.relative_to(out))
            item = dict(sha256=sha(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            if seen.get(relative) == item:
                continue
            seen[relative] = item
            entry = dict(path=relative, **item)
            if case in SELECTED and path.suffix in {'.py', '.ptx', '.llir', '.ttir', '.ttgir', '.json'}:
                destination = out / 'selected-cache' / item['sha256'] / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copyfile(path, destination)
                entry['retained'] = str(destination.relative_to(out))
            changed.append(entry)
    return dict(case=case, step=step, observed_at=now(), changed=changed)


def run_leg(base, leg):
    declared = json.loads((base / 'matrix.json').read_text())
    assert canonical(declared['matrix']) == canonical(declaration())
    out = base / leg
    out.mkdir(exist_ok=False)
    for name in CACHE_NAMES:
        path = out / name; path.mkdir(); os.environ[name] = str(path)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    report = dict(leg=leg, started_at=now(), declaration=identity(base / 'matrix.json'), cases=[],
                  before=gpu_snapshot(), source=source_identities(), root=str(ROOT),
                  git_commit=command('git', 'rev-parse', 'HEAD'), git_status=command('git', 'status', '--porcelain'),
                  python=identity(sys.executable), python_version=sys.version,
                  nvcc_inventory=command('nvcc', '--version'), cache_initially_empty=True,
                  environment={k: os.environ.get(k) for k in ENV_NAMES + CACHE_NAMES},
                  scope='Private graph/hints; no public cache or per-call selected executable certificate.',
                  bit_capture='Synchronized cuMemcpyDtoH_v2; finite bits exact; NaN payload equality not required')
    if leg == 'native':
        assert any(Path(p['path']) == ROOT / 'src/cuda/erf_provider.ptx' for p in declared['provider_inputs']), 'declare the actual embedded provider PTX'
        report['provider_inputs'] = []
        for item in declared['provider_inputs']:
            original = inside(item['path'])
            assert identity(original) == item, 'provider input changed after declaration'
            retained = out / 'provider-inputs' / item['sha256'] / original.name
            retained.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, retained)
            report['provider_inputs'].append(dict(**item, retained=str(retained.relative_to(out))))
    write(out / 'report.json', report)
    if leg == 'native':
        import torch_rs as fw
        from torch_rs import torch_rs as bridge
        report['extension'] = identity(bridge.__file__)
        assert Path(bridge.__file__).resolve().is_relative_to(ROOT), 'extension must be built inside worktree'
        assert report['extension'] in mapped_libraries(), 'extension not mapped'
    else:
        import torch as fw
        assert fw.__version__ == '2.13.0+cu130'
        import triton
        report['reference_libdevice'] = identity(Path(triton.__file__).parent / 'backends/nvidia/lib/libdevice.10.bc')
        assert report['reference_libdevice']['sha256'] == '5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee'
        report['reference_cuda'] = fw.version.cuda
    fw.set_num_threads(1)
    report['framework_version'] = fw.__version__
    reader = DeviceBytes(); driver = ctypes.c_int()
    reader.call('cuDriverGetVersion', ctypes.byref(driver))
    report['driver_api_version'] = driver.value
    seen = {}; cache_history = []
    for case in declared['matrix']['cases']:
        row = dict(name=case['name'], steps=[], started_at=now()); report['cases'].append(row)
        directory = out / case['name']; directory.mkdir()
        try:
            nodes = tuple(tuple(n) for n in case['nodes']); roots = tuple(case['roots']); order = tuple(case['order'])
            if leg == 'reference':
                namespace = dict(torch=fw, F=fw.nn.functional)
                exec(case['source'], namespace)
                # Exactly one ordinary default wrapper for the entire history.
                compiled = fw.compile(namespace['f'])
            kernels = {}; preparations = {}; retained = []
            for step, (count, scalar) in enumerate(case['history']):
                words = (fixtures.fixture() * ((count + 147) // 148))[:count]
                if step % 2:
                    words.reverse()
                values = [struct.unpack('<f', struct.pack('<I', w))[0] for w in words]
                inputs = (fw.tensor(values, dtype=fw.float32).to('cuda:0'),)
                if case['arity'] == 2:
                    inputs += (fw.tensor([-v for v in values], dtype=fw.float32).to('cuda:0'),)
                observation = dict(call=step, started_at=now(), scalar=float(scalar).hex(),
                                   inputs=[reader.record(t, directory / f'{step}-input-{i}.u32le') for i, t in enumerate(inputs)])
                row['steps'].append(observation)
                start = time.perf_counter_ns()
                if leg == 'reference':
                    result = compiled(*inputs, scalar)
                    outputs = result if type(result) is tuple else (result,)
                else:
                    current = nodes
                    runtime = case['name'] == 'scalar_history' and step > 0
                    if case['name'] == 'scalar_history' and not runtime:
                        current = tuple(('constant', 0, 0, fixtures.bits(scalar)) if n[0] == 'scalar' else n for n in nodes)
                    if current not in kernels:
                        kernel = bridge._pointwise_compile(inputs, current, roots); kernels[current] = kernel
                        (directory / f'{step}-native.cu').write_text(kernel.source)
                        (directory / f'{step}-native.ptx').write_text(kernel.ptx)
                        row['nvrtc'] = kernel.nvrtc_version; row['options'] = kernel.options
                        observation['link_inputs'] = dict(executor=identity(directory / f'{step}-native.ptx'),
                                                         providers=report['provider_inputs'])
                    kernel = kernels[current]
                    hint = count if step == 0 or count == case['history'][0][0] and len({s[0] for s in case['history'][:step+1]}) == 1 else case['history'][1][0]
                    key = (current, count, hint)
                    if key not in preparations:
                        preparations[key] = kernel.prepare(inputs, hint, order)
                    prepared = preparations[key]
                    (directory / f'{step}-plan.txt').write_text(kernel.plan(hint, order))
                    assert bridge._pointwise_plan(current, roots, case['arity'], [list(t.shape) for t in inputs], hint, order) == kernel.plan(hint, order)
                    assert bridge._pointwise_source(current, roots, case['arity'], [list(t.shape) for t in inputs]) == kernel.source
                    result = prepared.run(inputs, (scalar,) if runtime else ())
                    outputs = tuple(result[i] for i in order)
                    observation['preparations'] = len(preparations)
                observation['call_host_ns'] = time.perf_counter_ns() - start
                observation['outputs'] = [reader.record(t, directory / f'{step}-output-{i}.u32le') for i, t in enumerate(outputs)]
                observation['inputs_after'] = [reader.record(t, directory / f'{step}-input-after-{i}.u32le') for i, t in enumerate(inputs)]
                assert [r['sha256'] for r in observation['inputs']] == [r['sha256'] for r in observation['inputs_after']]
                output_ptrs = [t.data_ptr() for t in outputs]
                if leg == 'native':
                    assert {t.data_ptr() for t in inputs}.isdisjoint(output_ptrs)
                    assert len(set(output_ptrs)) == len(output_ptrs)
                observation['retained'] = []
                for prior, previous, records in retained:
                    snapshots = [reader.record(t, directory / f'retained-current-{step}-prior-{prior}-leaf-{i}.u32le') for i, t in enumerate(previous)]
                    assert [r['sha256'] for r in snapshots] == [r['sha256'] for r in records]
                    if leg == 'native':
                        assert {r['pointer'] for r in snapshots}.isdisjoint(output_ptrs)
                    observation['retained'].append(dict(prior_call=prior, outputs=snapshots))
                retained.append((step, outputs, observation['outputs']))
                observation['finished_at'] = now()
                cache_history.append(observe_cache(out, seen, case['name'], step))
        except Exception:
            row['error'] = traceback.format_exc()
        row['finished_at'] = now()
        write(out / 'report.json', report)
        write(out / 'cache-index.json', dict(files=seen, history=cache_history))
        print(leg, case['name'], 'error' if row.get('error') else 'captured', flush=True)
    if leg == 'native':
        assert 'torch' not in sys.modules
        assert identity(bridge.__file__) == report['extension'], 'extension file changed during run'
    report['mapped_libraries'] = mapped_libraries()
    report['runtime_versions'] = runtime_versions(report['mapped_libraries'])
    report['source_after'] = source_identities()
    report['source_unchanged'] = report['source'] == report['source_after']
    report['after'] = gpu_snapshot(); report['finished_at'] = now()
    report['passed_execution'] = report['source_unchanged'] and not any(c.get('error') for c in report['cases'])
    write(out / 'report.json', report)
    return 0 if report['passed_execution'] else 1


def compare(base):
    policy = load(POLICY, 'unchanged_policy', POLICY_SHA)
    matrix = declaration(); reports = {}
    declared = json.loads((base / 'matrix.json').read_text())
    assert canonical(declared['matrix']) == canonical(matrix)
    for leg in ('native', 'reference'):
        receipt = json.loads((base / f'{leg}-exit.json').read_text())
        assert receipt['returncode'] == 0, (leg, receipt)
        reports[leg] = json.loads((base / leg / 'report.json').read_text())
        assert [r['name'] for r in reports[leg]['cases']] == [c['name'] for c in matrix['cases']]
    failures = []; compared = 0; bit_differences = 0; finite_differences = 0; signed_zero_differences = 0
    for case, nc, rc in zip(matrix['cases'], reports['native']['cases'], reports['reference']['cases'], strict=True):
        assert not nc.get('error') and not rc.get('error')
        assert len(nc['steps']) == len(rc['steps']) == len(case['history'])
        for step, (ns, rs) in enumerate(zip(nc['steps'], rc['steps'], strict=True)):
            assert ns['scalar'] == rs['scalar']
            assert [r['words'] for r in ns['inputs']] == [r['words'] for r in rs['inputs']]
            for leg, observation in (('native', ns), ('reference', rs)):
                directory = base / leg / case['name']
                records = observation['inputs'] + observation['inputs_after'] + observation['outputs']
                records += [r for old in observation['retained'] for r in old['outputs']]
                for record in records:
                    path = inside(directory / record['raw'])
                    raw = path.read_bytes()
                    assert hashlib.sha256(raw).hexdigest() == record['sha256']
                    assert len(raw) == record['shape'][0] * 4
                    assert [f'{w[0]:08x}' for w in struct.iter_unpack('<I', raw)] == record['words']
                    assert [float(v[0]).hex() for v in struct.iter_unpack('<f', raw)] == record['values']
            assert len(ns['outputs']) == len(rs['outputs']) == len(case['order'])
            for leaf, (nv, rv) in enumerate(zip(ns['outputs'], rs['outputs'], strict=True)):
                compared += 1
                for nw, rw in zip(nv['words'], rv['words'], strict=True):
                    n, r = (struct.unpack('<f', struct.pack('<I', int(w, 16)))[0] for w in (nw, rw))
                    bit_differences += nw != rw
                    finite_differences += nw != rw and math.isfinite(n) and math.isfinite(r)
                    signed_zero_differences += n == r == 0 and nw != rw
                try:
                    policy.close_outputs(nv, rv)
                except AssertionError as error:
                    failures.append(dict(case=nc['name'], step=step, leaf=leaf, error=str(error)))
    assert compared == 335
    result = dict(compared=compared, bit_differences=bit_differences, finite_bit_differences=finite_differences,
                  signed_zero_differences=signed_zero_differences, failures=failures, passed=not failures,
                  measured_at=now(), policy_sha256=POLICY_SHA)
    write(base / 'comparison.json', result)
    print(json.dumps(result))
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('leg', choices=['declare', 'native', 'reference', 'compare', 'pack'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--provider-input', type=Path, action='append', default=[], help='Worktree-local provider PTX/source/generation receipt; pin at declaration')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(); base = inside(args.directory)
    if args.leg == 'declare':
        base.mkdir(parents=True, exist_ok=False)
        write(base / 'matrix.json', dict(declared_at=now(), matrix=declaration(), provider_inputs=[identity(inside(p)) for p in args.provider_input]))
        shutil.copyfile(__file__, base / 'probe.py')
        return 0
    if args.provider_input:
        parser.error('--provider-input belongs to declare')
    if args.leg == 'compare':
        return compare(base)
    if args.leg == 'pack':
        paths = [p for p in sorted(base.rglob('*')) if p.is_file() and not any(n in p.relative_to(base).parts for n in CACHE_NAMES)]
        archive = base.parent / (base.name + '-essential.tar.gz')
        manifest = {str(p.relative_to(base)): dict(sha256=sha(p), bytes=p.stat().st_size) for p in paths}
        with tarfile.open(archive, 'w:gz') as tar:
            for p in paths:
                tar.add(p, arcname=str(p.relative_to(base)), recursive=False)
        write(base.parent / (base.name + '-essential-manifest.json'), dict(archive=identity(archive), files=manifest))
        return 0
    if args.worker:
        return run_leg(base, args.leg)
    receipt = dict(started_at=now(), command=[sys.executable, '-B', str(Path(__file__).resolve()), args.leg, str(base), '--worker'],
                   cwd=str(ROOT), environment={k: os.environ.get(k) for k in ENV_NAMES})
    with (base / f'{args.leg}.log').open('w') as log:
        process = subprocess.run(receipt['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(returncode=process.returncode, finished_at=now())
    write(base / f'{args.leg}-exit.json', receipt)
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
