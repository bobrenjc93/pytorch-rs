"""Frozen, non-scoring public-call original/changed/changed/original diagnostic.

Correctness precedes timing in separate processes. No benchmark/evaluator is
replaced. All writes are under this checkout; original installations are read-only.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[3]
SHAPES = ((128, 256), (193, 256), (202, 118), (207, 292), (128, 256))
SOURCE = 'def f(x, extra):\n    return x.sin() + extra\n'
ORDER = ('original', 'changed', 'changed', 'original')
PROTOCOL_SHA = '9bd4579bbfab0690deea856b0248d0d97adaf64ef14b8a724a7cc8de2e1bbf6b'
GPU = 'GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1'
HELPER = ROOT / 'docs/diagnostics/compile-gelu-linked/probe.py'
POLICY = ROOT / 'docs/diagnostics/compile-pointwise-positional/warm-dispatch/gpu-dispatch.py'
POLICY_SHA = 'de002c8bf7c2c1fc46936b0a7b03273822f362897bdc6f22ea1d4a7b7043c811'
HELPER_SHA = 'f5f99ceb9ea941fbcc7c46a840374f4482c4fa94ccc8ad9033c98ada32f32e4f'
CACHES = ('TMPDIR', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')
SOURCE_PATHS = ('src', 'python', 'crates', '.cargo', 'build.rs', 'Cargo.toml',
                'Cargo.lock', 'pyproject.toml', 'uv.lock', 'rust-toolchain.toml')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path.read_bytes()))


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def inside(path):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT) and '.burner' not in path.relative_to(ROOT).parts
    return path


def load(path, name, expected):
    assert sha(path.read_bytes()) == expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_binding(root):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True,
                                       env=dict(os.environ, GIT_OPTIONAL_LOCKS='0')).strip()
    assert Path(git('rev-parse', '--show-toplevel')).resolve() == root
    paths = git('ls-files', '--', *SOURCE_PATHS).splitlines()
    return dict(root=str(root), commit=git('rev-parse', 'HEAD'),
                status=git('status', '--porcelain=v1', '--untracked-files=all'),
                files={p: sha((root/p).read_bytes()) for p in paths})


def binding(build):
    root, wheel = Path(build['source']), Path(build['wheel'])
    package = Path(importlib.util.find_spec('torch_rs').origin).resolve().parent
    assert package.is_relative_to(Path(sys.prefix).resolve()) and package.is_relative_to(root)
    members = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.startswith('torch_rs/') or name.endswith('/'):
                continue
            path = package.parent/name
            assert path.read_bytes() == archive.read(name), name
            if name.endswith('.py'):
                assert (root/'python'/name).read_bytes() == archive.read(name), name
            members[name] = identity(path)
    installed = {'torch_rs/'+str(p.relative_to(package)) for p in package.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts}
    assert installed == set(members), (installed-set(members), set(members)-installed)
    dependencies = {}
    for dist in importlib.metadata.distributions():
        dependencies[dist.metadata['Name']] = dict(version=dist.version,
                                                   metadata_sha256=sha(dist.read_text('METADATA').encode()))
    return dict(source=source_binding(root), wheel=identity(wheel), members=members,
                interpreter=identity(sys.executable), python=sys.version, prefix=sys.prefix,
                dependencies=dependencies)


class Synchronize:
    """Identical explicit CUDA runtime barrier in all native/reference workers."""
    def __init__(self, path):
        self.library = ctypes.CDLL(path)
        self.library.cudaSetDevice.argtypes = [ctypes.c_int]
        self.library.cudaSetDevice.restype = ctypes.c_int
        self.library.cudaDeviceSynchronize.argtypes = []
        self.library.cudaDeviceSynchronize.restype = ctypes.c_int
        self.library.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.library.cudaRuntimeGetVersion.restype = ctypes.c_int
        version = ctypes.c_int()
        assert self.library.cudaRuntimeGetVersion(ctypes.byref(version)) == 0
        self.metadata = dict(file=identity(path), version=version.value, api='cudaDeviceSynchronize')

    def __call__(self):
        assert self.library.cudaSetDevice(0) == 0
        assert self.library.cudaDeviceSynchronize() == 0


def tensor(fw, shape, phase):
    values = [((i+phase) % 23-11)*0.03125 for i in range(math.prod(shape))]
    return fw.tensor(values, dtype=fw.float32).to('cuda:0').reshape(shape)


def capture(reader, tensor_, out, name):
    record = reader.record(tensor_.reshape((math.prod(tensor_.shape),)), out/(name+'.u32le'))
    record.update(shape=list(tensor_.shape), stride=list(tensor_.stride()))
    record.pop('words'); record.pop('values')
    return record


def cache(target):
    c = target._torch_rs_pointwise_cache
    # Passive complete cache ordering/identities and relevant mutable entry state.
    return dict(graphs=[dict(key=repr(k), owner=id(v), lowerings=[(repr(a), id(b)) for a,b in v.lowerings.items()],
                            observations=[(repr(a), b) for a,b in v.observations.items()], numerical_hint=v.numerical_hint)
                        for k,v in c.graphs.items()],
                executors=[dict(key=repr(k), owner=id(v), source=sha(v.source.encode()),
                                ptx=sha(v.ptx.encode()), options=v.options, nvrtc=v.nvrtc_version)
                           for k,v in c.executors.items()],
                prepared=[dict(key=repr(k), owner=id(v[0]), bytes=v[1]) for k,v in c.prepared.items()],
                prepared_bytes=c.prepared_bytes)


def checked(target, original, args, fw):
    def forbid(frame, event, arg):
        if event == 'call' and frame.f_code is original.__code__:
            raise AssertionError('original body replay')
        if event == 'c_call' and getattr(arg, '__name__', None) in ('sin', 'add', '__add__'):
            if type(getattr(arg, '__self__', None)) is fw.Tensor or arg is fw.sin or arg is fw.add:
                raise AssertionError('eager arithmetic replay')
    assert sys.getprofile() is None and sys.gettrace() is None
    sys.setprofile(forbid)
    try:
        return target(*args)
    finally:
        sys.setprofile(None)


def worker(out, declaration, label, kind):
    report = dict(started=now(), label=label, kind=kind, rows=[], passed=False)
    try:
        assert sys.flags.isolated and sys.dont_write_bytecode
        assert identity(declaration['cudart']) == declaration['cuda_runtime']
        assert identity(declaration['nvrtc']) == declaration['nvrtc_library']
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
        assert all(os.environ[n] == '1' for n in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'))
        for name in CACHES:
            path = inside(os.environ[name]); assert path.is_relative_to(out) and not any(path.iterdir())
        build = declaration['builds']['changed' if label == 'reference' else label]
        report['before'] = binding(build)
        if label == 'original':
            assert report['before']['wheel']['sha256'] == '1f3998184e4e5a542ff480c111e00a241b9167ff35917e9c4825a12fe0f55bf8'
            assert report['before']['interpreter']['sha256'] == '202c17d1671602a4ef1d43e9b2fdbef0769443f37bf5e51f6b603e0b2c27d9d8'
            assert any(v['sha256'] == 'b9db6f35d6129c9681b6d47e190558c5b1897ce997ad1c6c7dbe011249f420c2' for v in report['before']['members'].values())
        p = load(HELPER, 'guard_bytes', HELPER_SHA)
        policy = load(POLICY, 'guard_policy', POLICY_SHA)
        report['gpu_before'] = policy.gpu_identity(GPU)
        fw = importlib.import_module('torch' if label == 'reference' else 'torch_rs')
        fw.set_num_threads(1)
        if label == 'reference': assert fw.__version__ == '2.13.0+cu130'
        else: report['native_extension'] = identity(importlib.import_module('torch_rs.torch_rs').__file__)
        sync = Synchronize(declaration['cudart']); reader = p.DeviceBytes()
        report['synchronization'] = sync.metadata
        namespace = {}; exec(compile(SOURCE, '<method-guard-matched-function>', 'exec'), namespace)
        original = namespace['f']; target = fw.compile(original)
        retained = []
        history = [(shape, 0) for shape in SHAPES]
        if kind == 'correctness': history += [(SHAPES[0], 5), (SHAPES[0], 10)]
        for index, (shape, phase) in enumerate(history):
            args = (tensor(fw, shape, phase), tensor(fw, (shape[-1],), phase+3))
            prefix = str(index)
            row = dict(shape=shape, phase=phase, inputs=[capture(reader,t,out,prefix+'-input-'+str(i)) for i,t in enumerate(args)])
            report['rows'].append(row)
            if kind == 'correctness':
                result = target(*args) if label == 'reference' else checked(target, original, args, fw)
                sync(); row['output'] = capture(reader,result,out,prefix+'-output')
                assert result.data_ptr() not in {t.data_ptr() for t in args} | {t.data_ptr() for t,_ in retained}
                row['retained'] = [capture(reader,t,out,prefix+'-retained-'+str(i)) for i,(t,_) in enumerate(retained)]
                assert [r['sha256'] for r in row['retained']] == [r['sha256'] for _,r in retained]
                retained.append((result,row['output']))
            else:
                result = None
                def timed():
                    nonlocal result
                    result = None  # Release previous output outside this clock.
                    assert sys.getprofile() is None and sys.gettrace() is None
                    sync(); start = time.perf_counter_ns()
                    result = target(*args); sync()
                    return time.perf_counter_ns()-start
                row['first_ns'] = timed()
                row['first_output'] = capture(reader,result,out,prefix+'-first')
                row['warmup_ns'] = []
                for _ in range(5): row['warmup_ns'].append(timed())
                row['cache_before'] = cache(target)
                row['samples_ns'] = []
                for _ in range(17): row['samples_ns'].append(timed())
                row['cache_after'] = cache(target)
                row['last_output'] = capture(reader,result,out,prefix+'-last')
                assert row['cache_before'] == row['cache_after'], 'warm cache changed'
                samples = row['samples_ns']; median = statistics.median(samples)
                row['statistics_ns'] = dict(median=median, mad=statistics.median(abs(s-median) for s in samples), minimum=min(samples), maximum=max(samples))
                result = None
            row['inputs_after'] = [capture(reader,t,out,prefix+'-input-after-'+str(i)) for i,t in enumerate(args)]
            assert [r['sha256'] for r in row['inputs']] == [r['sha256'] for r in row['inputs_after']]
            write(out/'report.json', report)  # Only after the full warm block.
        report['after'] = binding(build); assert report['before'] == report['after']
        report['mapped_libraries'] = p.mapped_libraries()
        report['runtime_versions'] = p.runtime_versions(report['mapped_libraries'])
        report['gpu_after'] = policy.gpu_identity(GPU)
        if label != 'reference':
            assert 'torch' not in sys.modules
            assert report['native_extension'] in report['mapped_libraries']
        report['passed'] = True
    except BaseException:
        report['failure'] = traceback.format_exc()
    report['finished'] = now(); write(out/'report.json', report)
    return 0 if report['passed'] else 1


def execute(base, declaration, label, kind, number):
    out = base/f'{number:02d}-{kind}-{label}'; out.mkdir()
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME'): env.pop(key, None)
    for name in CACHES:
        path = out/name; path.mkdir(); env[name] = str(path)
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', TORCH_RS_CUDART=declaration['cudart'],
               TORCH_RS_NVRTC=declaration['nvrtc'])
    build = declaration['builds']['changed' if label == 'reference' else label]
    command = [build['python'], '-I', '-B', str(Path(__file__).resolve()), 'worker', str(base),
               '--label', label, '--kind', kind, '--number', str(number)]
    receipt = dict(command=command, cwd=str(ROOT), environment={k:env.get(k) for k in CACHES+('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','TORCH_RS_CUDART','TORCH_RS_NVRTC','LD_LIBRARY_PATH')}, started=now())
    with (out/'process.log').open('w') as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(returncode=process.returncode, finished=now()); write(out/'exit.json',receipt)
    assert process.returncode == 0, str(out)


def record_values(record, out):
    raw = (out/record['raw']).read_bytes(); assert sha(raw) == record['sha256']
    assert len(raw) == math.prod(record['shape'])*4
    return dict(record, values=[float(x[0]).hex() for x in struct.iter_unpack('<f',raw)])


def compare(base, timing=False):
    policy = load(POLICY, 'guard_compare', POLICY_SHA)
    reference_dir = base/'02-correctness-reference'
    reference = json.loads((reference_dir/'report.json').read_text())
    assert reference['passed']
    assert json.loads((reference_dir/'exit.json').read_text())['returncode'] == 0
    comparisons = 0; workers = []; identities = {}
    paths = [base/f'{i:02d}-timing-{label}' for i,label in enumerate(ORDER,3)] if timing else [base/'00-correctness-original',base/'01-correctness-changed']
    for out in paths:
        report = json.loads((out/'report.json').read_text()); assert report['passed']
        assert json.loads((out/'exit.json').read_text())['returncode'] == 0
        assert len(report['rows']) == (5 if timing else 7)
        assert report['before'] == report['after']
        label = report['label']
        if label in identities: assert report['before'] == identities[label], 'build changed between workers'
        identities[label] = report['before']
        if timing:
            correct_dir = base/('00-correctness-original' if label == 'original' else '01-correctness-changed')
            correct = json.loads((correct_dir/'report.json').read_text())
            assert report['before'] == correct['before'] == correct['after'], 'timed build differs from correctness build'
        for i,row in enumerate(report['rows']):
            rr = reference['rows'][i]
            for role in ('inputs', 'inputs_after'):
                assert [r['sha256'] for r in row[role]] == [r['sha256'] for r in rr[role]]
                for rec in row[role]: record_values(rec,out)
                for rec in rr[role]: record_values(rec,reference_dir)
            for name in ('first_output','last_output') if timing else ('output',):
                policy.close_outputs(record_values(row[name],out),record_values(rr['output'],reference_dir)); comparisons += 1
            for n,rec in enumerate(row.get('retained', [])):
                policy.close_outputs(record_values(rec,out),record_values(rr['retained'][n],reference_dir)); comparisons += 1
        workers.append(report)
    result = dict(passed=True, comparisons=comparisons, policy=identity(POLICY), captured_at=now())
    if timing:
        result['shapes'] = [dict(shape=shape, workers=[dict(label=r['label'], samples_ns=r['rows'][i]['samples_ns'], **r['rows'][i]['statistics_ns']) for r in workers],
                                changed_over_original=[workers[j]['rows'][i]['statistics_ns']['median']/workers[k]['rows'][i]['statistics_ns']['median'] for j,k in ((1,0),(2,3))]) for i,shape in enumerate(SHAPES)]
        result['build_confounds'] = {name: [r['before'][name] for r in workers] for name in ('interpreter','dependencies')}
        result['native_hashes'] = [r['native_extension']['sha256'] for r in workers]
        result['limits'] = 'Two workers/version; no precise confidence interval or causal grouping-only claim when build identities differ. Only first/last timing outputs captured; no certification of intermediate outputs. Every sample retained.'
    write(base/('timing-comparison.json' if timing else 'correctness-comparison.json'),result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('declare','correctness','timing','worker','compare'))
    parser.add_argument('directory', type=Path)
    for build in ('original','changed'):
        for field in ('source','python','wheel'): parser.add_argument('--'+build+'-'+field,type=Path)
    parser.add_argument('--cudart',type=Path); parser.add_argument('--nvrtc',type=Path)
    parser.add_argument('--label',choices=('original','changed','reference'))
    parser.add_argument('--kind',choices=('correctness','timing')); parser.add_argument('--number',type=int)
    args = parser.parse_args(); base = inside(args.directory)
    if args.action == 'declare':
        builds = {b:{f:str(getattr(args,b+'_'+f).absolute()) for f in ('source','python','wheel')} for b in ('original','changed')}
        for field in ('source','python','wheel'): inside(builds['changed'][field])
        base.mkdir(parents=True,exist_ok=False)
        write(base/'declaration.json',dict(declared=now(),driver=identity(__file__),builds=builds,protocol_sha256=PROTOCOL_SHA,
              source=SOURCE,shapes=SHAPES,phases=[0,3],order=ORDER,warmups=5,samples=17,
              cudart=str(args.cudart.resolve()),nvrtc=str(args.nvrtc.resolve()),
              cuda_runtime=identity(args.cudart),nvrtc_library=identity(args.nvrtc),
              policy=identity(POLICY),helper=identity(HELPER),scope=__doc__))
        return 0
    declaration = json.loads((base/'declaration.json').read_text())
    assert declaration['driver'] == identity(__file__), 'driver changed after declaration'
    if args.action == 'worker': return worker(base/f'{args.number:02d}-{args.kind}-{args.label}',declaration,args.label,args.kind)
    if args.action == 'correctness':
        for i,label in enumerate(('original','changed','reference')): execute(base,declaration,label,'correctness',i)
        compare(base)
    if args.action == 'timing':
        assert json.loads((base/'correctness-comparison.json').read_text())['passed']
        for i,label in enumerate(ORDER,3): execute(base,declaration,label,'timing',i)
        compare(base,True)
    if args.action == 'compare': compare(base,True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
