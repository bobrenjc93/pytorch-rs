"""Unscored tensor-leaf madd evidence, in separate ordinary-default processes.

Run a fresh leg with IMPLEMENTATION OUTPUT_DIRECTORY BUILD_RECORD, then compare its report
with a leg from the other implementation using --compare LEFT RIGHT OUTPUT.
This diagnostic does not change evaluator commands, cases, or expectations.
"""
import contextlib
import gzip
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import sysconfig
import subprocess
import tarfile
import time
import traceback
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
os.environ['GIT_OPTIONAL_LOCKS'] = '0'  # Read-only worktree provenance probes.
HELPER = ROOT / 'docs/diagnostics/compile-pointwise-positional/warm-dispatch/gpu-dispatch.py'
spec = importlib.util.spec_from_file_location('dispatch_helpers', HELPER)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
# Different irregular sizes from the original broadcast diagnostic. Each group
# stays within unchanged default specialization limits and keeps its wrappers.
CASES = (
    ('product_left', 'x*y+x', [((239,1),(239,271)), ((239,271),(239,1)), ((257,1),(257,263))]),
    ('product_right', 'x+x*y', [((255,),(1,255)), ((1,513),(513,)), ((1025,1),(1,3))]),
    ('square_left', 'x*x+y', [((7,1),(3,1,13)), ((3,1,13),(7,1)), ((11,1,17),(1,19,1))]),
    ('square_right', 'y+x*x', [((),(17,)), ((17,),()), ((1,),(1,1)), ((0,1),(1,13)), ((1,13),(0,1))]),
    ('repeated_unused', 'x*x+x', [((7,),(11,)), ((7,),(0,13)), ((7,),()), ((7,),(7,))]),
    ('identities', 'y+x*y', [((17,),(17,)), ((17,),(17,)), ((17,),(17,))]),
)
VALUES = (0., -0., 1e-45, -1e-45, 1e-38, -1e-38, 2e38, -2e38,
          float('inf'), -float('inf'), float('nan'), 4096., -4098.,
          1.0000001192092896, -0.9999999403953552, -1.5, .713, -1.137, 3e38, -3e38, 1.9e19, -1.9e19)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    helpers.write(path, value)


def tensor_record(tensor):
    return helpers.tensor_record(tensor)


def digest(record):
    return sha(json.dumps(record, sort_keys=True, allow_nan=False).encode())


def source_record():
    paths = helpers.command('git', 'ls-files', '--', 'src', 'python', 'Cargo.toml',
                            'Cargo.lock', 'pyproject.toml', 'uv.lock', 'rust-toolchain.toml').splitlines()
    return {'head': helpers.command('git', 'rev-parse', 'HEAD'),
            'status': helpers.command('git', 'status', '--short'),
            'files': {p: sha((ROOT/p).read_bytes()) for p in paths}}


def make_inputs(fw, shapes, changed, alias):
    args, bases = [], []
    for index, shape in enumerate(shapes):
        values = [VALUES[(i+index*7+changed*3) % len(VALUES)] for i in range(math.prod(shape))]
        offset = 1+changed
        base = fw.tensor([91.]*offset+values+[92.], dtype=fw.float32).to('cuda:0')
        bases.append(base)
        args.append(base[offset:offset+len(values)].reshape(shape))
    if alias:
        args[1] = args[0]
    return tuple(args), bases


@contextlib.contextmanager
def native_guard(fw, fn, calls):
    bridge = importlib.import_module('torch_rs.torch_rs')
    trace = importlib.import_module('torch_rs._compile_trace')
    def reject(*args, **kwargs):
        raise AssertionError('eager or per-node replay')
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code is fn.__code__:
            raise AssertionError('original body replay')
        if event == 'c_call':
            name = getattr(arg, '__name__', '')
            if name in ('sin','cos','relu','__add__','__sub__','__mul__','__neg__'):
                reject()
            if name == 'run' and isinstance(getattr(arg, '__self__', None), bridge._PointwiseKernel):
                calls.append(arg.__self__)
    previous = sys.getprofile()
    with contextlib.ExitStack() as stack:
        for owner, names in ((fw, ('_execute_native_eager_compile_graph',)),
                             (trace, ('_execute_operation',)),
                             (bridge, ('_compile_trace_cuda_graph','_compile_trace_unary',
                                       '_compile_trace_binary','_compile_trace_scalar'))):
            for name in names:
                stack.enter_context(patch.object(owner, name, reject))
        sys.setprofile(profile)
        try:
            yield
        finally:
            sys.setprofile(previous)


def history(fw, implementation, expression, shapes, row, directory, synchronize):
    namespace = {}
    exec('def f(x,y):\n return '+expression, namespace)
    fn = namespace['f']
    compiled = fw.compile(fn)  # Untouched public defaults, including autotuning.
    row['states'] = []
    for index, pair in enumerate(shapes):
        for changed in (0,1):
            alias = row['name'] == 'identities' and index == 1
            args, bases = make_inputs(fw, pair, changed, alias)
            inputs, storage = [tensor_record(t) for t in args], [tensor_record(t) for t in bases]
            state = {'shapes': pair, 'changed': changed, 'aliased': alias, 'inputs': inputs, 'calls': []}
            row['states'].append(state)
            previous = None
            for phase, count in (('cold',1), ('warmup',5), ('sample',17)):
                for sample in range(count):
                    actual_calls = []
                    guard = native_guard(fw,fn,actual_calls) if implementation == 'native' else contextlib.nullcontext()
                    with guard:
                        synchronize()
                        start = time.perf_counter_ns()
                        output = compiled(*args)
                        synchronize()
                        elapsed = time.perf_counter_ns()-start
                    observed = tensor_record(output)  # Materialize outside timing.
                    record = {'phase':phase, 'sample':sample, 'elapsedNs':elapsed, 'outputSha256':digest(observed)}
                    state['calls'].append(record)
                    if phase == 'cold':
                        state['output'] = observed
                    if observed != state['output']:
                        record['changedOutput'] = observed
                        raise AssertionError('warm values changed')
                    if output.numel():
                        assert output.data_ptr() not in [t.data_ptr() for t in args]
                        if previous is not None:
                            assert output.data_ptr() != previous.data_ptr()
                    assert output is not previous
                    previous = output
                    assert [tensor_record(t) for t in args] == inputs
                    assert [tensor_record(t) for t in bases] == storage
                    if implementation == 'native':
                        selected = next(reversed(compiled._torch_rs_pointwise_cache.executors.values()))
                        assert actual_calls == [selected], 'must capture the module actually dispatched'
                        assert selected.source.count('fmaf(') == 1
                        assert selected.ptx.count('.visible .entry') == 1
                        assert selected.ptx.count('fma.rn.f32') == 1
                        key = sha(selected.source.encode())
                        (directory/f'{key}.cu').write_text(selected.source)
                        ptx_key = sha(selected.ptx.encode())
                        (directory/f'{ptx_key}.ptx').write_text(selected.ptx)
                        record['kernelSourceSha256'] = key
                        record['kernelPtxSha256'] = ptx_key
                        state['nvrtc'] = list(selected.nvrtc_version)
                        state['options'] = selected.options
    # Both sides finish the complete history before the same reset boundary.
    fw.compiler.reset()


def verify_build(path, source):
    path = helpers.inside(path)
    build = helpers.read(path)
    assert build['passed'], 'build failed'
    assert build['source'] == source['files'], 'build source differs from checkout'
    wheel = helpers.inside(build['wheel']['path'])
    assert sha(wheel.read_bytes()) == build['wheel']['sha256']
    site = helpers.inside(sysconfig.get_paths()['purelib'])
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('torch_rs/') and name.endswith(('.py','.so')):
                installed = helpers.inside(site/name)
                assert installed.read_bytes() == archive.read(name), f'stale install: {name}'
    return {'path':str(path), 'sha256':sha(path.read_bytes()), 'wheel':build['wheel'],
            'extension':build['extension']}


def build(directory):
    """Build/install a fresh local wheel and retain its source and command logs."""
    directory = helpers.inside(directory)
    directory.mkdir()
    before = source_record()
    record = {'source':before['files'], 'head':before['head'], 'status':before['status'],
              'kind':'development source capture; not a clean candidate/main evaluation',
              'python':sys.executable, 'commands':[], 'passed':False}
    try:
        for name in ('CARGO_HOME','UV_CACHE_DIR','TMPDIR','CUDA_CACHE_PATH'):
            helpers.inside(os.environ[name])
        with tarfile.open(directory/'source.tar.gz','w:gz') as archive:
            for path in before['files']:
                archive.add(ROOT/path,arcname=path)
        environment = dict(os.environ, CARGO_TARGET_DIR=str(directory/'cargo-target'),
                           VIRTUAL_ENV=sys.prefix, PYO3_PYTHON=sys.executable)
        def run(name, command):
            with (directory/(name+'.log')).open('w') as log:
                result = subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
            record['commands'].append({'command':command,'returncode':result.returncode})
            assert result.returncode == 0, name+' failed; see retained log'
        run('build',[str(helpers.inside(Path(sys.prefix)/'bin/maturin')),'build','--release',
                     '--locked','--offline','--out',str(directory/'wheels')])
        wheels = list((directory/'wheels').glob('*.whl'))
        assert len(wheels) == 1
        wheel = wheels[0]
        run('install',['uv','pip','install','--python',sys.executable,'--no-deps',
                       '--force-reinstall',str(wheel)])
        assert source_record()['files'] == before['files'], 'source changed during build'
        extension = importlib.import_module('torch_rs.torch_rs')
        record.update(wheel={'path':str(wheel),'sha256':sha(wheel.read_bytes())},
                      extension={'path':str(helpers.inside(extension.__file__)),
                                 'sha256':sha(Path(extension.__file__).read_bytes())},
                      rustc=helpers.command('rustc','--version'), nvcc=helpers.command('nvcc','--version'),
                      profile='release; thin LTO; codegen-units=1; extension-module; abi3-py310')
        # verify_build checks every installed Python/extension byte against wheel.
        record['passed'] = True
        write(directory/'build.json',record)
        verify_build(directory/'build.json',before)
        record['passed'] = True
    except Exception:
        record['passed'] = False
        record['failure'] = traceback.format_exc()
    finally:
        write(directory/'build.json',record)
    print(json.dumps({'buildPassed':record['passed']}))
    return record['passed']


def leg(implementation, directory, build_record):
    assert implementation in ('native','reference')
    directory = helpers.inside(directory)
    directory.mkdir()  # Never reuse a process cache or overwrite prior evidence.
    record = {'implementation':implementation, 'cases':[], 'passed':False}
    try:
        helpers.environment(ROOT, directory)
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
        record.update(source=source_record(), scriptSha256=sha(Path(__file__).read_bytes()),
                      helperSha256=sha(HELPER.read_bytes()), python=sys.executable,
                      startedAt=helpers.command('date','-u','+%FT%TZ'), gpuBefore=helpers.inventory())
        record['build'] = verify_build(build_record,record['source'])
        physical = next(line for line in record['gpuBefore'].splitlines() if line.split(',')[0].strip() == '0')
        record['gpuUuid'] = physical.split(',')[1].strip()
        helpers.gpu_identity(record['gpuUuid'])
        fw = importlib.import_module('torch_rs' if implementation == 'native' else 'torch')
        fw.set_num_threads(1)
        assert fw.cuda.is_available() and fw.cuda.device_count() == 1
        if implementation == 'reference':
            assert fw.__version__ == '2.13.0+cu130'
        record.update(frameworkPath=str(helpers.inside(fw.__file__)), frameworkVersion=fw.__version__,
                      frameworkSha256=sha(Path(fw.__file__).read_bytes()))
        if implementation == 'reference':
            record['referenceExtension'] = {'path':str(helpers.inside(fw._C.__file__)),
                                            'sha256':sha(Path(fw._C.__file__).read_bytes())}
            record['referenceCuda'] = fw.version.cuda
        if implementation == 'native':
            extension = importlib.import_module('torch_rs.torch_rs')
            record['extension'] = {'path':str(helpers.inside(extension.__file__)), 'sha256':sha(Path(extension.__file__).read_bytes())}
            assert record['extension'] == record['build']['extension']
        synchronize = helpers.CudaSynchronizer()
        record['synchronization'] = synchronize.metadata
        for name, expression, shapes in CASES:
            row = {'name':name,'expression':expression,'passed':False}
            record['cases'].append(row)
            try:
                history(fw,implementation,expression,shapes,row,directory,synchronize)
                row['passed'] = True
            except Exception:
                row['failure'] = traceback.format_exc()
        if implementation == 'native':
            assert 'torch' not in sys.modules
        record['runtime'] = helpers.runtime_identity()
        assert source_record()['files'] == record['source']['files'], 'source changed during capture'
        record['passed'] = all(row['passed'] for row in record['cases'])
    except Exception:
        record['failure'] = traceback.format_exc()
    finally:
        for key, probe in (('gpuAfter',helpers.inventory),
                           ('finishedAt',lambda: helpers.command('date','-u','+%FT%TZ'))):
            try:
                record[key] = probe()
            except Exception:
                record[key+'Failure'] = traceback.format_exc()
                record['passed'] = False
        write(directory/'report.json.gz',record)
    print(json.dumps({'passed':record['passed'], 'cases':[(r['name'],r['passed']) for r in record['cases']]}))
    return record['passed']


def compare(left, right, output):
    output = helpers.inside(output)
    assert not output.exists(), 'refusing to overwrite comparison evidence'
    left, right = helpers.inside(left), helpers.inside(right)
    result = {'passed':False, 'comparisons':[], 'failures':[], 'reports':[str(left),str(right)]}
    try:
        result['reportSha256'] = [sha(left.read_bytes()),sha(right.read_bytes())]
        left, right = helpers.read(left), helpers.read(right)
        assert {left['implementation'],right['implementation']} == {'native','reference'}
        assert left['passed'] and right['passed'], 'leg execution failed; raw failures retained'
        assert left['scriptSha256'] == right['scriptSha256']
        assert left['build'] == right['build']
        assert left['gpuUuid'] == right['gpuUuid']
        assert left['synchronization'] == right['synchronization']
        assert left['source']['files'] == right['source']['files']
        assert len(left['cases']) == len(right['cases']) == len(CASES)
        for a,b in zip(left['cases'],right['cases'],strict=True):
            assert a['name'] == b['name']
            for index,(x,y) in enumerate(zip(a['states'],b['states'],strict=True)):
                try:
                    for key in ('shapes','changed','aliased','inputs','output'):
                        # float.hex preserves zero signs, every finite bit, and
                        # nonfinite classes; NaN payloads intentionally coalesce.
                        assert x[key] == y[key], f'{key} mismatch'
                    assert len(x['calls']) == len(y['calls']) == 23
                    assert all(c['outputSha256'] == digest(x['output']) for c in x['calls']+y['calls'])
                    result['comparisons'].append([a['name'],index])
                except Exception:
                    result['failures'].append({'case':a['name'],'state':index,'failure':traceback.format_exc()})
        result['passed'] = not result['failures']
    except Exception:
        result['failures'].append({'failure':traceback.format_exc()})
    write(helpers.inside(output), result)
    print(json.dumps(result))
    return result['passed']


if __name__ == '__main__':
    if sys.argv[1] == '--build':
        passed = build(Path(sys.argv[2]))
    elif sys.argv[1] == '--compare':
        passed = compare(*map(Path,sys.argv[2:]))
    else:
        passed = leg(sys.argv[1],Path(sys.argv[2]),Path(sys.argv[3]))
    raise SystemExit(0 if passed else 1)
