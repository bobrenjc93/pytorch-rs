"""Fixed, non-scoring admission comparison; each command preserves its attempt."""
import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location('program_identity_common', HERE.parent / 'program-identity/consumer.py')
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)
PROTOCOL = 'bf9404943317343aca37258c26ecc2a3b921745892257e2fac0a651f9f625e11'
BASE = '8006d7e05636b3322da86ae89c41040900ceee00'
ORDER = ('B', 'R', 'C', 'C', 'R', 'B')
CASES = (
    ('unaryflat', (509, 32749), 'def f(x):\n return (x.sin()+x*0.125).relu()\n'),
    ('binaryflat', (509, 32749), 'def f(x,y):\n return (x+y)*0.75-x*0.125\n'),
    ('nested', (509, 32749), "def f(data):\n x,y=data['pair']\n p=x*y\n return {'out':(p+x,-p),'alias':x}\n"),
    ('heldout', (1153, 49157), 'def f(x,y):\n p=(x-y).cos()\n return (p*x+y).sin()\n'),
)
require, sha, encode, leaves = common.require, common.sha, common.encode, common.leaves


def bindings():
    require(sha((HERE / 'protocol.md').read_bytes()) == PROTOCOL, 'Frozen protocol changed')
    return {'protocol': PROTOCOL, 'consumer': sha(Path(__file__).read_bytes()),
            'common': sha(Path(common.__file__).read_bytes()), 'cases': sha(json.dumps(CASES).encode())}


def source_files(root):
    paths = [root / name for name in ('Cargo.toml', 'Cargo.lock', 'pyproject.toml', 'rust-toolchain.toml')]
    paths += list((root / 'src').rglob('*.rs')) + list((root / 'src/cuda').rglob('*'))
    paths += list((root / 'python').rglob('*.py'))
    return {str(path.relative_to(root)): sha(path.read_bytes()) for path in sorted(set(paths)) if path.is_file()}


def source(root, directory):
    """Snapshot an owned checkout without rewriting its dirty/commit identity."""
    require(Path(common.command('git', 'rev-parse', '--show-toplevel', cwd=root)).resolve() == root,
            'Source root must own its checkout')
    result = {'root': str(root), 'commit': common.command('git', 'rev-parse', 'HEAD', cwd=root),
              'tree': common.command('git', 'rev-parse', 'HEAD^{tree}', cwd=root),
              'status': common.command('git', 'status', '--porcelain=v1', '--untracked-files=all', cwd=root),
              'files': source_files(root)}
    (directory / 'source.patch').write_text(common.command('git', 'diff', 'HEAD', '--', 'src', 'python',
                                                        'Cargo.toml', 'pyproject.toml', cwd=root))
    for name in result['files']:
        target = directory / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    return result


def installed(args, observed_source):
    """Bind the installed Python/native bytes to the supplied successful build."""
    wheel = common.inside(args.wheel)
    build = common.read(common.inside(args.build_record))
    require(build['passed'] and build['sourceFiles'] == observed_source['files'], 'Build/source mismatch')
    require(build['sourceRoot'] == observed_source['root'], 'Build source root mismatch')
    require(build['wheel'] == str(wheel) and build['wheelSha256'] == sha(wheel.read_bytes()), 'Build/wheel mismatch')
    package = common.inside(Path(importlib.util.find_spec('torch_rs').origin).parent, Path(sys.prefix).resolve())
    files = common.wheel_files(wheel)
    require(any(name.endswith('.so') for name in files), 'Missing native extension')
    for name, data in files.items():
        require((package.parent / name).read_bytes() == data, 'Installed/wheel mismatch: ' + name)
        if name.endswith('.py'):
            require((Path(observed_source['root']) / 'python' / name).read_bytes() == data, 'Python/source mismatch: ' + name)
    return {'package': str(package), 'wheel': str(wheel), 'wheelSha256': build['wheelSha256'],
            'buildRecord': str(Path(args.build_record).resolve()), 'buildRecordSha256': sha(Path(args.build_record).read_bytes()),
            'files': {name: sha(data) for name, data in files.items()}}


def inputs(framework, name, size, phase):
    def tensor(y=False):
        return framework.tensor(common.values((size,), phase, y), dtype=framework.float32).to('cuda:0')
    x = tensor()
    if name == 'unaryflat':
        return (x,)
    y = tensor(True)
    return ({'pair': (x, y)},) if name == 'nested' else (x, y)


def matrix(framework, sync, record):
    """Run the frozen families and owner lifetimes, retaining partial rows on error."""
    record.update(cells=[], churn=[])
    for name, sizes, text in CASES:
        compiled, factory_ns = common.factory(framework, text, name)
        for size in sizes:
            row = {'family': name, 'size': size, 'factoryNs': factory_ns, 'samples': [], 'warmups': [], 'passed': False}
            record['cells'].append(row)
            arguments = [inputs(framework, name, size, phase) for phase in (0, 1, 2, 3, 4, 5, *range(8, 25))]
            row['inputs'] = [encode(args, leaves(args)) for args in arguments]
            result, row['firstNs'] = common.timed_call(compiled, arguments[0], sync)
            row['first'] = encode(result, leaves(arguments[0]))
            del result
            for phase in range(1, 6):
                result = compiled(*arguments[phase])
                sync()
                del result
                row['warmups'].append(phase)
            results = [None] * 17
            for index in range(17):
                results[index], elapsed = common.timed_call(compiled, arguments[index + 6], sync)
                row['samples'].append(elapsed)
            row['ownership'] = common.output_check(results, arguments[6:])
            row['outputs'] = [encode(result, leaves(args)) for result, args in zip(results, arguments[6:])]
            require(row['inputs'] == [encode(args, leaves(args)) for args in arguments], 'Input mutation')
            if record['implementation'] == 'native':
                row['compilerExecutors'] = [{'options': list(executor.options), 'nvrtcVersion': list(executor.nvrtc_version)}
                    for executor in compiled._torch_rs_pointwise_cache.executors.values()]
            row.update(statistics=common.summary(row['samples']), passed=True)
            del results, arguments
        args = inputs(framework, name, sizes[0], 25)
        entry = {'family': name, 'size': sizes[0], 'phase': 25, 'input': encode(args, leaves(args))}
        record['churn'].append(entry)
        result, entry['elapsedNs'] = common.timed_call(compiled, args, sync)
        entry['output'] = encode(result, leaves(args))
        require(entry['input'] == encode(args, leaves(args)), 'Churn input mutation')
        del result, args, compiled


def worker(args, record, directory):
    """Execute one fresh-process leg with pinned libraries and local provenance."""
    common.environment(directory)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Only GPU0 is admitted')
    runtime, nvrtc = common.library_file(args.runtime), common.library_file(args.nvrtc)
    os.environ.update(TORCH_RS_CUDART=str(runtime), TORCH_RS_NVRTC=str(nvrtc))
    root = common.inside(args.source_root)
    record.update(index=args.index, label=ORDER[args.index], implementation='reference' if ORDER[args.index] == 'R' else 'native',
                  source=source(root, directory), gpu=common.GPU, gpuBefore=common.gpu_identity())
    if record['label'] == 'B':
        require(record['source']['commit'] == BASE and not record['source']['status'], 'B must be clean rejected source')
    record['installed'] = installed(args, record['source'])
    compiler = common.CompilerPin(nvrtc)
    sync = common.Synchronizer(runtime)
    record.update(compiler=compiler.metadata, synchronization=sync.metadata)
    framework = importlib.import_module('torch' if record['label'] == 'R' else 'torch_rs')
    record.update(frameworkPath=str(common.inside(framework.__file__, Path(sys.prefix).resolve())),
                  frameworkVersion=framework.__version__, frameworkSha256=sha(Path(framework.__file__).read_bytes()))
    if record['label'] == 'R':
        require(framework.__version__ == '2.13.0+cu130', 'Wrong reference version')
        identity = 'GPU-' + str(framework.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
        require(identity == common.GPU, 'Reference device differs')
        record['referenceUuid'] = identity
        extension = common.inside(framework._C.__file__, Path(sys.prefix).resolve())
        record['referenceExtension'] = {'path': str(extension), 'sha256': sha(extension.read_bytes())}
    framework.set_num_threads(1)
    require(framework.cuda.is_available() and framework.cuda.device_count() == 1, 'Expected one CUDA device')
    record['environment'] = {key: os.environ.get(key) for key in (
        'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TORCH_RS_CUDART', 'TORCH_RS_NVRTC',
        'TMPDIR', 'XDG_CACHE_HOME', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'TORCH_HOME')}
    record['libraries'] = common.libraries()
    common.validate_libraries(record)
    matrix(framework, sync, record)
    record['libraries'] = common.libraries()
    common.validate_libraries(record)
    require(source_files(root) == record['source']['files'], 'Source changed during measurement')
    require(record['label'] == 'R' or 'torch' not in sys.modules, 'Native forwarded/imported reference')


def verify_records(reports):
    """Reject incomplete or incomparable legs before deriving non-scoring ratios."""
    require(len(reports) == 6, 'Exactly six ordered legs required')
    expected = [(name, size) for name, sizes, _ in CASES for size in sizes]
    for index, report in enumerate(reports):
        require(report['passed'] and report['index'] == index and report['label'] == ORDER[index], 'Failed or reordered leg')
        require(report['bindings'] == bindings(), 'Protocol/consumer bindings changed')
        require(report['gpu'] == common.GPU and report['gpuBefore'] and report['gpuAfter'], 'GPU evidence missing')
        require(report['interpreter'].startswith(str(ROOT)) and report['source']['root'].startswith(str(ROOT)), 'Foreign paths')
        common.validate_libraries(report)
        require(report['environment']['OMP_NUM_THREADS'] == '1'
                and report['environment']['MKL_NUM_THREADS'] == '1', 'Expected one host thread')
        require(report['synchronization']['api'] == 'cudaDeviceSynchronize'
                and report['synchronization']['logicalDevice'] == 0, 'Different completion barrier/device')
        for key in ('path', 'sha256', 'version'):
            require(report['synchronization'][key] == reports[0]['synchronization'][key], 'Different runtime')
            require(report['compiler'][key] == reports[0]['compiler'][key], 'Different compiler')
        require([(row['family'], row['size']) for row in report['cells']] == expected, 'Incomplete matrix')
        require([(row['family'], row['size'], row['phase']) for row in report['churn']] ==
                [(name, sizes[0], 25) for name, sizes, _ in CASES], 'Wrong churn history')
        for row in report['cells']:
            require(row['passed'] and row['warmups'] == list(range(1, 6)) and len(row['outputs']) == 17
                    and len(row['inputs']) == 23 and row['firstNs'] > 0, 'Incomplete cell')
            require(row['statistics'] == common.summary(row['samples']), 'Invalid summary')
            ownership = row['ownership']
            require(set(ownership) == {'retainedOutputs', 'computedPointers', 'inputPointers'}
                    and type(ownership['retainedOutputs']) is int and ownership['retainedOutputs'] == 17,
                    'Missing retained output ownership')
            computed, inputs = ownership['computedPointers'], ownership['inputPointers']
            require(type(computed) is list and type(inputs) is list
                    and all(type(pointer) is int and pointer > 0 for pointer in computed + inputs),
                    'Invalid storage pointer evidence')
            require(len(computed) == (34 if row['family'] == 'nested' else 17)
                    and len(inputs) == (17 if row['family'] == 'unaryflat' else 34)
                    and len(set(computed)) == len(computed) and sorted(set(inputs)) == inputs
                    and set(computed).isdisjoint(inputs), 'Retained storage ownership mismatch')
        if index:
            require(reports[index - 1]['finished'] <= report['started'], 'Overlapping or reversed processes')
    require(reports[0]['source']['commit'] == BASE and not reports[0]['source']['status'], 'Incorrect B provenance')
    for path in ('python/torch_rs/_compile_pointwise.py', 'src/python_pointwise.rs'):
        require(reports[0]['source']['files'][path] != reports[2]['source']['files'][path],
                'Candidate must contain the admitted native admission change')
    for left, right in ((0, 5), (1, 4), (2, 3), (1, 2)):
        for key in ('source', 'installed', 'interpreter', 'interpreterSha256'):
            require(reports[left][key] == reports[right][key], 'Changed source/build/environment identity')
    for report in (reports[1], reports[4]):
        require(report['frameworkVersion'] == '2.13.0+cu130' and report['referenceUuid'] == common.GPU, 'Wrong reference')
        require(report['referenceExtension'] == reports[1]['referenceExtension'], 'Changed reference build')
    ratios = []
    for order, (bi, ri, ci) in enumerate(((0, 1, 2), (5, 4, 3))):
        baseline, reference, candidate = (reports[i] for i in (bi, ri, ci))
        for b, r, c in zip(baseline['cells'], reference['cells'], candidate['cells']):
            common.compare(b['inputs'], c['inputs'], True)
            common.compare(c['inputs'], r['inputs'], True)
            for field in ('first', 'outputs'):
                common.compare(b[field], c[field], True)
                common.compare(c[field], r[field])
            ratios.append({'order': order, 'family': c['family'], 'size': c['size'],
                           'baselineOverCandidate': b['statistics']['median'] / c['statistics']['median'],
                           'referenceOverCandidate': r['statistics']['median'] / c['statistics']['median'],
                           'firstNs': {'B': b['firstNs'], 'R': r['firstNs'], 'C': c['firstNs']},
                           'statistics': {'B': b['statistics'], 'R': r['statistics'], 'C': c['statistics']}})
        for b, r, c in zip(baseline['churn'], reference['churn'], candidate['churn']):
            for field in ('input', 'output'):
                common.compare(b[field], c[field], True)
                common.compare(c[field], r[field], field == 'input')
            require(all(row['elapsedNs'] > 0 for row in (b, r, c)), 'Invalid churn time')
    return {'ratios': ratios,
            'churn': [{'index': i, 'label': ORDER[i], 'calls': [
                {key: call[key] for key in ('family', 'size', 'elapsedNs')} for call in report['churn']]}
                for i, report in enumerate(reports)], 'qualificationOrScoreProduced': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    run = commands.add_parser('worker')
    run.add_argument('--index', type=int, choices=range(6), required=True)
    for name in ('source-root', 'wheel', 'build-record', 'runtime', 'nvrtc'):
        run.add_argument('--' + name, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('reports', nargs=6)
    for command in (run, verify):
        command.add_argument('--output', required=True)
    args = parser.parse_args()
    directory = common.inside(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    record = {'passed': False, 'kind': args.kind, 'command': sys.argv, 'started': datetime.now(timezone.utc).isoformat(),
              'interpreter': sys.executable, 'interpreterSha256': sha(Path(sys.executable).read_bytes())}
    try:
        record['bindings'] = bindings()
        if args.kind == 'worker':
            worker(args, record, directory)
        else:
            record['reports'] = [{'path': str(common.inside(path)), 'sha256': sha(Path(path).read_bytes())} for path in args.reports]
            record.update(verify_records([common.read(path) for path in args.reports]))
        record['passed'] = True
    except BaseException:
        record['error'] = traceback.format_exc()
        print(record['error'], file=sys.stderr)
    finally:
        if args.kind == 'worker':
            for key, query in (('gpuAfter', common.inventory), ('libraries', common.libraries)):
                try:
                    record[key] = query()
                except BaseException:
                    record['passed'] = False
                    record[key + 'Error'] = traceback.format_exc()
        record['finished'] = datetime.now(timezone.utc).isoformat()
        common.write(directory / 'report.json.gz', record)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
