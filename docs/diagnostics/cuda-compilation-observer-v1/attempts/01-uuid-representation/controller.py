#!/usr/bin/env python3
"""Pin, reproduce and audit the independent six-slot CUDA observer campaign.

This orchestrator does not implement operations or change evaluator accounting.
Only `run` needs CUDA; `validate` and the unit tests are portable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path('scripts/campaigns/cuda-compilation-observer-v1.json')
CAMPAIGN_FILES = [MANIFEST, Path('scripts/cuda_compilation_campaign.py'),
                  Path('scripts/evaluate_cuda_compilation.py'),
                  Path('tests/test_cuda_compilation_evaluator.py'),
                  Path('tests/test_cuda_observer_campaign.py')]
MANAGED = {'README.md', 'docs/burner-evaluation-history.json',
           'docs/burner-evaluation-progress.svg'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read(path):
    def reject(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True)


def pins(root, manifest):
    actual = {p: sha(root / p) for p in manifest['observer_files']}
    require(actual == manifest['observer_files'], 'observer/common/matrix hash mismatch; no overlay allowed')
    return actual


def production(root, manifest, role):
    expected = manifest[role]['production']
    files = {p: sha(root / p) for p in expected['files']}
    require(files == expected['files'], role + ' production bytes changed')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    require(digest == expected['sha256'], role + ' production digest mismatch')
    return digest


def status(root):
    return {'commit': git(root, 'rev-parse', 'HEAD').strip(),
            'status': git(root, 'status', '--short', '--untracked-files=all'),
            'diff_stat': git(root, 'diff', 'HEAD', '--stat')}


def allowed_baseline_path(path):
    return (path in MANAGED or Path(path) in CAMPAIGN_FILES
            or path == 'docs/cuda-compilation-observer-campaign.md'
            or path.startswith('docs/diagnostics/cuda-compilation-observer-v1/'))


def baseline_scope(root, manifest):
    base = manifest['baseline']['production_commit']
    changed = git(root, 'diff', '--name-only', base, 'HEAD').splitlines()
    require(all(allowed_baseline_path(p) for p in changed),
            'baseline contains changes outside the evaluator-only campaign: ' + repr(changed))
    return changed


def load_manifest():
    manifest = read(ROOT / MANIFEST)
    require(manifest['schema_version'] == 1, 'unknown campaign schema')
    pins(ROOT, manifest)
    sys.path.insert(0, str(ROOT / 'scripts'))
    import evaluate_cuda_compilation as evaluator
    require(manifest['case_set'] == evaluator.corpus(), 'fixed case set changed')
    require(manifest['options'] == evaluator.OPTIONS, 'compile options changed')
    require(manifest['denominator'] == len(evaluator.corpus()['cases']) == 6, 'six slots required')
    require(len(manifest['seed_selection']['seeds']) == 2
            and evaluator.common.valid_seeds(manifest['seed_selection']['seeds']), 'invalid seeds')
    return manifest, evaluator


def inventory():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version,compute_cap',
        '--format=csv'], text=True)


def check_gpu(snapshot, manifest):
    rows = [line.split(', ') for line in snapshot.strip().splitlines()[1:]]
    zero = next(row for row in rows if row[0] == '0')
    require(zero[1] == manifest['device']['uuid'] and zero[2] == 'NVIDIA H100', 'GPU0 identity mismatch')


def command(root, evidence, name, argv, env):
    """Retain exact commands, all stdout/stderr and failed attempts."""
    start = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    result = subprocess.run([str(a) for a in argv], cwd=root, env=env,
                            capture_output=True, text=True)
    (evidence / (name + '.stdout')).write_text(result.stdout)
    (evidence / (name + '.stderr')).write_text(result.stderr)
    dump(evidence / (name + '.command.json'), {
        'argv': [str(a) for a in argv], 'cwd': str(root), 'started_at': start,
        'elapsed_seconds': time.monotonic() - clock, 'returncode': result.returncode,
        'environment': {k: env[k] for k in sorted(env) if k in {
            'CUDA_VISIBLE_DEVICES', 'TMPDIR', 'XDG_CACHE_HOME', 'UV_CACHE_DIR',
            'UV_PYTHON_INSTALL_DIR', 'CARGO_HOME', 'CARGO_TARGET_DIR', 'PYO3_PYTHON',
            'VIRTUAL_ENV', 'TORCH_RS_CUDART', 'TORCH_RS_CUBLAS', 'PATH',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'RUSTUP_AUTO_INSTALL', 'PYTHONDONTWRITEBYTECODE', 'LD_LIBRARY_PATH'}}})
    require(result.returncode == 0, f'{name} failed ({result.returncode}); see {evidence}')
    return result.stdout


def local_environment(root):
    cache = root / 'target/campaign-cache'
    cache.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k not in {
        'PYTHONPATH', 'PYTHONHOME', 'CONDA_PREFIX', 'VIRTUAL_ENV', 'LD_LIBRARY_PATH',
        'TORCH_RS_CUDART', 'TORCH_RS_CUBLAS', 'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS'}}
    env.update(CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
               RUSTUP_AUTO_INSTALL='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TMPDIR=str(cache), XDG_CACHE_HOME=str(cache),
               UV_CACHE_DIR=str(cache / 'uv'), UV_PYTHON_INSTALL_DIR=str(cache / 'python'),
               CARGO_HOME=str(cache / 'cargo'), CARGO_TARGET_DIR=str(root / 'target/build'),
               VIRTUAL_ENV=str(root / '.venv'), PYO3_PYTHON=str(root / '.venv/bin/python'),
               CUDA_CACHE_PATH=str(cache / 'cuda'), TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
               TRITON_CACHE_DIR=str(cache / 'triton'))
    return env


def checkout(commit, root, evidence, env):
    # Own .git and copied objects. No `git worktree` metadata in the parent.
    root.mkdir()
    command(ROOT, evidence, 'git-init', ['git', 'init', root], env)
    command(root, evidence, 'git-fetch', ['git', 'fetch', '--no-tags', str(ROOT), commit], env)
    # Never materialize or edit even the tracked .burner files.
    command(root, evidence, 'git-sparse', ['git', 'sparse-checkout', 'set', '--no-cone', '/*', '!/.burner/'], env)
    command(root, evidence, 'git-checkout', ['git', 'checkout', '--detach', commit], env)
    require(status(root)['status'] == '', 'new checkout is not clean')


IDENTITY_CODE = r'''
import hashlib, importlib.metadata as m, json, pathlib, sys, torch
root=pathlib.Path.cwd()
local=root/'.venv'
assert pathlib.Path(sys.executable).resolve().is_relative_to(local)
assert pathlib.Path(torch.__file__).resolve().is_relative_to(local)
assert torch.__version__ == '2.13.0+cu130', torch.__version__
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == 'NVIDIA H100'
deps=[]
for d in m.distributions():
    path=pathlib.Path(d.locate_file('')).resolve()
    assert path.is_relative_to(local), path
    deps.append(dict(name=d.metadata['Name'],version=d.version,path=str(path)))
print(json.dumps(dict(executable=sys.executable, executable_resolved=str(pathlib.Path(sys.executable).resolve()),
    executable_sha256=hashlib.sha256(pathlib.Path(sys.executable).read_bytes()).hexdigest(),
    prefix=sys.prefix,base_prefix=sys.base_prefix,python=sys.version,torch=torch.__version__,
    torch_path=torch.__file__,cuda=torch.version.cuda,gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
    dependencies=sorted(deps,key=lambda d:d['name'].lower())),indent=2))
'''


def build_and_run(root, evidence, role, manifest, evaluator):
    env = local_environment(root)
    before = status(root)
    production_digest = production(root, manifest, role)
    pinned = pins(root, manifest)
    python = root / '.venv/bin/python'
    uv = shutil.which('uv')
    require(uv is not None, 'uv required')
    command(root, evidence, 'venv', ['/usr/bin/python3.12', '-I', '-B', '-m', 'venv',
                                    '--copies', '--without-pip', root / '.venv'], env)
    command(root, evidence, 'dependencies', [uv, 'sync', '--locked', '--no-install-project',
                                           '--group', 'reference', '--link-mode', 'copy'], env)
    # Library paths come from this checkout's installed locked dependencies.
    runtimes = list((root / '.venv').glob('lib/python*/site-packages/nvidia/cu13/lib/libcudart.so.13'))
    blas = list((root / '.venv').glob('lib/python*/site-packages/nvidia/cu13/lib/libcublas.so.13'))
    require(len(runtimes) == len(blas) == 1, 'local CUDA 13 runtime/cuBLAS missing')
    env.update(TORCH_RS_CUDART=str(runtimes[0]), TORCH_RS_CUBLAS=str(blas[0]))
    snapshot_before = inventory()
    check_gpu(snapshot_before, manifest)
    (evidence / 'gpu-before.csv').write_text(snapshot_before)
    identity = json.loads(command(root, evidence, 'runtime', [python, '-I', '-B', '-c', IDENTITY_CODE], env))
    require(identity['gpu_uuid'] == manifest['device']['uuid'], 'reference GPU UUID differs')
    dump(evidence / 'runtime.json', identity)
    build_argv = [root / '.venv/bin/maturin', 'build', '--release', '--locked',
                  '--interpreter', python, '--out', root / 'target/wheels']
    command(root, evidence, 'build', build_argv, env)
    wheels = list((root / 'target/wheels').glob('*.whl'))
    require(len(wheels) == 1, 'exactly one fresh wheel required')
    wheel = wheels[0]
    command(root, evidence, 'install', [uv, 'pip', 'install', '--python', python,
                                      '--no-deps', wheel], env)
    with zipfile.ZipFile(wheel) as archive:
        extensions = [n for n in archive.namelist() if n.startswith('torch_rs/') and n.endswith('.so')]
        require(len(extensions) == 1, 'wheel native extension missing')
        for name in archive.namelist():
            if name.startswith('torch_rs/') and name.endswith('.py'):
                require(archive.read(name) == (root / 'python' / name).read_bytes(), 'wheel/source mismatch')
        extension = root / 'python' / extensions[0]
        extension.write_bytes(archive.read(extensions[0]))
    tools = {name: command(root, evidence, name, [name, '--version'], env)
             for name in ('rustc', 'cargo', 'nvcc')}
    source = json.loads(command(root, evidence, 'source', [python, '-B', '-c',
        "import sys,json; sys.path.insert(0,'scripts'); import evaluate_cuda_math as e; print(json.dumps(e.source_provenance()))"], env))
    build = dict(source, extension_sha256=sha(extension), wheel_sha256=sha(wheel),
                 build_command=' '.join(map(str, build_argv)), **tools)
    dump(evidence / 'build.json', build)
    seeds = manifest['seed_selection']['seeds']
    argv = [python, '-B', root / 'scripts/evaluate_cuda_compilation.py',
            '--reference-python', python, '--candidate-python', python,
            '--seed', str(seeds[0]), '--seed', str(seeds[1]),
            '--timeout', str(manifest['timeout_seconds']), '--build-record', evidence / 'build.json',
            '--output', root / 'target/result.json']
    command(root, evidence, 'evaluate', argv, env)
    shutil.copyfile(root / 'target/result.json', evidence / 'result.json')
    # Same campaign-owned rejection controls against each interpreter; portable
    # controls do not invoke the HardwareTests class or modify candidate files.
    command(root, evidence, 'controls', [python, '-B', '-m', 'unittest', '-v',
        'test_cuda_compilation_evaluator.AccountingTests',
        'test_cuda_compilation_evaluator.IsolationTests'],
        dict(env, PYTHONPATH=str(ROOT / 'tests')))
    snapshot_after = inventory()
    check_gpu(snapshot_after, manifest)
    (evidence / 'gpu-after.csv').write_text(snapshot_after)
    after = status(root)
    require(before == after, 'checkout changed during build/run')
    require(pins(root, manifest) == pinned, 'observer changed during run')
    require(production(root, manifest, role) == production_digest, 'production changed during run')
    dump(evidence / 'checkout.json', dict(before=before, after=after, observer_files=pinned,
        production_sha256=production_digest, root=str(root),
        native_runtime_sha256=sha(runtimes[0]), native_cublas_sha256=sha(blas[0])))


def validate(directory, manifest, evaluator):
    """Audit retained raw values and fixed slots, never promote failures to credit."""
    receipt = read(directory / 'campaign.json')
    require(receipt['manifest_sha256'] == sha(ROOT / MANIFEST), 'manifest receipt mismatch')
    require(receipt['seeds'] == manifest['seed_selection']['seeds'], 'seed mismatch')
    require(receipt['mode'] in ('development', 'clean-commit'), 'unknown evidence mode')
    for name, digest in receipt['files'].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), 'evidence path escape')
        require(sha(path) == digest, 'evidence hash mismatch: ' + name)
    reports, counts = {}, {}
    for role in ('baseline', 'candidate'):
        folder = directory / role
        report, checkout_record = read(folder / 'result.json'), read(folder / 'checkout.json')
        reports[role] = report
        require(checkout_record['before'] == checkout_record['after'], 'dirty during run')
        if role == 'candidate' or receipt['mode'] == 'clean-commit':
            require(checkout_record['before']['status'] == '' and checkout_record['before']['diff_stat'] == '',
                    'clean evidence requires a clean checkout')
        commit = receipt['baseline_commit'] if role == 'baseline' else manifest['candidate']['checkout_commit']
        require(checkout_record['before']['commit'] == report['source']['commit'] == commit, 'commit mismatch')
        require(checkout_record['production_sha256'] == manifest[role]['production']['sha256'], 'production mismatch')
        require(checkout_record['observer_files'] == manifest['observer_files'], 'checkout observer mismatch')
        require(report['source_unchanged_during_run'] is True, 'unstable source')
        require(report['seeds'] == receipt['seeds'] and report['compile_options'] == manifest['options'], 'asymmetric inputs/options')
        require(report['CUDA_VISIBLE_DEVICES'] == '0', 'wrong device mask')
        for p, digest in manifest['observer_files'].items():
            require(report['file_sha256'][p] == digest, 'result observer mismatch')
        require(report['build_record'] == read(folder / 'build.json'), 'build receipt mismatch')
        for position in ('before', 'after'):
            check_gpu((folder / f'gpu-{position}.csv').read_text(), manifest)
        runtime = read(folder / 'runtime.json')
        require(runtime['gpu_uuid'] == manifest['device']['uuid'], 'runtime GPU mismatch')
        local = Path(checkout_record['root'])
        require(Path(runtime['executable_resolved']).is_relative_to(local / '.venv'), 'external interpreter')
        require(all(Path(d['path']).is_relative_to(local / '.venv') for d in runtime['dependencies']), 'external dependency')
        expected_keys = [(case['id'], seed) for case in manifest['case_set']['cases'] for seed in receipt['seeds']]
        require([(t['case_id'], t['seed']) for t in report['trials']] == expected_keys, 'missing/duplicate/reordered slots')
        score = evaluator.account(manifest['case_set'], receipt['seeds'], report['trials'],
                                  report['build_record'], report['source'])
        require(score == report['accounting'] and score['denominator'] == 6, 'accounting mismatch')
        independent = []
        raw_values = 0
        for case in manifest['case_set']['cases']:
            successes = []
            for trial in [t for t in report['trials'] if t['case_id'] == case['id']]:
                ref, native = trial['reference'], trial['candidate']
                require(evaluator.valid_execution(ref, case, trial['seed'], 'reference'), 'reference incomplete')
                require(ref['gpu']['uuid'] == manifest['device']['uuid'], 'worker GPU mismatch')
                require(Path(ref['package']).is_relative_to(local / '.venv'), 'external reference')
                require(Path(native['package']).is_relative_to(local / 'python'), 'external native package')
                require(native['extension']['sha256'] == report['build_record']['extension_sha256'], 'native binary mismatch')
                require(native['blocked_imports'] == [] and native['loaded_torch_modules'] == [], 'forwarding observed')
                for row in (ref, native):
                    require(Path(row['executable']).is_relative_to(local / '.venv'), 'external worker interpreter')
                    for runtime_lib in row['cuda_runtimes']:
                        require(Path(runtime_lib['path']).is_relative_to(local / '.venv'), 'external CUDA runtime')
                valid = evaluator.valid_execution(native, case, trial['seed'], 'candidate')
                correct = valid
                if valid:
                    for a, b in zip(ref['executions'], native['executions']):
                        require(a['inputs'] == b['inputs'], 'different reference/native inputs')
                        expected, actual = a['output']['values'], b['output']['values']
                        require(len(expected) == len(actual), 'truncated output')
                        raw_values += len(actual)
                        correct &= all(abs(x-y) <= manifest['case_set']['atol'] + manifest['case_set']['rtol']*abs(x)
                                       for x,y in zip(expected, actual))
                successes.append(bool(correct))
            independent.append(int(all(successes)))
        require(independent == [c['credit'] for c in score['cases']], 'independent raw accounting disagrees')
        counts[role] = dict(passed=sum(independent), denominator=6, slots=independent,
                            compared_native_output_values=raw_values)
    for a, b in zip(reports['baseline']['trials'], reports['candidate']['trials']):
        require([x['inputs'] for x in a['reference']['executions']] ==
                [x['inputs'] for x in b['reference']['executions']], 'revisions used different input values')
    for field in ('dependencies',):
        strip_paths = lambda r: [(d['name'], d['version']) for d in read(directory / r / 'runtime.json')[field]]
        require(strip_paths('baseline') == strip_paths('candidate'), 'dependency versions differ')
    return {'mode': receipt['mode'], 'results': counts,
            'expected_outcomes_observed': counts['baseline']['slots'] == [1,1,1,1,0,1]
                                        and counts['candidate']['slots'] == [1,1,1,1,1,1],
            'human_approval': False, 'independent_review': False, 'implementation_credit_claimed': False}


def run(args, manifest, evaluator):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'run requires CUDA_VISIBLE_DEVICES=0')
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / 'target') and not output.exists(), 'use a fresh directory under target/')
    require(len(args.baseline_commit) == 40, 'supply full baseline commit')
    output.mkdir(parents=True)
    env = local_environment(output)
    dump(output / 'selection.json', dict(manifest_sha256=sha(ROOT / MANIFEST),
        seeds=manifest['seed_selection']['seeds'], baseline_commit=args.baseline_commit,
        candidate_commit=manifest['candidate']['checkout_commit']))
    for role, commit in [('baseline', args.baseline_commit), ('candidate', manifest['candidate']['checkout_commit'])]:
        evidence = output / role
        evidence.mkdir()
        root = output / (role + '-checkout')
        checkout(commit, root, evidence, env)
        if role == 'baseline':
            baseline_scope(root, manifest)
            if args.development:
                require(commit == manifest['baseline']['production_commit'], 'development baseline must be main')
                for path in CAMPAIGN_FILES:
                    (root / path).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / path, root / path)
            else:
                for path in CAMPAIGN_FILES:
                    require(sha(ROOT / path) == sha(root / path), 'final campaign commit/files mismatch')
        pins(root, manifest)  # candidate is never overlaid
        build_and_run(root, evidence, role, manifest, evaluator)
    files = {str(p.relative_to(output)): sha(p) for role in ('baseline', 'candidate')
             for p in sorted((output / role).iterdir()) if p.is_file()}
    files['selection.json'] = sha(output / 'selection.json')
    dump(output / 'campaign.json', dict(schema_version=1, manifest_sha256=sha(ROOT / MANIFEST),
        mode='development' if args.development else 'clean-commit', baseline_commit=args.baseline_commit,
        seeds=manifest['seed_selection']['seeds'], files=files,
        completed_at=datetime.now(timezone.utc).isoformat(),
        independent_reviews=[], human_approval=False))
    result = validate(output, manifest, evaluator)
    dump(output / 'validation.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    runner = sub.add_parser('run')
    runner.add_argument('--baseline-commit', required=True)
    runner.add_argument('--output', type=Path, required=True)
    runner.add_argument('--development', action='store_true')
    validator = sub.add_parser('validate')
    validator.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    manifest, evaluator = load_manifest()
    if args.action == 'run':
        run(args, manifest, evaluator)
    else:
        print(json.dumps(validate(args.evidence.resolve(), manifest, evaluator), indent=2))


if __name__ == '__main__':
    main()
