"""Fixed paired validation; never changes the repository benchmark or scores."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = Path('/data/users/bobren/a/pytorch-rs-burner')
REFERENCE_PYTHON = REPO / '.venv/bin/python'
COMMITS = {'baseline': 'e4cddf0ecb45c23c683953fa2f8c45904b170a80',
           'candidate': 'c41a850446c0cd94e70027b13d197a9c8413ff15'}
ORDER = ['baseline', 'candidate', 'candidate', 'baseline', 'baseline', 'candidate']
CPU = 24 if 24 in os.sched_getaffinity(0) else min(os.sched_getaffinity(0))

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def output(args, cwd, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()

def environment(name):
    code = ROOT / name
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('PYO3_', 'PYTHON', 'CARGO_', 'UV_')) or key in (
                'VIRTUAL_ENV', 'CONDA_PREFIX', 'RUSTFLAGS', 'RUSTC_WRAPPER',
                'RUSTC_WORKSPACE_WRAPPER', 'TORCH_RS_CUDART'):
            env.pop(key)
    for key, relative in {
        'CARGO_HOME': 'cargo-home', 'CARGO_TARGET_DIR': 'native-build',
        'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'cache', 'CUDA_CACHE_PATH': 'cuda-cache',
        'TORCHINDUCTOR_CACHE_DIR': 'inductor-cache', 'TRITON_CACHE_DIR': 'triton-cache',
        'TORCH_HOME': 'torch-cache', 'TORCH_EXTENSIONS_DIR': 'torch-extensions',
    }.items():
        path = code / 'target/paired' / relative
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(UV_CACHE_DIR=str(ROOT / 'uv-cache'), HTTPS_PROXY='http://fwdproxy:8080',
               CARGO_HTTP_PROXY='http://fwdproxy:8080', CUDA_VISIBLE_DEVICES='0',
               CARGO_BUILD_JOBS='4', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
               PYO3_PYTHON=str(code / '.venv/bin/python'))
    for tool in ('rustc', 'rustdoc'):
        env[tool.upper()] = output(['rustup', 'which', '--toolchain', '1.92.0', tool], code)
    return env

def snapshot(code):
    status = output(['git', 'status', '--porcelain'], code)
    if status:
        raise RuntimeError(f'Source tree is not clean: {code}: {status}')
    return {'commit': output(['git', 'rev-parse', 'HEAD'], code),
            'tree': output(['git', 'rev-parse', 'HEAD^{tree}'], code),
            'benchmark_sha256': sha(code / 'scripts/benchmark_compile_cuda.py'),
            'lock_sha256': sha(code / 'uv.lock'), 'git_status': status}

def logged(args, code, env, path, timeout=1800):
    start = time.time()
    print(f'Starting {path.name}', flush=True)
    with path.open('w') as log:
        result = subprocess.run(args, cwd=code, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    record = {'command': [str(x) for x in args], 'cwd': str(code),
              'started_epoch': start, 'seconds': time.time() - start,
              'exit_code': result.returncode, 'log': str(path), 'log_sha256': sha(path)}
    save(path.with_suffix('.receipt.json'), record)
    print(f'Finished {path.name}: exit={result.returncode}, {record["seconds"]:.2f}s', flush=True)
    return record

def prepare(name):
    code = ROOT / name
    if code.exists():
        raise RuntimeError(f'Refusing to overwrite {code}')
    subprocess.run(['git', 'clone', '--quiet', '--shared', '--no-hardlinks',
                    '--no-checkout', str(REPO), str(code)], check=True)
    subprocess.run(['git', 'checkout', '--quiet', '--detach', COMMITS[name]], cwd=code, check=True)
    env = environment(name)
    reports = code / 'target/paired/reports'
    reports.mkdir(parents=True)
    before = snapshot(code)
    assert before['commit'] == COMMITS[name]
    commands = [
        ['uv', '--no-config', 'sync', '--locked', '--group', 'reference',
         '--no-install-project', '--python', str(REFERENCE_PYTHON)],
        [str(code / '.venv/bin/python'), '-m', 'maturin', 'build', '--release',
         '--locked', '--out', 'target/paired/wheels'],
    ]
    for label, command in zip(('dependencies', 'native-build'), commands):
        receipt = logged(command, code, env, reports / f'{label}.log')
        if receipt['exit_code']:
            raise RuntimeError(f'{name}: {label} failed; see {receipt["log"]}')
        assert snapshot(code) == before
    wheels = list((code / 'target/paired/wheels').glob('*.whl'))
    assert len(wheels) == 1
    for label, command in [
        ('install', ['uv', '--no-config', 'pip', 'install', '--python',
                     str(code / '.venv/bin/python'), '--force-reinstall', '--no-deps', str(wheels[0])]),
        ('provenance', [str(code / '.venv/bin/python'), '.github/scripts/verify_native_extension.py']),
    ]:
        receipt = logged(command, code, env, reports / f'{label}.log')
        if receipt['exit_code']:
            raise RuntimeError(f'{name}: {label} failed')
    site = next((code / '.venv/lib').glob('python*/site-packages'))
    package = site / 'torch_rs'
    for source in (code / 'python/torch_rs').rglob('*.py'):
        assert sha(source) == sha(package / source.relative_to(code / 'python/torch_rs'))
    native = next(package.glob('*.abi3.so'))
    versions = output([str(code / '.venv/bin/python'), '-B', '-c',
        'import json,sys,torch,numpy; print(json.dumps(dict(python=sys.version,executable=sys.executable,torch=torch.__version__,torch_cuda=torch.version.cuda,numpy=numpy.__version__)))'], code, env)
    assert snapshot(code) == before
    save(reports / 'build-record.json', {**before, 'wheel': str(wheels[0]),
         'wheel_sha256': sha(wheels[0]), 'native': str(native), 'native_sha256': sha(native),
         'versions': json.loads(versions), 'environment': {k: env[k] for k in (
             'CARGO_HOME', 'CARGO_TARGET_DIR', 'PYO3_PYTHON', 'RUSTC', 'RUSTDOC',
             'CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED', 'UV_CACHE_DIR')},
         'rustc': output([env['RUSTC'], '--version'], code, env),
         'nvcc': output(['nvcc', '--version'], code, env),
         'build_cache': 'new empty native target; shared download cache inside paired-validation root'})
    print(f'Prepared {name}: {before["commit"]}', flush=True)

def measure():
    if (ROOT / 'results.json').exists():
        raise RuntimeError('Refusing to overwrite a measurement series')
    builds = {name: json.loads((ROOT / name / 'target/paired/reports/build-record.json').read_text())
              for name in COMMITS}
    assert builds['baseline']['benchmark_sha256'] == builds['candidate']['benchmark_sha256']
    assert builds['baseline']['lock_sha256'] == builds['candidate']['lock_sha256']
    for key in ('python', 'torch', 'torch_cuda', 'numpy'):
        assert builds['baseline']['versions'][key] == builds['candidate']['versions'][key]
    save(ROOT / 'plan.json', {'order': ORDER, 'cpu': CPU, 'gpu': 0, 'commits': COMMITS,
         'cache_policy': 'first invocation per checkout cold, subsequent invocations reuse that checkout caches',
         'benchmark_options': ['--include-unprepared-comparison'], 'runs': 6,
         'no_retry_or_selection': True, 'purpose': 'paired diagnostic; does not alter Burner scores'})
    records = []
    save(ROOT / 'results.json', records)
    for number, name in enumerate(ORDER, 1):
        dashboard = json.loads(output(['curl', '-fsS', '--max-time', '15',
                                      'http://127.0.0.1:4321/api/dashboard'], ROOT))
        runtime = dashboard['runtime']
        if dashboard['state']['orchestrator']['enabled'] or any(runtime[k] for k in (
                'runningAgents', 'runningEvaluations', 'runningComposites')):
            raise RuntimeError('Burner must be paused and idle for paired GPU measurements')
        code = ROOT / name
        env = environment(name)
        before = snapshot(code)
        assert before['commit'] == COMMITS[name]
        assert sha(Path(builds[name]['native'])) == builds[name]['native_sha256']
        report = code / f'target/paired/reports/run-{number}.json'
        receipt = logged(['taskset', '-c', str(CPU), str(code / '.venv/bin/python'), '-B',
                          'scripts/benchmark_compile_cuda.py', '--include-unprepared-comparison',
                          '--output', str(report)], code, env, report.with_suffix('.log'))
        assert snapshot(code) == before
        row = {'number': number, 'variant': name, 'source': before, 'receipt': receipt,
               'report': str(report), 'native_sha256': builds[name]['native_sha256']}
        if report.is_file():
            data = json.loads(report.read_text())
            row.update(report_sha256=sha(report), aggregates=data.get('aggregates'),
                       environment=data.get('environment'))
        records.append(row)
        save(ROOT / 'results.json', records)
        print(json.dumps({'run': number, 'variant': name, 'exit_code': receipt['exit_code'],
                          'score': (row.get('aggregates') or {}).get('coverage_adjusted_overall_percent')}), flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(prepare, COMMITS))
    elif sys.argv[1] == 'measure':
        measure()
    else:
        raise SystemExit('Expected prepare or measure')
