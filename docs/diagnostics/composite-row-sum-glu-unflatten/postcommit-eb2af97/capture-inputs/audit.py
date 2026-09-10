"""Verify captured evidence against this unchanged clean implementation."""
import hashlib, json, pathlib, subprocess, sys, zipfile
root = pathlib.Path.cwd()
out = root / 'target/postcommit-eb2af97'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
sys.path.insert(0, str(root / 'scripts'))
from evaluate_cuda_math import source_provenance, account, corpus
receipt = json.loads((out / 'build-capture/build-record.json').read_text())
report = json.loads((out / 'cuda-math.json').read_text())
assert not subprocess.check_output(['git', 'status', '--porcelain'])
source = source_provenance()
assert source['commit'] == 'eb2af972ac6fd538951024bb109db89079a6c77e'
assert {k:receipt[k] for k in source} == source == report['source']
assert receipt == report['build_record']
assert receipt['measurement_kind'] == 'clean-code' and receipt['clean_checkout'] is True
assert receipt['git_status_before_build'] == ''
assert receipt['source_unchanged_during_build'] and report['source_unchanged_during_run']
assert sha(receipt['wheel_path']) == receipt['wheel_sha256']
assert sha(receipt['interpreter_path']) == receipt['interpreter_sha256']
assert sha(receipt['rustc_path']) == receipt['rustc_sha256']
import torch_rs
extension = pathlib.Path(torch_rs._C.__file__).resolve()
assert str(extension) == receipt['extension_path']
assert sha(extension) == receipt['extension_sha256'] == sha(receipt['source_extension_path'])
installed = pathlib.Path(torch_rs.__file__).resolve().parent
assert installed.is_relative_to(root / '.venv')
python_files = {str(p.relative_to(root)):sha(p) for p in (root/'python/torch_rs').rglob('*.py')}
with zipfile.ZipFile(receipt['wheel_path']) as wheel:
    for relative, digest in python_files.items():
        p = pathlib.Path(relative)
        assert digest == sha(installed / p.relative_to('python/torch_rs'))
        assert digest == hashlib.sha256(wheel.read(str(p.relative_to('python')))).hexdigest()
    assert receipt['extension_sha256'] == hashlib.sha256(wheel.read('torch_rs/'+extension.name)).hexdigest()
assert report['evaluator_sha256'] == sha(root/'scripts/evaluate_cuda_math.py')
assert report['matrix_sha256'] == sha(root/'docs/hardware-heterogeneity-matrix-v1.json')
assert report['accounting'] == account(corpus(), report['seeds'], report['trials'], receipt, source)
assert report['accounting']['denominator'] == 6 and report['accounting']['passed'] == 5
assert report['seeds'] == [7763153567161607008, 2618969910755569448]
assert len(report['trials']) == 12
worker_files, runtimes = set(), set()
for trial in report['trials']:
    ref, cand = trial['reference'], trial['candidate']
    assert ref['pid'] != cand['pid']
    assert ref['status'] == 'passed'
    assert cand['loaded_torch_modules'] == cand['blocked_imports'] == []
    assert cand['extension']['sha256'] == receipt['extension_sha256']
    expected = 'failed' if trial['case_id'] == 'cuda_f32_matmul' else 'passed'
    assert cand['status'] == expected, (trial['case_id'], cand)
    if expected == 'passed':
        assert cand['inputs'] == cand['inputs_after']
    for worker in (ref, cand):
        for key in ('package', 'executable'):
            p = pathlib.Path(worker[key]).absolute()
            assert p.is_relative_to(root), p
            worker_files.add(p)
        assert worker['cuda_runtimes']
        for runtime in worker['cuda_runtimes']:
            p = pathlib.Path(runtime['path']).resolve()
            assert p.is_relative_to(root), p
            assert runtime['version'] == 13000 and runtime['status'] == 0
            runtimes.add(p)
for role in ('candidate', 'reference'):
    record = json.loads((out/f'numerical-{role}.json').read_text())
    for runtime in record['cuda_runtimes']:
        p = pathlib.Path(runtime['path']).resolve()
        assert p.is_relative_to(root) and runtime['version'] == 13000
        runtimes.add(p)
    if role == 'candidate':
        assert record['loaded_torch_modules'] == record['blocked_imports'] == []
        assert record['extension']['sha256'] == receipt['extension_sha256']
    assert len(record['cases']) == 46
commands = []
for path in sorted(out.glob('*.receipt.json')):
    record = json.loads(path.read_text())
    assert record['exit_status'] == 0, path
    assert record['git_status_before'] == record['git_status_after'] == ''
    assert sha(record['log']) == record['log_sha256']
    commands.append(str(path))
for record in receipt['commands']:
    assert record['exit_status'] == 0 and sha(record['log']) == record['log_sha256']
inventory = json.loads((out/'candidate-inventory.json').read_text())
for p, digest in {**inventory['candidate_files'], **inventory['protected_files']}.items():
    assert sha(root/p) == digest, p
historical = root/'docs/diagnostics/cuda-sum-rows'
precommit = root/'docs/diagnostics/composite-row-sum-glu-unflatten'
reference = root/'.venv/lib/python3.12/site-packages/torch/include/ATen/native/cuda/Reduce.cuh'
result = dict(measurement_kind='clean-code', source=source,
    build_record_sha256=sha(out/'build-capture/build-record.json'), report_sha256=sha(out/'cuda-math.json'),
    production_sources_installed_and_wheel_match=True, installed_python_files=python_files,
    wheel_extension_matches_installed_and_source=True,
    reference_header=dict(path=str(reference), sha256=sha(reference)),
    worker_file_hashes={str(p):sha(p) for p in sorted(worker_files)},
    runtime_hashes={str(p):sha(p) for p in sorted(runtimes)},
    command_receipts=commands, evaluator_accounting_recomputed=True,
    historical_and_precommit_files_unchanged=True,
    preserved_evidence_hashes={str(p.relative_to(root)):sha(p) for parent in (historical, precommit) for p in parent.rglob('*') if p.is_file()},
    protected_files_unchanged=inventory['protected_files'], no_performance_credit=True)
(out/'evidence-audit.json').write_text(json.dumps(result, indent=2)+'\n')
print('Clean commit, build, wheel, installed/source extensions,',len(python_files),'Python sources, evaluator accounting, worker processes, runtimes, preserved evidence and protected files verified.')
print('Source:',source)
print('Runtime hashes:',result['runtime_hashes'])
