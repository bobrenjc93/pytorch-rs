"""Audit precommit indexing-repair diagnostics without rewriting provenance."""
import hashlib, json, pathlib, subprocess, sys, zipfile
root=pathlib.Path.cwd(); out=root/'target/indexing-repair'; snapshot=root/'target/python314-checkout'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(root/'scripts'))
from evaluate_cuda_math import source_provenance, account, corpus
source=source_provenance()
receipt=json.loads((out/'capture-final/build-record.json').read_text())
report=json.loads((out/'cuda-math.json').read_text())
assert {k:receipt[k] for k in source} == source == report['source']
assert receipt == report['build_record']
assert receipt['measurement_kind']=='precommit-diagnostic' and receipt['clean_checkout'] is False
assert receipt['source_unchanged_during_build'] and report['source_unchanged_during_run']
assert sha(receipt['wheel_path'])==receipt['wheel_sha256']
assert sha(receipt['extension_path'])==receipt['extension_sha256']==sha(receipt['source_extension_path'])
assert sha(receipt['interpreter_path'])==receipt['interpreter_sha256']
assert sha(receipt['rustc_path'])==receipt['rustc_sha256']
import torch_rs
assert str(pathlib.Path(torch_rs._C.__file__).resolve())==receipt['extension_path']
installed=pathlib.Path(torch_rs.__file__).resolve().parent
python_files={str(p.relative_to(root)):sha(p) for p in (root/'python/torch_rs').rglob('*.py')}
with zipfile.ZipFile(receipt['wheel_path']) as wheel:
    for p,digest in python_files.items():
        relative=pathlib.Path(p).relative_to('python/torch_rs')
        assert digest==sha(installed/relative)==sha(snapshot/p)
        assert digest==hashlib.sha256(wheel.read('torch_rs/'+str(relative))).hexdigest()
    assert hashlib.sha256(wheel.read('torch_rs/torch_rs.abi3.so')).hexdigest()==receipt['extension_sha256']
command=[str(snapshot/'.venv/bin/python'),'-I','-B','-c',
    'import hashlib,json,pathlib,sys,torch_rs,torch; p=pathlib.Path(torch_rs._C.__file__).resolve(); print(json.dumps(dict(reference_version=torch.__version__,runtime_hashes={v:hashlib.sha256(pathlib.Path(v).read_bytes()).hexdigest() for v in {line.split()[-1] for line in pathlib.Path("/proc/self/maps").read_text().splitlines() if "/libcudart.so" in line}},python=sys.version,executable=sys.executable,interpreter_path=str(pathlib.Path(sys.executable).resolve()),interpreter_sha256=hashlib.sha256(pathlib.Path(sys.executable).read_bytes()).hexdigest(),package=str(pathlib.Path(torch_rs.__file__).resolve()),extension=str(p),extension_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))']
py314=json.loads(subprocess.check_output(command,text=True))
assert py314['extension_sha256']==receipt['extension_sha256']==sha(snapshot/'python/torch_rs/torch_rs.abi3.so')
assert pathlib.Path(py314['extension']).is_relative_to(snapshot/'.venv')
assert py314['runtime_hashes']
for p,digest in py314['runtime_hashes'].items():
    assert pathlib.Path(p).resolve().is_relative_to(root) and sha(p)==digest
paths=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard','--','src','python','tests','scripts','Cargo.toml','Cargo.lock','pyproject.toml','uv.lock','rust-toolchain.toml'],text=True).split('\0')
for p in filter(None,paths):assert sha(root/p)==sha(snapshot/p),p
assert report['evaluator_sha256']==sha(root/'scripts/evaluate_cuda_math.py')
assert report['matrix_sha256']==sha(root/'docs/hardware-heterogeneity-matrix-v1.json')
assert report['accounting']==account(corpus(),report['seeds'],report['trials'],receipt,source)
assert report['accounting']['denominator']==6 and report['accounting']['passed']==5
worker_files,runtimes=set(),set()
for trial in report['trials']:
    ref,cand=trial['reference'],trial['candidate']
    assert ref['pid']!=cand['pid'] and ref['status']=='passed'
    assert cand['loaded_torch_modules']==cand['blocked_imports']==[]
    assert cand['extension']['sha256']==receipt['extension_sha256']
    for worker in (ref,cand):
        for key in ('package','executable'):
            p=pathlib.Path(worker[key]).absolute(); assert p.is_relative_to(root)
            worker_files.add(p)
        assert worker['cuda_runtimes']
        for runtime in worker['cuda_runtimes']:
            p=pathlib.Path(runtime['path']).resolve()
            assert p.is_relative_to(root) and runtime['version']==13000 and runtime['status']==0
            runtimes.add(p)
for role in ('candidate','reference'):
    record=json.loads((out/f'numerical-{role}.json').read_text())
    assert len(record['cases'])==46
    if role=='candidate':
        assert record['loaded_torch_modules']==record['blocked_imports']==[]
        assert record['extension']['sha256']==receipt['extension_sha256']
    for runtime in record['cuda_runtimes']:
        p=pathlib.Path(runtime['path']).resolve();assert p.is_relative_to(root)
        runtimes.add(p)
preserved=json.loads((out/'reference-and-preservation.json').read_text())
for p,digest in {**preserved['protected_files'],**preserved['prior_evidence_files']}.items():assert sha(root/p)==digest,p
commands=[]
for path in sorted(out.glob('*.receipt.json')):
    record=json.loads(path.read_text())
    assert record['exit_status']==(101 if path.name=='clippy.receipt.json' else 0),path
    assert sha(record['log'])==record['log_sha256'],path
    commands.append(str(path))
for record in receipt['commands']:assert record['exit_status']==0 and sha(record['log'])==record['log_sha256']
result=dict(measurement_kind='precommit-diagnostic',clean_checkout=False,source=source,
    pending_clean_capture_after_burner_commit=True,
    build_record_sha256=sha(out/'capture-final/build-record.json'),report_sha256=sha(out/'cuda-math.json'),
    installed_python_sources=python_files,python314=py314,python314_inspection_command=command,
    source_snapshot_matches=True,test_source_hashes={p:sha(root/p) for p in paths if p.startswith('tests/')},
    worker_file_hashes={str(p):sha(p) for p in sorted(worker_files)},
    runtime_hashes={str(p):sha(p) for p in sorted(runtimes)},command_receipts=commands,
    protected_and_prior_evidence_files_unchanged=True,evaluator_accounting_recomputed=True,
    no_performance_credit=True)
(out/'evidence-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('Verified final dirty-source build, source/wheel/installed/3.14 snapshot identities, evaluator, separate workers, runtime hashes, all command logs, and unchanged protected/prior evidence files.')
print('Source:',source)
print('Extension:',receipt['extension_sha256'])
