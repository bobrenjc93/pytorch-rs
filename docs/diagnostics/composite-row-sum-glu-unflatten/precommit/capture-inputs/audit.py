import hashlib, importlib.util, json, pathlib, subprocess, sys, zipfile
root=pathlib.Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
receipt=json.loads((root/'target/integration/capture/build-record.json').read_text())
sys.path.insert(0,str(root/'scripts'))
from evaluate_cuda_math import source_provenance
assert {k:receipt[k] for k in ('commit','source_sha256','production_diff_sha256')} == source_provenance()
assert receipt['clean_checkout'] is False
import torch_rs
ext=pathlib.Path(torch_rs._C.__file__).resolve()
assert sha(ext)==receipt['extension_sha256']==sha(receipt['source_extension_path'])
assert sha(receipt['wheel_path'])==receipt['wheel_sha256']
installed=pathlib.Path(torch_rs.__file__).resolve().parent
snapshot=root/'target/python314-checkout'
python_files={str(p.relative_to(root)):sha(p) for p in (root/'python/torch_rs').rglob('*.py')}
for relative,digest in python_files.items():
    p=pathlib.Path(relative)
    assert digest==sha(installed/p.relative_to('python/torch_rs'))
    assert digest==sha(snapshot/relative)
with zipfile.ZipFile(receipt['wheel_path']) as wheel:
    for relative,digest in python_files.items():
        assert hashlib.sha256(wheel.read(str(pathlib.Path(relative).relative_to('python')))).hexdigest()==digest
paths=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard','--','src','tests','scripts','Cargo.toml','Cargo.lock','pyproject.toml','uv.lock','rust-toolchain.toml'],cwd=root).decode().split('\0')
for p in paths:
    if p:
        assert sha(root/p)==sha(snapshot/p),p
reference = root/'.venv/lib/python3.12/site-packages/torch/include/ATen/native/cuda/Reduce.cuh'
result=dict(measurement_kind='precommit-diagnostic', source=source_provenance(),
            installed_python_files=python_files, snapshot_implementation_tests_harness_match=True,
            reference_header=dict(path=str(reference),sha256=sha(reference)),
            build_record_sha256=sha(root/'target/integration/capture/build-record.json'))
(root/'target/integration/source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('source, installed wheel, source extension, Python files, and 3.14 snapshot verified')
