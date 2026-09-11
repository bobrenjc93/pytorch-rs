import hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
from evaluate_cuda_math import source_provenance,sha256
root=Path.cwd(); dest=root/'docs/diagnostics/cuda-relu/development'
artifacts=json.loads((dest/'artifact-sha256.json').read_text())
for path,expected in artifacts.items():assert sha256(dest/path)==expected,path
for receipt in (dest/'checks').glob('*.json'):
    if receipt.name.endswith('.manifest.json'):continue
    r=json.loads(receipt.read_text())
    assert sha256(receipt.parent/r['log'])==r['log_sha256']
    assert sha256(receipt.parent/r['source_manifest'])==r['source_manifest_sha256']
for label in ('baseline','initial','bridge','acceptance'):
    r=json.loads((dest/('builds/'+label+'-build-record.json')).read_text())
    for command in r['commands']:
        assert sha256(dest/('builds/'+label+'-'+Path(command['log']).name))==command['log_sha256']
latest=json.loads((dest/'checks/focused-corrected.json').read_text())
manifest=json.loads((dest/'checks'/latest['source_manifest']).read_text())
for path,expected in manifest.items():assert sha256(root/path)==expected,path
acceptance=json.loads((dest/'acceptance.json').read_text())
baseline=json.loads((dest/'baseline.json').read_text())
build=json.loads((dest/'builds/acceptance-build-record.json').read_text())
assert acceptance['source_sha256']==build['source_sha256']==source_provenance()['source_sha256']
assert acceptance['extension_sha256']==build['extension_sha256']
assert acceptance['native_eager_bits']==acceptance['native_compiled_bits']==acceptance['reference_eager_bits']==acceptance['reference_compiled_bits']
for row in baseline['programs']:
    assert row['reference_eager']['status']==row['reference_compiled']['status']=='pass'
    assert row['native_eager']['status']==row['native_compiled']['status']=='error'
import numpy as np
for row in acceptance['programs']:
    for label in ('native_eager','native_compiled','reference_eager','reference_compiled'):
        assert row[label]['status']=='pass'
        np.testing.assert_allclose(row[label]['output'],row['reference_eager']['output'],rtol=1e-5,atol=1e-6)
for path in ('scripts','tests/test_compile_corpus.py','docs/hardware-heterogeneity-matrix-v1.json','uv.lock','Cargo.lock'):
    assert not subprocess.check_output(['git','diff','HEAD','--',path]),path
print(json.dumps({'artifacts_verified':len(artifacts),'source':source_provenance(),'baseline_gap_verified':True,'acceptance_bits_and_programs_verified':True,'evaluators_and_locks_unchanged':True},indent=2))
