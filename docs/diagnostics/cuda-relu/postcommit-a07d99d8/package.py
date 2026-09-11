"""Audit and package measurements; does not run or change workloads."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

root = Path.cwd()
out = root / 'target/relu-postcommit-a07d99d8'
dest = root / 'docs/diagnostics/cuda-relu/postcommit-a07d99d8'
historical = root / 'docs/diagnostics/cuda-relu/development'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return json.loads(Path(path).read_text())
def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')

campaign = read(out/'campaign.json')
state = campaign['initial_state']
assert state['git_status'] == ''
assert campaign['final_state'] == state
assert subprocess.check_output(['git','status','--porcelain']) == b''
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == state['commit']
for step in campaign['steps']:
    assert step['state_before'] == step['state_after'] == state
    assert step['exit_status'] == 0
assert sha(out/'capture.py') == campaign['capture_script_sha256']
assert sha(out/'environment.sh') == campaign['environment_script_sha256']
assert sha(historical/'artifact-sha256.json') == campaign['historical_bundle_manifest_sha256']
for name,digest in read(historical/'artifact-sha256.json').items():
    assert sha(historical/name) == digest, name

build = read(out/'release/build-record.json')
assert build['clean_checkout'] and build['git_status_before_build'] == ''
assert build['source_unchanged_during_build']
assert build['commit'] == state['commit']
assert build['production_diff_sha256'] == hashlib.sha256(b'').hexdigest()
for path_key, hash_key in [('extension_path','extension_sha256'),('source_extension_path','extension_sha256'),('wheel_path','wheel_sha256'),('interpreter_path','interpreter_sha256')]:
    p = Path(build[path_key])
    assert p.resolve().is_relative_to(root)
    assert sha(p) == build[hash_key]
for command in build['commands']:
    assert command['exit_status'] == 0
    assert Path(command['log']).resolve().is_relative_to(root)
    assert sha(command['log']) == command['log_sha256']
locked = tomllib.loads((root/'uv.lock').read_text())['package']
normalize = lambda s: re.sub(r'[-_.]+','-',s).lower()
versions = {(normalize(p['name']),p['version']) for p in locked}
assert all((normalize(n),v) in versions for n,v in build['dependencies'].items())
assert build['uv_lock_sha256'] == sha(root/'uv.lock')

acceptance = read(out/'acceptance.json')
for key in ('commit','source_sha256','production_diff_sha256','extension_sha256'):
    assert acceptance[key] == build[key]
assert acceptance['probe_sha256'] == sha(historical/'acceptance-probe.py')
assert Path(acceptance['native_runtime']['library']).resolve().is_relative_to(root)
assert acceptance['native_runtime']['version'] == 13000
assert acceptance['torch'] == '2.13.0+cu130'
bits = [int(x,16) for x in acceptance['edge_input_bits']]
expected = [f'{(0 if (b & 0x80000000 and b & 0x7fffffff <= 0x7f800000) else b):08x}' for b in bits]
for key in ('native_eager_bits','native_compiled_bits','reference_eager_bits','reference_compiled_bits'):
    assert acceptance[key] == expected
baseline = read(historical/'baseline.json')
assert acceptance['seed'] == baseline['seed']
for row,original in zip(acceptance['programs'],baseline['programs'],strict=True):
    assert row['program'] == original['program'] and row['inputs'] == original['inputs']
    for mode in ('native_eager','native_compiled','reference_eager','reference_compiled'):
        assert row[mode]['status'] == 'pass'
        np.testing.assert_allclose(row[mode]['output'],row['reference_eager']['output'],rtol=1e-5,atol=1e-6)

receipts = {}
shared_manifests = {}
executables = {}
for step in campaign['steps']:
    if step['name'] in ('historical-audit','release-build'):
        continue
    name = 'postcommit-a07d99d8-'+step['name']
    path = root/'target/relu-checks'/f'{name}.json'
    receipt = read(path)
    for key in ('commit','source_sha256','production_diff_sha256'):
        assert receipt[key] == build[key]
    assert receipt['source_unchanged'] and receipt['exit_status'] == 0
    assert receipt['native_sha256'] == build['extension_sha256']
    assert receipt['native_path'] == build['extension_path']
    assert receipt['environment']['CUDA_VISIBLE_DEVICES'] == step['CUDA_VISIBLE_DEVICES']
    assert Path(receipt['environment']['TORCH_RS_CUDART']).resolve().is_relative_to(root)
    log = path.parent/receipt['log']
    assert sha(log) == receipt['log_sha256']
    manifest = path.parent/receipt['source_manifest']
    shared = historical/'checks'/receipt['source_manifest']
    assert sha(manifest) == sha(shared) == receipt['source_manifest_sha256']
    for source,digest in read(manifest).items():
        assert sha(root/source) == digest, source
    shared_manifests[receipt['source_manifest']] = '../development/checks/'+receipt['source_manifest']
    receipts[step['name']] = {'receipt': 'checks/'+path.name, 'exit_status': receipt['exit_status']}
    for binary in re.findall(r'Running .*? \((target/[^)]+)\)',log.read_text()):
        p = root/binary
        assert p.resolve().is_relative_to(root)
        executables[binary] = sha(p)

assert not dest.exists()
(dest/'checks').mkdir(parents=True)
(dest/'build').mkdir()
for name in ('campaign.json','acceptance.json','capture.log','environment.sh','capture.py','package.py'):
    shutil.copyfile(out/name,dest/name)
for name in ('build-record.json','commands.json','build.log','install.log'):
    shutil.copyfile(out/'release'/name,dest/'build'/name)
for item in receipts.values():
    name = Path(item['receipt']).name
    shutil.copyfile(root/'target/relu-checks'/name,dest/'checks'/name)
    log = read(dest/'checks'/name)['log']
    shutil.copyfile(root/'target/relu-checks'/log,dest/'checks'/log)
write(dest/'audit.json', {'audited_at':datetime.now(timezone.utc).isoformat(), 'command':sys.argv, 'audit_script_sha256':sha(__file__), 'commit':state['commit'], 'historical_artifacts_unchanged':85, 'clean_steps':len(campaign['steps']), 'release_and_installed_package_verified':True, 'all_installed_dependency_versions_in_uv_lock':True, 'ieee_patterns_verified':len(bits), 'programs_verified':len(acceptance['programs']), 'baseline_inputs_unchanged':True, 'receipt_native_and_manifest_hashes_verified':True, 'rust_executable_sha256':executables})
write(dest/'index.json', {'kind':'clean-commit semantic diagnostic; no timing or scoring claims', 'measured_commit':state['commit'], 'base_commit':'9407a208d12a6a650ced579e943154849973e037', 'campaign':'campaign.json', 'build':'build/build-record.json', 'acceptance':'acceptance.json', 'audit':'audit.json', 'checks':receipts, 'source_manifest_resolution':shared_manifests, 'source_manifest_note':'Raw receipts identify manifests by basename. Resolve through this map to the unchanged development bundle; one immutable manifest is shared by all current checks without duplicating its bytes.', 'historical_evidence':'../development/index.json', 'reproduction':'capture.py is the executed command-orchestration record, using committed helpers without workload changes. Source environment.sh and run it from the repository root only in a fresh clean checkout with unused target paths; capture and build helpers refuse overwrites. package.py audits and copies the completed captures. Generated build products and caches remain ignored under target/.', 'scope':'Fresh release identity, seeded eager and eager-backend compiled reference differentials, IEEE bits, focused ReLU Python/Rust checks, two-device restoration and portable CUDA-hidden skips. Broader author validation and its failures remain in development/.', 'limitations':['No timing or general torch.compile/accelerator/training parity claim.','GPU inventory/utilization snapshots do not reserve hardware.','This evidence does not approve the branch or replace independent review or merge gates.']})
artifacts = {str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
write(dest/'artifact-sha256.json', artifacts)
for name,digest in read(dest/'artifact-sha256.json').items():
    assert sha(dest/name) == digest
print(json.dumps({'published':str(dest),'artifacts':len(artifacts),'shared_manifests':len(shared_manifests),'measured_commit':state['commit']},indent=2))
