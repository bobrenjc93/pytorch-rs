"""Orchestrate committed measurement helpers; publish only after clean runs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

root = Path.cwd()
out = root / 'target/relu-postcommit-5b2a315f'
commit = '5b2a315f8c433a839e56844f5b0766580cf55a6b'
def git(*args):
    return subprocess.check_output(['git', *args], text=True)
def stamp():
    return datetime.now(timezone.utc).isoformat()
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state():
    return {'commit': git('rev-parse', 'HEAD').strip(), 'tree': git('rev-parse', 'HEAD^{tree}').strip(), 'git_status': git('status', '--porcelain')}
initial = state()
assert initial['commit'] == commit and initial['git_status'] == ''
assert Path(sys.prefix) == root / '.venv'
assert Path(sys.executable).resolve().is_relative_to(Path(os.environ['UV_PYTHON_INSTALL_DIR']))
record = {'measurement_kind': 'clean-commit non-scoring semantic diagnostic', 'cwd': str(root), 'initial_state': initial, 'started_at': stamp(), 'python_executable': sys.executable, 'interpreter': str(Path(sys.executable).resolve()), 'python': sys.version, 'environment_script_sha256': sha(out/'environment.sh'), 'capture_script_sha256': sha(__file__), 'cache_state': 'fresh release build target and CUDA/XDG/Inductor/Triton caches; populated worktree-local Cargo registry, uv cache and locked .venv; fresh Rust test target', 'historical_bundle_manifest_sha256': sha(root/'docs/diagnostics/cuda-relu/development/artifact-sha256.json'), 'steps': []}
def save():
    (out/'campaign.json').write_text(json.dumps(record, indent=2)+'\n')
def run(name, command, devices='0', receipt=True):
    before = state()
    assert before == initial, before
    step = {'name': name, 'state_before': before, 'started_at': stamp(), 'CUDA_VISIBLE_DEVICES': devices}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices)
    if receipt:
        command = [sys.executable, 'docs/diagnostics/cuda-relu/development/run-check.py', 'postcommit-5b2a315f-'+name, *command]
    step['command'] = command
    record['steps'].append(step)
    save()
    result = subprocess.run(command, env=env)
    step.update(ended_at=stamp(), exit_status=result.returncode, state_after=state())
    save()
    assert step['state_after'] == initial, step
    result.check_returncode()

save()
run('historical-audit', [sys.executable, 'docs/diagnostics/cuda-relu/development/audit-evidence.py'], receipt=False)
run('release-build', [sys.executable, 'scripts/capture_depth_concat_build.py', '--output', str(out/'release')], receipt=False)
run('verify-release', [sys.executable, 'docs/diagnostics/cuda-relu/development/verify-build.py', str(out/'release/build-record.json')])
run('acceptance', [sys.executable, 'docs/diagnostics/cuda-relu/development/acceptance-probe.py', str(out/'acceptance.json')])
run('focused', [sys.executable, '-m', 'unittest', 'tests.test_cuda_relu', 'tests.test_compile_cuda_relu', '-v'])
run('two-device', [sys.executable, '-m', 'unittest', 'tests.test_cuda_relu.CudaReluDeviceTests', 'tests.test_compile_cuda_relu.CompileCudaReluDeviceTests', '-v'], devices='0,1')
run('cuda-hidden', [sys.executable, '-m', 'unittest', 'tests.test_cuda_relu', 'tests.test_compile_cuda_relu', '-v'], devices='')
run('rust-bindings-relu', ['cargo', 'test', '--locked', '--offline', '--features', 'python-bindings', 'relu', '--', '--nocapture'])
run('rust-default-relu', ['cargo', 'test', '--locked', '--offline', '--no-default-features', '--test', 'cuda_relu', '--', '--nocapture'])
run('rust-kernel-slots', ['cargo', 'test', '--locked', '--offline', '--lib', 'kernel_slots_match_driver_entry_order', '--', '--nocapture'])
record['ended_at'] = stamp()
record['final_state'] = state()
save()
