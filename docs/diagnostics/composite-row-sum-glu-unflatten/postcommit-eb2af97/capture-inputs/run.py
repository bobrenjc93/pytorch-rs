import datetime, hashlib, json, os, pathlib, subprocess, sys, time
root = pathlib.Path.cwd()
out = root / 'target/postcommit-eb2af97'
name, *command = sys.argv[1:]
log = out / (name + '.log')
assert not log.exists()
status = lambda: subprocess.check_output(['git', 'status', '--porcelain'], text=True)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert commit == 'eb2af972ac6fd538951024bb109db89079a6c77e'
before = status()
assert not before, before
stamp = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
started, tick = stamp(), time.perf_counter()
with log.open('w') as stream:
    result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
after = status()
record = dict(command=command, cwd=str(root), commit=commit, started_at=started, ended_at=stamp(), wall_seconds=time.perf_counter()-tick, exit_status=result.returncode, git_status_before=before, git_status_after=after, log=str(log), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), environment={k:v for k,v in os.environ.items() if k.startswith(('CARGO_', 'UV_', 'CUDA_', 'TORCH_', 'TRITON_', 'PYO3_')) and 'PROXY' not in k or k in ('TMPDIR','XDG_CACHE_HOME','VIRTUAL_ENV','PYTHONNOUSERSITE')})
(out / (name + '.receipt.json')).write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
assert not after, after
sys.exit(result.returncode)
