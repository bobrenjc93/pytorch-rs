import datetime, hashlib, json, os, pathlib, subprocess, sys, time
root = pathlib.Path(__file__).resolve().parents[2]
name, *command = sys.argv[1:]
out = root / 'target/integration'
log = out / (name + '.log')
assert not log.exists(), log
stamp = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
status = lambda: subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
before, start, tick = status(), stamp(), time.monotonic()
with log.open('w') as stream:
    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
record = dict(command=command, cwd=os.getcwd(), started_at=start, ended_at=stamp(),
              seconds=time.monotonic()-tick, exit_status=result.returncode,
              git_status_before=before, git_status_after=status(),
              log=str(log), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
              environment={k:v for k,v in os.environ.items() if k.startswith(('CARGO_', 'UV_', 'CUDA_', 'TORCH_', 'TRITON_', 'PYO3_')) or k in ('TMPDIR','XDG_CACHE_HOME','VIRTUAL_ENV')})
(out / (name + '-receipt.json')).write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k:record[k] for k in ('command','exit_status','seconds','log')}), flush=True)
sys.exit(result.returncode)
