import datetime, hashlib, json, os, pathlib, subprocess, sys, time
root = pathlib.Path.cwd().resolve()
label, *command = sys.argv[1:]
out = root / 'target/integration-evidence'
out.mkdir(parents=True, exist_ok=True)
receipt = out / (label + '.receipt.json')
assert not receipt.exists(), receipt
record = dict(command=command, cwd=str(root), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), source_status=subprocess.check_output(['git','status','--porcelain'],text=True), environment={k: v for k,v in os.environ.items() if k in ('TMPDIR','XDG_CACHE_HOME','UV_CACHE_DIR','UV_PYTHON_INSTALL_DIR','CARGO_HOME','CARGO_TARGET_DIR','CUDA_VISIBLE_DEVICES','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RUSTUP_TOOLCHAIN','PYO3_PYTHON','PYTHONDONTWRITEBYTECODE','PYTHONNOUSERSITE')})
receipt.write_text(json.dumps(record, indent=2)+'\n')
start = time.monotonic()
with (out / (label + '.log')).open('w') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
record.update(returncode=result.returncode, elapsed_seconds=time.monotonic()-start, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), log_sha256=hashlib.sha256((out/(label+'.log')).read_bytes()).hexdigest())
receipt.write_text(json.dumps(record, indent=2)+'\n')
print(label, 'exit', result.returncode, 'seconds', record['elapsed_seconds'], flush=True)
raise SystemExit(result.returncode)
