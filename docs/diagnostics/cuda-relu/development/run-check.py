import hashlib,json,os,subprocess,sys,time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
from evaluate_cuda_math import source_provenance, sha256
root=Path.cwd(); name=sys.argv[1]; command=sys.argv[2:]
out=root/'target/relu-checks'; out.mkdir(exist_ok=True)
receipt=out/(name+'.json'); log=out/(name+'.log')
assert not receipt.exists() and not log.exists()
before=source_provenance()
paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z','--','tests','src','python','scripts','Cargo.toml','Cargo.lock','rust-toolchain.toml','pyproject.toml','uv.lock'],text=True).split('\0')
files={p:sha256(p) for p in paths if p and Path(p).is_file()}
manifest=json.dumps(files,sort_keys=True,indent=2)+'\n'; digest=hashlib.sha256(manifest.encode()).hexdigest()
manifest_path=out/(digest+'.manifest.json')
if not manifest_path.exists(): manifest_path.write_text(manifest)
def snapshot(): return subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,compute_cap,utilization.gpu,memory.used,memory.total','--format=csv'],text=True)
record={**before,'command':command,'source_manifest':manifest_path.name,'source_manifest_sha256':digest,'gpu_before':snapshot(),'started_at':datetime.now(timezone.utc).isoformat(),'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TORCH_RS_CUDART','CARGO_HOME','CARGO_TARGET_DIR','UV_PYTHON_INSTALL_DIR','UV_CACHE_DIR','TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','RUSTUP_TOOLCHAIN')}}
import torch_rs
record['native_path']=torch_rs._C.__file__; record['native_sha256']=sha256(torch_rs._C.__file__)
start=time.monotonic()
with log.open('w') as f:
    result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
record.update(exit_status=result.returncode,wall_seconds=time.monotonic()-start,ended_at=datetime.now(timezone.utc).isoformat(),gpu_after=snapshot(),log=log.name,log_sha256=sha256(log),source_unchanged=before==source_provenance())
receipt.write_text(json.dumps(record,indent=2)+'\n')
print(name,result.returncode,log)
sys.exit(result.returncode)
