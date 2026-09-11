import ctypes, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import torch_rs as native
sys.path.insert(0, str(Path.cwd() / 'scripts'))
from evaluate_cuda_math import source_provenance, sha256
from tests.test_cuda_add import runtime
lib = runtime()
lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
lib.cudaMemcpy.restype = ctypes.c_int
bits = np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3fa00000,0xbfa00000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00000,0xffc12345,0x7f800001,0xff800001,0x7fa12345,0xffa12345,0x7fffffff,0xffffffff], dtype=np.uint32)
def download(x):
    out = np.empty(x.numel(), dtype=np.uint32)
    torch.cuda.synchronize()
    assert lib.cudaMemcpy(out.ctypes.data, x.data_ptr(), out.nbytes, 2) == 0
    return [f'{v:08x}' for v in out]
def relu(x): return x.relu()
def reduce(x): return (x*0.5).relu().sum(1)
def affine(x,w): return (x@w).relu()
report = dict(source_provenance(), extension_sha256=sha256(native._C.__file__), command=sys.argv, python=sys.version, torch=torch.__version__, torch_cuda=torch.version.cuda, probe_sha256=sha256(__file__))
report['inventory'] = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,compute_cap,utilization.gpu,memory.used,memory.total','--format=csv'],text=True)
version = ctypes.c_int()
assert lib.cudaRuntimeGetVersion(ctypes.byref(version)) == 0
report['native_runtime'] = {'library': lib._name, 'version':version.value}
assert lib.cudaDriverGetVersion(ctypes.byref(version)) == 0
report['driver_api_version'] = version.value
report['nvcc'] = subprocess.check_output(['nvcc','--version'],text=True)
report['nvcc_used'] = False
report['device_properties'] = str(torch.cuda.get_device_properties(0))
t = torch.from_numpy(bits.view(np.float32)).to('cuda:0')
report['edge_input_bits'] = download(t)
report['reference_eager_bits'] = download(t.relu())
report['reference_compiled_bits'] = download(torch.compile(relu,backend='eager',fullgraph=True)(t))
assert report['reference_eager_bits'] == report['reference_compiled_bits']
native_edge = native.zeros((bits.size,), device='cuda:0')
assert lib.cudaMemcpy(native_edge.data_ptr(), bits.ctypes.data, bits.nbytes, 1) == 0
report['native_eager_bits'] = download(native_edge.relu())
report['native_compiled_bits'] = download(native.compile(relu, backend='eager', fullgraph=True)(native_edge))
assert report['native_eager_bits'] == report['native_compiled_bits'] == report['reference_eager_bits']

rng = np.random.default_rng(1978)
report['seed'] = 1978
report['programs'] = []
for program, shapes in [(relu,[(3,7)]),(reduce,[(3,7)]),(affine,[(3,7),(7,5)])]:
    arrays = [rng.normal(size=s).astype(np.float32) for s in shapes]
    args = [native.tensor(a.tolist()).to('cuda:0') for a in arrays]
    refs = [torch.from_numpy(a).to('cuda:0') for a in arrays]
    row = {'program':program.__name__, 'inputs':[{'shape':a.shape,'sha256':__import__('hashlib').sha256(a.tobytes()).hexdigest()} for a in arrays]}
    for name, call in [('reference_eager',program),('reference_compiled',torch.compile(program,backend='eager',fullgraph=True)),('native_eager',program),('native_compiled',native.compile(program,backend='eager',fullgraph=True))]:
        try:
            result = call(*(refs if name.startswith('reference') else args))
            row[name] = {'status':'pass','output':result.cpu().tolist()}
        except Exception as e:
            row[name] = {'status':'error','exception':type(e).__name__,'message':str(e)}
    report['programs'].append(row)
report['inventory_after'] = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv'],text=True)
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
