import ctypes, hashlib, json, pathlib, sys
import torch
import torch_rs as native
from tests.test_cuda_add import runtime
root=pathlib.Path.cwd()
build=json.loads((root/'docs/diagnostics/composite-row-sum-glu-unflatten/postcommit-02535d5/capture-build-record.json').read_text())
ext=pathlib.Path(native._C.__file__).resolve()
assert hashlib.sha256(ext.read_bytes()).hexdigest()==build['extension_sha256']
print('verified extension',str(ext),build['extension_sha256'],flush=True)
print('reference',torch.__version__,torch.version.git_version,torch.cuda.get_device_name(0),flush=True)
lib=runtime();lib.cudaMemcpy.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int];lib.cudaMemcpy.restype=ctypes.c_int
for width in (536870912,536870916):
    x=native.zeros((1,width),device='cuda:0');ref=torch.zeros((1,width),device='cuda:0')
    for index,value in ((513002932,1e8),(279983480,-1e8),(6175757,-1.)):
        cell=ctypes.c_float(value)
        assert lib.cudaMemcpy(x.data_ptr()+index*4,ctypes.byref(cell),4,1)==0
        ref[0,index]=value
    torch.cuda.synchronize()
    for keepdim in (False,True):
        print(dict(width=width,keepdim=keepdim,native=x.sum(1,keepdim).cpu().tolist(),reference=ref.sum(1,keepdim).cpu().tolist()),flush=True)
    del x,ref
