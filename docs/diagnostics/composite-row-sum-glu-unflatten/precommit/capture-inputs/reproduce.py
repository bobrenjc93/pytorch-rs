"""Precommit numerical receipts on the extension bound by the build record."""
import hashlib, json, pathlib, sys
import numpy as np
import torch
import torch_rs as native
from tests.test_cuda_add import upload
root=pathlib.Path.cwd()
receipt=json.loads((root/'target/integration/capture/build-record.json').read_text())
ext=pathlib.Path(native._C.__file__).resolve()
assert hashlib.sha256(ext.read_bytes()).hexdigest()==receipt['extension_sha256']
print('build_record:',root/'target/integration/capture/build-record.json')
print('extension:',ext,receipt['extension_sha256'])
print('interpreter:',sys.executable,sys.version)
print('reference:',torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))
for rows in (1,2,17):
    patterns=[[1e8,1,1,-1e8]]
    for m in (np.float32(3e38),np.finfo(np.float32).max):
        patterns += [[m,0,m,-m],[m,-m,m,-m]]
    for pattern in patterns:
        values=np.tile(np.asarray(pattern,dtype=np.float32),(rows,1))
        a,b=[upload(module,values.ravel(),values.shape) for module in (native,torch)]
        for keepdim in (False,True):
            actual,expected=a.sum(1,keepdim).cpu().tolist(),b.sum(1,keepdim).cpu().tolist()
            np.testing.assert_array_equal(actual,expected)
            print(dict(rows=rows,pattern=[float(v) for v in pattern],keepdim=keepdim,native=actual,reference=expected))
for width in (65539,1_000_000,1_000_003):
    for value in (0.1,-0.1,0.01,1.1):
        a=native.full((2,width),value).to('cuda:0')
        b=torch.full((2,width),value,device='cuda:0')
        actual,expected=a.sum(1).cpu().tolist(),b.sum(1).cpu().tolist()
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-4)
        print(dict(width=width,value=value,native=actual,reference=expected))
print('GLU:')
for module in (native,torch):
    x=module.tensor([1e20,-20.],requires_grad=True)
    (module.nn.functional.glu(x)*1e20).sum().backward()
    result=x.grad.tolist()
    np.testing.assert_allclose(result,[2.0611537e11,2.0611537e31],rtol=2e-6)
    print(module.__name__,result)
print('all numerical regressions passed; correctness only')
