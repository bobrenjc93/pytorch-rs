import torch,torch_rs as native,json
from pathlib import Path
torch.set_num_threads(1)
records=[]
assert 'target/main-binding-probe-package' in native.__file__
print('baseline native package:',native.__file__)
for origin in ('capture',):
 for initial in (0.,-0.):
  ns={}
  exec('def f(s,x):\n return x*s' if origin=='parameter' else 'def f(x):\n return x*s',ns)
  fn=ns['f']; rn={};exec('def f(s,x):\n return x*s' if origin=='parameter' else 'def f(x):\n return x*s',rn)
  nc,rc=native.compile(fn),torch.compile(rn['f'])
  for shape,scale in ((2,initial),(3,-initial),(2,initial),(4,-initial),(2,initial)):
   nx=native.tensor([1.]*shape).to('cuda:0');rx=torch.ones(shape,device='cuda')
   if origin=='parameter': a,e=nc(scale,nx),rc(scale,rx)
   else:
    ns['s']=rn['s']=scale;a,e=nc(nx),rc(rx)
   actual=torch.tensor(a.cpu().tolist()).view(torch.int32).tolist();expected=e.cpu().view(torch.int32).tolist()
   records.append(dict(origin=origin,initial=repr(initial),shape=shape,scale=repr(scale),actual=actual,expected=expected,match=actual==expected))
  native.compiler.reset();torch.compiler.reset()
Path('target/checks/main-capture-shape-probe.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
