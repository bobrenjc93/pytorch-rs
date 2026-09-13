import torch, json, struct
from pathlib import Path
torch.set_num_threads(1)
records=[]
for origin in ('parameter','global','closure'):
 for body in ('((x*0)+s)-16777216.0','x*s'):
  for seq in ([16777217.,16777218.,16777217.,float('inf'),float('nan'),-0.,0.,1e300],[-0.,0.,-0.,1.25,0.,-0.],[False,True,0.,False,1.,True], [float('inf'),16777217.,float('nan'),16777217.,16777218.]):
   ns={}
   if origin=='parameter':
    exec('def f(s,x):\n return '+body,ns); f=ns['f']
   elif origin=='global':
    exec('def f(x):\n return '+body,ns); f=ns['f']
   else:
    exec('def factory():\n s=1.\n def f(x):\n  return '+body+'\n def setter(v):\n  nonlocal s\n  s=v\n return f,setter',ns)
    f,setter=ns['factory']()
   c=torch.compile(f)
   for v in seq:
    x=torch.tensor([1.,-1.,0.,-0.,3e38],device='cuda')
    if origin=='parameter': y=c(v,x)
    else:
     if origin=='global': ns['s']=v
     else: setter(v)
     y=c(x)
    torch.cuda.synchronize()
    records.append(dict(origin=origin,body=body,value=repr(v),kind=type(v).__name__,bits=y.cpu().view(torch.int32).tolist()))
   torch.compiler.reset()
Path('target/checks/reference-probe.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
