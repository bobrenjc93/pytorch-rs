import json, torch
from pathlib import Path
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
torch.set_num_threads(1)
records=[]
for body,history in (
 ('x*s',[(2,),(3,),(2,)]),
 ('x*s',[(2,3),(4,3),(2,3),(4,5),(2,3)]),
 ('x*s',[(2,2),(3,3),(3,4),(2,2)]),
):
 ns={};exec('def f(s,x):\n return '+body,ns);f=ns['f'];c=torch.compile(f)
 for i,shape in enumerate(history):
  scalar=-0. if i%2 else 0.
  result=c(scalar,torch.ones(shape,device='cuda'))
  entries=_debug_get_cache_entry_list(f)
  record={'shape':shape,'scalar':repr(scalar),'negative':result.signbit().all().item(),'guards':[]}
  for e in entries:
   g=e.guard_manager
   record['guards'].append(str(g))
  records.append(record)
 torch.compiler.reset()
Path('target/checks/shape-guards.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
