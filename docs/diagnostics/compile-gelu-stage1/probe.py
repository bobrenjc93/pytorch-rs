"""Non-scoring stage gate: actual private native planner versus default Inductor.

Explicit SSA fixtures exercise production owners; this is neither a frontend nor
an interpreter. No public GELU callable is installed for this diagnostic.
Run reference/native legs in separate fresh processes, then compare.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
COMPARATOR = ROOT / 'docs/diagnostics/compile-pointwise-positional/warm-dispatch/gpu-dispatch.py'


def bits(value):
    return struct.unpack('<Q', struct.pack('<d', float(value)))[0]


class Fixture:
    """Append explicit private payloads, without evaluating tensor expressions."""
    def __init__(self, arity=1):
        self.nodes = [('input', i, 0, 0) for i in range(arity)]
        self.arity = arity

    def op(self, tag, a, b=0):
        self.nodes.append((tag, a, b, 0))
        return len(self.nodes) - 1

    def constant(self, value):
        tag = 'boolean' if type(value) is bool else 'integer' if type(value) is int else 'constant'
        self.nodes.append((tag, 0, 0, int(value) if tag == 'boolean' else bits(value)))
        return len(self.nodes) - 1

    def gelu(self, x):
        half = self.op('mul', x, self.constant(0.5))
        arg = self.op('mul', x, self.constant(0.70710678118654752440))
        e = self.op('erf', arg)
        one = self.op('add', self.constant(1), e)
        return self.op('mul', half, one)


def cases():
    result = []
    def save(name, g, outputs, body, history=None):
        roots = sorted(set(outputs))
        result.append(dict(name=name, nodes=g.nodes, roots=roots,
                           order=[roots.index(x) for x in outputs], arity=g.arity,
                           source='def f(x' + (', y' if g.arity == 2 else '') + ', a):\n' + body,
                           history=history or [[148, .75], [148, .75]]))
    for name in ('erf', 'gelu', 'erf_affine', 'gelu_affine', 'shared', 'nested',
                 'products_first', 'sum_first', 'mixed_trig', 'recomputed'):
        g = Fixture(2 if name in ('products_first', 'sum_first') else 1)
        if name == 'erf':
            out = [g.op('erf', 0)]; body = ' return torch.erf(x)\n'
        elif name == 'gelu':
            out = [g.gelu(0)]; body = ' return F.gelu(x)\n'
        elif name == 'erf_affine':
            e = g.op('erf', g.op('mul', 0, g.constant(.70710678118654752440)))
            out = [g.op('mul', g.op('sub', e, g.constant(.5)), g.constant(1048576.))]
            body = ' return (torch.erf(x*.70710678118654752440)-.5)*1048576.\n'
        elif name == 'gelu_affine':
            v = g.op('sub', g.op('mul', 0, g.constant(.5)), g.constant(.25))
            out = [g.op('add', g.op('mul', g.gelu(v), g.constant(3.)), g.constant(.125))]
            body = ' return F.gelu(x*.5-.25)*3.+.125\n'
        elif name == 'shared':
            v = g.gelu(0)
            out = [v, g.op('add', g.op('mul', v, v), g.op('mul', 0, g.constant(.125)))]
            body = ' g=F.gelu(x)\n return g,g*g+x*.125\n'
        elif name == 'nested':
            out = [g.gelu(g.gelu(0)), g.op('erf', g.op('erf', 0))]
            body = ' return F.gelu(F.gelu(x)),torch.erf(torch.erf(x))\n'
        elif name in ('products_first', 'sum_first'):
            v = g.gelu(0); p = g.op('mul', v, v); q = g.op('mul', 0, 1); s = g.op('add', p, q)
            out = [p, s] if name == 'products_first' else [s, p]
            body = ' g=F.gelu(x)\n p=g*g\n q=x*y\n return ' + ('p,p+q' if name == 'products_first' else 'p+q,p') + '\n'
        elif name == 'mixed_trig':
            out = [g.op('add', g.op('sin', g.op('erf', 0)), g.op('erf', g.op('cos', 0)))]
            body = ' return torch.sin(torch.erf(x))+torch.erf(torch.cos(x))\n'
        else:
            v, w = g.gelu(0), g.gelu(0)
            out = [v, w, g.op('sub', v, w)]
            body = ' g=F.gelu(x)\n h=F.gelu(x)\n return g,h,g-h\n'
        save(name, g, out, body)
    for zero in (0, False, 0.0):
        for value in (.75, .875, -0., float('inf'), float('-inf'), float('nan'), 2**-149, -(2**-149), .4769362807273865):
            g = Fixture(); z = g.op('add', g.op('mul', 0, g.constant(zero)), g.constant(value))
            e = g.op('erf', z); v = g.gelu(z)
            c = g.op('mul', g.op('sub', e, g.constant(.5)), g.constant(1048576.))
            # Also expose cancellation after GELU's surrounding constant arithmetic.
            d = g.op('mul', g.op('sub', v, g.constant(.58)), g.constant(1048576.))
            literal = repr(value) if math.isfinite(value) else 'float("'+str(value)+'")'
            save('constant_'+type(zero).__name__+'_'+float(value).hex(), g, [e, v, c, d],
                 f' z=x*{zero!r}+{literal}\n e=torch.erf(z)\n g=F.gelu(z)\n return e,g,(e-.5)*1048576.,(g-.58)*1048576.\n')
    for length in (28, 30, 32, 98, 100, 102):
        g = Fixture(); v = 0
        for _ in range(length): v = g.op('sin', v)
        e = g.op('erf', v); p = g.gelu(v)
        out = [e, g.op('add', p, g.op('mul', v, v))]
        save('threshold_'+str(length), g, out,
             ' v=x\n'+' v=torch.sin(v)\n'*length+' return torch.erf(v),F.gelu(v)+v*v\n',
             [[2,.75],[13,.75],[2,.75],[257,.75],[2,.75]])
    g = Fixture(); z = g.op('mul', 0, g.constant(0)); g.nodes.append(('scalar',0,0,0)); z = g.op('add', z, len(g.nodes)-1)
    e=g.op('erf',z); v=g.gelu(z); c=g.op('mul',g.op('sub',e,g.constant(.5)),g.constant(1048576.))
    save('scalar_history',g,[e,v,c], ' z=x*0+a\n e=torch.erf(z)\n return e,F.gelu(z),(e-.5)*1048576.\n',
         [[148,a] for a in (.75,.875,.75,-0.,0.,float('inf'),float('-inf'),float('nan'),.75)])
    return result


def fixture():
    grid = [struct.unpack('<I',struct.pack('<f', k/8))[0] for k in range(-64,65)]
    return grid+[0x80000000,1,0x80000001,0x7fffff,0x807fffff,0x800000,0x80800000,
                 0x3f7fffff,0x3f800001,0x407fffff,0x40800001,0xc07fffff,0xc0800001,
                 0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001,0xffc00001]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def record(tensor, path):
    values=tensor.cpu().tolist()
    words=[struct.unpack('<I',struct.pack('<f',x))[0] for x in values]
    raw=struct.pack('<'+'I'*len(words),*words); path.write_bytes(raw)
    return dict(shape=list(tensor.shape),stride=list(tensor.stride()),dtype=str(tensor.dtype),
                device=str(tensor.device),requiresGrad=tensor.requires_grad,offset=tensor.storage_offset(),
                values=[float(x).hex() for x in values],words=[f'{x:08x}' for x in words],
                sha256=hashlib.sha256(raw).hexdigest(),raw=path.name)


def command(*args):
    return subprocess.check_output(args,text=True).strip()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('leg',choices=['declare','native','reference','compare']); parser.add_argument('directory',type=Path)
    args=parser.parse_args(); base=args.directory.resolve(); assert base.is_relative_to(ROOT)
    matrix=cases()
    if args.leg=='declare':
        base.mkdir(parents=True,exist_ok=False)
        write(base/'matrix.json',dict(cases=matrix,calls=sum(len(c['history']) for c in matrix),
                                     leaves=sum(len(c['history'])*len(c['order']) for c in matrix),input_words=fixture(),
                                     policy_sha256=hashlib.sha256(COMPARATOR.read_bytes()).hexdigest()))
        return
    if args.leg=='compare':
        spec=importlib.util.spec_from_file_location('policy',COMPARATOR); policy=importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
        n=json.loads((base/'native/report.json').read_text()); r=json.loads((base/'reference/report.json').read_text())
        failures=[]; compared=0; bit_differences=0
        for nc,rc in zip(n['cases'],r['cases'],strict=True):
            assert nc['name']==rc['name']
            if nc.get('error') or rc.get('error'): failures.append(dict(case=nc['name'],infrastructure=[nc.get('error'),rc.get('error')])); continue
            for step,(ns,rs) in enumerate(zip(nc['steps'],rc['steps'],strict=True)):
                for leaf,(nv,rv) in enumerate(zip(ns['outputs'],rs['outputs'],strict=True)):
                    compared+=1; bit_differences+=sum(x!=y for x,y in zip(nv['words'],rv['words'],strict=True))
                    try: policy.close_outputs(nv,rv)
                    except AssertionError as error: failures.append(dict(case=nc['name'],step=step,leaf=leaf,error=str(error)))
        write(base/'comparison.json',dict(compared=compared,bit_differences=bit_differences,failures=failures,passed=not failures))
        print(json.dumps(dict(compared=compared,failures=len(failures),examples=failures[:8])))
        return
    out=base/args.leg; out.mkdir(exist_ok=False)
    for name in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','XDG_CACHE_HOME','TMPDIR'):
        p=out/name; p.mkdir(); os.environ[name]=str(p)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    report=dict(leg=args.leg,cases=[],before=command('nvidia-smi','--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used','--format=csv'),
                source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*pointwise*.rs')},
                nvcc=command('nvcc','--version'),python=sys.version)
    if args.leg=='native':
        import torch_rs as fw
        from torch_rs import torch_rs as bridge
        report['extension']=fw.__file__
    else:
        import torch as fw
        from torch._inductor.utils import run_and_get_code
        assert fw.__version__=='2.13.0+cu130'
    fw.set_num_threads(1); report['framework_version']=fw.__version__
    for case in matrix:
        row=dict(name=case['name'],steps=[]); report['cases'].append(row)
        directory=out/case['name']; directory.mkdir()
        try:
            nodes=tuple(tuple(n) for n in case['nodes']); roots=tuple(case['roots']); order=tuple(case['order'])
            if args.leg=='reference':
                namespace=dict(torch=fw,F=fw.nn.functional); exec(case['source'],namespace); compiled=fw.compile(namespace['f'])
            kernels={}; preparations={}; retained=[]
            for step,(count,scalar) in enumerate(case['history']):
                words=(fixture()*((count+147)//148))[:count]
                if step%2: words.reverse()
                values=[struct.unpack('<f',struct.pack('<I',w))[0] for w in words]
                x=fw.tensor(values,dtype=fw.float32).to('cuda:0'); inputs=(x,)
                if case['arity']==2: inputs+=(fw.tensor([-v for v in values],dtype=fw.float32).to('cuda:0'),)
                observation=dict(inputs=[record(t,directory/f'{step}-input-{i}.u32le') for i,t in enumerate(inputs)],scalar=float(scalar).hex())
                if args.leg=='reference':
                    result,code=run_and_get_code(compiled,*inputs,scalar)
                    for i,text in enumerate(code): (directory/f'{step}-wrapper-{i}.py').write_text(text)
                    outputs=result if type(result) is tuple else (result,)
                else:
                    current=nodes
                    runtime=case['name']=='scalar_history' and step>0
                    if case['name']=='scalar_history' and not runtime:
                        current=tuple(('constant',0,0,bits(scalar)) if n[0]=='scalar' else n for n in nodes)
                    if current not in kernels:
                        kernel=bridge._pointwise_compile(inputs,current,roots); kernels[current]=kernel
                        (directory/f'{step}-native.cu').write_text(kernel.source); (directory/f'{step}-native.ptx').write_text(kernel.ptx)
                        row['nvrtc']=kernel.nvrtc_version; row['options']=kernel.options
                    kernel=kernels[current]
                    # Existing frontend promotes shape hints after the first distinct shape.
                    hint=count if step==0 or count==case['history'][0][0] and len({s[0] for s in case['history'][:step+1]})==1 else case['history'][1][0]
                    key=(current,count,hint)
                    if key not in preparations: preparations[key]=kernel.prepare(inputs,hint,order)
                    prepared=preparations[key]
                    (directory/f'{step}-plan.txt').write_text(kernel.plan(hint,order))
                    assert bridge._pointwise_plan(current,roots,case['arity'],[list(t.shape) for t in inputs],hint,order)==kernel.plan(hint,order)
                    assert bridge._pointwise_source(current,roots,case['arity'],[list(t.shape) for t in inputs])==kernel.source
                    result=prepared.run(inputs,(scalar,) if runtime else ())
                    outputs=tuple(result[i] for i in order)
                    observation['preparations']=len(preparations)
                observation['outputs']=[record(t,directory/f'{step}-output-{i}.u32le') for i,t in enumerate(outputs)]
                assert all(t.data_ptr()!=x.data_ptr() for t in outputs)
                assert len({t.data_ptr() for t in outputs})==len(outputs)
                for previous,records in retained:
                    assert [record(t,directory/f'retained-{step}-{i}.u32le')['words'] for i,t in enumerate(previous)]==[r['words'] for r in records]
                retained.append((outputs,observation['outputs'])); row['steps'].append(observation)
        except Exception: row['error']=traceback.format_exc()
        write(out/'report.json',report); print(args.leg,case['name'],'error' if row.get('error') else 'captured',flush=True)
    if args.leg=='native': assert 'torch' not in sys.modules
    report['mapped_libraries']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if any(s in line for s in ('libcuda','libnvrtc'))} if Path(p).is_file()}
    report['after']=command('nvidia-smi','--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used','--format=csv')
    write(out/'report.json',report)


if __name__=='__main__': main()
