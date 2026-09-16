"""Bounded public GELU diagnostic, separate eager and default-compiled comparisons.

Non-scoring: no fixed evaluator/corpus is modified or replaced. Timing reports
first-call cost per wrapper and steady calls, not process-cold cost for every
row. Earlier rows may warm module/compiler caches. Native body-replay checks run
outside timing. One wrapper executes each complete numerical history.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path(__file__).with_name('probe.py')
spec = importlib.util.spec_from_file_location('private_diagnostic_helpers', PRIVATE)
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
LEGS = ('native-eager', 'reference-eager', 'native-compiled', 'reference-compiled', 'native-no-nvrtc')


def cases():
    ordinary = [dict(shape=[148], scalar=.75, offset=0, reverse=False),
                dict(shape=[148], scalar=.75, offset=0, reverse=True)]
    definitions = [
        ('namespace', ' return fw.nn.functional.gelu(x)\n', 1),
        ('functional_alias', ' return F.gelu(x)\n', 1),
        ('function_alias', ' return gelu(x)\n', 1),
        ('nn_alias', ' return NN.functional.gelu(x)\n', 1),
        ('shared', ' g=F.gelu(x)\n return g,g*g+x*.125\n', 1),
        ('nested', ' return F.gelu(F.gelu(x))\n', 1),
        ('products_first', ' g=F.gelu(x)\n p=g*g\n q=x*y\n return p,p+q\n', 2),
        ('sum_first', ' g=F.gelu(x)\n p=g*g\n q=x*y\n return p+q,p\n', 2),
        ('mixed_trig', ' return fw.sin(F.gelu(x))+fw.cos(x)\n', 1),
        ('recomputed', ' g=F.gelu(x)\n h=F.gelu(x)\n return g,h,g-h\n', 1),
    ]
    result = [dict(name=name, arity=arity, source='def f(x'+(', y' if arity==2 else '')+', a):\n'+body,
                   history=ordinary) for name, body, arity in definitions]
    result.append(dict(name='shape_history', arity=1, source='def f(x,a):\n return F.gelu(x)\n',
                       history=[dict(shape=[n], scalar=.75, offset=0, reverse=bool(i%2)) for i,n in enumerate((2,13,2,257,2))]))
    result.append(dict(name='scalar_history', arity=1, source='def f(x,a):\n return F.gelu(x*0+a)\n',
                       history=[dict(shape=[148], scalar=a, offset=0, reverse=bool(i%2))
                                for i,a in enumerate((.75,.875,.75,-0.,0.,float('inf'),float('-inf'),float('nan'),.75))]))
    result.append(dict(name='metadata', arity=1, source='def f(x,a):\n return F.gelu(x)\n',
                       history=[dict(shape=shape, scalar=.75, offset=offset, reverse=False)
                                for shape,offset in (([],0),([0],0),([2,3],0),([2,1,3],0),([4],2),([],0))]))
    for zero in (0,False,0.0):
        for value in (.75,.875,-0.,float('inf'),float('-inf'),float('nan'),2**-149,-(2**-149),.4769362807273865):
            # Nonfinite values are immutable scalar globals; calling float(str)
            # inside the captured function is outside this compiler's surface.
            literal=repr(value) if math.isfinite(value) else 'constant_value'
            result.append(dict(name='constant_'+type(zero).__name__+'_'+float(value).hex(),arity=1,
                               source=f'def f(x,a):\n z=x*{zero!r}+{literal}\n g=F.gelu(z)\n return g,(g-.58)*1048576.\n',
                               bindings={'constant_value':value},history=ordinary))
    for length in (28,30,32,98,100,102):
        result.append(dict(name='threshold_'+str(length),arity=1,
                           source='def f(x,a):\n v=x\n'+' v=fw.sin(v)\n'*length+' g=F.gelu(v)\n return g,g+v*v\n',
                           history=[dict(shape=[n],scalar=.75,offset=0,reverse=bool(i%2)) for i,n in enumerate((2,13,2,257,2))]))
    for case in result:
        returned=ast.parse(case['source']).body[0].body[-1].value
        case['outputs']=len(returned.elts) if isinstance(returned,ast.Tuple) else 1
    return result


def timing_cases():
    return [dict(name=name, source='def f(x,a):\n'+body, shape=[n])
            for name, body in (
                ('gelu', ' return F.gelu(x)\n'),
                ('affine_control', ' return x*.75+x\n'),
                ('trig_control', ' return fw.sin(x)+fw.cos(x)\n'))
            for n in (257, 131072)]


def leg_matrix(leg):
    compiled=leg.endswith('compiled')
    eager_names={'namespace','functional_alias','function_alias','nn_alias','nested','shape_history','metadata'}
    selected=[c for c in cases() if compiled or c['name'] in eager_names]
    timings=[c for c in timing_cases() if compiled or c['name']!='trig_control']
    return dict(cases=selected,timings=timings,calls=sum(len(c['history']) for c in selected),
                history_leaves=sum(len(c['history'])*c['outputs'] for c in selected),
                timing_leaves=len(timings)*3)


def declaration():
    return dict(legs={leg:leg_matrix(leg) for leg in LEGS}, warmups=5, samples=17,
                subset_reason='Ordinary native eager supports bounded GELU, nesting, aliases and metadata; scalar-add/tensor-product/trig compositions are compiled-only. Both reference modes use their corresponding native matrix. Affine eager control uses supported scalar multiplication and tensor addition. This diagnostic is not a fixed scoring corpus.',
                private_helper=p.identity(PRIVATE), policy=p.identity(p.POLICY),
                source=p.identity(Path(__file__)), seed='fixed historical binary32 fixture; alternating reversal')


def values(count, reverse=False, finite=False):
    words = p.fixtures.fixture()
    if finite:
        words = [struct.unpack('<I',struct.pack('<f',k/8))[0] for k in range(-32,33)]
    words = (words*((count+len(words)-1)//len(words)))[:count]
    if reverse: words.reverse()
    return [struct.unpack('<f',struct.pack('<I',w))[0] for w in words]


def tensor(fw, shape, reverse=False, offset=0, finite=False, negate=False):
    n = math.prod(shape)
    data = values(n+offset, reverse, finite)
    if negate: data = [-v for v in data]
    result = fw.tensor(data, dtype=fw.float32).to('cuda:0')
    if offset: result = result[offset:offset+n]
    return result.reshape(tuple(shape))


def record(reader, t, path):
    shape = list(t.shape); n = math.prod(shape)
    if n:
        result = reader.record(t.reshape((n,)), path)
    else:
        path.write_bytes(b'')
        result = dict(pointer=t.data_ptr(), values=[], words=[], sha256=hashlib.sha256(b'').hexdigest(), raw=path.name)
    result.update(shape=shape, stride=list(t.stride()), dtype=str(t.dtype), device=str(t.device),
                  requiresGrad=t.requires_grad, offset=t.storage_offset())
    return result


def function(fw, source, bindings=None):
    namespace = dict(fw=fw, F=fw.nn.functional, NN=fw.nn, gelu=fw.nn.functional.gelu)
    namespace.update(bindings or {})
    exec(source, namespace)
    return namespace['f']


def checked_call(callable_, original, args, native_compiled, eager_gelu):
    if not native_compiled:
        return callable_(*args)
    def forbid(frame, event, arg):
        if event == 'call' and frame.f_code is original.__code__:
            raise AssertionError('original Python body replayed by native compiler')
        if event == 'c_call' and arg is eager_gelu:
            raise AssertionError('native compiler called the eager GELU builtin')
    previous = sys.getprofile(); sys.setprofile(forbid)
    try:
        return callable_(*args)
    finally:
        sys.setprofile(previous)


def outputs(result):
    return result if type(result) is tuple else (result,)


def record_kernels(target,out):
    cache=target._torch_rs_pointwise_cache
    artifacts=[]
    for kernel in cache.executors.values():
        key=hashlib.sha256(kernel.ptx.encode()).hexdigest()
        directory=out/'native-kernels'/key
        directory.mkdir(parents=True,exist_ok=True)
        for name,text in (('executor.cu',kernel.source),('executor.ptx',kernel.ptx)):
            path=directory/name
            if not path.exists(): path.write_text(text)
            assert path.read_text()==text
        artifacts.append(dict(source=p.identity(directory/'executor.cu'),ptx=p.identity(directory/'executor.ptx'),
                              nvrtc_version=kernel.nvrtc_version,options=kernel.options))
    return dict(executors=artifacts,graphs=len(cache.graphs),preparations=len(cache.prepared),
                preparation_retained_bytes=cache.prepared_bytes,
                scope='Passive native cache observation; preparation bytes exclude shared kernels')


def source_identity():
    result = p.source_identities()
    result[str(Path(__file__).relative_to(ROOT))] = p.sha(Path(__file__))
    result.update({str(path.relative_to(ROOT)): p.sha(path) for path in (ROOT/'src/cuda').glob('*gelu*') if path.is_file()})
    return result


def run(base, leg):
    declared = json.loads((base/'matrix.json').read_text())
    assert p.canonical(declared['matrix']) == p.canonical(declaration())
    out = base/leg; out.mkdir()
    for name in p.CACHE_NAMES:
        path=out/name; path.mkdir(); os.environ[name]=str(path)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    if leg=='native-no-nvrtc':
        missing=base/'intentionally-absent-nvrtc.so'
        assert not missing.exists()
        os.environ['TORCH_RS_NVRTC']=str(missing)
    native=leg.startswith('native'); compiled=leg.endswith('compiled')
    matrix=declared['matrix']['legs'][leg]
    report=dict(leg=leg, started_at=p.now(), source=source_identity(), before=p.gpu_snapshot(), cases=[], timings=[],
                declaration=p.identity(base/'matrix.json'), git_commit=p.command('git','rev-parse','HEAD'),
                git_status=p.command('git','status','--porcelain'), root=str(ROOT), python=p.identity(sys.executable),
                environment={k:os.environ.get(k) for k in p.ENV_NAMES+p.CACHE_NAMES},
                matrix=matrix,subset_reason=declared['matrix']['subset_reason'],
                timing_scope='One fresh process per leg; first call per wrapper; global caches may be warm after earlier rows; five warmups and seventeen synchronized unprofiled samples')
    p.write(out/'report.json',report)
    if native:
        import torch_rs as fw
        from torch_rs import torch_rs as bridge
        report['extension']=p.identity(bridge.__file__)
        assert Path(bridge.__file__).resolve().is_relative_to(ROOT)
        assert report['extension'] in p.mapped_libraries()
        if compiled:
            provider=ROOT/'src/cuda/erf_provider.ptx'
            shutil.copyfile(provider,out/'erf_provider.ptx')
            report['linked_provider']=dict(input=p.identity(provider),retained=p.identity(out/'erf_provider.ptx'))
    else:
        import torch as fw
        assert fw.__version__=='2.13.0+cu130'
        report['reference_cuda']=fw.version.cuda
    fw.set_num_threads(1); reader=p.DeviceBytes()
    report['framework_version']=fw.__version__
    seen={}; cache_history=[]
    # Timings precede correctness histories so the first GELU call includes lazy
    # eager module loading or the first compiler/provider link in this process.
    for index,case in enumerate(matrix['timings']):
        row=dict(name=case['name'], shape=case['shape'], started_at=p.now()); report['timings'].append(row)
        directory=out/f'timing-{index}'; directory.mkdir()
        try:
            x=tensor(fw,case['shape'],finite=True)
            row['input']=record(reader,x,directory/'input.u32le')
            original=function(fw,case['source'],case.get('bindings'))
            target=fw.compile(original) if compiled else original
            reader.call('cuCtxSynchronize'); start=time.perf_counter_ns()
            result=target(x,.75)
            reader.call('cuCtxSynchronize'); row['first_call_ns']=time.perf_counter_ns()-start
            row['cold_outputs']=[record(reader,t,directory/f'cold-{i}.u32le') for i,t in enumerate(outputs(result))]
            for _ in range(5):
                result=target(x,.75); reader.call('cuCtxSynchronize')
            samples=[]
            for _ in range(17):
                reader.call('cuCtxSynchronize'); start=time.perf_counter_ns()
                result=target(x,.75)
                reader.call('cuCtxSynchronize'); samples.append(time.perf_counter_ns()-start)
            row['samples_ns']=samples; row['median_ns']=statistics.median(samples)
            row['outputs']=[record(reader,t,directory/f'warm-{i}.u32le') for i,t in enumerate(outputs(result))]
            changed=tensor(fw,case['shape'],reverse=True,finite=True)
            changed_result=checked_call(target,original,(changed,.75),native and compiled,fw.nn.functional.gelu)
            row['changed_input']=record(reader,changed,directory/'changed-input.u32le')
            row['changed_outputs']=[record(reader,t,directory/f'changed-{i}.u32le') for i,t in enumerate(outputs(changed_result))]
            row['input_after']=record(reader,x,directory/'input-after.u32le')
            assert row['input']['sha256']==row['input_after']['sha256']
            row['body_replay_check']=native and compiled
            if native and compiled: row['native_cache']=record_kernels(target,out)
        except Exception:
            row['error']=traceback.format_exc()
        row['finished_at']=p.now(); p.write(out/'report.json',report)
        cache_history.append(p.observe_cache(out,seen,case['name'],f'timing-{index}'))
        p.write(out/'cache-index.json',dict(files=seen,history=cache_history))
    for case in matrix['cases']:
        row=dict(name=case['name'],steps=[]); report['cases'].append(row)
        directory=out/case['name']; directory.mkdir()
        try:
            original=function(fw,case['source'],case.get('bindings'))
            target=fw.compile(original) if compiled else original
            retained=[]
            for call,h in enumerate(case['history']):
                ts=(tensor(fw,h['shape'],h['reverse'],h['offset']),)
                if case['arity']==2: ts+=(tensor(fw,h['shape'],h['reverse'],h['offset'],negate=True),)
                observation=dict(call=call,started_at=p.now(),scalar=float(h['scalar']).hex(),
                                 inputs=[record(reader,t,directory/f'{call}-input-{i}.u32le') for i,t in enumerate(ts)])
                row['steps'].append(observation)
                result=checked_call(target,original,(*ts,h['scalar']),native and compiled,fw.nn.functional.gelu)
                actual=outputs(result)
                observation['outputs']=[record(reader,t,directory/f'{call}-output-{i}.u32le') for i,t in enumerate(actual)]
                observation['inputs_after']=[record(reader,t,directory/f'{call}-input-after-{i}.u32le') for i,t in enumerate(ts)]
                assert [r['sha256'] for r in observation['inputs']]==[r['sha256'] for r in observation['inputs_after']]
                if native and math.prod(h['shape']):
                    assert {t.data_ptr() for t in ts}.isdisjoint(t.data_ptr() for t in actual)
                observation['retained']=[]
                for prior,old,records in retained:
                    snapshots=[record(reader,t,directory/f'retained-current-{call}-prior-{prior}-{i}.u32le') for i,t in enumerate(old)]
                    assert [r['sha256'] for r in snapshots]==[r['sha256'] for r in records]
                    observation['retained'].append(dict(prior_call=prior,outputs=snapshots))
                retained.append((call,actual,observation['outputs']))
                observation['finished_at']=p.now()
                if native and compiled: observation['native_cache']=record_kernels(target,out)
                cache_history.append(p.observe_cache(out,seen,case['name'],call))
            row['body_replay_check']=native and compiled
        except Exception:
            row['error']=traceback.format_exc()
        p.write(out/'report.json',report)
        p.write(out/'cache-index.json',dict(files=seen,history=cache_history))
        print(leg,case['name'],'error' if row.get('error') else 'captured',flush=True)
    report['mapped_libraries']=p.mapped_libraries()
    report['runtime_versions']=p.runtime_versions(report['mapped_libraries'])
    if native:
        assert 'torch' not in sys.modules
        assert p.identity(bridge.__file__)==report['extension']
    if leg=='native-no-nvrtc':
        assert not any('libnvrtc' in item['path'] for item in report['mapped_libraries']), 'no-NVRTC eager loaded NVRTC'
    report['source_after']=source_identity(); report['source_unchanged']=report['source']==report['source_after']
    report['after']=p.gpu_snapshot(); report['finished_at']=p.now()
    report['passed_execution']=report['source_unchanged'] and not any(c.get('error') for c in report['cases']+report['timings'])
    p.write(out/'report.json',report)
    return 0 if report['passed_execution'] else 1


def verify_raw(value,directory):
    if isinstance(value,dict):
        if 'raw' in value:
            raw=(directory/value['raw']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==value['sha256']
            assert len(raw)==math.prod(value['shape'])*4
            assert [f'{x[0]:08x}' for x in struct.iter_unpack('<I',raw)]==value['words']
            assert [float(x[0]).hex() for x in struct.iter_unpack('<f',raw)]==value['values']
        else:
            for child in value.values(): verify_raw(child,directory)
    elif isinstance(value,list):
        for child in value: verify_raw(child,directory)


def compare(base):
    policy=p.load(p.POLICY,'unchanged_policy',p.POLICY_SHA)
    declared=json.loads((base/'matrix.json').read_text())
    assert p.canonical(declared['matrix'])==p.canonical(declaration())
    reports={}
    for leg in LEGS:
        receipt=json.loads((base/f'{leg}-exit.json').read_text()); assert receipt['returncode']==0,(leg,receipt)
        report=json.loads((base/leg/'report.json').read_text()); assert report['passed_execution']
        matrix=declared['matrix']['legs'][leg]
        assert [c['name'] for c in report['cases']]==[c['name'] for c in matrix['cases']]
        assert [(r['name'],r['shape']) for r in report['timings']]==[(r['name'],r['shape']) for r in matrix['timings']]
        for case in report['cases']: verify_raw(case,base/leg/case['name'])
        for i,row in enumerate(report['timings']): verify_raw(row,base/leg/f'timing-{i}')
        reports[leg]=report
    pairs=[]
    for native,reference in (('native-eager','reference-eager'),('native-compiled','reference-compiled'),('native-no-nvrtc','reference-eager')):
        result=dict(native=native,reference=reference,failures=[],compared=0,timings=[])
        def check(a,b,location):
            result['compared']+=1
            try: policy.close_outputs(a,b)
            except AssertionError as error: result['failures'].append(dict(location=location,error=str(error)))
        assert p.canonical(leg_matrix(native))==p.canonical(leg_matrix(reference))
        for definition,nc,rc in zip(leg_matrix(native)['cases'],reports[native]['cases'],reports[reference]['cases'],strict=True):
            assert len(nc['steps'])==len(rc['steps'])==len(definition['history'])
            for step,(ns,rs) in enumerate(zip(nc['steps'],rc['steps'],strict=True)):
                assert [r['words'] for r in ns['inputs']]==[r['words'] for r in rs['inputs']]
                assert len(ns['outputs'])==len(rs['outputs'])==definition['outputs']
                for leaf,(nv,rv) in enumerate(zip(ns['outputs'],rs['outputs'],strict=True)):
                    check(nv,rv,f"{nc['name']}/{step}/{leaf}")
        for i,(nr,rr) in enumerate(zip(reports[native]['timings'],reports[reference]['timings'],strict=True)):
            assert nr['input']['words']==rr['input']['words'] and nr['changed_input']['words']==rr['changed_input']['words']
            before=len(result['failures'])
            for kind in ('cold_outputs','outputs','changed_outputs'):
                for leaf,(nv,rv) in enumerate(zip(nr[kind],rr[kind],strict=True)): check(nv,rv,f'timing/{i}/{kind}/{leaf}')
            result['timings'].append(dict(name=nr['name'],shape=nr['shape'],correct=len(result['failures'])==before,
                                          native_first_ns=nr['first_call_ns'],reference_first_ns=rr['first_call_ns'],
                                          native_median_ns=nr['median_ns'],reference_median_ns=rr['median_ns'],
                                          reference_over_native=rr['median_ns']/nr['median_ns']))
        expected=leg_matrix(native)
        assert result['compared']==expected['history_leaves']+expected['timing_leaves']
        result['passed']=not result['failures']; pairs.append(result)
    result=dict(pairs=pairs,passed=all(pair['passed'] for pair in pairs),measured_at=p.now(),policy_sha256=p.POLICY_SHA)
    p.write(base/'comparison.json',result); print(json.dumps(result)); return 0 if result['passed'] else 1


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('leg',choices=('declare','compare','pack')+LEGS)
    parser.add_argument('directory',type=Path); parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(); base=p.inside(args.directory)
    if args.leg=='declare':
        base.mkdir(parents=True,exist_ok=False)
        p.write(base/'matrix.json',dict(declared_at=p.now(),matrix=declaration()))
        shutil.copyfile(__file__,base/'public_probe.py'); return 0
    if args.leg=='compare': return compare(base)
    if args.leg=='pack':
        import tarfile
        paths=[f for f in sorted(base.rglob('*')) if f.is_file() and not any(name in f.relative_to(base).parts for name in p.CACHE_NAMES)]
        archive=base.parent/(base.name+'-essential.tar.gz')
        with tarfile.open(archive,'w:gz') as tar:
            for path in paths: tar.add(path,arcname=str(path.relative_to(base)),recursive=False)
        p.write(base.parent/(base.name+'-manifest.json'),dict(archive=p.identity(archive),files={str(f.relative_to(base)):p.identity(f) for f in paths}))
        return 0
    if args.worker: return run(base,args.leg)
    receipt=dict(started_at=p.now(),command=[sys.executable,'-B',str(Path(__file__).resolve()),args.leg,str(base),'--worker'],
                 cwd=str(ROOT),environment={k:os.environ.get(k) for k in p.ENV_NAMES})
    with (base/f'{args.leg}.log').open('w') as log:
        process=subprocess.run(receipt['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    receipt.update(returncode=process.returncode,finished_at=p.now()); p.write(base/f'{args.leg}-exit.json',receipt)
    return process.returncode


if __name__=='__main__': raise SystemExit(main())
