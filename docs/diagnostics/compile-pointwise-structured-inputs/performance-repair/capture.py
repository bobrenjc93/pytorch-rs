"""One-process, non-scoring frontend diagnostic; see DECLARATION.md first."""
import argparse
import cProfile
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pstats
import random
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--frontend', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--profile', action='store_true')
args = parser.parse_args()
source, output = Path(args.frontend).resolve(), Path(args.output).resolve()
assert source.is_relative_to(ROOT) and output.is_relative_to(ROOT)
import torch_rs as fw
from torch_rs import torch_rs as bridge
# Each process replaces the module before creating any compiled wrapper.
spec = importlib.util.spec_from_file_location('torch_rs._compile_pointwise', source)
frontend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = frontend
spec.loader.exec_module(frontend)
fw._compile_pointwise = frontend
from tests.test_compile_pointwise_helpers import no_bodies
assert fw._compile_pointwise is frontend
assert Path(bridge.__file__).resolve().is_relative_to(ROOT)
fw.set_num_threads(1)
runtime_path = Path(sys.prefix) / 'lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13'
runtime = ctypes.CDLL(str(runtime_path))
def sync():
    assert runtime.cudaDeviceSynchronize() == 0

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git(*words):
    return subprocess.check_output(['git', *words], text=True).strip()

def implementation_of(wrapper):
    for cell in wrapper.__closure__ or ():
        value = cell.cell_contents
        if callable(value) and getattr(value, '__globals__', None) is frontend.__dict__:
            return value
    raise AssertionError('public compile did not select the requested frontend')

cases = [
    ('flat_one', 'def f(x):\n return ((x*0.75-1.125).relu(),x)'),
    ('flat_two', 'def f(x,y):\n return (x*1.25-y*0.5,x)'),
    ('flat_runtime', 'def f(gain,x,bias):\n return ((x*gain-bias).relu(),x)'),
    ('flat_shape', 'def f(gain,x,bias):\n if x.shape[0]<128:\n  return (x*gain-bias,x)\n return (x*gain+bias,x)'),
    ('structured_small', 'def f(p):\n x=p["x"]\n return (x*p["gain"]-p["bias"],x)'),
    ('structured_wide', 'def f(p):\n x=p["x"]\n return (x*p["gain"]-p["bias"],x)'),
]
seed = [(37,.375,.125),(521,.625,.25),(2053,.875,.375),(37,1.125,.5),(521,.625,.25)]
record = {'kind':'common-native/frontend comparison; unscored', 'frontend':str(source),
          'frontend_sha256':sha(source), 'script':str(Path(__file__).resolve()),
          'script_sha256':sha(__file__), 'declaration_sha256':sha(Path(__file__).with_name('DECLARATION.md')),
          'commit':git('rev-parse','HEAD'),'status':git('status','--short'),
          'python':sys.version,'executable':sys.executable,'package':fw.__file__,
          'package_sha256':sha(fw.__file__),'extension':bridge.__file__,
          'extension_sha256':sha(bridge.__file__),'profile':args.profile,
          'gpu_before':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,compute_cap,utilization.gpu,memory.used','--format=csv'],text=True),
          'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TMPDIR','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','XDG_CACHE_HOME','PYTHONHASHSEED')},'cases':[]}
try:
    rng=random.Random(19471)
    for name,code in cases:
        row={'name':name,'program':code,'seed':seed,'blocks_ns':[]}
        record['cases'].append(row)
        namespace={};exec(code,namespace);fn=namespace['f'];compiled=fw.compile(fn)
        selected=implementation_of(compiled)
        assert selected.__code__.co_filename == str(source)
        row['selected_implementation_file']=selected.__code__.co_filename
        inputs={}
        for size in (37,521,2053):
            xdata=[rng.randint(-24,24)/8 for _ in range(size)]
            ydata=[rng.randint(-16,16)/8 for _ in range(size)]
            x=fw.tensor(xdata,dtype=fw.float32).to('cuda:0')
            y=fw.tensor(ydata,dtype=fw.float32).to('cuda:0')
            inputs[size]=(x,y,xdata,ydata)
        def call_state(state):
            size,gain,bias=state;x,y,xdata,ydata=inputs[size]
            if name=='flat_one':
                operands=(x,);expected=[max(v*.75-1.125,0) for v in xdata]
            elif name=='flat_two':
                operands=(x,y);expected=[v*1.25-w*.5 for v,w in zip(xdata,ydata)]
            elif name=='flat_runtime':
                operands=(gain,x,bias);expected=[max(v*gain-bias,0) for v in xdata]
            elif name=='flat_shape':
                operands=(gain,x,bias);expected=[v*gain+(-bias if size<128 else bias) for v in xdata]
            else:
                p={'x':x,'gain':gain,'bias':bias}
                if name=='structured_wide':p['unused']=[False]*4090
                operands=(p,);expected=[v*gain-bias for v in xdata]
            return operands,x,expected
        states=[call_state(s) for s in seed]
        def check(actual,state):
            _,x,expected=state
            assert actual[1] is x
            assert tuple(actual[0].shape)==tuple(x.shape) and actual[0].dtype==x.dtype
            assert str(actual[0].device)==str(x.device)
            assert actual[0].cpu().tolist()==expected
        sync();start=time.perf_counter_ns()
        try:
            actual=compiled(*states[0][0]);sync()
        except NotImplementedError as exc:
            row.update(status='unsupported',error=str(exc))
            if name.startswith('structured') and 'InputTree' not in frontend.__dict__:
                continue
            raise
        row['cold_completed_ns']=time.perf_counter_ns()-start
        check(actual,states[0])
        with no_bodies(fn):
            for state in states[1:]:check(compiled(*state[0]),state)
            for i in range(30):check(compiled(*states[i%3][0]),states[i%3])
        row['logical_specializations']=len(compiled._torch_rs_pointwise_cache.graphs)
        # Prebuild argument/expectation cycles; keep all bookkeeping out of timing.
        calls=[states[i%3] for i in range(300)]
        if args.profile:
            profile=cProfile.Profile();sync();profile.enable()
            for i in range(500):actual=compiled(*states[i%3][0])
            profile.disable();sync();check(actual,states[499%3])
            stats=pstats.Stats(profile)
            row['cpu_profile']=[{'file':key[0],'line':key[1],'name':key[2],
                                'primitive_calls':value[0],'calls':value[1],
                                'self_seconds':value[2],'cumulative_seconds':value[3]}
                               for key,value in sorted(stats.stats.items(),key=lambda kv:kv[1][3],reverse=True)]
        else:
            for block in range(12):
                held=[];sync();start=time.perf_counter_ns()
                for operands,_,_ in calls:held.append(compiled(*operands))
                sync();elapsed=time.perf_counter_ns()-start
                row['blocks_ns'].append(elapsed)
                for result,state in zip(held,calls):check(result,state)
                del held
            samples=[v/300 for v in row['blocks_ns']]
            row['ns_per_call']={'median':statistics.median(samples),
                                'quartiles':statistics.quantiles(samples,n=4),
                                'min':min(samples),'max':max(samples)}
        row['status']='passed'
    record['successful']=True
except BaseException:
    record['successful']=False;record['error']=traceback.format_exc()
    raise
finally:
    libraries=[]
    for path in sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if any(s in line for s in ('libcudart.so','libcuda.so','libnvrtc.so'))}):
        libraries.append({'path':path,'sha256':sha(path)})
    version=ctypes.c_int();assert runtime.cudaRuntimeGetVersion(ctypes.byref(version))==0
    record['cuda_runtime_version']=version.value;record['libraries']=libraries
    record['gpu_after']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,compute_cap,utilization.gpu,memory.used','--format=csv'],text=True)
    output.write_text(json.dumps(record,indent=2)+'\n')
