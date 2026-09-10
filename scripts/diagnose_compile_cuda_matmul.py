#!/usr/bin/env python3
"""Separate public compiled CUDA matmul diagnostic; never a Burner scoring corpus.

Uses native backend=eager capture versus PyTorch 2.13 backend=inductor. Each
cell has equal weight, including failures (zero); no result-dependent selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

# Read-only reuse of provenance and timing summaries; scoring files stay fixed.
from benchmark_cuda_matmul import ROOT, local, sha256, source_provenance, stamp, summary
from evaluate_cuda_math import runtime_provenance


def cells(seed):
    import random
    rng = random.Random(seed)
    shapes = [(1, 65539, 1), (64, 64, 64), (512, 512, 512), (127, 1025, 65)]
    side = rng.randrange(97, 769)
    shapes += [(side, side, side), tuple(rng.randrange(67, 803) for _ in range(3))]
    return [{'shape_mkn': shape, 'offset': offset, 'program': program,
             'status': 'pending', 'capped_parity': 0., 'orders': []}
            for shape in shapes for offset, program in ((0, 'matmul'), (3, 'composed'))]


def aggregate(cases):
    values = [c['capped_parity'] if c['status'] == 'passed' else 0. for c in cases]
    if not values or any(v <= 0 for v in values):
        return 0.
    return math.exp(statistics.mean(math.log(min(1., v)) for v in values))


def program(module, bias, composed):
    expression = '-((x * 0.5) @ (y * -2)) + bias' if composed else 'x @ y'
    namespace = {'bias': bias}
    exec(f'def workload(x, y):\n    return {expression}\n', namespace)
    return namespace['workload']


def sample_order(calls, inputs, order, synchronize, check, entry):
    # Attach observations immediately so an exception retains complete blocks
    # and even the individual calls in an unfinished or failed block. The
    # caller's exception handler publishes this same entry with the failed cell.
    samples = {key: [] for key in order}
    entry['samples'] = {key: {'samples_us': samples[key], 'calls_us': [],
                              'checked_calls': 0} for key in order}
    for _ in range(31):
        for key in order:
            elapsed = 0
            record = entry['samples'][key]
            for _ in range(5):
                synchronize(); start = time.perf_counter_ns()
                result = calls[key](*inputs[key]); synchronize()
                duration = time.perf_counter_ns() - start
                elapsed += duration
                record['calls_us'].append(duration / 1000)
                check(result)  # Materialization/checking stays outside timing.
                record['checked_calls'] += 1
            samples[key].append(elapsed / 5000)
            record.update(summary(samples[key]))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=798431)
    parser.add_argument('--allow-dirty', action='store_true')
    args = parser.parse_args()
    output = local(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the attempt before setup; even failed setup is retained.
    with output.open('x') as file:
        file.write('{}\n')
    report = {'diagnostic': 'compiled_cuda_matmul_v1', 'started_at': stamp(),
              'command': [sys.executable, *sys.argv], 'cwd': str(Path.cwd()),
              'seed': args.seed, 'cases': cells(args.seed), 'status': 'setup',
              'harness_sha256': sha256(__file__),
              'policy': {'native_backend': 'eager', 'reference_backend': 'inductor',
                         'fullgraph': True, 'dynamic': False, 'mode': 'default',
                         'warmups_per_order': 10, 'samples_per_order': 31, 'calls_per_sample': 5,
                         'orders': [['native','pytorch'], ['pytorch','native']],
                         'timing': 'wall-clock; CUDA synchronize before/after every call; output retained to completion',
                         'proof_instrumentation': 'untimed changed-data calls block lowering and Python-body execution; all timing calls unprofiled',
                         'first_call': 'new wrapper and Dynamo reset per order; compiler disk caches retained and disclosed',
                         'aggregation': 'fixed equal-weight geometric mean of capped reference/native medians; failed/missing cells zero',
                         'rtol': 1e-5, 'atol': 1e-4, 'tf32': False}}
    def save():
        output.write_text(json.dumps(report, indent=2) + '\n')
    try:
        assert Path.cwd().resolve() == ROOT
        before = source_provenance()
        report['source'] = before
        status = subprocess.check_output(['git','status','--porcelain'], text=True)
        report['git_status'] = status
        report['measurement_kind'] = 'development-uncommitted' if status else 'clean-commit'
        if status and not args.allow_dirty:
            raise RuntimeError('Burner must commit implementation before clean evidence; use --allow-dirty only for development')
        build_path = local(args.build_record)
        build = json.loads(build_path.read_text())
        for key in ('commit','source_sha256','production_diff_sha256'):
            assert build[key] == before[key], f'stale build record: {key}'
        report['build_record'] = {'path': str(build_path), 'sha256': sha256(build_path), 'record': build}
        for key in ('TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR',
                    'TORCH_RS_CUDART','TORCH_RS_CUBLAS','CARGO_HOME','CARGO_TARGET_DIR'):
            local(os.environ[key])
        report['environment'] = {key: value for key,value in os.environ.items()
                                 if key.startswith(('TORCH','TRITON','CUDA','CARGO','OMP_','MKL_','OPENBLAS_','PYO3_')) or key in ('TMPDIR','XDG_CACHE_HOME')}
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
        local(sys.executable); local(sys.base_prefix)
        import numpy as np
        import torch
        import torch_rs as native
        extension = importlib.import_module('torch_rs.torch_rs')
        report['imports'] = {mod.__name__: str(local(mod.__file__)) for mod in (np,torch,native,extension)}
        assert local(native.__file__).is_relative_to(ROOT / '.venv'), 'install a current wheel'
        assert sha256(extension.__file__) == build['extension_sha256']
        # Check every installed torch_rs Python file against this checkout,
        # not just the extension or editable package entry point.
        package = Path(native.__file__).parent
        manifest = {}
        for source in sorted((ROOT/'python/torch_rs').rglob('*.py')):
            installed = package / source.relative_to(ROOT/'python/torch_rs')
            assert sha256(installed) == sha256(source), f'stale installed source: {source}'
            manifest[str(source.relative_to(ROOT))] = sha256(source)
        report['installed_source_manifest'] = manifest
        report['executable'] = str(local(sys.executable)); report['python'] = sys.version
        assert torch.__version__.split('+')[0] == '2.13.0' and torch.cuda.is_available()
        report['pytorch'] = torch.__version__; report['pytorch_cuda'] = torch.version.cuda
        report['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,compute_cap,memory.used','--format=csv'],text=True)
        report['nvcc'] = subprocess.check_output(['nvcc','--version'],text=True)
        report['native_cuda_compiler'] = 'nvcc unused; cuBLAS SGEMM and driver-JIT embedded PTX'
        report['rustc'] = subprocess.check_output(['rustc','--version'],text=True)
        import triton
        from triton.backends.nvidia.compiler import get_ptxas
        major, minor = torch.cuda.get_device_capability(0)
        ptxas = local(get_ptxas(major * 10 + minor).path)
        report['reference_triton_compiler'] = {
            'triton_version': triton.__version__, 'triton_path': str(local(triton.__file__)),
            'ptxas_path': str(ptxas), 'ptxas_sha256': sha256(ptxas),
            'ptxas_version': subprocess.check_output([str(ptxas), '--version'], text=True),
            'pytorch_build': torch.__config__.show(),
        }
        report['cache_initial_files'] = {key: len(list(local(os.environ[key]).rglob('*'))) for key in ('TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR')}
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_num_threads(1)
        rng = np.random.default_rng(args.seed)
        from torch_rs import _compile_bytecode
        for cell in report['cases']:
            try:
                m,k,n = cell['shape_mkn']; offset = cell['offset']
                vals = [rng.normal(size=offset+s).astype(np.float32) for s in (m*k,k*n)]
                bases = {key: [mod.tensor(v.tolist(), dtype=mod.float32).to('cuda:0') for v in vals]
                         for key,mod in (('native',native),('pytorch',torch))}
                inputs = {key: (v[0][offset:].reshape(m,k), v[1][offset:].reshape(k,n)) for key,v in bases.items()}
                bias_values = rng.normal(size=n).astype(np.float32)
                biases = {key: mod.tensor(bias_values.tolist()).to('cuda:0') for key,mod in (('native',native),('pytorch',torch))}
                fns = {key: program(mod,biases[key],cell['program']=='composed') for key,mod in (('native',native),('pytorch',torch))}
                def array(value):
                    return np.asarray(value.cpu().tolist(),dtype=np.float32).reshape(tuple(value.shape))
                expected = array(fns['pytorch'](*inputs['pytorch']))
                output_hashes = []
                cell['output_hashes'] = output_hashes
                cell['input_hashes'] = [hashlib.sha256(v.tobytes()).hexdigest() for v in vals]
                def check(value):
                    assert tuple(value.shape) == (m,n) and value.stride() == (n,1)
                    assert str(value.device) == 'cuda:0' and str(value.dtype) == 'torch.float32'
                    assert value.storage_offset() == 0 and not value.requires_grad
                    actual = array(value)
                    output_hashes.append(hashlib.sha256(actual.tobytes()).hexdigest())
                    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-4,equal_nan=True)
                all_samples = {'native': [], 'pytorch': []}
                for order in report['policy']['orders']:
                    torch._dynamo.reset()
                    entry = {'order': order, 'first_call_ms': {}, 'wrapper_ms': {}, 'samples': {}}
                    cell['orders'].append(entry); save()
                    calls = {}
                    for key in order:
                        mod = native if key == 'native' else torch
                        start = time.perf_counter_ns()
                        compiled = mod.compile(fns[key], backend='eager' if key=='native' else 'inductor', fullgraph=True, dynamic=False)
                        entry['wrapper_ms'][key] = (time.perf_counter_ns()-start)/1e6
                        calls[key] = compiled
                        # Body-replay proof runs separately from timing.
                        def reject(frame,event,arg):
                            if event=='call' and frame.f_code is fns['native'].__code__:
                                raise AssertionError('candidate body replay')
                        torch.cuda.synchronize(); start = time.perf_counter_ns()
                        result = compiled(*inputs[key]); torch.cuda.synchronize()
                        entry['first_call_ms'][key] = (time.perf_counter_ns()-start)/1e6
                        check(result)
                        retained = result
                        # Changed values with identical guards must produce new
                        # outputs without re-lowering; restore timing inputs after.
                        changed_bases = [v * 0.25 for v in bases[key]]
                        changed = (changed_bases[0][offset:].reshape(m,k), changed_bases[1][offset:].reshape(k,n))
                        changed_expected = array(fns['pytorch'](*(v * 0.25 for v in inputs['pytorch'])))
                        from unittest.mock import patch
                        context = patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('stale graph/cache miss'))
                        old = sys.getprofile()
                        try:
                            if key=='native': sys.setprofile(reject)
                            with context:
                                result = compiled(*changed)
                        finally:
                            sys.setprofile(old)
                        np.testing.assert_allclose(array(result),changed_expected,rtol=1e-5,atol=1e-4)
                        assert result.data_ptr() != retained.data_ptr()
                        for value in (*inputs[key], *changed): assert result.data_ptr() != value.data_ptr()
                        for value, original in zip(changed_bases, vals):
                            np.testing.assert_array_equal(array(value).view(np.uint32), (original * np.float32(0.25)).view(np.uint32))
                    for key in order:
                        for _ in range(10):
                            torch.cuda.synchronize(); result = calls[key](*inputs[key]); torch.cuda.synchronize()
                    samples = sample_order(calls, inputs, order, torch.cuda.synchronize, check, entry)
                    for key in order:
                        all_samples[key].extend(samples[key])
                        for value,original in zip(bases[key],vals):
                            np.testing.assert_array_equal(array(value).view(np.uint32),original.view(np.uint32))
                        np.testing.assert_array_equal(array(biases[key]),bias_values)
                    save()
                cell['timings'] = {key: summary(samples) for key,samples in all_samples.items()}
                cell['capped_parity'] = min(1.,statistics.median(all_samples['pytorch'])/statistics.median(all_samples['native']))
                cell['output_hashes'] = sorted(set(output_hashes))
                cell['checks'] = 'every timed output, changed inputs, fresh storage, all source/bias bits, shape/stride/device/dtype'
                cell['status'] = 'passed'
            except Exception:
                cell['status'] = 'failed'; cell['error'] = traceback.format_exc(); cell['capped_parity'] = 0.
            save()
            print(json.dumps({key:cell[key] for key in ('shape_mkn','offset','status','capped_parity')}),flush=True)
        libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                            if any(name in line for name in ('/libcudart.so','/libcublas.so','/libcublasLt.so'))})
        report['loaded_cuda_libraries'] = [{'path':str(local(p)),'sha256':sha256(p)} for p in libraries]
        report['runtime'] = runtime_provenance()
        report['source_after'] = source_provenance()
        assert report['source_after'] == before, 'source changed during capture'
        report['status'] = 'passed' if all(c['status']=='passed' for c in report['cases']) else 'failed'
    except BaseException:
        report['status'] = 'failed'; report['error'] = traceback.format_exc()
    finally:
        report['capped_geometric_parity'] = aggregate(report['cases'])
        report['ended_at'] = stamp(); save()
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
