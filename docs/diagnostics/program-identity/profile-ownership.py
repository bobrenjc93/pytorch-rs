"""Bounded native ownership call attribution; profiler times are not latency.

Run against a worktree-local installed release wheel, before and after a repair.
This never participates in the frozen public timing or scoring protocol.
"""
import argparse
import cProfile
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('consumer', Path(__file__).with_name('consumer.py'))
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def negation(x):
    return -x


def profile_call(label, compiled, lengths, inputs, sync, record):
    cache = compiled._torch_rs_pointwise_cache
    row = {'label': label, 'lengths': lengths, 'preparedBefore': len(cache.prepared)}
    record['phases'].append(row)
    outputs = []
    profile = cProfile.Profile()
    sync()
    profile.enable()
    try:
        for length in lengths:
            outputs.append(compiled(inputs[length]))
    finally:
        profile.disable()
    sync()
    rows = []
    for item in profile.getstats():
        code = item.code
        name = code if isinstance(code, str) else f'{code.co_filename}:{code.co_firstlineno}:{code.co_name}'
        rows.append({'function': name, 'calls': item.callcount, 'recursiveCalls': item.reccallcount,
                     'profilerInlineSeconds': item.inlinetime, 'profilerTotalSeconds': item.totaltime})
    row.update(profile=sorted(rows, key=lambda x: x['function']),
               ownershipQueries=sum(x['calls'] for x in rows if 'belongs_to' in x['function']),
               preparedAfter=len(cache.prepared), outputsRetained=len(outputs))
    for length, output in zip(lengths, outputs):
        assert output.cpu().tolist() == [-0.25] * length
    row['outputsChecked'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = c.inside(args.output)
    output.parent.mkdir(parents=True, exist_ok=False)
    record = {'kind': 'native-ownership-profiler-attribution', 'passed': False,
              'ordinaryLatencyOrScore': False, 'command': sys.orig_argv,
              'startedAt': datetime.now(timezone.utc).isoformat(),
              'head': c.command('git', 'rev-parse', 'HEAD'),
              'status': c.command('git', 'status', '--short'),
              'scriptSha256': c.sha(Path(__file__).read_bytes()), 'phases': []}
    try:
        c.environment(output.parent)
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
        wheel = c.inside(args.wheel)
        files = c.wheel_files(wheel)
        import torch_rs as native
        package = c.inside(Path(native.__file__).parent)
        for name, data in files.items():
            assert (package.parent / name).read_bytes() == data
            if name.endswith('.py'):
                assert (ROOT / 'python' / name).read_bytes() == data
        record.update(wheel=str(wheel), wheelSha256=c.sha(wheel.read_bytes()),
                      package=str(package), installedHashes={n: c.sha(b) for n, b in files.items()},
                      interpreter=str(c.inside(sys.executable)), python=sys.version,
                      sourceHashes={name: c.sha((ROOT / name).read_bytes()) for name in (
                          'python/torch_rs/_compile_pointwise.py', 'src/python_pointwise.rs',
                          'src/tensor_pointwise.rs')}, gpuBefore=c.gpu_identity())
        runtime = c.library_file(os.environ['TORCH_RS_CUDART'])
        compiler = c.CompilerPin(os.environ['TORCH_RS_NVRTC'])
        sync = c.Synchronizer(runtime)
        record.update(implementation='native', synchronization=sync.metadata, compiler=compiler.metadata,
                      environment={name: os.environ[name] for name in ('TORCH_RS_CUDART', 'TORCH_RS_NVRTC')})
        native.set_num_threads(1)
        inputs = {n: native.tensor([0.25] * n).to('cuda:0') for n in (3, 5, 7, 9, 11, 13, 15)}
        compiled = native.compile(negation)
        profile_call('cold', compiled, [3], inputs, sync, record)
        profile_call('one-preparation-newest', compiled, [3] * 16, inputs, sync, record)
        profile_call('warm-miss', compiled, [5], inputs, sync, record)
        compiled(inputs[3])  # Admit shape 3 under the promoted numerical hint.
        profile_call('three-preparations-newest-and-older', compiled, [3, 5] * 16, inputs, sync, record)
        for n in (7, 9, 11, 13, 15):
            compiled(inputs[n])
        profile_call('eight-preparations-newest-and-older', compiled, [15, 5] * 16, inputs, sync, record)
        record.update(libraries=c.libraries(), gpuAfter=c.gpu_identity())
        c.validate_libraries(record)
        assert 'torch' not in sys.modules
        record['passed'] = True
    except BaseException:
        record['failure'] = traceback.format_exc()
        raise
    finally:
        record['finishedAt'] = datetime.now(timezone.utc).isoformat()
        c.write(output, record)


if __name__ == '__main__':
    main()
