"""One bounded before/after public-call profile; never a parity benchmark."""
import argparse
import cProfile
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('admission_evidence', HERE / 'consumer.py')
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)
c = evidence.common
ROOT = c.ROOT
CASES = (
    ('unaryflat', 'def f(x, scale):\n return -x * scale\n'),
    ('unarynested', 'def f(data):\n return -data["x"][0] * data["scale"]\n'),
    ('binaryflat', 'def f(x, y, scale):\n return -x + y * scale\n'),
    ('binarynested', 'def f(data):\n return -data["x"][0] + data["other"][0] * data["other"][1]\n'),
)
SIZE, WARMUPS, SAMPLES = 1021, 5, 64


def arguments(native, name, phase):
    xs = [(i % 17 - 8) / 8 + (phase % 4) / 16 for i in range(SIZE)]
    x = native.tensor(xs).to('cuda:0')
    if name.startswith('unary'):
        args = (x, 0.5) if name.endswith('flat') else ({'x': [x], 'scale': 0.5},)
        return args, {'x': xs, 'scale': 0.5}, [-v * 0.5 for v in xs]
    ys = [(i % 13 - 6) / 4 - (phase % 3) / 8 for i in range(SIZE)]
    y = native.tensor(ys).to('cuda:0')
    args = (x, y, 0.5) if name.endswith('flat') else ({'x': [x], 'other': (y, 0.5)},)
    return args, {'x': xs, 'y': ys, 'scale': 0.5}, [-v + w * 0.5 for v, w in zip(xs, ys)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', required=True, choices=('before', 'after'))
    parser.add_argument('--wheel', required=True, type=Path)
    parser.add_argument('--build-record', required=True, type=Path)
    args = parser.parse_args()
    folder = c.inside(ROOT / 'target/warm-repair' / ('profile-' + args.side))
    folder.mkdir(parents=True, exist_ok=False)  # A side cannot overwrite/reroll an attempt.
    record = {'side': args.side, 'passed': False, 'ordinaryLatencyOrScore': False, 'implementation': 'native',
              'startedAt': datetime.now(timezone.utc).isoformat(), 'command': sys.orig_argv,
              'cases': CASES, 'size': SIZE, 'warmups': WARMUPS, 'samples': SAMPLES,
              'scriptSha256': c.sha(Path(__file__).read_bytes()), 'phases': [],
              'compiledInvocations': 0, 'profiledInvocations': 0}
    try:
        c.environment(folder)
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
        record['source'] = evidence.source(ROOT, folder)
        record['installed'] = evidence.installed(args, record['source'])
        record.update(interpreter=str(c.inside(sys.executable)),
                      interpreterSha256=c.sha(Path(sys.executable).read_bytes()), python=sys.version,
                      gpuBefore=c.gpu_identity(), scopeSha256=c.sha((HERE / 'warm-repair-scope.md').read_bytes()))
        compiler = c.CompilerPin(os.environ['TORCH_RS_NVRTC'])
        sync = c.Synchronizer(os.environ['TORCH_RS_CUDART'])
        record.update(compiler=compiler.metadata, synchronization=sync.metadata,
                      environment={k: os.environ.get(k) for k in (
                          'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                          'TORCH_RS_NVRTC', 'TORCH_RS_CUDART')})
        c.write(folder / 'entry.json.gz', record)
        import torch_rs as native
        native.set_num_threads(1)
        for name, source in CASES:
            namespace = {'__builtins__': __builtins__}
            exec(compile(source, '<warm-profile-'+name+'>', 'exec'), namespace)
            fn = namespace['f']
            compiled = native.compile(fn)
            retained = []
            row = {'name': name, 'inputs': [], 'outputs': [], 'compiledInvocations': 0,
                   'profiledInvocations': 0}
            record['phases'].append(row)
            profile = cProfile.Profile()
            try:
                for phase in range(1 + WARMUPS + SAMPLES):
                    inputs, capture, expected = arguments(native, name, phase)
                    row['inputs'].append(capture)
                    sync()
                    profiled = phase >= 1 + WARMUPS
                    record['compiledInvocations'] += 1
                    row['compiledInvocations'] += 1
                    assert record['compiledInvocations'] <= 512
                    if profiled:
                        record['profiledInvocations'] += 1
                        row['profiledInvocations'] += 1
                        profile.enable()
                    try:
                        result = compiled(*inputs)
                    finally:
                        if profiled:
                            profile.disable()
                    sync()
                    actual = result.cpu().tolist()
                    row['outputs'].append(actual)
                    assert actual == expected
                    retained.append(result)
            finally:
                profile.dump_stats(str(folder / (name + '.prof')))
                row['profile'] = sorted([
                    {'function': str(item.code) if isinstance(item.code, str) else
                     f'{item.code.co_filename}:{item.code.co_firstlineno}:{item.code.co_name}',
                     'calls': item.callcount, 'recursiveCalls': item.reccallcount,
                     'inlineSeconds': item.inlinetime, 'totalSeconds': item.totaltime}
                    for item in profile.getstats()], key=lambda r: r['function'])
            assert not any(item.code is fn.__code__ for item in profile.getstats())
            row['outputsRetained'] = len(retained)
            row['passed'] = True
        assert record['compiledInvocations'] == 280 and record['profiledInvocations'] == 256
        record.update(libraries=c.libraries(), gpuAfter=c.gpu_identity())
        c.validate_libraries(record)
        assert 'torch' not in sys.modules
        record['passed'] = True
    except BaseException:
        record['failure'] = traceback.format_exc()
        raise
    finally:
        record['finishedAt'] = datetime.now(timezone.utc).isoformat()
        c.write(folder / 'report.json.gz', record)
    print(args.side, 'passed:', record['compiledInvocations'], 'calls;', record['profiledInvocations'], 'profiled')


if __name__ == '__main__':
    main()
