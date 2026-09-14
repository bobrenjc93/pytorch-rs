"""One-shot hardware-free frontend timing/count diagnostic; no CUDA attribution."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import torch_rs as native
from torch_rs import _compile_pointwise as frontend
from tests.test_compile_pointwise_helpers import HelperCache
from tests.test_compile_pointwise_jit import program

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
report = {'kind': 'hardware-free mocked-native warm frontend diagnostic',
          'started_ns': time.time_ns(), 'commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'dirty': subprocess.check_output(['git','status','--porcelain=v1'],text=True),
          'frontend': frontend.__file__, 'frontend_sha256': hashlib.sha256(Path(frontend.__file__).read_bytes()).hexdigest(),
          'executable': sys.executable, 'batch_calls': 1000, 'warmup_batches': 5, 'sample_batches': 17, 'cases': []}
for name, source in [('arithmetic', 'def f(x):\n return x*x+x'),
                     ('literal_loop', 'def f(x):\n for i in range(3):\n  x=x+0.125\n return x')]:
    fixture = HelperCache()
    fixture.setUp()
    try:
        compiled = native.compile(program(source))
        first = compiled(fixture.x)
        row = {'name': name, 'warmup_ns': [], 'sample_ns': [],
               'graph': first[:2]}
        for collection, count in [('warmup_ns', 5), ('sample_ns', 17)]:
            for _ in range(count):
                start = time.perf_counter_ns()
                for _ in range(1000): compiled(fixture.x)
                row[collection].append(time.perf_counter_ns()-start)
        counts = collections.Counter()
        def profile(frame, event, arg):
            if event == 'call' and frame.f_code.co_filename == frontend.__file__:
                counts[frame.f_code.co_name] += 1
        sys.setprofile(profile)
        try:
            for _ in range(100): compiled(fixture.x)
        finally:
            sys.setprofile(None)
        row['calls_per_100_warm_invocations'] = dict(counts)
        row['median_ns_per_call'] = statistics.median(row['sample_ns']) / 1000
        report['cases'].append(row)
    finally:
        fixture.doCleanups()
report['finished_ns'] = time.time_ns()
with args.output.open('x') as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report, indent=2))
