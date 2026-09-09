#!/usr/bin/env python3
"""Matched-thread rank-2 mean diagnostics; no evaluator or compile benchmark.

Use build_cuda_add_diagnostic.py to install and record an isolated release
build. Dirty exports are scratch diagnostics only, never retained evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from diagnose_cuda_add import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-record', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 4, 8])
    parser.add_argument('--implementation', choices=['native', 'torch'],
                        help='supplemental single-engine process; retain default co-resident results too')
    parser.add_argument('--seed', type=int, default=914207)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--samples', type=int, default=9)
    args = parser.parse_args()
    if min(args.threads + [args.warmups, args.samples]) < 1:
        parser.error('thread counts, warmups and samples must be positive')
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.relative_to(root)
    build = json.loads(args.build_record.read_text())
    if not build['source_matches_commit'] and root / 'target' not in output.parents:
        parser.error('uncommitted-source diagnostics must stay under target/')
    for path, digest in build['source_files'].items():
        if sha(Path(build['source_root']) / path) != digest:
            raise RuntimeError(f'source snapshot changed: {path}')

    import numpy as np
    import torch
    import torch_rs as native
    extension = Path(importlib.import_module('torch_rs.torch_rs').__file__).resolve()
    if sha(extension) != build['installed_native_sha256']:
        raise RuntimeError('installed native extension differs from recorded build')
    if torch.__version__.split('+')[0] != '2.13.0':
        raise RuntimeError('requires PyTorch 2.13 reference')
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < max(args.threads):
        raise RuntimeError('insufficient CPUs in affinity mask')
    os.sched_setaffinity(0, affinity[:max(args.threads)])
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    shapes = [(2048, 4096), (8192, 513), (257, 16381), (33, 65537)]
    shapes += [tuple(int(n) for n in rng.integers(701, 2100, size=2)) for _ in range(2)]
    rows = []
    for shape in shapes:
        # Exact binary inputs avoid hiding errors behind noisy reference sums.
        values = rng.integers(-128, 128, size=shape).astype(np.float32) / 64
        inputs = {'native': native.tensor(values.tolist()), 'torch': torch.tensor(values)}
        for threads in args.threads:
            torch.set_num_threads(1 if args.implementation == "native" else threads)
            if hasattr(native, 'set_num_threads'):
                native.set_num_threads(1 if args.implementation == "torch" else threads)
            if args.implementation != "torch" and native.get_num_threads() != threads:
                raise RuntimeError('native thread budget does not match requested/reference budget')
            for transposed in (False, True):
                for axis in (0, 1):
                    logical = values.T if transposed else values
                    expected = logical.astype(np.float64).mean(axis=axis).astype(np.float32)
                    for form in ('method', 'function'):
                        for batch in (1, 32):
                            raw = {'native': [], 'torch': []}
                            checks = {}
                            for order in [('native', 'torch'), ('torch', 'native')]:
                                for name in order:
                                    if args.implementation and name != args.implementation:
                                        continue
                                    module = native if name == 'native' else torch
                                    x = inputs[name].t() if transposed else inputs[name]
                                    def block():
                                        for _ in range(batch):
                                            result = x.mean(axis) if form == 'method' else module.mean(x, axis)
                                        return result
                                    for _ in range(args.warmups):
                                        out = block()
                                    for _ in range(args.samples):
                                        del out
                                        start = time.perf_counter_ns()
                                        out = block()
                                        raw[name].append((time.perf_counter_ns() - start) / batch)
                                        actual = np.array(out.tolist(), dtype=np.float32)
                                        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
                                    checks[name] = {'matches_reference': True,
                                                    'max_abs_error': float(np.max(np.abs(actual - expected))),
                                                    'sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                                                    'shape': list(out.shape), 'stride': list(out.stride()),
                                                    'dtype': str(out.dtype), 'device': str(out.device)}
                                    del out, x
                            medians = {name: statistics.median(samples) for name, samples in raw.items() if samples}
                            ratio = medians['torch'] / medians['native'] if len(medians) == 2 else None
                            rows.append({'shape': shape, 'transposed': transposed, 'axis': axis,
                                         'form': form, 'calls_per_sample': batch,
                                         'native_threads': native.get_num_threads(), 'torch_threads': torch.get_num_threads(),
                                         'samples_ns_per_call': raw, 'median_ns': medians,
                                         'stdev_ns': {name: statistics.pstdev(samples) for name, samples in raw.items() if samples},
                                         'uncapped_parity': ratio, 'capped_parity': min(1, ratio) if ratio is not None else None, 'checks': checks})
        del inputs
    aggregates = {}
    for threads in ([] if args.implementation else args.threads):
        for batch in (1, 32):
            ratios = [r['capped_parity'] for r in rows if r['native_threads'] == threads and r['calls_per_sample'] == batch]
            aggregates[f'{threads} threads/{batch} calls'] = 100 * math.exp(statistics.mean(math.log(r) for r in ratios))
    report = {'schema': 1, 'purpose': 'rank-2 CPU mean diagnostic, not an evaluator score',
              'seed': args.seed, 'warmups_per_order': args.warmups, 'samples_per_order': args.samples,
              'orders': ([[args.implementation], [args.implementation]] if args.implementation else
                         [['native', 'torch'], ['torch', 'native']]),
              'timing': 'host wall time around 1/32 synchronous public calls; inputs and output readback outside timing on both sides',
              'cache': 'one fresh process; warmed ordered shape/thread/layout matrix, no cache flush',
              'build': build, 'runner': str(Path(__file__).resolve()), 'runner_sha256': sha(__file__),
              'environment': {'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
                              'native_extension': str(extension), 'native_sha256': sha(extension), 'package': native.__file__,
                              'torch': torch.__version__, 'platform': platform.platform(),
                              'cpu': subprocess.check_output(['lscpu'], text=True),
                              'affinity': sorted(os.sched_getaffinity(0)), 'interop_threads': 1,
                              'implementation': args.implementation or 'co-resident comparison',
                              'thread_environment': {key: os.environ.get(key) for key in
                                                     ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_WAIT_POLICY', 'KMP_BLOCKTIME']},
                              'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip()},
              'diagnostic_capped_geometric_parity_percent': aggregates, 'rows': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(aggregates, indent=2))


if __name__ == '__main__':
    main()
