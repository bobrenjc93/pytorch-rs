import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import numpy as np
import torch
import torch_rs

torch.set_num_threads(1)
rng = np.random.default_rng(682019)
rows = []
for size in (16, 4096, 1048576):
    canonical = rng.integers(0, 2, size, dtype=np.uint8)
    sparse = canonical.copy()
    sparse[::257] = 254
    dense = rng.integers(0, 256, size, dtype=np.uint8)
    for pattern, storage in (("canonical", canonical), ("sparse", sparse), ("dense", dense)):
        for layout, selection in (("contiguous", slice(None)), ("reversed", slice(None, None, -1)), ("strided", slice(1, None, 2))):
            source = memoryview(storage.tobytes()).cast("?")[selection]
            expected = np.asarray(list(source), dtype=np.float32)
            iterations = 100 if size == 16 else 20 if size == 4096 else 1
            funcs = {name: (lambda module=module: module.tensor(source, dtype=module.float32, device="cpu")) for name, module in (("native", torch_rs), ("torch", torch))}
            values = {name: [] for name in funcs}
            correct = {}
            for name, func in funcs.items():
                actual = np.asarray(func())
                correct[name] = bool(np.array_equal(actual, expected))
            for order in (("native", "torch"), ("torch", "native")):
                for name in order:
                    for _ in range(3):
                        np.asarray(funcs[name]())
                for _ in range(5):
                    for name in order:
                        start = time.perf_counter_ns()
                        for _ in range(iterations):
                            output = funcs[name]()
                            shape = tuple(output.shape)
                        elapsed = (time.perf_counter_ns() - start) / iterations / 1000
                        materialized = np.asarray(output)
                        checksum = float(materialized.sum(dtype=np.float64))
                        assert shape == (len(source),)
                        assert bool(np.array_equal(materialized, expected)) == correct[name]
                        values[name].append(elapsed)
            rows.append(dict(size=size, elements=len(source), pattern=pattern, layout=layout, iterations=iterations, correct=correct, checksum=checksum, timings={name: dict(median_us=statistics.median(samples), min_us=min(samples), max_us=max(samples), samples_us=samples) for name, samples in values.items()}))
so = next(Path(torch_rs.__file__).parent.glob('*.so'))
print(json.dumps(dict(python=sys.version, executable=sys.executable, platform=platform.platform(), torch=torch.__version__, numpy=np.__version__, native=str(so), native_sha256=hashlib.sha256(so.read_bytes()).hexdigest(), threads=1, affinity=sorted(os.sched_getaffinity(0)), warmups_per_order=3, samples_per_order=5, seed=682019, rows=rows), indent=2))
