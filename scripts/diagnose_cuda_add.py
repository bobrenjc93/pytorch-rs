#!/usr/bin/env python3
"""Public CUDA-add diagnostics, not an evaluator or a historical-report updater.

Use a release wheel built from the source snapshot recorded by --build-record.
Each cache condition runs in a fresh child process. All timings include completion;
output readback is outside timing on both sides. No compile benchmark is used.
"""
from __future__ import annotations

import argparse
import ctypes
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def worker(args):
    import numpy as np
    import torch
    import torch_rs as native
    from torch_rs._cuda_public_storage import _candidate_libraries

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.__version__.split("+")[0] == "2.13.0"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {affinity[0]})
    rng = np.random.default_rng(args.seed)
    lib = ctypes.CDLL(os.environ.get("TORCH_RS_CUDART") or _candidate_libraries()[0])
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    runtime_version = ctypes.c_int()
    assert lib.cudaRuntimeGetVersion(ctypes.byref(runtime_version)) == 0

    def read(tensor):
        values = np.empty(tuple(tensor.shape), dtype=np.float32)
        if values.size:
            assert lib.cudaMemcpy(values.ctypes.data, tensor.data_ptr(), values.nbytes, 2) == 0
        return values

    modules = {"native": native, "torch": torch}
    saturation_sizes = [1000 + i * 7919 for i in range(48)] + [2**20 + i * 1009 for i in range(20)]
    if args.worker == "saturated":
        for module in modules.values():
            # The same held-live allocation burst and drops on both sides fills
            # entry and byte budgets; no unsupported native API is involved.
            junk = [module.zeros((n,), device="cuda:0") for n in saturation_sizes]
            del junk
        torch.cuda.synchronize()

    shapes = [(), (0,), (3079,), (65557,), (1_048_603,), (4_194_359,),
              (17_000_003,), (33_554_467,), (67_108_867,)]
    shapes += [tuple(int(x) for x in rng.integers(21, 91, size=3)) for _ in range(2)]
    rows = []
    for index, shape in enumerate(shapes):
        # Nonzero, exactly representable random inputs; size-independent values
        # keep repeated chain rounding exactly comparable at every iteration.
        n = math.prod(shape)
        a = rng.integers(-4096, 4096, size=n).astype(np.float32) / 128
        b = rng.integers(-4096, 4096, size=n).astype(np.float32) / 128
        inputs = {}
        for name, module in modules.items():
            def upload(values):
                if index == 3:
                    padded = np.concatenate((np.full(7, 99, dtype=np.float32), values))
                    return module.tensor(padded.tolist(), dtype=module.float32).to("cuda:0")[7:].reshape(shape)
                return module.tensor(values.tolist(), dtype=module.float32).reshape(shape).to("cuda:0")
            inputs[name] = (upload(a), upload(b))
        # All primary shapes use +; additional cells exercise both other public
        # forms and independently offset contiguous aliases.
        forms = ["operator"] if index >= 4 else ["operator", "method", "function"]
        for form in forms:
            for batch, chained in [(1, False), (32, False), (64, False), (32, True)]:
                raw = {name: [] for name in modules}
                expected = a.copy().reshape(shape)
                if chained:
                    for _ in range(batch):
                        expected = expected + b.reshape(shape)
                else:
                    expected = expected + b.reshape(shape)
                checks = {}
                for order in [("native", "torch"), ("torch", "native")]:
                    for name in order:
                        module = modules[name]
                        x, y = inputs[name]
                        def call(left, right):
                            if form == "operator":
                                return left + right
                            if form == "method":
                                return left.add(right)
                            return module.add(left, right)
                        def block():
                            result = x
                            for _ in range(batch):
                                result = call(result if chained else x, y)
                            return result
                        for _ in range(args.warmups):
                            out = block()
                        torch.cuda.synchronize()
                        for _ in range(args.samples):
                            # Drop the previous result before timing, symmetrically.
                            del out
                            torch.cuda.synchronize()
                            start = time.perf_counter_ns()
                            out = block()
                            torch.cuda.synchronize()
                            raw[name].append((time.perf_counter_ns() - start) / batch)
                            actual = read(out)
                            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                        checks[name] = {"bitwise_equal": True, "shape": list(out.shape),
                                        "stride": list(out.stride()), "dtype": str(out.dtype),
                                        "device": str(out.device), "sha256": hashlib.sha256(actual.tobytes()).hexdigest()}
                        # Do not let the last implementation's loop bindings
                        # retain inputs beyond the shared inputs dictionary.
                        # This matters when the next shape pressures its pool.
                        del out, x, y
                medians = {name: statistics.median(values) for name, values in raw.items()}
                ratio = medians["torch"] / medians["native"]
                rows.append({"shape": list(shape), "layout": "offset_7" if index == 3 else "contiguous", "form": form, "calls_per_sample": batch,
                             "chained": chained, "samples_ns_per_call": raw, "median_ns": medians,
                             "stdev_ns": {k: statistics.pstdev(v) for k, v in raw.items()},
                             "uncapped_parity": ratio, "capped_parity": min(1, ratio), "checks": checks})
        del inputs
    aggregates = {}
    for boundary in [1, 32, 64]:
        ratios = [row["capped_parity"] for row in rows if row["calls_per_sample"] == boundary and not row["chained"]]
        aggregates[str(boundary)] = 100 * math.exp(statistics.mean(math.log(r) for r in ratios))
    extension = Path(importlib.import_module("torch_rs.torch_rs").__file__).resolve()
    return {"cache_state": args.worker, "saturation_sizes": saturation_sizes if args.worker == "saturated" else [],
            "rows": rows, "diagnostic_capped_geometric_parity_percent": aggregates,
            "environment": {"python": sys.version, "executable": sys.executable,
                            "prefix": sys.prefix, "native_extension": str(extension),
                            "native_sha256": sha(extension), "package": native.__file__,
                            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                            "cuda_runtime": runtime_version.value,
                            "cuda_libraries": sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                                      if 'libcuda' in line and '/' in line}),
                            "gpu": command("nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,compute_cap", "--format=csv"),
                            "nvcc": command("nvcc", "--version"), "rustc": command("rustc", "--version"),
                            "os": platform.platform(), "cpu": command("lscpu"),
                            "affinity": sorted(os.sched_getaffinity(0)), "threads": 1,
                            "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                            "CUDA_CACHE_PATH": os.environ.get("CUDA_CACHE_PATH")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-record", type=Path)
    parser.add_argument("--seed", type=int, default=937514)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--worker", choices=["clean", "saturated"])
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
        return
    if args.output is None or args.build_record is None:
        parser.error("--output and --build-record are required")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.relative_to(root)
    build = json.loads(args.build_record.read_text())
    for path, digest in build["source_files"].items():
        if sha(Path(build["source_root"]) / path) != digest:
            raise RuntimeError(f"source snapshot changed: {path}")
    cases = []
    for state in ["clean", "saturated"]:
        result = subprocess.check_output([sys.executable, str(Path(__file__).resolve()), "--worker", state,
                                          "--seed", str(args.seed), "--samples", str(args.samples),
                                          "--warmups", str(args.warmups)], text=True)
        case = json.loads(result)
        if case["environment"]["native_sha256"] != build["installed_native_sha256"]:
            raise RuntimeError("installed extension differs from the recorded build")
        cases.append(case)
    report = {"schema": 1, "purpose": "CUDA-add feature diagnostic, not an evaluator score",
              "seed": args.seed, "warmups_per_order": args.warmups, "samples_per_order": args.samples,
              "orders": [["native", "torch"], ["torch", "native"]],
              "timing": "host wall time around 1/32/64 public calls plus device synchronization; inputs and bitwise readback outside timing on both sides",
              "cache": "two fresh processes, ordered shape matrix; clean means no saturation prelude, not cold per-call allocation",
              "build": build, "runner": str(Path(__file__).resolve()), "runner_sha256": sha(__file__), "cases": cases}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for case in cases:
        print(case["cache_state"], case["diagnostic_capped_geometric_parity_percent"])


if __name__ == "__main__":
    main()
