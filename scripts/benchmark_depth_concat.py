#!/usr/bin/env python3
"""Release dstack/cat diagnostic; targeted timings, never a coverage score.

Run in the checkout's real .venv after installing a source-matched release
wheel. --build-record must contain evaluate_cuda_math.source_provenance(),
extension_sha256 and the actual build command. Dirty runs remain precommit
diagnostics; Burner must repeat after committing the integrated source.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import benchmark_top_level_stack as common
from evaluate_cuda_math import sha256, source_provenance

ROOT = Path(__file__).resolve().parents[1]


def workloads():
    # Shape-based coverage includes the reported regression and unrelated sizes,
    # views, scalars and empty outputs. Public rank-three inputs stay rejected;
    # generic rank-three native concatenation is covered by Rust tests.
    yield "scalar", "dstack", (), "dense", 2
    yield "vector", "dstack", (263,), "dense", 3
    for shape in ((3, 7), (257, 263), (129, 521)):
        for layout in ("dense", "transpose", "offset", "strided"):
            yield f"matrix_{shape[0]}x{shape[1]}_{layout}", "dstack", shape, layout, 2
    for layout in ("dense", "transpose", "offset", "strided"):
        yield f"three_matrices_{layout}", "dstack", (37, 41), layout, 3
    for axis in range(2):
        yield f"cat_axis_{axis}", f"cat{axis}", (37, 41), "transpose", 3
    yield "empty", "dstack", (257, 0), "dense", 2


def operands(module, np, shape, layout, count, seed):
    rng = np.random.default_rng(seed)
    result = []
    for index in range(count):
        current = list(shape)
        physical = list(current)
        if layout == "transpose":
            physical[0], physical[1] = physical[1], physical[0]
        elif layout == "offset":
            physical.insert(0, 2)
        elif layout == "strided":
            physical.append(2)
        values = rng.standard_normal(physical).astype(np.float32)
        tensor = module.tensor(values.tolist(), dtype=module.float32).reshape(physical)
        if layout == "transpose":
            tensor = tensor.transpose(0, 1)
        elif layout == "offset":
            tensor = tensor[1]
        elif layout == "strided":
            # A non-dense column selection, preserving the requested shape.
            tensor = tensor.transpose(0, len(current))[1]
            if len(current) > 1:
                tensor = tensor.transpose(0, len(current) - 1)
        result.append(tensor)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=15)
    parser.add_argument("--samples", type=int, default=81)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=19351936)
    args = parser.parse_args()
    if min(args.warmups, args.samples, args.repeats) < 1:
        parser.error("warmups, samples and repeats must be positive")
    output = common._output_path(args.output)
    assert Path(sys.prefix).resolve() == ROOT / ".venv"
    common._configure_thread_environment(1, "")
    affinity = common._pin_cpu(None)
    np, native, reference = common._import_backends()
    common._validate_reference_version(reference)
    common._configure_reference_threads(reference, 1)
    native.set_num_threads(1)
    common._validate_thread_configuration(native, reference, 1)
    before = source_provenance()
    receipt = json.loads(args.build_record.read_text())
    assert all(receipt[key] == value for key, value in before.items())
    extension = Path(native._C.__file__).resolve()
    assert extension.is_relative_to(ROOT / ".venv")
    assert receipt["extension_sha256"] == sha256(extension)
    report = {
        "schema": "depth_concat_diagnostic_v1", "started_at": datetime.now(timezone.utc).isoformat(),
        "source": before, "git": common._git_provenance(),
        "measurement_kind": "precommit-diagnostic" if common._git_provenance()["status_short"] else "clean-code",
        "build_record": receipt, "build_record_sha256": sha256(args.build_record),
        "command": [sys.executable, *sys.argv], "cwd": str(ROOT),
        "python": sys.version, "python_executable": sys.executable,
        "interpreter_realpath": str(Path(sys.executable).resolve()),
        "interpreter_sha256": sha256(Path(sys.executable).resolve()),
        "extension_path": str(extension), "extension_sha256": sha256(extension),
        "harness_sha256": sha256(Path(__file__)),
        "common_harness_sha256": sha256(Path(common.__file__)),
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "module_paths": {m.__name__: m.__file__ for m in (np, native, reference)},
        "cpu": common._cpu_model_name(), "affinity": affinity,
        "threads": {"native": native.get_num_threads(), "reference": reference.get_num_threads(),
                    "native_interop": native.get_num_interop_threads(), "reference_interop": reference.get_num_interop_threads()},
        "device": "cpu", "warmups": args.warmups, "samples": args.samples,
        "repeats": args.repeats, "seed": args.seed,
        "timing_scope": "API call only; eager outputs materialized and bit-checked after every block; symmetric warmups and reversed orders; construction excluded",
        "credit": "targeted diagnostic only; no repository-wide percentage", "rows": [],
    }
    for name, api, shape, layout, count in workloads():
        inputs = {m.__name__: operands(m, np, shape, layout, count, args.seed) for m in (native, reference)}
        def call(module):
            tensors = inputs[module.__name__]
            return module.dstack(tensors) if api == "dstack" else module.cat(tensors, dim=int(api[-1]))
        expected = common._tensor_value_bits(np, call(reference))
        row = {"name": name, "api": api, "inputs": [common._tensor_metadata(x) for x in inputs[native.__name__]], "passes": []}
        for order in ((native, reference), (reference, native)):
            for module in order:
                samples = []
                gc.collect()
                was_enabled = gc.isenabled()
                gc.disable()
                try:
                    for sample in range(args.warmups + args.samples):
                        start = time.perf_counter_ns()
                        for _ in range(args.repeats):
                            result = call(module)
                        elapsed = time.perf_counter_ns() - start
                        np.testing.assert_array_equal(common._tensor_value_bits(np, result), expected)
                        if sample >= args.warmups:
                            samples.append(elapsed)
                finally:
                    if was_enabled:
                        gc.enable()
                row["passes"].append({"order": [m.__name__ for m in order], "implementation": module.__name__,
                                      **common._summarize_samples(samples, args.repeats)})
        report["rows"].append(row)
    assert source_provenance() == before
    assert sha256(extension) == receipt["extension_sha256"]
    report["ended_at"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
