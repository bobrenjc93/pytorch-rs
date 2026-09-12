#!/usr/bin/env python3
"""Fail-closed public-default compile coverage and CUDA performance measurement.

The only compile invocation is framework.compile(program), without overrides.
Worker processes never share imported frameworks or compiler caches. See
docs/torch-compile-default-evaluator.md for the finite corpus and score contract.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import gzip
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

from torch_compile_default_corpus import (
    CASES,
    CATEGORY_WEIGHTS,
    VARIANTS,
    VERSION,
    build,
)


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / "target/default-compile-eval/venv"
WARMUPS = 5
SAMPLES = 17
RTOL = 1e-4
ATOL = 1e-4
RTOL_OVERRIDES = {"bfloat16_roundtrip": 8e-3}
SOURCE_PATHS = (
    "src",
    "python",
    "crates",
    ".cargo",
    "build.rs",
    "Cargo.toml",
    "Cargo.lock",
    "pyproject.toml",
    "uv.lock",
    "rust-toolchain.toml",
    ".burner/evaluations.json",
    "scripts/torch_compile_default_corpus.py",
    "scripts/evaluate_torch_compile_default.py",
    "scripts/evaluate_torch_compile_default.sh",
    ".github/scripts/verify_native_extension.py",
)


class InvalidMeasurement(RuntimeError):
    """An infrastructure/reference/contract failure; never a numeric zero."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def source_identity():
    paths = git("ls-files", "--", *SOURCE_PATHS).splitlines()
    manifest = {p: digest((ROOT / p).read_bytes()) for p in paths}
    # Include new evaluator files during explicitly unscored diagnostic development.
    for p in SOURCE_PATHS:
        if (ROOT / p).is_file():
            manifest[p] = digest((ROOT / p).read_bytes())
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "working_tree_changes": git("status", "--porcelain", "--untracked-files=all"),
        "source_sha256": digest(json_bytes(manifest)),
        "files": manifest,
    }


def check_environment():
    if Path(sys.prefix).resolve() != VENV.resolve():
        raise InvalidMeasurement(
            f"use the worktree-local evaluator environment: {VENV}"
        )
    if not Path(sys.executable).resolve().is_relative_to(ROOT):
        raise InvalidMeasurement(
            "the base interpreter must also belong to this worktree"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise InvalidMeasurement("CUDA_VISIBLE_DEVICES must select one real GPU")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "interpreter_sha256": digest(Path(sys.executable).read_bytes()),
        "prefix": sys.prefix,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "cuda_visible_devices": visible,
    }


def gpu_snapshot():
    fields = "index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,clocks.sm,pstate"
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=" + fields, "--format=csv"], text=True, timeout=20
    ).strip()


class BlockReferenceImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError(
                "candidate evaluation forbids installed-PyTorch forwarding"
            )
        return None


class CudaSynchronizer:
    """Identical real device-wide synchronization, independent of Python API gaps."""

    def __init__(self):
        libraries = list(
            VENV.glob("lib/python*/site-packages/nvidia/cu13/lib/libcudart.so.13")
        )
        if len(libraries) != 1:
            raise InvalidMeasurement(
                "expected the locked environment's single CUDA 13 runtime"
            )
        self.path = libraries[0].resolve()
        self.runtime = ctypes.CDLL(str(self.path))
        self.runtime.cudaSetDevice.argtypes = [ctypes.c_int]
        self.runtime.cudaSetDevice.restype = ctypes.c_int
        self.runtime.cudaDeviceSynchronize.argtypes = []
        self.runtime.cudaDeviceSynchronize.restype = ctypes.c_int
        self.runtime.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.runtime.cudaRuntimeGetVersion.restype = ctypes.c_int
        self.check(self.runtime.cudaSetDevice(0))
        version = ctypes.c_int()
        self.check(self.runtime.cudaRuntimeGetVersion(ctypes.byref(version)))
        self.metadata = {
            "path": str(self.path),
            "sha256": digest(self.path.read_bytes()),
            "version": version.value,
            "api": "cudaDeviceSynchronize",
            "logical_device": 0,
        }
        self()

    @staticmethod
    def check(status):
        if status != 0:
            raise InvalidMeasurement(f"CUDA runtime synchronization error {status}")

    def __call__(self):
        self.check(self.runtime.cudaSetDevice(0))
        self.check(self.runtime.cudaDeviceSynchronize())


def verify_candidate(fw):
    native = importlib.import_module("torch_rs.torch_rs")
    if not isinstance(native.__spec__.loader, importlib.machinery.ExtensionFileLoader):
        raise InvalidMeasurement("candidate is not a native extension")
    installed = Path(fw.__file__).resolve().parent
    native_path = Path(native.__file__).resolve()
    if not installed.is_relative_to(VENV.resolve()) or not native_path.is_relative_to(
        VENV.resolve()
    ):
        raise InvalidMeasurement(
            "candidate resolved outside the current worktree environment"
        )
    source = ROOT / "python/torch_rs"
    expected = {
        str(p.relative_to(source)): digest(p.read_bytes()) for p in source.rglob("*.py")
    }
    actual = {
        str(p.relative_to(installed)): digest(p.read_bytes())
        for p in installed.rglob("*.py")
    }
    if actual != expected:
        raise InvalidMeasurement(
            "installed candidate Python sources do not match this checkout"
        )
    return {
        "package": str(installed),
        "native_extension": str(native_path),
        "native_sha256": digest(native_path.read_bytes()),
        "python_sources_sha256": digest(json_bytes(actual)),
    }


def verify_reference(fw):
    distribution = importlib.metadata.distribution("torch")
    records = {str(item): item for item in distribution.files or ()}
    origins = {}
    for name in (
        "torch",
        "torch._C",
        "torch._dynamo.eval_frame",
        "torch._inductor.compile_fx",
    ):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(VENV.resolve()):
            raise InvalidMeasurement(f"reference {name} resolved outside this worktree")
        relative = str(path.relative_to(Path(distribution.locate_file("")).resolve()))
        record = records.get(relative)
        checksum = hashlib.sha256(path.read_bytes()).digest()
        encoded = base64.urlsafe_b64encode(checksum).decode().rstrip("=")
        if (
            record is None
            or record.hash is None
            or record.hash.mode != "sha256"
            or record.hash.value != encoded
        ):
            raise InvalidMeasurement(
                f"reference {name} does not match its installed wheel RECORD"
            )
        origins[name] = {"path": str(path), "sha256": checksum.hex()}
    return {"distribution_version": distribution.version, "origins": origins}


def snapshot(value, fw, seen=None):
    """Materialize all values plus observable structure/metadata, not a checksum."""
    if seen is None:
        seen = {}
    if isinstance(value, fw.Tensor):
        if id(value) in seen:
            return {"alias_of": seen[id(value)]}
        index = len(seen)
        seen[id(value)] = index
        return {
            "tensor": index,
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype).split(".")[-1],
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
            "values": value.detach().cpu().tolist(),
        }
    if isinstance(value, dict):
        return {
            "dict": [[key, snapshot(item, fw, seen)] for key, item in value.items()]
        }
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [snapshot(item, fw, seen) for item in value]}
    if value is None or type(value) in (int, float, str, bool):
        return value
    raise TypeError(f"non-public or unsupported output type: {type(value).__name__}")


def compare(actual, expected, path="output", *, rtol=RTOL, atol=ATOL):
    """Compare every value; metadata/container/alias mismatches are not toleranced."""
    import numpy as np

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: structure differs")
        for key in expected:
            if key == "values":
                np.testing.assert_allclose(
                    actual[key],
                    expected[key],
                    rtol=rtol,
                    atol=atol,
                    equal_nan=False,
                    err_msg=path,
                )
            else:
                compare(
                    actual[key], expected[key], f"{path}.{key}", rtol=rtol, atol=atol
                )
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list length/type differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{path}[{index}]", rtol=rtol, atol=atol)
    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def tensors(value, fw):
    if isinstance(value, fw.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item, fw)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item, fw)


def invoke(function, args, case, fw):
    with fw.set_grad_enabled(case.training):
        output = function(*args)
        if case.training:
            output.sum().backward()
    return output


def observe(output, args, program, case, fw, device):
    leaves = list(tensors(output, fw))
    if not leaves:
        raise AssertionError("program returned no public Tensor")
    if any(str(t.device).split(":")[0] != device for t in leaves):
        raise AssertionError(f"output silently moved off {device}")
    result = {"output": snapshot(output, fw)}
    if case.training:
        differentiable = [t for t in tensors(args, fw) if t.requires_grad] + list(
            program.parameters
        )
        if not differentiable or any(t.grad is None for t in differentiable):
            raise AssertionError("missing required input or parameter gradient")
        result["gradients"] = snapshot([t.grad for t in differentiable], fw)
    if case.observe_inputs:
        result["inputs_after"] = snapshot(args, fw)
    if case.alias_probe:
        # A returned view must continue to alias after the compiled call returns.
        with fw.no_grad():
            output[0].add_(0.125)
        result["after_external_view_mutation"] = snapshot((output, args), fw)
    return result


def entry_code(function):
    return getattr(
        function,
        "__code__",
        getattr(getattr(function, "forward", None), "__code__", None),
    )


def reject_eager_wrapper(function, compiled, call):
    """An identity compiler or a warm wrapper rerunning the original body fails."""
    if compiled is function:
        raise AssertionError("compile returned the original Python callable")
    code = entry_code(function)
    if code is None:
        raise AssertionError("corpus entrypoint has no inspectable Python code")
    calls = 0

    def profile(frame, event, arg):
        nonlocal calls
        if event == "call" and frame.f_code is code:
            calls += 1

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = call()
    finally:
        sys.setprofile(previous)
    if calls:
        raise AssertionError(
            f"warm compiled call executed the original Python body {calls} time(s)"
        )
    return result


def latency_summary(samples):
    median = statistics.median(samples)
    if not math.isfinite(median) or median <= 0 or len(samples) != SAMPLES:
        raise InvalidMeasurement("invalid steady-state timing samples")
    return {
        "samples_ms": samples,
        "median_ms": median,
        "mad_ms": statistics.median(abs(v - median) for v in samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def measure_case(case, fw, implementation, device, synchronize):
    rtol = RTOL_OVERRIDES.get(case.name, RTOL)
    if implementation == "reference":
        fw.compiler.reset()
        from torch._dynamo.utils import counters
        from torch._inductor import metrics

        counters.clear()
        metrics.reset()
    program = build(case, fw, device)
    start = time.perf_counter_ns()
    compiled = fw.compile(program.function)
    factory_ms = (time.perf_counter_ns() - start) / 1e6
    if compiled is program.function:
        raise AssertionError("compile returned the original callable")
    result = {"status": "passed", "factory_ms": factory_ms, "variants": []}
    for variant in VARIANTS:

        def fresh(sample=0):
            for parameter in program.parameters:
                parameter.grad = None
            return program.inputs(variant, sample)

        args = fresh()
        synchronize()
        start = time.perf_counter_ns()
        output = invoke(compiled, args, case, fw)
        synchronize()
        cold_ms = (time.perf_counter_ns() - start) / 1e6
        observed = observe(output, args, program, case, fw, device)
        if implementation == "reference":
            eager = build(case, fw, device)
            eager_args = eager.inputs(variant)
            expected = observe(
                invoke(eager.function, eager_args, case, fw),
                eager_args,
                eager,
                case,
                fw,
                device,
            )
            compare(observed, expected, "reference compiled versus eager", rtol=rtol)

        for _ in range(WARMUPS):
            args = fresh()
            invoke(compiled, args, case, fw)
            synchronize()
        samples = []
        # All inputs, state resets, and allocations are outside each timed call.
        for _ in range(SAMPLES):
            args = fresh()
            synchronize()
            start = time.perf_counter_ns()
            output = invoke(compiled, args, case, fw)
            synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        warm_observed = observe(output, args, program, case, fw, device)
        compare(warm_observed, observed, "cold versus warm", rtol=rtol)
        args = fresh(sample=1)
        output = reject_eager_wrapper(
            program.function, compiled, lambda: invoke(compiled, args, case, fw)
        )
        if implementation == "candidate":
            if any(
                name == "torch" or name.startswith("torch.") for name in sys.modules
            ):
                raise AssertionError("candidate imported installed PyTorch")
        synchronize()
        changed = observe(output, args, program, case, fw, device)
        if implementation == "reference":
            eager = build(case, fw, device)
            eager_args = eager.inputs(variant, sample=1)
            expected = observe(
                invoke(eager.function, eager_args, case, fw),
                eager_args,
                eager,
                case,
                fw,
                device,
            )
            compare(
                changed,
                expected,
                "changed-input reference versus eager",
                rtol=rtol,
            )
        result["variants"].append(
            {
                "variant": variant,
                "cold_call_ms": cold_ms,
                "observed": observed,
                "changed_observed": changed,
                "changed_observed_sha256": digest(json_bytes(changed)),
                "observed_sha256": digest(json_bytes(observed)),
                **latency_summary(samples),
            }
        )
    if implementation == "reference":
        graphs = int(counters["stats"]["unique_graphs"])
        if graphs < 1:
            raise AssertionError("default reference produced no compiled graph")
        result["compiler_evidence"] = {
            "backend": fw.compiler.get_default_backend(),
            "unique_graphs": graphs,
            "generated_kernel_count": metrics.generated_kernel_count,
            "inductor": dict(counters["inductor"]),
            "graph_breaks": dict(counters["graph_break"]),
        }
    else:
        result["compiler_evidence"] = {
            "default_backend": fw.compiler.get_default_backend(),
            "original_body_on_warm_call": False,
            "installed_torch_imported": False,
        }
    return result


def worker(implementation, device, output_path):
    environment = check_environment()
    for variable, suffix in (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
    ):
        cache = Path(
            os.environ.setdefault(
                variable, str(output_path.parent / (output_path.stem + "-" + suffix))
            )
        )
        if not cache.resolve().is_relative_to(ROOT):
            raise InvalidMeasurement(f"{variable} must belong to this worktree")
    if implementation == "candidate":
        sys.meta_path.insert(0, BlockReferenceImport())
    fw = importlib.import_module(
        "torch" if implementation == "reference" else "torch_rs"
    )
    environment["framework_version"] = fw.__version__
    if implementation == "reference":
        if fw.__version__.split("+")[0] != "2.13.0":
            raise InvalidMeasurement("this corpus is pinned to PyTorch 2.13.0")
        environment["reference_provenance"] = verify_reference(fw)
        if fw.compiler.get_default_backend() != "inductor":
            raise InvalidMeasurement("the reference default backend is not Inductor")
        if fw._dynamo.config.suppress_errors:
            raise InvalidMeasurement(
                "reference error suppression would allow eager fallback"
            )
        if device == "cuda" and not fw.cuda.is_available():
            raise InvalidMeasurement("real CUDA unavailable to the reference")
        environment["cuda_runtime"] = fw.version.cuda
        environment["float32_matmul_precision"] = fw.get_float32_matmul_precision()
        if device == "cuda":
            properties = fw.cuda.get_device_properties(0)
            environment["selected_gpu"] = {
                "name": properties.name,
                "uuid": str(properties.uuid),
                "compute_capability": [properties.major, properties.minor],
                "total_memory": properties.total_memory,
            }
    else:
        environment.update(verify_candidate(fw))
    fw.set_num_threads(1)
    environment["threads"] = fw.get_num_threads()
    synchronize = CudaSynchronizer() if device == "cuda" else lambda: None
    if device == "cuda":
        environment["synchronization"] = synchronize.metadata
    report = {
        "implementation": implementation,
        "device": device,
        "environment": environment,
        "cases": {},
    }
    for case in CASES:
        print(f"{implementation}/{device}: {case.name}", file=sys.stderr, flush=True)
        try:
            report["cases"][case.name] = measure_case(
                case, fw, implementation, device, synchronize
            )
        except InvalidMeasurement:
            raise
        except Exception as error:
            report["cases"][case.name] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
    with gzip.open(output_path, "wt") as output:
        json.dump(report, output, allow_nan=False)


def run_worker(implementation, device, round_number, directory):
    label = f"{device}-{round_number}-{implementation}"
    output = directory / f"{label}.json.gz"
    log = directory / f"{label}.log"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("TORCH", "TRITON"))
    }
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "TORCHINDUCTOR_CACHE_DIR": str(directory / f"{label}-inductor-cache"),
            "TRITON_CACHE_DIR": str(directory / f"{label}-triton-cache"),
        }
    )
    print(f"running {label}; log: {log}", file=sys.stderr, flush=True)
    started = time.time()
    with log.open("w") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                implementation,
                "--device",
                device,
                "--worker-output",
                str(output),
            ],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=stream,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=1200)
        except BaseException:
            # Own process group only: do not leave compiler workers orphaned.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            except ProcessLookupError:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    if returncode != 0:
        raise InvalidMeasurement(f"{label} worker failed ({returncode}); inspect {log}")
    with gzip.open(output, "rt") as stream:
        report = json.load(stream)
    report["artifacts"] = {
        "output": str(output),
        "output_sha256": digest(output.read_bytes()),
        "log": str(log),
        "log_sha256": digest(log.read_bytes()),
        "wall_seconds": time.time() - started,
        "cache_state": "unique empty process-local Inductor/Triton directories at worker start",
    }
    return report


def geometric_mean(values):
    if not values:
        return None
    if any(value <= 0 for value in values):
        return 0.0
    return math.exp(statistics.mean(math.log(value) for value in values))


def aggregate(cells, metric):
    """Missing candidate cells are explicit zeros in the fixed denominator."""
    score = 0.0
    categories = {}
    for category, weight in CATEGORY_WEIGHTS.items():
        rows = [cell for cell in cells if cell["category"] == category]
        if not rows:
            raise InvalidMeasurement(f"missing reference category: {category}")
        if metric == "coverage":
            value = sum(cell["passed"] for cell in rows) / len(rows)
        else:
            value = geometric_mean(
                [min(1.0, cell["ratio"]) if cell["passed"] else 0.0 for cell in rows]
            )
        categories[category] = {
            "passed": sum(cell["passed"] for cell in rows),
            "eligible": len(rows),
            "weight": weight,
            "contribution": weight * value,
        }
        score += weight * value
    return {
        "score": score,
        "passed": sum(cell["passed"] for cell in cells),
        "eligible": len(cells),
        "categories": categories,
        "common_success_geomean_ratio": geometric_mean(
            [
                cell["ratio"]
                for cell in cells
                if cell["passed"] and cell.get("ratio") is not None
            ]
        ),
    }


def compare_workers(reference, candidate):
    """Never shrink eligibility after discovering a failed reference case."""
    cells = []
    expected_names = {case.name for case in CASES}
    if (
        set(reference["cases"]) != expected_names
        or set(candidate["cases"]) != expected_names
    ):
        raise InvalidMeasurement("incomplete or changed workload manifest")
    for case in CASES:
        ref = reference["cases"][case.name]
        native = candidate["cases"][case.name]
        if ref["status"] != "passed":
            raise InvalidMeasurement(
                f"reference case {case.name} failed: {ref.get('error')}"
            )
        if [item["variant"] for item in ref["variants"]] != list(VARIANTS):
            raise InvalidMeasurement(
                f"reference {case.name}: missing/reordered variants"
            )
        for index, variant in enumerate(VARIANTS):
            ref_value = ref["variants"][index]
            row = {
                "case": case.name,
                "category": case.category,
                "variant": variant,
                "device": reference["device"],
                "passed": False,
                "ratio": 0.0,
                "reference": {
                    key: value
                    for key, value in ref_value.items()
                    if key not in ("observed", "changed_observed")
                },
            }
            if native["status"] != "passed":
                row["error"] = native.get("error", "candidate did not pass")
            else:
                try:
                    value = native["variants"][index]
                    if value["variant"] != variant:
                        raise AssertionError("missing/reordered candidate variant")
                    rtol = RTOL_OVERRIDES.get(case.name, RTOL)
                    compare(value["observed"], ref_value["observed"], rtol=rtol)
                    compare(
                        value["changed_observed"],
                        ref_value["changed_observed"],
                        "same-shape new inputs",
                        rtol=rtol,
                    )
                    ratio = ref_value["median_ms"] / value["median_ms"]
                    if not math.isfinite(ratio) or ratio <= 0:
                        raise AssertionError("invalid latency ratio")
                    row.update(
                        passed=True,
                        ratio=ratio,
                        candidate={
                            key: val
                            for key, val in value.items()
                            if key not in ("observed", "changed_observed")
                        },
                    )
                except (
                    AssertionError,
                    KeyError,
                    IndexError,
                    TypeError,
                    ZeroDivisionError,
                ) as error:
                    row["error"] = f"{type(error).__name__}: {error}"
            cells.append(row)
    return cells


def merge_rounds(rounds):
    if len({len(rows) for rows in rounds}) != 1:
        raise InvalidMeasurement("round denominators differ")
    result = []
    for rows in zip(*rounds):
        identities = {(row["case"], row["variant"], row["device"]) for row in rows}
        if len(identities) != 1:
            raise InvalidMeasurement("round workload identities differ")
        merged = {
            key: value
            for key, value in rows[0].items()
            if key not in ("reference", "candidate")
        }
        merged["passed"] = all(row["passed"] for row in rows)
        merged["ratio"] = (
            geometric_mean([row["ratio"] for row in rows]) if merged["passed"] else 0.0
        )
        merged["rounds"] = list(rows)
        result.append(merged)
    return result


def wheel_identity(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT / "target/default-compile-eval"):
        raise InvalidMeasurement("wheel must be newly built inside this worktree")
    with zipfile.ZipFile(path) as wheel:
        extensions = [
            name
            for name in wheel.namelist()
            if name.startswith("torch_rs/") and name.endswith(".so")
        ]
        if len(extensions) != 1:
            raise InvalidMeasurement("wheel must contain one native extension")
        installed = next(VENV.glob("lib/python*/site-packages")) / extensions[0]
        if installed.read_bytes() != wheel.read(extensions[0]):
            raise InvalidMeasurement(
                "installed native extension does not match the measured wheel"
            )
    return {
        "path": str(path),
        "sha256": digest(path.read_bytes()),
        "build_command": "maturin build --release --locked",
        "profile": "release",
    }


def setup_identity(build_identity, timestamps, provenance):
    built = json.loads(build_identity)
    if built != {
        key: provenance[key]
        for key in ("commit", "source_sha256", "working_tree_changes")
    }:
        raise InvalidMeasurement("checkout changed between wheel build and evaluation")
    times = [int(value) for value in timestamps.split(",")]
    if len(times) != 4 or times != sorted(times) or times[0] <= 0:
        raise InvalidMeasurement("missing or invalid setup timestamps")
    return {
        "build_identity": built,
        "started_at_unix_ns": times[0],
        "finished_at_unix_ns": times[-1],
        "environment_and_dependencies_seconds": (times[1] - times[0]) / 1e9,
        "native_build_seconds": (times[2] - times[1]) / 1e9,
        "install_and_verification_seconds": (times[3] - times[2]) / 1e9,
        "total_seconds": (times[-1] - times[0]) / 1e9,
        "cache_state": "worktree-local dependency/build caches may be warm; a fresh wheel is always built and installed",
    }


def evaluate(args):
    directory = (
        ROOT
        / "target/default-compile-eval"
        / (
            "run-"
            + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            + "-"
            + uuid.uuid4().hex[:8]
        )
    )
    directory.mkdir(parents=True)
    provenance = source_identity()
    report = {
        "version": VERSION,
        "valid": False,
        "diagnostic": args.diagnostic,
        "provenance": provenance,
        "metric": args.metric,
        "compile_call": "framework.compile(program)",
        "warmups": WARMUPS,
        "samples": SAMPLES,
        "rtol": RTOL,
        "atol": ATOL,
        "workers": [],
        "run_directory": str(directory),
    }
    try:
        if args.output:
            destination = Path(args.output).resolve()
            if not destination.is_relative_to(ROOT) or destination.exists():
                raise InvalidMeasurement(
                    "output must be a new report path inside this worktree; existing evidence is never overwritten"
                )
        if not args.diagnostic:
            if provenance["working_tree_changes"]:
                raise InvalidMeasurement("scoring requires a clean committed checkout")
            git(
                "ls-files",
                "--error-unmatch",
                "scripts/torch_compile_default_corpus.py",
                "scripts/evaluate_torch_compile_default.py",
                "scripts/evaluate_torch_compile_default.sh",
            )
        report["environment"] = check_environment()
        report["wheel"] = wheel_identity(Path(args.wheel))
        report["setup"] = setup_identity(
            args.build_identity, args.setup_timestamps, provenance
        )
        report["rtol_overrides"] = RTOL_OVERRIDES
        report["gpu_before"] = gpu_snapshot()
        report["rustc"] = subprocess.check_output(
            ["rustc", "--version"], text=True
        ).strip()
        all_cells = []
        for device in ("cuda",) if args.metric == "cuda-perf" else ("cpu", "cuda"):
            rounds = []
            orders = (
                (("reference", "candidate"), ("candidate", "reference"))
                if device == "cuda"
                else (("reference", "candidate"),)
            )
            for number, order in enumerate(orders):
                measured = {}
                for implementation in order:
                    measured[implementation] = run_worker(
                        implementation, device, number, directory
                    )
                    report["workers"].append(
                        {
                            key: value
                            for key, value in measured[implementation].items()
                            if key != "cases"
                        }
                        | {
                            "case_status": {
                                name: {
                                    key: value
                                    for key, value in item.items()
                                    if key not in ("variants", "traceback")
                                }
                                for name, item in measured[implementation][
                                    "cases"
                                ].items()
                            }
                        }
                    )
                rounds.append(
                    compare_workers(measured["reference"], measured["candidate"])
                )
            all_cells.extend(merge_rounds(rounds))
        if source_identity() != provenance:
            raise InvalidMeasurement("source or HEAD changed during measurement")
        report["gpu_after"] = gpu_snapshot()
        report["cells"] = all_cells
        report["scores"] = {}
        if args.metric != "cuda-perf":
            report["scores"]["coverage"] = aggregate(all_cells, "coverage")
        report["scores"]["cuda-perf"] = aggregate(
            [cell for cell in all_cells if cell["device"] == "cuda"], "cuda-perf"
        )
        report["valid"] = not args.diagnostic
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        write_json(directory / "report.json", report)
        print(
            f"invalid measurement; no score: {report['error']}; report: {directory / 'report.json'}",
            file=sys.stderr,
        )
        return 2
    write_json(directory / "report.json", report)
    if args.output:
        destination = Path(args.output).resolve()
        if not destination.is_relative_to(ROOT):
            raise InvalidMeasurement(
                "published report must remain inside this worktree"
            )
        write_json(destination, report)
    if args.diagnostic or args.metric == "both":
        print(
            json.dumps(
                {
                    "valid": report["valid"],
                    "diagnostic": args.diagnostic,
                    "scores": report["scores"],
                    "report": str(directory / "report.json"),
                }
            )
        )
    else:
        result = report["scores"][args.metric]
        print(
            json.dumps(
                {
                    "score": result["score"],
                    "summary": f"{VERSION} {args.metric}: {result['score']:.6g}/100; {result['passed']}/{result['eligible']} fixed public-default cells pass. This is finite-corpus parity, not all Python programs.",
                    "evidence": [
                        f"Report: {directory / 'report.json'}",
                        f"Source {provenance['commit']} ({provenance['source_sha256']}); public default Inductor reference; CUDA_VISIBLE_DEVICES={report['environment']['cuda_visible_devices']}",
                        f"Common-success geometric mean ratio: {result['common_success_geomean_ratio']}; failed candidate cells remain zero.",
                    ],
                    "suggestions": [
                        "Implement the missing public default compiler paths; do not change this independently reviewed corpus in feature branches."
                    ]
                    if result["score"] < 100
                    else [],
                }
            )
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric", choices=("coverage", "cuda-perf", "both"), default="both"
    )
    parser.add_argument("--wheel")
    parser.add_argument("--output")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="allow development changes; never emit a Burner score",
    )
    parser.add_argument(
        "--source-identity", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--build-identity", help=argparse.SUPPRESS)
    parser.add_argument("--setup-timestamps", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker", choices=("reference", "candidate"), help=argparse.SUPPRESS
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.source_identity:
        identity = source_identity()
        print(
            json.dumps(
                {
                    key: identity[key]
                    for key in ("commit", "source_sha256", "working_tree_changes")
                }
            )
        )
        return 0
    if args.worker:
        worker(args.worker, args.device, args.worker_output)
        return 0
    if not args.wheel or not args.build_identity or not args.setup_timestamps:
        parser.error(
            "build receipt is required; use scripts/evaluate_torch_compile_default.sh"
        )
    return evaluate(args)


if __name__ == "__main__":

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(main())
