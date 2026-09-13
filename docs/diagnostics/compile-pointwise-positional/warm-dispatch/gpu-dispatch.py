"""Unscored clean-wheel H100 dispatch diagnostic: build, one leg, verify 16 legs."""
import argparse
import ctypes
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[4]
VERSION = "warm-dispatch-gpu-v2"
BEFORE = "885264b5319d76165e6be8d6df405450b3830c47"
SOURCE_PATHS = ("src", "python", "crates", ".cargo", "build.rs", "Cargo.toml",
                "Cargo.lock", "pyproject.toml", "uv.lock", "rust-toolchain.toml")
CASES = (
    ("literal_unary", "def f(x):\n return x*1.25+0.75", "single"),
    ("tensor_arithmetic", "def f(x,y):\n return x*y+x", "pair"),
    ("broadcast_add", "def f(x,y):\n return x+y", "broadcast"),
    ("repeated_alias", "def f(x,y):\n return x+y", "alias"),
    ("shape_revisit", "def f(x,y):\n return x*y+y", "shapes"),
    ("promoted_capture", "def f(x):\n return x*scale+x", "promoted"),
    ("eight_boolean_entries", "def f(x):\n return x*a+x*b+x*c", "booleans"),
)
# Each entry names its build and implementation. R legs remain associated with
# their native build; every invocation uses a fresh process and cache directory.
ORDER = (("B", "native"), ("B", "reference"), ("C", "native"), ("C", "reference"),
         ("C", "reference"), ("C", "native"), ("B", "reference"), ("B", "native"),
         ("C", "native"), ("C", "reference"), ("B", "native"), ("B", "reference"),
         ("B", "reference"), ("B", "native"), ("C", "reference"), ("C", "native"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inside(path, base=ROOT):
    path = Path(path).resolve()
    assert path.is_relative_to(base), (str(path), str(base))
    return path


def command(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def source(root):
    assert Path(command("git", "rev-parse", "--show-toplevel", cwd=root)).resolve() == root
    status = command("git", "status", "--porcelain=v1", "--untracked-files=all", cwd=root)
    assert not status, status
    paths = command("git", "ls-files", "--", *SOURCE_PATHS, cwd=root).splitlines()
    return {"root": str(root), "commit": command("git", "rev-parse", "HEAD", cwd=root),
            "status": status, "files": {p: sha((root/p).read_bytes()) for p in paths}}


def write(path, record):
    data = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def inventory():
    return command("nvidia-smi", "--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used",
                   "--format=csv,noheader")


def gpu_identity(expected):
    snapshot = inventory()
    row = next(line for line in snapshot.splitlines() if line.split(",")[0].strip() == "0")
    assert row.split(",")[1].strip() == expected and "H100" in row, row
    driver = ctypes.CDLL("libcuda.so.1")
    device, identity = ctypes.c_int(), (ctypes.c_ubyte * 16)()
    assert driver.cuInit(0) == 0 and driver.cuDeviceGet(ctypes.byref(device), 0) == 0
    assert driver.cuDeviceGetUuid(ctypes.byref(identity), device) == 0
    assert "GPU-" + str(uuid.UUID(bytes=bytes(identity))) == expected
    return snapshot


def runtime_identity():
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if "libcudart.so" in line or "libnvrtc.so" in line}
    result = []
    for name in sorted(paths):
        library = ctypes.CDLL(name)
        version, minor = ctypes.c_int(), ctypes.c_int()
        if "libcudart" in name:
            assert library.cudaRuntimeGetVersion(ctypes.byref(version)) == 0
            version = version.value
        else:
            assert library.nvrtcVersion(ctypes.byref(version), ctypes.byref(minor)) == 0
            version = [version.value, minor.value]
        result.append({"path": name, "sha256": sha(Path(name).read_bytes()), "version": version})
    return result


class CudaSynchronizer:
    """Use the same device-wide CUDA runtime barrier for both frameworks."""

    def __init__(self):
        libraries = list(inside(sys.prefix).glob(
            "lib/python*/site-packages/nvidia/cu13/lib/libcudart.so.13"))
        assert len(libraries) == 1, "Expected the locked environment's single CUDA 13 runtime"
        path = inside(libraries[0])
        self.runtime = ctypes.CDLL(str(path))
        self.runtime.cudaSetDevice.argtypes = [ctypes.c_int]
        self.runtime.cudaSetDevice.restype = ctypes.c_int
        self.runtime.cudaDeviceSynchronize.argtypes = []
        self.runtime.cudaDeviceSynchronize.restype = ctypes.c_int
        self.runtime.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.runtime.cudaRuntimeGetVersion.restype = ctypes.c_int
        version = ctypes.c_int()
        self.check(self.runtime.cudaRuntimeGetVersion(ctypes.byref(version)))
        self.metadata = {"path": str(path), "sha256": sha(path.read_bytes()),
                         "version": version.value, "api": "cudaDeviceSynchronize",
                         "logicalDevice": 0}
        self()

    @staticmethod
    def check(status):
        if status != 0:
            raise RuntimeError(f"CUDA runtime synchronization error {status}")

    def __call__(self):
        self.check(self.runtime.cudaSetDevice(0))
        self.check(self.runtime.cudaDeviceSynchronize())


def environment(root, directory):
    assert inside(sys.executable).is_file() and inside(sys.prefix).is_dir()
    assert not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME")
    sys.dont_write_bytecode = True
    for name, relative in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"),
                           ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"),
                           ("CUDA_CACHE_PATH", "cuda-cache"), ("TORCH_HOME", "torch-home")):
        path = inside(directory/relative)
        path.mkdir(parents=True)
        os.environ[name] = str(path)
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONNOUSERSITE="1",
                      PYTHONDONTWRITEBYTECODE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    for name, relative in (("CARGO_HOME", "target/cargo-home"), ("CARGO_TARGET_DIR", "target"),
                           ("UV_CACHE_DIR", "target/uv-cache")):
        os.environ[name] = str(inside(root/relative))


def build(args, record, directory):
    root = inside(args.source_root)
    environment(root, directory)
    initial = source(root)
    assert initial["commit"] == args.commit
    record["source"] = initial
    wheels = directory/"wheels"
    wheels.mkdir()
    command_line = [str(inside(Path(sys.prefix)/"bin/maturin")), "build", "--release", "--locked", "--out", str(wheels)]
    record["buildCommand"] = command_line
    os.environ.update(PYO3_PYTHON=sys.executable, VIRTUAL_ENV=sys.prefix, CARGO_BUILD_JOBS="8")
    with (directory/"build.log").open("w") as log:
        subprocess.run(command_line, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    record["buildLogSha256"] = sha((directory/"build.log").read_bytes())
    assert source(root) == initial
    found = list(wheels.glob("*.whl"))
    assert len(found) == 1
    wheel = found[0]
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("torch_rs/") and name.endswith(".py"):
                assert archive.read(name) == (root/"python"/name).read_bytes(), name
    record.update(wheel=str(wheel), wheelSha256=sha(wheel.read_bytes()),
                  rustc=command("rustc", "-Vv", cwd=root), nvcc=command("nvcc", "--version"))


def installed(build_record):
    root = inside(build_record["source"]["root"])
    assert source(root) == build_record["source"]
    wheel = inside(build_record["wheel"])
    assert sha(wheel.read_bytes()) == build_record["wheelSha256"]
    spec = importlib.util.find_spec("torch_rs")
    package = inside(Path(spec.origin).parent, inside(sys.prefix))
    with zipfile.ZipFile(wheel) as archive:
        entries = [n for n in archive.namelist() if n.startswith("torch_rs/") and (n.endswith(".py") or n.endswith(".so"))]
        assert any(n.endswith(".so") for n in entries)
        for name in entries:
            data = archive.read(name)
            assert (package.parent/name).read_bytes() == data, name
            if name.endswith(".py"):
                assert (root/"python"/name).read_bytes() == data, name
    return {"package": str(package), "wheel": str(wheel), "wheelSha256": build_record["wheelSha256"]}


def states(torch, kind):
    # Exactly representable deterministic values, no framework-dependent RNG.
    def tensor(shape, phase):
        count = math.prod(shape)
        return torch.tensor([((i + phase) % 23 - 11) * 0.03125 for i in range(count)],
                            dtype=torch.float32).reshape(shape).to("cuda:0")
    x, y = tensor((31, 7), 0), tensor((31, 7), 3)
    if kind == "single": return [((x,), {})]
    if kind == "pair": return [((x, y), {})]
    if kind == "broadcast": return [((x, tensor((7,), 3)), {})]
    if kind == "alias": return [((x, x), {})]
    if kind == "shapes": return [((x, y), {}), ((tensor((47, 7), 0), tensor((47, 7), 3)), {})]
    if kind == "promoted": return [((x,), {"scale": value}) for value in (1.25, 2.5, 0.75)]
    assert kind == "booleans"
    return [((x,), dict(zip(("a", "b", "c"), (bool(mask & (1 << bit)) for bit in range(3))))) for mask in range(8)]


def encoded_values(values):
    # Hex strings preserve signed zero and nonfinite classes in strict JSON.
    return [encoded_values(value) for value in values] if isinstance(values, list) else float(values).hex()


def tensor_record(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "requiresGrad": tensor.requires_grad,
            "values": encoded_values(tensor.cpu().tolist())}


def history(torch, source_text, kind, row, synchronize):
    namespace = {}
    exec(source_text, namespace)
    model = namespace["f"]
    compiled = torch.compile(model)  # Ordinary public default, no overrides.
    calls = states(torch, kind)
    inputs = [tensor_record(t) for args, _ in calls for t in args]
    row.update(inputs=inputs, states=[updates for _, updates in calls], callsPerSample=len(calls), traversals=[])
    previous_outputs = []
    for phase, count in (("setup", 2), ("warmup", 5), ("sample", 17)):
        for index in range(count):
            outputs = []
            synchronize()
            start = time.perf_counter_ns()
            for args, updates in calls:
                namespace.update(updates)
                outputs.append(compiled(*args))
            synchronize()
            elapsed = time.perf_counter_ns() - start
            observations = [tensor_record(output) for output in outputs]
            row["traversals"].append({"phase": phase, "index": index, "elapsedNs": elapsed, "outputs": observations})
            # Checks/materialization are outside the timed interval.
            assert [tensor_record(t) for args, _ in calls for t in args] == inputs
            pointers = [output.data_ptr() for output in outputs]
            assert len(set(pointers)) == len(pointers), "Outputs alias within a traversal"
            assert not set(pointers).intersection(t.data_ptr() for args, _ in calls for t in args)
            assert not set(pointers).intersection(t.data_ptr() for t in previous_outputs)
            previous_outputs = outputs
            for output in outputs:
                assert output.dtype == torch.float32 and str(output.device) == "cuda:0"
                assert not output.requires_grad and output.is_contiguous()
    row["sampleNs"] = [x["elapsedNs"] for x in row["traversals"] if x["phase"] == "sample"]
    row["medianNsPerCall"] = statistics.median(row["sampleNs"])/len(calls)


def leg(args, record, directory):
    build_record = read(inside(args.build_record))
    assert build_record["passed"] and build_record["commandKind"] == "build"
    root = inside(build_record["source"]["root"])
    environment(root, directory)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert (args.build_label, args.implementation) == ORDER[args.leg_index]
    assert (build_record["source"]["commit"] == BEFORE) == (args.build_label == "B")
    record.update(buildRecord=str(inside(args.build_record)), buildRecordSha256=sha(inside(args.build_record).read_bytes()),
                  source=build_record["source"], nativeInstall=installed(build_record),
                  legIndex=args.leg_index, buildLabel=args.build_label, implementation=args.implementation,
                  gpuUuid=args.gpu_uuid, gpuBefore=gpu_identity(args.gpu_uuid), cases=[])
    torch = importlib.import_module("torch_rs" if args.implementation == "native" else "torch")
    record["frameworkPath"] = str(inside(torch.__file__, inside(sys.prefix)))
    record["frameworkVersion"] = torch.__version__
    if args.implementation == "reference":
        assert torch.__version__ == "2.13.0+cu130"
    torch.set_num_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    synchronize = CudaSynchronizer()
    record["synchronization"] = synchronize.metadata
    for name, text, kind in CASES:
        row = {"name": name, "program": text, "passed": False}
        record["cases"].append(row)
        try:
            history(torch, text, kind, row, synchronize)
            row["passed"] = True
        except Exception:
            row["failure"] = traceback.format_exc()
    record["gpuAfter"] = gpu_identity(args.gpu_uuid)
    record["runtime"] = runtime_identity()
    assert source(root) == build_record["source"]
    record["numericalComparison"] = "Deferred until both separate-process legs finish; verify compares every recorded output."
    assert all(row["passed"] for row in record["cases"])
    if args.implementation == "native":
        assert "torch" not in sys.modules


def close_outputs(left, right):
    for key in ("shape", "stride", "dtype", "device", "requiresGrad"):
        assert left[key] == right[key], key
    def compare(a, b):
        if isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b)
            for x, y in zip(a, b): compare(x, y)
        else:
            a, b = float.fromhex(a), float.fromhex(b)
            # Match the existing pointwise JIT comparison: reference-relative
            # absolute-plus-relative tolerance, equal NaNs, and IEEE zero signs.
            if math.isnan(a) and math.isnan(b):
                return
            if a == 0 and b == 0:
                assert math.copysign(1, a) == math.copysign(1, b), (a, b)
                return
            assert a == b or (math.isfinite(a) and math.isfinite(b)
                              and abs(a-b) <= 1e-6 + 1e-5*abs(b)), (a, b)
    compare(left["values"], right["values"])


def verify(args, record, directory):
    record.update(reports=[], pairs=[], comparisonFailures=[], passed=False)
    errors = record["comparisonFailures"]

    def failed(location, error):
        errors.append({"location": location, "error": str(error)})

    def check(condition, location, message):
        if not condition:
            failed(location, message)

    # Preserve every input identity before inspecting any comparison. A missing
    # or malformed report must not hide later independent pairs or their failures.
    reports = []
    for index, path in enumerate(args.reports):
        item = {"path": str(path)}
        record["reports"].append(item)
        try:
            path = inside(path)
            item.update(path=str(path), sha256=sha(path.read_bytes()))
            reports.append(read(path))
        except Exception as error:
            failed(f"report[{index}]", error)
            reports.append(None)
    check(len(reports) == 16, "reports", "Expected all 16 legs")
    gpu, corrected = set(), set()
    for index, report in enumerate(reports):
        try:
            assert report["passed"] and report["legIndex"] == index
            assert (report["buildLabel"], report["implementation"]) == ORDER[index]
            assert len(report["cases"]) == 7
            assert report["scriptSha256"] == record["scriptSha256"]
            assert report["diagnosticVersion"] == VERSION
            synchronization = report["synchronization"]
            assert synchronization["api"] == "cudaDeviceSynchronize"
            assert synchronization["logicalDevice"] == 0
            assert synchronization == reports[0]["synchronization"], "Synchronization identity differs"
            gpu.add(report["gpuUuid"])
            if report["buildLabel"] == "C": corrected.add(report["source"]["commit"])
            if index:
                assert reports[index-1]["finishedAt"] <= report["startedAt"], "Leg order changed or overlapped"
        except Exception as error:
            failed(f"report[{index}].contract", repr(error))
    check(len(gpu) == len(corrected) == 1, "reports.identity", "GPU/corrected commit differs or is absent")
    for index in range(0, 16, 2):
        for case_index, expected in enumerate(CASES):
            row = {"legIndices": [index, index+1], "history": expected[0],
                   "outputsCompared": 0, "passed": False}
            record["pairs"].append(row)
            where = f"pair[{index},{index+1}].{expected[0]}"
            before_errors = len(errors)
            try:
                pair = {r["implementation"]: r for r in reports[index:index+2]}
                native, reference = pair["native"], pair["reference"]
                check(native["source"] == reference["source"], where, "Source identities differ")
                left, right = native["cases"][case_index], reference["cases"][case_index]
                check(left["name"] == right["name"] == expected[0], where, "History names differ")
                check(left["inputs"] == right["inputs"] and left["states"] == right["states"],
                      where, "Inputs or state histories differ")
                a_traversals, b_traversals = left["traversals"], right["traversals"]
                check(len(a_traversals) == len(b_traversals) == 24, where, "Expected all 24 traversals")
                row.update(nativeMedianNsPerCall=left.get("medianNsPerCall"),
                           referenceMedianNsPerCall=right.get("medianNsPerCall"))
            except Exception as error:
                failed(where, repr(error))
                continue
            for traversal_index in range(max(len(a_traversals), len(b_traversals))):
                traversal_where = f"{where}.traversal[{traversal_index}]"
                try:
                    a, b = a_traversals[traversal_index], b_traversals[traversal_index]
                    check((a["phase"], a["index"]) == (b["phase"], b["index"]),
                          traversal_where, "Traversal phases or indices differ")
                    a_outputs, b_outputs = a["outputs"], b["outputs"]
                    check(len(a_outputs) == len(b_outputs), traversal_where, "Output counts differ")
                except Exception as error:
                    failed(traversal_where, repr(error))
                    continue
                for output_index in range(max(len(a_outputs), len(b_outputs))):
                    row["outputsCompared"] += 1
                    try:
                        close_outputs(a_outputs[output_index], b_outputs[output_index])
                    except Exception as error:
                        failed(f"{traversal_where}.output[{output_index}]", repr(error))
            row["passed"] = len(errors) == before_errors
    record["passed"] = not errors
    assert record["passed"], f"{len(errors)} retained comparison/contract failures"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command_kind", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--source-root", required=True, type=Path)
    build_parser.add_argument("--commit", required=True)
    leg_parser = sub.add_parser("leg")
    leg_parser.add_argument("--build-record", required=True, type=Path)
    leg_parser.add_argument("--build-label", choices=("B", "C"), required=True)
    leg_parser.add_argument("--implementation", choices=("native", "reference"), required=True)
    leg_parser.add_argument("--leg-index", type=int, choices=range(16), required=True)
    leg_parser.add_argument("--gpu-uuid", required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("reports", nargs=16, type=Path)
    for child in (build_parser, leg_parser, verify_parser):
        child.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = inside(args.output)
    assert not output.exists() and not output.parent.exists(), "Each invocation requires a fresh output directory"
    output.parent.mkdir(parents=True)
    record = {"diagnosticVersion": VERSION, "commandKind": args.command_kind,
              "startedAt": datetime.now(timezone.utc).isoformat(), "command": sys.orig_argv,
              "scriptPath": str(Path(__file__).resolve()), "scriptSha256": sha(Path(__file__).read_bytes()),
              "executable": str(inside(sys.executable)), "executableSha256": sha(Path(sys.executable).read_bytes()),
              "python": sys.version, "qualificationOrScoreProduced": False, "passed": False}
    try:
        globals()[args.command_kind](args, record, output.parent)
        record["passed"] = True
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["finishedAt"] = datetime.now(timezone.utc).isoformat()
        write(output, record)


if __name__ == "__main__":
    main()
