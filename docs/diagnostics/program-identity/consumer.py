"""Non-scoring Program-identity protocol: clean build, freeze, serial legs, offline verify.

Every command writes a new immutable attempt directory, including on failure.
No command changes a checkout, installs a wheel, or acquires a GPU lease.
"""
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

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
VERSION = "program-identity-public-v2"
BASELINE = "60202557b4f110d07777f585e804ab5f55e1ff7b"
PROTOCOL_SHA = "d6e8f51d024b451f8afed33d743647d46396425724b918a599f7b32274f95d5b"
GPU = "GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1"
COMPILER_OPTIONS = ("--gpu-architecture=compute_90", "--fmad=true", "--ftz=false",
                    "--prec-div=true", "--prec-sqrt=true")
ORDER = (("B", "native"), ("B", "reference"), ("C", "native"), ("C", "reference"),
         ("C", "reference"), ("C", "native"), ("B", "reference"), ("B", "native"))
LENGTHS = (3, 5, 5, 7, 9, 11, 13, 15, 17, 19, 21, 5)
WIDTHS = (3, 5, 7, 9, 11, 13, 15, 17, 19, 3)
CASES = (
    ("affine", "def f(x):\n    return x * 1.25 + 0.75\n"),
    ("trig", "def f(x, y):\n    return (x.sin() + y.cos()).relu()\n"),
    ("shared", "def f(x, y):\n    p = x * y\n    for _ in range(37):\n        p = p * 0.9375 + x * 0.03125\n    return p\n"),
    ("broadcast", "def f(x, y):\n    return x + y\n"),
    ("structured", "def f(data):\n    x, y = data['pair']\n    p = x * y\n    return {'value': (p, p.sin()), 'alias': x}\n"),
    ("over_cap", "def f(x, y):\n    p = x * y\n    for _ in range(320):\n        p = p * x + y\n    return p\n"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inside(path, root=ROOT):
    path = Path(path).resolve()
    require(path.is_relative_to(root.resolve()), f"Outside worktree: {path}")
    return path


def command(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith(".gz") else data)


def write(path, record):
    data = (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(gzip.compress(data, mtime=0) if str(path).endswith(".gz") else data)


def bindings():
    require(sha((HERE / "protocol.md").read_bytes()) == PROTOCOL_SHA, "Frozen protocol changed")
    return {"protocol": PROTOCOL_SHA, "consumer": sha(Path(__file__).read_bytes()),
            "fixtures": sha(json.dumps(CASES).encode()),
            "tests": sha((ROOT / "tests/test_program_identity_consumer.py").read_bytes())}


def source(root):
    root = inside(root)
    require(Path(command("git", "rev-parse", "--show-toplevel", cwd=root)).resolve() == root,
            "Source must own its checkout")
    status = command("git", "status", "--porcelain=v1", "--untracked-files=all", cwd=root)
    require(not status, f"Dirty source: {status}")
    paths = command("git", "ls-files", cwd=root).splitlines()
    return {"root": str(root), "commit": command("git", "rev-parse", "HEAD", cwd=root),
            "tree": command("git", "rev-parse", "HEAD^{tree}", cwd=root), "status": status,
            "files": {p: sha((root / p).read_bytes()) for p in paths if (root / p).is_file()}}


def environment(directory):
    require(Path(sys.executable).absolute().is_relative_to(ROOT), "Use a worktree-local interpreter")
    inside(sys.prefix)
    require(not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME"), "Injected import path")
    sys.dont_write_bytecode = True
    for name, relative in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"),
                           ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"),
                           ("CUDA_CACHE_PATH", "cuda"), ("TORCH_HOME", "torch"),
                           ("UV_CACHE_DIR", "uv"), ("CARGO_HOME", "cargo"),
                           ("CARGO_TARGET_DIR", "target")):
        path = inside(directory / relative)
        path.mkdir()
        os.environ[name] = str(path)
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONNOUSERSITE="1",
                      PYTHONDONTWRITEBYTECODE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")


def wheel_files(wheel):
    with zipfile.ZipFile(wheel) as archive:
        return {name: archive.read(name) for name in archive.namelist()
                if name.startswith("torch_rs/") and name.endswith((".py", ".so"))}


def build(args, record, directory):
    environment(directory)
    root = inside(args.source_root)
    before = source(root)
    require(before["commit"] == args.commit, "Wrong source commit")
    wheels = directory / "wheels"
    wheels.mkdir()
    invocation = [str(inside(Path(sys.prefix) / "bin/maturin")), "build", "--release", "--locked", "--out", str(wheels)]
    record.update(source=before, buildCommand=invocation, buildEnvironment={key: value for key, value in os.environ.items()
                                    if key in ("CARGO_HOME", "CARGO_TARGET_DIR", "CUDA_VISIBLE_DEVICES",
                                               "LD_LIBRARY_PATH", "PATH", "NVRTC_LIBRARY_PATH", "CUDA_HOME")},
                  rustc=command("rustc", "-Vv"), nvcc=command("nvcc", "--version"))
    os.environ.update(PYO3_PYTHON=sys.executable, VIRTUAL_ENV=sys.prefix, CARGO_BUILD_JOBS="8")
    with (directory / "build.log").open("x") as stream:
        subprocess.run(invocation, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
    found = list(wheels.glob("*.whl"))
    require(len(found) == 1, "Expected one release wheel")
    wheel = found[0]
    files = wheel_files(wheel)
    require(any(name.endswith(".so") for name in files), "Missing extension")
    for name, data in files.items():
        if name.endswith(".py"):
            require(data == (root / "python" / name).read_bytes(), f"Wheel/source mismatch: {name}")
    require(source(root) == before, "Build changed source")
    record.update(wheel=str(wheel), wheelSha256=sha(wheel.read_bytes()),
                  files={name: sha(data) for name, data in files.items()},
                  buildLogSha256=sha((directory / "build.log").read_bytes()))


def installed(build_record):
    require(build_record["passed"], "Failed build")
    root = inside(build_record["source"]["root"])
    require(source(root) == build_record["source"], "Source identity changed")
    wheel = inside(build_record["wheel"])
    require(sha(wheel.read_bytes()) == build_record["wheelSha256"], "Wheel changed")
    spec = importlib.util.find_spec("torch_rs")
    package = inside(Path(spec.origin).parent, inside(sys.prefix))
    files = wheel_files(wheel)
    for name, data in files.items():
        require((package.parent / name).read_bytes() == data, f"Installed/wheel mismatch: {name}")
        if name.endswith(".py"):
            require((root / "python" / name).read_bytes() == data, f"Wheel/source mismatch: {name}")
    require({name: sha(data) for name, data in files.items()} == build_record["files"], "Wheel inventory changed")
    return {"package": str(package), "wheelSha256": sha(wheel.read_bytes()), "files": build_record["files"]}


def check(args, record, directory):
    import unittest
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_program_identity_consumer.py")
    with (directory / "controls.log").open("x") as stream:
        outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    record.update(testsRun=outcome.testsRun, logSha256=sha((directory / "controls.log").read_bytes()))
    require(outcome.testsRun >= 10 and outcome.wasSuccessful(), "Hardware-free controls failed")


def freeze(args, record, directory):
    candidate = source(ROOT)
    require(candidate["commit"] != BASELINE, "Candidate must be a committed implementation")
    record.update(source=candidate, controls=read(inside(args.controls)))
    require(record["controls"]["passed"] and record["controls"]["commandKind"] == "check"
            and record["controls"]["bindings"] == bindings(),
            "Hardware-free controls must pass against these bytes")


def inventory():
    return command("nvidia-smi", "--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,pci.bus_id",
                   "--format=csv,noheader")


def gpu_identity():
    snapshot = inventory()
    row = next(line for line in snapshot.splitlines() if line.split(",")[0].strip() == "0")
    require(row.split(",")[1].strip() == GPU and "H100" in row, "GPU0 identity changed")
    driver = ctypes.CDLL("libcuda.so.1")
    device, count, identity = ctypes.c_int(), ctypes.c_int(), (ctypes.c_ubyte * 16)()
    require(driver.cuInit(0) == 0 and driver.cuDeviceGetCount(ctypes.byref(count)) == 0
            and count.value == 1 and driver.cuDeviceGet(ctypes.byref(device), 0) == 0,
            "CUDA driver requires exactly one visible device")
    require(driver.cuDeviceGetUuid_v2(ctypes.byref(identity), device) == 0, "Driver UUID query")
    require("GPU-" + str(uuid.UUID(bytes=bytes(identity))) == GPU, "Driver UUID mismatch")
    return snapshot


def library_file(path):
    path = inside(path)
    require(path.is_file() and path.stat().st_size > 0, "Pinned library must be a nonempty worktree-local file")
    return path


def library_symbol(library, symbol, expected):
    """Bind the actual ELF provider of a resolved symbol, not merely dlopen input."""
    class DlInfo(ctypes.Structure):
        _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                    ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]
    info = DlInfo()
    process = ctypes.CDLL(None)
    process.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    process.dladdr.restype = ctypes.c_int
    require(process.dladdr(ctypes.cast(getattr(library, symbol), ctypes.c_void_p), ctypes.byref(info)) != 0
            and info.filename, "Cannot identify actual loaded library symbol")
    actual = library_file(os.fsdecode(info.filename))
    require(actual == expected, "Resolved symbol uses a different library than the requested pin")
    return {"path": str(actual), "sha256": sha(actual.read_bytes())}


class CompilerPin:
    def __init__(self, path):
        path = library_file(path)
        self.library = ctypes.CDLL(str(path))  # Kept alive through the entire worker.
        self.metadata = library_symbol(self.library, "nvrtcVersion", path)
        major, minor = ctypes.c_int(), ctypes.c_int()
        require(self.library.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) == 0, "NVRTC version query")
        self.metadata["version"] = [major.value, minor.value]


def pci_identity(value):
    # NVML prints an eight-digit domain; CUDA uses four digits.
    return tuple(int(part, 16) for part in value.replace(".", ":").split(":"))


def physical_devices(snapshot):
    devices = []
    for line in snapshot.splitlines():
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 7, "Incomplete physical GPU inventory")
        devices.append({"index": int(fields[0]), "uuid": fields[1], "pci.bus_id": fields[6]})
    return devices


def bind_runtime_identity(observed, devices):
    require(observed["visible_count"] == 1 and observed["logical_index"] == 0,
            "Runtime identity requires exactly one visible device at logical cuda:0")
    matches = [device for device in devices
               if pci_identity(device["pci.bus_id"]) == pci_identity(observed["pci.bus_id"])]
    require(len(matches) == 1, "Runtime PCI identity absent or ambiguous in physical inventory")
    matched = matches[0]
    require(matched["index"] == 0 and matched["uuid"] == GPU, "Runtime PCI identity does not select expected GPU0 UUID")
    return {**observed, "uuid": matched["uuid"], "physical_index": matched["index"],
            "uuid_source": "physical inventory matched by runtime PCI bus ID"}


class Synchronizer:
    def __init__(self, path):
        path = library_file(path)
        self.runtime = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        actual = library_symbol(self.runtime, "cudaDeviceSynchronize", path)
        version = ctypes.c_int()
        require(self.runtime.cudaRuntimeGetVersion(ctypes.byref(version)) == 0, "Runtime version query")
        require(self.runtime.cudaSetDevice(0) == 0, "Runtime device selection")
        self.metadata = {**actual, "version": version.value,
                         "api": "cudaDeviceSynchronize", "logicalDevice": 0}
        count, device = ctypes.c_int(), ctypes.c_int()
        bus = ctypes.create_string_buffer(64)
        require(self.runtime.cudaGetDeviceCount(ctypes.byref(count)) == 0, "Runtime device count query")
        require(self.runtime.cudaGetDevice(ctypes.byref(device)) == 0, "Runtime logical device query")
        self.runtime.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.runtime.cudaDeviceGetPCIBusId.restype = ctypes.c_int
        require(self.runtime.cudaDeviceGetPCIBusId(bus, len(bus), device.value) == 0, "Runtime PCI bus query")
        devices = physical_devices(inventory())
        observed = {"visible_count": count.value, "logical_index": device.value, "pci.bus_id": bus.value.decode()}
        self.metadata["runtimeIdentity"] = bind_runtime_identity(observed, devices)
        self.metadata["physicalInventory"] = devices
        self()

    def __call__(self):
        require(self.runtime.cudaDeviceSynchronize() == 0, "CUDA completion failure")


def libraries():
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if any(name in line for name in ("libcuda.so", "libcudart.so", "libnvrtc", "libnvJitLink"))}
    result = []
    for path in sorted(paths):
        row = {"path": path, "sha256": sha(Path(path).read_bytes())}
        library = ctypes.CDLL(path)
        major, minor = ctypes.c_int(), ctypes.c_int()
        if "libnvrtc.so" in path:
            require(library.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) == 0, "NVRTC version query")
            row["version"] = [major.value, minor.value]
        elif "libcudart.so" in path:
            require(library.cudaRuntimeGetVersion(ctypes.byref(major)) == 0, "Runtime version query")
            row["version"] = major.value
        elif "libcuda.so" in path:
            require(library.cuDriverGetVersion(ctypes.byref(major)) == 0, "Driver version query")
            row["version"] = major.value
        result.append(row)
    return result


def factory(framework, text, name):
    scope = {}
    exec(compile(text, f"<program-identity:{name}>", "exec"), scope)
    start = time.perf_counter_ns()
    compiled = framework.compile(scope["f"])
    return compiled, time.perf_counter_ns() - start


def values(shape, phase, y=False):
    return [(((i + 3 * phase + 5) % 31) - 15) / 64 if y
            else (((i + phase) % 29) - 14) / 64 for i in range(math.prod(shape))]


def inputs(framework, name, size, phase, width=7):
    shape = ((37 if size == 257 else 9363), width) if name == "broadcast" else (size,)
    def tensor(dims, y=False):
        return framework.tensor(values(dims, phase, y), dtype=framework.float32).reshape(dims).to("cuda:0")
    x = tensor(shape)
    if name in ("affine", "negation"):
        return (x,)
    y = tensor((width,) if name == "broadcast" else shape, True)
    return ({"pair": (x, y)},) if name == "structured" else (x, y)


def leaves(value):
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in leaves(item)]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in leaves(item)]
    return [value]


def encode(value, input_tensors):
    if isinstance(value, (dict, tuple, list)):
        if isinstance(value, dict):
            return {"kind": "dict", "items": [[key, encode(item, input_tensors)] for key, item in value.items()]}
        return {"kind": type(value).__name__, "items": [encode(item, input_tensors) for item in value]}
    def bits(items):
        return [bits(item) for item in items] if isinstance(items, list) else float(items).hex()
    pointer = value.data_ptr()
    return {"kind": "tensor", "shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device), "requiresGrad": value.requires_grad,
            "alias": [i for i, tensor in enumerate(input_tensors) if tensor.data_ptr() == pointer],
            "values": bits(value.cpu().tolist())}


def output_check(outputs, arguments):
    all_inputs = [tensor for args in arguments for tensor in leaves(args)]
    input_pointers = {tensor.data_ptr() for tensor in all_inputs}
    computed = []
    for output, args in zip(outputs, arguments):
        current = {tensor.data_ptr() for tensor in leaves(args)}
        for tensor in leaves(output):
            pointer = tensor.data_ptr()
            require(str(tensor.dtype) == "torch.float32" and str(tensor.device) == "cuda:0"
                    and not tensor.requires_grad, "Output metadata mismatch")
            if pointer in current:
                continue
            require(pointer not in input_pointers and pointer not in computed, "Computed outputs share storage")
            computed.append(pointer)
    return {"retainedOutputs": len(outputs), "computedPointers": computed, "inputPointers": sorted(input_pointers)}


def timed_call(compiled, args, sync):
    sync()
    start = time.perf_counter_ns()
    result = compiled(*args)
    sync()
    return result, time.perf_counter_ns() - start


def summary(samples):
    require(len(samples) == 17 and all(type(x) is int and x > 0 for x in samples), "Expected 17 positive raw times")
    ordered = sorted(samples)
    return {"median": ordered[8], "q1": ordered[4], "q3": ordered[12],
            "mad": statistics.median(abs(x - ordered[8]) for x in samples), "min": ordered[0], "max": ordered[16]}


def matrix(framework, sync, record):
    record["cells"] = []
    for name, text in CASES:
        compiled, factory_ns = factory(framework, text, name)
        for size in (257, 65537):
            row = {"family": name, "size": size, "factoryNs": factory_ns, "samples": [], "warmups": [], "passed": False}
            record["cells"].append(row)
            arguments = [inputs(framework, name, size, phase) for phase in (0, 1, 2, 3, 4, 5, *range(8, 25))]
            before = [encode(args, leaves(args)) for args in arguments]
            row["inputs"] = before
            result, row["firstNs"] = timed_call(compiled, arguments[0], sync)
            row["first"] = encode(result, leaves(arguments[0]))
            del result
            row["firstOwnerReleasedBeforeWarmup"] = True
            for phase in range(1, 6):
                result = compiled(*arguments[phase])
                sync()
                del result
                row["warmups"].append({"phase": phase, "completed": True, "ownerReleased": True})
            results = [None] * 17
            for index in range(17):
                result, elapsed = timed_call(compiled, arguments[index + 6], sync)
                results[index] = result
                row["samples"].append(elapsed)
            del result
            row["ownership"] = output_check(results, arguments[6:])
            row["outputs"] = [encode(result, leaves(args)) for result, args in zip(results, arguments[6:])]
            require(before == [encode(args, leaves(args)) for args in arguments], "Input mutation")
            if record["implementation"] == "native":
                row["compilerExecutors"] = [
                    {"options": list(executor.options), "nvrtcVersion": list(executor.nvrtc_version)}
                    for executor in compiled._torch_rs_pointwise_cache.executors.values()]
            row.update(statistics=summary(row["samples"]), passed=True)
            del results, arguments


def churn(framework, sync, record):
    compiled, factory_ns = factory(framework, "def f(x):\n    return -x\n", "negation")
    record["churn"] = {"factoryNs": factory_ns, "calls": []}
    for index, size in enumerate(LENGTHS):
        args = inputs(framework, "negation", size, index)
        result, elapsed = timed_call(compiled, args, sync)
        record["churn"]["calls"].append({"size": size, "elapsedNs": elapsed,
                                        "input": encode(args, leaves(args)), "output": encode(result, leaves(args))})
        del result, args


def selected(prepared):
    return {"identity": bytes(prepared.executable_identity).hex(), "kind": prepared.kind,
            "instructions": prepared.instruction_count, "registers": prepared.register_count,
            "source": prepared.source, "sourceSha256": sha(prepared.source.encode()),
            "ptx": prepared.ptx, "ptxSha256": sha(prepared.ptx.encode()),
            "nvrtcVersion": list(prepared.nvrtc_version), "options": list(prepared.options),
            "device": prepared.device, "context": prepared.context,
            "retainedBytes": prepared.retained_bytes}


def controls(framework, sync, record):
    """Companion invocations; never a claim about any timed invocation."""
    control = record["controls"] = {"timingData": False, "matrix": [], "negation": [], "broadcast": [],
                                     "unobservableCounters": ["NVRTC compile", "module load", "Program::build", "instruction upload"]}
    for name, text in CASES:
        compiled, _ = factory(framework, text, name)
        for size in (257, 65537):
            # Recreate ordinary phases and output-owner lifetimes, without timing.
            arguments = [inputs(framework, name, size, phase) for phase in (0, 1, 2, 3, 4, 5, *range(8, 25))]
            before_inputs = [encode(args, leaves(args)) for args in arguments]
            result = compiled(*arguments[0])
            sync()
            encode(result, leaves(arguments[0]))
            del result
            for args in arguments[1:6]:
                result = compiled(*args)
                sync()
                del result
            retained_results = [None] * 17
            for i, args in enumerate(arguments[6:]):
                retained_results[i] = compiled(*args)
                sync()
            result, prepared = compiled._torch_rs_pointwise_receipt(*arguments[-1])
            sync()
            ownership = output_check([*retained_results, result], [*arguments[6:], arguments[-1]])
            require(before_inputs == [encode(args, leaves(args)) for args in arguments], "Control input mutation")
            observation = selected(prepared)
            if name == "over_cap":
                require(observation["instructions"] > 256 and observation["kind"] == "vm",
                        "Fixed range(320) over-cap fixture prerequisite failed; stop, do not adjust fixture")
            else:
                require(observation["kind"] == "direct", "Eligible fixture did not select direct")
            control["matrix"].append({"family": name, "size": size, "selected": observation,
                                      "output": encode(result, leaves(arguments[-1])), "companionInvocation": True,
                                      "ownership": ownership})
            del result, args, prepared, retained_results, arguments
    compiled, _ = factory(framework, "def f(x):\n    return -x\n", "negation-control")
    pointwise = importlib.import_module("torch_rs._compile_pointwise")
    native = pointwise._native
    original_plan, original_account = native._pointwise_host_plan, pointwise._prepared_entry_bytes
    counts = {"hostPlan": 0, "accounting": 0}
    def counted_plan(*args, **kwargs):
        counts["hostPlan"] += 1
        return original_plan(*args, **kwargs)
    def counted_account(*args, **kwargs):
        counts["accounting"] += 1
        return original_account(*args, **kwargs)
    owners, executors, outputs, arguments, keys, original_inputs = [], [], [], [], [], []
    native._pointwise_host_plan = counted_plan
    pointwise._prepared_entry_bytes = counted_account
    try:
        for index, size in enumerate(LENGTHS):
            args = inputs(framework, "negation", size, index)
            cache = compiled._torch_rs_pointwise_cache
            original_inputs.append(encode(args, leaves(args)))
            before_preparations = {id(entry[0]) for entry in cache.prepared.values()}
            before_counts = dict(counts)
            result, prepared = compiled._torch_rs_pointwise_receipt(*args)
            sync()
            matches = [(key, entry) for key, entry in cache.prepared.items() if entry[0] is prepared]
            require(len(matches) == 1, "Receipt does not identify one retained preparation")
            key, entry = matches[0]
            executor = entry[3]
            require(cache.executors.get(entry[2]) is executor and prepared.belongs_to(executor),
                    "Preparation/executor owner association is stale")
            hit = id(prepared) in before_preparations
            delta = {name: counts[name] - before_counts[name] for name in counts}
            require(delta == ({"hostPlan": 0, "accounting": 0} if hit else {"hostPlan": 1, "accounting": 1}),
                    "Preparation hit rebuilt or recounted host data")
            keys.append(key)
            owners.append(prepared)
            executors.append(executor)
            outputs.append(result)
            arguments.append(args)
            control["negation"].append({"size": size, "selected": selected(prepared),
                                        "preparationObject": id(prepared), "executorObject": id(executor),
                                        "preparationKey": repr(key), "hit": hit, "counts": delta})
    finally:
        native._pointwise_host_plan = original_plan
        pointwise._prepared_entry_bytes = original_account
    identities = {row["selected"]["identity"] for row in control["negation"]}
    require(len(identities) == 1 and all(executor is executors[0] for executor in executors),
            "Equivalent negation failed to reuse the retained executable owner")
    require(owners[1] is owners[2] and owners[1] is not owners[-1] and keys[1] == keys[2] == keys[-1],
            "Stable-hint hit/eviction history failed")
    require(original_inputs == [encode(args, leaves(args)) for args in arguments], "Churn control input mutation")
    control["ownershipBeforeReset"] = output_check(outputs, arguments)
    before = [encode(output, leaves(args)) for output, args in zip(outputs, arguments)]
    framework.compiler.reset()
    require(before == [encode(output, leaves(args)) for output, args in zip(outputs, arguments)], "Reset invalidated retained output")
    result, after = compiled._torch_rs_pointwise_receipt(*arguments[-1])
    sync()
    outputs.append(result)
    arguments.append(arguments[-1])
    control["ownershipAfterReset"] = output_check(outputs, arguments)
    control["retainedHandleAfterReset"] = selected(owners[-1])
    control["newPreparationAfterReset"] = selected(after)
    compiled, _ = factory(framework, dict(CASES)["broadcast"], "broadcast-control")
    retained, broadcast_executors = [], []
    for width in WIDTHS:
        # Separate control has exactly five rows, independent of matrix sizes.
        x = framework.tensor(values((5, width), 0), dtype=framework.float32).reshape((5, width)).to("cuda:0")
        y = framework.tensor(values((width,), 0, True), dtype=framework.float32).to("cuda:0")
        result, prepared = compiled._torch_rs_pointwise_receipt(x, y)
        sync()
        retained.append(prepared)
        cache = compiled._torch_rs_pointwise_cache
        entries = [entry for entry in cache.prepared.values() if entry[0] is prepared]
        require(len(entries) == 1 and prepared.belongs_to(entries[0][3]), "Broadcast receipt owner mismatch")
        broadcast_executors.append(entries[0][3])
        require(all(cache.executors.get(entry[2]) is entry[3] and entry[0].belongs_to(entry[3])
                    for entry in cache.prepared.values()), "Stale broadcast preparation after pruning")
        control["broadcast"].append({"width": width, "selected": selected(prepared),
                                     "executorObject": id(entries[0][3]), "retainedExecutors": len(cache.executors),
                                     "retainedPreparations": len(cache.prepared)})
        del result
    require(len({row["selected"]["identity"] for row in control["broadcast"][:9]}) == 9,
            "Different broadcast addresses lost executable identity")
    require(control["broadcast"][0]["selected"]["identity"] == control["broadcast"][-1]["selected"]["identity"],
            "Repeated address formula changed exact identity")
    require(broadcast_executors[0] is not broadcast_executors[-1], "Evicted executable unexpectedly retained")


def leg(args, record, directory):
    environment(directory)
    runtime_path, compiler_path = library_file(args.runtime), library_file(args.nvrtc)
    os.environ["TORCH_RS_CUDART"] = str(runtime_path)
    os.environ["TORCH_RS_NVRTC"] = str(compiler_path)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Reserve only GPU0 before process startup")
    frozen = read(inside(args.freeze))
    require(frozen["passed"] and frozen["bindings"] == bindings(), "Invalid clean freeze")
    require(source(ROOT) == frozen["source"], "Consumer source changed after freeze")
    build_record = read(inside(args.build_record))
    label, implementation = ("C", "native") if args.command_kind == "preflight" else ORDER[args.index]
    require((build_record["source"]["commit"] == BASELINE) == (label == "B"), "Wrong paired build")
    if label == "C":
        require(build_record["source"]["commit"] == frozen["source"]["commit"], "Candidate build is not frozen commit")
    if args.command_kind == "leg":
        prerequisite = read(inside(args.preflight))
        require(prerequisite["passed"] and prerequisite["commandKind"] == "preflight"
                and prerequisite["bindings"] == bindings()
                and prerequisite["source"]["commit"] == frozen["source"]["commit"]
                and prerequisite["freezeSha256"] == sha(inside(args.freeze).read_bytes()),
                "Clean candidate fixture/control preflight must pass before ordinary timing")
        record["preflightSha256"] = sha(inside(args.preflight).read_bytes())
    record.update(index=getattr(args, "index", None), label=label, implementation=implementation, source=build_record["source"],
                  freezeSha256=sha(inside(args.freeze).read_bytes()), buildRecordSha256=sha(inside(args.build_record).read_bytes()),
                  installed=installed(build_record), gpu=GPU, gpuBefore=gpu_identity())
    compiler = CompilerPin(compiler_path) if implementation == "native" else None
    sync = Synchronizer(runtime_path)
    record["synchronization"] = sync.metadata
    if compiler is not None:
        record["compiler"] = compiler.metadata
    framework = importlib.import_module("torch_rs" if implementation == "native" else "torch")
    record.update(frameworkPath=str(inside(framework.__file__, inside(sys.prefix))),
                  frameworkSha256=sha(Path(framework.__file__).read_bytes()), frameworkVersion=framework.__version__)
    if implementation == "reference":
        require(framework.__version__ == "2.13.0+cu130", "Wrong reference version")
        properties = framework.cuda.get_device_properties(0)
        reference_uuid = "GPU-" + str(properties.uuid).removeprefix("GPU-")
        require(reference_uuid == GPU and framework.cuda.current_device() == 0, "Reference runtime UUID/device mismatch")
        record["referenceUuid"] = reference_uuid
        extension = inside(framework._C.__file__, inside(sys.prefix))
        record["referenceExtension"] = {"path": str(extension), "sha256": sha(extension.read_bytes())}
    framework.set_num_threads(1)
    require(framework.cuda.is_available() and framework.cuda.device_count() == 1, "One CUDA device required")
    record["environment"] = {key: os.environ.get(key) for key in (
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TMPDIR",
        "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_HOME",
        "LD_LIBRARY_PATH", "CUDA_HOME", "TORCH_RS_NVRTC", "TORCH_RS_CUDART")}
    record["librariesBefore"] = libraries()
    record["libraries"] = record["librariesBefore"]
    validate_libraries(record)
    if args.command_kind == "leg":
        require(prerequisite["synchronization"] == sync.metadata, "Preflight used a different runtime pin")
        require(prerequisite["compiler"]["path"] == str(compiler_path)
                and prerequisite["compiler"]["sha256"] == sha(compiler_path.read_bytes()),
                "Preflight used a different compiler pin")
    try:
        if args.command_kind == "preflight":
            controls(framework, sync, record)
        else:
            matrix(framework, sync, record)
            churn(framework, sync, record)
            if label == "C" and implementation == "native":
                controls(framework, sync, record)
        require(implementation != "native" or "torch" not in sys.modules, "Native worker imported reference")
        require(source(ROOT) == frozen["source"], "Source changed during capture")
    finally:
        record.update(gpuAfter=gpu_identity(), libraries=libraries())
        validate_libraries(record)


def validate_libraries(record):
    runtime = record["synchronization"]
    identity = runtime["runtimeIdentity"]
    require(identity == bind_runtime_identity(identity, runtime["physicalInventory"]),
            "Runtime UUID/PCI provenance does not match physical inventory")
    loaded = record["libraries"]
    runtimes = [row for row in loaded if "libcudart.so" in Path(row["path"]).name]
    require(len(runtimes) == 1 and all(runtimes[0][key] == runtime[key] for key in ("path", "sha256", "version")),
            "Actual loaded CUDA runtime differs from the single pinned barrier/runtime")
    require(record["environment"]["TORCH_RS_CUDART"] == runtime["path"], "Native runtime pin differs from barrier")
    if record["implementation"] == "native":
        compiler = record["compiler"]
        compilers = [row for row in loaded if "libnvrtc.so" in Path(row["path"]).name]
        require(len(compilers) == 1 and compilers[0] == compiler, "Actual loaded NVRTC differs from pinned compiler")
        require(record["environment"]["TORCH_RS_NVRTC"] == compiler["path"], "Native NVRTC pin differs")
        for cell in record.get("cells", []):
            require(cell["compilerExecutors"], "Missing actual native compiler/options observations")
            for executor in cell["compilerExecutors"]:
                require(executor["options"] == list(COMPILER_OPTIONS) and executor["nvrtcVersion"] == compiler["version"],
                        "Native executable used different NVRTC version/options")
        for row in record.get("controls", {}).get("matrix", []):
            observation = row["selected"]
            require(observation["options"] == list(COMPILER_OPTIONS)
                    and observation["nvrtcVersion"] == compiler["version"], "Selected invocation compiler/options differ")


def preflight(args, record, directory):
    leg(args, record, directory)


def compare(left, right, exact=False):
    """Compare structure/metadata exactly; finite reference-relative values only."""
    require(type(left) is type(right), "Container type mismatch")
    if isinstance(left, dict):
        require(left.keys() == right.keys(), "Record keys mismatch")
        for key in left:
            if key == "values":
                compare_values(left[key], right[key], exact)
            else:
                compare(left[key], right[key], exact)
    elif isinstance(left, list):
        require(len(left) == len(right), "Container length mismatch")
        for a, b in zip(left, right):
            compare(a, b, exact)
    else:
        require(left == right, "Metadata mismatch")


def compare_values(left, right, exact):
    if isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), "Value shape mismatch")
        for a, b in zip(left, right):
            compare_values(a, b, exact)
    else:
        require(isinstance(left, str) and isinstance(right, str), "Expected hexadecimal float records")
        a, b = float.fromhex(left), float.fromhex(right)
        require(math.isfinite(a) and math.isfinite(b), "Nonfinite output")
        require(left == right if exact else abs(a - b) <= 1e-4 + 1e-4 * abs(b),
                "Native bits differ" if exact else "Reference tolerance failure")


def bit_differences(left, right):
    if isinstance(left, dict):
        return sum(bit_differences(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return sum(bit_differences(a, b) for a, b in zip(left, right))
    return int(isinstance(left, str) and left != right)


def verify_records(reports, expected_bindings):
    """Pure offline verification; retain all per-leg/pair failures without hiding later legs."""
    result = {"passed": False, "failures": [], "ratios": [], "qualificationOrScoreProduced": False}
    def check(where, function):
        try:
            function()
        except Exception as error:
            result["failures"].append({"location": where, "error": str(error)})
    require(len(reports) == 8, "All eight ordered legs are mandatory")
    expected_cells = [(name, size) for name, _ in CASES for size in (257, 65537)]
    for index, report in enumerate(reports):
        def validate_leg():
            require(report["passed"] and report["version"] == VERSION, "Failed or foreign leg")
            require(report["index"] == index and (report["label"], report["implementation"]) == ORDER[index], "Wrong order")
            require(report["bindings"] == expected_bindings, "Consumer/protocol binding differs")
            require(report["source"]["status"] == "", "Dirty source")
            paired_index = 0 if ORDER[index][0] == "B" else 2
            paired = reports[paired_index]
            require(report["source"] == paired["source"] and
                    report["buildRecordSha256"] == paired["buildRecordSha256"] and
                    report["installed"]["wheelSha256"] == paired["installed"]["wheelSha256"] and
                    report["installed"]["files"] == paired["installed"]["files"], "Paired source/wheel identities differ")
            require(bool(report["source"]["files"]) and bool(report["installed"]["files"]), "Missing source/import inventory")
            for digest in (report["source"]["commit"], report["source"]["tree"]):
                require(len(digest) == 40 and all(c in "0123456789abcdef" for c in digest), "Invalid commit/tree identity")
            for digest in (report["buildRecordSha256"], report["installed"]["wheelSha256"],
                           report["interpreterSha256"], report["frameworkSha256"]):
                require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Invalid provenance digest")
            validate_libraries(report)
            if ORDER[index][1] == "native":
                require(report["compiler"] == reports[0]["compiler"], "B/C native compiler bytes/path/version differ")
            if ORDER[index][1] == "reference":
                require(report["frameworkVersion"] == "2.13.0+cu130" and report["referenceUuid"] == GPU,
                        "Reference runtime identity differs")
                require(report["referenceExtension"]["sha256"] == reports[1]["referenceExtension"]["sha256"],
                        "Reference extension changed")
            require((report["source"]["commit"] == BASELINE) == (ORDER[index][0] == "B"), "Wrong build identity")
            require(report["freezeSha256"] == reports[0]["freezeSha256"], "Freeze changed between legs")
            require(report["preflightSha256"] == reports[0]["preflightSha256"], "Fixture prerequisite changed between legs")
            require(report["gpu"] == GPU and report["gpuBefore"] and report["gpuAfter"], "GPU evidence missing")
            sync = report["synchronization"]
            require(sync["api"] == "cudaDeviceSynchronize" and sync["logicalDevice"] == 0, "Wrong barrier")
            require(sync == reports[0]["synchronization"], "Runtime barrier identity differs")
            require([(cell["family"], cell["size"]) for cell in report["cells"]] == expected_cells, "Incomplete or reordered matrix")
            require([call["size"] for call in report["churn"]["calls"]] == list(LENGTHS), "Wrong churn history")
            for call in report["churn"]["calls"]:
                require(type(call["elapsedNs"]) is int and call["elapsedNs"] > 0, "Missing churn time")
            if index:
                require(reports[index - 1]["finishedAt"] <= report["startedAt"], "Overlapping/reordered workers")
            if ORDER[index] == ("C", "native"):
                control = report["controls"]
                require(control["timingData"] is False, "Controls relabeled as timing")
                require([(row["family"], row["size"]) for row in control["matrix"]] == expected_cells,
                        "Missing candidate selected-invocation controls")
                for row in control["matrix"]:
                    selected_record = row["selected"]
                    require(row["companionInvocation"] is True, "Receipt relabeled timed invocation")
                    require(sha(selected_record["source"].encode()) == selected_record["sourceSha256"] and
                            sha(selected_record["ptx"].encode()) == selected_record["ptxSha256"], "Source/PTX association changed")
                    require(bool(bytes.fromhex(selected_record["identity"])), "Missing executable identity")
                    if row["family"] == "over_cap":
                        require(selected_record["instructions"] > 256 and selected_record["kind"] == "vm", "Over-cap prerequisite failed")
                    else:
                        require(selected_record["kind"] == "direct" and selected_record["instructions"] <= 256 and
                                selected_record["registers"] <= 128 and len(selected_record["source"].encode()) <= 65536,
                                "Direct caps or selection incorrect")
                require([row["size"] for row in control["negation"]] == list(LENGTHS), "Control history incomplete")
                require(len({row["selected"]["identity"] for row in control["negation"]}) == 1, "Identical code differs")
                objects = [row["preparationObject"] for row in control["negation"]]
                require(objects[1] == objects[2] and objects[1] != objects[-1], "Missing retained hit/eviction")
                require(len({row["executorObject"] for row in control["negation"]}) == 1, "Executable owner changed")
                keys = [row["preparationKey"] for row in control["negation"]]
                require(keys[1] == keys[2] == keys[-1], "Revisit used a different preparation key")
                for i, row in enumerate(control["negation"]):
                    require(row["hit"] == (i == 2), "Unexpected preparation hit/miss")
                    require(row["counts"] == ({"hostPlan": 0, "accounting": 0} if i == 2 else
                                               {"hostPlan": 1, "accounting": 1}), "Construction/accounting counts differ")
                require([row["width"] for row in control["broadcast"]] == list(WIDTHS), "Address controls incomplete")
                identities = [row["selected"]["identity"] for row in control["broadcast"]]
                require(len(set(identities[:9])) == 9 and identities[0] == identities[-1], "Address identity failed")
                require(control["broadcast"][0]["executorObject"] != control["broadcast"][-1]["executorObject"],
                        "Evicted broadcast executable owner did not change")
                require(all(row["retainedExecutors"] <= 8 and row["retainedPreparations"] <= 8
                            for row in control["broadcast"]), "LRU exceeded frozen limit")
        check(f"leg[{index}]", validate_leg)
        for cell_index, cell in enumerate(report.get("cells", [])):
            def validate_cell():
                require(cell["passed"] and len(cell["outputs"]) == 17 and len(cell["inputs"]) == 23, "Incomplete cell")
                require(cell["statistics"] == summary(cell["samples"]), "Incorrect statistics")
                require(type(cell["firstNs"]) is int and cell["firstNs"] > 0, "Missing cold observation")
                require(type(cell["factoryNs"]) is int and cell["factoryNs"] > 0, "Missing factory observation")
                require(cell["firstOwnerReleasedBeforeWarmup"] is True, "Wrong first output lifetime")
                require(cell["warmups"] == [{"phase": phase, "completed": True, "ownerReleased": True}
                                           for phase in range(1, 6)], "Wrong warmup lifetime/history")
                ownership = cell["ownership"]
                require(ownership["retainedOutputs"] == 17, "Wrong retained output lifetime")
                computed = ownership["computedPointers"]
                require(len(set(computed)) == len(computed) and not set(computed).intersection(ownership["inputPointers"]),
                        "Computed output aliases retained storage")
                for output in (cell["first"], *cell["outputs"]):
                    compare(output, output, exact=True)  # Reject nonfinite even if both implementations agree.
            check(f"leg[{index}].cell[{cell_index}]", validate_cell)
    for numerator, denominator, exact, label in ((0, 2, True, "B/C-forward"), (7, 5, True, "B/C-reverse"),
                                                (1, 0, False, "reference/B-forward"), (3, 2, False, "reference/C-forward"),
                                                (4, 5, False, "reference/C-reverse"), (6, 7, False, "reference/B-reverse")):
        for cell_index, (name, size) in enumerate(expected_cells):
            def validate_pair():
                left, right = reports[numerator]["cells"][cell_index], reports[denominator]["cells"][cell_index]
                require(left["inputs"] == right["inputs"], "Different prebuilt inputs")
                # Reference is the right operand of the asymmetric tolerance.
                native, reference = (left, right) if exact else (right, left)
                compare(native["first"], reference["first"], exact)
                for a, b in zip(native["outputs"], reference["outputs"]):
                    compare(a, b, exact)
                result["ratios"].append({"comparison": label, "family": name, "size": size,
                                          "rawBitDifferences": bit_differences(native["first"], reference["first"])
                                          + sum(bit_differences(a, b) for a, b in zip(native["outputs"], reference["outputs"])),
                                          "medianRatio": summary(left["samples"])["median"] / summary(right["samples"])["median"],
                                          "firstRatio": left["firstNs"] / right["firstNs"]})
            check(f"{label}.{name}.{size}", validate_pair)
        def validate_churn():
            left = reports[numerator]["churn"]["calls"]
            right = reports[denominator]["churn"]["calls"]
            require(len(left) == len(right) == len(LENGTHS), "Incomplete churn")
            for a, b in zip(left, right):
                require(a["input"] == b["input"], "Churn inputs differ")
                compare(a["output"] if exact else b["output"], b["output"] if exact else a["output"], exact)
        check(label + ".churn", validate_churn)
    result["ledger"] = {"cells": 96, "samples": 1632, "warmups": 480, "firstCalls": 96}
    result["passed"] = not result["failures"]
    return result


def verify(args, record, directory):
    reports = [read(inside(path)) for path in args.reports]
    record["inventory"] = [{"path": str(inside(path)), "sha256": sha(inside(path).read_bytes())} for path in args.reports]
    record.update(verify_records(reports, bindings()))
    require(record["passed"], "Offline verification failed; see retained failures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command_kind", required=True)
    checker = commands.add_parser("check")
    builder = commands.add_parser("build")
    builder.add_argument("--source-root", required=True, type=Path)
    builder.add_argument("--commit", required=True)
    freezer = commands.add_parser("freeze")
    freezer.add_argument("--controls", required=True, type=Path)
    prerequisite = commands.add_parser("preflight")
    worker = commands.add_parser("leg")
    for native_worker in (prerequisite, worker):
        native_worker.add_argument("--build-record", required=True, type=Path)
        native_worker.add_argument("--freeze", required=True, type=Path)
        native_worker.add_argument("--runtime", required=True, type=Path)
        native_worker.add_argument("--nvrtc", required=True, type=Path)
    worker.add_argument("--index", required=True, type=int, choices=range(8))
    worker.add_argument("--preflight", required=True, type=Path)
    verifier = commands.add_parser("verify")
    verifier.add_argument("reports", nargs=8, type=Path)
    for child in (checker, builder, freezer, prerequisite, worker, verifier):
        child.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = inside(args.output)
    require(not output.parent.exists(), "Each attempt requires a fresh directory")
    output.parent.mkdir(parents=True)
    record = {"version": VERSION, "bindings": bindings(), "commandKind": args.command_kind,
              "startedAt": datetime.now(timezone.utc).isoformat(), "command": sys.orig_argv,
              "python": sys.version, "interpreter": sys.executable,
              "interpreterSha256": sha(Path(sys.executable).read_bytes()), "pid": os.getpid(), "passed": False,
              "qualificationOrScoreProduced": False}
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
