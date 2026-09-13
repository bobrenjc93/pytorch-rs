"""Unscored source-derived Python dispatch diagnostic; all native work is mocked."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import types


ROOT = "/data/users/bobren/a/pytorch-rs-burner"
FRONTEND = "python/torch_rs/_compile_pointwise.py"
STATE = "python/torch_rs/_compiler_state.py"
PUBLIC = "python/torch_rs/__init__.py"
SOURCES = {
    "main": {
        "commit": "a281503f3391bbd98a0ab6de2ff8f0c0a55a12d4",
        FRONTEND: "919adb3bc6e6518395a0eab05067f199cefe29e286b3c507b53409000e2da24a",
        STATE: "368627af1e2771d80206ec6a7cd16e103c3cbcf8718a8b22190db536d91b467f",
        PUBLIC: "d8841a4c0960808f2996188eb4269e0e7eeff8270f4f9beddcf0667598f79532",
    },
    "candidate": {
        "commit": "885264b5319d76165e6be8d6df405450b3830c47",
        FRONTEND: "6da5e0e7ce54fcb9b8443536e19e17e98258b706432214c2a55caee7e402fbe2",
        STATE: "6d4499841eebf52fbb762fcf91d21636c1f3a2e1cc17abdc5ecaa6bd60e028b3",
        PUBLIC: "d8841a4c0960808f2996188eb4269e0e7eeff8270f4f9beddcf0667598f79532",
    },
}
PROFILE_CALLS = 128
TIMING_CALLS = 1024
ORDERS = (("main", "candidate"), ("candidate", "main"),
          ("candidate", "main"), ("main", "candidate"))
CASES = (
    ("literal_unary", "def f(x):\n return x*1.25+0.75", "single"),
    ("tensor_arithmetic", "def f(x,y):\n return x*y+x", "pair"),
    ("broadcast_add", "def f(x,y):\n return x+y", "broadcast"),
    ("repeated_alias", "def f(x,y):\n return x+y", "alias"),
    ("shape_revisit", "def f(x,y):\n return x*y+y", "shapes"),
    ("promoted_capture", "def f(x):\n return x*scale+x", "promoted"),
    ("eight_boolean_entries", "def f(x):\n return x*a+x*b+x*c", "booleans"),
)


def source_bytes(commit, path):
    return subprocess.check_output(
        ["git", "show", commit + ":" + path], cwd=ROOT, timeout=30)


def sha(value):
    return hashlib.sha256(value).hexdigest()


class Boundary:
    """Synthetic Tensor/package/native boundary, identical for both source trees."""

    def __init__(self, label):
        self.calls = Counter()
        self.sentinel = object()

        class Tensor:
            def __init__(self, shape):
                stride, strides = 1, []
                for dimension in reversed(shape):
                    strides.insert(0, stride)
                    stride *= max(dimension, 1)
                self.metadata = (tuple(shape), tuple(strides), False, "torch.float32", "cuda:0", 0)

            def __add__(self, other):
                raise AssertionError("Original Python tensor arithmetic executed")

            __mul__ = __add__

        self.Tensor = Tensor
        self.root = types.ModuleType("_dispatch_profile_" + label)
        self.root.Tensor = Tensor
        self.root.overrides = types.SimpleNamespace(_get_current_function_mode=lambda: None)
        sys.modules[self.root.__name__] = self.root

    def metadata(self, tensor):
        assert type(tensor) is self.Tensor
        self.calls["metadata"] += 1
        return tensor.metadata

    def validate(self, tensors):
        assert len(tensors) in (1, 2)
        assert all(type(tensor) is self.Tensor for tensor in tensors)
        self.calls["validate"] += 1

    def compile(self, tensors, nodes, output):
        self.calls["compile"] += 1
        boundary = self

        class Executor:
            def run(self, actual, scalars):
                assert len(actual) == len(tensors)
                assert all(type(tensor) is boundary.Tensor for tensor in actual)
                assert type(scalars) is tuple
                boundary.calls["launch"] += 1
                return boundary.sentinel

        return Executor()


def load_frontend(label, sources):
    boundary = Boundary(label)
    state_tree = ast.parse(sources[STATE])
    cache_class = next(node for node in state_tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "NativeEagerCompileCache")
    state_namespace = {"threading": threading}
    exec(compile(ast.Module(body=[cache_class], type_ignores=[]), STATE, "exec"), state_namespace)
    # Direct construction intentionally omits the WeakSet registry/global-reset
    # lifecycle. This compares warm dispatch, not public reset/ownership behavior.
    public_tree = ast.parse(sources[PUBLIC])
    default_limit = next(ast.literal_eval(node.value) for node in public_tree.body
                         if isinstance(node, ast.Assign) and any(
                             isinstance(target, ast.Name) and target.id == "_COMPILE_DEFAULT_RECOMPILE_LIMIT"
                             for target in node.targets))
    assert default_limit == 8
    frontend = types.ModuleType(boundary.root.__name__ + "._compile_pointwise")
    frontend.__package__ = boundary.root.__name__
    frontend.__dict__.update(
        _native=types.SimpleNamespace(_compile_trace_tensor_metadata=boundary.metadata,
                                      _pointwise_validate_inputs=boundary.validate,
                                      _pointwise_compile=boundary.compile),
        _state=types.SimpleNamespace(new_native_eager_compile_cache=state_namespace["NativeEagerCompileCache"]))
    sys.modules[frontend.__name__] = frontend
    tree = ast.parse(sources[FRONTEND])
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level)]
    exec(compile(tree, SOURCES[label]["commit"] + ":" + FRONTEND, "exec"), frontend.__dict__)
    return frontend, boundary, default_limit


def prepare(frontend, boundary, limit, source, kind):
    namespace = {}
    exec(source, namespace)
    model = namespace["f"]
    compiled = frontend.implementation(model, limit)
    x, y = boundary.Tensor((31, 7)), boundary.Tensor((31, 7))
    if kind == "single":
        states = [((x,), {})]
    elif kind == "pair":
        states = [((x, y), {})]
    elif kind == "broadcast":
        states = [((x, boundary.Tensor((7,))), {})]
    elif kind == "alias":
        states = [((x, x), {})]
    elif kind == "shapes":
        states = [((x, y), {}), ((boundary.Tensor((47, 7)), boundary.Tensor((47, 7))), {})]
    elif kind == "promoted":
        states = [((x,), {"scale": value}) for value in (1.25, 2.5, 0.75)]
    else:
        assert kind == "booleans"
        states = [((x,), dict(zip(("a", "b", "c"), (bool(mask & (1 << bit)) for bit in range(3)))))
                  for mask in range(8)]

    def batch(count):
        for index in range(count):
            args, updates = states[index % len(states)]
            namespace.update(updates)
            assert compiled(*args) is boundary.sentinel

    batch(len(states))
    # Finish warm setup with a whole traversal; later runs may only hit these owners.
    batch(len(states))
    cache = compiled._torch_rs_pointwise_cache
    cardinality = (len(cache.graphs), len(getattr(cache, "executors", {})))
    assert 0 < cardinality[0] <= limit
    if kind == "booleans":
        assert cardinality[0] == 8
    return types.SimpleNamespace(batch=batch, model=model, cache=cache, cardinality=cardinality,
                                 tensorsPerCall=len(states[0][0]), states=len(states))


def boundary_delta(before, after, calls, tensor_count):
    expected = {"metadata": calls * tensor_count, "validate": calls, "launch": calls, "compile": 0}
    actual = {key: after[key] - before[key] for key in expected}
    assert actual == expected, (actual, expected)
    return actual


def main():
    assert sys.version_info[:2] == (3, 12), sys.version
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    assert sys.getprofile() is None and sys.gettrace() is None
    loaded, inputs = {}, {}
    for label, spec in SOURCES.items():
        data = {path: source_bytes(spec["commit"], path) for path in (FRONTEND, STATE, PUBLIC)}
        for path, value in data.items():
            assert sha(value) == spec[path], (label, path)
        inputs[label] = data
        loaded[label] = load_frontend(label, data)
    result = {"scope": "Mocked production Python frontend/cache only; not native validation, public-entrypoint parity or GPU latency",
              "omittedLifecycle": "Direct cache construction bypasses the WeakSet registry and global reset lifecycle",
              "cpuTimingIncludes": "Harness iteration, namespace updates, sentinel assertion and mocked native bookkeeping; not GPU latency",
              "python": sys.version, "sources": SOURCES, "profileCalls": PROFILE_CALLS,
              "timingCallsPerBatch": TIMING_CALLS, "timingOrders": ORDERS, "cases": [],
              "qualificationOrScoreProduced": False, "realFrameworkOrGpuUsed": False}
    for name, source, kind in CASES:
        setups, row = {}, {"name": name, "program": source, "sourceResults": {}}
        for label, (frontend, boundary, limit) in loaded.items():
            setup = prepare(frontend, boundary, limit, source, kind)
            setups[label] = setup
            counts = Counter()

            def profile(frame, event, arg):
                if event != "call":
                    return
                if frame.f_code is setup.model.__code__:
                    counts["original_body"] += 1
                if frame.f_globals.get("__name__") == frontend.__name__:
                    owner = frame.f_locals.get("self")
                    label = type(owner).__name__ + "." if owner is not None else ""
                    counts[label + frame.f_code.co_name] += 1

            before = boundary.calls.copy()
            sys.setprofile(profile)
            try:
                setup.batch(PROFILE_CALLS)
            finally:
                sys.setprofile(None)
            for forbidden in ("original_body", "analyze", "lower"):
                assert counts[forbidden] == 0, (name, label, forbidden)
            assert counts["compiled"] == PROFILE_CALLS, (name, label, counts)
            if label == "candidate":
                assert counts["Graph.__hash__"] > 0, (name, label, counts)
            delta = boundary_delta(before, boundary.calls, PROFILE_CALLS, setup.tensorsPerCall)
            assert setup.cardinality == (len(setup.cache.graphs), len(getattr(setup.cache, "executors", {})))
            row["sourceResults"][label] = {"cacheCardinality": setup.cardinality,
                "states": setup.states, "productionCallCounts": dict(sorted(counts.items())),
                "mockBoundaryCallCounts": delta, "unprofiledCpuBatchUsPerCall": []}
        for order in ORDERS:
            for label in order:
                _, boundary, _ = loaded[label]
                setup = setups[label]
                assert sys.getprofile() is None and sys.gettrace() is None
                before = boundary.calls.copy()
                start = time.perf_counter_ns()
                setup.batch(TIMING_CALLS)
                elapsed = time.perf_counter_ns() - start
                boundary_delta(before, boundary.calls, TIMING_CALLS, setup.tensorsPerCall)
                assert setup.cardinality == (len(setup.cache.graphs), len(getattr(setup.cache, "executors", {})))
                row["sourceResults"][label]["unprofiledCpuBatchUsPerCall"].append(elapsed / TIMING_CALLS / 1000)
        for observation in row["sourceResults"].values():
            observation["medianUnprofiledCpuUsPerCall"] = statistics.median(observation["unprofiledCpuBatchUsPerCall"])
        result["cases"].append(row)
    for label, data in inputs.items():
        for path, value in data.items():
            assert source_bytes(SOURCES[label]["commit"], path) == value
    assert not any(name == "torch" or name.startswith("torch.") or name == "torch_rs" or name.startswith("torch_rs.")
                   for name in sys.modules)
    result["scriptSha256"] = sha(Path(__file__).read_bytes())
    result["sourceHashesRechecked"] = True
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
