"""Read-only, stdlib-only reproduction of native NaN guard over-specialization.

Execute the production frontend AST, redirecting only its relative imports to a
mock native boundary and the actual cache class. No torch/torch_rs import, GPU,
build, numerical kernel execution, repository mutation, or source edit occurs.
The report distinguishes a guard/cache-domain failure from NaN output payloads.
"""

import argparse
import ast
import hashlib
import json
import math
import struct
import sys
import threading
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worktree",
        type=Path,
        default=Path("/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_14879886"),
    )
    worktree = parser.parse_args().worktree
    frontend_path = worktree / "python/torch_rs/_compile_pointwise.py"
    state_path = worktree / "python/torch_rs/_compiler_state.py"
    public_path = worktree / "python/torch_rs/__init__.py"
    frontend_bytes = frontend_path.read_bytes()
    state_bytes = state_path.read_bytes()
    public_bytes = public_path.read_bytes()

    state_tree = ast.parse(state_bytes, str(state_path))
    state_namespace = {"threading": threading}
    state_class = next(
        node for node in state_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeEagerCompileCache"
    )
    exec(
        compile(ast.Module(body=[state_class], type_ignores=[]), str(state_path), "exec"),
        state_namespace,
    )
    public_tree = ast.parse(public_bytes, str(public_path))
    default_limit = next(
        ast.literal_eval(node.value)
        for node in public_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_COMPILE_DEFAULT_RECOMPILE_LIMIT"
            for target in node.targets
        )
    )

    root_name = "_scalar_guard_readonly_review"
    root = types.ModuleType(root_name)

    class Tensor:
        pass

    root.Tensor = Tensor
    root.overrides = types.SimpleNamespace(_get_current_function_mode=lambda: None)
    sys.modules[root_name] = root
    calls = {"compile": 0, "launch": 0, "validate": 0}
    metadata = ((2,), (1,), "float32", False, "cuda:0", 0)

    def validate_inputs(inputs):
        assert len(inputs) == 1 and type(inputs[0]) is Tensor
        calls["validate"] += 1

    class Executor:
        def __init__(self, nodes, output):
            self.nodes, self.output = nodes, output

        def run(self, inputs, scalars):
            calls["launch"] += 1
            assert len(inputs) == 1 and scalars == ()
            return self.nodes, self.output

    def compile_native(inputs, nodes, output):
        calls["compile"] += 1
        return Executor(nodes, output)

    frontend = types.ModuleType(root_name + "._compile_pointwise")
    frontend.__package__ = root_name
    frontend.__dict__.update(
        _native=types.SimpleNamespace(
            _compile_trace_tensor_metadata=lambda arg: metadata,
            _pointwise_validate_inputs=validate_inputs,
            _pointwise_compile=compile_native,
        ),
        _state=types.SimpleNamespace(
            new_native_eager_compile_cache=state_namespace["NativeEagerCompileCache"]
        ),
    )
    sys.modules[frontend.__name__] = frontend
    frontend_tree = ast.parse(frontend_bytes, str(frontend_path))
    frontend_tree.body = [
        node for node in frontend_tree.body
        if not (isinstance(node, ast.ImportFrom) and node.level)
    ]
    exec(compile(frontend_tree, str(frontend_path), "exec"), frontend.__dict__)
    namespace = {}
    exec("def f(scale, x):\n return x*scale", namespace)
    compiled = frontend.implementation(namespace["f"], default_limit)
    owner = compiled._torch_rs_pointwise_cache
    x = Tensor()
    encodings = [0x7FF8000000000000 + i for i in range(1, 6)] + [
        0xFFF8000000000000 + i for i in range(1, 5)
    ]
    rows = []
    for index, bits in enumerate(encodings, 1):
        value = struct.unpack("=d", struct.pack("=Q", bits))[0]
        assert type(value) is float and math.isnan(value)
        row = {"distinct_value_call": index, "bits": f"0x{bits:016x}"}
        try:
            compiled(value, x)
            row.update(
                cold_graphs=len(owner.graphs),
                cold_executors=len(owner.executors),
                cold_compile_calls=calls["compile"],
            )
            compiled(value, x)
            row.update(
                warm_graphs=len(owner.graphs),
                warm_executors=len(owner.executors),
                warm_compile_calls=calls["compile"],
            )
        except Exception as exc:
            row.update(
                exception_type=type(exc).__name__,
                exception=str(exc),
                graphs_after_failure=len(owner.graphs),
                executors_after_failure=len(owner.executors),
                compile_calls_after_failure=calls["compile"],
            )
        rows.append(row)

    hashes_before = {
        str(frontend_path): hashlib.sha256(frontend_bytes).hexdigest(),
        str(state_path): hashlib.sha256(state_bytes).hexdigest(),
        str(public_path): hashlib.sha256(public_bytes).hexdigest(),
    }
    hashes_after = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (frontend_path, state_path, public_path)
    }
    print(json.dumps({
        "scope": "actual frontend AST with only relative imports redirected to an isolated mock native boundary; actual cache class; no framework imports/GPU execution",
        "concern": "false recompilation and domain rejection, not NaN output payload equality",
        "worktree": str(worktree),
        "default_recompile_limit_read_from_public_caller": default_limit,
        "fixed_metadata": metadata,
        "rows": rows,
        "native_boundary_call_counts": calls,
        "reproduced_ninth_nan_default_limit_failure": (
            rows[-1].get("exception_type") == "NotImplementedError"
            and rows[-1].get("exception")
            == "torch.compile(): native CUDA pointwise: hit recompile_limit=8"
        ),
        "source_sha256_before": hashes_before,
        "source_sha256_after": hashes_after,
        "torch_imported": "torch" in sys.modules,
        "torch_rs_imported": "torch_rs" in sys.modules,
    }, indent=2))


if __name__ == "__main__":
    main()
