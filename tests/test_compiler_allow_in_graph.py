import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch


COMPILER_EXPORTS = [
    "assume_constant_result",
    "reset",
    "allow_in_graph",
    "list_backends",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "set_enable_guard_collectives",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "keep_portable_guards_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "keep_tensor_guards_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
]


@torch.compiler.allow_in_graph
def _picklable_allowed_function(value, *, increment=1):
    return value + increment


class _CallableObject:
    def __init__(self):
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return value


class CompilerAllowInGraphTests(unittest.TestCase):
    def test_callable_inputs_return_exact_objects_and_preserve_eager_calls(self):
        calls = []

        def calculate(value: int, *, scale: int = 1) -> int:
            """Calculate eagerly."""
            calls.append((value, scale))
            return value * scale + len(calls)

        marker = object()
        calculate.custom_attribute = marker
        before_dict = dict(calculate.__dict__)

        returned = torch.compiler.allow_in_graph(calculate)

        self.assertIs(returned, calculate)
        self.assertEqual(calculate.__dict__, before_dict)
        self.assertIs(calculate.custom_attribute, marker)
        self.assertFalse(hasattr(calculate, "_dynamo_marked_constant"))
        self.assertFalse(hasattr(calculate, "_torchdynamo_disable"))
        self.assertFalse(hasattr(calculate, "__wrapped__"))
        self.assertEqual(calculate(3, scale=2), 7)
        self.assertEqual(calculate(3, scale=2), 8)
        self.assertEqual(calls, [(3, 2), (3, 2)])
        self.assertEqual(
            str(inspect.signature(calculate)),
            "(value: int, *, scale: int = 1) -> int",
        )
        self.assertEqual(calculate.__name__, "calculate")
        self.assertIn("<locals>.calculate", calculate.__qualname__)
        self.assertEqual(calculate.__module__, __name__)
        self.assertEqual(calculate.__doc__, "Calculate eagerly.")
        self.assertEqual(
            calculate.__annotations__,
            {"value": int, "scale": int, "return": int},
        )

        lambda_function = lambda value: value + 1
        self.assertIs(torch.compiler.allow_in_graph(lambda_function), lambda_function)
        self.assertEqual(lambda_function(4), 5)
        self.assertEqual(lambda_function.__name__, "<lambda>")
        self.assertEqual(lambda_function.__dict__, {})

        self.assertIs(torch.compiler.allow_in_graph(len), len)
        self.assertEqual(len([1, 2, 3]), 3)

        callable_object = _CallableObject()
        self.assertIs(torch.compiler.allow_in_graph(callable_object), callable_object)
        self.assertEqual(callable_object("value"), "value")
        self.assertEqual(callable_object.calls, ["value"])
        self.assertEqual(callable_object.__dict__, {"calls": ["value"]})

    def test_list_and_tuple_inputs_return_fresh_lists_of_exact_callables(self):
        def first(value):
            return value + 1

        second = lambda value: value * 2
        original = (first, [second, len])

        returned = torch.compiler.allow_in_graph(original)

        self.assertIsInstance(returned, list)
        self.assertIsNot(returned, original)
        self.assertIs(returned[0], first)
        self.assertIsInstance(returned[1], list)
        self.assertIsNot(returned[1], original[1])
        self.assertIs(returned[1][0], second)
        self.assertIs(returned[1][1], len)
        self.assertEqual(returned[0](3), 4)
        self.assertEqual(returned[1][0](3), 6)
        self.assertEqual(returned[1][1]([1, 2]), 2)

    def test_noncallable_targets_raise_pytorch_2_13_assertion(self):
        valid_function = lambda: None
        cases = (None, 1, object(), "value", [valid_function, 1])
        for target in cases:
            with self.subTest(target=target):
                with self.assertRaises(AssertionError) as raised:
                    torch.compiler.allow_in_graph(target)
                self.assertEqual(
                    str(raised.exception),
                    "allow_in_graph expects a callable",
                )
                self.assertEqual(
                    raised.exception.args,
                    ("allow_in_graph expects a callable",),
                )

    def test_signature_metadata_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.allow_in_graph

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(fn)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "allow_in_graph")
        self.assertEqual(function.__qualname__, "allow_in_graph")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertIn(
            "Tells the compiler frontend (Dynamo) to skip symbolic introspection",
            inspect.cleandoc(function.__doc__),
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_wildcard_reload_copy_and_pickle_use_canonical_objects(self):
        compiler = torch.compiler
        function = compiler.allow_in_graph

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            set(COMPILER_EXPORTS),
        )
        for name in COMPILER_EXPORTS:
            self.assertIs(compiler_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("allow_in_graph", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("allow_in_graph", top_level_namespace)

        for copied_function in (function, _picklable_allowed_function):
            self.assertIs(copy.copy(copied_function), copied_function)
            self.assertIs(copy.deepcopy(copied_function), copied_function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(
                    function=copied_function.__name__,
                    protocol=protocol,
                ):
                    payload = pickle.dumps(copied_function, protocol=protocol)
                    self.assertIs(pickle.loads(payload), copied_function)
                    if copied_function is function:
                        self.assertIn(b"torch_rs.compiler", payload)

        self.assertEqual(_picklable_allowed_function(4, increment=3), 7)
        self.assertFalse(hasattr(_picklable_allowed_function, "_dynamo_marked_constant"))
        self.assertFalse(hasattr(_picklable_allowed_function, "_torchdynamo_disable"))

        old_exports = compiler.__all__
        old_function = compiler.allow_in_graph
        reloaded = importlib.reload(compiler)
        self.assertIs(reloaded, compiler)
        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIsNot(compiler.__all__, old_exports)
        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        self.assertIsNot(compiler.allow_in_graph, old_function)
        self.assertIs(copy.copy(compiler.allow_in_graph), compiler.allow_in_graph)

    def test_call_shape_errors_match_pytorch_2_13(self):
        marker = torch.compiler.allow_in_graph
        function = lambda: None
        cases = (
            (
                lambda: marker(),
                "allow_in_graph() missing 1 required positional argument: 'fn'",
            ),
            (
                lambda: marker(function, function),
                "allow_in_graph() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: marker(function, fn=function),
                "allow_in_graph() got multiple values for argument 'fn'",
            ),
            (
                lambda: marker(function, extra=True),
                "allow_in_graph() got an unexpected keyword argument 'extra'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_marker_does_not_enable_compiler_execution_paths(self):
        @torch.compiler.allow_in_graph
        def state():
            return (
                torch.compiler.is_compiling(),
                torch.compiler.is_dynamo_compiling(),
                torch.compiler.is_exporting(),
            )

        self.assertEqual(state(), (False, False, False))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))
        self.assertFalse(hasattr(torch.compiler, "substitute_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "cudagraph_mark_step_begin"))

    def test_import_and_marking_do_not_import_pytorch_or_compiler_backends(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

modules_before_call = set(sys.modules)
calls = []

@torch.compiler.allow_in_graph
def function(value):
    calls.append(value)
    return value + 1

assert function(1) == 2
assert function(2) == 3
assert calls == [1, 2]
assert set(sys.modules) == modules_before_call
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name.startswith("torch_rs._dynamo")
    or name.startswith("torch_rs.compiler.backends")
    or name == "torch_rs.compiler.registry"
    for name in sys.modules
)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
