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


class _Callable:
    def __init__(self):
        self.existing = []

    def __call__(self, value):
        self.existing.append(value)
        return value + len(self.existing)


class CompilerAllowInGraphTests(unittest.TestCase):
    def test_callable_inputs_return_exact_objects_without_wrapping_or_metadata_changes(
        self,
    ):
        calls = []

        def function(value: int, *, scale: int = 1) -> int:
            """Calculate eagerly."""
            calls.append((value, scale))
            return value * scale + len(calls)

        function.custom_attribute = []
        before_dict = dict(function.__dict__)
        marked = torch.compiler.allow_in_graph(function)
        keyword_marked = torch.compiler.allow_in_graph(fn=function)

        self.assertIs(marked, function)
        self.assertIs(keyword_marked, function)
        self.assertEqual(function.__dict__, before_dict)
        self.assertEqual(function(3, scale=2), 7)
        self.assertEqual(function(3, scale=2), 8)
        self.assertEqual(calls, [(3, 2), (3, 2)])
        self.assertEqual(
            str(inspect.signature(function)),
            "(value: int, *, scale: int = 1) -> int",
        )
        self.assertEqual(function.__name__, "function")
        self.assertEqual(function.__module__, __name__)

        lambda_function = lambda value: value + 2
        self.assertIs(torch.compiler.allow_in_graph(lambda_function), lambda_function)
        self.assertEqual(lambda_function(3), 5)
        self.assertEqual(lambda_function.__dict__, {})

        self.assertIs(torch.compiler.allow_in_graph(len), len)
        self.assertEqual(len([1, 2, 3]), 3)

        callable_object = _Callable()
        self.assertIs(torch.compiler.allow_in_graph(callable_object), callable_object)
        self.assertEqual(callable_object(4), 5)
        self.assertEqual(callable_object.__dict__, {"existing": [4]})

    def test_sequence_inputs_return_new_lists_with_same_callables(self):
        def function(value):
            return value + 1

        for target in ([], (), [function, len], (function, len)):
            with self.subTest(target=target):
                result = torch.compiler.allow_in_graph(target)
                self.assertIsInstance(result, list)
                self.assertIsNot(result, target)
                self.assertEqual(len(result), len(target))
                for actual, expected in zip(result, target):
                    self.assertIs(actual, expected)

        nested = torch.compiler.allow_in_graph([function, (len,)])
        self.assertIs(nested[0], function)
        self.assertIsInstance(nested[1], list)
        self.assertIs(nested[1][0], len)

        for target in ([1], (1,)):
            with self.subTest(target=target):
                with self.assertRaises(AssertionError) as raised:
                    torch.compiler.allow_in_graph(target)
                self.assertEqual(
                    str(raised.exception),
                    "allow_in_graph expects a callable",
                )

    def test_non_callable_targets_raise_pytorch_2_13_assertion(self):
        for target in (None, 1, object(), types.SimpleNamespace(existing="value"), {}):
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

    def test_signature_documentation_and_module_identity(self):
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
        self.assertIn("Tells the compiler frontend", function.__doc__)
        self.assertIn("Dynamo", function.__doc__)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_wildcard_copy_pickle_and_reload_use_canonical_objects(self):
        compiler = torch.compiler
        function = compiler.allow_in_graph

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        from torch_rs.compiler import allow_in_graph

        self.assertIs(allow_in_graph, function)

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
        self.assertFalse(hasattr(torch, "allow_in_graph"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("allow_in_graph", top_level_namespace)

        for copied_function in (function, _picklable_allowed_function):
            self.assertIs(copy.copy(copied_function), copied_function)
            self.assertIs(copy.deepcopy(copied_function), copied_function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(
                    function=copied_function.__name__, protocol=protocol
                ):
                    self.assertIs(
                        pickle.loads(pickle.dumps(copied_function, protocol)),
                        copied_function,
                    )

        self.assertEqual(_picklable_allowed_function(4, increment=3), 7)
        self.assertEqual(_picklable_allowed_function.__dict__, {})

        old_function = function
        old_exports = compiler.__all__
        reloaded = importlib.reload(compiler)
        new_function = reloaded.allow_in_graph

        self.assertIs(reloaded, compiler)
        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIsNot(new_function, old_function)
        self.assertIsNot(compiler.__all__, old_exports)
        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)

        def after_reload(value):
            return value + 1

        self.assertIs(new_function(after_reload), after_reload)
        self.assertEqual(after_reload(4), 5)
        self.assertIs(copy.copy(old_function), old_function)
        self.assertIs(copy.deepcopy(old_function), old_function)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(pickle.dumps(new_function, protocol)),
                new_function,
            )

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

    def test_marker_does_not_enable_compilation_or_change_eager_state(self):
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
        self.assertFalse(hasattr(torch, "list_backends"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "substitute_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))
        self.assertFalse(torch.cuda.is_available())
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

calls = []

@torch.compiler.allow_in_graph
def function(value):
    calls.append(value)
    return value + 1

assert torch.compiler.allow_in_graph(len) is len
assert torch.compiler.allow_in_graph((len,))[0] is len
assert function(1) == 2
assert function(2) == 3
assert calls == [1, 2]
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
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
