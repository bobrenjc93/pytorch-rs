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


FUNCTION_DOC = """
    Mark a callable as safe to insert into compiler graphs.

    This PyTorch-compatible eager marker validates ``fn`` and returns callable
    objects unchanged. The current implementation does not register the object
    with Dynamo, call into PyTorch, compile the callable, or change eager
    execution behavior. ``torch.compile``, backend registration,
    ``substitute_in_graph``, graph execution, Dynamo execution state, and
    compiler CUDA behavior remain unsupported.
    """


def _picklable_function(value, *, increment=1):
    return value + increment


_picklable_function = torch.compiler.allow_in_graph(_picklable_function)


class _CallableObject:
    def __init__(self):
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return value + 3


class CompilerAllowInGraphTests(unittest.TestCase):
    def test_callable_inputs_return_the_same_object_and_run_eagerly(self):
        calls = []

        def function(value, *, scale=1):
            calls.append((value, scale))
            return value * scale + len(calls)

        function.custom_attribute = ["preserved"]
        original_dict = dict(function.__dict__)

        positional = torch.compiler.allow_in_graph(function)
        keyword = torch.compiler.allow_in_graph(fn=function)

        self.assertIs(positional, function)
        self.assertIs(keyword, function)
        self.assertEqual(function.__dict__, original_dict)
        self.assertEqual(function(3, scale=2), 7)
        self.assertEqual(function(3, scale=2), 8)
        self.assertEqual(calls, [(3, 2), (3, 2)])
        self.assertEqual(str(inspect.signature(function)), "(value, *, scale=1)")
        self.assertEqual(function.__name__, "function")
        self.assertIn("<locals>.function", function.__qualname__)
        self.assertEqual(function.__module__, __name__)

        lambda_function = lambda value: value + 2
        self.assertIs(torch.compiler.allow_in_graph(lambda_function), lambda_function)
        self.assertEqual(lambda_function(4), 6)

        self.assertIs(torch.compiler.allow_in_graph(len), len)
        self.assertEqual(torch.compiler.allow_in_graph(len)([1, 2, 3]), 3)

        callable_object = _CallableObject()
        object_dict = dict(callable_object.__dict__)
        self.assertIs(torch.compiler.allow_in_graph(callable_object), callable_object)
        self.assertEqual(callable_object.__dict__, object_dict)
        self.assertEqual(callable_object(5), 8)
        self.assertEqual(callable_object.calls, [5])

    def test_tuple_and_list_inputs_return_fresh_lists_of_original_callables(self):
        def function(value):
            return value + 1

        targets = [function, len]
        result = torch.compiler.allow_in_graph(targets)
        self.assertIs(type(result), list)
        self.assertIsNot(result, targets)
        self.assertEqual(result, targets)
        self.assertIs(result[0], function)
        self.assertIs(result[1], len)

        tuple_result = torch.compiler.allow_in_graph((function, len))
        self.assertIs(type(tuple_result), list)
        self.assertEqual(tuple_result, targets)
        self.assertIs(tuple_result[0], function)
        self.assertIs(tuple_result[1], len)

    def test_non_callables_raise_assertion_without_mutation(self):
        namespace = types.SimpleNamespace(existing="preserved")
        targets = (None, 1, object(), "not callable", namespace)
        for target in targets:
            with self.subTest(target=target):
                before = dict(getattr(target, "__dict__", {}))
                with self.assertRaises(AssertionError) as raised:
                    torch.compiler.allow_in_graph(target)
                self.assertEqual(str(raised.exception), "allow_in_graph expects a callable")
                self.assertEqual(
                    raised.exception.args,
                    ("allow_in_graph expects a callable",),
                )
                self.assertEqual(getattr(target, "__dict__", {}), before)

        with self.assertRaisesRegex(
            AssertionError,
            "^allow_in_graph expects a callable$",
        ):
            torch.compiler.allow_in_graph([lambda: None, 1])

    def test_non_callable_rejection_survives_optimized_python(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

if sys.flags.optimize <= 0:
    raise AssertionError("subprocess did not run with optimization enabled")

for target in (None, 1, "not callable"):
    try:
        torch.compiler.allow_in_graph(target)
    except AssertionError as error:
        if str(error) != "allow_in_graph expects a callable":
            raise AssertionError(str(error))
        if error.args != ("allow_in_graph expects a callable",):
            raise AssertionError(error.args)
    else:
        raise AssertionError(f"optimized mode accepted {target!r}")

try:
    torch.compiler.allow_in_graph([lambda: None, 1])
except AssertionError as error:
    if str(error) != "allow_in_graph expects a callable":
        raise AssertionError(str(error))
else:
    raise AssertionError("optimized mode accepted a non-callable sequence entry")

if torch.compiler.allow_in_graph(len) is not len:
    raise AssertionError("callable identity was not preserved")
if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
    raise AssertionError("PyTorch was imported")
"""
        completed = subprocess.run(
            [sys.executable, "-O", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
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
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_wildcard_reload_copy_and_pickle_use_canonical_objects(self):
        compiler = torch.compiler
        function = compiler.allow_in_graph

        self.assertEqual(
            compiler.__all__,
            [
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
            ],
        )
        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            set(compiler.__all__),
        )
        for name in compiler.__all__:
            self.assertIs(compiler_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("allow_in_graph", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("allow_in_graph", top_level_namespace)

        for copied_function in (function, _picklable_function):
            self.assertIs(copy.copy(copied_function), copied_function)
            self.assertIs(copy.deepcopy(copied_function), copied_function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(
                    function=copied_function.__name__,
                    protocol=protocol,
                ):
                    payload = pickle.dumps(copied_function, protocol=protocol)
                    if copied_function is function:
                        self.assertIn(b"torch_rs.compiler", payload)
                    self.assertIs(pickle.loads(payload), copied_function)

        self.assertEqual(_picklable_function(4, increment=3), 7)
        self.assertEqual(_picklable_function.__dict__, {})

        old_function = function
        old_exports = compiler.__all__
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            reloaded = importlib.reload(compiler)
            new_function = reloaded.allow_in_graph

            self.assertIs(reloaded, compiler)
            self.assertIs(torch.compiler, compiler)
            self.assertIs(sys.modules["torch_rs.compiler"], compiler)
            self.assertIsNot(new_function, old_function)
            self.assertIsNot(compiler.__all__, old_exports)
            self.assertEqual(compiler.__all__, old_exports)
            self.assertIs(compiler.get_default_backend(), backend)
            self.assertIs(new_function(len), len)
            self.assertIs(copy.copy(old_function), old_function)
            self.assertIs(copy.deepcopy(old_function), old_function)
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(old_function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertIs(
                    pickle.loads(pickle.dumps(new_function, protocol)),
                    new_function,
                )
        finally:
            compiler.set_default_backend(original_backend)

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

    def test_marker_does_not_enable_compile_or_import_pytorch(self):
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

assert torch.compiler.allow_in_graph(function) is function
assert not hasattr(function, "_dynamo_marked_constant")
assert not hasattr(function, "_torchdynamo_disable")
assert function(1) == 2
assert function(2) == 3
assert calls == [1, 2]
assert torch.compiler.is_compiling() is False
assert torch.compiler.is_dynamo_compiling() is False
assert torch.compiler.is_exporting() is False
assert not hasattr(torch, "compile")
assert not hasattr(torch.compiler, "compile")
assert not hasattr(torch.compiler, "register_backend")
assert not hasattr(torch.compiler, "substitute_in_graph")
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
