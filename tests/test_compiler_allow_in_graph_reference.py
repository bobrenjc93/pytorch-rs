import copy
import importlib
import inspect
import pickle
import pickletools
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@torch.compiler.allow_in_graph
def _actual_picklable_function(value, *, increment=1):
    return value + increment


def _reference_picklable_function(value, *, increment=1):
    return value + increment


if reference_torch is not None:
    _reference_picklable_function = reference_torch.compiler.allow_in_graph(
        _reference_picklable_function
    )


class _Callable:
    def __init__(self):
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerAllowInGraphReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.allow_in_graph differentials require pinned "
                "PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def decorator_outcome(self, module):
        calls = []

        @module.compiler.allow_in_graph
        def calculate(value, *, scale=1):
            calls.append((value, scale))
            return value * scale + len(calls)

        return (
            calculate(3, scale=2),
            calculate(3, scale=2),
            calls,
            str(inspect.signature(calculate)),
            calculate.__name__,
            calculate.__module__.replace("torch_rs", "torch"),
            dict(calculate.__dict__),
            hasattr(calculate, "__wrapped__"),
        )

    def callable_inputs_outcome(self, module):
        def function(value):
            return value + 1

        lambda_function = lambda value: value + 2
        callable_object = _Callable()
        function.custom_metadata = ["preserved"]
        lambda_function.custom_metadata = {"preserved": True}

        returned_function = module.compiler.allow_in_graph(function)
        returned_lambda = module.compiler.allow_in_graph(lambda_function)
        returned_builtin = module.compiler.allow_in_graph(len)
        returned_object = module.compiler.allow_in_graph(callable_object)

        return (
            returned_function is function,
            returned_lambda is lambda_function,
            returned_builtin is len,
            returned_object is callable_object,
            function.__dict__,
            lambda_function.__dict__,
            callable_object.__dict__,
            returned_function(4),
            returned_lambda(4),
            returned_builtin([1, 2, 3]),
            returned_object("value"),
            callable_object.calls,
        )

    def recursive_container_outcome(self, module):
        def first():
            return "first"

        def second():
            return "second"

        list_input = [first, second]
        tuple_input = (first, second)
        list_result = module.compiler.allow_in_graph(list_input)
        tuple_result = module.compiler.allow_in_graph(tuple_input)
        empty_result = module.compiler.allow_in_graph(())
        return (
            type(list_result),
            list_result is list_input,
            [item is original for item, original in zip(list_result, list_input)],
            [item() for item in list_result],
            type(tuple_result),
            tuple_result is tuple_input,
            [item is original for item, original in zip(tuple_result, tuple_input)],
            [item() for item in tuple_result],
            type(empty_result),
            empty_result,
        )

    def test_decorator_use_and_eager_behavior_match_pytorch_2_13(self):
        self.assertEqual(
            self.decorator_outcome(torch),
            self.decorator_outcome(reference_torch),
        )

    def test_callable_inputs_return_exact_objects_and_metadata_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_inputs_outcome(torch),
            self.callable_inputs_outcome(reference_torch),
        )

    def test_list_and_tuple_inputs_match_pytorch_2_13(self):
        self.assertEqual(
            self.recursive_container_outcome(torch),
            self.recursive_container_outcome(reference_torch),
        )

    def test_noncallable_errors_match_pytorch_2_13(self):
        actual_function = lambda: None
        expected_function = lambda: None
        cases = (
            (
                lambda: torch.compiler.allow_in_graph(None),
                lambda: reference_torch.compiler.allow_in_graph(None),
            ),
            (
                lambda: torch.compiler.allow_in_graph(1),
                lambda: reference_torch.compiler.allow_in_graph(1),
            ),
            (
                lambda: torch.compiler.allow_in_graph("value"),
                lambda: reference_torch.compiler.allow_in_graph("value"),
            ),
            (
                lambda: torch.compiler.allow_in_graph(object()),
                lambda: reference_torch.compiler.allow_in_graph(object()),
            ),
            (
                lambda: torch.compiler.allow_in_graph([actual_function, 1]),
                lambda: reference_torch.compiler.allow_in_graph(
                    [expected_function, 1]
                ),
            ),
            (
                lambda: torch.compiler.allow_in_graph((1,)),
                lambda: reference_torch.compiler.allow_in_graph((1,)),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_signature_documentation_and_ownership_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.allow_in_graph
        expected = expected_compiler.allow_in_graph

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_compiler)
        self.assertIs(inspect.getmodule(expected), expected_compiler)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.allow_in_graph
        expected = expected_compiler.allow_in_graph
        supported = {
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
        }

        self.assertEqual(
            actual_compiler.__all__,
            [name for name in expected_compiler.__all__ if name in supported],
        )
        self.assertEqual(
            torch.__all__.count("compiler"),
            reference_torch.__all__.count("compiler"),
        )
        self.assertEqual(
            torch.__all__.count("allow_in_graph"),
            reference_torch.__all__.count("allow_in_graph"),
        )

        for module in (actual_compiler, expected_compiler):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in supported:
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("allow_in_graph", namespace)

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

        for function in (_actual_picklable_function, _reference_picklable_function):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            self.assertEqual(function(4, increment=3), 7)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(marked=function.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol)), function
                    )

    def test_call_shape_errors_match_pytorch_2_13(self):
        actual = torch.compiler.allow_in_graph
        expected = reference_torch.compiler.allow_in_graph
        actual_function = lambda: None
        expected_function = lambda: None
        cases = (
            (lambda: actual(), lambda: expected()),
            (
                lambda: actual(actual_function, actual_function),
                lambda: expected(expected_function, expected_function),
            ),
            (
                lambda: actual(actual_function, fn=actual_function),
                lambda: expected(expected_function, fn=expected_function),
            ),
            (
                lambda: actual(actual_function, extra=True),
                lambda: expected(expected_function, extra=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_reload_matches_pytorch_2_13_in_an_isolated_process(self):
        script = r"""
import copy
import importlib
import pickle
import sys

import torch as reference_torch
import torch_rs as torch

if reference_torch.__version__.split("+")[0] != "2.13.0":
    raise SystemExit(f"expected PyTorch 2.13.0, got {reference_torch.__version__}")

supported = {
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
}

def pickle_old_function_error(function):
    try:
        pickle.dumps(function)
    except Exception as error:
        return type(error), str(error)
    return None, None

def reload_outcome(package, module_name):
    compiler = importlib.import_module(module_name)
    old_function = compiler.allow_in_graph
    old_exports = compiler.__all__
    reloaded = importlib.reload(compiler)
    new_function = reloaded.allow_in_graph
    error_type, _ = pickle_old_function_error(old_function)
    return (
        reloaded is compiler,
        package.compiler is compiler,
        sys.modules[module_name] is compiler,
        new_function is old_function,
        compiler.__all__ is old_exports,
        new_function(len) is len,
        copy.copy(old_function) is old_function,
        copy.deepcopy(old_function) is old_function,
        error_type,
        compiler.__all__,
    )

actual = reload_outcome(torch, "torch_rs.compiler")
expected = reload_outcome(reference_torch, "torch.compiler")
assert actual[:9] == expected[:9], (actual[:9], expected[:9])
assert actual[9] == [name for name in expected[9] if name in supported], (
    actual[9],
    expected[9],
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

    def test_graph_execution_and_unsupported_compiler_surface_remain_unsupported(self):
        @torch.compiler.allow_in_graph
        def function():
            return (
                torch.compiler.is_compiling(),
                torch.compiler.is_dynamo_compiling(),
                torch.compiler.is_exporting(),
            )

        self.assertEqual(function(), (False, False, False))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))
        self.assertFalse(hasattr(torch.compiler, "substitute_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "cudagraph_mark_step_begin"))
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(hasattr(reference_torch, "export"))
        self.assertTrue(hasattr(reference_torch.compiler, "substitute_in_graph"))
        self.assertTrue(hasattr(reference_torch.compiler, "cudagraph_mark_step_begin"))


if __name__ == "__main__":
    unittest.main()
