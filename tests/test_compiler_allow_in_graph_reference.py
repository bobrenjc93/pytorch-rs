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
def _actual_picklable_function(value):
    return value + 1


def _reference_picklable_function(value):
    return value + 1


if reference_torch is not None:
    _reference_picklable_function = reference_torch.compiler.allow_in_graph(
        _reference_picklable_function
    )


class _Callable:
    def __init__(self):
        self.values = []

    def __call__(self, value):
        self.values.append(value)
        return value + len(self.values)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerAllowInGraphReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.allow_in_graph differentials require pinned PyTorch 2.13.0"
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

    def marker_outcome(self, module):
        calls = []

        def function(value: int, *, scale: int = 1) -> int:
            """Calculate eagerly."""
            calls.append((value, scale))
            return value * scale + len(calls)

        function.custom_attribute = []
        before_dict = dict(function.__dict__)
        lambda_function = lambda value: value + 2
        callable_object = _Callable()

        marked = module.compiler.allow_in_graph(function)
        keyword_marked = module.compiler.allow_in_graph(fn=function)
        marked_lambda = module.compiler.allow_in_graph(lambda_function)
        marked_builtin = module.compiler.allow_in_graph(len)
        marked_object = module.compiler.allow_in_graph(callable_object)
        list_target = [function, len]
        tuple_target = (function, len)
        nested_target = [function, (len,)]
        marked_list = module.compiler.allow_in_graph(list_target)
        marked_tuple = module.compiler.allow_in_graph(tuple_target)
        marked_nested = module.compiler.allow_in_graph(nested_target)

        return (
            marked is function,
            keyword_marked is function,
            function.__dict__ == before_dict,
            function(3, scale=2),
            function(3, scale=2),
            calls,
            str(inspect.signature(function)),
            function.__name__,
            function.__module__.split(".")[-1],
            marked_lambda is lambda_function,
            lambda_function(3),
            lambda_function.__dict__,
            marked_builtin is len,
            len([1, 2, 3]),
            marked_object is callable_object,
            callable_object(4),
            callable_object.__dict__,
            type(marked_list),
            marked_list is not list_target,
            tuple(
                item is original for item, original in zip(marked_list, list_target)
            ),
            type(marked_tuple),
            tuple(
                item is original for item, original in zip(marked_tuple, tuple_target)
            ),
            type(marked_nested[1]),
            marked_nested[0] is function,
            marked_nested[1][0] is len,
        )

    def test_callable_inputs_and_eager_behavior_match_pytorch_2_13(self):
        self.assertEqual(
            self.marker_outcome(torch),
            self.marker_outcome(reference_torch),
        )

    def test_non_callable_rejection_matches_pytorch_2_13(self):
        target_factories = (
            lambda: None,
            lambda: 1,
            object,
            lambda: types.SimpleNamespace(existing="value"),
            dict,
            lambda: [1],
            lambda: (1,),
        )
        for case, target_factory in enumerate(target_factories):
            with self.subTest(case=case):
                actual_target = target_factory()
                expected_target = target_factory()
                self.assert_error_matches(
                    lambda: torch.compiler.allow_in_graph(actual_target),
                    lambda: reference_torch.compiler.allow_in_graph(expected_target),
                )

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

    def test_exports_wildcard_copy_pickle_and_reload_match_pytorch_2_13(self):
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
            self.assertEqual(function(4), 5)
            self.assertEqual(function.__dict__, {})
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
import importlib
import sys

import torch as reference_torch
import torch_rs as torch

def outcome(module):
    compiler = module.compiler
    old_function = compiler.allow_in_graph
    old_exports = compiler.__all__
    reloaded = importlib.reload(compiler)
    new_function = reloaded.allow_in_graph

    def after_reload(value):
        return value + 1

    return (
        reloaded is compiler,
        module.compiler is compiler,
        new_function is not old_function,
        compiler.__all__ is not old_exports,
        compiler.allow_in_graph(after_reload) is after_reload,
        after_reload(4),
    )

actual = outcome(torch)
expected = outcome(reference_torch)
if actual != expected:
    raise AssertionError((actual, expected))
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

    def test_compilation_and_compiler_extensions_remain_unsupported(self):
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
        self.assertFalse(hasattr(torch.compiler, "substitute_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))
        self.assertFalse(torch.cuda.is_available())
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertFalse(torch.cuda.is_initialized())

        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(hasattr(reference_torch, "export"))
        self.assertTrue(callable(reference_torch.compiler.substitute_in_graph))


if __name__ == "__main__":
    unittest.main()
