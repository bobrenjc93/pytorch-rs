import copy
import importlib
import inspect
import pickle
import pickletools
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


SUPPORTED_COMPILER_EXPORTS = {
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


class _CallableObject:
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

    def callable_outcome(self, module):
        calls = []

        def calculate(value: int, *, scale: int = 1) -> int:
            """Calculate eagerly."""
            calls.append((value, scale))
            return value * scale + len(calls)

        marker = object()
        calculate.custom_attribute = marker
        before_dict = dict(calculate.__dict__)
        returned = module.compiler.allow_in_graph(calculate)
        lambda_function = lambda value: value + 1
        lambda_returned = module.compiler.allow_in_graph(lambda_function)
        callable_object = _CallableObject()
        callable_returned = module.compiler.allow_in_graph(callable_object)

        return (
            returned is calculate,
            calculate.__dict__ == before_dict,
            calculate.custom_attribute is marker,
            hasattr(calculate, "_dynamo_marked_constant"),
            hasattr(calculate, "_torchdynamo_disable"),
            hasattr(calculate, "__wrapped__"),
            calculate(3, scale=2),
            calculate(3, scale=2),
            calls,
            str(inspect.signature(calculate)),
            calculate.__name__,
            calculate.__doc__,
            calculate.__annotations__,
            lambda_returned is lambda_function,
            lambda_function(4),
            lambda_function.__name__,
            lambda_function.__dict__,
            module.compiler.allow_in_graph(len) is len,
            len([1, 2, 3]),
            callable_returned is callable_object,
            callable_object("value"),
            callable_object.__dict__,
        )

    def test_callable_inputs_and_eager_behavior_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_outcome(torch),
            self.callable_outcome(reference_torch),
        )

    def test_list_and_tuple_inputs_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):

            def first(value):
                return value + 1

            second = lambda value: value * 2
            original = (first, [second, len])
            returned = module.compiler.allow_in_graph(original)
            outcomes.append(
                (
                    isinstance(returned, list),
                    returned is original,
                    returned[0] is first,
                    isinstance(returned[1], list),
                    returned[1] is original[1],
                    returned[1][0] is second,
                    returned[1][1] is len,
                    returned[0](3),
                    returned[1][0](3),
                    returned[1][1]([1, 2]),
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_noncallable_errors_match_pytorch_2_13(self):
        target_factories = (
            lambda: None,
            lambda: 1,
            object,
            lambda: "value",
            lambda: [lambda: None, 1],
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
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
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

        self.assertEqual(
            actual_compiler.__all__,
            [
                name
                for name in expected_compiler.__all__
                if name in SUPPORTED_COMPILER_EXPORTS
            ],
        )
        self.assertEqual(
            torch.__all__.count("compiler"),
            reference_torch.__all__.count("compiler"),
        )
        self.assertEqual(
            torch.__all__.count("allow_in_graph"),
            reference_torch.__all__.count("allow_in_graph"),
        )

        actual_namespace = {}
        exec("from torch_rs.compiler import *", actual_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            SUPPORTED_COMPILER_EXPORTS,
        )
        for name in SUPPORTED_COMPILER_EXPORTS:
            self.assertIs(actual_namespace[name], getattr(actual_compiler, name))

        expected_namespace = {}
        exec("from torch.compiler import *", expected_namespace)
        for name in SUPPORTED_COMPILER_EXPORTS:
            self.assertIs(expected_namespace[name], getattr(expected_compiler, name))

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
                actual_payload = pickle.dumps(actual, protocol=protocol)
                expected_payload = pickle.dumps(expected, protocol=protocol)
                self.assertIs(pickle.loads(actual_payload), actual)
                self.assertIs(pickle.loads(expected_payload), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

        for function in (_actual_picklable_function, _reference_picklable_function):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            self.assertEqual(function(4), 5)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(marked=function.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol)),
                        function,
                    )

        old_actual_all = actual_compiler.__all__
        old_actual = actual_compiler.allow_in_graph
        self.assertIs(importlib.reload(actual_compiler), actual_compiler)
        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertEqual(old_actual_all is actual_compiler.__all__, False)
        self.assertEqual(old_actual is actual_compiler.allow_in_graph, False)
        self.assertEqual(
            actual_compiler.__all__,
            [
                name
                for name in expected_compiler.__all__
                if name in SUPPORTED_COMPILER_EXPORTS
            ],
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

    def test_execution_paths_remain_unsupported(self):
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
        self.assertTrue(hasattr(reference_torch.compiler, "substitute_in_graph"))


if __name__ == "__main__":
    unittest.main()
