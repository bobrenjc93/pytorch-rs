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


class _TrackingList(list):
    def __init__(self, values):
        super().__init__(values)
        self.visited = []

    def __iter__(self):
        for index, value in enumerate(super().__iter__()):
            self.visited.append(index)
            yield value


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

    def callable_outcome(self, module):
        calls = []

        def calculate(value: int, *, scale: int = 1) -> int:
            """Calculate eagerly."""
            calls.append((value, scale))
            return value * scale + len(calls)

        calculate.custom_metadata = object()
        original = calculate
        original_dict = calculate.__dict__.copy()
        decorated = module.compiler.allow_in_graph(calculate)
        return (
            decorated is original,
            decorated.__dict__ == original_dict,
            decorated(3, scale=2),
            decorated(3, scale=2),
            calls,
            str(inspect.signature(decorated)),
            decorated.__name__,
            decorated.__doc__,
        )

    def test_callable_identity_and_eager_calls_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_outcome(torch),
            self.callable_outcome(reference_torch),
        )

    def method_outcome(self, module):
        class Accumulator:
            def __init__(self):
                self.total = 0

            def add(self, value):
                self.total += value
                return self.total

            @classmethod
            def identify(cls, value):
                return cls, value

            @staticmethod
            def double(value):
                return value * 2

        original_add = Accumulator.add
        Accumulator.add = module.compiler.allow_in_graph(Accumulator.add)
        left = Accumulator()
        right = Accumulator()
        bound = left.add
        allowed_bound = module.compiler.allow_in_graph(bound)
        return (
            Accumulator.add is original_add,
            isinstance(left.add, types.MethodType),
            left.add.__self__ is left,
            left.add.__func__ is Accumulator.add,
            allowed_bound is bound,
            left.add(2),
            left.add(3),
            right.add(7),
            Accumulator.identify("value")[1],
            Accumulator.double(4),
            allowed_bound(1),
        )

    def test_method_binding_matches_pytorch_2_13(self):
        self.assertEqual(
            self.method_outcome(torch),
            self.method_outcome(reference_torch),
        )

    def sequence_outcome(self, module):
        def first(value):
            return value + 1

        def second(value):
            return value * 2

        source_list = [first, second]
        from_list = module.compiler.allow_in_graph(source_list)
        source_tuple = (first, second)
        from_tuple = module.compiler.allow_in_graph(source_tuple)
        nested_source = (first, [second, (first,)])
        nested = module.compiler.allow_in_graph(nested_source)
        empty_list = []
        empty_result = module.compiler.allow_in_graph(empty_list)
        return (
            type(from_list) is list,
            from_list is not source_list,
            from_list[0] is first,
            from_list[1] is second,
            type(from_tuple) is list,
            from_tuple[0] is first,
            from_tuple[1] is second,
            type(nested) is list,
            nested[0] is first,
            type(nested[1]) is list,
            nested[1] is not nested_source[1],
            nested[1][0] is second,
            type(nested[1][1]) is list,
            nested[1][1][0] is first,
            empty_result,
            empty_result is not empty_list,
            module.compiler.allow_in_graph(()) == [],
        )

    def test_sequence_materialization_matches_pytorch_2_13(self):
        self.assertEqual(
            self.sequence_outcome(torch),
            self.sequence_outcome(reference_torch),
        )

    def test_noncallable_and_call_shape_errors_match_pytorch_2_13(self):
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
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(1), lambda: expected(1)),
            (lambda: actual("callable"), lambda: expected("callable")),
            (lambda: actual(iter(())), lambda: expected(iter(()))),
            (
                lambda: actual((actual_function, None)),
                lambda: expected((expected_function, None)),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def validation_order(self, module):
        def first():
            return "first"

        def unvisited():
            return "unvisited"

        entries = _TrackingList([first, None, unvisited])
        try:
            module.compiler.allow_in_graph(entries)
        except BaseException as error:
            outcome = (type(error).__name__, str(error), error.args)
        else:
            outcome = None
        return entries.visited, outcome

    def test_left_to_right_validation_matches_pytorch_2_13(self):
        self.assertEqual(
            self.validation_order(torch),
            self.validation_order(reference_torch),
        )
        self.assertEqual(self.validation_order(torch)[0], [0, 1])

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

    def test_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.allow_in_graph
        expected = expected_compiler.allow_in_graph
        supported = {
            "assume_constant_result",
            "reset",
            "allow_in_graph",
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

        for function in (
            _actual_picklable_function,
            _reference_picklable_function,
        ):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            self.assertEqual(function.__dict__, {})
            self.assertEqual(function(4), 5)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol)), function
                    )

    def test_reload_behavior_matches_pytorch_2_13(self):
        script = r"""
import copy
import importlib
import pickle
import sys

import torch_rs as torch
import torch as reference_torch


def reload_outcome(module, compiler_module_name):
    compiler = importlib.import_module(compiler_module_name)
    old_function = compiler.allow_in_graph
    old_exports = compiler.__all__
    reloaded = importlib.reload(compiler)
    new_function = reloaded.allow_in_graph

    def eager(value):
        return value + 1

    try:
        pickle.dumps(old_function)
    except BaseException as error:
        old_pickle_error = (
            type(error).__name__,
            "not the same object" in str(error),
        )
    else:
        old_pickle_error = None

    new_pickle_results = tuple(
        pickle.loads(pickle.dumps(new_function, protocol)) is new_function
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
    )
    return (
        reloaded is compiler,
        module.compiler is compiler,
        sys.modules[compiler_module_name] is compiler,
        old_function is new_function,
        old_exports is compiler.__all__,
        new_function(eager) is eager,
        eager(3),
        copy.copy(old_function) is old_function,
        copy.deepcopy(old_function) is old_function,
        old_pickle_error,
        new_pickle_results,
    )


assert reload_outcome(torch, "torch_rs.compiler") == reload_outcome(
    reference_torch, "torch.compiler"
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

    def state_outcome(self, module):
        before = (
            module.compiler.is_compiling(),
            module.compiler.is_dynamo_compiling(),
            module.compiler.is_exporting(),
            module.is_grad_enabled(),
        )

        def function():
            return module.is_grad_enabled()

        original_dict = function.__dict__.copy()
        allowed = module.compiler.allow_in_graph(function)
        after = (
            module.compiler.is_compiling(),
            module.compiler.is_dynamo_compiling(),
            module.compiler.is_exporting(),
            module.is_grad_enabled(),
        )
        return (
            before,
            after,
            allowed is function,
            function.__dict__ == original_dict,
            allowed(),
        )

    def test_eager_state_matches_pytorch_2_13(self):
        self.assertEqual(
            self.state_outcome(torch),
            self.state_outcome(reference_torch),
        )
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(hasattr(reference_torch, "export"))


if __name__ == "__main__":
    unittest.main()
