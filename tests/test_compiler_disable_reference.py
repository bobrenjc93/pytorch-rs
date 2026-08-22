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


@torch.compiler.disable
def _actual_picklable_function(value):
    return value + 1


def _reference_picklable_function(value):
    return value + 1


if reference_torch is not None:
    _reference_picklable_function = reference_torch.compiler.disable(
        _reference_picklable_function
    )


class _CallableInstance:
    def __call__(self):
        return "called"


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerDisableReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.disable differentials require pinned PyTorch 2.13.0"
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

    def disable_outcome(self, module, recursive, reason):
        calls = []

        def calculate(value: int, *, scale=1) -> int:
            """Calculate a stateful eager result."""
            calls.append((value, scale))
            return value * scale + len(calls)

        calculate.label = "preserved"
        wrapper = module.compiler.disable(
            calculate,
            recursive=recursive,
            reason=reason,
        )
        return (
            wrapper is calculate,
            wrapper(3, scale=2),
            wrapper(3, scale=2),
            calls,
            wrapper.__name__,
            wrapper.__qualname__.split(".")[-1],
            wrapper.__doc__,
            wrapper.__annotations__,
            str(inspect.signature(wrapper)),
            wrapper.label,
            wrapper.__wrapped__ is calculate,
            wrapper._torchdynamo_disable,
            wrapper._torchdynamo_disable_msg,
            wrapper._torchdynamo_orig_callable is calculate,
            wrapper._torchdynamo_wrapper_id == id(wrapper),
            wrapper._torchdynamo_disable_recursive,
            hasattr(calculate, "_torchdynamo_disable"),
        )

    def test_direct_eager_wrappers_match_pytorch_2_13(self):
        for recursive in (True, False, [True], []):
            with self.subTest(recursive=recursive):
                self.assertEqual(
                    self.disable_outcome(torch, recursive, "reason"),
                    self.disable_outcome(reference_torch, recursive, "reason"),
                )

    def test_repeated_disable_unwraps_to_the_same_innermost_function(self):
        outcomes = []
        for module in (torch, reference_torch):

            def function(value):
                return value + 1

            first = module.compiler.disable(function, reason="first")
            first.outer_only = "not copied"
            second = module.compiler.disable(
                first,
                recursive=False,
                reason="second",
            )
            outcomes.append(
                (
                    second is first,
                    second.__wrapped__ is function,
                    second._torchdynamo_orig_callable is function,
                    second._torchdynamo_disable_msg,
                    second._torchdynamo_disable_recursive,
                    hasattr(second, "outer_only"),
                    second(4),
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_method_binding_matches_pytorch_2_13(self):
        def outcome(module):
            class Accumulator:
                def __init__(self):
                    self.total = 0

                @module.compiler.disable
                def add(self, value):
                    self.total += value
                    return self.total

            left = Accumulator()
            right = Accumulator()
            bound = right.add
            rebound = module.compiler.disable(
                bound,
                recursive=False,
                reason="bound",
            )
            return (
                left.add.__self__ is left,
                left.add.__func__ is Accumulator.add,
                left.add(2),
                left.add(3),
                right.add(7),
                rebound(1),
                rebound.__wrapped__ is bound,
                rebound._torchdynamo_orig_callable is bound,
                rebound._torchdynamo_disable_recursive,
                rebound._torchdynamo_disable_msg,
            )

        self.assertEqual(outcome(torch), outcome(reference_torch))

    def test_signature_documentation_and_ownership_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.disable
        expected = expected_compiler.disable

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

    def test_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        supported = {
            "assume_constant_result",
            "reset",
            "disable",
            "set_default_backend",
            "get_default_backend",
            "is_compiling",
            "is_dynamo_compiling",
            "is_exporting",
        }

        self.assertEqual(
            actual_compiler.__all__,
            [name for name in expected_compiler.__all__ if name in supported],
        )
        self.assertEqual(
            torch.__all__.count("disable"),
            reference_torch.__all__.count("disable"),
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
            self.assertNotIn("disable", namespace)

        for function in (actual_compiler.disable, expected_compiler.disable):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                actual = actual_compiler.disable
                expected = expected_compiler.disable
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
            self.assertEqual(function(4), 5)
            self.assertIs(function._torchdynamo_disable, True)
            self.assertIs(function._torchdynamo_disable_recursive, True)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(wrapper=function.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol)),
                        function,
                    )

    def test_invalid_direct_call_and_call_shape_errors_match_pytorch_2_13(self):
        self.assert_error_matches(
            lambda: torch.compiler.disable(1),
            lambda: reference_torch.compiler.disable(1),
        )

        actual_function = lambda: None
        expected_function = lambda: None
        cases = (
            (
                lambda: torch.compiler.disable(actual_function, True, "reason"),
                lambda: reference_torch.compiler.disable(
                    expected_function,
                    True,
                    "reason",
                ),
            ),
            (
                lambda: torch.compiler.disable(
                    actual_function,
                    True,
                    recursive=False,
                ),
                lambda: reference_torch.compiler.disable(
                    expected_function,
                    True,
                    recursive=False,
                ),
            ),
            (
                lambda: torch.compiler.disable(actual_function, extra=True),
                lambda: reference_torch.compiler.disable(
                    expected_function,
                    extra=True,
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_factory_classes_builtins_and_compilation_remain_unsupported(self):
        class Target:
            def __init__(self):
                self.value = 1

            def __call__(self):
                return self.value

        for target in (None, Target, len, _CallableInstance()):
            with self.subTest(target=target):
                with self.assertRaises(NotImplementedError):
                    torch.compiler.disable(target)

        self.assertTrue(callable(reference_torch.compiler.disable()))
        self.assertEqual(reference_torch.compiler.disable(len)([1, 2]), 2)
        self.assertIs(reference_torch.compiler.disable(Target), Target)
        self.assertTrue(callable(reference_torch.compile))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
