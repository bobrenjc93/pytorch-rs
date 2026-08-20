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


def _pickle_target(value, offset=2, *, scale=3):
    return (value + offset) * scale


class _CallableTarget:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args, kwargs


class _SlotCallable:
    __slots__ = ()

    def __call__(self):
        return "called"


class _RejectingCallable:
    def __setattr__(self, name, value):
        raise RuntimeError(f"rejected {name}={value!r}")

    def __call__(self):
        return "called"


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerAssumeConstantResultReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.assume_constant_result differentials require pinned "
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

    def function_outcome(self, decorator):
        calls = []

        def target(value: int, offset=2, *, scale: int = 3) -> int:
            """Return a scaled offset."""
            calls.append((value, offset, scale))
            return (value + offset) * scale

        target.user_metadata = "preserved"
        signature = inspect.signature(target)
        annotations = target.__annotations__.copy()
        first = decorator(target)
        second = decorator(target)
        return (
            first is target,
            second is target,
            target._dynamo_marked_constant is True,
            target(4, scale=5),
            calls,
            inspect.signature(target) == signature,
            target.__annotations__ == annotations,
            target.__name__,
            target.__doc__,
            target.user_metadata,
        )

    def callable_outcome(self, decorator):
        target = _CallableTarget()
        result = decorator(fn=target)
        call_result = result(1, key="value")
        return (
            result is target,
            target._dynamo_marked_constant is True,
            call_result,
            target.calls,
        )

    def test_function_callable_and_idempotent_behavior_match_pytorch_2_13(self):
        self.assertEqual(
            self.function_outcome(torch.compiler.assume_constant_result),
            self.function_outcome(reference_torch.compiler.assume_constant_result),
        )
        self.assertEqual(
            self.callable_outcome(torch.compiler.assume_constant_result),
            self.callable_outcome(reference_torch.compiler.assume_constant_result),
        )

    def test_unconditional_assignment_without_callability_validation_matches(self):
        for decorator in (
            torch.compiler.assume_constant_result,
            reference_torch.compiler.assume_constant_result,
        ):
            target = types.SimpleNamespace(_dynamo_marked_constant="old")
            result = decorator(target)
            self.assertIs(result, target)
            self.assertIs(target._dynamo_marked_constant, True)
            self.assertFalse(callable(target))

    def test_signature_documentation_and_ownership_match(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.assume_constant_result
        expected = expected_compiler.assume_constant_result

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

    def test_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.assume_constant_result
        expected = expected_compiler.assume_constant_result
        supported = {
            "assume_constant_result",
            "is_compiling",
            "is_dynamo_compiling",
            "is_exporting",
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
            torch.__all__.count("assume_constant_result"),
            reference_torch.__all__.count("assume_constant_result"),
        )
        self.assertEqual(
            hasattr(torch, "assume_constant_result"),
            hasattr(reference_torch, "assume_constant_result"),
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
            self.assertNotIn("assume_constant_result", namespace)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="decorator", protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

        actual_target = actual(_pickle_target)
        expected_target = expected(_pickle_target)
        self.assertIs(actual_target, expected_target)
        self.assertIs(copy.copy(actual_target), actual_target)
        self.assertIs(copy.deepcopy(expected_target), expected_target)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="decorated", protocol=protocol):
                restored = pickle.loads(
                    pickle.dumps(actual_target, protocol=protocol)
                )
                self.assertIs(restored, actual_target)
                self.assertIs(restored._dynamo_marked_constant, True)
                self.assertEqual(restored(2, scale=4), 16)

    def test_invalid_target_errors_match_pytorch_2_13(self):
        class MethodOwner:
            def method(self):
                return "called"

        cases = (
            (lambda: None, lambda: None),
            (lambda: 1, lambda: 1),
            (lambda: len, lambda: len),
            (lambda: list, lambda: list),
            (lambda: _SlotCallable(), lambda: _SlotCallable()),
            (lambda: MethodOwner().method, lambda: MethodOwner().method),
            (lambda: _RejectingCallable(), lambda: _RejectingCallable()),
        )
        for case, (actual_target, expected_target) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda: torch.compiler.assume_constant_result(actual_target()),
                    lambda: reference_torch.compiler.assume_constant_result(
                        expected_target()
                    ),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.assume_constant_result
        expected = reference_torch.compiler.assume_constant_result
        actual_target = lambda: None
        expected_target = lambda: None
        cases = (
            (lambda: actual(), lambda: expected()),
            (
                lambda: actual(actual_target, actual_target),
                lambda: expected(expected_target, expected_target),
            ),
            (
                lambda: actual(target=actual_target),
                lambda: expected(target=expected_target),
            ),
            (
                lambda: actual(actual_target, fn=actual_target),
                lambda: expected(expected_target, fn=expected_target),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_queries_stay_eager_and_compilation_remains_unsupported(self):
        self.assertIs(torch.compiler.is_compiling(), False)
        self.assertIs(torch.compiler.is_dynamo_compiling(), False)
        self.assertIs(torch.compiler.is_exporting(), False)
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(hasattr(reference_torch, "export"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))

        unsupported = set(reference_torch.compiler.__all__) - {
            "assume_constant_result",
            "is_compiling",
            "is_dynamo_compiling",
            "is_exporting",
        }
        self.assertTrue(unsupported)
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.compiler, name))


if __name__ == "__main__":
    unittest.main()
