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


class SlottedCallable:
    __slots__ = ()

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

    def eager_outcome(self, module):
        calls = []

        def target(value, *, offset=0):
            calls.append((value, offset))
            return value + offset + len(calls)

        target._dynamo_marked_constant = False
        original_signature = inspect.signature(target)
        original_name = target.__name__
        original_qualname = target.__qualname__
        original_module = target.__module__
        first = module.compiler.assume_constant_result(target)
        first_marker = target._dynamo_marked_constant
        second = module.compiler.assume_constant_result(fn=target)
        attribute_target = types.SimpleNamespace()
        attribute_result = module.compiler.assume_constant_result(attribute_target)
        return (
            first is target,
            second is target,
            first_marker is True,
            target._dynamo_marked_constant is True,
            inspect.signature(target) == original_signature,
            target.__name__ == original_name,
            target.__qualname__ == original_qualname,
            target.__module__ == original_module,
            target(3, offset=4),
            target(3, offset=4),
            calls,
            attribute_result is attribute_target,
            attribute_target._dynamo_marked_constant is True,
        )

    def test_eager_identity_marking_calls_and_idempotence_match_pytorch_2_13(self):
        self.assertEqual(
            self.eager_outcome(torch),
            self.eager_outcome(reference_torch),
        )

    def test_decorator_syntax_and_callable_object_behavior_match(self):
        def outcome(module):
            @module.compiler.assume_constant_result
            def decorated(value):
                return value * 2

            class Callable:
                def __init__(self):
                    self.calls = 0

                def __call__(self, value):
                    self.calls += 1
                    return value + self.calls

            target = Callable()
            result = module.compiler.assume_constant_result(target)
            return (
                decorated._dynamo_marked_constant is True,
                decorated(4),
                result is target,
                target._dynamo_marked_constant is True,
                target(4),
                target(4),
                target.calls,
            )

        self.assertEqual(outcome(torch), outcome(reference_torch))

    def test_signature_annotations_documentation_and_identity_match(self):
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
            self.assertEqual(
                {name for name in namespace if not name.startswith("__")},
                supported if module is actual_compiler else set(module.__all__),
            )
            self.assertIs(
                namespace["assume_constant_result"],
                module.assume_constant_result,
            )

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
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_invalid_target_errors_match_pytorch_2_13(self):
        actual = torch.compiler.assume_constant_result
        expected = reference_torch.compiler.assume_constant_result
        cases = (
            (None, None),
            (1, 1),
            ("value", "value"),
            (len, len),
            (SlottedCallable(), SlottedCallable()),
        )
        for case, (actual_target, expected_target) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda: actual(actual_target),
                    lambda: expected(expected_target),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.assume_constant_result
        expected = reference_torch.compiler.assume_constant_result

        def actual_target():
            return None

        def expected_target():
            return None

        cases = (
            (lambda: actual(), lambda: expected()),
            (
                lambda: actual(actual_target, actual_target),
                lambda: expected(expected_target, expected_target),
            ),
            (
                lambda: actual(actual_target, fn=actual_target),
                lambda: expected(expected_target, fn=expected_target),
            ),
            (
                lambda: actual(actual_target, extra=True),
                lambda: expected(expected_target, extra=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_state_queries_stay_eager_and_graph_execution_stays_unsupported(self):
        self.assertIs(torch.compiler.is_compiling(), False)
        self.assertIs(torch.compiler.is_dynamo_compiling(), False)
        self.assertIs(torch.compiler.is_exporting(), False)
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "allow_in_graph"))

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
