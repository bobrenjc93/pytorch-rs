import copy
import importlib
import inspect
import pickle
import pickletools
import types
import typing
import unittest

import torch_rs as torch

if __package__:
    from .signature_utils import expose_reference_compiler_register_backend
else:
    from signature_utils import expose_reference_compiler_register_backend

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None

expose_reference_compiler_register_backend(reference_torch)


@torch.compiler.assume_constant_result
def _actual_picklable_function(value):
    return value + 1


def _reference_picklable_function(value):
    return value + 1


if reference_torch is not None:
    _reference_picklable_function = reference_torch.compiler.assume_constant_result(
        _reference_picklable_function
    )


class _SlotCallable:
    __slots__ = ()

    def __call__(self):
        return "called"


class _RejectingCallable:
    def __setattr__(self, name, value):
        raise RuntimeError("attribute writes forbidden")

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

    def decorator_outcome(self, module):
        calls = []

        @module.compiler.assume_constant_result
        def calculate(value, *, scale=1):
            calls.append((value, scale))
            return value * scale + len(calls)

        return (
            calculate._dynamo_marked_constant is True,
            calculate(3, scale=2),
            calculate(3, scale=2),
            calls,
            str(inspect.signature(calculate)),
            calculate.__name__,
            calculate.__module__,
        )

    def test_decorator_use_and_eager_behavior_match_pytorch_2_13(self):
        self.assertEqual(
            self.decorator_outcome(torch),
            self.decorator_outcome(reference_torch),
        )

    def test_direct_calls_and_noncallable_targets_match_pytorch_2_13(self):
        for module in (torch, reference_torch):
            with self.subTest(module=module.__name__):

                def function(value):
                    return value + 1

                positional = module.compiler.assume_constant_result(function)
                keyword = module.compiler.assume_constant_result(fn=function)
                self.assertIs(positional, function)
                self.assertIs(keyword, function)
                self.assertIs(function._dynamo_marked_constant, True)
                self.assertEqual(function(4), 5)

                target = types.SimpleNamespace(existing="preserved")
                self.assertIs(module.compiler.assume_constant_result(target), target)
                self.assertEqual(target.existing, "preserved")
                self.assertIs(target._dynamo_marked_constant, True)

    def test_repeated_marking_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):

            def function():
                return "eager"

            sentinel = object()
            function._dynamo_marked_constant = sentinel
            function.other_metadata = "preserved"
            first = module.compiler.assume_constant_result(function)
            second = module.compiler.assume_constant_result(first)
            outcomes.append(
                (
                    first is function,
                    second is function,
                    function._dynamo_marked_constant is True,
                    function.other_metadata,
                    function(),
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_invalid_target_errors_match_pytorch_2_13(self):
        target_factories = (
            lambda: None,
            lambda: 1,
            list,
            lambda: len,
            _SlotCallable,
            _RejectingCallable,
        )
        for case, target_factory in enumerate(target_factories):
            with self.subTest(case=case):
                actual_target = target_factory()
                expected_target = target_factory()
                self.assert_error_matches(
                    lambda: torch.compiler.assume_constant_result(actual_target),
                    lambda: reference_torch.compiler.assume_constant_result(
                        expected_target
                    ),
                )

    def test_signature_documentation_and_ownership_match_pytorch_2_13(self):
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

    def test_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.assume_constant_result
        expected = expected_compiler.assume_constant_result
        supported = {
            "assume_constant_result",
            "reset",
            "list_backends",
            "register_backend",
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
            torch.__all__.count("assume_constant_result"),
            reference_torch.__all__.count("assume_constant_result"),
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
            self.assertIs(function._dynamo_marked_constant, True)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            self.assertEqual(function(4), 5)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(marked=function.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol)), function
                    )

    def test_call_shape_errors_match_pytorch_2_13(self):
        actual = torch.compiler.assume_constant_result
        expected = reference_torch.compiler.assume_constant_result
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

    def test_state_queries_stay_eager_and_graph_execution_stays_unsupported(self):
        @torch.compiler.assume_constant_result
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
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(hasattr(reference_torch, "export"))


if __name__ == "__main__":
    unittest.main()
