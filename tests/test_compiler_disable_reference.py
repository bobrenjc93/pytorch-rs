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


def _actual_picklable_function(value):
    return value + 1


_actual_picklable_function = torch.compiler.disable(_actual_picklable_function)


@torch.compiler.disable(recursive=False, reason="factory pickling test")
def _actual_factory_picklable_function(value):
    return value + 1


def _reference_picklable_function(value):
    return value + 1


def _reference_factory_picklable_function(value):
    return value + 1


if reference_torch is not None:
    _reference_picklable_function = reference_torch.compiler.disable(
        _reference_picklable_function
    )
    _reference_factory_picklable_function = reference_torch.compiler.disable(
        recursive=False,
        reason="factory pickling test",
    )(_reference_factory_picklable_function)


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
                argument = argument.replace("_actual_", "_")
                argument = argument.replace("_reference_", "_")
            shape.append((opcode.name, argument))
        return shape

    def wrapped_outcome(self, module, recursive):
        calls = []
        reason = object()

        def calculate(value: int, *, scale: int = 1) -> int:
            """Calculate eagerly."""
            calls.append((value, scale))
            return value * scale + len(calls)

        original = calculate
        shared_attribute = []
        original.custom_attribute = shared_attribute
        wrapped = module.compiler.disable(
            original,
            recursive=recursive,
            reason=reason,
        )
        metadata = (
            wrapped._torchdynamo_disable is True,
            wrapped._torchdynamo_disable_msg is reason,
            wrapped._torchdynamo_orig_callable is original,
            wrapped._torchdynamo_wrapper_id == id(wrapped),
            wrapped._torchdynamo_disable_recursive is recursive,
            wrapped.__wrapped__ is original,
        )
        reflection = (
            type(wrapped) is types.FunctionType,
            wrapped is not original,
            inspect.unwrap(wrapped) is original,
            wrapped.custom_attribute is shared_attribute,
            not hasattr(original, "_torchdynamo_disable"),
            str(inspect.signature(wrapped)),
            wrapped.__name__,
            wrapped.__qualname__,
            wrapped.__module__.split(".")[-1],
            wrapped.__doc__,
            wrapped.__annotations__ is original.__annotations__,
        )
        values = (wrapped(3, scale=2), wrapped(3, scale=2), calls)
        return metadata, reflection, values

    def test_direct_wrapping_and_eager_calls_match_pytorch_2_13(self):
        for recursive in (True, False):
            with self.subTest(recursive=recursive):
                self.assertEqual(
                    self.wrapped_outcome(torch, recursive),
                    self.wrapped_outcome(reference_torch, recursive),
                )

    def factory_wrapped_outcome(self, module, recursive, explicit_none):
        calls = []
        reason = object()
        if explicit_none:
            decorator = module.compiler.disable(
                fn=None,
                recursive=recursive,
                reason=reason,
            )
        else:
            decorator = module.compiler.disable(
                recursive=recursive,
                reason=reason,
            )

        def calculate(value: int, *, scale: int = 1) -> int:
            """Calculate eagerly."""
            calls.append((value, scale))
            return value * scale + len(calls)

        original = calculate
        shared_attribute = []
        original.custom_attribute = shared_attribute
        wrapped = decorator(original)
        metadata = (
            wrapped._torchdynamo_disable is True,
            wrapped._torchdynamo_disable_msg is reason,
            wrapped._torchdynamo_orig_callable is original,
            wrapped._torchdynamo_wrapper_id == id(wrapped),
            wrapped._torchdynamo_disable_recursive is recursive,
            wrapped.__wrapped__ is original,
        )
        reflection = (
            callable(decorator),
            type(wrapped) is types.FunctionType,
            wrapped is not original,
            inspect.unwrap(wrapped) is original,
            wrapped.custom_attribute is shared_attribute,
            not hasattr(original, "_torchdynamo_disable"),
            str(inspect.signature(wrapped)),
            wrapped.__name__,
            wrapped.__qualname__,
            wrapped.__module__.split(".")[-1],
            wrapped.__doc__,
            wrapped.__annotations__ is original.__annotations__,
        )
        values = (wrapped(3, scale=2), wrapped(3, scale=2), calls)
        return metadata, reflection, values

    def test_factory_wrapping_and_eager_calls_match_pytorch_2_13(self):
        for recursive in (True, False):
            for explicit_none in (True, False):
                with self.subTest(
                    recursive=recursive,
                    explicit_none=explicit_none,
                ):
                    self.assertEqual(
                        self.factory_wrapped_outcome(
                            torch,
                            recursive,
                            explicit_none,
                        ),
                        self.factory_wrapped_outcome(
                            reference_torch,
                            recursive,
                            explicit_none,
                        ),
                    )

    def test_factory_recursive_truthiness_is_fixed_at_creation(self):
        outcomes = []
        for module in (torch, reference_torch):
            truthy_recursive = [1]
            falsey_recursive = []
            truthy_factory = module.compiler.disable(recursive=truthy_recursive)
            falsey_factory = module.compiler.disable(recursive=falsey_recursive)
            truthy_recursive.clear()
            falsey_recursive.append(1)

            def first():
                return "first"

            def second():
                return "second"

            truthy_wrapped = truthy_factory(first)
            falsey_wrapped = falsey_factory(second)
            outcomes.append(
                (
                    truthy_wrapped(),
                    truthy_wrapped._torchdynamo_disable_recursive,
                    falsey_wrapped(),
                    falsey_wrapped._torchdynamo_disable_recursive,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_method_binding_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):

            class Accumulator:
                def __init__(self):
                    self.total = 0

                def add(self, value):
                    self.total += value
                    return self.total

            original = Accumulator.add
            Accumulator.add = module.compiler.disable(
                Accumulator.add,
                recursive=False,
                reason="method",
            )
            left = Accumulator()
            right = Accumulator()
            outcomes.append(
                (
                    isinstance(left.add, types.MethodType),
                    left.add.__self__ is left,
                    left.add.__func__ is Accumulator.add,
                    Accumulator.add.__wrapped__ is original,
                    Accumulator.add._torchdynamo_orig_callable is original,
                    Accumulator.add._torchdynamo_disable_recursive is False,
                    Accumulator.add._torchdynamo_disable_msg,
                    left.add(2),
                    left.add(3),
                    right.add(7),
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_factory_method_binding_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):

            class Accumulator:
                def __init__(self):
                    self.total = 0

                def add(self, value):
                    self.total += value
                    return self.total

            original = Accumulator.add
            Accumulator.add = module.compiler.disable(
                recursive=False,
                reason="method",
            )(Accumulator.add)
            left = Accumulator()
            right = Accumulator()
            outcomes.append(
                (
                    isinstance(left.add, types.MethodType),
                    left.add.__self__ is left,
                    left.add.__func__ is Accumulator.add,
                    Accumulator.add.__wrapped__ is original,
                    Accumulator.add._torchdynamo_orig_callable is original,
                    Accumulator.add._torchdynamo_disable_recursive is False,
                    Accumulator.add._torchdynamo_disable_msg,
                    left.add(2),
                    left.add(3),
                    right.add(7),
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_direct_bound_methods_and_repeated_wrapping_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):

            class Counter:
                def add(self, value):
                    return value + 1

            counter = Counter()
            bound = counter.add
            wrapped_bound = module.compiler.disable(bound, recursive=False)

            def function(value):
                return value + 2

            first = module.compiler.disable(function, reason="first")
            second = module.compiler.disable(
                first,
                recursive=False,
                reason="second",
            )
            outcomes.append(
                (
                    wrapped_bound(3),
                    wrapped_bound.__wrapped__ is bound,
                    wrapped_bound._torchdynamo_orig_callable is bound,
                    second(3),
                    second is not first,
                    second.__wrapped__ is function,
                    second._torchdynamo_orig_callable is function,
                    second._torchdynamo_disable_msg,
                    second._torchdynamo_disable_recursive,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_factory_bound_methods_and_repeated_wrapping_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):

            class Counter:
                def add(self, value):
                    return value + 1

            decorator = module.compiler.disable(
                recursive=False,
                reason="factory",
            )
            counter = Counter()
            bound = counter.add
            wrapped_bound = decorator(bound)

            def function(value):
                return value + 2

            first = decorator(function)
            second = decorator(first)
            outcomes.append(
                (
                    wrapped_bound(3),
                    wrapped_bound.__wrapped__ is bound,
                    wrapped_bound._torchdynamo_orig_callable is bound,
                    second(3),
                    second is not first,
                    second.__wrapped__ is function,
                    second._torchdynamo_orig_callable is function,
                    second._torchdynamo_disable_msg,
                    second._torchdynamo_disable_recursive,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_truthiness_and_reason_metadata_match_pytorch_2_13(self):
        for recursive in (0, 1, [], [1], None, "recursive"):
            reason = object()
            outcomes = []
            for module in (torch, reference_torch):

                def function():
                    return "eager"

                wrapped = module.compiler.disable(
                    function,
                    recursive=recursive,
                    reason=reason,
                )
                outcomes.append(
                    (
                        wrapped(),
                        wrapped._torchdynamo_disable_recursive,
                        type(wrapped._torchdynamo_disable_recursive),
                        wrapped._torchdynamo_disable_msg is reason,
                    )
                )

            with self.subTest(recursive=recursive):
                self.assertEqual(outcomes[0], outcomes[1])

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
        actual = actual_compiler.disable
        expected = expected_compiler.disable
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
            torch.__all__.count("compiler"),
            reference_torch.__all__.count("compiler"),
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
            self.assertIs(function._torchdynamo_disable, True)
            self.assertIs(function._torchdynamo_disable_recursive, True)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(wrapped=function.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol)), function
                    )

        factory_functions = (
            _actual_factory_picklable_function,
            _reference_factory_picklable_function,
        )
        for function in factory_functions:
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            self.assertEqual(function(4), 5)
            self.assertIs(function._torchdynamo_disable, True)
            self.assertIs(function._torchdynamo_disable_recursive, False)
            self.assertEqual(
                function._torchdynamo_disable_msg,
                "factory pickling test",
            )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(factory_wrapped=True, protocol=protocol):
                actual_loaded = pickle.loads(
                    pickle.dumps(_actual_factory_picklable_function, protocol)
                )
                expected_loaded = pickle.loads(
                    pickle.dumps(_reference_factory_picklable_function, protocol)
                )
                self.assertIs(actual_loaded, _actual_factory_picklable_function)
                self.assertIs(expected_loaded, _reference_factory_picklable_function)
                self.assertEqual(
                    self.pickle_shape(
                        _actual_factory_picklable_function,
                        protocol,
                    ),
                    self.pickle_shape(
                        _reference_factory_picklable_function,
                        protocol,
                    ),
                )

    def test_factory_wrapped_errors_match_pytorch_2_13(self):
        def make_wrapped(module, recursive):
            def positional_only(value, /):
                return value

            return module.compiler.disable(recursive=recursive)(positional_only)

        for recursive in (True, False):
            actual_wrapped = make_wrapped(torch, recursive)
            expected_wrapped = make_wrapped(reference_torch, recursive)
            with self.subTest(recursive=recursive):
                self.assert_error_matches(
                    lambda: actual_wrapped(value=1),
                    lambda: expected_wrapped(value=1),
                )

    def test_supported_errors_match_pytorch_2_13(self):
        actual = torch.compiler.disable
        expected = reference_torch.compiler.disable
        actual_function = lambda value, /: value
        expected_function = lambda value, /: value
        cases = (
            (
                lambda: actual(actual_function, True, "reason"),
                lambda: expected(expected_function, True, "reason"),
            ),
            (
                lambda: actual(actual_function, fn=actual_function),
                lambda: expected(expected_function, fn=expected_function),
            ),
            (
                lambda: actual(actual_function, recursive=True, extra=True),
                lambda: expected(expected_function, recursive=True, extra=True),
            ),
            (
                lambda: actual(actual_function)(value=1),
                lambda: expected(expected_function)(value=1),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_compilation_stays_unsupported_and_eager_state_stays_false(self):
        @torch.compiler.disable
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
