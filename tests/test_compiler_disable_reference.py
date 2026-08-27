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


def _reference_picklable_function(value):
    return value + 1


if reference_torch is not None:
    _reference_picklable_function = reference_torch.compiler.disable(
        _reference_picklable_function
    )


@torch.compiler.disable(recursive=False, reason="factory pickling test")
def _actual_factory_picklable_function(value):
    return value + 1


def _reference_factory_picklable_function(value):
    return value + 1


if reference_torch is not None:
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

    def test_factory_forms_and_method_binding_match_pytorch_2_13(self):
        outcomes = []
        reason = object()
        for module in (torch, reference_torch):
            default_factory = module.compiler.disable()
            configured_factory = module.compiler.disable(
                fn=None,
                recursive=False,
                reason=reason,
            )

            class Accumulator:
                def __init__(self):
                    self.total = 0

                def add(self, value):
                    self.total += value
                    return self.total

                def fail(self):
                    raise RuntimeError(self.total)

            original_add = Accumulator.add
            original_fail = Accumulator.fail
            Accumulator.add = default_factory(Accumulator.add)
            Accumulator.fail = configured_factory(Accumulator.fail)
            left = Accumulator()
            right = Accumulator()
            try:
                left.fail()
            except RuntimeError as error:
                failure = (type(error), error.args)
            else:
                failure = None
            outcomes.append(
                (
                    callable(default_factory),
                    callable(configured_factory),
                    isinstance(left.add, types.MethodType),
                    left.add.__self__ is left,
                    left.add.__func__ is Accumulator.add,
                    Accumulator.add.__wrapped__ is original_add,
                    Accumulator.add._torchdynamo_orig_callable is original_add,
                    Accumulator.add._torchdynamo_disable_recursive,
                    Accumulator.add._torchdynamo_disable_msg,
                    Accumulator.fail.__wrapped__ is original_fail,
                    Accumulator.fail._torchdynamo_orig_callable is original_fail,
                    Accumulator.fail._torchdynamo_disable_recursive,
                    Accumulator.fail._torchdynamo_disable_msg is reason,
                    left.add(2),
                    left.add(3),
                    right.add(7),
                    failure,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_factory_none_rejection_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            module_outcomes = []
            for recursive in (True, False):
                factory = module.compiler.disable(recursive=recursive)
                try:
                    factory(None)
                except Exception:
                    module_outcomes.append(True)
                else:
                    module_outcomes.append(False)
            outcomes.append(module_outcomes)

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], [True, True])

    def test_factory_recursive_snapshot_and_reuse_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            mutable_recursive = []
            mutable_factory = module.compiler.disable(
                recursive=mutable_recursive,
            )
            mutable_recursive.append(True)

            class StatefulTruthiness:
                def __init__(self):
                    self.calls = 0

                def __bool__(self):
                    self.calls += 1
                    return self.calls > 1

            stateful_recursive = StatefulTruthiness()
            stateful_factory = module.compiler.disable(
                recursive=stateful_recursive,
            )

            def first():
                return "first"

            def second():
                return "second"

            mutable_wrappers = (mutable_factory(first), mutable_factory(second))
            stateful_wrappers = (
                stateful_factory(first),
                stateful_factory(second),
            )
            outcomes.append(
                (
                    tuple(
                        wrapper._torchdynamo_disable_recursive
                        for wrapper in mutable_wrappers
                    ),
                    stateful_recursive.calls,
                    tuple(
                        wrapper._torchdynamo_disable_recursive
                        for wrapper in stateful_wrappers
                    ),
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], ((False, False), 1, (False, False)))

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

    def test_factory_repeated_wrapping_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):

            def function(value):
                return value + 2

            first = module.compiler.disable()(function)
            second = module.compiler.disable(
                recursive=False,
                reason="second",
            )(first)
            outcomes.append(
                (
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
            "keep_portable_guards_unsafe",
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

        for function in (
            _actual_picklable_function,
            _reference_picklable_function,
            _actual_factory_picklable_function,
            _reference_factory_picklable_function,
        ):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            self.assertEqual(function(4), 5)
            self.assertIs(function._torchdynamo_disable, True)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(wrapped=function.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol)), function
                    )

        for function in (_actual_picklable_function, _reference_picklable_function):
            self.assertIs(function._torchdynamo_disable_recursive, True)
        for function in (
            _actual_factory_picklable_function,
            _reference_factory_picklable_function,
        ):
            self.assertIs(function._torchdynamo_disable_recursive, False)
            self.assertEqual(
                function._torchdynamo_disable_msg,
                "factory pickling test",
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
