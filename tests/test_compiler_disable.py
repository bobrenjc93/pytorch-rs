import copy
import functools
import importlib
import inspect
import pickle
import re
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch


FUNCTION_DOC = """
    This function provides a decorator to disable compilation on a function.
    It also provides the option of recursively disabling called functions.

    Args:
        fn (optional): The function to disable
        recursive (optional): A boolean value indicating whether the disabling should be recursive.
        reason (optional): A string value indicating the reason for disabling the function.
    """


def _picklable_function(value, *, increment=1):
    return value + increment


_picklable_function = torch.compiler.disable(
    _picklable_function,
    recursive=False,
    reason="pickling test",
)


@torch.compiler.disable(recursive=False, reason="factory pickling test")
def _factory_picklable_function(value, *, increment=1):
    return value + increment


class _Callable:
    def __call__(self):
        return "called"


class _UnsupportedClass:
    pass


class CompilerDisableTests(unittest.TestCase):
    def assert_disable_metadata(self, wrapper, original, recursive, reason):
        self.assertIs(wrapper._torchdynamo_disable, True)
        self.assertIs(wrapper._torchdynamo_disable_msg, reason)
        self.assertIs(wrapper._torchdynamo_orig_callable, original)
        self.assertEqual(wrapper._torchdynamo_wrapper_id, id(wrapper))
        self.assertIs(wrapper._torchdynamo_disable_recursive, recursive)
        self.assertIs(wrapper.__wrapped__, original)

    def test_direct_calls_wrap_functions_and_preserve_eager_execution(self):
        for recursive in (True, False):
            with self.subTest(recursive=recursive):
                calls = []
                reason = object()

                def calculate(value: int, *, scale: int = 1) -> int:
                    """Calculate eagerly."""
                    calls.append((value, scale))
                    return value * scale + len(calls)

                original = calculate
                shared_attribute = []
                original.custom_attribute = shared_attribute
                wrapped = torch.compiler.disable(
                    original,
                    recursive=recursive,
                    reason=reason,
                )

                self.assertIsNot(wrapped, original)
                self.assertEqual(wrapped(3, scale=2), 7)
                self.assertEqual(wrapped(3, scale=2), 8)
                self.assertEqual(calls, [(3, 2), (3, 2)])
                self.assert_disable_metadata(wrapped, original, recursive, reason)
                self.assertIs(inspect.unwrap(wrapped), original)
                self.assertIs(wrapped.custom_attribute, shared_attribute)
                self.assertFalse(hasattr(original, "_torchdynamo_disable"))
                self.assertEqual(
                    str(inspect.signature(wrapped)),
                    "(value: int, *, scale: int = 1) -> int",
                )
                self.assertEqual(wrapped.__name__, original.__name__)
                self.assertEqual(wrapped.__qualname__, original.__qualname__)
                self.assertEqual(wrapped.__module__, original.__module__)
                self.assertEqual(wrapped.__doc__, original.__doc__)
                self.assertEqual(wrapped.__annotations__, original.__annotations__)

    def test_wrapped_errors_and_return_values_are_transparent(self):
        sentinel = object()

        def return_sentinel():
            return sentinel

        class CustomError(Exception):
            pass

        error = CustomError("eager failure")

        def fail():
            raise error

        self.assertIs(torch.compiler.disable(return_sentinel)(), sentinel)
        with self.assertRaises(CustomError) as raised:
            torch.compiler.disable(fail)()
        self.assertIs(raised.exception, error)

        def positional_only(value, /, *, flag=False):
            return value, flag

        wrapped = torch.compiler.disable(positional_only)
        self.assertEqual(wrapped(3, flag=True), (3, True))
        with self.assertRaises(TypeError) as wrapped_error:
            wrapped(value=3)
        with self.assertRaises(TypeError) as original_error:
            positional_only(value=3)
        self.assertEqual(str(wrapped_error.exception), str(original_error.exception))
        self.assertEqual(wrapped_error.exception.args, original_error.exception.args)

    def test_decorated_methods_keep_descriptor_binding(self):
        class Accumulator:
            def __init__(self):
                self.total = 0

            @torch.compiler.disable
            def add(self, value):
                self.total += value
                return self.total

            @torch.compiler.disable
            def fail(self):
                raise RuntimeError(self.total)

            @classmethod
            @torch.compiler.disable
            def identify(cls, value):
                return cls, value

            @staticmethod
            @torch.compiler.disable
            def double(value):
                return value * 2

        left = Accumulator()
        right = Accumulator()

        self.assertIsInstance(left.add, types.MethodType)
        self.assertIs(left.add.__self__, left)
        self.assertIs(left.add.__func__, Accumulator.add)
        self.assertEqual(left.add(2), 2)
        self.assertEqual(left.add(3), 5)
        self.assertEqual(right.add(7), 7)
        self.assertEqual(Accumulator.identify("value"), (Accumulator, "value"))
        self.assertEqual(Accumulator.double(4), 8)
        with self.assertRaisesRegex(RuntimeError, "^5$"):
            left.fail()

        original = Accumulator.add.__wrapped__
        self.assert_disable_metadata(Accumulator.add, original, True, None)

    def test_factory_forms_wrap_functions_and_preserve_configuration(self):
        default_calls = []

        @torch.compiler.disable()
        def default(value):
            default_calls.append(value)
            return value + len(default_calls)

        reason = object()

        @torch.compiler.disable(fn=None, recursive=False, reason=reason)
        def configured(value):
            return value * 2

        self.assertEqual(default(3), 4)
        self.assertEqual(default(3), 5)
        self.assertEqual(default_calls, [3, 3])
        self.assert_disable_metadata(default, default.__wrapped__, True, None)
        self.assertEqual(configured(4), 8)
        self.assert_disable_metadata(
            configured,
            configured.__wrapped__,
            False,
            reason,
        )

    def test_factory_rejects_none_as_a_decorator_target(self):
        message = "torch.compiler.disable() currently supports only Python functions"
        for factory in (
            torch.compiler.disable(),
            torch.compiler.disable(recursive=False),
        ):
            with self.subTest(factory=factory):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    f"^{re.escape(message)}$",
                ):
                    factory(None)

    def test_factory_snapshots_recursive_truthiness_and_is_reusable(self):
        recursive = []
        reason = object()
        factory = torch.compiler.disable(recursive=recursive, reason=reason)
        recursive.append(True)

        def first():
            return "first"

        def second():
            return "second"

        first_wrapper = factory(first)
        second_wrapper = factory(second)
        self.assert_disable_metadata(first_wrapper, first, False, reason)
        self.assert_disable_metadata(second_wrapper, second, False, reason)

        class StatefulTruthiness:
            def __init__(self):
                self.calls = 0

            def __bool__(self):
                self.calls += 1
                return self.calls > 1

        stateful = StatefulTruthiness()
        stateful_factory = torch.compiler.disable(recursive=stateful)
        self.assertEqual(stateful.calls, 1)
        stateful_first = stateful_factory(first)
        stateful_second = stateful_factory(second)
        self.assertEqual(stateful.calls, 1)
        self.assertIs(stateful_first._torchdynamo_disable_recursive, False)
        self.assertIs(stateful_second._torchdynamo_disable_recursive, False)

    def test_factory_decorated_methods_keep_descriptor_binding(self):
        reason = object()

        class Accumulator:
            def __init__(self):
                self.total = 0

            @torch.compiler.disable()
            def add(self, value):
                self.total += value
                return self.total

            @torch.compiler.disable(recursive=False, reason=reason)
            def fail(self):
                raise RuntimeError(self.total)

        left = Accumulator()
        right = Accumulator()

        self.assertIsInstance(left.add, types.MethodType)
        self.assertIs(left.add.__self__, left)
        self.assertIs(left.add.__func__, Accumulator.add)
        self.assertEqual(left.add(2), 2)
        self.assertEqual(left.add(3), 5)
        self.assertEqual(right.add(7), 7)
        with self.assertRaisesRegex(RuntimeError, "^5$"):
            left.fail()

        self.assert_disable_metadata(
            Accumulator.add,
            Accumulator.add.__wrapped__,
            True,
            None,
        )
        self.assert_disable_metadata(
            Accumulator.fail,
            Accumulator.fail.__wrapped__,
            False,
            reason,
        )

    def test_direct_bound_method_and_repeated_wrapping_match_function_behavior(self):
        class Counter:
            def add(self, value):
                return value + 1

        counter = Counter()
        bound = counter.add
        wrapped_bound = torch.compiler.disable(
            bound,
            recursive=False,
            reason="bound",
        )
        self.assertEqual(wrapped_bound(3), 4)
        self.assert_disable_metadata(wrapped_bound, bound, False, "bound")

        def function(value):
            return value + 2

        first = torch.compiler.disable(function, reason="first")
        second = torch.compiler.disable(first, recursive=False, reason="second")
        self.assertIsNot(second, first)
        self.assertEqual(second(3), 5)
        self.assert_disable_metadata(second, function, False, "second")

        factory_first = torch.compiler.disable()(function)
        factory_second = torch.compiler.disable(
            recursive=False,
            reason="factory second",
        )(factory_first)
        self.assertIsNot(factory_second, factory_first)
        self.assertEqual(factory_second(3), 5)
        self.assert_disable_metadata(
            factory_second,
            function,
            False,
            "factory second",
        )

    def test_recursive_uses_truthiness_and_reason_is_preserved_by_identity(self):
        def function():
            return "eager"

        for recursive, expected in ((0, False), (1, True), ([], False), ([1], True)):
            reason = object()
            with self.subTest(recursive=recursive):
                wrapped = torch.compiler.disable(
                    fn=function,
                    recursive=recursive,
                    reason=reason,
                )
                self.assertIs(wrapped._torchdynamo_disable_recursive, expected)
                self.assertIs(wrapped._torchdynamo_disable_msg, reason)
                self.assertEqual(wrapped(), "eager")

    def test_signature_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.disable

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(fn=None, recursive=True, *, reason=None)",
        )
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "disable")
        self.assertEqual(function.__qualname__, "disable")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertEqual(function.__defaults__, (None, True))
        self.assertEqual(function.__kwdefaults__, {"reason": None})
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copying_and_pickling_use_canonical_objects(self):
        compiler = torch.compiler
        function = compiler.disable

        self.assertEqual(
            compiler.__all__,
            [
                "assume_constant_result",
                "reset",
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
            ],
        )
        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            set(compiler.__all__),
        )
        for name in compiler.__all__:
            self.assertIs(compiler_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("disable", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("disable", top_level_namespace)

        for copied_function in (
            function,
            _picklable_function,
            _factory_picklable_function,
        ):
            self.assertIs(copy.copy(copied_function), copied_function)
            self.assertIs(copy.deepcopy(copied_function), copied_function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(
                    function=copied_function.__name__, protocol=protocol
                ):
                    self.assertIs(
                        pickle.loads(pickle.dumps(copied_function, protocol)),
                        copied_function,
                    )

        self.assertEqual(_picklable_function(4, increment=3), 7)
        self.assertIs(_picklable_function._torchdynamo_disable, True)
        self.assertIs(_picklable_function._torchdynamo_disable_recursive, False)
        self.assertEqual(
            _picklable_function._torchdynamo_disable_msg,
            "pickling test",
        )
        self.assertEqual(_factory_picklable_function(4, increment=3), 7)
        self.assertIs(_factory_picklable_function._torchdynamo_disable, True)
        self.assertIs(
            _factory_picklable_function._torchdynamo_disable_recursive,
            False,
        )
        self.assertEqual(
            _factory_picklable_function._torchdynamo_disable_msg,
            "factory pickling test",
        )

    def test_supported_call_shape_errors_match_pytorch_2_13(self):
        disable = torch.compiler.disable
        function = lambda: None
        cases = (
            (
                lambda: disable(function, True, "reason"),
                "disable() takes from 0 to 2 positional arguments but 3 were given",
            ),
            (
                lambda: disable(function, fn=function),
                "disable() got multiple values for argument 'fn'",
            ),
            (
                lambda: disable(function, recursive=True, extra=True),
                "disable() got an unexpected keyword argument 'extra'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_factories_are_callable_but_not_context_managers(self):
        for factory in (
            torch.compiler.disable(),
            torch.compiler.disable(None),
            torch.compiler.disable(fn=None, recursive=False),
            torch.compiler.disable(recursive=False, reason="factory"),
        ):
            with self.subTest(factory=factory):
                self.assertTrue(callable(factory))
                self.assertFalse(hasattr(factory, "__enter__"))
                self.assertFalse(hasattr(factory, "__exit__"))
                with self.assertRaises(TypeError):
                    with factory:
                        pass

    def test_unsupported_targets_fail_without_mutation_through_both_forms(self):
        target_message = (
            "torch.compiler.disable() currently supports only Python functions"
        )
        original_init = _UnsupportedClass.__init__
        targets = (
            _UnsupportedClass,
            len,
            [].append,
            _Callable(),
            functools.partial(lambda: None),
            1,
        )
        for target in targets:
            for decorator in (
                torch.compiler.disable,
                torch.compiler.disable(),
                torch.compiler.disable(recursive=False, reason="factory"),
            ):
                with self.subTest(target=target, decorator=decorator):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        f"^{re.escape(target_message)}$",
                    ):
                        decorator(target)
        self.assertIs(_UnsupportedClass.__init__, original_init)

    def test_wrapping_does_not_enable_compilation_or_import_pytorch(self):
        @torch.compiler.disable
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

        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

calls = []

@torch.compiler.disable(recursive=False, reason="factory")
def function(value):
    calls.append(value)
    return value + 1

assert function._torchdynamo_disable is True
assert function._torchdynamo_disable_recursive is False
assert function._torchdynamo_disable_msg == "factory"
assert function(1) == 2
assert function(2) == 3
assert calls == [1, 2]
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
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


if __name__ == "__main__":
    unittest.main()
