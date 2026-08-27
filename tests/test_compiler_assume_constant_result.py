import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch


FUNCTION_DOC = """
    This function is used to mark a function `fn` as having a constant result.
    This allows the compiler to optimize away your function.
    Returns The same function `fn`

    Args:
        fn: The function to be marked as having a constant result.

    .. warning::
        `assume_constant_result` can if invalid cause safety and soundness issues, :func:`torch.compile`
        will not attempt to validate whether the constant assumption is true or not

    """


@torch.compiler.assume_constant_result
def _picklable_marked_function(value, *, increment=1):
    return value + increment


class _SlotCallable:
    __slots__ = ()

    def __call__(self):
        return "called"


class _RejectingCallable:
    def __setattr__(self, name, value):
        raise RuntimeError("attribute writes forbidden")

    def __call__(self):
        return "called"


class CompilerAssumeConstantResultTests(unittest.TestCase):
    def test_decorator_marks_the_original_function_without_changing_eager_calls(self):
        calls = []

        @torch.compiler.assume_constant_result
        def calculate(value, *, scale=1):
            calls.append((value, scale))
            return value * scale + len(calls)

        self.assertIs(calculate._dynamo_marked_constant, True)
        self.assertEqual(calculate(3, scale=2), 7)
        self.assertEqual(calculate(3, scale=2), 8)
        self.assertEqual(calls, [(3, 2), (3, 2)])
        self.assertEqual(str(inspect.signature(calculate)), "(value, *, scale=1)")
        self.assertEqual(calculate.__name__, "calculate")
        self.assertIn("<locals>.calculate", calculate.__qualname__)
        self.assertEqual(calculate.__module__, __name__)

    def test_decorator_preserves_method_binding_and_per_instance_state(self):
        class Accumulator:
            def __init__(self):
                self.total = 0

            @torch.compiler.assume_constant_result
            def add(self, value):
                self.total += value
                return self.total

        left = Accumulator()
        right = Accumulator()

        self.assertIs(Accumulator.add._dynamo_marked_constant, True)
        self.assertEqual(left.add(2), 2)
        self.assertEqual(left.add(3), 5)
        self.assertEqual(right.add(7), 7)

    def test_direct_positional_and_keyword_calls_return_the_exact_target(self):
        def function(value):
            return value + 1

        positional = torch.compiler.assume_constant_result(function)
        keyword = torch.compiler.assume_constant_result(fn=function)

        self.assertIs(positional, function)
        self.assertIs(keyword, function)
        self.assertIs(function._dynamo_marked_constant, True)
        self.assertEqual(function(4), 5)

        target = types.SimpleNamespace(existing="preserved")
        self.assertIs(torch.compiler.assume_constant_result(target), target)
        self.assertEqual(target.existing, "preserved")
        self.assertIs(target._dynamo_marked_constant, True)

    def test_repeated_marking_is_idempotent_and_overwrites_the_marker_with_true(self):
        def function():
            return "eager"

        sentinel = object()
        function._dynamo_marked_constant = sentinel
        function.other_metadata = "preserved"

        first = torch.compiler.assume_constant_result(function)
        second = torch.compiler.assume_constant_result(first)

        self.assertIs(first, function)
        self.assertIs(second, function)
        self.assertIs(function._dynamo_marked_constant, True)
        self.assertEqual(function.other_metadata, "preserved")
        self.assertEqual(function(), "eager")

    def test_invalid_targets_raise_attribute_assignment_errors(self):
        immutable_attribute_suffix = (
            " and no __dict__ for setting new attributes"
            if sys.version_info >= (3, 13)
            else ""
        )
        cases = (
            (
                None,
                "'NoneType' object has no attribute "
                f"'_dynamo_marked_constant'{immutable_attribute_suffix}",
            ),
            (
                1,
                "'int' object has no attribute "
                f"'_dynamo_marked_constant'{immutable_attribute_suffix}",
            ),
            (
                [],
                "'list' object has no attribute "
                f"'_dynamo_marked_constant'{immutable_attribute_suffix}",
            ),
            (
                len,
                "'builtin_function_or_method' object has no attribute "
                f"'_dynamo_marked_constant'{immutable_attribute_suffix}",
            ),
            (
                _SlotCallable(),
                "'_SlotCallable' object has no attribute "
                f"'_dynamo_marked_constant'{immutable_attribute_suffix}",
            ),
        )
        for target, message in cases:
            with self.subTest(target=target):
                with self.assertRaises(AttributeError) as raised:
                    torch.compiler.assume_constant_result(target)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        with self.assertRaisesRegex(RuntimeError, "^attribute writes forbidden$"):
            torch.compiler.assume_constant_result(_RejectingCallable())

    def test_signature_annotations_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.assume_constant_result

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(fn)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "assume_constant_result")
        self.assertEqual(function.__qualname__, "assume_constant_result")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copy_and_pickle_use_canonical_objects(self):
        compiler = torch.compiler
        decorator = compiler.assume_constant_result

        self.assertEqual(
            compiler.__all__,
            [
                "assume_constant_result",
                "reset",
                "disable",
                "set_default_backend",
                "get_default_backend",
                "is_compiling",
                "is_dynamo_compiling",
                "is_exporting",
                "keep_portable_guards_unsafe",
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
        self.assertNotIn("assume_constant_result", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("assume_constant_result", top_level_namespace)

        for function in (decorator, _picklable_marked_function):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )

        self.assertIs(_picklable_marked_function._dynamo_marked_constant, True)
        self.assertEqual(_picklable_marked_function(4, increment=3), 7)

    def test_call_shape_errors_match_pytorch_2_13(self):
        decorator = torch.compiler.assume_constant_result
        function = lambda: None
        cases = (
            (
                lambda: decorator(),
                "assume_constant_result() missing 1 required positional argument: 'fn'",
            ),
            (
                lambda: decorator(function, function),
                "assume_constant_result() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: decorator(function, fn=function),
                "assume_constant_result() got multiple values for argument 'fn'",
            ),
            (
                lambda: decorator(function, extra=True),
                "assume_constant_result() got an unexpected keyword argument 'extra'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_marking_does_not_enable_compilation_or_change_state_queries(self):
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
        self.assertFalse(hasattr(torch.compiler, "allow_in_graph"))

    def test_import_and_marking_do_not_import_pytorch(self):
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

@torch.compiler.assume_constant_result
def function(value):
    calls.append(value)
    return value + 1

assert function._dynamo_marked_constant is True
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
