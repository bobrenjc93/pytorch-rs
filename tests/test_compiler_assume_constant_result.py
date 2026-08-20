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


class CompilerAssumeConstantResultTests(unittest.TestCase):
    def test_decorator_marks_and_returns_exact_function_without_wrapping(self):
        calls = []

        def target(value: int, offset=2, *, scale: int = 3) -> int:
            """Return a scaled offset."""
            calls.append((value, offset, scale))
            return (value + offset) * scale

        target.user_metadata = "preserved"
        identity = id(target)
        code = target.__code__
        signature = inspect.signature(target)
        annotations = target.__annotations__.copy()
        result = torch.compiler.assume_constant_result(target)

        self.assertIs(result, target)
        self.assertEqual(id(result), identity)
        self.assertIs(result.__code__, code)
        self.assertEqual(inspect.signature(result), signature)
        self.assertEqual(result.__annotations__, annotations)
        self.assertEqual(result.__name__, "target")
        self.assertIn("Return a scaled offset.", result.__doc__)
        self.assertEqual(result.user_metadata, "preserved")
        self.assertIs(result._dynamo_marked_constant, True)
        self.assertEqual(result(4, scale=5), 30)
        self.assertEqual(calls, [(4, 2, 5)])

    def test_decorator_syntax_callable_objects_and_idempotence(self):
        @torch.compiler.assume_constant_result
        def decorated(value):
            return value + 1

        self.assertIs(decorated._dynamo_marked_constant, True)
        self.assertEqual(decorated(4), 5)
        self.assertIs(torch.compiler.assume_constant_result(decorated), decorated)
        self.assertIs(torch.compiler.assume_constant_result(decorated), decorated)
        self.assertIs(decorated._dynamo_marked_constant, True)

        callable_target = _CallableTarget()
        result = torch.compiler.assume_constant_result(fn=callable_target)
        self.assertIs(result, callable_target)
        self.assertIs(callable_target._dynamo_marked_constant, True)
        self.assertEqual(result(1, key="value"), ((1,), {"key": "value"}))
        self.assertEqual(callable_target.calls, [((1,), {"key": "value"})])

    def test_assignment_is_unconditional_and_does_not_validate_callability(self):
        target = types.SimpleNamespace(_dynamo_marked_constant="old")

        result = torch.compiler.assume_constant_result(target)

        self.assertIs(result, target)
        self.assertIs(target._dynamo_marked_constant, True)
        self.assertFalse(callable(target))

    def test_signature_documentation_and_module_ownership(self):
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
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        compiler = torch.compiler
        function = compiler.assume_constant_result

        self.assertEqual(
            compiler.__all__,
            [
                "assume_constant_result",
                "is_compiling",
                "is_dynamo_compiling",
                "is_exporting",
            ],
        )
        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            {
                "assume_constant_result",
                "is_compiling",
                "is_dynamo_compiling",
                "is_exporting",
            },
        )
        self.assertIs(compiler_namespace["assume_constant_result"], function)

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("assume_constant_result", torch.__all__)
        self.assertFalse(hasattr(torch, "assume_constant_result"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("assume_constant_result", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="decorator", protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

        decorated = function(_pickle_target)
        self.assertIs(copy.copy(decorated), decorated)
        self.assertIs(copy.deepcopy(decorated), decorated)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="decorated", protocol=protocol):
                restored = pickle.loads(pickle.dumps(decorated, protocol=protocol))
                self.assertIs(restored, decorated)
                self.assertIs(restored._dynamo_marked_constant, True)
                self.assertEqual(restored(2, scale=4), 16)

    def test_invalid_targets_propagate_attribute_assignment_errors(self):
        cases = (
            (
                None,
                AttributeError,
                "'NoneType' object has no attribute '_dynamo_marked_constant'",
            ),
            (
                _SlotCallable(),
                AttributeError,
                "'_SlotCallable' object has no attribute '_dynamo_marked_constant'",
            ),
            (
                _RejectingCallable(),
                RuntimeError,
                "rejected _dynamo_marked_constant=True",
            ),
        )
        for target, error_type, message in cases:
            with self.subTest(target=type(target).__name__):
                with self.assertRaises(error_type) as raised:
                    torch.compiler.assume_constant_result(target)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.compiler.assume_constant_result
        target = lambda: None
        cases = (
            (
                lambda: function(),
                "assume_constant_result() missing 1 required positional argument: 'fn'",
            ),
            (
                lambda: function(target, target),
                "assume_constant_result() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(target=target),
                "assume_constant_result() got an unexpected keyword argument 'target'",
            ),
            (
                lambda: function(target, fn=target),
                "assume_constant_result() got multiple values for argument 'fn'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_compilation_and_graph_execution_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        for name in (
            "compile",
            "allow_in_graph",
            "disable",
            "list_backends",
            "reset",
            "substitute_in_graph",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.compiler, name))

    def test_importing_and_decorating_do_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

@torch.compiler.assume_constant_result
def answer(value):
    return value + 1

assert answer._dynamo_marked_constant is True
assert answer(41) == 42
assert torch.compiler.is_compiling() is False
assert torch.compiler.is_dynamo_compiling() is False
assert torch.compiler.is_exporting() is False
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
