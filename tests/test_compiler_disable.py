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
    This function provides a decorator to disable compilation on a function.
    It also provides the option of recursively disabling called functions.

    Args:
        fn (optional): The function to disable
        recursive (optional): A boolean value indicating whether the disabling should be recursive.
        reason (optional): A string value indicating the reason for disabling the function.
    """


@torch.compiler.disable
def _picklable_disabled_function(value, *, increment=1):
    return value + increment


class _CallableInstance:
    def __call__(self):
        return "called"


class CompilerDisableTests(unittest.TestCase):
    def assert_disable_metadata(self, wrapper, original, recursive, reason):
        self.assertIs(wrapper._torchdynamo_disable, True)
        self.assertIs(wrapper._torchdynamo_disable_msg, reason)
        self.assertIs(wrapper._torchdynamo_orig_callable, original)
        self.assertEqual(wrapper._torchdynamo_wrapper_id, id(wrapper))
        self.assertIs(wrapper._torchdynamo_disable_recursive, recursive)
        self.assertIs(wrapper.__wrapped__, original)

    def test_direct_call_wraps_function_without_changing_eager_behavior(self):
        calls = []

        def calculate(value: int, *, scale=1) -> int:
            """Calculate a stateful eager result."""
            calls.append((value, scale))
            return value * scale + len(calls)

        calculate.label = "preserved"
        wrapper = torch.compiler.disable(calculate)

        self.assertIsNot(wrapper, calculate)
        self.assertEqual(wrapper(3, scale=2), 7)
        self.assertEqual(wrapper(3, scale=2), 8)
        self.assertEqual(calls, [(3, 2), (3, 2)])
        self.assert_disable_metadata(wrapper, calculate, True, None)
        self.assertEqual(wrapper.label, "preserved")
        self.assertEqual(wrapper.__name__, calculate.__name__)
        self.assertEqual(wrapper.__qualname__, calculate.__qualname__)
        self.assertEqual(wrapper.__module__, calculate.__module__)
        self.assertEqual(wrapper.__doc__, calculate.__doc__)
        self.assertEqual(wrapper.__annotations__, calculate.__annotations__)
        self.assertEqual(inspect.signature(wrapper), inspect.signature(calculate))
        self.assertFalse(hasattr(calculate, "_torchdynamo_disable"))

    def test_recursive_and_reason_metadata_follow_pytorch_truthiness(self):
        reason = object()

        def function(value):
            return value

        recursive = torch.compiler.disable(
            function,
            recursive=[True],
            reason=reason,
        )
        nonrecursive = torch.compiler.disable(
            fn=function,
            recursive=[],
            reason=reason,
        )

        self.assert_disable_metadata(recursive, function, True, reason)
        self.assert_disable_metadata(nonrecursive, function, False, reason)
        self.assertEqual(recursive("recursive"), "recursive")
        self.assertEqual(nonrecursive("nonrecursive"), "nonrecursive")

    def test_repeated_disable_wraps_the_innermost_function(self):
        def function(value):
            return value + 1

        first = torch.compiler.disable(function, reason="first")
        first.outer_only = "not copied"
        second = torch.compiler.disable(first, recursive=False, reason="second")

        self.assertIsNot(second, first)
        self.assertIs(second.__wrapped__, function)
        self.assert_disable_metadata(second, function, False, "second")
        self.assertFalse(hasattr(second, "outer_only"))
        self.assertEqual(second(4), 5)

    def test_decorated_methods_keep_descriptor_binding(self):
        class Accumulator:
            def __init__(self):
                self.total = 0

            @torch.compiler.disable
            def add(self, value):
                self.total += value
                return self.total

        left = Accumulator()
        right = Accumulator()

        self.assertIs(left.add.__self__, left)
        self.assertIs(left.add.__func__, Accumulator.add)
        self.assert_disable_metadata(
            Accumulator.add,
            Accumulator.add.__wrapped__,
            True,
            None,
        )
        self.assertEqual(left.add(2), 2)
        self.assertEqual(left.add(3), 5)
        self.assertEqual(right.add(7), 7)

        bound = right.add
        rebound = torch.compiler.disable(bound, recursive=False, reason="bound")
        self.assert_disable_metadata(rebound, bound, False, "bound")
        self.assertEqual(rebound(1), 8)

    def test_copy_and_pickle_keep_canonical_wrappers(self):
        function = _picklable_disabled_function

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(function(4, increment=3), 7)
        self.assertIs(function._torchdynamo_disable, True)
        self.assertIs(function._torchdynamo_disable_recursive, True)

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

    def test_exports_use_the_canonical_compiler_function(self):
        compiler = torch.compiler

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
            ],
        )
        namespace = {}
        exec("from torch_rs.compiler import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")},
            set(compiler.__all__),
        )
        self.assertIs(namespace["disable"], compiler.disable)
        self.assertNotIn("disable", torch.__all__)

        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("disable", top_level_namespace)

        self.assertIs(copy.copy(compiler.disable), compiler.disable)
        self.assertIs(copy.deepcopy(compiler.disable), compiler.disable)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(compiler.disable, protocol)),
                    compiler.disable,
                )

    def test_unsupported_valid_pytorch_forms_are_rejected_without_mutation(self):
        message = (
            "torch_rs.compiler.disable only supports direct calls with a Python "
            "function"
        )
        for call in (
            lambda: torch.compiler.disable(),
            lambda: torch.compiler.disable(fn=None),
            lambda: torch.compiler.disable(len),
            lambda: torch.compiler.disable(_CallableInstance()),
        ):
            with self.subTest(call=call):
                with self.assertRaises(NotImplementedError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        class Target:
            def __init__(self):
                self.value = 1

            def __call__(self):
                return self.value

        original_init = Target.__init__
        original_call = Target.__call__
        with self.assertRaisesRegex(NotImplementedError, "^torch_rs\\.compiler"):
            torch.compiler.disable(Target)
        self.assertIs(Target.__init__, original_init)
        self.assertIs(Target.__call__, original_call)

        with self.assertRaises(AssertionError) as raised:
            torch.compiler.disable(1)
        self.assertEqual(str(raised.exception), "fn must be callable")
        self.assertEqual(raised.exception.args, ("fn must be callable",))

    def test_call_shape_errors_match_pytorch_2_13(self):
        function = lambda: None
        cases = (
            (
                lambda: torch.compiler.disable(function, True, "reason"),
                "disable() takes from 0 to 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.compiler.disable(function, True, recursive=False),
                "disable() got multiple values for argument 'recursive'",
            ),
            (
                lambda: torch.compiler.disable(function, extra=True),
                "disable() got an unexpected keyword argument 'extra'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_disabling_does_not_enable_compilation_or_import_pytorch(self):
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

def function(value):
    calls.append(value)
    return value + 1

disabled = torch.compiler.disable(function, recursive=False, reason="eager only")
assert disabled(1) == 2
assert disabled(2) == 3
assert calls == [1, 2]
assert disabled._torchdynamo_disable is True
assert disabled._torchdynamo_disable_recursive is False
assert disabled._torchdynamo_disable_msg == "eager only"
assert not hasattr(torch, "compile")
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
