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
def decorated_module_function(value, *, offset=0):
    return value + offset


class PicklableCallable:
    def __init__(self, offset):
        self.offset = offset

    def __call__(self, value):
        return value + self.offset


class SlottedCallable:
    __slots__ = ()

    def __call__(self):
        return "called"


class CompilerAssumeConstantResultTests(unittest.TestCase):
    def test_marks_and_returns_the_exact_function_without_wrapping_eager_calls(self):
        calls = []

        def function(value, *, offset=0):
            calls.append((value, offset))
            return value + offset + len(calls)

        original_signature = inspect.signature(function)
        original_name = function.__name__
        original_module = function.__module__
        marked = torch.compiler.assume_constant_result(function)

        self.assertIs(marked, function)
        self.assertIs(function._dynamo_marked_constant, True)
        self.assertEqual(inspect.signature(function), original_signature)
        self.assertEqual(function.__name__, original_name)
        self.assertEqual(function.__module__, original_module)
        self.assertEqual(function(3, offset=4), 8)
        self.assertEqual(function(3, offset=4), 9)
        self.assertEqual(calls, [(3, 4), (3, 4)])

    def test_decorator_syntax_callable_objects_and_idempotence(self):
        self.assertIs(decorated_module_function._dynamo_marked_constant, True)
        self.assertEqual(decorated_module_function(3, offset=4), 7)

        target = PicklableCallable(5)
        target._dynamo_marked_constant = False
        first = torch.compiler.assume_constant_result(target)
        second = torch.compiler.assume_constant_result(fn=target)

        self.assertIs(first, target)
        self.assertIs(second, target)
        self.assertIs(target._dynamo_marked_constant, True)
        self.assertEqual(target(7), 12)

        attribute_target = types.SimpleNamespace()
        self.assertIs(
            torch.compiler.assume_constant_result(attribute_target), attribute_target
        )
        self.assertIs(attribute_target._dynamo_marked_constant, True)

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
        self.assertIs(copy.copy(decorated_module_function), decorated_module_function)
        self.assertIs(
            copy.deepcopy(decorated_module_function), decorated_module_function
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

                decorated_payload = pickle.dumps(
                    decorated_module_function, protocol=protocol
                )
                self.assertIs(
                    pickle.loads(decorated_payload), decorated_module_function
                )

        target = torch.compiler.assume_constant_result(PicklableCallable(6))
        copied = copy.copy(target)
        deepcopied = copy.deepcopy(target)
        restored = pickle.loads(pickle.dumps(target))
        for candidate in (copied, deepcopied, restored):
            self.assertIs(candidate._dynamo_marked_constant, True)
            self.assertEqual(candidate(4), 10)

    def test_rejects_invalid_targets_with_pytorch_2_13_errors(self):
        immutable_attribute_suffix = (
            " and no __dict__ for setting new attributes"
            if sys.version_info >= (3, 14)
            else ""
        )
        cases = (
            (
                None,
                "'NoneType' object has no attribute '_dynamo_marked_constant'"
                + immutable_attribute_suffix,
            ),
            (
                1,
                "'int' object has no attribute '_dynamo_marked_constant'"
                + immutable_attribute_suffix,
            ),
            (
                len,
                "'builtin_function_or_method' object has no attribute "
                "'_dynamo_marked_constant'"
                + immutable_attribute_suffix,
            ),
            (
                SlottedCallable(),
                "'SlottedCallable' object has no attribute "
                "'_dynamo_marked_constant'"
                + immutable_attribute_suffix,
            ),
        )
        for target, message in cases:
            with self.subTest(target=target):
                with self.assertRaises(AttributeError) as raised:
                    torch.compiler.assume_constant_result(target)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_rejects_invalid_arguments_with_pytorch_2_13_errors(self):
        function = torch.compiler.assume_constant_result

        def target():
            return None

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
                lambda: function(target, fn=target),
                "assume_constant_result() got multiple values for argument 'fn'",
            ),
            (
                lambda: function(target, extra=True),
                "assume_constant_result() got an unexpected keyword argument 'extra'",
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
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "allow_in_graph"))
        self.assertIs(torch.compiler.is_compiling(), False)
        self.assertIs(torch.compiler.is_dynamo_compiling(), False)
        self.assertIs(torch.compiler.is_exporting(), False)

    def test_importing_and_marking_do_not_import_pytorch(self):
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
def eager(value):
    return value + 1

assert eager._dynamo_marked_constant is True
assert eager(2) == 3
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
