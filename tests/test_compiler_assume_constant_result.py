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


def _picklable_constant_function(value):
    return value + 1


torch.compiler.assume_constant_result(_picklable_constant_function)


class CompilerAssumeConstantResultTests(unittest.TestCase):
    def test_plain_function_is_marked_in_place_and_remains_eager(self):
        sentinel = object()
        calls = []

        def function(value, *, option=sentinel):
            """function documentation"""
            calls.append((value, option))
            return len(calls), value, option

        function.custom_attribute = sentinel
        before = (
            function.__name__,
            function.__qualname__,
            function.__doc__,
            function.__annotations__.copy(),
            function.__defaults__,
            function.__kwdefaults__.copy(),
        )

        result = torch.compiler.assume_constant_result(function)

        self.assertIs(result, function)
        self.assertIs(function._dynamo_marked_constant, True)
        self.assertIs(function.custom_attribute, sentinel)
        self.assertEqual(
            (
                function.__name__,
                function.__qualname__,
                function.__doc__,
                function.__annotations__,
                function.__defaults__,
                function.__kwdefaults__,
            ),
            before,
        )
        self.assertEqual(function("first"), (1, "first", sentinel))
        self.assertEqual(function("second", option="value"), (2, "second", "value"))
        self.assertEqual(calls, [("first", sentinel), ("second", "value")])

    def test_methods_callable_instances_overwrites_and_idempotence(self):
        class Example:
            @torch.compiler.assume_constant_result
            def method(self, value):
                return value + 1

        raw_method = Example.__dict__["method"]
        self.assertIs(Example.method, raw_method)
        self.assertIs(raw_method._dynamo_marked_constant, True)
        self.assertEqual(Example().method(4), 5)

        class CallableTarget:
            def __init__(self):
                self.calls = []

            def __call__(self, value):
                self.calls.append(value)
                return len(self.calls), value * 2

        target = CallableTarget()
        previous_marker = object()
        target._dynamo_marked_constant = previous_marker

        first = torch.compiler.assume_constant_result(target)
        second = torch.compiler.assume_constant_result(first)

        self.assertIs(first, target)
        self.assertIs(second, target)
        self.assertIs(target._dynamo_marked_constant, True)
        self.assertEqual(target(3), (1, 6))
        self.assertEqual(target(5), (2, 10))
        self.assertEqual(target.calls, [3, 5])

    def test_writable_noncallable_follows_pytorch_assignment_semantics(self):
        class Target:
            pass

        target = Target()
        self.assertIs(torch.compiler.assume_constant_result(target), target)
        self.assertIs(target._dynamo_marked_constant, True)

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
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

        def keyword_target():
            return "eager result"

        self.assertIs(function(fn=keyword_target), keyword_target)
        self.assertIs(keyword_target._dynamo_marked_constant, True)
        self.assertEqual(keyword_target(), "eager result")

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        compiler = torch.compiler
        function = compiler.assume_constant_result
        supported = {
            "assume_constant_result",
            "is_compiling",
            "is_dynamo_compiling",
            "is_exporting",
        }

        self.assertEqual(
            compiler.__all__,
            [
                "assume_constant_result",
                "is_compiling",
                "is_dynamo_compiling",
                "is_exporting",
            ],
        )
        self.assertEqual(
            {name for name in vars(compiler) if not name.startswith("_")},
            supported,
        )
        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            supported,
        )
        for name in supported:
            self.assertIs(compiler_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("assume_constant_result", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("assume_constant_result", top_level_namespace)
        self.assertFalse(hasattr(torch, "assume_constant_result"))

        for value in (function, _picklable_constant_function):
            with self.subTest(value=value):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIs(pickle.loads(payload), value)

        self.assertIs(_picklable_constant_function._dynamo_marked_constant, True)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)

    def test_rejects_invalid_calls_with_pytorch_2_13_errors(self):
        function = torch.compiler.assume_constant_result
        immutable_attribute_suffix = (
            " and no __dict__ for setting new attributes"
            if sys.version_info >= (3, 14)
            else ""
        )

        class Example:
            def method(self):
                return None

        class SlottedCallable:
            __slots__ = ()

            def __call__(self):
                return None

        cases = (
            (
                lambda: function(),
                TypeError,
                "assume_constant_result() missing 1 required positional argument: 'fn'",
            ),
            (
                lambda: function(lambda: None, lambda: None),
                TypeError,
                "assume_constant_result() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(function=lambda: None),
                TypeError,
                "assume_constant_result() got an unexpected keyword argument "
                "'function'",
            ),
            (
                lambda: function(lambda: None, fn=lambda: None),
                TypeError,
                "assume_constant_result() got multiple values for argument 'fn'",
            ),
            (
                lambda: function(None),
                AttributeError,
                "'NoneType' object has no attribute "
                f"'_dynamo_marked_constant'{immutable_attribute_suffix}",
            ),
            (
                lambda: function(1),
                AttributeError,
                "'int' object has no attribute "
                f"'_dynamo_marked_constant'{immutable_attribute_suffix}",
            ),
            (
                lambda: function(len),
                AttributeError,
                "'builtin_function_or_method' object has no attribute "
                f"'_dynamo_marked_constant'{immutable_attribute_suffix}",
            ),
            (
                lambda: function(Example().method),
                AttributeError,
                "'method' object has no attribute "
                f"'_dynamo_marked_constant'{immutable_attribute_suffix}",
            ),
            (
                lambda: function(property()),
                AttributeError,
                "'property' object has no attribute "
                f"'_dynamo_marked_constant'{immutable_attribute_suffix}",
            ),
            (
                lambda: function(SlottedCallable()),
                AttributeError,
                "'SlottedCallable' object has no attribute "
                f"'_dynamo_marked_constant'{immutable_attribute_suffix}",
            ),
        )
        for call, exception_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(exception_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_compiler_queries_remain_eager_and_graph_apis_remain_unsupported(self):
        queries = (
            torch.compiler.is_compiling,
            torch.compiler.is_dynamo_compiling,
            torch.compiler.is_exporting,
        )
        self.assertEqual(tuple(query() for query in queries), (False, False, False))

        def function(value):
            return value + 1

        torch.compiler.assume_constant_result(function)
        self.assertEqual(function(2), 3)
        self.assertEqual(tuple(query() for query in queries), (False, False, False))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))

    def test_importing_and_using_the_decorator_does_not_import_pytorch(self):
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
    return len(calls)

assert function._dynamo_marked_constant is True
assert function(1) == 1
assert function(2) == 2
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
