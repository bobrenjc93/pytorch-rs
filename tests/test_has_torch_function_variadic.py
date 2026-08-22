import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = """Special case of `has_torch_function` that skips tuple creation.

    This uses the METH_FASTCALL protocol introduced in Python 3.7

    Instead of:
      `has_torch_function((a, b))`
    call:
      `has_torch_function_variadic(a, b)`
    which skips unnecessary packing and unpacking work.
    """


class HasTorchFunctionVariadicTests(unittest.TestCase):
    def test_zero_multiple_tensor_and_custom_arguments(self):
        function = torch.overrides.has_torch_function_variadic

        class ClassOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        class PlainOverride:
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        class StaticOverride:
            @staticmethod
            def __torch_function__(func, types, args=(), kwargs=None):
                return NotImplemented

        class NoneOverride:
            __torch_function__ = None

        class FalseOverride:
            __torch_function__ = False

        class MissingOverride:
            pass

        tensor = torch.tensor([1.0])
        cases = (
            ((), False),
            ((None,), False),
            ((None, 1, tensor, object()), False),
            ((tensor,), False),
            ((torch.Tensor,), True),
            ((None, torch.Tensor), True),
            ((torch.Tensor.__base__,), False),
            ((ClassOverride(),), True),
            ((None, ClassOverride), True),
            ((PlainOverride(),), True),
            ((StaticOverride(),), True),
            ((NoneOverride(),), True),
            ((FalseOverride(),), True),
            ((MissingOverride(),), False),
            (((ClassOverride(),),), False),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=repr(arguments)):
                self.assertIs(function(*arguments), expected)

    def test_descriptor_failures_and_short_circuiting(self):
        function = torch.overrides.has_torch_function_variadic
        events = []

        class Descriptor:
            def __init__(self, name, *, raises=False):
                self.name = name
                self.raises = raises

            def __get__(self, instance, owner):
                events.append((self.name, instance is None, owner.__name__))
                if self.raises:
                    raise RuntimeError(f"{self.name} failed")
                return None

        class BrokenFirst:
            __torch_function__ = Descriptor("broken-first", raises=True)

        class Override:
            __torch_function__ = Descriptor("override")

        class Unreachable:
            __torch_function__ = Descriptor("unreachable", raises=True)

        self.assertIs(
            function(object(), BrokenFirst(), Override(), Unreachable()),
            True,
        )
        self.assertEqual(
            events,
            [
                ("broken-first", False, "BrokenFirst"),
                ("override", False, "Override"),
            ],
        )

        events.clear()
        self.assertIs(function(Override(), Unreachable()), True)
        self.assertEqual(events, [("override", False, "Override")])

        events.clear()
        self.assertIs(function(BrokenFirst(), object()), False)
        self.assertEqual(events, [("broken-first", False, "BrokenFirst")])

        events.clear()
        self.assertIs(function(object(), Override), True)
        self.assertEqual(events, [("override", True, "Override")])

    def test_active_modes_require_an_argument_and_skip_descriptor_access(self):
        function = torch.overrides.has_torch_function_variadic
        tensor = torch.tensor([1.0])

        class RaisingDescriptor:
            def __init__(self):
                self.calls = 0

            def __get__(self, instance, owner):
                self.calls += 1
                raise RuntimeError("descriptor should not be read")

        descriptor = RaisingDescriptor()

        class BrokenOverride:
            __torch_function__ = descriptor

        class Mode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        self.assertIs(function(), False)
        self.assertIs(function(tensor, 1), False)
        self.assertIs(function(1, BrokenOverride()), False)
        self.assertEqual(descriptor.calls, 1)

        with Mode():
            self.assertIs(function(), False)
            self.assertIs(function(tensor), True)
            self.assertIs(function(1), True)
            self.assertIs(function(1, BrokenOverride()), True)
            self.assertEqual(descriptor.calls, 1)

        self.assertIs(function(tensor), False)

    def test_public_and_private_names_are_one_builtin_with_matching_metadata(self):
        function = torch.overrides.has_torch_function_variadic
        self.assertIs(function, torch._C._has_torch_function_variadic)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "_has_torch_function_variadic")
        self.assertEqual(function.__qualname__, "_has_torch_function_variadic")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertFalse(hasattr(function, "__annotations__"))
        self.assertEqual(
            repr(function),
            "<built-in function _has_torch_function_variadic>",
        )
        self.assertIs(function.__self__, torch._C)
        self.assertEqual(function.__reduce__(), "_has_torch_function_variadic")
        with self.assertRaises(ValueError):
            inspect.signature(function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

        self.assertIs(importlib.import_module("torch_rs._C"), torch._C)
        native_namespace = {}
        exec(
            "from torch_rs._C import _has_torch_function_variadic",
            native_namespace,
        )
        self.assertIs(native_namespace["_has_torch_function_variadic"], function)
        public_namespace = {}
        exec(
            "from torch_rs.overrides import has_torch_function_variadic",
            public_namespace,
        )
        self.assertIs(public_namespace["has_torch_function_variadic"], function)

    def test_probe_is_not_wildcard_or_top_level_exported(self):
        function = torch.overrides.has_torch_function_variadic
        self.assertNotIn("has_torch_function_variadic", torch.overrides.__all__)
        self.assertNotIn("_has_torch_function_variadic", torch._C.__all__)
        self.assertNotIn("has_torch_function_variadic", torch.__all__)
        self.assertNotIn("_has_torch_function_variadic", torch.__all__)
        self.assertFalse(hasattr(torch, "has_torch_function_variadic"))
        self.assertFalse(hasattr(torch, "_has_torch_function_variadic"))

        overrides_namespace = {}
        exec("from torch_rs.overrides import *", overrides_namespace)
        self.assertNotIn("has_torch_function_variadic", overrides_namespace)
        native_namespace = {}
        exec("from torch_rs._C import *", native_namespace)
        self.assertNotIn("_has_torch_function_variadic", native_namespace)

        self.assertFalse(hasattr(torch.overrides, "has_torch_function"))
        self.assertFalse(hasattr(torch._C, "_has_torch_function"))
        self.assertIs(torch.overrides.has_torch_function_variadic, function)

    def test_keyword_errors_match_pytorch_2_13(self):
        function = torch.overrides.has_torch_function_variadic
        cases = (
            lambda: function(input=None),
            lambda: function(objects=()),
            lambda: function(None, unexpected=True),
        )
        message = (
            "torch._C._has_torch_function_variadic() takes no keyword arguments"
        )
        for call in cases:
            with self.subTest(call=call):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(**{}), False)
        self.assertIs(function(None, **{}), False)

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

function = torch.overrides.has_torch_function_variadic
assert function is torch._C._has_torch_function_variadic
assert function() is False
assert function(None, torch.tensor([1.0])) is False
assert function(None, torch.Tensor) is True

class Override:
    __torch_function__ = None

assert function(None, Override()) is True
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
