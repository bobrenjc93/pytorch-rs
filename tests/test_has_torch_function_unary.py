import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = """Special case of `has_torch_function` for single inputs.
    Instead of:
      `has_torch_function((t,))`
    call:
      `has_torch_function_unary(t)`
    which skips unnecessary packing and unpacking work.
    """


class HasTorchFunctionUnaryTests(unittest.TestCase):
    def test_exact_tensors_tensor_classes_and_custom_overrides(self):
        function = torch.overrides.has_torch_function_unary

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

        cases = (
            (torch.tensor([1.0]), False),
            (torch.Tensor, True),
            (torch.Tensor.__base__, False),
            (ClassOverride(), True),
            (ClassOverride, True),
            (PlainOverride(), True),
            (StaticOverride(), True),
            (NoneOverride(), True),
            (FalseOverride(), True),
            (MissingOverride(), False),
            (None, False),
            (1, False),
            (1.0, False),
            ([], False),
            ((), False),
            ({}, False),
            (object(), False),
            (int, False),
        )
        for value, expected in cases:
            with self.subTest(value=repr(value)):
                self.assertIs(function(value), expected)

    def test_descriptor_lookup_is_single_and_failures_are_suppressed(self):
        function = torch.overrides.has_torch_function_unary

        class RecordingDescriptor:
            def __init__(self):
                self.calls = []

            def __get__(self, instance, owner):
                self.calls.append((instance, owner))
                return None

        descriptor = RecordingDescriptor()

        class Override:
            __torch_function__ = descriptor

        instance = Override()
        self.assertIs(function(instance), True)
        self.assertEqual(descriptor.calls, [(instance, Override)])

        descriptor.calls.clear()
        self.assertIs(function(Override), True)
        self.assertEqual(len(descriptor.calls), 1)

        class RaisingDescriptor:
            def __init__(self):
                self.calls = 0

            def __get__(self, instance, owner):
                self.calls += 1
                raise RuntimeError("descriptor failed")

        raising_descriptor = RaisingDescriptor()

        class BrokenOverride:
            __torch_function__ = raising_descriptor

        self.assertIs(function(BrokenOverride()), False)
        self.assertIs(function(BrokenOverride), False)
        self.assertEqual(raising_descriptor.calls, 2)

        class DynamicOverride:
            def __getattribute__(self, name):
                if name == "__torch_function__":
                    return None
                return object.__getattribute__(self, name)

        class BrokenDynamicOverride:
            def __getattribute__(self, name):
                if name == "__torch_function__":
                    raise RuntimeError("dynamic lookup failed")
                return object.__getattribute__(self, name)

        self.assertIs(function(DynamicOverride()), True)
        self.assertIs(function(BrokenDynamicOverride()), False)

    def test_active_modes_make_every_input_relevant_without_descriptor_access(self):
        function = torch.overrides.has_torch_function_unary
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

        self.assertIs(function(tensor), False)
        self.assertIs(function(1), False)
        self.assertIs(function(BrokenOverride()), False)
        self.assertEqual(descriptor.calls, 1)

        with Mode():
            self.assertIs(function(tensor), True)
            self.assertIs(function(torch.Tensor), True)
            self.assertIs(function(1), True)
            self.assertIs(function(BrokenOverride()), True)
            self.assertEqual(descriptor.calls, 1)

        self.assertIs(function(tensor), False)

    def test_public_and_private_names_are_one_builtin_with_matching_metadata(self):
        function = torch.overrides.has_torch_function_unary
        self.assertIs(function, torch._C._has_torch_function_unary)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "_has_torch_function_unary")
        self.assertEqual(function.__qualname__, "_has_torch_function_unary")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertFalse(hasattr(function, "__annotations__"))
        self.assertEqual(
            repr(function),
            "<built-in function _has_torch_function_unary>",
        )
        self.assertIs(function.__self__, torch._C)
        self.assertEqual(function.__reduce__(), "_has_torch_function_unary")

        if sys.version_info >= (3, 13):
            self.assertEqual(function.__text_signature__, "($self, object, /)")
            self.assertEqual(str(inspect.signature(function)), "(object, /)")
        else:
            self.assertIsNone(function.__text_signature__)
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
            "from torch_rs._C import _has_torch_function_unary",
            native_namespace,
        )
        self.assertIs(native_namespace["_has_torch_function_unary"], function)
        public_namespace = {}
        exec(
            "from torch_rs.overrides import has_torch_function_unary",
            public_namespace,
        )
        self.assertIs(public_namespace["has_torch_function_unary"], function)

    def test_probe_is_not_wildcard_exported_and_broader_probes_stay_unsupported(self):
        function = torch.overrides.has_torch_function_unary
        self.assertNotIn("has_torch_function_unary", torch.overrides.__all__)
        self.assertNotIn("_has_torch_function_unary", torch._C.__all__)
        self.assertNotIn("has_torch_function_unary", torch.__all__)
        self.assertNotIn("_has_torch_function_unary", torch.__all__)
        self.assertFalse(hasattr(torch, "has_torch_function_unary"))
        self.assertFalse(hasattr(torch, "_has_torch_function_unary"))

        overrides_namespace = {}
        exec("from torch_rs.overrides import *", overrides_namespace)
        self.assertNotIn("has_torch_function_unary", overrides_namespace)
        native_namespace = {}
        exec("from torch_rs._C import *", native_namespace)
        self.assertNotIn("_has_torch_function_unary", native_namespace)

        for module, names in (
            (
                torch.overrides,
                ("has_torch_function", "has_torch_function_variadic"),
            ),
            (
                torch._C,
                ("_has_torch_function", "_has_torch_function_variadic"),
            ),
        ):
            for name in names:
                with self.subTest(module=module.__name__, name=name):
                    self.assertFalse(hasattr(module, name))
                    self.assertNotIn(name, getattr(module, "__all__", ()))

        self.assertIs(torch.overrides.has_torch_function_unary, function)

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.overrides.has_torch_function_unary
        cases = (
            (
                lambda: function(),
                "torch._C._has_torch_function_unary() takes exactly one argument (0 given)",
            ),
            (
                lambda: function(None, None),
                "torch._C._has_torch_function_unary() takes exactly one argument (2 given)",
            ),
            (
                lambda: function(input=None),
                "torch._C._has_torch_function_unary() takes no keyword arguments",
            ),
            (
                lambda: function(None, unexpected=True),
                "torch._C._has_torch_function_unary() takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

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

function = torch.overrides.has_torch_function_unary
assert function is torch._C._has_torch_function_unary
assert function(torch.tensor([1.0])) is False
assert function(torch.Tensor) is True

class Override:
    __torch_function__ = None

assert function(Override()) is True
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
