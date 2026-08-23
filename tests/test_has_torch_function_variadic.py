import copy
import ctypes
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
    def test_zero_multiple_exact_tensor_and_custom_override_arguments(self):
        function = torch.overrides.has_torch_function_variadic
        tensor = torch.tensor([1.0])

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
            ((), False),
            ((tensor,), False),
            ((tensor, tensor, None, 1), False),
            ((torch.Tensor,), True),
            ((None, torch.Tensor, tensor), True),
            ((torch.Tensor.__base__,), False),
            ((ClassOverride(),), True),
            ((None, ClassOverride), True),
            ((PlainOverride(),), True),
            ((StaticOverride(),), True),
            ((NoneOverride(),), True),
            ((FalseOverride(),), True),
            ((MissingOverride(), None, 1, 1.0, [], (), {}, object(), int), False),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=repr(arguments)):
                self.assertIs(function(*arguments), expected)

    def test_descriptor_failures_are_suppressed_and_probes_short_circuit(self):
        function = torch.overrides.has_torch_function_variadic
        events = []

        class Descriptor:
            def __init__(self, name, raises=False):
                self.name = name
                self.raises = raises

            def __get__(self, instance, owner):
                events.append((self.name, instance is None, owner.__name__))
                if self.raises:
                    raise RuntimeError(f"{self.name} failed")
                return None

        class Broken:
            __torch_function__ = Descriptor("broken", raises=True)

        class Override:
            __torch_function__ = Descriptor("override")

        class Skipped:
            __torch_function__ = Descriptor("skipped", raises=True)

        self.assertIs(function(Broken(), 1, Override(), Skipped()), True)
        self.assertEqual(
            events,
            [
                ("broken", False, "Broken"),
                ("override", False, "Override"),
            ],
        )

        events.clear()
        self.assertIs(function(Override(), Skipped(), Broken()), True)
        self.assertEqual(events, [("override", False, "Override")])

        events.clear()
        self.assertIs(function(Broken(), Broken), False)
        self.assertEqual(
            events,
            [
                ("broken", False, "Broken"),
                ("broken", True, "Broken"),
            ],
        )

        class DynamicOverride:
            def __getattribute__(self, name):
                if name == "__torch_function__":
                    events.append(("dynamic", False, "DynamicOverride"))
                    return None
                return object.__getattribute__(self, name)

        class BrokenDynamicOverride:
            def __getattribute__(self, name):
                if name == "__torch_function__":
                    events.append(("broken_dynamic", False, "BrokenDynamicOverride"))
                    raise RuntimeError("dynamic lookup failed")
                return object.__getattribute__(self, name)

        events.clear()
        self.assertIs(
            function(BrokenDynamicOverride(), DynamicOverride(), Skipped()),
            True,
        )
        self.assertEqual(
            events,
            [
                ("broken_dynamic", False, "BrokenDynamicOverride"),
                ("dynamic", False, "DynamicOverride"),
            ],
        )

    def test_active_modes_make_each_nonempty_call_relevant_without_lookup(self):
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

        broken = BrokenOverride()
        self.assertIs(function(), False)
        self.assertIs(function(tensor, 1), False)
        self.assertIs(function(broken), False)
        self.assertEqual(descriptor.calls, 1)

        with Mode():
            self.assertIs(function(), False)
            self.assertIs(function(tensor), True)
            self.assertIs(function(torch.Tensor), True)
            self.assertIs(function(torch.Tensor.__base__), True)
            self.assertIs(function(1, None, broken), True)
            self.assertEqual(descriptor.calls, 1)

        self.assertIs(function(tensor, 1), False)
        self.assertIs(function(broken), False)
        self.assertEqual(descriptor.calls, 2)

    def test_public_and_private_names_are_one_fastcall_builtin(self):
        function = torch.overrides.has_torch_function_variadic
        self.assertIs(function, torch._C._has_torch_function_variadic)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "_has_torch_function_variadic")
        self.assertEqual(function.__qualname__, "_has_torch_function_variadic")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertFalse(hasattr(function, "__annotations__"))
        self.assertEqual(
            repr(function),
            "<built-in function _has_torch_function_variadic>",
        )
        self.assertIs(function.__self__, torch._C)
        self.assertEqual(function.__reduce__(), "_has_torch_function_variadic")

        get_flags = ctypes.pythonapi.PyCFunction_GetFlags
        get_flags.argtypes = [ctypes.py_object]
        get_flags.restype = ctypes.c_int
        self.assertTrue(get_flags(function) & 0x0080)

        if sys.version_info >= (3, 13):
            self.assertEqual(function.__text_signature__, "($self, /, *args)")
            self.assertEqual(str(inspect.signature(function)), "(*args)")
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

    def test_probe_is_not_wildcard_exported_and_sequence_probe_stays_unsupported(self):
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

    def test_keyword_errors_match_pytorch_and_precede_argument_probes(self):
        function = torch.overrides.has_torch_function_variadic

        class RaisingLookup:
            calls = 0

            def __getattribute__(self, name):
                if name == "__torch_function__":
                    type(self).calls += 1
                    raise RuntimeError("argument should not be probed")
                return object.__getattribute__(self, name)

        cases = (
            lambda: function(input=None),
            lambda: function(None, unexpected=True),
            lambda: function(RaisingLookup(), unexpected=True),
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
        self.assertEqual(RaisingLookup.calls, 0)
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
assert function(torch.tensor([1.0]), None) is False
assert function(None, torch.Tensor) is True

class Override:
    __torch_function__ = None

assert function(None, Override(), object()) is True
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
