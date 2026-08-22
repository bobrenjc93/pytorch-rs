import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import torch_rs as torch


class IsVulkanAvailableTests(unittest.TestCase):
    def test_returns_exact_false_and_ignores_every_argument(self):
        function = torch.is_vulkan_available

        class ExplosiveOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise AssertionError("ignored arguments must not be dispatched")

        ignored = ExplosiveOverride()
        tensor = torch.tensor([1.0])
        calls = (
            lambda: function(),
            lambda: function(None),
            lambda: function(ignored, tensor, 3),
            lambda: function(ignored=True),
            lambda: function(
                ignored,
                tensor,
                arbitrary=object(),
                **{"embedded\x00null": ignored, "κ": tensor},
            ),
        )
        for case, call in enumerate(calls):
            with self.subTest(case=case):
                result = call()
                self.assertIs(type(result), bool)
                self.assertIs(result, False)

        class RaisingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise AssertionError("is_vulkan_available must ignore active modes")

        with RaisingMode():
            self.assertIs(function(tensor, ignored=tensor), False)

    def test_null_metadata_owner_exports_copying_and_pickling(self):
        function = torch.is_vulkan_available
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "is_vulkan_available")
        self.assertEqual(
            function.__qualname__,
            "_VariableFunctionsClass.is_vulkan_available",
        )
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__doc__)
        self.assertIsNone(function.__text_signature__)
        self.assertFalse(hasattr(function, "__annotations__"))
        self.assertIsNone(function.__self__)
        self.assertRegex(
            repr(function),
            r"^<built-in method is_vulkan_available of type object at "
            r"0x[0-9a-f]+>$",
        )
        with self.assertRaisesRegex(ValueError, "^no signature found for builtin"):
            inspect.signature(function)

        reducer, (owner, name) = function.__reduce__()
        self.assertIs(reducer, getattr)
        self.assertEqual(name, "is_vulkan_available")
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.is_vulkan_available, function)

        for action in (
            lambda: setattr(owner, "is_vulkan_available", None),
            lambda: delattr(owner, "is_vulkan_available"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.is_vulkan_available, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("is_vulkan_available"), 1)
        self.assertEqual(torch._C.__all__.count("is_vulkan_available"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["is_vulkan_available"], function)

    def test_package_and_native_reload_preserve_callable_identity(self):
        package = torch
        native = torch._C
        function = torch.is_vulkan_available
        owner = torch._C._VariableFunctionsClass

        def assert_stable_surface():
            self.assertIs(torch.is_vulkan_available, function)
            self.assertIs(torch._C._VariableFunctionsClass, owner)
            self.assertIs(owner.is_vulkan_available, function)
            self.assertEqual(torch.__all__.count("is_vulkan_available"), 1)
            self.assertEqual(torch._C.__all__.count("is_vulkan_available"), 1)
            self.assertIs(torch.is_vulkan_available(), False)

        assert_stable_surface()
        self.assertIs(importlib.reload(package), package)
        assert_stable_surface()
        self.assertIs(importlib.reload(native), native)
        assert_stable_surface()

    def test_vulkan_execution_surface_remains_unsupported(self):
        self.assertFalse(hasattr(torch, "vulkan"))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.vulkan")

        for specification in ("vulkan", "vulkan:0"):
            with self.subTest(specification=specification, action="device"):
                with self.assertRaisesRegex(RuntimeError, "only 'cpu' is implemented"):
                    torch.device(specification)
            with self.subTest(specification=specification, action="kernel"):
                with self.assertRaisesRegex(RuntimeError, "only 'cpu' is implemented"):
                    torch.zeros((1,), device=specification)

        tensor = torch.tensor([1.0])
        self.assertFalse(hasattr(torch.Tensor, "to"))
        self.assertFalse(hasattr(torch.Tensor, "vulkan"))
        with self.assertRaises(AttributeError):
            tensor.to("vulkan")
        with self.assertRaises(AttributeError):
            tensor.vulkan()


if __name__ == "__main__":
    unittest.main()
