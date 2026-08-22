import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class IsVulkanAvailableReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_vulkan_available differentials require pinned PyTorch 2.13.0"
            )

    def call_contract(self, module):
        function = module.is_vulkan_available

        class ExplosiveOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise AssertionError("ignored arguments must not be dispatched")

        ignored = ExplosiveOverride()
        tensor = module.tensor([1.0])
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
        results = []
        for call in calls:
            result = call()
            results.append((result is False, type(result).__name__))

        class RaisingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise AssertionError("is_vulkan_available must ignore active modes")

        with RaisingMode():
            mode_result = function(tensor, ignored=tensor)
        return tuple(results), mode_result is False, type(mode_result).__name__

    def test_return_and_permissive_calling_match_pytorch_2_13(self):
        self.assertEqual(
            self.call_contract(torch),
            self.call_contract(reference_torch),
        )

    def metadata_contract(self, module):
        function = module.is_vulkan_available
        reducer, (owner, name) = function.__reduce__()
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                str(error).split(" for ", 1)[0],
            )
        else:
            signature_error = None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "has_annotations": hasattr(function, "__annotations__"),
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "reducer_is_getattr": reducer is getattr,
            "reduce_name": name,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_identity": owner is module._C._VariableFunctionsClass,
            "owner_function_identity": owner.is_vulkan_available is function,
            "self_is_none": function.__self__ is None,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "all_count": module.__all__.count("is_vulkan_available"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["is_vulkan_available"]
            is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol))
                is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_metadata_ownership_exports_copying_and_pickling_match(self):
        actual = self.metadata_contract(torch)
        expected = self.metadata_contract(reference_torch)
        self.assertIsNone(actual["doc"])
        self.assertEqual(actual, expected)

    def reload_contract(self, module):
        native = module._C
        function = module.is_vulkan_available
        owner = native._VariableFunctionsClass
        reloaded = importlib.reload(native)
        return (
            reloaded is native,
            module.is_vulkan_available is function,
            native._VariableFunctionsClass is owner,
            owner.is_vulkan_available is function,
            module.__all__.count("is_vulkan_available"),
            module.is_vulkan_available() is False,
        )

    def test_native_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "PyTorch CUDA is unavailable",
    )
    def test_real_cuda_availability_and_tensor_arguments_do_not_change_result(self):
        device = reference_torch.device("cuda", 0)
        tensor = reference_torch.tensor([1.0, 2.0], device=device)
        device_name = reference_torch.cuda.get_device_name(device)
        self.assertTrue(device_name)
        self.assertIs(reference_torch.cuda.is_available(), True)
        self.assertEqual(tensor.device.type, "cuda")
        self.assertIs(tensor.is_cuda, True)

        for function in (
            torch.is_vulkan_available,
            reference_torch.is_vulkan_available,
        ):
            with self.subTest(function=function.__module__, gpu=device_name):
                self.assertIs(function(), False)
                self.assertIs(function(tensor), False)
                self.assertIs(function(cuda_tensor=tensor), False)

        self.assertEqual((tensor + 1).tolist(), [2.0, 3.0])
        reference_torch.cuda.synchronize(device)

    def test_availability_query_does_not_enable_the_vulkan_backend(self):
        self.assertIs(torch.is_vulkan_available(), False)
        self.assertIs(reference_torch.is_vulkan_available(), False)
        self.assertFalse(hasattr(torch, "vulkan"))
        self.assertFalse(hasattr(reference_torch, "vulkan"))
        self.assertEqual(reference_torch.device("vulkan").type, "vulkan")

        for specification in ("vulkan", "vulkan:0"):
            with self.subTest(specification=specification):
                with self.assertRaisesRegex(RuntimeError, "only 'cpu' is implemented"):
                    torch.tensor([1.0], device=specification)
        self.assertFalse(hasattr(torch.Tensor, "to"))
        self.assertFalse(hasattr(torch.Tensor, "vulkan"))


if __name__ == "__main__":
    unittest.main()
