import copy
import importlib
import inspect
import json
import math
import pickle
import subprocess
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorIsVulkanReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("is_vulkan differentials require pinned PyTorch 2.13.0")

    def tensor_cases(self, module, device):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            device=device,
            requires_grad=True,
        )
        tracked = leaf * 2.0
        tracked_view = tracked.transpose(0, 1)
        detached = tracked.detach()
        detached_view = tracked_view.detach()
        tracked.sum().backward()

        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            device=device,
        )
        strided_view = source.transpose(0, 1)
        offset_view = strided_view[1]
        extreme_empty = (
            module.zeros((0,), device=device)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        return (
            *(
                (f"scalar {value!r}", module.tensor(value, device=device))
                for value in (
                    -math.inf,
                    -1.0,
                    -0.0,
                    0.0,
                    1.0,
                    math.inf,
                    math.nan,
                )
            ),
            ("ordinary tensor", source),
            ("empty", module.zeros((2, 0, 3), device=device)),
            ("strided view", strided_view),
            ("offset strided view", offset_view),
            ("extreme empty view", extreme_empty),
            ("detached non-leaf", detached),
            ("detached strided view", detached_view),
            ("autograd leaf", leaf),
            ("autograd non-leaf", tracked),
            ("autograd non-leaf view", tracked_view),
            ("accumulated gradient", leaf.grad),
        )

    def vulkan_contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            str(tensor.dtype),
            str(tensor.device),
            str(tensor.layout),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )
        data_pointer = tensor.data_ptr()
        result = tensor.is_vulkan
        return {
            "value": result,
            "value_type": type(result).__name__,
            "metadata": metadata,
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                str(tensor.dtype),
                str(tensor.device),
                str(tensor.layout),
                tensor.requires_grad,
                tensor.is_leaf,
                tensor.output_nr,
            ),
            "storage_identity_unchanged": data_pointer == tensor.data_ptr(),
        }

    def test_supported_cpu_tensor_results_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch, "cpu")
        expected_cases = self.tensor_cases(reference_torch, "cpu")
        for (actual_name, actual), (expected_name, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            with self.subTest(case=actual_name, shape=actual.shape):
                self.assertEqual(actual_name, expected_name)
                self.assertEqual(
                    self.vulkan_contract(actual),
                    self.vulkan_contract(expected),
                )

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "PyTorch CUDA is unavailable",
    )
    def test_reference_pytorch_real_cuda_tensors_report_false(self):
        device = reference_torch.device("cuda", 0)
        device_name = reference_torch.cuda.get_device_name(device)
        self.assertTrue(device_name)

        for case, tensor in self.tensor_cases(reference_torch, device):
            with self.subTest(case=case, shape=tensor.shape, gpu=device_name):
                metadata = self.vulkan_contract(tensor)
                self.assertEqual(tensor.device.type, "cuda")
                self.assertIs(tensor.is_cuda, True)
                self.assertIs(type(tensor.is_vulkan), bool)
                self.assertIs(tensor.is_vulkan, False)
                self.assertTrue(metadata["metadata_unchanged"])
                self.assertTrue(metadata["storage_identity_unchanged"])
        reference_torch.cuda.synchronize(device)

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.is_vulkan unexpectedly accepted the operation")

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "is_vulkan")
        tensor = module.tensor([1.0])
        actions = (
            lambda: setattr(tensor, "is_vulkan", True),
            lambda: delattr(tensor, "is_vulkan"),
            lambda: descriptor.__set__(tensor, True),
            lambda: descriptor.__delete__(tensor),
        )
        return {
            "descriptor_type": type(descriptor).__name__,
            "is_getset": type(descriptor) is types.GetSetDescriptorType,
            "callable": callable(descriptor),
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "doc": descriptor.__doc__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "has_module": hasattr(descriptor, "__module__"),
            "repr": repr(descriptor),
            "class_identity": module.Tensor.is_vulkan is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value": descriptor.__get__(tensor, module.Tensor),
            "value_type": type(
                descriptor.__get__(tensor, module.Tensor)
            ).__name__,
            "mutation_errors": tuple(self.error(action) for action in actions),
            "receiver_error": self.error(lambda: descriptor.__get__(1, int)),
        }

    def test_descriptor_documentation_and_errors_match_pytorch_2_13(self):
        actual = self.descriptor_contract(torch)
        expected = self.descriptor_contract(reference_torch)
        self.assertIsNone(actual["doc"])
        self.assertEqual(actual, expected)

    def mode_dispatch_observation(self, module_name):
        source = r'''
import importlib
import inspect
import json

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "is_vulkan")
marker = object()

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append((func, types, args, kwargs))
        return self.result

recording = RecordingMode(marker)
with recording:
    intercepted = tensor.is_vulkan
function, dispatch_types, args, kwargs = recording.calls[0]

order = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label):
        self.label = label

    def __torch_function__(self, func, types, args=(), kwargs=None):
        order.append(self.label)
        return func(*args, **(kwargs or {}))

with ForwardingMode("lower"):
    with ForwardingMode("upper"):
        forwarded = tensor.is_vulkan

print(json.dumps({
    "intercepted": intercepted is marker,
    "call_count": len(recording.calls),
    "function_type": type(function).__name__,
    "function_name": function.__name__,
    "function_qualname": function.__qualname__,
    "function_self": function.__self__ is descriptor,
    "function_equals_descriptor_get": function == descriptor.__get__,
    "types": dispatch_types == (module.Tensor,),
    "args": len(args) == 1 and args[0] is tensor,
    "kwargs_is_none": kwargs is None,
    "forwarding_order": order,
    "forwarded": forwarded,
    "forwarded_type": type(forwarded).__name__,
    "stack_depth": len(module.overrides._get_current_function_mode_stack()),
}))
'''
        result = subprocess.run(
            [sys.executable, "-c", f"MODULE = {module_name!r}\n" + source],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_torch_function_mode_forwarding_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation("torch_rs"),
            self.mode_dispatch_observation("torch"),
        )

    def test_vulkan_backend_surface_remains_intentionally_unsupported(self):
        tensor = torch.tensor([1.0])
        self.assertFalse(hasattr(torch, "vulkan"))
        self.assertFalse(hasattr(reference_torch, "vulkan"))
        self.assertEqual(reference_torch.device("vulkan").type, "vulkan")

        for specification in ("vulkan", "vulkan:0"):
            with self.subTest(specification=specification, action="device"):
                with self.assertRaisesRegex(
                    RuntimeError, r"only 'cpu' is implemented"
                ):
                    torch.device(specification)
            with self.subTest(specification=specification, action="create"):
                with self.assertRaisesRegex(
                    RuntimeError, r"only 'cpu' is implemented"
                ):
                    torch.tensor([1.0], device=specification)
            with self.subTest(specification=specification, action="to"):
                with self.assertRaisesRegex(
                    NotImplementedError, r"device conversions are not supported"
                ):
                    tensor.to(specification)

        self.assertTrue(hasattr(torch.Tensor, "to"))
        self.assertFalse(hasattr(torch.Tensor, "vulkan"))


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class IsVulkanAvailableReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_vulkan_available differentials require pinned PyTorch 2.13.0"
            )

    def callable_contract(self, module):
        function = module.is_vulkan_available
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = type(error).__name__
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
            "self_is_none": function.__self__ is None,
            "has_annotations": hasattr(function, "__annotations__"),
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.is_vulkan_available is function,
            "signature_error": signature_error,
            "all_count": module.__all__.count("is_vulkan_available"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["is_vulkan_available"]
            is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def test_permissive_argument_contract_matches_pytorch_2_13(self):
        class ExplosiveOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise AssertionError("availability query dispatched an argument")

        for module in (torch, reference_torch):
            function = module.is_vulkan_available
            calls = (
                lambda: function(),
                lambda: function(None),
                lambda: function(1, "cuda", object()),
                lambda: function(enabled=True),
                lambda: function(None, device="cuda:0", enabled=True),
                lambda: function(
                    ExplosiveOverride(), candidate=ExplosiveOverride()
                ),
            )
            for call in calls:
                with self.subTest(module=module.__name__, call=call):
                    result = call()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)

    def native_reload_contract(self, module):
        native = module._C
        function = module.is_vulkan_available
        owner = native._VariableFunctionsClass
        reloaded = importlib.reload(native)
        return (
            reloaded is native,
            module.is_vulkan_available is function,
            module._C._VariableFunctionsClass is owner,
            owner.is_vulkan_available is function,
            function() is False,
        )

    def test_native_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.native_reload_contract(torch),
            self.native_reload_contract(reference_torch),
        )

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "PyTorch CUDA is unavailable",
    )
    def test_real_cuda_availability_does_not_change_vulkan_result(self):
        device = reference_torch.device("cuda", 0)
        tensor = reference_torch.arange(4, device=device)
        self.assertTrue(reference_torch.cuda.get_device_name(device))
        self.assertIs(reference_torch.cuda.is_available(), True)
        self.assertEqual(tensor.sum().item(), 6)

        for module in (torch, reference_torch):
            with self.subTest(module=module.__name__):
                self.assertIs(module.is_vulkan_available(), False)
                self.assertIs(
                    module.is_vulkan_available(tensor, cuda_available=True),
                    False,
                )
        reference_torch.cuda.synchronize(device)


if __name__ == "__main__":
    unittest.main()
