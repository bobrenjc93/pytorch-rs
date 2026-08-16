import inspect
import math
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorIsMkldnnReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_mkldnn differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        strided_view = source.transpose(0, 1)
        offset_view = strided_view[1]
        extreme_empty = (
            module.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        return (
            *(
                module.tensor(value)
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
            module.zeros((2, 0, 3)),
            strided_view,
            offset_view,
            extreme_empty,
            leaf,
            tracked,
            leaf.grad,
        )

    def mkldnn_contract(self, module, tensor):
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        result = tensor.is_mkldnn
        return {
            "value": result,
            "value_type": type(result).__name__,
            "layout_is_canonical_strided": tensor.layout is module.strided,
            "metadata": metadata,
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                str(tensor.dtype),
                str(tensor.device),
                tensor.requires_grad,
                tensor.is_leaf,
            ),
        }

    def test_supported_tensor_states_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)

        self.assertFalse(actual_cases[8].is_contiguous())
        self.assertFalse(expected_cases[8].is_contiguous())
        self.assertGreater(actual_cases[9].storage_offset(), 0)
        self.assertGreater(expected_cases[9].storage_offset(), 0)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape):
                self.assertEqual(
                    self.mkldnn_contract(torch, actual),
                    self.mkldnn_contract(reference_torch, expected),
                )

    def test_reference_true_state_is_bounded_by_missing_mkldnn_layout(self):
        actual = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expected = reference_torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        self.assertEqual(
            self.mkldnn_contract(torch, actual),
            self.mkldnn_contract(reference_torch, expected),
        )
        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        self.assertTrue(hasattr(reference_torch.Tensor, "to_mkldnn"))

        if not reference_torch.backends.mkldnn.is_available():
            self.skipTest("PyTorch oneDNN backend is unavailable")

        genuine_mkldnn = expected.to_mkldnn()
        self.assertIs(type(genuine_mkldnn.is_mkldnn), bool)
        self.assertIs(genuine_mkldnn.is_mkldnn, True)
        self.assertEqual(str(genuine_mkldnn.layout), "torch._mkldnn")
        self.assertIs(actual.is_mkldnn, False)
        self.assertIs(actual.layout, torch.strided)

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.is_mkldnn unexpectedly accepted the operation")

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "is_mkldnn")
        tensor = module.tensor([1.0])
        actions = (
            lambda: setattr(tensor, "is_mkldnn", True),
            lambda: delattr(tensor, "is_mkldnn"),
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
            "class_identity": module.Tensor.is_mkldnn is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value": descriptor.__get__(tensor, module.Tensor),
            "value_type": type(
                descriptor.__get__(tensor, module.Tensor)
            ).__name__,
            "mutation_errors": tuple(self.error(action) for action in actions),
            "receiver_error": self.error(lambda: descriptor.__get__(1, int)),
        }

    def test_descriptor_ownership_absent_documentation_and_errors_match(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_dispatch_contract(self, module):
        tensor = module.tensor([1.0])
        descriptor = inspect.getattr_static(module.Tensor, "is_mkldnn")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            intercepted = tensor.is_mkldnn
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
                forwarded = tensor.is_mkldnn

        return {
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
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_contract(torch),
            self.mode_dispatch_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
