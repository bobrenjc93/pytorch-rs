import inspect
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorNameReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.name differentials require pinned PyTorch 2.13.0")

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("the operation unexpectedly succeeded")

    def tensor_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)[1]
        offset_view = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        ).transpose(0, 1)[1]
        tracked.sum().backward()
        tensors = (
            module.tensor(-0.0),
            module.zeros((2, 0, 3)),
            offset_view,
            leaf,
            tracked,
            leaf.grad,
            tracked.detach(),
        )

        def metadata(tensor):
            return (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                tensor.requires_grad,
                tensor.is_leaf,
            )

        before = tuple(metadata(tensor) for tensor in tensors)
        values = tuple(tensor.name is None for tensor in tensors)
        repeated = tuple(tensor.name is None for tensor in tensors)
        after = tuple(metadata(tensor) for tensor in tensors)
        return {
            "all_none": values,
            "repeated_none": repeated,
            "metadata_unchanged": before == after,
            "metadata": before,
            "gradient": leaf.grad.tolist(),
        }

    def test_supported_tensor_values_match_pytorch_2_13(self):
        self.assertEqual(
            self.tensor_contract(torch),
            self.tensor_contract(reference_torch),
        )

    def descriptor_contract(self, module):
        tensor = module.tensor([1.0])
        descriptor = inspect.getattr_static(module.Tensor, "name")
        assignments = (
            lambda: setattr(tensor, "name", None),
            lambda: setattr(tensor, "name", "batch"),
            lambda: delattr(tensor, "name"),
            lambda: descriptor.__set__(tensor, None),
            lambda: descriptor.__delete__(tensor),
        )
        wrong_receivers = (
            lambda: descriptor.__get__(1, int),
            lambda: descriptor.__set__(1, None),
            lambda: descriptor.__delete__(1),
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
            "has_text_signature": hasattr(descriptor, "__text_signature__"),
            "repr": repr(descriptor),
            "class_identity": module.Tensor.name is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value_is_none": descriptor.__get__(tensor, module.Tensor) is None,
            "assignment_errors": tuple(
                self.error(action) for action in assignments
            ),
            "wrong_receiver_errors": tuple(
                self.error(action) for action in wrong_receivers
            ),
        }

    def test_descriptor_and_mutation_contract_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.tensor([1.0], requires_grad=True)
        descriptor = inspect.getattr_static(module.Tensor, "name")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            intercepted = tensor.name
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
                forwarded = tensor.name

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
            "forwarded_is_none": forwarded is None,
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def unsupported_contract(self, module):
        return tuple(
            hasattr(module.Tensor, name)
            for name in ("names", "rename", "rename_", "refine_names")
        )

    def test_named_dimension_apis_remain_equally_unsupported(self):
        self.assertEqual(
            self.unsupported_contract(torch),
            self.unsupported_contract(reference_torch),
        )
        self.assertEqual(self.unsupported_contract(torch), (False,) * 4)


if __name__ == "__main__":
    unittest.main()
