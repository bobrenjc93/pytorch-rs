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

    def tensor_cases(self, module):
        ordinary = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        ordinary_view = ordinary.transpose(0, 1)
        leaf = module.tensor([1.0, 2.0], requires_grad=True)
        tracked = leaf * 2.0
        tracked_view = tracked.transpose(0, 0)
        empty_leaf = module.zeros((2, 0, 3), requires_grad=True)
        empty_tracked = empty_leaf * 2.0

        with module.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_view = leaf.transpose(0, 0)

        tracked.sum().backward()

        return (
            module.tensor(1.0),
            module.zeros((0,)),
            ordinary,
            ordinary_view,
            leaf,
            tracked,
            tracked_view,
            tracked.detach(),
            tracked_view.detach(),
            empty_leaf,
            empty_tracked,
            empty_tracked.transpose(0, 2),
            no_grad_output,
            no_grad_view,
            leaf.grad,
        )

    def value_contract(self, module):
        outcomes = []
        for tensor in self.tensor_cases(module):
            metadata = (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                tensor.requires_grad,
                tensor.is_leaf,
            )
            value = tensor.name
            outcomes.append(
                (
                    value is None,
                    type(value).__name__,
                    metadata
                    == (
                        tuple(tensor.shape),
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                )
            )
        return tuple(outcomes)

    def test_values_match_pytorch_2_13(self):
        self.assertEqual(
            self.value_contract(torch),
            self.value_contract(reference_torch),
        )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.name operation unexpectedly succeeded")

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "name")
        tensor = module.tensor([1.0])
        actions = (
            lambda: setattr(tensor, "name", "batch"),
            lambda: delattr(tensor, "name"),
            lambda: descriptor.__set__(tensor, "batch"),
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
            "class_identity": module.Tensor.name is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value_is_none": descriptor.__get__(tensor, module.Tensor) is None,
            "mutation_errors": tuple(self.error(action) for action in actions),
            "receiver_error": self.error(lambda: descriptor.__get__(1, int)),
        }

    def test_descriptor_documentation_and_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_dispatch_contract(self, module):
        tensor = module.tensor([1.0], requires_grad=True) * 2.0
        descriptor = inspect.getattr_static(module.Tensor, "name")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            intercepted = tensor.name
        function, dispatch_types, args, kwargs = recording.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
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
            self.mode_dispatch_contract(torch),
            self.mode_dispatch_contract(reference_torch),
        )

    def named_api_contract(self, module):
        tensor = module.tensor([[1.0]])
        return tuple(
            (hasattr(module.Tensor, attribute), hasattr(tensor, attribute))
            for attribute in ("names", "rename", "rename_", "refine_names")
        )

    def test_named_dimension_apis_remain_as_unsupported_as_pytorch_2_13(self):
        self.assertEqual(
            self.named_api_contract(torch),
            self.named_api_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
