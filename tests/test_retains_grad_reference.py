import inspect
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorRetainsGradReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "retains_grad differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        ordinary = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = leaf * 2.0
        tracked_view = tracked.transpose(0, 1)
        detached_tracked = tracked.detach()
        detached_view = tracked_view.detach()

        with module.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_leaf_view = leaf.transpose(0, 1)
            no_grad_non_leaf_view = tracked.transpose(0, 1)

        recorded_after_no_grad = no_grad_leaf_view + 1.0
        tracked.sum().backward()

        return (
            ordinary,
            ordinary + 1.0,
            ordinary.transpose(0, 1),
            leaf,
            tracked,
            tracked_view,
            detached_tracked,
            detached_view,
            no_grad_output,
            no_grad_leaf_view,
            no_grad_non_leaf_view,
            recorded_after_no_grad,
            leaf.grad,
        )

    def default_contract(self, tensor):
        value = tensor.retains_grad
        return {
            "value": value,
            "value_type": type(value).__name__,
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
        }

    def test_false_defaults_match_pytorch_2_13_for_every_supported_state(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                actual_contract = self.default_contract(actual)
                expected_contract = self.default_contract(expected)
                self.assertIs(actual_contract["value"], False)
                self.assertIs(expected_contract["value"], False)
                self.assertEqual(actual_contract, expected_contract)

    def test_reference_true_state_is_bounded_by_missing_retain_grad(self):
        actual = torch.tensor([1.0], requires_grad=True) * 2.0
        expected = reference_torch.tensor([1.0], requires_grad=True) * 2.0

        self.assertEqual(
            self.default_contract(actual), self.default_contract(expected)
        )
        self.assertFalse(hasattr(torch.Tensor, "retain_grad"))
        self.assertTrue(hasattr(reference_torch.Tensor, "retain_grad"))

        expected.retain_grad()
        self.assertIs(expected.retains_grad, True)
        expected.sum().backward()
        self.assertIsNotNone(expected.grad)

        actual.sum().backward()
        self.assertIs(actual.retains_grad, False)
        self.assertIsNone(actual.grad)

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.retains_grad unexpectedly accepted the operation")

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "retains_grad")
        tensor = module.tensor([1.0])
        actions = (
            lambda: setattr(tensor, "retains_grad", True),
            lambda: delattr(tensor, "retains_grad"),
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
            "class_identity": module.Tensor.retains_grad is descriptor,
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
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_dispatch_contract(self, module):
        tensor = module.tensor([1.0], requires_grad=True) * 2.0
        descriptor = inspect.getattr_static(module.Tensor, "retains_grad")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            intercepted = tensor.retains_grad
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
                forwarded = tensor.retains_grad

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
