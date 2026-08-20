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
                "retain_grad differentials require pinned PyTorch 2.13.0"
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

    def leaf_retain_grad_contract(self, module):
        leaf = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        empty = module.zeros((2, 0, 3), requires_grad=True)
        source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with module.no_grad():
            view = source.transpose(0, 1)

        leaf_results = (leaf.retain_grad(), leaf.retain_grad())
        empty_result = empty.retain_grad()
        view_result = view.retain_grad()
        leaf.sum().backward()
        leaf.sum().backward()
        (empty + 2.0).sum().backward()
        (view * view).sum().backward()
        return {
            "leaf_results_are_none": all(
                result is None for result in leaf_results
            ),
            "empty_result_is_none": empty_result is None,
            "view_result_is_none": view_result is None,
            "retains_grad": (
                leaf.retains_grad,
                empty.retains_grad,
                view.retains_grad,
            ),
            "leaf_grad": leaf.grad.tolist(),
            "empty_grad_shape": tuple(empty.grad.shape),
            "empty_grad_stride": empty.grad.stride(),
            "empty_grad": empty.grad.tolist(),
            "view_state": (view.requires_grad, view.is_leaf),
            "source_grad_is_none": source.grad is None,
            "view_grad_is_none": view.grad is None,
        }

    def test_leaf_noop_and_gradient_accumulation_match_pytorch_2_13(self):
        self.assertEqual(
            self.leaf_retain_grad_contract(torch),
            self.leaf_retain_grad_contract(reference_torch),
        )

    def test_non_leaf_retention_remains_deliberately_unsupported(self):
        actual_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        actual = actual_leaf * 3.0
        expected_leaf = reference_torch.tensor(
            [1.0, 2.0], requires_grad=True
        )
        expected = expected_leaf * 3.0

        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.retain_grad\(\) only supports leaf tensors$",
        ):
            actual.retain_grad()
        self.assertIs(actual.retains_grad, False)
        self.assertIsNone(actual.grad)

        self.assertIs(expected.retain_grad(), None)
        self.assertIs(expected.retains_grad, True)
        actual.sum().backward()
        expected.sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())
        self.assertIsNone(actual.grad)
        self.assertIsNotNone(expected.grad)

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("operation unexpectedly succeeded")

    def signature_outcome(self, callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def retain_grad_callable_contract(self, module):
        tensor = module.tensor([1.0], requires_grad=True)
        descriptor = inspect.getattr_static(module.Tensor, "retain_grad")
        bound = tensor.retain_grad
        calls = (
            lambda: tensor.retain_grad(1),
            lambda: bound(1),
            lambda: tensor.retain_grad(1, 2),
            lambda: tensor.retain_grad(input=tensor),
            lambda: descriptor(tensor, unexpected=True),
            lambda: descriptor(),
            lambda: descriptor(1),
            lambda: descriptor(self=tensor),
        )
        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
            "repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "signatures": (
                self.signature_outcome(descriptor),
                self.signature_outcome(bound),
            ),
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "class_identity": module.Tensor.retain_grad is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "descriptor_result": descriptor(tensor),
            "bound_result": bound(),
            "call_errors": tuple(self.error(call) for call in calls),
        }

    def test_retain_grad_callable_metadata_and_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.retain_grad_callable_contract(torch),
            self.retain_grad_callable_contract(reference_torch),
        )

    def test_requires_grad_false_error_matches_pytorch_2_13(self):
        for shape in ((1,), (2, 0, 3)):
            actual = torch.zeros(shape)
            expected = reference_torch.zeros(shape)
            with self.subTest(shape=shape):
                self.assertEqual(
                    self.error(actual.retain_grad),
                    self.error(expected.retain_grad),
                )
                self.assertIs(actual.retains_grad, False)

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

    def retain_grad_mode_contract(self, module):
        tensor = module.tensor([1.0], requires_grad=True)
        descriptor = inspect.getattr_static(module.Tensor, "retain_grad")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            intercepted = tensor.retain_grad()
        function, dispatch_types, args, kwargs = recording.calls[0]

        untracked = module.tensor([1.0])
        replacement = RecordingMode()
        with replacement:
            untracked_result = untracked.retain_grad()

        rejected = RecordingMode()
        try:
            with rejected:
                tensor.retain_grad(unexpected=True)
        except Exception as error:
            invalid_error = type(error).__name__, str(error)
        else:
            invalid_error = None

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.retain_grad()

        return {
            "intercepted": intercepted is marker,
            "call_count": len(recording.calls),
            "function_type": type(function).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_is_descriptor": function is descriptor,
            "types": dispatch_types == (module.Tensor,),
            "args": len(args) == 1 and args[0] is tensor,
            "kwargs_is_none": kwargs is None,
            "untracked_replaced": untracked_result is marker,
            "replacement_calls": len(replacement.calls),
            "invalid_error": invalid_error,
            "rejected_calls": len(rejected.calls),
            "forwarding_order": order,
            "forwarded_is_none": forwarded is None,
            "retains_grad": tensor.retains_grad,
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_retain_grad_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.retain_grad_mode_contract(torch),
            self.retain_grad_mode_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
