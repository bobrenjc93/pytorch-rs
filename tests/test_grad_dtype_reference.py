import inspect
import types
import unittest
import warnings

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorGradDtypeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "grad_dtype differentials require pinned PyTorch 2.13.0"
            )

    def leaf_cases(self, module):
        ordinary = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = leaf * 2.0
        tracked_view = tracked.transpose(0, 1)

        with module.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_leaf_view = leaf.transpose(0, 1)
            no_grad_non_leaf_view = tracked.transpose(0, 1)

        tracked.sum().backward()
        return (
            ordinary,
            ordinary + 1.0,
            ordinary.transpose(0, 1),
            leaf,
            module.zeros((2, 0, 3)),
            module.zeros((2, 0, 3), requires_grad=True),
            tracked.detach(),
            tracked_view.detach(),
            no_grad_output,
            no_grad_leaf_view,
            no_grad_non_leaf_view,
            leaf.grad,
        )

    def leaf_contract(self, module, tensor):
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        value = tensor.grad_dtype
        return {
            "repr": repr(value),
            "type": type(value).__name__,
            "is_float32": value is module.float32,
            "is_tensor_dtype": value is tensor.dtype,
            "metadata": metadata,
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                tensor.requires_grad,
                tensor.is_leaf,
            ),
        }

    def test_leaf_defaults_match_pytorch_2_13(self):
        actual_cases = self.leaf_cases(torch)
        expected_cases = self.leaf_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                actual_contract = self.leaf_contract(torch, actual)
                expected_contract = self.leaf_contract(
                    reference_torch, expected
                )
                self.assertTrue(actual_contract["is_float32"])
                self.assertTrue(expected_contract["is_float32"])
                self.assertEqual(actual_contract, expected_contract)

    def setter_contract(self, module, tensor):
        descriptor = inspect.getattr_static(module.Tensor, "grad_dtype")
        gradient = tensor.grad
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            repr(tensor.dtype),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        assignment_result = setattr(tensor, "grad_dtype", module.float32)
        descriptor_result = descriptor.__set__(tensor, module.float)
        return {
            "assignment_result": assignment_result,
            "descriptor_result": descriptor_result,
            "grad_dtype": repr(tensor.grad_dtype),
            "is_float32": tensor.grad_dtype is module.float32,
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                repr(tensor.dtype),
                tensor.requires_grad,
                tensor.is_leaf,
            ),
            "gradient_identity_unchanged": tensor.grad is gradient,
            "gradient": None if gradient is None else gradient.tolist(),
        }

    def test_float32_setter_matches_pytorch_2_13_on_supported_leaves(self):
        actual_cases = self.leaf_cases(torch)
        expected_cases = self.leaf_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                self.assertEqual(
                    self.setter_contract(torch, actual),
                    self.setter_contract(reference_torch, expected),
                )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.grad_dtype unexpectedly accepted the operation")

    def non_leaf_contract(self, module):
        leaf = module.tensor([1.0, 2.0], requires_grad=True)
        tracked = leaf * 2.0
        descriptor = inspect.getattr_static(module.Tensor, "grad_dtype")
        tensors = (
            tracked,
            tracked.transpose(0, 0),
            tracked.reshape(2),
            tracked[0],
            module.zeros((2, 0, 3), requires_grad=True) * 2.0,
        )
        errors = tuple(
            (
                self.error(lambda tensor=tensor: tensor.grad_dtype),
                self.error(
                    lambda tensor=tensor: descriptor.__get__(
                        tensor, module.Tensor
                    )
                ),
            )
            for tensor in tensors
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            setter_errors = tuple(
                (
                    self.error(
                        lambda tensor=tensor: setattr(
                            tensor, "grad_dtype", module.float32
                        )
                    ),
                    self.error(
                        lambda tensor=tensor: descriptor.__set__(
                            tensor, module.float
                        )
                    ),
                    self.error(
                        lambda tensor=tensor: setattr(
                            tensor, "grad_dtype", None
                        )
                    ),
                )
                for tensor in tensors
            )
        tracked.sum().backward()
        return {
            "errors": errors,
            "setter_errors": setter_errors,
            "leaf_gradient": leaf.grad.tolist(),
            "leaf_gradient_dtype": repr(leaf.grad.dtype),
            "leaf_grad_dtype": repr(leaf.grad_dtype),
        }

    def test_non_leaf_error_and_gradient_state_match_pytorch_2_13(self):
        actual = self.non_leaf_contract(torch)
        expected = self.non_leaf_contract(reference_torch)
        self.assertEqual(actual, expected)
        for pair in actual["errors"]:
            for error in pair:
                self.assertEqual(
                    error,
                    (
                        "RuntimeError",
                        "grad_dtype can only be accessed on leaf tensors.",
                    ),
                )
        for group in actual["setter_errors"]:
            for error in group:
                self.assertEqual(
                    error,
                    (
                        "RuntimeError",
                        "grad_dtype can only be set on leaf tensors.",
                    ),
                )

    def gradient_contract(self, module):
        leaf = module.tensor([1.0, 2.0], requires_grad=True)
        weights = module.tensor([3.0, -4.0])
        before = repr(leaf.grad_dtype), leaf.grad
        leaf.grad_dtype = module.float
        (leaf * weights).sum().backward()
        first = (
            leaf.grad.tolist(),
            repr(leaf.grad.dtype),
            repr(leaf.grad_dtype),
        )
        leaf.grad_dtype = module.float32
        (leaf * weights).sum().backward()
        second = (
            leaf.grad.tolist(),
            repr(leaf.grad.dtype),
            repr(leaf.grad_dtype),
        )

        empty = module.zeros((2, 0, 3), requires_grad=True)
        empty.grad_dtype = module.float32
        (empty * 2.0).sum().backward()
        return {
            "before": before,
            "first": first,
            "second": second,
            "empty": (
                tuple(empty.grad.shape),
                empty.grad.numel(),
                repr(empty.grad.dtype),
                repr(empty.grad_dtype),
            ),
        }

    def test_float32_gradient_accumulation_matches_pytorch_2_13(self):
        self.assertEqual(
            self.gradient_contract(torch),
            self.gradient_contract(reference_torch),
        )

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "grad_dtype")
        tensor = module.tensor([1.0])
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
            "class_identity": module.Tensor.grad_dtype is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value": repr(descriptor.__get__(tensor, module.Tensor)),
            "setter_name": descriptor.__set__.__name__,
            "setter_qualname": descriptor.__set__.__qualname__,
            "setter_self": descriptor.__set__.__self__ is descriptor,
            "setter_result": descriptor.__set__(tensor, module.float32),
            "receiver_error": self.error(
                lambda: descriptor.__get__(1, int)
            ),
            "setter_receiver_error": self.error(
                lambda: descriptor.__set__(1, module.float32)
            ),
        }

    def test_descriptor_ownership_and_documentation_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_dispatch_contract(self, module):
        non_leaf = module.tensor([1.0], requires_grad=True) * 2.0
        descriptor = inspect.getattr_static(module.Tensor, "grad_dtype")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types_, args=(), kwargs=None):
                self.calls.append((func, types_, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            intercepted = non_leaf.grad_dtype
        function, dispatch_types, args, kwargs = recording.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types_, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        leaf = module.tensor([1.0], requires_grad=True)
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = leaf.grad_dtype

        return {
            "intercepted": intercepted is marker,
            "call_count": len(recording.calls),
            "function_type": type(function).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_self": function.__self__ is descriptor,
            "function_equals_descriptor_get": function == descriptor.__get__,
            "types": dispatch_types == (module.Tensor,),
            "args": len(args) == 1 and args[0] is non_leaf,
            "kwargs_is_none": kwargs is None,
            "order": order,
            "forwarded": forwarded is module.float32,
        }

    def test_getter_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_contract(torch),
            self.mode_dispatch_contract(reference_torch),
        )

    def setter_mode_dispatch_contract(self, module):
        non_leaf = module.tensor([1.0], requires_grad=True) * 2.0
        descriptor = inspect.getattr_static(module.Tensor, "grad_dtype")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types_, args=(), kwargs=None):
                self.calls.append((func, types_, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            assignment_result = setattr(
                non_leaf, "grad_dtype", module.float32
            )
        function, dispatch_types, args, kwargs = recording.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types_, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        leaf = module.tensor([1.0], requires_grad=True)
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarding_result = descriptor.__set__(leaf, module.float)

        return {
            "assignment_result": assignment_result,
            "intercepted": len(recording.calls) == 1,
            "function_type": type(function).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_self": function.__self__ is descriptor,
            "function_equals_descriptor_set": function == descriptor.__set__,
            "types": dispatch_types == (module.Tensor,),
            "args": len(args) == 2
            and args[0] is non_leaf
            and args[1] is module.float32,
            "kwargs_is_none": kwargs is None,
            "order": order,
            "forwarding_result": forwarding_result,
            "forwarded": leaf.grad_dtype is module.float32,
        }

    def test_setter_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.setter_mode_dispatch_contract(torch),
            self.setter_mode_dispatch_contract(reference_torch),
        )

    def test_unsupported_setter_states_are_rejected_without_mutation(self):
        descriptor = inspect.getattr_static(torch.Tensor, "grad_dtype")
        tensor = torch.tensor([1.0], requires_grad=True)
        (tensor * 3.0).sum().backward()
        gradient = tensor.grad

        self.assertEqual(
            self.error(lambda: setattr(tensor, "grad_dtype", object())),
            self.error(
                lambda: setattr(
                    reference_torch.tensor([1.0]),
                    "grad_dtype",
                    object(),
                )
            ),
        )
        unsupported = (
            (
                None,
                (
                    "NotImplementedError",
                    "torch_rs only supports setting grad_dtype to torch.float32",
                ),
            ),
            (
                reference_torch.float16,
                (
                    "RuntimeError",
                    "grad_dtype must be a torch.dtype or None, but got dtype",
                ),
            ),
        )
        for value, expected_error in unsupported:
            for mutation in (
                lambda value=value: setattr(tensor, "grad_dtype", value),
                lambda value=value: descriptor.__set__(tensor, value),
            ):
                with self.subTest(value=value, mutation=mutation):
                    self.assertEqual(self.error(mutation), expected_error)
                    self.assertIs(tensor.grad_dtype, torch.float32)
                    self.assertIs(tensor.grad, gradient)
                    self.assertEqual(gradient.tolist(), [3.0])

        for deletion in (
            lambda: delattr(tensor, "grad_dtype"),
            lambda: descriptor.__delete__(tensor),
        ):
            self.assertEqual(
                self.error(deletion),
                (
                    "AttributeError",
                    "attribute 'grad_dtype' of 'torch._C.TensorBase' objects "
                    "is not writable",
                ),
            )
            self.assertIs(tensor.grad_dtype, torch.float32)
            self.assertIs(tensor.grad, gradient)

        expected_float32 = reference_torch.tensor(
            [1.0], requires_grad=True
        )
        expected_float32.grad_dtype = reference_torch.float32
        self.assertIs(expected_float32.grad_dtype, reference_torch.float32)

        expected_none = reference_torch.tensor([1.0], requires_grad=True)
        expected_none.grad_dtype = None
        self.assertIsNone(expected_none.grad_dtype)

        expected_float16 = reference_torch.tensor(
            [1.0], requires_grad=True
        )
        expected_float16.grad_dtype = reference_torch.float16
        self.assertIs(expected_float16.grad_dtype, reference_torch.float16)


if __name__ == "__main__":
    unittest.main()
