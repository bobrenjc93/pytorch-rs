import copy
import inspect
import math
import pickle
import re
import types
import unittest
from decimal import Decimal

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


METHOD_DOC = (
    "\nadd(other, *, alpha=1) -> Tensor\n\n"
    "Add a scalar or tensor to :attr:`self` tensor. If both :attr:`alpha`\n"
    "and :attr:`other` are specified, each element of :attr:`other` is scaled by\n"
    ":attr:`alpha` before being used.\n\n"
    "When :attr:`other` is a tensor, the shape of :attr:`other` must be\n"
    ":ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying\n"
    "tensor\n\n"
    "See :func:`torch.add`\n"
)


class TensorAddTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertIs(actual.layout, torch.strided)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def test_tensor_and_real_scalar_calls_delegate_to_plus(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [3.0], [4.0]])
        self.assert_tensor_matches(left.add(right), left + right, case="tensor")
        self.assert_tensor_matches(
            left.add(other=right), left + right, case="tensor keyword"
        )
        self.assert_tensor_matches(left.add(x2=right), left + right, case="x2")
        self.assert_tensor_matches(
            left.add(x2=right, alpha=1), left + right, case="x2 alpha"
        )
        self.assert_tensor_matches(
            left.add(other=right, alpha=np.float32(1.0)),
            left + right,
            case="numpy alpha",
        )

        offset = left[1]
        for scalar in (True, -2, 2.5, np.bool_(False), np.int64(3), np.float32(-0.0)):
            with self.subTest(scalar=scalar):
                self.assert_tensor_matches(
                    offset.add(scalar), offset + scalar, case="scalar"
                )
                self.assert_tensor_matches(
                    offset.add(other=scalar), offset + scalar, case="scalar keyword"
                )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            empty.add(broadcast), empty + broadcast, case="empty broadcast"
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        self.assert_tensor_matches(
            special.add(np.float32(-0.0)),
            special + np.float32(-0.0),
            case="signed zero and non-finites",
        )

    def test_autograd_shared_operands_empties_and_no_grad_match_plus(self):
        method_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        method_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        operator_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        operator_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)

        method_output = method_left.transpose(0, 1).add(
            other=method_right.transpose(0, 1)
        )
        operator_output = operator_left.transpose(0, 1) + operator_right.transpose(0, 1)
        self.assert_tensor_matches(method_output, operator_output, case="tracked views")
        method_output.sum().backward()
        operator_output.sum().backward()
        self.assert_tensor_matches(method_left.grad, operator_left.grad, case="left grad")
        self.assert_tensor_matches(
            method_right.grad, operator_right.grad, case="right grad"
        )

        method_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        operator_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        method_shared.add(method_shared).sum().backward()
        (operator_shared + operator_shared).sum().backward()
        self.assert_tensor_matches(
            method_shared.grad, operator_shared.grad, case="shared operand grad"
        )

        method_empty = torch.zeros((2, 0, 3), requires_grad=True)
        operator_empty = torch.zeros((2, 0, 3), requires_grad=True)
        method_empty.add(torch.ones((1, 1, 3))).sum().backward()
        (operator_empty + torch.ones((1, 1, 3))).sum().backward()
        self.assert_tensor_matches(
            method_empty.grad, operator_empty.grad, case="empty grad"
        )

        no_grad_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        no_grad_right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            tensor_output = no_grad_left.transpose(0, 1).add(
                no_grad_right.transpose(0, 1)
            )
            scalar_output = no_grad_left.add(2.0)
        self.assertFalse(tensor_output.requires_grad)
        self.assertFalse(scalar_output.requires_grad)
        self.assertTrue(no_grad_left.add(2.0).requires_grad)

    def test_descriptor_metadata_copy_pickle_and_unbound_calls(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "add")
        bound = tensor.add

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(repr(descriptor), "<method 'add' of 'torch._C.TensorBase' objects>")
        self.assertEqual(descriptor.__name__, "add")
        self.assertEqual(descriptor.__qualname__, "TensorBase.add")
        self.assertEqual(bound.__name__, "add")
        self.assertEqual(bound.__qualname__, "Tensor.add")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)
        with self.assertRaises(ValueError):
            inspect.signature(bound)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(torch.Tensor.add, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(copy.copy(descriptor), descriptor)
        self.assertIs(copy.deepcopy(descriptor), descriptor)
        self.assertIs(copy.copy(bound), bound)
        self.assertIs(copy.deepcopy(bound), bound)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(descriptor, protocol=protocol)),
                    descriptor,
                )

        self.assert_tensor_matches(
            descriptor(tensor, other=tensor), tensor + tensor, case="unbound other"
        )
        self.assert_tensor_matches(
            descriptor(tensor, x2=tensor, alpha=1), tensor + tensor, case="unbound x2"
        )

    def test_rejected_arguments_do_not_mutate_inputs_or_add_top_level_function(self):
        tensor = torch.tensor([1.0])
        destination = torch.tensor([17.0])

        cases = (
            (
                lambda: tensor.add(),
                TypeError,
                'add() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.add(tensor, tensor),
                TypeError,
                "add() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.add(tensor, other=tensor),
                TypeError,
                "add() got multiple values for argument 'other'",
            ),
            (
                lambda: tensor.add(tensor, out=destination),
                TypeError,
                "add() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: tensor.add(tensor, out=None),
                TypeError,
                "add() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: tensor.add(wat=tensor),
                TypeError,
                'add() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.add([]),
                TypeError,
                "add(): argument 'other' (position 1) must be Tensor, not list",
            ),
            (
                lambda: tensor.add(other=None),
                TypeError,
                "add(): argument 'other' must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.add(Decimal("1.0")),
                TypeError,
                "add(): argument 'other' (position 1) must be Tensor, not decimal.Decimal",
            ),
            (
                lambda: tensor.add(1 + 2j),
                TypeError,
                "add(): argument 'other' (position 1) must be Tensor, not complex",
            ),
            (
                lambda: tensor.add(torch.float32),
                TypeError,
                "add(): argument 'other' (position 1) must be Tensor, not torch.dtype",
            ),
            (
                lambda: tensor.add(torch.device("cpu")),
                TypeError,
                "add(): argument 'other' (position 1) must be Tensor, not torch.device",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()
                self.assertEqual(destination.tolist(), [17.0])

        for alpha in (0, 2, np.float32(-1.0), math.nan, math.inf):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^Tensor\.add\(\) only supports alpha=1$",
                ):
                    tensor.add(tensor, alpha=alpha)

        alpha_errors = (
            (
                lambda: tensor.add(tensor, alpha=True),
                RuntimeError,
                "Boolean alpha only supported for Boolean results.",
            ),
            (
                lambda: tensor.add(tensor, alpha=[]),
                TypeError,
                "add(): argument 'alpha' must be Number, not list",
            ),
            (
                lambda: tensor.add(tensor, alpha=np.uint64(2**63)),
                TypeError,
                "an integer is required",
            ),
            (
                lambda: tensor.add(tensor, alpha=2**64),
                OverflowError,
                "int too big to convert",
            ),
            (
                lambda: tensor.add(tensor, alpha=-(2**63) - 1),
                OverflowError,
                "can't convert negative int to unsigned",
            ),
        )
        for call, error_type, message in alpha_errors:
            with self.subTest(alpha_error=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()

        self.assertFalse(hasattr(torch, "add"))
        self.assertNotIn("add", torch.__all__)

    def test_torch_function_modes_and_operand_overrides(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "add")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        wide_uint_alpha = np.uint64(2**63)
        positive_overflow_alpha = 2**100
        negative_overflow_alpha = -(2**63) - 1

        for label, call in (
            ("tensor", lambda: tensor.add(tensor)),
            ("scalar", lambda: tensor.add(2.0)),
            ("keyword", lambda: tensor.add(other=tensor, alpha=1)),
            ("bool alpha", lambda: tensor.add(tensor, alpha=True)),
        ):
            recording = RecordingMode(marker)
            with self.subTest(label=label):
                with recording:
                    self.assertIs(call(), marker)
                self.assertEqual(len(recording.calls), 1)
                function, dispatch_types, args, kwargs = recording.calls[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, ())
                self.assertIs(args[0], tensor)
                if label == "keyword":
                    self.assertEqual(len(args), 1)
                    self.assertEqual(set(kwargs), {"other", "alpha"})
                    self.assertIs(kwargs["other"], tensor)
                    self.assertEqual(kwargs["alpha"], 1)
                elif label == "bool alpha":
                    self.assertEqual(len(args), 2)
                    self.assertIs(args[1], tensor)
                    self.assertEqual(kwargs, {"alpha": True})
                else:
                    self.assertEqual(len(args), 2)
                    self.assertIsNone(kwargs)

        for label, alpha_value in (
            ("wide uint alpha", wide_uint_alpha),
            ("positive overflow alpha", positive_overflow_alpha),
            ("negative overflow alpha", negative_overflow_alpha),
        ):
            recording = RecordingMode(marker)
            with self.subTest(label=label):
                with recording:
                    self.assertIs(tensor.add(tensor, alpha=alpha_value), marker)
                self.assertEqual(len(recording.calls), 1)
                function, dispatch_types, args, kwargs = recording.calls[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(len(args), 2)
                self.assertIs(args[0], tensor)
                self.assertIs(args[1], tensor)
                self.assertIs(kwargs["alpha"], alpha_value)

        non_number_recording = RecordingMode(marker)
        with self.assertRaisesRegex(
            TypeError, r"^add\(\): argument 'alpha' must be Number, not list$"
        ):
            with non_number_recording:
                tensor.add(tensor, alpha=[])
        self.assertEqual(non_number_recording.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.add(tensor)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(forwarded, tensor + tensor, case="forwarded modes")

        declining = RecordingMode(NotImplemented)
        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.Tensor\.add'; all "
            r"__torch_function__ handlers returned NotImplemented:",
        ):
            with declining:
                tensor.add(tensor)
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        Override.calls.clear()
        self.assertIs(tensor.add(value), marker)
        self.assertEqual(len(Override.calls), 1)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], tensor)
        self.assertIs(args[1], value)
        self.assertIsNone(kwargs)

        Override.calls.clear()
        self.assertIs(tensor.add(value, alpha=True), marker)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], tensor)
        self.assertIs(args[1], value)
        self.assertEqual(kwargs, {"alpha": True})

        for label, alpha_value in (
            ("wide uint alpha", wide_uint_alpha),
            ("positive overflow alpha", positive_overflow_alpha),
            ("negative overflow alpha", negative_overflow_alpha),
        ):
            Override.calls.clear()
            with self.subTest(override_alpha=label):
                self.assertIs(tensor.add(value, alpha=alpha_value), marker)
                function, dispatch_types, args, kwargs = Override.calls[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(len(args), 2)
                self.assertIs(args[0], tensor)
                self.assertIs(args[1], value)
                self.assertIs(kwargs["alpha"], alpha_value)

        alpha = Override()
        Override.calls.clear()
        self.assertIs(tensor.add(tensor, alpha=alpha), marker)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], tensor)
        self.assertIs(args[1], tensor)
        self.assertIs(kwargs["alpha"], alpha)

        Override.calls.clear()
        with self.assertRaisesRegex(
            TypeError, r"^add\(\) got an unexpected keyword argument 'out'$"
        ):
            tensor.add(tensor, out=Override())
        self.assertEqual(Override.calls, [])


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorAddReferenceTests(unittest.TestCase):
    def assert_matches_reference(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertIs(actual.layout, torch.strided)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    def test_default_alpha_values_and_metadata_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        actual_right = torch.tensor([[2.0], [3.0], [4.0]])
        expected_right = reference_torch.tensor([[2.0], [3.0], [4.0]])

        calls = (
            (
                "tensor",
                actual_left.add(actual_right),
                expected_left.add(expected_right),
            ),
            (
                "tensor keyword",
                actual_left.add(other=actual_right, alpha=np.float32(1.0)),
                expected_left.add(other=expected_right, alpha=np.float32(1.0)),
            ),
            (
                "x2 alias",
                actual_left.add(x2=actual_right, alpha=1),
                expected_left.add(x2=expected_right, alpha=1),
            ),
            (
                "scalar",
                actual_left[1].add(np.int64(3)),
                expected_left[1].add(np.int64(3)),
            ),
            (
                "empty",
                torch.zeros((2, 0, 3)).transpose(0, 2).add(torch.ones((1, 1, 2))),
                reference_torch.zeros((2, 0, 3))
                .transpose(0, 2)
                .add(reference_torch.ones((1, 1, 2))),
            ),
        )
        for case, actual, expected in calls:
            self.assert_matches_reference(actual, expected, case=case)

        actual_descriptor = inspect.getattr_static(torch.Tensor, "add")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "add")
        self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)
        self.assertEqual(
            actual_descriptor.__qualname__, expected_descriptor.__qualname__
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )

    def test_default_alpha_autograd_and_modes_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        expected_left = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
        actual_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        expected_right = reference_torch.tensor(
            [[5.0], [7.0], [11.0]], requires_grad=True
        )
        actual_output = actual_left.transpose(0, 1).add(
            other=actual_right.transpose(0, 1)
        )
        expected_output = expected_left.transpose(0, 1).add(
            other=expected_right.transpose(0, 1)
        )
        self.assert_matches_reference(actual_output, expected_output, case="tracked")
        actual_output.sum().backward()
        expected_output.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_left.grad), expected_left.grad.numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(actual_right.grad), expected_right.grad.numpy()
        )

        actual_no_grad = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_no_grad = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            actual_untracked = actual_no_grad.add(2.0)
        with reference_torch.no_grad():
            expected_untracked = expected_no_grad.add(2.0)
        self.assert_matches_reference(actual_untracked, expected_untracked, case="no_grad")

        actual_descriptor = inspect.getattr_static(torch.Tensor, "add")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "add")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        actual_recording = RecordingMode(marker)
        with actual_recording:
            self.assertIs(actual_no_grad.add(actual_no_grad), marker)

        class ReferenceRecordingMode(reference_torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        expected_recording = ReferenceRecordingMode(marker)
        with expected_recording:
            self.assertIs(expected_no_grad.add(expected_no_grad), marker)

        self.assertEqual(len(actual_recording.calls), len(expected_recording.calls))
        actual_func, actual_types, actual_args, actual_kwargs = actual_recording.calls[0]
        expected_func, expected_types, expected_args, expected_kwargs = expected_recording.calls[0]
        self.assertIs(actual_func, actual_descriptor)
        self.assertIs(expected_func, expected_descriptor)
        self.assertEqual(actual_types, expected_types)
        self.assertEqual(len(actual_args), len(expected_args))
        self.assertEqual(actual_kwargs, expected_kwargs)


if __name__ == "__main__":
    unittest.main()
