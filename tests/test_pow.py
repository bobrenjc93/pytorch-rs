import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


POW_DOC = """
pow(exponent) -> Tensor

See :func:`torch.pow`
"""
UNSUPPORTED_EXPONENT = "Tensor.pow only supports the real scalar exponent 2"


class TensorPowTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                np.asarray(expected, dtype=np.float32).reshape(-1).view(np.uint32),
            )

    @staticmethod
    def value_cases():
        base = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        return (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            (
                "signed zero and non-finites",
                torch.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

    @staticmethod
    def autograd_case(case):
        if case == "scalar":
            leaf = torch.tensor(-3.0, requires_grad=True)
            return leaf, leaf, None
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return leaf, leaf.transpose(0, 2)[1], None

        leaf = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        if case == "offset":
            source = leaf[1]
            weights = torch.tensor(
                np.arange(1, 13, dtype=np.float32).reshape(3, 4).tolist()
            )
            return leaf, source, weights
        if case == "noncontiguous":
            source = leaf.transpose(0, 2)
            weights = torch.tensor(
                np.arange(1, 25, dtype=np.float32).reshape(4, 3, 2).tolist()
            )
            return leaf, source, weights
        raise AssertionError(f"unknown pow autograd case: {case}")

    def test_exponent_two_reuses_square_values_layout_and_fresh_storage(self):
        for case, source in self.value_cases():
            output = source.pow(2)
            expected = source.square()
            self.assert_tensor_matches(output, expected, case=case)
            self.assertFalse(output.is_set_to(source))

        class IntExponent(int):
            pass

        class FloatExponent(float):
            pass

        exponents = (
            2,
            2.0,
            IntExponent(2),
            FloatExponent(2.0),
            np.int8(2),
            np.uint64(2),
            np.float16(2),
            np.float32(2),
            np.float64(2),
        )
        source = torch.tensor([-0.0, -3.0, float("inf"), float("nan")])
        expected = source.square()
        for index, exponent in enumerate(exponents):
            output = (
                source.pow(exponent)
                if index % 2 == 0
                else source.pow(exponent=exponent)
            )
            self.assert_tensor_matches(output, expected, case=type(exponent).__name__)
            self.assertFalse(output.is_set_to(source))

    def test_autograd_repeated_backward_and_no_grad_reuse_square(self):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            pow_leaf, pow_input, pow_weights = self.autograd_case(case)
            square_leaf, square_input, square_weights = self.autograd_case(case)
            pow_output = pow_input.pow(2)
            square_output = square_input.square()
            self.assert_tensor_matches(
                pow_output, square_output, case=(case, "forward")
            )

            if pow_weights is None:
                pow_loss = pow_output if case == "scalar" else pow_output.sum()
                square_loss = (
                    square_output if case == "scalar" else square_output.sum()
                )
            else:
                pow_loss = (pow_output * pow_weights).sum()
                square_loss = (square_output * square_weights).sum()
            pow_loss.backward()
            square_loss.backward()
            self.assert_tensor_matches(
                pow_leaf.grad, square_leaf.grad, case=(case, "gradient")
            )

        accumulated = torch.tensor([2.0, -3.0], requires_grad=True)
        accumulated.pow(2).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(accumulated.grad), np.asarray([4.0, -6.0], dtype=np.float32)
        )
        accumulated.pow(exponent=2.0).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(accumulated.grad), np.asarray([8.0, -12.0], dtype=np.float32)
        )

        freed = torch.tensor([2.0, -3.0], requires_grad=True)
        loss = freed.pow(2).sum()
        loss.backward()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

        probability = torch.tensor([2.0], requires_grad=True).pow(2)
        with self.assertRaisesRegex(ValueError, r"grad_fn=<PowBackward0>"):
            torch.nn.functional.dropout(
                torch.tensor([1.0]), p=probability, training=False
            )

        for case in ("scalar", "empty", "offset", "noncontiguous"):
            _, source, _ = self.autograd_case(case)
            with torch.no_grad():
                output = source.pow(2)
            expected = source.detach().square()
            self.assert_tensor_matches(output, expected, case=(case, "no_grad"))
            self.assertFalse(output.is_set_to(source))

    def test_tensorbase_descriptor_metadata_and_argument_binding(self):
        tensor = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "pow")
        bound = tensor.pow

        self.assertIs(torch.Tensor.pow, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(repr(descriptor), "<method 'pow' of 'torch._C.TensorBase' objects>")
        self.assertEqual(descriptor.__name__, "pow")
        self.assertEqual(descriptor.__qualname__, "TensorBase.pow")
        self.assertEqual(bound.__name__, "pow")
        self.assertEqual(bound.__qualname__, "Tensor.pow")
        self.assertEqual(descriptor.__doc__, POW_DOC)
        self.assertEqual(bound.__doc__, POW_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        expected_missing = (
            "pow() received an invalid combination of arguments - got (), but "
            "expected one of:\n * (Tensor exponent)\n * (Number exponent)\n"
        )
        expected_extra = (
            "pow() received an invalid combination of arguments - got (int, int), "
            "but expected one of:\n * (Tensor exponent)\n * (Number exponent)\n"
        )
        expected_duplicate = (
            "pow() received an invalid combination of arguments - got (int, "
            "exponent=int), but expected one of:\n * (Tensor exponent)\n"
            " * (Number exponent)\n"
        )
        expected_keyword = (
            "pow() received an invalid combination of arguments - got (other=int, ), "
            "but expected one of:\n * (Tensor exponent)\n"
            "      didn't match because some of the keywords were incorrect: other\n"
            " * (Number exponent)\n"
            "      didn't match because some of the keywords were incorrect: other\n"
        )
        cases = (
            (lambda: tensor.pow(), expected_missing),
            (lambda: bound(), expected_missing),
            (lambda: descriptor(tensor), expected_missing),
            (lambda: tensor.pow(2, 3), expected_extra),
            (lambda: tensor.pow(2, exponent=2), expected_duplicate),
            (lambda: tensor.pow(other=2), expected_keyword),
            (lambda: descriptor(), "unbound method TensorBase.pow() needs an argument"),
            (
                lambda: descriptor(1, 2),
                "descriptor 'pow' for 'torch._C.TensorBase' objects doesn't apply "
                "to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor, exponent=2),
                "unbound method TensorBase.pow() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        self.assertEqual(tensor.pow(2).tolist(), [4.0])
        self.assertEqual(tensor.pow(exponent=2).tolist(), [4.0])
        self.assertEqual(descriptor(tensor, exponent=2).tolist(), [4.0])

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.tensor([2.0, -3.0], requires_grad=True)
        tensor_exponent = torch.tensor(2.0)
        descriptor = inspect.getattr_static(torch.Tensor, "pow")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        calls = (
            (lambda: tensor.pow(2), (tensor, 2), None),
            (lambda: tensor.pow(exponent=2), (tensor,), {"exponent": 2}),
            (lambda: tensor.pow(3), (tensor, 3), None),
            (lambda: tensor.pow(tensor_exponent), (tensor, tensor_exponent), None),
        )
        for call, expected_args, expected_kwargs in calls:
            mode = RecordingMode()
            with mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(len(args), len(expected_args))
            for actual, expected in zip(args, expected_args, strict=True):
                if isinstance(expected, torch.Tensor):
                    self.assertIs(actual, expected)
                else:
                    self.assertEqual(actual, expected)
            self.assertEqual(kwargs, expected_kwargs)

        invalid_mode = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid_mode:
                tensor.pow(object())
        self.assertEqual(invalid_mode.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.pow(exponent=2)
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [4.0, 9.0])
        forwarded.sum().backward()
        self.assertEqual(tensor.grad.tolist(), [4.0, -6.0])

        declining = RecordingMode(NotImplemented)
        with self.assertRaisesRegex(
            TypeError, r"^Multiple dispatch failed for 'torch\.Tensor\.pow'"
        ):
            with declining:
                tensor.pow(2)
        self.assertEqual(len(declining.calls), 1)

    def test_unsupported_exponents_fail_without_mutating_inputs(self):
        source = torch.tensor([2.0, -3.0], requires_grad=True)
        tensor_exponent = torch.tensor(2.0)
        source_snapshot = (
            np.asarray(source.detach()).copy().view(np.uint32),
            source.shape,
            source.stride(),
            source.storage_offset(),
            source.data_ptr(),
            source.requires_grad,
            source.is_leaf,
        )
        exponent_snapshot = np.asarray(tensor_exponent).copy().view(np.uint32)

        unsupported = (
            3,
            2.0000000000000004,
            True,
            np.float32(3),
            2 + 0j,
            tensor_exponent,
        )
        for exponent in unsupported:
            with self.subTest(exponent=repr(exponent)):
                with self.assertRaisesRegex(
                    NotImplementedError, f"^{re.escape(UNSUPPORTED_EXPONENT)}$"
                ):
                    source.pow(exponent)

        np.testing.assert_array_equal(
            np.asarray(source.detach()).view(np.uint32), source_snapshot[0]
        )
        self.assertEqual(
            (
                source.shape,
                source.stride(),
                source.storage_offset(),
                source.data_ptr(),
                source.requires_grad,
                source.is_leaf,
            ),
            source_snapshot[1:],
        )
        np.testing.assert_array_equal(
            np.asarray(tensor_exponent).view(np.uint32), exponent_snapshot
        )
        self.assertIsNone(source.grad)

        extreme = torch.zeros((0,)).reshape((0, sys.maxsize, 3))
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            extreme.pow(2)
        for exponent in (3, tensor_exponent):
            with self.assertRaisesRegex(
                NotImplementedError, f"^{re.escape(UNSUPPORTED_EXPONENT)}$"
            ):
                extreme.pow(exponent)

    def test_top_level_inplace_and_operator_forms_remain_unsupported(self):
        tensor = torch.tensor([2.0])
        self.assertFalse(hasattr(torch, "pow"))
        self.assertNotIn("pow", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "pow_"))
        self.assertFalse(hasattr(tensor, "pow_"))
        with self.assertRaises(TypeError):
            tensor**2


if __name__ == "__main__":
    unittest.main()
