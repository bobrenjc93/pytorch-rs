import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = """
mean(dim=None, keepdim=False, *, dtype=None) -> Tensor

See :func:`torch.mean`
"""

EXPECTED_OVERLOADS = (
    "but expected one of:\n"
    " * (*, torch.dtype dtype = None)\n"
    " * (tuple of ints dim, bool keepdim = False, *, "
    "torch.dtype dtype = None)\n"
)


def bits_to_tensor(bits, shape):
    values = np.asarray(bits, dtype=np.uint32).view(np.float32)
    return torch.tensor(memoryview(values), dtype=torch.float32).reshape(shape)


def scalar_bits(tensor):
    return np.asarray(tensor, dtype=np.float32).view(np.uint32).item()


class TensorMeanTests(unittest.TestCase):
    def assert_scalar(self, value, expected_bits, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(value.shape, ())
            self.assertEqual(value.stride(), ())
            self.assertEqual(value.storage_offset(), 0)
            self.assertEqual(value.numel(), 1)
            self.assertTrue(value.is_contiguous())
            self.assertIs(value.dtype, torch.float32)
            self.assertEqual(value.device, torch.device("cpu"))
        with self.subTest(case=case, value=True):
            self.assertEqual(scalar_bits(value), expected_bits)

    @staticmethod
    def supported_calls(source):
        return (
            ("default", lambda: source.mean()),
            ("dtype none", lambda: source.mean(dtype=None)),
            ("dtype float32", lambda: source.mean(dtype=torch.float32)),
            ("dtype float alias", lambda: source.mean(dtype=torch.float)),
        )

    def test_dtype_only_forms_use_full_sum_and_scalar_scale(self):
        dense = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        noncontiguous = dense.transpose(0, 2)
        cancellation = (
            np.where(
                np.arange(120) % 2 == 0, np.float32(1.0e8), np.float32(-1.0e8)
            )
            + (np.arange(120, dtype=np.float32) % 7)
        ).reshape(3, 40)
        noncontiguous_cancellation = torch.tensor(cancellation.tolist()).transpose(0, 1)
        cases = (
            ("scalar", torch.tensor(-3.5), np.float32(-3.5).view(np.uint32).item()),
            ("negative zero", torch.tensor(-0.0), 0x0000_0000),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1], 0xFFC0_0000),
            ("contiguous offset", dense[1], np.float32(17.5).view(np.uint32).item()),
            ("offset", noncontiguous[1], np.float32(11.0).view(np.uint32).item()),
            (
                "noncontiguous",
                noncontiguous,
                np.float32(11.5).view(np.uint32).item(),
            ),
            (
                "finite cancellation",
                torch.tensor([0.0, 0.0, 1.0, 3.0, 123456789.0]),
                0x4BBC_614F,
            ),
            ("noncontiguous cancellation", noncontiguous_cancellation, 0x0000_0000),
            ("positive NaN", bits_to_tensor([0x7FC1_2345, 0x3F80_0000], [2]), 0x7FC1_2345),
            ("negative NaN", bits_to_tensor([0xFFC5_4321, 0x3F80_0000], [2]), 0xFFC5_4321),
            ("infinity", torch.tensor([float("inf"), 1.0]), 0x7F80_0000),
            (
                "opposite infinities",
                torch.tensor([float("inf"), -float("inf")]),
                0xFFC0_0000,
            ),
        )

        for name, source, expected_bits in cases:
            for form, call in self.supported_calls(source):
                self.assert_scalar(call(), expected_bits, case=(name, form))

    def test_dtype_forms_preserve_autograd_accumulation_and_empty_gradients(self):
        leaf = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], requires_grad=True
        )
        loss = leaf.transpose(0, 1).mean(dtype=torch.float32)
        self.assertTrue(loss.requires_grad)
        self.assertFalse(loss.is_leaf)
        loss.backward()
        loss.backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.full((2, 3), np.float32(1.0 / 3.0), dtype=np.float32),
        )

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty_loss = empty.transpose(0, 2).mean(dtype=None)
        self.assertEqual(scalar_bits(empty_loss), 0xFFC0_0000)
        empty_loss.backward()
        empty_loss.backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.tolist(), [[], []])

        with torch.no_grad():
            untracked = leaf.mean(dtype=torch.float)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertTrue(leaf.mean(dtype=torch.float32).requires_grad)

    def test_descriptor_documentation_and_unbound_dtype_calls(self):
        tensor = torch.tensor([1.0, 2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "mean")
        bound = tensor.mean

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "mean")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(descriptor(tensor).item(), 1.5)
        self.assertEqual(descriptor(tensor, dtype=None).item(), 1.5)
        self.assertEqual(descriptor(tensor, dtype=torch.float32).item(), 1.5)

    def test_invalid_dtype_and_argument_errors_document_the_supported_boundary(self):
        tensor = torch.ones((2, 3))
        invalid = "mean() received an invalid combination of arguments - got "
        cases = (
            (
                lambda: tensor.mean(dtype=1),
                f"{invalid}(dtype=int, ), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: tensor.mean(dtype=object()),
                f"{invalid}(dtype=object, ), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: tensor.mean(torch.float32),
                f"{invalid}(torch.dtype), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: tensor.mean(extra=True),
                f"{invalid}(extra=bool, ), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: tensor.mean(0, False, torch.float32),
                "mean() takes from 1 to 2 positional arguments but 3 were given",
            ),
            (
                lambda: tensor.mean(out=None),
                f"{invalid}(out=NoneType, ), {EXPECTED_OVERLOADS}",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_dimension_keepdim_out_and_non_float32_dtype_remain_unsupported(self):
        tensor = torch.ones((2, 3))
        self.assertFalse(hasattr(torch, "bool"))
        self.assertFalse(hasattr(torch, "int64"))
        self.assertFalse(hasattr(torch, "float64"))

        cases = (
            ("positional dim", lambda: tensor.mean(0)),
            ("positional none dim", lambda: tensor.mean(None)),
            ("keyword dim", lambda: tensor.mean(dim=0)),
            ("none dim", lambda: tensor.mean(dim=None)),
            ("positional keepdim", lambda: tensor.mean(0, False)),
            ("keyword keepdim", lambda: tensor.mean(dim=0, keepdim=True)),
            ("out", lambda: tensor.mean(out=tensor)),
        )
        for case, call in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^mean\(\) received an invalid combination of arguments",
                ):
                    call()


if __name__ == "__main__":
    unittest.main()
