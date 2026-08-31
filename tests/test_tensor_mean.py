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


class TensorMeanTests(unittest.TestCase):
    def assert_scalar(self, value, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(value.shape, ())
            self.assertEqual(value.stride(), ())
            self.assertEqual(value.storage_offset(), 0)
            self.assertEqual(value.numel(), 1)
            self.assertTrue(value.is_contiguous())
            self.assertIs(value.dtype, torch.float32)
            self.assertEqual(value.device, torch.device("cpu"))
        with self.subTest(case=case, value=True):
            self.assertEqual(
                np.float32(value.item()).view(np.uint32).item(),
                np.float32(expected).view(np.uint32).item(),
            )

    @staticmethod
    def supported_calls(source):
        return (
            ("default", lambda: source.mean()),
            ("positional none dim", lambda: source.mean(None)),
            ("keyword none dim", lambda: source.mean(dim=None)),
            ("none dim keepdim false", lambda: source.mean(None, False)),
            ("dtype none", lambda: source.mean(dtype=None)),
            ("dtype float32", lambda: source.mean(dtype=torch.float32)),
            ("dtype float alias", lambda: source.mean(dtype=torch.float)),
            (
                "none dim dtype float32",
                lambda: source.mean(dim=None, keepdim=False, dtype=torch.float32),
            ),
        )

    @staticmethod
    def expected_full_mean(source):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.float32(source.sum().item()) / np.float32(source.numel())

    @staticmethod
    def rank_one_strided_vector(values, *, requires_grad=False):
        rows = len(values)
        columns = 5
        selected_column = 2
        matrix = np.full((rows, columns), np.float32(0.5), dtype=np.float32)
        matrix[:, selected_column] = np.asarray(values, dtype=np.float32)
        source = torch.tensor(
            matrix.tolist(), dtype=torch.float32, requires_grad=requires_grad
        )
        return source, source.transpose(0, 1)[selected_column]

    def test_dtype_only_forms_reuse_full_sum_values_and_metadata(self):
        dense = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        noncontiguous = dense.transpose(0, 2)
        cases = (
            ("scalar", torch.tensor(-3.5)),
            ("negative zero", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("singleton", torch.tensor([5.0])),
            ("contiguous offset", dense[1]),
            ("offset", noncontiguous[1]),
            ("noncontiguous", noncontiguous),
        )

        for name, source in cases:
            expected = self.expected_full_mean(source)
            for form, call in self.supported_calls(source):
                self.assert_scalar(call(), expected, case=(name, form))

    def test_mean_edges_cover_signed_zero_nan_infinity_and_strided_offsets(self):
        cases = (
            ("signed zero", [-0.0, 0.0, -0.0, 0.0]),
            ("nan", [1.0, np.nan, 2.0, -3.0]),
            ("positive infinity", [1.0, np.inf, 2.0, 3.0]),
            ("negative infinity", [1.0, -np.inf, 2.0, 3.0]),
            ("sequential cancellation", [1.0e20, -1.0e20, 3.0, -0.0]),
        )

        for case, values in cases:
            _, view = self.rank_one_strided_vector(values)
            self.assertEqual(view.stride(), (5,))
            self.assertEqual(view.storage_offset(), 2)
            self.assertFalse(view.is_contiguous())
            self.assert_scalar(
                view.mean(),
                self.expected_full_mean(view),
                case=("rank-one transpose-selected offset", case),
            )

    def test_autograd_empty_no_grad_and_repeated_backward(self):
        leaf = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], requires_grad=True
        )
        loss = leaf.transpose(0, 1).mean(dim=None, keepdim=False, dtype=torch.float32)
        self.assertTrue(loss.requires_grad)
        self.assertFalse(loss.is_leaf)
        loss.backward()
        np.testing.assert_allclose(
            np.asarray(leaf.grad),
            np.full((2, 3), np.float32(1.0 / 6.0), dtype=np.float32),
            rtol=0,
            atol=0,
        )
        loss.backward()
        np.testing.assert_allclose(
            np.asarray(leaf.grad),
            np.full((2, 3), np.float32(2.0 / 6.0), dtype=np.float32),
            rtol=0,
            atol=0,
        )

        repeated = torch.tensor([1.0, 2.0], requires_grad=True)
        repeated_loss = repeated.mean()
        repeated_loss.backward()
        repeated_loss.backward()
        self.assertEqual(repeated.grad.tolist(), [1.0, 1.0])

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty_loss = empty.transpose(0, 2).mean(None, False, dtype=None)
        empty_loss.backward()
        empty_loss.backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.tolist(), [[], []])

        with torch.no_grad():
            untracked = leaf.mean(dim=None, dtype=torch.float)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertTrue(leaf.mean(None, dtype=torch.float32).requires_grad)

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

    def test_invalid_dtype_and_argument_errors_match_the_pytorch_overload(self):
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
                lambda: tensor.mean(dim=None, dtype=1),
                "mean(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: tensor.mean(None, False, dtype=object()),
                "mean(): argument 'dtype' must be torch.dtype, not object",
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

    def test_dimension_keepdim_and_out_forms_remain_unsupported(self):
        tensor = torch.ones((2, 3))
        cases = (
            ("positional dim", lambda: tensor.mean(0)),
            ("keyword dim", lambda: tensor.mean(dim=0)),
            ("tuple dim", lambda: tensor.mean((0, 1))),
            ("list dim", lambda: tensor.mean([0, 1])),
            ("positional keepdim", lambda: tensor.mean(0, False)),
            ("none dim keepdim true", lambda: tensor.mean(None, True)),
            ("keyword keepdim", lambda: tensor.mean(dim=0, keepdim=True)),
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
