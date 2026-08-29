import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = """
sum(dim=None, keepdim=False, dtype=None) -> Tensor

See :func:`torch.sum`
"""

EXPECTED_OVERLOADS = (
    "but expected one of:\n"
    " * (*, torch.dtype dtype = None)\n"
    " * (tuple of ints dim, bool keepdim = False, *, "
    "torch.dtype dtype = None)\n"
)


class TensorSumTests(unittest.TestCase):
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
            ("default", lambda: source.sum()),
            ("dtype none", lambda: source.sum(dtype=None)),
            ("dtype float32", lambda: source.sum(dtype=torch.float32)),
            ("dtype float alias", lambda: source.sum(dtype=torch.float)),
        )

    def test_dtype_only_forms_reuse_full_reduction_values_and_metadata(self):
        dense = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        noncontiguous = dense.transpose(0, 2)
        cases = (
            ("scalar", torch.tensor(-3.5), np.float32(-3.5)),
            ("negative zero", torch.tensor(-0.0), np.float32(0.0)),
            (
                "empty",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                np.float32(0.0),
            ),
            ("contiguous offset", dense[1], np.float32(210.0)),
            ("offset", noncontiguous[1], np.float32(66.0)),
            ("noncontiguous", noncontiguous, np.float32(276.0)),
        )

        for name, source, expected in cases:
            for form, call in self.supported_calls(source):
                self.assert_scalar(call(), expected, case=(name, form))

    def test_dtype_forms_preserve_autograd_accumulation_and_empty_gradients(self):
        leaf = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], requires_grad=True
        )
        loss = leaf.transpose(0, 1).sum(dtype=torch.float32)
        self.assertTrue(loss.requires_grad)
        self.assertFalse(loss.is_leaf)
        loss.backward()
        loss.backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty.transpose(0, 2).sum(dtype=None).backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.tolist(), [[], []])

        with torch.no_grad():
            untracked = leaf.sum(dtype=torch.float)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertTrue(leaf.sum(dtype=torch.float32).requires_grad)

    def test_rank_seven_sum_covers_offsets_singletons_empty_backward_and_no_grad(self):
        values = np.arange(384, dtype=np.float32).reshape(
            2, 2, 3, 2, 2, 2, 2, 2
        )
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        view = leaf[1].permute(3, 1, 6, 0, 4, 2, 5)
        self.assertFalse(view.is_contiguous())
        self.assertGreater(view.storage_offset(), 0)
        self.assert_scalar(
            view.sum(), np.float32(values[1].sum()), case="rank-7 offset"
        )

        loss = view.sum(dtype=torch.float32)
        loss.backward()
        loss.backward()
        expected_grad = np.zeros_like(values)
        expected_grad[1] = np.float32(2.0)
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected_grad)

        singleton_values = np.arange(96, dtype=np.float32).reshape(
            2, 2, 1, 3, 2, 1, 2, 2
        )
        singleton = torch.tensor(singleton_values.tolist())[1].permute(
            2, 0, 3, 5, 6, 4, 1
        )
        self.assertFalse(singleton.is_contiguous())
        self.assert_scalar(
            singleton.sum(),
            np.float32(singleton_values[1].sum()),
            case="rank-7 singleton",
        )

        empty = torch.zeros((2, 2, 0, 3, 4, 2, 1, 2), requires_grad=True)
        empty_view = empty[1].permute(6, 1, 0, 5, 4, 3, 2)
        self.assert_scalar(empty_view.sum(), np.float32(0.0), case="rank-7 empty")
        empty_view.sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.numel(), 0)

        no_grad_leaf = torch.ones((2, 2, 3, 2, 2, 2, 2, 2), requires_grad=True)
        no_grad_view = no_grad_leaf[1].permute(6, 4, 2, 0, 5, 3, 1)
        with torch.no_grad():
            no_grad_sum = no_grad_view.sum(dtype=torch.float)
        self.assert_scalar(no_grad_sum, np.float32(192.0), case="rank-7 no_grad")
        self.assertFalse(no_grad_sum.requires_grad)
        self.assertTrue(no_grad_sum.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)

    def test_descriptor_documentation_and_unbound_dtype_calls(self):
        tensor = torch.tensor([1.0, 2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "sum")
        bound = tensor.sum

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "sum")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(descriptor(tensor).item(), 3.0)
        self.assertEqual(descriptor(tensor, dtype=None).item(), 3.0)
        self.assertEqual(descriptor(tensor, dtype=torch.float32).item(), 3.0)

    def test_invalid_dtype_and_argument_errors_match_the_pytorch_overload(self):
        tensor = torch.ones((2, 3))
        invalid = "sum() received an invalid combination of arguments - got "
        cases = (
            (
                lambda: tensor.sum(dtype=1),
                f"{invalid}(dtype=int, ), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: tensor.sum(dtype=object()),
                f"{invalid}(dtype=object, ), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: tensor.sum(torch.float32),
                f"{invalid}(torch.dtype), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: tensor.sum(extra=True),
                f"{invalid}(extra=bool, ), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: tensor.sum(0, False, torch.float32),
                "sum() takes from 1 to 2 positional arguments but 3 were given",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_dimension_keepdim_and_out_forms_remain_unsupported(self):
        tensor = torch.ones((2, 3))
        cases = (
            ("positional dim", lambda: tensor.sum(0)),
            ("positional none dim", lambda: tensor.sum(None)),
            ("keyword dim", lambda: tensor.sum(dim=0)),
            ("none dim", lambda: tensor.sum(dim=None)),
            ("positional keepdim", lambda: tensor.sum(0, False)),
            ("keyword keepdim", lambda: tensor.sum(dim=0, keepdim=True)),
            ("out", lambda: tensor.sum(out=None)),
        )
        for case, call in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^sum\(\) received an invalid combination of arguments",
                ):
                    call()


if __name__ == "__main__":
    unittest.main()
