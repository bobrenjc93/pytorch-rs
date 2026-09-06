import unittest

import numpy as np
import torch_rs as torch


class TopLevelCatTests(unittest.TestCase):
    def assert_cat_matches(self, actual, expected_values, *, case):
        expected = np.asarray(expected_values, dtype=np.float32)
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, (len(expected_values),))
            self.assertEqual(actual.stride(), (1,))
            self.assertEqual(actual.storage_offset(), 0)
            self.assertTrue(actual.is_contiguous())
            self.assertFalse(actual.requires_grad)
            self.assertTrue(actual.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                expected.reshape(-1).view(np.uint32),
            )

    def test_list_tuple_empty_operands_order_metadata_and_fresh_storage(self):
        base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        offset_noncontiguous = base.transpose(0, 1)[1]
        self.assertEqual(offset_noncontiguous.tolist(), [2.0, 5.0])
        self.assertEqual(offset_noncontiguous.stride(), (3,))
        self.assertEqual(offset_noncontiguous.storage_offset(), 1)

        empty = torch.tensor([])
        tail = torch.tensor([-0.0, 7.5])
        result = torch.cat([offset_noncontiguous, empty, tail], dim=0)
        self.assert_cat_matches(result, [2.0, 5.0, -0.0, 7.5], case="list")
        self.assertFalse(result.is_set_to(offset_noncontiguous))
        self.assertFalse(result.is_set_to(tail))
        self.assertNotEqual(result.data_ptr(), offset_noncontiguous.data_ptr())
        self.assertNotEqual(result.data_ptr(), tail.data_ptr())

        tuple_result = torch.cat((torch.tensor([]), torch.tensor([3.25])), dim=-1)
        self.assert_cat_matches(tuple_result, [3.25], case="tuple dim -1")

        keyword_result = torch.cat(
            tensors=[torch.tensor([8.0]), torch.tensor([]), torch.tensor([9.0])],
            out=None,
        )
        self.assert_cat_matches(keyword_result, [8.0, 9.0], case="keywords")

        axis_result = torch.cat([torch.tensor([14.0]), torch.tensor([15.0])], axis=0)
        self.assert_cat_matches(axis_result, [14.0, 15.0], case="axis alias")

        single = torch.tensor([11.0, 13.0])
        single_result = torch.cat([single])
        self.assert_cat_matches(single_result, [11.0, 13.0], case="single tensor")
        self.assertFalse(single_result.is_set_to(single))
        self.assertNotEqual(single_result.data_ptr(), single.data_ptr())

    def test_no_grad_allows_grad_requiring_operands_without_recording(self):
        left = torch.tensor([1.0, 2.0], requires_grad=True)
        right = torch.tensor([3.0], requires_grad=True)

        with self.assertRaisesRegex(
            RuntimeError, r"^cat\(\): autograd recording is not supported$"
        ):
            torch.cat([left, right])
        self.assertIsNone(left.grad)
        self.assertIsNone(right.grad)

        with torch.no_grad():
            result = torch.cat([left, right], dim=-1)
        self.assert_cat_matches(result, [1.0, 2.0, 3.0], case="no_grad")
        self.assertIsNone(left.grad)
        self.assertIsNone(right.grad)

    def test_rejects_unsupported_inputs_before_touching_out(self):
        destination = torch.tensor([9.0, 10.0])
        before = destination.tolist()
        with self.assertRaisesRegex(
            RuntimeError, r"^cat\(\): the 'out' argument is not supported$"
        ):
            torch.cat([torch.tensor([1.0]), torch.tensor([2.0])], out=destination)
        self.assertEqual(destination.tolist(), before)

        for sequence in ([], ()):
            with self.subTest(sequence=type(sequence).__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^torch\.cat\(\): expected a non-empty list of Tensors$",
                ):
                    torch.cat(sequence)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^zero-dimensional tensor \(at position 0\) cannot be concatenated$",
        ):
            torch.cat([torch.tensor(1.0)])

        with self.assertRaisesRegex(
            NotImplementedError,
            r"^cat\(\): only exact native CPU float32 1-D Tensor inputs are supported$",
        ):
            torch.cat([torch.tensor([[1.0]])])

        with self.assertRaisesRegex(
            TypeError,
            r"^expected Tensor as element 1 in argument 0, but got int$",
        ):
            torch.cat([torch.tensor([1.0]), 1])

        with self.assertRaisesRegex(
            TypeError,
            r"^cat\(\): argument 'tensors' \(position 1\) must be tuple of Tensors, not Tensor$",
        ):
            torch.cat(torch.tensor([1.0]))

    def test_rejects_unsupported_dimensions(self):
        tensor = torch.tensor([1.0])
        for dimension in (1, -2):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"^Dimension out of range \(expected to be in range of \[-1, 0\], but got {dimension}\)$",
                ):
                    torch.cat([tensor], dim=dimension)

        with self.assertRaisesRegex(
            IndexError,
            r"^Dimension out of range \(expected to be in range of \[-1, 0\], but got 1\)$",
        ):
            torch.cat([tensor], axis=1)

        for dimension, type_name in ((True, "bool"), (None, "NoneType"), ("0", "str")):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^cat\(\): argument 'dim' must be int, not {type_name}$",
                ):
                    torch.cat([tensor], dim=dimension)

        with self.assertRaisesRegex(
            TypeError, r"^cat\(\): argument 'dim' must be int, not str$"
        ):
            torch.cat([tensor], axis="0")

        for call in (
            lambda: torch.cat([tensor], dim=0, axis=0),
            lambda: torch.cat([tensor], 0, axis=0),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^cat\(\) got an unexpected keyword argument 'axis'$",
                ):
                    call()

    def test_public_callable_surface(self):
        self.assertTrue(hasattr(torch, "cat"))
        self.assertIn("cat", torch.__all__)
        self.assertEqual(torch.__all__.count("cat"), 1)
        self.assertEqual(torch.cat.__name__, "cat")
        self.assertEqual(torch.cat.__qualname__, "_VariableFunctionsClass.cat")
        self.assertIs(torch._C._VariableFunctionsClass.cat, torch.cat)


if __name__ == "__main__":
    unittest.main()
