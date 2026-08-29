import gc
import re
import unittest

import numpy as np
import torch_rs as torch


class UnsqueezeTests(unittest.TestCase):
    def endpoint_cases(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        return (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3))),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 3)[1]),
        )

    def assert_unsqueeze_view(self, actual, source, *, axis):
        self.assertIsNot(actual, source)
        self.assertEqual(actual.data_ptr(), source.data_ptr())
        self.assertEqual(actual.storage_offset(), source.storage_offset())
        self.assertFalse(actual.is_set_to(source))
        self.assertIs(actual.dtype, source.dtype)
        self.assertEqual(actual.device, source.device)
        expected = np.expand_dims(np.asarray(source), axis=axis)
        np.testing.assert_array_equal(np.asarray(actual), expected)

    def test_method_and_top_level_front_and_back_are_shared_storage_views(self):
        for case, source in self.endpoint_cases():
            rank = source.ndim
            for form, view in (
                ("method front", source.unsqueeze(0)),
                ("method negative front", source.unsqueeze(-(rank + 1))),
                ("top-level front", torch.unsqueeze(source, 0)),
                ("top-level keyword front", torch.unsqueeze(input=source, dim=0)),
            ):
                with self.subTest(case=case, form=form):
                    self.assert_unsqueeze_view(view, source, axis=0)

            for form, view in (
                ("method back", source.unsqueeze(rank)),
                ("method negative back", source.unsqueeze(-1)),
                ("top-level back", torch.unsqueeze(source, rank)),
                ("top-level keyword back", torch.unsqueeze(input=source, dim=-1)),
            ):
                with self.subTest(case=case, form=form):
                    self.assert_unsqueeze_view(view, source, axis=-1)

    def test_endpoint_strides_offsets_and_source_lifetime_are_preserved(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        offset = base[1]
        noncontiguous = base.transpose(0, 3)[1]
        zero_width = torch.zeros((0, 3))

        expected = (
            (offset.unsqueeze(0), offset, (1, 2, 3, 4), (24, 12, 4, 1), 24),
            (offset.unsqueeze(-1), offset, (2, 3, 4, 1), (12, 4, 1, 1), 24),
            (
                noncontiguous.unsqueeze(0),
                noncontiguous,
                (1, 2, 3, 2),
                (24, 12, 4, 24),
                1,
            ),
            (
                torch.unsqueeze(noncontiguous, -1),
                noncontiguous,
                (2, 3, 2, 1),
                (12, 4, 24, 1),
                1,
            ),
            (
                zero_width.unsqueeze(0),
                zero_width,
                (1, 0, 3),
                (0, 3, 1),
                0,
            ),
        )
        for view, source, shape, stride, offset in expected:
            with self.subTest(shape=shape):
                self.assertEqual(view.shape, shape)
                self.assertEqual(view.stride(), stride)
                self.assertEqual(view.storage_offset(), offset)
                self.assertEqual(view.data_ptr(), source.data_ptr())

        def retained_view():
            local = torch.tensor(values.tolist()).transpose(0, 3)[1]
            return torch.unsqueeze(local, 0)

        surviving = retained_view()
        gc.collect()
        self.assertEqual(surviving.shape, (1, 2, 3, 2))
        self.assertEqual(surviving.stride(), (24, 12, 4, 24))
        self.assertEqual(surviving.storage_offset(), 1)
        np.testing.assert_array_equal(
            np.asarray(surviving), values.transpose(3, 1, 2, 0)[1][None]
        )

    def test_index_protocol_dimensions(self):
        class IndexOnly:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        class IntSubclass(int):
            pass

        source = torch.zeros((2, 3, 4))
        self.assertEqual(source.unsqueeze(IndexOnly(0)).shape, (1, 2, 3, 4))
        self.assertEqual(source.unsqueeze(np.int64(-1)).shape, (2, 3, 4, 1))
        self.assertEqual(torch.unsqueeze(source, IntSubclass(3)).shape, (2, 3, 4, 1))
        with self.assertRaisesRegex(ValueError, "Overflow when unpacking long long"):
            source.unsqueeze(IndexOnly(2**100))

    def test_autograd_empty_and_no_grad_views(self):
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        weights = np.linspace(-2.0, 3.0, num=6, dtype=np.float32).reshape(1, 2, 3)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        view = torch.unsqueeze(leaf, 0)
        self.assertTrue(view.requires_grad)
        self.assertFalse(view.is_leaf)
        self.assertEqual(view.data_ptr(), leaf.data_ptr())

        (view * torch.tensor(weights.tolist())).sum().backward()
        np.testing.assert_array_equal(np.asarray(leaf.grad), weights.reshape(2, 3))

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty.unsqueeze(-1).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(empty.grad), np.zeros((2, 0, 3), dtype=np.float32)
        )

        with torch.no_grad():
            untracked = leaf.unsqueeze(-1)
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.shape, (2, 3, 1))
        self.assertEqual(untracked.stride(), (3, 1, 1))
        self.assertEqual(untracked.data_ptr(), leaf.data_ptr())

    def test_middle_out_subclass_non_tensor_and_sequence_dims_are_unsupported(self):
        source = torch.zeros((2, 3, 4))
        for call in (
            lambda: source.unsqueeze(1),
            lambda: source.unsqueeze(-2),
            lambda: torch.unsqueeze(source, 2),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    NotImplementedError, r"^unsqueeze\(\) only supports inserting"
                ):
                    call()

        for call in (
            lambda: source.unsqueeze([0]),
            lambda: source.unsqueeze(dim=(0,)),
            lambda: torch.unsqueeze(source, [0]),
            lambda: torch.unsqueeze(input=source, dim=(0,)),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(TypeError, "argument 'dim'.*must be int"):
                    call()

        for call in (
            lambda: source.unsqueeze(0, out=source),
            lambda: torch.unsqueeze(source, 0, out=source),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    TypeError, r"unexpected keyword argument 'out'"
                ):
                    call()

        class Override:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls += 1
                return object()

        for value in (Override(), [1.0], np.zeros((1,), dtype=np.float32)):
            with self.subTest(value_type=type(value)):
                with self.assertRaisesRegex(
                    TypeError, f"^{re.escape('unsqueeze() only supports exact native Tensor input')}$"
                ):
                    torch.unsqueeze(value, 0)
        self.assertEqual(Override.calls, 0)

        with self.assertRaises(IndexError):
            source.unsqueeze(4)
        with self.assertRaises(TypeError):
            source.unsqueeze(True)


if __name__ == "__main__":
    unittest.main()
