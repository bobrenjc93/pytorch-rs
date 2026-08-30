import gc
import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorNewAxisIndexTests(unittest.TestCase):
    def layout_cases(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        return (
            ("scalar", torch.tensor(-0.0), (1,), (1,), 0),
            ("empty", torch.zeros((2, 0, 3)), (1, 2, 0, 3), (6, 3, 3, 1), 0),
            ("offset", base[1], (1, 2, 3, 4), (24, 12, 4, 1), 24),
            (
                "noncontiguous",
                base.transpose(0, 3)[1],
                (1, 2, 3, 2),
                (24, 12, 4, 24),
                1,
            ),
        )

    def trailing_layout_cases(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        return (
            ("scalar", torch.tensor(-0.0), (1,), (1,), 0),
            ("empty", torch.zeros((2, 0, 3)), (2, 0, 3, 1), (3, 3, 1, 1), 0),
            ("offset", base[1], (2, 3, 4, 1), (12, 4, 1, 1), 24),
            (
                "noncontiguous",
                base.transpose(0, 3)[1],
                (2, 3, 2, 1),
                (12, 4, 24, 1),
                1,
            ),
        )

    def assert_leading_unsqueeze(self, source, result, shape, stride, offset):
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.stride(), stride)
        self.assertEqual(result.storage_offset(), offset)
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertFalse(result.is_set_to(source))
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        self.assertEqual(result.tolist(), [source.tolist()])

    def assert_trailing_unsqueeze(self, source, result, shape, stride, offset):
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.stride(), stride)
        self.assertEqual(result.storage_offset(), offset)
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertFalse(result.is_set_to(source))
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        self.assertEqual(
            result.tolist(), np.expand_dims(np.asarray(source), axis=-1).tolist()
        )

    def test_newaxis_is_the_public_none_alias(self):
        self.assertIsNone(torch.newaxis)
        self.assertEqual(torch.__all__.count("newaxis"), 1)
        self.assertFalse(hasattr(torch._C, "newaxis"))

        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIsNone(namespace["newaxis"])

    def test_bare_none_and_torch_newaxis_use_the_leading_unsqueeze_view(self):
        for spelling, index in (("None", None), ("torch.newaxis", torch.newaxis)):
            for case, source, shape, stride, offset in self.layout_cases():
                with self.subTest(spelling=spelling, case=case):
                    result = source[index]
                    self.assert_leading_unsqueeze(
                        source, result, shape, stride, offset
                    )

        scalar = torch.tensor(-0.0)[None]
        self.assertEqual(np.asarray(scalar).view(np.uint32).item(), 0x8000_0000)

    def test_exact_trailing_none_and_newaxis_use_the_trailing_unsqueeze_view(self):
        indices = (
            ("None", (Ellipsis, None)),
            ("torch.newaxis", (Ellipsis, torch.newaxis)),
        )
        for spelling, index in indices:
            for case, source, shape, stride, offset in self.trailing_layout_cases():
                with self.subTest(spelling=spelling, case=case):
                    result = source[index]
                    self.assert_trailing_unsqueeze(
                        source, result, shape, stride, offset
                    )

        scalar = torch.tensor(-0.0)[..., None]
        self.assertEqual(np.asarray(scalar).view(np.uint32).item(), 0x8000_0000)

    @unittest.skipUnless(
        sys.maxsize == (1 << 63) - 1,
        "signed 64-bit stride wrapping requires a 64-bit Python build",
    )
    def test_extreme_empty_leading_stride_uses_signed_wrapping(self):
        non_concrete = torch.zeros((0,)).reshape((1 << 62, 0, 2))
        with self.assertRaisesRegex(
            RuntimeError,
            "SymIntArrayRef expected to contain only concrete integers",
        ):
            non_concrete[None]

        negative_boundary = torch.zeros((0,)).reshape((1 << 62, 0, 3))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^as_strided: Negative strides are not supported at the moment, "
            r"got strides: \[-4611686018427387904, 3, 3, 1\]$",
        ):
            negative_boundary[torch.newaxis]

        wrapped_negative = torch.zeros((0,)).reshape((sys.maxsize, 0, 2))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^as_strided: Negative strides are not supported at the moment, "
            r"got strides: \[-2, 2, 2, 1\]$",
        ):
            wrapped_negative[None]

        wrapped_positive = torch.zeros((0,)).reshape((sys.maxsize, 0, 3))
        result = wrapped_positive[torch.newaxis]
        self.assertEqual(result.shape, (1, sys.maxsize, 0, 3))
        self.assertEqual(result.stride(), (sys.maxsize - 2, 3, 3, 1))
        self.assertEqual(result.storage_offset(), 0)
        self.assertEqual(result.data_ptr(), wrapped_positive.data_ptr())
        self.assertFalse(result.is_set_to(wrapped_positive))

    def make_autograd_case(self, case):
        if case == "scalar":
            leaf = torch.tensor(-2.0, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return leaf, leaf

        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        if case == "offset":
            return leaf, leaf[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 3)[1]
        raise AssertionError(f"unknown case: {case}")

    def expected_gradient(self, case):
        if case == "scalar":
            return np.asarray(1.0, dtype=np.float32)
        if case == "empty":
            return np.zeros((2, 0, 3), dtype=np.float32)
        if case == "offset":
            expected = np.zeros((2, 2, 3, 4), dtype=np.float32)
            expected[1] = 1.0
            return expected
        if case == "noncontiguous":
            expected = np.zeros((2, 2, 3, 4), dtype=np.float32)
            expected[:, :, :, 1] = 1.0
            return expected
        raise AssertionError(f"unknown case: {case}")

    def test_autograd_and_no_grad_cover_every_supported_layout(self):
        for case, _, shape, stride, offset in self.layout_cases():
            with self.subTest(case=case, mode="autograd"):
                leaf, source = self.make_autograd_case(case)
                result = source[None]
                self.assert_leading_unsqueeze(
                    source, result, shape, stride, offset
                )
                self.assertTrue(result.requires_grad)
                self.assertFalse(result.is_leaf)
                weights = torch.ones(tuple(result.shape))
                (result * weights).sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(leaf.grad), self.expected_gradient(case)
                )

            with self.subTest(case=case, mode="no_grad"):
                leaf, source = self.make_autograd_case(case)
                with torch.no_grad():
                    result = source[torch.newaxis]
                self.assert_leading_unsqueeze(
                    source, result, shape, stride, offset
                )
                self.assertTrue(result.requires_grad)
                self.assertTrue(result.is_leaf)
                self.assertIsNone(leaf.grad)

        diagnostic_leaf = torch.tensor([2.0], requires_grad=True)
        with self.assertRaisesRegex(
            ValueError,
            r"^dropout probability has to be between 0 and 1, but got "
            r"tensor\(\[\[2\.\]\], grad_fn=<UnsqueezeBackward0>\)$",
        ):
            torch.nn.functional.dropout(
                None, p=diagnostic_leaf[None], training=False
            )

    def test_trailing_autograd_and_no_grad_cover_every_supported_layout(self):
        for case, _, shape, stride, offset in self.trailing_layout_cases():
            with self.subTest(case=case, mode="autograd"):
                leaf, source = self.make_autograd_case(case)
                result = source[..., None]
                self.assert_trailing_unsqueeze(
                    source, result, shape, stride, offset
                )
                self.assertTrue(result.requires_grad)
                self.assertFalse(result.is_leaf)
                weights = torch.ones(tuple(result.shape))
                (result * weights).sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(leaf.grad), self.expected_gradient(case)
                )

            with self.subTest(case=case, mode="no_grad"):
                leaf, source = self.make_autograd_case(case)
                with torch.no_grad():
                    result = source[..., torch.newaxis]
                self.assert_trailing_unsqueeze(
                    source, result, shape, stride, offset
                )
                self.assertTrue(result.requires_grad)
                self.assertTrue(result.is_leaf)
                self.assertIsNone(leaf.grad)

        diagnostic_leaf = torch.tensor([2.0], requires_grad=True)
        with self.assertRaisesRegex(
            ValueError,
            r"^dropout probability has to be between 0 and 1, but got "
            r"tensor\(\[\[2\.\]\], grad_fn=<UnsqueezeBackward0>\)$",
        ):
            torch.nn.functional.dropout(
                None, p=diagnostic_leaf[..., None], training=False
            )

    def test_storage_and_autograd_survive_source_lifetime(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retained_view():
            source = torch.tensor(values.tolist()).transpose(0, 2)[1]
            return source[None]

        surviving = retained_view()
        gc.collect()
        self.assertEqual(surviving.shape, (1, 3, 2))
        self.assertEqual(surviving.stride(), (12, 4, 12))
        self.assertEqual(surviving.storage_offset(), 1)
        np.testing.assert_array_equal(
            np.asarray(surviving), values[:, :, 1].T[None]
        )

        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_autograd_view():
            source = (leaf * 2.0).transpose(0, 2)[1]
            return source[torch.newaxis]

        tracked = retained_autograd_view()
        gc.collect()
        weights = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        (tracked * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[:, :, 1] = 2.0 * np.asarray(weights)[0].T
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def test_trailing_storage_and_autograd_survive_source_lifetime(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retained_view():
            source = torch.tensor(values.tolist()).transpose(0, 2)[1]
            return source[..., None]

        surviving = retained_view()
        gc.collect()
        self.assertEqual(surviving.shape, (3, 2, 1))
        self.assertEqual(surviving.stride(), (4, 12, 1))
        self.assertEqual(surviving.storage_offset(), 1)
        np.testing.assert_array_equal(
            np.asarray(surviving), values[:, :, 1].T[..., None]
        )

        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_autograd_view():
            source = (leaf * 2.0).transpose(0, 2)[1]
            return source[..., torch.newaxis]

        tracked = retained_autograd_view()
        gc.collect()
        weights = torch.tensor(
            [[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]
        )
        (tracked * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[:, :, 1] = 2.0 * np.asarray(weights)[..., 0].T
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def test_torch_function_mode_observes_none_before_indexing(self):
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func,
                        dispatch_types,
                        args,
                        kwargs,
                        tuple(torch.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        descriptor = inspect.getattr_static(torch.Tensor, "__getitem__")
        mode = RecordingMode()
        for case, source, _, _, _ in self.layout_cases():
            with self.subTest(case=case, mode="recording"):
                mode.calls.clear()
                with mode:
                    result = source[torch.newaxis]
                    self.assertEqual(
                        torch.overrides._get_current_function_mode_stack(), [mode]
                    )
                self.assertIs(result, marker)
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs, handler_stack = mode.calls[0]
                self.assertIs(type(function), types.WrapperDescriptorType)
                self.assertIs(function, descriptor)
                self.assertEqual(function.__qualname__, "TensorBase.__getitem__")
                self.assertEqual(function.__objclass__.__name__, "TensorBase")
                self.assertEqual(function.__objclass__.__module__, "torch._C")
                self.assertEqual(dispatch_types, ())
                self.assertEqual(len(args), 2)
                self.assertIs(args[0], source)
                self.assertIsNone(args[1])
                self.assertIsNone(kwargs)
                self.assertEqual(handler_stack, ())
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), []
                )

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        for case, source, shape, stride, offset in self.layout_cases():
            with self.subTest(case=case, mode="forwarding"):
                with ForwardingMode():
                    result = source[None]
                self.assert_leading_unsqueeze(
                    source, result, shape, stride, offset
                )

    def test_torch_function_mode_observes_the_exact_trailing_tuple(self):
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func,
                        dispatch_types,
                        args,
                        kwargs,
                        tuple(torch.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        descriptor = inspect.getattr_static(torch.Tensor, "__getitem__")
        mode = RecordingMode()
        for case, source, _, _, _ in self.trailing_layout_cases():
            with self.subTest(case=case, mode="recording"):
                index = (Ellipsis, torch.newaxis)
                mode.calls.clear()
                with mode:
                    result = source[index]
                    self.assertEqual(
                        torch.overrides._get_current_function_mode_stack(), [mode]
                    )
                self.assertIs(result, marker)
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs, handler_stack = mode.calls[0]
                self.assertIs(type(function), types.WrapperDescriptorType)
                self.assertIs(function, descriptor)
                self.assertEqual(function.__qualname__, "TensorBase.__getitem__")
                self.assertEqual(function.__objclass__.__name__, "TensorBase")
                self.assertEqual(function.__objclass__.__module__, "torch._C")
                self.assertEqual(dispatch_types, ())
                self.assertEqual(len(args), 2)
                self.assertIs(args[0], source)
                self.assertIs(args[1], index)
                self.assertIsNone(kwargs)
                self.assertEqual(handler_stack, ())
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), []
                )

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        for case, source, shape, stride, offset in self.trailing_layout_cases():
            with self.subTest(case=case, mode="forwarding"):
                with ForwardingMode():
                    result = source[..., None]
                self.assert_trailing_unsqueeze(
                    source, result, shape, stride, offset
                )

    def test_other_newaxis_forms_remain_unsupported_and_public_unsqueeze_is_separate(
        self,
    ):
        tensor = torch.ones((2, 3))
        leading_mixed = (
            (None,),
            (None, None),
            (None, 0),
            (None, Ellipsis),
            (slice(None), None),
            (0, None),
        )
        for index in leading_mixed:
            with self.subTest(index=repr(index)):
                with self.assertRaisesRegex(IndexError, "only integers"):
                    tensor[index]

        repeated_or_extended = (
            (Ellipsis, None, None),
            (Ellipsis, None, 0),
            (Ellipsis, 0, None),
        )
        for index in repeated_or_extended:
            with self.subTest(index=repr(index)):
                with self.assertRaises(IndexError):
                    tensor[index]

        self.assertTrue(hasattr(torch, "unsqueeze"))
        self.assertIn("unsqueeze", torch.__all__)
        self.assertTrue(hasattr(torch.Tensor, "unsqueeze"))
        self.assertTrue(tensor.unsqueeze(0).is_set_to(tensor[None]))
        self.assertTrue(torch.unsqueeze(tensor, -1).is_set_to(tensor[..., torch.newaxis]))


if __name__ == "__main__":
    unittest.main()
