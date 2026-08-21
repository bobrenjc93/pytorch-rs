import gc
import inspect
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorEllipsisIndexTests(unittest.TestCase):
    def layout_cases(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        return (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3))),
            ("offset", base[1]),
            ("offset-noncontiguous", base.transpose(0, 3)[1]),
        )

    def assert_metadata_alias(self, source, alias):
        self.assertIsNot(alias, source)
        self.assertEqual(alias.shape, source.shape)
        self.assertEqual(alias.stride(), source.stride())
        self.assertEqual(alias.storage_offset(), source.storage_offset())
        self.assertTrue(alias.is_set_to(source))
        self.assertEqual(alias.data_ptr(), source.data_ptr())
        self.assertIs(alias.dtype, source.dtype)
        self.assertEqual(alias.device, source.device)
        self.assertEqual(alias.tolist(), source.tolist())

    def assert_matching_view(self, actual, expected):
        self.assertIsNot(actual, expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertTrue(actual.is_set_to(expected))
        self.assertEqual(actual.data_ptr(), expected.data_ptr())
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertEqual(actual.tolist(), expected.tolist())

    def test_bare_ellipsis_returns_a_distinct_exact_metadata_alias(self):
        for case, source in self.layout_cases():
            with self.subTest(case=case):
                alias = source[...]
                self.assert_metadata_alias(source, alias)

        scalar = torch.tensor(-0.0)[...]
        self.assertEqual(np.asarray(scalar).view(np.uint32).item(), 0x8000_0000)

    def test_singleton_tuple_ellipsis_returns_a_distinct_exact_metadata_alias(self):
        for case, source in self.layout_cases():
            with self.subTest(case=case):
                alias = source[(Ellipsis,)]
                self.assert_metadata_alias(source, alias)

        scalar = torch.tensor(-0.0)[(Ellipsis,)]
        self.assertEqual(np.asarray(scalar).view(np.uint32).item(), 0x8000_0000)

    def test_trailing_ellipsis_reuses_integer_tuple_views(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        cases = (
            ("one-index", base, (1, Ellipsis), (1,)),
            ("partial", base, (-1, 1, Ellipsis), (-1, 1)),
            ("full-rank", base, (1, -1, -2, -3, Ellipsis), (1, -1, -2, -3)),
            ("empty", torch.zeros((2, 0, 3)), (1, Ellipsis), (1,)),
            (
                "noncontiguous",
                base.transpose(0, 3),
                (1, -1, Ellipsis),
                (1, -1),
            ),
            ("offset", base[1], (1, Ellipsis), (1,)),
        )
        for case, source, trailing, integer_only in cases:
            with self.subTest(case=case):
                self.assert_matching_view(source[trailing], source[integer_only])

    def assert_alias_autograd_gradient_and_no_grad_leaf_status(self, index):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        alias = leaf[index]
        self.assert_metadata_alias(leaf, alias)
        self.assertTrue(alias.requires_grad)
        self.assertFalse(alias.is_leaf)
        weights = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        (alias * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), weights.tolist())

        diagnostic_leaf = torch.tensor([2.0], requires_grad=True)
        with self.assertRaisesRegex(
            ValueError,
            r"^dropout probability has to be between 0 and 1, but got "
            r"tensor\(\[2\.\], grad_fn=<AliasBackward0>\)$",
        ):
            torch.nn.functional.dropout(
                None, p=diagnostic_leaf[index], training=False
            )

        no_grad_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        no_grad_source = no_grad_leaf.transpose(0, 1)
        with torch.no_grad():
            no_grad_alias = no_grad_source[index]
        self.assert_metadata_alias(no_grad_source, no_grad_alias)
        self.assertTrue(no_grad_alias.requires_grad)
        self.assertTrue(no_grad_alias.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)

    def test_bare_ellipsis_alias_autograd_gradient_and_no_grad_leaf_status(self):
        self.assert_alias_autograd_gradient_and_no_grad_leaf_status(Ellipsis)

    def test_singleton_tuple_alias_autograd_gradient_and_no_grad_leaf_status(self):
        self.assert_alias_autograd_gradient_and_no_grad_leaf_status((Ellipsis,))

    def test_trailing_ellipsis_preserves_autograd_and_no_grad_view_status(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        source = (leaf * 2.0).transpose(0, 2)
        indexed = source[1, 1, Ellipsis]
        self.assert_matching_view(indexed, source[1, 1])
        self.assertTrue(indexed.requires_grad)
        self.assertFalse(indexed.is_leaf)

        weights = torch.tensor([3.0, 5.0])
        (indexed * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[:, 1, 1] = [6.0, 10.0]
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

        no_grad_leaf = torch.tensor(values.tolist(), requires_grad=True)
        no_grad_source = no_grad_leaf.transpose(0, 2)
        with torch.no_grad():
            no_grad_indexed = no_grad_source[-1, 0, Ellipsis]
            no_grad_expected = no_grad_source[-1, 0]
        self.assert_matching_view(no_grad_indexed, no_grad_expected)
        self.assertTrue(no_grad_indexed.requires_grad)
        self.assertTrue(no_grad_indexed.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)

    def assert_alias_storage_and_autograd_survive_source_lifetime(self, index):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retained_view():
            source = torch.tensor(values.tolist()).transpose(0, 2)[1]
            return source[index]

        surviving = retained_view()
        gc.collect()
        self.assertEqual(surviving.shape, (3, 2))
        self.assertEqual(surviving.stride(), (4, 12))
        self.assertEqual(surviving.storage_offset(), 1)
        np.testing.assert_array_equal(np.asarray(surviving), values[:, :, 1].T)

        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_autograd_view():
            source = (leaf * 2.0).transpose(0, 2)[1]
            return source[index]

        tracked = retained_autograd_view()
        gc.collect()
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        (tracked * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[:, :, 1] = 2.0 * np.asarray(weights).T
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def test_bare_ellipsis_alias_survives_source_lifetime(self):
        self.assert_alias_storage_and_autograd_survive_source_lifetime(Ellipsis)

    def test_singleton_tuple_alias_survives_source_lifetime(self):
        self.assert_alias_storage_and_autograd_survive_source_lifetime((Ellipsis,))

    def test_trailing_ellipsis_view_and_autograd_survive_source_lifetime(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_view():
            source = (leaf * 2.0).transpose(0, 2)
            return source[1, 1, Ellipsis]

        surviving = retained_view()
        gc.collect()
        self.assertEqual(surviving.shape, (2,))
        self.assertEqual(surviving.stride(), (12,))
        self.assertEqual(surviving.storage_offset(), 5)
        self.assertEqual(surviving.tolist(), [10.0, 34.0])

        (surviving * torch.tensor([3.0, 5.0])).sum().backward()
        expected = np.zeros_like(values)
        expected[:, 1, 1] = [6.0, 10.0]
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def assert_dispatches_through_tensorbase_mode_before_parsing(
        self, index, integer_only=None
    ):
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
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

        mode = RecordingMode()
        with mode:
            result = source[index]
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [mode]
            )

        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs, handler_stack = mode.calls[0]
        descriptor = inspect.getattr_static(torch.Tensor, "__getitem__")
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
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        class IndexBomb:
            def __init__(self):
                self.calls = 0

            def __index__(self):
                self.calls += 1
                raise AssertionError("index parsing must be deferred")

        bomb = IndexBomb()
        mode.calls.clear()
        with mode:
            result = source[bomb]
        self.assertIs(result, marker)
        self.assertEqual(bomb.calls, 0)
        self.assertEqual(len(mode.calls), 1)
        self.assertIs(mode.calls[0][2][1], bomb)

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = source[index]
        if integer_only is None:
            self.assert_metadata_alias(source, forwarded)
        else:
            self.assert_matching_view(forwarded, source[integer_only])

    def test_bare_ellipsis_dispatches_through_tensorbase_mode_before_parsing(self):
        self.assert_dispatches_through_tensorbase_mode_before_parsing(Ellipsis)

    def test_singleton_tuple_dispatches_with_the_original_index(self):
        index = (Ellipsis,)
        self.assert_dispatches_through_tensorbase_mode_before_parsing(index)

    def test_trailing_ellipsis_dispatches_and_forwards_with_the_original_index(self):
        index = (1, Ellipsis)
        self.assert_dispatches_through_tensorbase_mode_before_parsing(index, (1,))

    def test_trailing_ellipsis_uses_integer_conversion_bounds_and_rank_rules(self):
        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        tensor = torch.zeros((2, 3, 4))
        first = IndexValue(-1)
        second = IndexValue(0)
        self.assert_matching_view(tensor[first, second, Ellipsis], tensor[-1, 0])
        self.assertEqual((first.calls, second.calls), (1, 1))

        first = IndexValue(2)
        later = IndexValue(0)
        with self.assertRaisesRegex(
            IndexError, "index 2 is out of bounds for dimension 0 with size 2"
        ):
            tensor[first, later, Ellipsis]
        self.assertEqual((first.calls, later.calls), (1, 0))

        too_many = tuple(IndexValue(0) for _ in range(4))
        with self.assertRaisesRegex(
            IndexError, "too many indices for tensor of dimension 3"
        ):
            tensor[(*too_many, Ellipsis)]
        self.assertEqual([index.calls for index in too_many], [0, 0, 0, 0])

        def error_contract(source, index):
            try:
                source[index]
            except Exception as error:
                return type(error), str(error)
            self.fail(f"index {index!r} unexpectedly succeeded")

        scalar = torch.tensor(1.0)
        empty = torch.zeros((2, 0, 3))
        cases = (
            (tensor, (2,), (2, Ellipsis)),
            (tensor, (0, 3), (0, 3, Ellipsis)),
            (tensor, (99, 0, 0, 0), (99, 0, 0, 0, Ellipsis)),
            (tensor, (1 << 100,), (1 << 100, Ellipsis)),
            (scalar, (0,), (0, Ellipsis)),
            (empty, (1, 0), (1, 0, Ellipsis)),
        )
        for source, integer_only, trailing in cases:
            with self.subTest(index=trailing):
                self.assertEqual(
                    error_contract(source, trailing),
                    error_contract(source, integer_only),
                )

    def test_integer_indexing_remains_supported_and_other_forms_stay_unsupported(self):
        tensor = torch.tensor(
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[4.0, 5.0], [6.0, 7.0]],
            ]
        )
        indexed = tensor[-1, 0]
        self.assertEqual(indexed.tolist(), [4.0, 5.0])
        self.assertEqual(indexed.stride(), (1,))
        self.assertEqual(indexed.storage_offset(), 4)

        unsupported = (
            None,
            (Ellipsis, 0),
            (0, Ellipsis, 0),
            (slice(None), Ellipsis),
            (0, slice(None), Ellipsis),
            (None, Ellipsis),
            ([0], Ellipsis),
            (Ellipsis, Ellipsis),
            (0, Ellipsis, Ellipsis),
        )
        for index in unsupported:
            with self.subTest(index=repr(index)):
                with self.assertRaisesRegex(IndexError, "only integers"):
                    tensor[index]


if __name__ == "__main__":
    unittest.main()
