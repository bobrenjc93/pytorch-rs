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

    def test_bare_ellipsis_returns_a_distinct_exact_metadata_alias(self):
        for case, source in self.layout_cases():
            with self.subTest(case=case):
                alias = source[...]
                self.assert_metadata_alias(source, alias)

        scalar = torch.tensor(-0.0)[...]
        self.assertEqual(np.asarray(scalar).view(np.uint32).item(), 0x8000_0000)

    def test_alias_autograd_gradient_and_no_grad_leaf_status(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        alias = leaf[...]
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
                None, p=diagnostic_leaf[...], training=False
            )

        no_grad_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        no_grad_source = no_grad_leaf.transpose(0, 1)
        with torch.no_grad():
            no_grad_alias = no_grad_source[...]
        self.assert_metadata_alias(no_grad_source, no_grad_alias)
        self.assertTrue(no_grad_alias.requires_grad)
        self.assertTrue(no_grad_alias.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)

    def test_alias_storage_and_autograd_survive_source_lifetime(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retained_view():
            source = torch.tensor(values.tolist()).transpose(0, 2)[1]
            return source[...]

        surviving = retained_view()
        gc.collect()
        self.assertEqual(surviving.shape, (3, 2))
        self.assertEqual(surviving.stride(), (4, 12))
        self.assertEqual(surviving.storage_offset(), 1)
        np.testing.assert_array_equal(np.asarray(surviving), values[:, :, 1].T)

        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_autograd_view():
            source = (leaf * 2.0).transpose(0, 2)[1]
            return source[...]

        tracked = retained_autograd_view()
        gc.collect()
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        (tracked * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[:, :, 1] = 2.0 * np.asarray(weights).T
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def test_bare_ellipsis_dispatches_through_tensorbase_mode_before_parsing(self):
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
            result = source[...]
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
        self.assertIs(args[1], Ellipsis)
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
            forwarded = source[...]
        self.assert_metadata_alias(source, forwarded)

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
            slice(None),
            None,
            (Ellipsis,),
            (Ellipsis, 0),
            (0, Ellipsis),
            (slice(None), Ellipsis),
        )
        for index in unsupported:
            with self.subTest(index=repr(index)):
                with self.assertRaisesRegex(IndexError, "only integers"):
                    tensor[index]


if __name__ == "__main__":
    unittest.main()
