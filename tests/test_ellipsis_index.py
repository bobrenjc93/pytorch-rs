import gc
import inspect
import unittest

import numpy as np
import torch_rs as torch


class TensorBareEllipsisIndexTests(unittest.TestCase):
    def assert_alias(self, source, alias):
        self.assertIsNot(alias, source)
        self.assertEqual(alias.shape, source.shape)
        self.assertEqual(alias.stride(), source.stride())
        self.assertEqual(alias.storage_offset(), source.storage_offset())
        self.assertEqual(alias.data_ptr(), source.data_ptr())
        self.assertTrue(alias.is_set_to(source))
        self.assertIs(alias.dtype, source.dtype)
        self.assertEqual(alias.device, source.device)
        np.testing.assert_array_equal(np.asarray(alias), np.asarray(source))
        np.testing.assert_array_equal(
            np.asarray(alias).reshape(-1).view(np.uint32),
            np.asarray(source).reshape(-1).view(np.uint32),
        )

    def test_scalar_empty_offset_and_noncontiguous_layouts_are_distinct_aliases(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        cases = (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3))),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 2)[1]),
        )

        for case, source in cases:
            with self.subTest(case=case):
                self.assert_alias(source, source[...])

    def test_alias_autograd_gradient_no_grad_and_source_lifetime(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        source = (leaf * 2.0).transpose(0, 1)
        alias = source[...]
        self.assert_alias(source, alias)
        self.assertTrue(alias.requires_grad)
        self.assertFalse(alias.is_leaf)
        self.assertEqual(alias.output_nr, 0)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(alias),
            ", grad_fn=<AliasBackward0>",
        )

        del source
        gc.collect()
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        (alias * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 6.0], [4.0, 8.0]])

        scalar = torch.tensor(-2.0, requires_grad=True)
        scalar_alias = scalar[...]
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(scalar_alias),
            ", grad_fn=<AliasBackward0>",
        )
        (scalar_alias * 7.0).backward()
        self.assertEqual(scalar.grad.item(), 7.0)

        for case, source in (
            ("leaf", torch.tensor([1.0, 2.0], requires_grad=True)),
            (
                "nonleaf",
                torch.tensor([1.0, 2.0], requires_grad=True) * 2.0,
            ),
        ):
            with self.subTest(case=case):
                with torch.no_grad():
                    alias = source[...]
                self.assert_alias(source, alias)
                self.assertTrue(alias.requires_grad)
                self.assertTrue(alias.is_leaf)
                self.assertEqual(alias.output_nr, 0)
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(alias),
                    ", requires_grad=True",
                )

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty_alias = empty[...]
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(empty_alias),
            ", grad_fn=<AliasBackward0>",
        )
        empty_alias.sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])

        def retain_temporary_alias():
            temporary = torch.tensor(
                np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
            )
            return temporary.transpose(0, 2)[1][...]

        retained = retain_temporary_alias()
        gc.collect()
        self.assertEqual(retained.shape, (3, 2))
        self.assertEqual(retained.stride(), (4, 12))
        self.assertEqual(retained.storage_offset(), 1)
        self.assertEqual(retained.tolist(), [[1.0, 13.0], [5.0, 17.0], [9.0, 21.0]])

    def test_integer_indexing_and_deliberately_unsupported_forms_are_unchanged(self):
        tensor = torch.tensor(
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[4.0, 5.0], [6.0, 7.0]],
            ]
        )
        self.assertEqual(tensor[-1].tolist(), [[4.0, 5.0], [6.0, 7.0]])
        self.assertEqual(tensor[1, -1].tolist(), [6.0, 7.0])
        self.assertTrue(tensor[()].is_set_to(tensor))

        unsupported = (
            slice(None),
            None,
            (Ellipsis,),
            (0, Ellipsis),
            (Ellipsis, 0),
        )
        for index in unsupported:
            with self.subTest(index=index):
                with self.assertRaisesRegex(IndexError, "only integers"):
                    tensor[index]

    def test_tensorbase_descriptor_dispatches_mode_before_parsing(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        descriptor = inspect.getattr_static(torch.Tensor, "__getitem__")
        self.assertEqual(descriptor.__qualname__, "TensorBase.__getitem__")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__text_signature__, "($self, key, /)")

        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor[...]
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], tensor)
        self.assertIs(args[1], Ellipsis)
        self.assertIsNone(kwargs)

        deferred = RecordingMode()
        with deferred:
            self.assertIs(tensor[slice(None)], marker)
        self.assertEqual(len(deferred.calls), 1)


if __name__ == "__main__":
    unittest.main()
