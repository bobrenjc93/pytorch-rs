import inspect
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class UnflattenReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), str(expected.device))
        np.testing.assert_array_equal(np.asarray(actual.detach()), expected.detach().numpy())

    def test_layouts_aliases_and_weighted_first_order_gradients(self):
        # Each case starts from a leaf so offset/transpose gradient mappings are
        # checked all the way back to the original storage, with unequal weights.
        cases = (
            ((2, 12), lambda x: x, 1, (3, -1)),
            ((2, 12), lambda x: x.transpose(0, 1), 0, (3, 4)),
            ((2, 12), lambda x: x.transpose(0, 1), -1, (1, 2, 1)),
            ((3, 4, 6), lambda x: x[1], -1, (2, 3)),
            ((3, 4, 6), lambda x: x.transpose(0, 1)[1], 1, (2, -1)),
            ((2, 12), lambda x: x[:, 2:10], 1, (2, 4)),
            ((2, 1, 6), lambda x: x.transpose(0, 1), 0, (1, 1, 1)),
            ((2, 1, 6), lambda x: x, 2, (6,)),
            ((0,), lambda x: x, 0, (2, -1)),
            ((2, 0, 6), lambda x: x.transpose(0, 2), 1, (0, 3)),
            ((2, 0, 6), lambda x: x.transpose(0, 2), 0, (2, -1)),
            ((3, 6), lambda x: x[3:3], 1, (2, 3)),
            ((3, 0, 6), lambda x: x[2], 1, (2, 3)),
        )
        for shape, transform, dim, sizes in cases:
            with self.subTest(shape=shape, dim=dim, sizes=sizes, transform=transform):
                values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                actual_leaf = torch.tensor(values.tolist(), requires_grad=True) if values.size else torch.zeros(shape, requires_grad=True)
                expected_leaf = reference_torch.tensor(values, requires_grad=True) if values.size else reference_torch.zeros(shape, requires_grad=True)
                source, reference = transform(actual_leaf), transform(expected_leaf)
                self.assert_matches(source, reference)
                result, expected = source.unflatten(dim, sizes), reference.unflatten(dim, sizes)
                self.assert_matches(result, expected)
                self.assertIsNot(result, source)
                self.assertEqual(result.data_ptr(), source.data_ptr())
                self.assertEqual(expected.data_ptr(), reference.data_ptr())
                # Matching shape/strides plus storage identity checks aliases
                # even when empty data_ptr() values are both null.
                identity = source.unflatten(dim, (source.shape[dim],))
                reference_identity = reference.unflatten(dim, (reference.shape[dim],))
                self.assertEqual(identity.is_set_to(source), reference_identity.is_set_to(reference))
                weights = np.arange(result.numel(), dtype=np.float32).reshape(result.shape) + 1
                actual_weights = torch.tensor(weights.tolist()) if weights.size else torch.zeros(result.shape)
                (result * actual_weights).sum().backward()
                (expected * reference_torch.tensor(weights)).sum().backward()
                self.assertEqual(actual_leaf.grad.shape, expected_leaf.grad.shape)
                np.testing.assert_array_equal(np.asarray(actual_leaf.grad), expected_leaf.grad.numpy())
                with torch.no_grad(), reference_torch.no_grad():
                    self.assert_matches(source.unflatten(dim, sizes), reference.unflatten(dim, sizes))

    def test_positional_keyword_and_integer_forms(self):
        actual, expected = torch.zeros((2, 6)), reference_torch.zeros((2, 6))
        self.assertEqual(inspect.signature(torch.Tensor.unflatten), inspect.signature(reference_torch.Tensor.unflatten))
        for args, kwargs in (
            ((1, (2, 3)), {}),
            ((-1,), {"sizes": [2, -1]}),
            ((), {"sizes": (2, 3), "dim": 1}),
            ((np.int64(1), (np.int32(2), 3)), {}),
        ):
            with self.subTest(args=args, kwargs=kwargs):
                self.assert_matches(actual.unflatten(*args, **kwargs), expected.unflatten(*args, **kwargs))
        self.assert_matches(actual.unflatten(1, torch.Size([2, 3])), expected.unflatten(1, reference_torch.Size([2, 3])))

    @unittest.skipUnless(reference_torch is not None and reference_torch.cuda.is_available(), "requires NVIDIA GPU")
    def test_native_scope_rejects_cuda(self):
        source = torch.zeros((12,), device="cuda:0")
        with self.assertRaisesRegex(NotImplementedError, "only exact native CPU float32"):
            source.unflatten(0, (3, 4))

    def assert_error_matches(self, actual, expected, args, kwargs):
        with self.assertRaises(Exception) as actual_error:
            actual.unflatten(*args, **kwargs)
        with self.assertRaises(Exception) as expected_error:
            expected.unflatten(*args, **kwargs)
        self.assertIs(type(actual_error.exception), type(expected_error.exception))
        # Some ATen unflatten diagnostics embed a build-specific C++ backtrace.
        # Compare the complete semantic message preceding that backtrace.
        def diagnostic(error):
            return str(error).split("\nException raised from ", 1)[0].rstrip('"\n')
        self.assertEqual(diagnostic(actual_error.exception), diagnostic(expected_error.exception))

    def test_binding_dimension_product_and_ambiguous_inference_errors(self):
        actual, expected = torch.zeros((2, 6)), reference_torch.zeros((2, 6))
        calls = (
            ((), {}), ((1,), {}), ((), {"sizes": (2, 3)}),
            ((1, (2, 3), 0), {}), ((1, (2, 3)), {"dim": 1}),
            ((1, (2, 3)), {"sizes": (2, 3)}),
            ((), {"dim": 1, "sizes": (2, 3), "extra": 0}),
            ((-3, (2, 3)), {}), ((2, (2, 3)), {}),
            ((2**100, (2, 3)), {}), ((True, (2, 3)), {}),
            ((None, (2, 3)), {}), ((1.0, (2, 3)), {}), (("named", (2, 3)), {}),
            ((1, (2, 4)), {}), ((-1, (4, -1)), {}),
            ((1, (-1, -1)), {}), ((1, (-2, 3)), {}), ((1, (0, -1)), {}),
            ((99, ()), {}), ((1, []), {}), ((1, None), {}),
            ((1, 6), {}), ((1, range(1, 3)), {}),
            ((1, (True, 6)), {}), ((1, (2.0, 3)), {}), ((1, (2**100,)), {}),
        )
        for args, kwargs in calls:
            with self.subTest(args=args, kwargs=kwargs):
                self.assert_error_matches(actual, expected, args, kwargs)
        for shape, dim, sizes in (
            ((), 0, (1,)), ((), -1, (1,)), ((), 1, (1,)),
            ((0, 6), 0, (0, -1)), ((0, 6), 0, (-1, 0)),
            ((0, 6), 1, (4, 2)), ((0, 6), 1, (4, -1)),
        ):
            with self.subTest(shape=shape, dim=dim, sizes=sizes):
                self.assert_error_matches(torch.zeros(shape), reference_torch.zeros(shape), (dim, sizes), {})


if __name__ == "__main__":
    unittest.main()
