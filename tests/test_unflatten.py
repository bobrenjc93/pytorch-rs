import inspect
import unittest

import numpy as np
import torch_rs as torch


class UnflattenTests(unittest.TestCase):
    def test_binding_and_inference(self):
        source = torch.tensor(np.arange(24, dtype=np.float32).reshape(4, 6).tolist())
        for result in (
            source.unflatten(1, (2, 3)),
            source.unflatten(-1, sizes=[2, -1]),
            source.unflatten(sizes=torch.Size([2, 3]), dim=1),
            torch.Tensor.unflatten(source, dim=1, sizes=(2, 3)),
        ):
            self.assertEqual(result.shape, (4, 2, 3))
            self.assertEqual(result.stride(), (6, 3, 1))
            self.assertEqual(result.data_ptr(), source.data_ptr())
            np.testing.assert_array_equal(np.asarray(result), np.arange(24).reshape(4, 2, 3))
        self.assertEqual(str(inspect.signature(torch.Tensor.unflatten)), "(self, dim, sizes)")
        self.assertFalse(hasattr(torch, "unflatten"))

    def test_alias_observes_gradient_storage_updates(self):
        leaf = torch.ones((2, 6), requires_grad=True)
        (leaf * 4).sum().backward()
        source = leaf.grad.transpose(0, 1)[1:5]
        result = source.unflatten(0, (2, 2))
        self.assertEqual(result.storage_offset(), 1)
        self.assertEqual(result.stride(), (2, 1, 6))
        (leaf * 5).sum().backward()
        self.assertEqual(result.tolist(), [[[9.0, 9.0]] * 2] * 2)
        del source, leaf
        self.assertEqual(result.sum().item(), 72.0)

    def test_empty_inference_is_local_to_selected_dimension(self):
        source = torch.zeros((0, 6))
        self.assertEqual(source.unflatten(1, (2, -1)).shape, (0, 2, 3))
        self.assertEqual(source.unflatten(0, (2, -1)).shape, (2, 0, 6))
        self.assertTrue(source.unflatten(0, (0,)).is_set_to(source))
        with self.assertRaisesRegex(RuntimeError, "don't multiply up"):
            source.unflatten(1, (2, 4))
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            source.unflatten(0, (0, -1))

    def test_invalid_inputs(self):
        source = torch.ones((6,))
        for dim in (-2, 1):
            with self.assertRaises(IndexError):
                source.unflatten(dim, (2, 3))
        for sizes in ((), [], (-1, -1), (-2, 3), (2, 4)):
            with self.assertRaises(RuntimeError):
                source.unflatten(0, sizes)
        for dim, sizes in ((True, (2, 3)), (0.0, (2, 3)), (0, (True, 6)), (0, (2.0, 3)), (0, 6)):
            with self.assertRaises(TypeError):
                source.unflatten(dim, sizes)
        with self.assertRaisesRegex(RuntimeError, "tensor has no dimensions"):
            torch.tensor(1.0).unflatten(0, (1,))


if __name__ == "__main__":
    unittest.main()
