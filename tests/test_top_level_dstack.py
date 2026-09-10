import unittest

import numpy as np
import torch_rs as torch


class TopLevelDstackTests(unittest.TestCase):
    def test_mixed_ranks_and_depth_order(self):
        result = torch.dstack(
            tensors=(torch.tensor(-0.0), torch.tensor([2.0]), torch.tensor([[3.0]])),
            out=None,
        )
        self.assertEqual(result.shape, (1, 1, 3))
        np.testing.assert_array_equal(
            np.asarray(result).view(np.uint32),
            np.asarray([[[-0.0, 2.0, 3.0]]], dtype=np.float32).view(np.uint32),
        )
        result = torch.dstack([torch.tensor([1., 2.]), torch.tensor([[3., 4.]])])
        self.assertEqual(result.tolist(), [[[1., 3.], [2., 4.]]])
        self.assertEqual(result.stride(), (4, 2, 1))
        self.assertEqual(result.storage_offset(), 0)

    def test_single_input_storage_is_independent(self):
        for data in (2.0, [1.0, 2.0], [[1.0], [2.0]]):
            with self.subTest(data=data):
                source = torch.tensor(data)
                result = torch.dstack([source])
                self.assertNotEqual(result.data_ptr(), source.data_ptr())
                np.asarray(result).reshape(-1)[0] = 17.0
                self.assertEqual(source.tolist(), data)

    def test_empty_dimensions_keep_normalized_shape(self):
        for shape, expected in (
            ((0,), (1, 0, 2)),
            ((0, 3), (0, 3, 2)),
            ((2, 0), (2, 0, 2)),
            ((0, 0), (0, 0, 2)),
        ):
            with self.subTest(shape=shape):
                source = torch.zeros(shape)
                result = torch.dstack([source, source])
                self.assertEqual(result.shape, expected)
                self.assertEqual(result.numel(), 0)
                self.assertTrue(result.is_contiguous())
                self.assertFalse(result.is_set_to(source))
        with self.assertRaisesRegex(RuntimeError, "Sizes of tensors must match except in dimension 2"):
            torch.dstack([torch.tensor([]), torch.zeros((2, 0))])

    def test_concrete_out_and_higher_input_ranks_are_rejected(self):
        source = torch.tensor([1.0])
        output = torch.tensor([[[9.0]]])
        with self.assertRaisesRegex(RuntimeError, "dstack.*'out' argument is not supported"):
            torch.dstack([source], out=output)
        self.assertEqual(source.tolist(), [1.0])
        self.assertEqual(output.tolist(), [[[9.0]]])
        for rank in (3, 4):
            for inputs in ([torch.zeros((1,) * rank)], [source, torch.zeros((0,) * rank)]):
                with self.subTest(rank=rank, count=len(inputs)):
                    with self.assertRaisesRegex(NotImplementedError, "dstack.*only exact native CPU float32"):
                        torch.dstack(inputs)

    def test_mode_intercepts_unsupported_native_calls_after_binding(self):
        marker = object()
        calls = []

        class Mode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return marker

        inputs = [torch.zeros((1, 1, 1))]
        output = torch.tensor([0.0])
        with Mode():
            self.assertIs(torch.dstack(tensors=inputs, out=output), marker)
            with self.assertRaises(TypeError):
                torch.dstack(inputs, dim=2)
        self.assertEqual(calls, [(torch.dstack, (), (), {"tensors": inputs, "out": output})])

    def test_public_export(self):
        self.assertEqual(torch.__all__.count("dstack"), 1)
        self.assertEqual(torch.dstack.__name__, "dstack")
        self.assertEqual(torch.dstack.__module__, "torch")
        self.assertEqual(torch.dstack.__qualname__, "_VariableFunctionsClass.dstack")
        self.assertIn("dstack(tensors, *, out=None) -> Tensor", torch.dstack.__doc__)
        self.assertIs(torch._C._VariableFunctionsClass.dstack, torch.dstack)
        self.assertFalse(hasattr(torch.functional, "dstack"))
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["dstack"], torch.dstack)


if __name__ == "__main__":
    unittest.main()
