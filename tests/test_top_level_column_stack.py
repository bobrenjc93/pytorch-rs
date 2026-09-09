import unittest

import numpy as np
import torch_rs as torch


class TopLevelColumnStackTests(unittest.TestCase):
    def test_scalars_vectors_and_columns(self):
        result = torch.column_stack(tensors=(torch.tensor(-0.0), torch.tensor([2.0]), torch.tensor([[3.0, 4.0]])), out=None)
        self.assertEqual(result.shape, (1, 4))
        np.testing.assert_array_equal(
            np.asarray(result).view(np.uint32),
            np.asarray([[-0.0, 2.0, 3.0, 4.0]], dtype=np.float32).view(np.uint32),
        )
        result = torch.column_stack([torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])])
        self.assertEqual(result.tolist(), [[1.0, 3.0], [2.0, 4.0]])
        self.assertEqual(result.stride(), (2, 1))
        self.assertEqual(result.storage_offset(), 0)

    def test_vectors_are_columns_and_empty_vectors_keep_their_row_count(self):
        result = torch.column_stack([torch.tensor([1., 2.]), torch.tensor([3., 4.])])
        self.assertEqual(result.tolist(), [[1., 3.], [2., 4.]])
        self.assertEqual(torch.column_stack([torch.tensor([])]).shape, (0, 1))
        self.assertEqual(torch.column_stack([torch.zeros((2, 0))] * 2).shape, (2, 0))
        with self.assertRaisesRegex(RuntimeError, "Sizes of tensors must match except in dimension 1"):
            torch.column_stack([torch.tensor([]), torch.ones((2, 1))])

    def test_single_input_has_independent_storage(self):
        for data in (2.0, [1.0, 2.0], [[1.0], [2.0]]):
            with self.subTest(data=data):
                source = torch.tensor(data)
                result = torch.column_stack([source])
                self.assertNotEqual(result.data_ptr(), source.data_ptr())
                np.asarray(result).reshape(-1)[0] = 17.0
                self.assertEqual(source.tolist(), data)

    def test_out_rank_and_argument_boundaries(self):
        source = torch.tensor([1.0])
        output = torch.tensor([9.0])
        with self.assertRaisesRegex(RuntimeError, "column_stack.*'out' argument is not supported"):
            torch.column_stack([source], out=output)
        self.assertEqual(output.tolist(), [9.0])
        for rank in (3, 4):
            for inputs in ([torch.zeros((1,) * rank)], [source, torch.zeros((0,) * rank)]):
                with self.subTest(rank=rank, count=len(inputs)):
                    with self.assertRaisesRegex(NotImplementedError, "column_stack.*only exact native CPU float32"):
                        torch.column_stack(inputs)
        for inputs in ([], ()):
            with self.assertRaisesRegex(RuntimeError, "^column_stack expects a non-empty TensorList$"):
                torch.column_stack(inputs)
        for call in (
            lambda: torch.column_stack(),
            lambda: torch.column_stack(source),
            lambda: torch.column_stack(iter([source])),
            lambda: torch.column_stack([source, 1]),
            lambda: torch.column_stack([source], None),
            lambda: torch.column_stack([source], out=1),
            lambda: torch.column_stack([source], dtype=torch.float32),
            lambda: torch.column_stack([source], device="cpu"),
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

    def test_mode_can_intercept_unsupported_native_inputs(self):
        marker = object()
        calls = []

        class Mode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return marker

        inputs = [torch.zeros((1, 1, 1))]
        output = torch.tensor([0.0])
        with Mode():
            self.assertIs(torch.column_stack(inputs, out=output), marker)
        self.assertEqual(calls, [(torch.column_stack, (), (inputs,), {"out": output})])

    def test_public_export(self):
        self.assertEqual(torch.__all__.count("column_stack"), 1)
        self.assertEqual(torch.column_stack.__module__, "torch")
        self.assertEqual(torch.column_stack.__qualname__, "_VariableFunctionsClass.column_stack")
        self.assertIn("column_stack(tensors, *, out=None) -> Tensor", torch.column_stack.__doc__)
        self.assertIs(torch._C._VariableFunctionsClass.column_stack, torch.column_stack)
        self.assertFalse(hasattr(torch.functional, "column_stack"))
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["column_stack"], torch.column_stack)


if __name__ == "__main__":
    unittest.main()
