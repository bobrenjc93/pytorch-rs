import unittest

import numpy as np
import torch_rs as torch


class TopLevelHstackTests(unittest.TestCase):
    def test_scalars_vectors_and_columns(self):
        result = torch.hstack(tensors=(torch.tensor(-0.0), torch.tensor([2.0, 3.0])), out=None)
        self.assertEqual(result.shape, (3,))
        np.testing.assert_array_equal(
            np.asarray(result).view(np.uint32),
            np.asarray([-0.0, 2.0, 3.0], dtype=np.float32).view(np.uint32),
        )
        result = torch.hstack([torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])])
        self.assertEqual(result.tolist(), [[1.0, 3.0], [2.0, 4.0]])
        self.assertEqual(result.stride(), (2, 1))
        self.assertEqual(result.storage_offset(), 0)

    def test_single_input_has_independent_storage(self):
        for data in (2.0, [1.0, 2.0], [[1.0], [2.0]]):
            with self.subTest(data=data):
                source = torch.tensor(data)
                result = torch.hstack([source])
                self.assertNotEqual(result.data_ptr(), source.data_ptr())
                np.asarray(result).reshape(-1)[0] = 17.0
                self.assertEqual(source.tolist(), data)

    def test_out_rank_and_argument_boundaries(self):
        source = torch.tensor([1.0])
        output = torch.tensor([9.0])
        with self.assertRaisesRegex(RuntimeError, "hstack.*'out' argument is not supported"):
            torch.hstack([source], out=output)
        self.assertEqual(output.tolist(), [9.0])
        for rank in (3, 4):
            for inputs in ([torch.zeros((1,) * rank)], [source, torch.zeros((0,) * rank)]):
                with self.subTest(rank=rank, count=len(inputs)):
                    with self.assertRaisesRegex(NotImplementedError, "hstack.*only exact native CPU float32"):
                        torch.hstack(inputs)
        for inputs in ([], ()):
            with self.assertRaisesRegex(RuntimeError, "^hstack expects a non-empty TensorList$"):
                torch.hstack(inputs)
        for call in (
            lambda: torch.hstack(),
            lambda: torch.hstack(source),
            lambda: torch.hstack(iter([source])),
            lambda: torch.hstack([source, 1]),
            lambda: torch.hstack([source], None),
            lambda: torch.hstack([source], out=1),
            lambda: torch.hstack([source], dtype=torch.float32),
            lambda: torch.hstack([source], device="cpu"),
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
            self.assertIs(torch.hstack(inputs, out=output), marker)
        self.assertEqual(calls, [(torch.hstack, (), (inputs,), {"out": output})])

    def test_public_export(self):
        self.assertEqual(torch.__all__.count("hstack"), 1)
        self.assertEqual(torch.hstack.__module__, "torch")
        self.assertEqual(torch.hstack.__qualname__, "_VariableFunctionsClass.hstack")
        self.assertIn("hstack(tensors, *, out=None) -> Tensor", torch.hstack.__doc__)
        self.assertIs(torch._C._VariableFunctionsClass.hstack, torch.hstack)
        self.assertFalse(hasattr(torch.functional, "hstack"))
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["hstack"], torch.hstack)


if __name__ == "__main__":
    unittest.main()
