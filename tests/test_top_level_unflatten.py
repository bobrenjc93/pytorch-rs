import inspect
import pickle
import unittest
from unittest.mock import patch

import torch_rs as torch


class TopLevelUnflattenTests(unittest.TestCase):
    def test_public_export_and_binding(self):
        self.assertEqual(torch.__all__.count('unflatten'), 1)
        self.assertTrue(inspect.isbuiltin(torch.unflatten))
        self.assertIs(pickle.loads(pickle.dumps(torch.unflatten)), torch.unflatten)
        namespace = {}
        exec('from torch_rs import *', namespace)
        self.assertIs(namespace['unflatten'], torch.unflatten)
        source = torch.ones((2, 6))
        for result in (
            torch.unflatten(source, 1, (2, 3)),
            torch.unflatten(source, -1, sizes=[2, -1]),
            torch.unflatten(input=source, dim=1, sizes=torch.Size([2, 3])),
        ):
            self.assertEqual(result.shape, (2, 2, 3))
            self.assertEqual(result.stride(), (6, 3, 1))
            self.assertEqual(result.data_ptr(), source.data_ptr())
            self.assertIsNot(result, source)

    def test_duck_method_is_not_called(self):
        class Duck:
            def unflatten(self, *args):
                raise AssertionError('must not invoke an arbitrary method')

        with self.assertRaisesRegex(TypeError, 'must be Tensor'):
            torch.unflatten(Duck(), 0, (1,))

    def test_native_execution_bypasses_replaceable_method_attributes(self):
        source = torch.tensor([0., 1., 2., 3., 4., 5.])
        with patch.object(torch.Tensor, 'unflatten', side_effect=AssertionError('public method lookup')), \
             patch.object(torch.Tensor, '_unflatten', side_effect=AssertionError('private method lookup')):
            result = torch.unflatten(source, 0, (2, 3))
        self.assertEqual(result.tolist(), [[0., 1., 2.], [3., 4., 5.]])
        self.assertEqual(result.data_ptr(), source.data_ptr())

    def test_dtype_extension_keyword_remains_rejected(self):
        source = torch.ones((6,))
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'dtype'"):
            torch.unflatten(source, 0, (2, 3), dtype=torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires an NVIDIA GPU')
    def test_real_cuda_input_remains_rejected(self):
        source = torch.zeros((6,), device='cuda:0')
        self.assertEqual(str(source.device), 'cuda:0')
        with self.assertRaisesRegex(NotImplementedError, 'only exact native CPU float32'):
            torch.unflatten(source, 0, (2, 3))


if __name__ == '__main__':
    unittest.main()
