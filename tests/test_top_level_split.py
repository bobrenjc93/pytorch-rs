import inspect
import pickle
import unittest

import torch_rs as torch


class TopLevelSplitTests(unittest.TestCase):
    def test_exports_signature_and_call_forms(self):
        self.assertIs(torch.split, torch.functional.split)
        self.assertEqual(torch.__all__.count('split'), 1)
        self.assertIn('split', torch.functional.__all__)
        self.assertIs(pickle.loads(pickle.dumps(torch.split)), torch.split)
        self.assertEqual(list(inspect.signature(torch.split).parameters),
                         ['tensor', 'split_size_or_sections', 'dim'])
        source = torch.tensor([float(i) for i in range(7)])
        for call in (
            lambda: torch.split(source, 3),
            lambda: torch.split(source, 3, -1),
            lambda: torch.split(source, split_size_or_sections=3),
            lambda: torch.split(dim=0, split_size_or_sections=3, tensor=source),
        ):
            outputs = call()
            self.assertIs(type(outputs), tuple)
            self.assertEqual([o.tolist() for o in outputs],
                             [[0., 1., 2.], [3., 4., 5.], [6.]])
            self.assertTrue(outputs[0].is_set_to(source.narrow(0, 0, 3)))

    def test_unsupported_metadata_is_explicit(self):
        class DuckTensor:
            dtype = torch.float32
            device = torch.device('cpu')

            def split(self, *args):
                raise AssertionError('must not execute arbitrary split methods')

        for source in (DuckTensor(), None, [1., 2.]):
            with self.assertRaisesRegex(NotImplementedError, 'exact native CPU float32'):
                torch.split(source, 1)

    def test_only_tensor_operand_dispatches_and_binding_precedes_dispatch(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise AssertionError('must not dispatch')

        source = torch.ones((4,))
        for call in (
            lambda: torch.split(source, Override()),
            lambda: torch.split(source, 2, Override()),
            lambda: torch.split(Override()),
            lambda: torch.split(Override(), 2, split_size=2),
            lambda: torch.split(Override(), 2, tensor=source),
        ):
            with self.assertRaises(TypeError):
                call()

    @unittest.skipUnless(torch.cuda.is_available(), 'requires an NVIDIA GPU')
    def test_real_cuda_input_is_rejected(self):
        source = torch.zeros((5,), device='cuda:0')
        self.assertEqual(str(source.device), 'cuda:0')
        for size in (2, [0, 2, 3]):
            with self.assertRaisesRegex(NotImplementedError, 'exact native CPU float32'):
                torch.split(source, size)


if __name__ == '__main__':
    unittest.main()
