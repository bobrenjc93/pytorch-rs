import unittest

import numpy as np
import torch_rs as torch


class TensorSplitTests(unittest.TestCase):
    def test_uneven_oversized_and_call_forms(self):
        source = torch.tensor([float(i) for i in range(7)])
        for call in (
            lambda: source.split(3),
            lambda: source.split(3, 0),
            lambda: source.split(split_size=3),
            lambda: source.split(dim=-1, split_size=3),
        ):
            outputs = call()
            self.assertIs(type(outputs), tuple)
            self.assertEqual([out.tolist() for out in outputs], [[0., 1., 2.], [3., 4., 5.], [6.]])
        for size in (7, 100, 2**63 - 1):
            outputs = source.split(size)
            self.assertEqual(len(outputs), 1)
            self.assertIsNot(outputs[0], source)
            self.assertTrue(outputs[0].is_set_to(source))

    def test_transposed_offset_views(self):
        source = torch.tensor([float(i) for i in range(30)]).reshape(2, 3, 5)[1].t()
        outputs = source.split(2)
        for output, start, length in zip(outputs, (0, 2, 4), (2, 2, 1), strict=True):
            direct = source.narrow(0, start, length)
            self.assertEqual(output.tolist(), direct.tolist())
            self.assertEqual(output.stride(), (1, 5))
            self.assertEqual(output.storage_offset(), 15 + start)
            self.assertEqual(output.data_ptr(), direct.data_ptr())
            self.assertTrue(output.is_set_to(direct))
        del source
        self.assertEqual(outputs[2].tolist(), [[19., 24., 29.]])

    def test_empty_dimensions(self):
        for shape, dim in (((0,), 0), ((2, 0, 3), 1), ((2, 3, 0), -1)):
            for size in (0, 2):
                source = torch.zeros(shape, requires_grad=True)
                outputs = source.split(size, dim)
                self.assertEqual(len(outputs), 1)
                self.assertTrue(outputs[0].is_set_to(source))
                self.assertEqual(tuple(outputs[0].shape), shape)
                outputs[0].sum().backward()
                self.assertEqual(tuple(source.grad.shape), shape)
        source = torch.zeros((0, 5))
        self.assertEqual([tuple(o.shape) for o in source.split(2, 1)], [(0, 2), (0, 2), (0, 1)])
        with self.assertRaisesRegex(RuntimeError, 'dimension size of 5'):
            source.split(0, 1)

    def test_backward_accumulates_selected_repeated_and_nested_outputs(self):
        leaf = torch.tensor([float(i) for i in range(30)], requires_grad=True)
        source = (leaf * 2.).reshape(2, 3, 5)[1].t()
        outputs = source.split(2)
        self.assertEqual([o.output_nr for o in outputs], [0, 1, 2])
        self.assertTrue(all(o.requires_grad and not o.is_leaf for o in outputs))
        nested = outputs[1].split(1)
        (nested[0].sum() + nested[0].sum() + outputs[2].sum() + leaf.sum()).backward()
        expected = np.ones(30, dtype=np.float32)
        expected[[17, 22, 27]] = 5.
        expected[[19, 24, 29]] = 3.
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def test_no_grad_views_and_untracked_output_numbers(self):
        for requires_grad in (False, True):
            leaf = torch.ones((5,), requires_grad=requires_grad)
            with torch.no_grad():
                outputs = leaf.split(2)
            self.assertEqual([o.output_nr for o in outputs], [0, 0, 0])
            self.assertTrue(all(o.is_leaf for o in outputs))
            self.assertEqual([o.requires_grad for o in outputs], [requires_grad] * 3)

    def test_argument_errors_and_unsupported_forms(self):
        source = torch.zeros((2, 3))
        class IndexOnly:
            def __index__(self):
                raise AssertionError('split must not coerce arbitrary objects')

        for size in (True, 1.5, None, '2', np.int64(2), IndexOnly()):
            with self.subTest(size=size), self.assertRaises(TypeError):
                source.split(size)
        for dim in (True, 1.5, None, '0', IndexOnly()):
            with self.subTest(dim=dim), self.assertRaises(TypeError):
                source.split(1, dim)
        for call in (lambda: source.split(), lambda: source.split(1, 0, 0),
                     lambda: source.split(1, split_size=1), lambda: source.split(1, extra=0)):
            with self.assertRaises(TypeError):
                call()
        for size in (-1, 0):
            with self.assertRaises(RuntimeError):
                source.split(size)
        for dim in (-3, 2):
            with self.assertRaises(IndexError):
                source.split(1, dim)
        for call in (lambda: source.split(2**100), lambda: source.split(1, 2**100)):
            with self.assertRaisesRegex(ValueError, 'Overflow when unpacking long long'):
                call()
        with self.assertRaisesRegex(RuntimeError, 'at least a 1-dimensional tensor'):
            torch.tensor(1.).split(1)
        for sizes in ([1, 1], (1, 1)):
            with self.assertRaisesRegex(NotImplementedError, 'section-list'):
                source.split(sizes)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires an NVIDIA GPU')
    def test_cuda_input_is_outside_the_native_cpu_contract(self):
        source = torch.zeros((5,), device='cuda:0')
        with self.assertRaisesRegex(NotImplementedError, 'exact native CPU float32'):
            source.split(2)


if __name__ == '__main__':
    unittest.main()
