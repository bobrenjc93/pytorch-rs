"""Native inference boundaries for the composed functional GLU."""
import unittest

import numpy as np
import torch_rs as torch
from torch_rs.nn.functional import glu


class FunctionalGluTests(unittest.TestCase):
    def test_active_recording_rejected_without_changing_input_or_graph(self):
        leaf = torch.tensor([[-2.0, -0.0, 1.0, 2.0], [3.0, 4.0, 5.0, 6.0]], requires_grad=True)
        with torch.no_grad():
            unrecorded_view = leaf[0]
        nonleaf = leaf + leaf
        cases = [leaf, leaf[1], leaf.transpose(0, 1), unrecorded_view, nonleaf,
                 torch.zeros((2, 0), requires_grad=True),
                 torch.tensor([float('inf'), float('nan')], requires_grad=True)]
        for source in cases:
            with self.subTest(shape=source.shape, stride=source.stride()):
                before = np.asarray(source).copy().view(np.uint32)
                metadata = (source.shape, source.stride(), source.storage_offset(), source.data_ptr())
                with self.assertRaisesRegex(RuntimeError, r'^glu\(\): autograd recording is not supported$'):
                    glu(source)
                self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(metadata, (source.shape, source.stride(), source.storage_offset(), source.data_ptr()))
                np.testing.assert_array_equal(np.asarray(source).view(np.uint32), before)
                with torch.no_grad():
                    output = glu(source)
                    self.assertFalse(output.requires_grad)
                    self.assertTrue(output.is_leaf)
                self.assertTrue(torch.is_grad_enabled())
                detached = glu(source.detach())
                np.testing.assert_array_equal(np.asarray(output), np.asarray(detached))
        self.assertIsNone(leaf.grad)
        nonleaf.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0] * 4] * 2)

    def test_rank_boundary_in_grad_and_no_grad_modes(self):
        for shape in [(1, 2, 2, 2), (1, 1, 2, 2, 2), (0, 1, 2, 2)]:
            source = torch.zeros(shape)
            for enabled in (True, False):
                with self.subTest(shape=shape, grad_enabled=enabled), torch.set_grad_enabled(enabled):
                    with self.assertRaisesRegex(NotImplementedError, 'ranks 1 through 3'):
                        glu(source)
                    self.assertEqual(torch.is_grad_enabled(), enabled)

    def test_failure_restores_forwarding_mode_and_no_grad(self):
        source = torch.ones((2, 4), requires_grad=True)
        calls = []

        class Forward(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                if func is glu:
                    calls.append(func)
                return func(*args, **(kwargs or {}))

        mode = Forward()
        with mode:
            with self.assertRaisesRegex(RuntimeError, 'autograd recording is not supported'):
                glu(source)
            self.assertIs(torch.overrides._get_current_function_mode(), mode)
            with torch.no_grad():
                glu(source)
                with self.assertRaises(IndexError):
                    glu(source, 2)
                self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(calls, [glu] * 3)
        self.assertIsNone(torch.overrides._get_current_function_mode())


if __name__ == '__main__':
    unittest.main()
