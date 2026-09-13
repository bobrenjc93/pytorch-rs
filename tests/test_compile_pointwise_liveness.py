"""Dead pointwise consumers must not change live floating-point contraction."""
import unittest

import torch_rs as native

from tests.test_compile_pointwise_jit import available, program


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class PointwiseLiveness(unittest.TestCase):
    def tearDown(self):
        import torch
        native.compiler.reset()
        torch.compiler.reset()

    def test_unused_consumers_do_not_change_live_product_contraction(self):
        import torch
        torch.set_num_threads(1)
        left = [2e38, -2e38, 1e38, -1e38, 1.25, -1.25]
        right = [1e38, -1e38, 2e38, -2e38, 0.5, -0.5]
        for unused in ('', 'unused = a * a', 'unused = a + a', 'unused = a.sin()'):
            source = 'def f(x, y):\n a = x * 2.0\n ' + unused + '\n return a - y * 2.0\n'
            compiled = native.compile(program(source))
            reference = torch.compile(program(source, torch))
            for values in ((left, right), (right, left)):
                args = [native.tensor(v, dtype=native.float32).to('cuda:0') for v in values]
                refs = [torch.tensor(v, dtype=torch.float32, device='cuda:0') for v in values]
                with self.subTest(unused=unused, values=values):
                    actual = torch.tensor(compiled(*args).cpu().tolist(), dtype=torch.float32)
                    expected = reference(*refs).cpu()
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True)


if __name__ == '__main__':
    unittest.main()
