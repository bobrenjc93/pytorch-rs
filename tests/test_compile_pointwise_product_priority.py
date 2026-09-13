"""Default-Inductor contraction ordering across raw and computed products."""
import unittest

import torch_rs as native
from tests.test_compile_pointwise_jit import available, cache, program


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class ProductPriority(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def check_families(self, families):
        torch = self.torch
        left = [2e38, 1e38, -2e38, -1e38, 0., -0., 1.137, -1.137]
        right = [-1e10, -2e10, 1e10, 2e10, -0., 0., 1.137, -1.137]
        for setup, first, second in families:
            for reversed_products in (False, True):
                native.compiler.reset()
                torch.compiler.reset()
                expression = f'{second}+{first}' if reversed_products else f'{first}+{second}'
                source = f'def f(x,y):\n {setup}return {expression}'
                compiled = native.compile(program(source))
                reference = torch.compile(program(source, torch))
                for warm in (False, True):
                    values = (left[::-1], right[::-1]) if warm else (left, right)
                    args = [native.tensor(value, dtype=native.float32).to('cuda:0') for value in values]
                    refs = [torch.tensor(value, dtype=torch.float32, device='cuda:0') for value in values]
                    with self.subTest(source=source, warm=warm):
                        output = compiled(*args)
                        expected = reference(*refs)
                        actual = torch.tensor(output.cpu().tolist(), dtype=torch.float32)
                        expected_host = expected.cpu()
                        torch.testing.assert_close(actual, expected_host, rtol=1e-5, atol=1e-6, equal_nan=True)
                        zeros = expected_host == 0
                        self.assertTrue(torch.equal(actual.signbit()[zeros], expected_host.signbit()[zeros]))
                        self.assertEqual(tuple(output.shape), tuple(expected.shape))
                        self.assertEqual(output.stride(), expected.stride())
                        self.assertEqual(str(output.dtype), str(expected.dtype))
                        self.assertEqual(str(output.device), str(expected.device))
                        self.assertEqual(output.requires_grad, expected.requires_grad)
                        for arg, ref in zip(args, refs):
                            self.assertNotEqual(output.data_ptr(), arg.data_ptr())
                            torch.testing.assert_close(torch.tensor(arg.cpu().tolist()), ref.cpu(), rtol=0, atol=0)
                self.assertEqual(len(cache(compiled).graphs), 1)

    def test_raw_product_addition_order(self):
        self.check_families([('', 'x*x', 'x*y')])

    def test_unary_producer_products_in_both_operand_orders(self):
        self.check_families([
            ('a=x.relu()\n ', 'a*a', 'x*y'),
            ('a=x.relu()\n ', 'a*x', 'x*y'),
            ('a=x.relu()\n ', 'x*a', 'x*y'),
        ])

    def test_arithmetic_producer_contraction_controls(self):
        self.check_families([
            ('a=x+1.0\n ', 'a*a', 'x*y'),
            ('a=x-1.0\n ', 'a*a', 'x*y'),
            ('a=x*1.137\n ', 'a*a', 'x*y'),
            ('a=-x\n ', 'a*a', 'x*y'),
            ('a=x+1.0\n b=x-1.0\n ', 'a*b', 'x*y'),
        ])


if __name__ == '__main__':
    unittest.main()
