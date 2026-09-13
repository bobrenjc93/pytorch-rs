"""Runtime sine and shared signed-product boundaries against default Inductor."""
import math
import struct
import unittest

import torch_rs as native
from tests.test_compile_pointwise_jit import available, program


def bits(value):
    return struct.unpack('=I', struct.pack('=f', value))[0]


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class NumericalBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def check(self, source, values):
        compiled = native.compile(program(source))
        reference = self.torch.compile(program(source))
        for data in (values, values[::-1]):
            with self.subTest(source=source, reversed=data is not values):
                x = native.tensor(data).to('cuda:0')
                tx = self.torch.tensor(data, device='cuda:0')
                output = compiled(x)
                actual, expected = output.cpu().tolist(), reference(tx).cpu().tolist()
                self.assertNotEqual(output.data_ptr(), x.data_ptr())
                self.assertEqual(output.shape, x.shape)
                self.assertEqual(output.dtype, x.dtype)
                self.assertEqual(output.device, x.device)
                for a, b in zip(actual, expected):
                    if math.isnan(b):
                        self.assertTrue(math.isnan(a))
                    else:
                        self.assertEqual(bits(a), bits(b), (a, b))
                self.assertEqual([bits(v) for v in x.cpu().tolist()], [bits(v) for v in data])

    def test_shared_signed_product_retains_reference_rounding(self):
        values = [-2e38, 2e38, -1.0, 1.0, -0.0, 0.0]
        for coefficient in (-2.0, -4.0, -3.713, -0.5, 2.0):
            self.check(f'def f(x):\n a=x*{coefficient!r}\n return (x+a)-a', values)
        for body in ('(a+x)-a', '(a-x)-a', '(a+x)+a', 'a+(x-a)',
                     '(x+x*-2.0)-x*-2.0', '(x+a)-(x*1.0)*-2.0',
                     '(x+a)-(True*x)*-2', '(x+a)+(x-a)'):
            self.check('def f(x):\n a=x*-2.0\n return '+body, values)
        # Unused consumers must not impose a rounding boundary on a live use.
        for unused in ('a*a', 'a+a', 'a.sin()'):
            self.check('def f(x):\n a=x*-2.0\n unused='+unused+'\n return x+a', values)

    def test_runtime_sine_subnormals_before_amplified_consumers(self):
        smallest_normal = struct.unpack('=f', struct.pack('=I', 0x00800000))[0]
        largest_subnormal = struct.unpack('=f', struct.pack('=I', 0x007fffff))[0]
        values = [1e-38, -1e-38, largest_subnormal, -largest_subnormal,
                  smallest_normal, -smallest_normal, 0.0, -0.0]
        for expression in ('x.sin()*1e38', 'x.sin()*1e300',
                           '(-x).sin()*1e38', '(x*0.5).sin()*1e38',
                           '(x.sin()+x.cos())*2.0'):
            self.check('def f(x):\n return '+expression, values)
        self.check('def f(x):\n return (x*1e-38).sin()*1e38', [1., -1., 2., -2.])

    def test_constant_only_sine_does_not_acquire_runtime_flush(self):
        for constant in (1e-38, -1e-38):
            for expression in ('a.sin()*1e38', 'a.relu().sin()*1e38',
                               '(-a).sin()*1e38', '(a+a).sin()*1e38'):
                self.check(f'def f(x):\n a=(x*0)+{constant!r}\n return '+expression,
                           [0., -0., 1., -1.])


if __name__ == '__main__':
    unittest.main()
