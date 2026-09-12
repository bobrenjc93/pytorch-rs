"""Constant trigonometric evaluation rounds at float32 input/output boundaries."""
import math
import struct
import unittest

import torch_rs as native
from tests.test_compile_pointwise_jit import available, program


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class ConstantUnaryPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def test_held_out_inputs_and_shared_constant_unary_consumers(self):
        for value in (0.1, 16777217., 1e-38, -1e-38, 3.14, 1e38):
            rounded = struct.unpack('=f', struct.pack('=f', value))[0]
            for op in ('sin', 'cos'):
                expected_scalar = getattr(math, op)(rounded)
                source = f'def f(x):\n a=((x*0)+{value!r}).{op}()\n b=a-{expected_scalar!r}\n return b*1e38 + b'
                compiled = native.compile(program(source))
                reference = self.torch.compile(program(source))
                for data in ([1., -2.], [-3., 4.]):
                    with self.subTest(value=value, op=op, data=data):
                        x, tx = native.tensor(data).to('cuda:0'), self.torch.tensor(data, device='cuda:0')
                        result, expected = compiled(x), reference(tx).cpu()
                        actual = self.torch.tensor(result.cpu().tolist())
                        self.assertTrue(self.torch.equal(actual.view(self.torch.int32), expected.view(self.torch.int32)))
                        self.assertNotEqual(result.data_ptr(), x.data_ptr())
                        self.assertEqual(x.cpu().tolist(), data)

    def test_constant_unary_nonfinites_and_signed_zero(self):
        for value in (0., -0., float('inf'), float('-inf'), float('nan')):
            for op in ('sin', 'cos'):
                for operand in ('(x*0)+constant', '-(x*0)'):
                    source = f'def f(x):\n a=({operand}).{op}()\n return a*2.0+a'
                    compiled = native.compile(program(source, constant=value))
                    reference = self.torch.compile(program(source, constant=value))
                    for data in ([1., -2.], [-3., 4.]):
                        with self.subTest(value=value, op=op, operand=operand):
                            x = native.tensor(data).to('cuda:0')
                            tx = self.torch.tensor(data, device='cuda:0')
                            expected = reference(tx).cpu()
                            actual = self.torch.tensor(compiled(x).cpu().tolist())
                            self.torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                            present = ~expected.isnan()
                            self.assertTrue(self.torch.equal(actual.view(self.torch.int32)[present],
                                                            expected.view(self.torch.int32)[present]))

    def test_changed_binding_switches_from_constant_to_runtime_unary(self):
        for op in ('sin', 'cos'):
            source = f'def f(x):\n return (((x*0)+constant).{op}()-expected_scalar)*1e38'
            scalar = getattr(math, op)(1.)
            fn = program(source, constant=1., expected_scalar=scalar)
            ref_fn = program(source, constant=1., expected_scalar=scalar)
            compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
            for value in (1., 2., 1., 1e-38):
                fn.__globals__['constant'] = ref_fn.__globals__['constant'] = value
                with self.subTest(op=op, value=value):
                    actual = self.torch.tensor(compiled(native.tensor([1., -1.]).to('cuda:0')).cpu().tolist())
                    expected = reference(self.torch.tensor([1., -1.], device='cuda:0')).cpu()
                    self.assertTrue(self.torch.equal(actual.view(self.torch.int32), expected.view(self.torch.int32)))


if __name__ == '__main__':
    unittest.main()
