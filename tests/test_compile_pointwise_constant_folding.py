"""Constant propagation must preserve precision and early-zero provenance."""
import struct
import unittest

import torch_rs as native
from tests.test_compile_pointwise_jit import available, cache, lower, program
from torch_rs import torch_rs as bridge


class ConstantAdmission(unittest.TestCase):
    def test_literals_keep_binary64_bits_until_runtime_materialization(self):
        graph = lower(program('def f(x):\n return x + 16777217.0'))
        literal = next(n for n in graph.nodes if n[0] == 'constant')
        self.assertEqual(literal[3], struct.unpack('=Q', struct.pack('=d', 16777217.0))[0])
        self.assertIn('0x4b800000u', bridge._pointwise_source(graph.nodes, graph.output, 1))


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class ConstantHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def check(self, source, scalars=(0, False), graph_count=None):
        fn, ref_fn = program(source, scale=0), program(source, scale=0)
        compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
        values = [-0., 0., 1., -1., 1e-38, float('inf'), -float('inf'), float('nan')]
        for scalar in scalars:
            fn.__globals__['scale'] = ref_fn.__globals__['scale'] = scalar
            for data in (values, values[::-1]):
                x = native.tensor(data).to('cuda:0')
                tx = self.torch.tensor(data, device='cuda:0')
                with self.subTest(source=source, scalar_type=type(scalar).__name__, reversed=data is not values):
                    result = compiled(x)
                    expected = reference(tx)
                    actual = self.torch.tensor(result.cpu().tolist(), dtype=self.torch.float32)
                    expected = expected.cpu()
                    self.torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                    # Exact output bits, except NaN payloads, including all zeros.
                    finite_or_inf = ~expected.isnan()
                    self.assertTrue(self.torch.equal(actual.view(self.torch.int32)[finite_or_inf],
                                                    expected.view(self.torch.int32)[finite_or_inf]))
                    self.assertEqual(result.shape, tuple(expected.shape))
                    self.assertEqual(result.stride(), expected.stride())
                    self.assertEqual(str(result.dtype), str(expected.dtype))
                    self.assertEqual(str(result.device), 'cuda:0')
                    self.assertFalse(result.requires_grad)
                    self.assertNotEqual(result.data_ptr(), x.data_ptr())
                    self.torch.testing.assert_close(self.torch.tensor(x.cpu().tolist()), tx.cpu(),
                                                   rtol=0, atol=0, equal_nan=True)
        self.assertEqual(len(cache(compiled).graphs),
                         len({type(s) for s in scalars}) if graph_count is None else graph_count)

    def test_changed_float_binding_becomes_a_runtime_scalar(self):
        self.check('def f(x):\n return ((x*0)+scale)-16777216.0',
                   scalars=(16777216.0, 16777217.0, 16777216.0, 16777218.0), graph_count=2)

    def test_constant_precision_cancellation_overflow_and_literal_rounding(self):
        for body in ('((x*scale)+16777216.0)+1.0-16777216.0',
                     '((x*scale)+2e38)*2.0-2e38',
                     '((x*scale)+16777217.0)-16777216.0',
                     '((x*scale)+16777217)-16777216',
                     '((x*scale)+1e300)-1e300',
                     '((x*scale)+9007199254740992.0)+1.0-9007199254740992.0',
                     '-(((x*scale)+33554432.0)+3.0-33554432.0)',
                     '(((x*scale)+0.1)*0.1)-0.01'):
            self.check('def f(x):\n return ' + body)

    def test_computed_zero_and_seed_zero_have_distinct_addition_rules(self):
        for zero in ('x*scale', 'x*scale-0.0', 'x*scale+0.0',
                     'x*scale*1.0', 'x*scale*2.0', 'x*scale-0', 'x*scale+0',
                     '(x*scale)+(x*scale)', '(x*scale)-(x*scale)',
                     '(x*scale+0.0)+(x*scale)', '(x*scale)+(x*scale+0.0)'):
            for body in (f'({zero})+x', f'x+({zero})', f'x-({zero})'):
                self.check('def f(x):\n return ' + body, scalars=(0, False, 0.0, 0))

    def test_materialization_is_per_consumer_and_unary_operations_stop_folding(self):
        for body in ('return a.relu()-16777216.0',
                     'return a*x-16777216.0*x',
                     'return a+x-16777216.0',
                     'b=a.relu()\n return (a-16777216.0)+(b-16777216.0)',
                     'b=a.sin()\n return (a-16777216.0)+(b-b)'):
            self.check('def f(x):\n a=(x*scale)+16777217.0\n ' + body)

    def test_subtraction_aliases_preserve_later_subtraction_contraction(self):
        source = 'def f(x,y):\n z=x*0\n a=x*2.0\n b=a-(z-z)\n return b-y*2.0'
        compiled = native.compile(program(source))
        reference = self.torch.compile(program(source))
        for values in ([2e38, -2e38], [-2e38, 2e38]):
            x, y = native.tensor(values).to('cuda:0'), native.tensor([1e38, -1e38]).to('cuda:0')
            tx = self.torch.tensor(values, device='cuda:0')
            ty = self.torch.tensor([1e38, -1e38], device='cuda:0')
            result = compiled(x, y)
            self.torch.testing.assert_close(self.torch.tensor(result.cpu().tolist()),
                                           reference(tx, ty).cpu(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
