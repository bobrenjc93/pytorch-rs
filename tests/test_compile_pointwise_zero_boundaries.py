"""Scalar-zero contraction and static captures across nonfinite interludes."""
import unittest

import torch_rs as native
from tests.test_compile_pointwise_jit import available, program


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class ZeroBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def compare(self, compiled, reference, values):
        x = native.tensor(values).to('cuda:0')
        tx = self.torch.tensor(values, device='cuda:0')
        result, expected = compiled(x), reference(tx).cpu()
        actual = self.torch.tensor(result.cpu().tolist())
        self.torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        present = ~expected.isnan()
        self.assertTrue(self.torch.equal(actual.view(self.torch.int32)[present],
                                         expected.view(self.torch.int32)[present]))
        self.assertNotEqual(result.data_ptr(), x.data_ptr())
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device, x.device)
        self.torch.testing.assert_close(self.torch.tensor(x.cpu().tolist()), tx.cpu(),
                                       rtol=0, atol=0, equal_nan=True)

    def test_scalar_zero_subtraction_preserves_contraction(self):
        for coefficient in (1.137, 0.71359, -1.137, 2.0, -2.0):
            for expression in ('-a+(a-0.0)', '(a-0.0)-a', 'a-(a-0.0)',
                               '-a+(a+0.0)', '(a+0.0)-a',
                               '-a+(a-(-0.0))', '(a-(-0.0))-a', 'a-a',
                               '(a-0.0)-(a-0.0)', '-a+((a-0.0)-0.0)',
                               '-a+(a-0)', '(a-0)-a',
                               '-a+(a-False)', '(a-False)-a'):
                source = f'def f(x):\n a=x*{coefficient!r}\n return '+expression
                compiled, reference = native.compile(program(source)), self.torch.compile(program(source))
                for values in ([1e10, 2e38], [-1e10, -2e38],
                               [-0., 0.], [float('inf'), float('nan')]):
                    with self.subTest(coefficient=coefficient, expression=expression, values=values):
                        self.compare(compiled, reference, values)
                native.compiler.reset()
                self.torch.compiler.reset()

    def test_nested_zero_subtractions_and_sign_flips(self):
        for coefficient in (1.137, 0.71359, -1.137):
            for expression in ('(b-0.0)+(a-0.0)', '(a-0.0)+(b-0.0)',
                               '(b-0.0)-(-a)', '(b-0.0)*-1.0-a',
                               '((b-0.0)*-1.0)*-1.0+(a-0.0)'):
                source = f'def f(x):\n a=x*{coefficient!r}\n b=(a-0.0)*-1.0\n return '+expression
                compiled = native.compile(program(source))
                reference = self.torch.compile(program(source))
                for values in ([1e10, 2e38], [-1e10, -2e38], [0., -0.]):
                    with self.subTest(coefficient=coefficient, expression=expression, values=values):
                        self.compare(compiled, reference, values)
                native.compiler.reset()
                self.torch.compiler.reset()

    def sequence(self, sequence, *, change_shape=False):
        source = 'def f(x):\n return ((x*0)+a)-16777216.0'
        fn, ref_fn = program(source, a=sequence[0]), program(source, a=sequence[0])
        compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
        for index, value in enumerate(sequence):
            fn.__globals__['a'] = ref_fn.__globals__['a'] = value
            for values in ([1., -1.], [-3., 4.]):
                if change_shape and index == len(sequence) - 1:
                    values = values + [2.]
                with self.subTest(index=index, binding=value):
                    self.compare(compiled, reference, values)
        native.compiler.reset()
        self.torch.compiler.reset()

    def test_nonfinite_interlude_reuses_existing_static_capture(self):
        for nonfinite in (float('inf'), float('-inf'), float('nan')):
            self.sequence([16777217., nonfinite, 16777217.])

    def test_finite_change_still_promotes_after_nonfinite_interlude(self):
        for nonfinite in (float('inf'), float('-inf'), float('nan')):
            self.sequence([16777217., nonfinite, 16777218., 16777217.])
            self.sequence([16777217., 16777218., nonfinite, 16777217.])

    def test_cache_miss_keeps_nonfinite_promotion_history(self):
        for nonfinite in (float('inf'), float('-inf'), float('nan')):
            self.sequence([nonfinite, 16777217.])
            self.sequence([16777217., nonfinite, 16777217.], change_shape=True)


if __name__ == '__main__':
    unittest.main()
