"""Expression identity and per-consumer rounding in the default native JIT."""
import unittest

import torch_rs as native
from torch_rs import torch_rs as bridge
from tests.test_compile_pointwise_jit import available, cache, kernel, lower, program


class RoundingAdmission(unittest.TestCase):
    def test_identical_expressions_share_rounding_before_contraction(self):
        graph = lower(program('def f(x, y):\n return x*y-x*y'), 2)
        source = bridge._pointwise_source(graph.nodes, graph.output, 2)
        self.assertEqual(source.count('__fmul_rn('), 1)
        self.assertIn('__fsub_rn(', source)
        self.assertNotIn('fmaf(', source)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class RoundingHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def compare(self, actual, expected):
        torch = self.torch
        observed = torch.tensor(actual.cpu().tolist(), dtype=torch.float32)
        expected = expected.cpu()
        torch.testing.assert_close(observed, expected, rtol=0, atol=0, equal_nan=True)
        zeros = (observed == 0) & (expected == 0)
        self.assertTrue(torch.equal(observed.signbit()[zeros], expected.signbit()[zeros]))

    def check_bodies(self, bodies, left, right):
        for body in bodies:
            source = 'def f(x, y):\n ' + body
            compiled = native.compile(program(source))
            reference = self.torch.compile(program(source, self.torch))
            for values in ((left, right), (right, left)):
                args = [native.tensor(v, dtype=native.float32).to('cuda:0') for v in values]
                refs = [self.torch.tensor(v, dtype=self.torch.float32, device='cuda:0') for v in values]
                with self.subTest(body=body, values=values):
                    actual = compiled(*args)
                    self.compare(actual, reference(*refs))
                    for arg, ref in zip(args, refs):
                        self.compare(arg, ref)
                        self.assertNotEqual(actual.data_ptr(), arg.data_ptr())
            self.assertEqual(len(cache(compiled).graphs), 1)

    def test_repeated_expressions_finite_cancellation_and_overflow(self):
        self.check_bodies([
            'return x*y-x*y', 'a=x*y\n return a-a',
            'return x*1.137-x*1.137', 'return (x*y).sin()-(x*y).sin()',
            # Operand order is significant to the reference's expression CSE.
            'return x*y-y*x', 'return x*y+x*y',
        ], [1e10, 2e38, -2e38, -1e10, 0., -0., float('inf'), float('nan')],
           [1.0000001192092896, 2., 2., 1.0000001192092896, -0., 0., 1., 1.])

    def test_repeated_input_identity_cold_warm_and_fresh_values(self):
        for first_same in (False, True):
            fn = program('def f(x, y):\n return x*1.137-y*1.137')
            compiled = native.compile(fn)
            reference = self.torch.compile(program('def f(x, y):\n return x*1.137-y*1.137'))
            for same, view, value in [(first_same, False, 1e10), (not first_same, False, 1e10),
                                      (False, True, 1e10), (True, False, 3e10),
                                      (False, False, -1e10), (first_same, False, 2e38)]:
                x = native.tensor([value]).to('cuda:0')
                tx = self.torch.tensor([value], device='cuda:0')
                y = x if same else x[:] if view else native.tensor([value]).to('cuda:0')
                ty = tx if same else tx[:] if view else self.torch.tensor([value], device='cuda:0')
                with self.subTest(first_same=first_same, same=same, view=view, value=value):
                    self.compare(compiled(x, y), reference(tx, ty))
            self.assertEqual(len(cache(compiled).graphs), 2)
            self.assertEqual(len({id(entry[1]) for entry in cache(compiled).graphs.values()}), 2)
            native.compiler.reset()
            self.assertEqual(cache(compiled).graphs, {})
            self.compare(compiled(x, x), reference(tx, tx))

    def test_identity_relationship_obeys_recompile_limit(self):
        fn = program('def f(x, y):\n return x*1.137-y*1.137')
        compiled = native.compile(fn, recompile_limit=1)
        x, y = [native.tensor([1e10]).to('cuda:0') for _ in range(2)]
        self.assertEqual(compiled(x, x).cpu().tolist(), [0.])
        original = kernel(compiled)
        with self.assertRaisesRegex(NotImplementedError, 'recompile_limit'):
            compiled(x, y)
        self.assertIs(kernel(compiled), original)
        self.assertEqual(compiled(x, x).cpu().tolist(), [0.])

    def test_normalized_signs_preserve_contraction_orientation(self):
        bodies = []
        for a, b in [(2., 2.), (3.713, 1.137), (1.137, 3.713), (4., 0.5)]:
            bodies.extend([f'return -(x*{a!r}-y*{b!r})',
                           f'return x*{-a!r}-y*{-b!r}',
                           f'return x*{-a!r}+y*{b!r}'])
        self.check_bodies(bodies,
            [2e38, -2e38, 1e38, -1e38, 1e10, -0., 0., 1e-38],
            [1e38, -1e38, 2e38, -2e38, 1.0000001192092896, 0., -0., 1e-38])

    def test_shared_products_contract_per_consumer_with_nonfinite_values(self):
        self.check_bodies([
            'a=x*2.0\n return (a-y)+a', 'a=x*2.0\n return a+(a-y)',
            'a=x*2.0\n return (a+y)-a', 'a=x*2.0\n return (a-y)-a',
            'a=x*y\n return (a-x)+a', 'a=x*2.0\n return (a-y)+a.relu()',
        ], [2e38, -2e38, 2e38, -2e38, 1e10, -0., 0., float('nan')],
           [float('inf'), -float('inf'), 1e38, -1e38, 1., 0., -0., 1.])


if __name__ == '__main__':
    unittest.main()
