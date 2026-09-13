"""Integer zero must remain distinct from floating zero during CUDA lowering."""
import unittest

import torch_rs as native

from tests.test_compile_pointwise_jit import available


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class IntegerScalarSemantics(unittest.TestCase):
    def tearDown(self):
        import torch
        native.compiler.reset()
        torch.compiler.reset()

    def test_scalar_type_changes_preserve_integer_zero_and_float_zero_semantics(self):
        import torch
        torch.set_num_threads(1)
        values = [float('inf'), -float('inf'), float('nan'), -1., -0., 0., 1., 1e-38]
        for expression in ('x * scale', 'scale * x'):
            source = 'def f(x):\n return ' + expression
            candidate_globals, reference_globals = {}, {}
            exec(source, candidate_globals)
            exec(source, reference_globals)
            candidate = native.compile(candidate_globals['f'])
            reference = torch.compile(reference_globals['f'])
            # Exercise cold integer-zero admission and repeated warm type changes
            # on identical input metadata. None of these cache keys may retain
            # input values or conflate an integer zero with a floating zero.
            # Do not transition between floating +0.0 and -0.0 here: Inductor's
            # captured-value equality guard can reuse the +0.0 graph for -0.0,
            # disagreeing with both eager and a fresh compile. Literal signed
            # zero semantics are covered independently by the JIT IEEE tests.
            for scalar in (0, 0.0, False, True, 1, 1.0, 0):
                candidate_globals['scale'] = reference_globals['scale'] = scalar
                for data in (values, values[::-1]):
                    x = native.tensor(data, dtype=native.float32).to('cuda:0')
                    tx = torch.tensor(data, dtype=torch.float32, device='cuda:0')
                    with self.subTest(expression=expression, scalar=scalar, kind=type(scalar).__name__, data=data):
                        actual = torch.tensor(candidate(x).cpu().tolist(), dtype=torch.float32)
                        expected = reference(tx).cpu()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                        zeros = (actual == 0) & (expected == 0)
                        self.assertTrue(torch.equal(actual.signbit()[zeros], expected.signbit()[zeros]))
            native.compiler.reset()
            torch.compiler.reset()


if __name__ == '__main__':
    unittest.main()
