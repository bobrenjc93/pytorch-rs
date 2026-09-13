"""Constant-only unary results must preserve later numerical boundaries."""
import unittest

import torch_rs as native
from tests.test_compile_pointwise_jit import available, program


@unittest.skipUnless(available(), "requires native and reference CUDA")
class ConstantTranscendentalBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def check(self, expression):
        source = "def f(x):\n return " + expression
        compiled = native.compile(program(source))
        reference = self.torch.compile(program(source))
        values = [1.0, -1.0, 0.5, 1.5, -0.0, 0.0]
        for data in (values, values[::-1]):
            with self.subTest(expression=expression, reversed=data is not values):
                x = native.tensor(data, dtype=native.float32).to("cuda:0")
                tx = self.torch.tensor(data, dtype=self.torch.float32, device="cuda:0")
                result = compiled(x)
                expected = reference(tx).cpu()
                actual = self.torch.tensor(result.cpu().tolist(), dtype=self.torch.float32)
                self.torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=True)
                zero = (actual == 0) & (expected == 0)
                self.assertTrue(self.torch.equal(actual.signbit()[zero], expected.signbit()[zero]))
                self.assertNotEqual(result.data_ptr(), x.data_ptr())
                self.assertEqual(result.shape, tuple(expected.shape))
                self.torch.testing.assert_close(self.torch.tensor(x.cpu().tolist()), tx.cpu(),
                                               rtol=0, atol=0, equal_nan=True)

    def test_constant_sine_cancellation_can_amplify_a_single_ulp(self):
        for zero in ("x*0", "x*False"):
            for value, sine in (("1.0", "0.8414709848078965"),
                                ("-1.0", "-0.8414709848078965")):
                self.check(f"((({zero}+{value}).sin())-({sine}))*1e38")

    def test_constant_cosine_cancellation_control(self):
        self.check("((((x*0)+1.0).cos())-0.5403023058681398)*1e38")

    def test_runtime_trigonometric_controls(self):
        for expression in ("(x.sin()-0.8414709848078965)*1e38",
                           "(x.cos()-0.5403023058681398)*1e38",
                           "(x.sin()-x.sin())*1e38"):
            self.check(expression)


if __name__ == "__main__":
    unittest.main()
