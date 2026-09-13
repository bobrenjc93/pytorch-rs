"""Warm binding promotion must survive reversion and remain per binding."""
import unittest

import torch_rs as native
from tests.test_compile_pointwise_jit import available, program


@unittest.skipUnless(available(), "requires native and reference CUDA")
class WarmBindingSequences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def check_sequence(self, expression, sequence):
        source = "def f(x):\n return " + expression
        fn = program(source, **sequence[0])
        reference_fn = program(source, **sequence[0])
        compiled = native.compile(fn)
        reference = self.torch.compile(reference_fn)
        # Compile once per framework. Neither wrapper nor reference cache is
        # reset between changed bindings, type transitions, or earlier values.
        values = [-0.0, 0.0, 1.0, -1.0, float("inf"), float("nan")]
        for step, bindings in enumerate(sequence):
            fn.__globals__.update(bindings)
            reference_fn.__globals__.update(bindings)
            for data in (values, values[::-1]):
                with self.subTest(step=step, bindings=bindings, reversed=data is not values):
                    x = native.tensor(data, dtype=native.float32).to("cuda:0")
                    tx = self.torch.tensor(data, dtype=self.torch.float32, device="cuda:0")
                    result = compiled(x)
                    expected = reference(tx).cpu()
                    actual = self.torch.tensor(result.cpu().tolist(), dtype=self.torch.float32)
                    self.torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                    present = ~expected.isnan()
                    self.assertTrue(self.torch.equal(actual.view(self.torch.int32)[present],
                                                     expected.view(self.torch.int32)[present]))
                    self.assertNotEqual(result.data_ptr(), x.data_ptr())
                    self.assertEqual(result.shape, tuple(expected.shape))
                    self.assertEqual(str(result.device), "cuda:0")
                    self.torch.testing.assert_close(self.torch.tensor(x.cpu().tolist()), tx.cpu(),
                                                   rtol=0, atol=0, equal_nan=True)

    def test_returning_to_original_float_keeps_runtime_boundary(self):
        self.check_sequence("((x*0)+a)-16777216.0", [
            {"a": 16777217.0}, {"a": 16777218.0}, {"a": 16777217.0},
        ])

    def test_two_captures_promote_independently(self):
        self.check_sequence("(((x*0)+a)-16777216.0)+(((x*0)+b)-16777216.0)", [
            {"a": 16777216.0, "b": 16777217.0},
            {"a": 16777217.0, "b": 16777217.0},
            {"a": 16777217.0, "b": 16777219.0},
            {"a": 16777216.0, "b": 16777217.0},
        ])

    def test_integer_interlude_does_not_forget_float_promotion(self):
        self.check_sequence("((x*0)+a)-16777216.0", [
            {"a": 16777217.0}, {"a": 16777218.0},
            {"a": 16777217}, {"a": 16777217.0},
        ])


if __name__ == "__main__":
    unittest.main()
