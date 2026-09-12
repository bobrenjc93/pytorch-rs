"""Nonfinite specializations must obey ordinary default-compiler cache behavior."""
import math
import unittest

import torch_rs as native
from tests import test_compile_pointwise_operator_warm_bindings as helpers
from tests.test_compile_pointwise_jit import available, program


@unittest.skipUnless(available(), "requires native and reference CUDA")
class NonfiniteBindingHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def check_independent_sequence(self, values):
        # A reset separates independent experiments, never steps or fresh-input
        # calls within a sequence. Each sequence reuses both compiled wrappers.
        native.compiler.reset()
        self.torch.compiler.reset()
        helpers.WarmBindingSequences.check_sequence(
            self, "((x*0)+a)-16777216.0", [{"a": value} for value in values])

    def test_static_hit_survives_nonfinite_only_history(self):
        for interlude in ((math.inf,), (-math.inf,), (math.nan,), (math.nan, math.inf)):
            with self.subTest(interlude=interlude):
                self.check_independent_sequence([16777217.0, *interlude, 16777217.0])

    def test_nonfinite_first_still_promotes_new_finite_values(self):
        for first in (math.inf, -math.inf, math.nan):
            with self.subTest(first=first):
                self.check_independent_sequence([first, 16777217.0, 16777218.0, 16777217.0])

    def test_finite_change_after_interlude_keeps_promotion(self):
        for middle in (math.inf, -math.inf, math.nan):
            with self.subTest(middle=middle):
                self.check_independent_sequence([16777217.0, middle, 16777218.0, 16777217.0])

    def test_prior_promotion_survives_nonfinite_interlude(self):
        for middle in (math.inf, -math.inf, math.nan):
            with self.subTest(middle=middle):
                self.check_independent_sequence([16777217.0, 16777218.0, middle, 16777217.0])

    def test_integer_entry_is_not_a_matching_static_float(self):
        for first in (16777216, 16777217):
            for middle in (math.inf, -math.inf, math.nan):
                with self.subTest(first=first, middle=middle):
                    self.check_independent_sequence([first, middle, 16777217.0])

    def test_contiguous_offset_change_keeps_the_static_specialization(self):
        source = "def f(x):\n return ((x*0)+a)-16777216.0"
        for offsets in ((0, 0, 1), (1, 0, 2), (0, 1, 0)):
            native.compiler.reset()
            self.torch.compiler.reset()
            fn, ref_fn = program(source, a=0.0), program(source, a=0.0)
            compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
            for value, offset in zip((16777217.0, math.inf, 16777217.0), offsets):
                fn.__globals__["a"] = ref_fn.__globals__["a"] = value
                for data in ([1.0, -1.0, 2.0, -2.0, 3.0], [3.0, -3.0, 4.0, -4.0, 5.0]):
                    with self.subTest(offsets=offsets, binding=value, offset=offset, data=data):
                        x = native.tensor(data, dtype=native.float32).to("cuda:0")[offset:offset + 2]
                        tx = self.torch.tensor(data, dtype=self.torch.float32, device="cuda:0")[offset:offset + 2]
                        result = compiled(x)
                        expected = reference(tx).cpu()
                        actual = self.torch.tensor(result.cpu().tolist(), dtype=self.torch.float32)
                        self.torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                        self.assertTrue(self.torch.equal(actual.view(self.torch.int32), expected.view(self.torch.int32)))
                        self.assertNotEqual(result.data_ptr(), x.data_ptr())
                        self.assertEqual(result.shape, tuple(expected.shape))
                        self.assertEqual(str(result.device), "cuda:0")
                        self.torch.testing.assert_close(self.torch.tensor(x.cpu().tolist()), tx.cpu(),
                                                       rtol=0, atol=0, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
