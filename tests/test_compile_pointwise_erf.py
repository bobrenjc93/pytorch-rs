"""Private Erf admission during the conditional GELU integration gate."""
import struct
import unittest

from torch_rs import torch_rs as bridge


def constant(value):
    return ('constant', 0, 0, struct.unpack('<Q', struct.pack('<d', value))[0])


class PrivateErf(unittest.TestCase):
    def test_strict_unary_operand_and_payload(self):
        prefix = (('input', 0, 0, 0),)
        for invalid in [('erf', 0, 1, 0), ('erf', 0, 0, 1), ('erf', 1, 0, 0),
                        ('erf', True, 0, 0)]:
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError, NotImplementedError)):
                bridge._pointwise_plan(prefix + (invalid,), (1,), 1)
        with self.assertRaisesRegex(ValueError, 'tensor expression'):
            bridge._pointwise_plan((constant(.75), ('erf', 0, 0, 0)), (1,), 1)
        with self.assertRaisesRegex(ValueError, 'earlier SSA'):
            bridge._pointwise_plan(prefix + (('erf', 2, 0, 0), ('neg', 0, 0, 0)), (2,), 1)

    def test_live_equal_actual_shape_boundary_includes_unused_inputs(self):
        nodes = (('input', 0, 0, 0), ('input', 1, 0, 0), ('erf', 0, 0, 0),
                 ('relu', 1, 0, 0), ('mul', 2, 0, 0))
        for shapes in [([2, 1], [1, 2]), ([1], [1, 1]), ([0], [1])]:
            with self.subTest(shapes=shapes):
                for roots in [(2,), (4,)]:
                    with self.assertRaisesRegex(RuntimeError, 'live Erf requires equal actual input shapes'):
                        bridge._pointwise_plan(nodes, roots, 2, shapes)
                # Dead Erf does not narrow the accepted broadcast-trig surface.
                bridge._pointwise_plan(nodes, (3,), 2, shapes)
        bridge._pointwise_plan(nodes, (2, 4), 2, [[2, 3], [2, 3]])

    def test_constant_erf_uses_float32_and_keeps_original_shape_validation(self):
        nodes = (('input', 0, 0, 0), ('integer', 0, 0, 0), ('mul', 0, 1, 0),
                 constant(.75), ('add', 2, 3, 0), ('erf', 4, 0, 0))
        plan = bridge._pointwise_plan(nodes, (5,), 1, [[13]])
        self.assertIn('torch_rs_erf(', plan)
        self.assertNotIn('x0[i]', plan)
        self.assertNotIn('(double)', plan)
        # Invalid dead broadcast is checked before liveness and zero folding.
        bad = nodes + (('input', 1, 0, 0), ('add', 0, 6, 0))
        with self.assertRaisesRegex(RuntimeError, 'broadcast'):
            bridge._pointwise_plan(bad, (5,), 2, [[2], [3]])

    def test_cse_does_not_merge_independent_output_slots(self):
        nodes = (('input', 0, 0, 0), ('erf', 0, 0, 0), ('erf', 0, 0, 0))
        plan = bridge._pointwise_plan(nodes, (1, 2), 1, [[13]])
        self.assertEqual(plan.count('torch_rs_erf('), 1)
        self.assertIn('out0[i]', plan)
        self.assertIn('out1[i]', plan)
        self.assertNotIn('torch_rs_erf', bridge._pointwise_plan(nodes + (('neg', 0, 0, 0),), (3,), 1))

    def test_private_erf_is_not_a_public_callable_or_tensor_method(self):
        import torch_rs
        for name in ('erf', 'gelu'):
            self.assertFalse(hasattr(torch_rs, name))
            self.assertFalse(hasattr(torch_rs.Tensor, name))


if __name__ == '__main__':
    unittest.main()
