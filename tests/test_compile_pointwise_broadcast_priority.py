"""Reject the historical contraction failures at original-IR admission.

The measured broad-domain failures remain in docs/diagnostics/compile-pointwise-
broadcast. These tests establish exclusion, not a numerical repair.
"""
import math
import os
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import torch_rs as bridge
from test_compile_pointwise_jit import Hardware, available, cache, kernel, lower, program

BOUNDARY = 'unequal input shapes require at most one arithmetic stage and no live sin/cos'
CANCELLATION = 'def f(x, y):\n a=x+1.0\n return x*y+a*a'


class Metadata(unittest.TestCase):
    def source(self, source, shapes):
        graph = lower(program(source), 2)
        return bridge._pointwise_source(graph.nodes, graph.outputs, 2, shapes)

    def test_original_depth_precedes_simplification_and_address_canonicalization(self):
        for expression in ('(x*1)*1', '(x+0)-0', '-(-x)', '(x-x)*0',
                           '(x+y)-(x+y)', '(x*y).relu()+0', 'x.sin()', 'y.cos()',
                           'x.sin()*False', 'x.cos()*0'):
            for shapes in (((2, 1), (1, 3)), ((2,), (1, 2)), ((), (1,))):
                with self.subTest(expression=expression, shapes=shapes):
                    with self.assertRaisesRegex(RuntimeError, BOUNDARY):
                        self.source('def f(x,y):\n return '+expression, shapes)
            # The pre-existing equal-shape numerical domain is unchanged.
            self.assertIn('torch_rs_pointwise', self.source(
                'def f(x,y):\n return '+expression, ((2,), (2,))))

    def test_relu_is_depth_neutral_and_dead_nodes_do_not_limit_live_capability(self):
        for expression in ('x.relu()', '-x.relu()', '(-x).relu()',
                           '(x.relu()+y.relu()).relu().relu()',
                           'x*x', 'x+(-1.25)', 'x*True'):
            source = 'def f(x,y):\n dead=x.sin().cos()*y+x\n return '+expression
            with self.subTest(expression=expression):
                code = self.source(source, ((2, 1), (1, 3)))
                self.assertNotIn('sinf(', code)
                self.assertNotIn('cosf(', code)
                self.assertNotIn('fmaf(', code)

    def test_dead_graph_still_validates_before_live_admission(self):
        graph = lower(program('def f(x,y):\n dead=x+y\n return -x'), 2)
        with self.assertRaisesRegex(RuntimeError, 'broadcast'):
            bridge._pointwise_source(graph.nodes, graph.outputs, 2, ((2,), (3,)))
        # Structural validation applies even to a dead invalid SSA operand.
        nodes = (('input', 0, 0, 0), ('neg', 0, 0, 0), ('add', 0, 99, 0))
        with self.assertRaisesRegex(ValueError, 'earlier SSA node'):
            bridge._pointwise_source(nodes, (1,), 2, ((2,), (1, 2)))

    def test_runtime_scalar_sign_is_metadata_not_tensor_negation(self):
        nodes = (('input', 0, 0, 0), ('input', 1, 0, 0),
                 ('scalar', 0, 1, 0), ('mul', 0, 2, 0), ('relu', 3, 0, 0))
        code = bridge._pointwise_source(nodes, (4,), 2, ((2,1), (1,3)))
        self.assertIn('float s0', code)
        self.assertNotIn('fmaf(', code)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class BroadcastPriority(unittest.TestCase):
    setUpClass = classmethod(Hardware.setUpClass.__func__)
    tearDown = Hardware.tearDown
    upload = Hardware.upload
    compare = Hardware.compare
    without_replay = Hardware.without_replay

    def rejected_history(self, source, patterns, values):
        fn = program(source)
        compiled = native.compile(fn)
        for shapes in patterns:
            args = [self.upload([data[i % len(data)] for i in range(math.prod(shape))], shape)
                    for shape, data in zip(shapes, values, strict=True)]
            for _ in range(2):
                with self.subTest(source=source, shapes=shapes):
                    # Rejection must precede NVRTC loading, including empty output.
                    with patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/bounded-test-nvrtc'):
                        with self.assertRaisesRegex(NotImplementedError, BOUNDARY):
                            self.without_replay(fn, compiled, args)
                    self.assertFalse(cache(compiled).graphs)
        native.compiler.reset()
        self.assertFalse(cache(compiled).graphs)

    def test_historical_finite_and_ieee_shape_histories_are_explicitly_unsupported(self):
        shapes = [((2,1), (1,2)), ((1,2), (2,1)), ((2,1), (2,2))]
        for values in (((4096.,), (-4098.,)), ((2e38,-2e38), (2e38,-2e38))):
            self.rejected_history(CANCELLATION, shapes, values)

    def test_historical_fresh_large_cancellation_is_explicitly_unsupported(self):
        self.rejected_history(CANCELLATION,
            [((257,1), (257,257)), ((257,257), (257,1)),
             ((259,1), (259,263)), ((17,1), (17,33))], ((4096.,), (-4098.,)))

    def test_products_trig_unused_inputs_empty_and_simplified_graphs_reject(self):
        for source in (CANCELLATION,
            'def f(x,y):\n a=y.relu()\n return a*a+x*y',
            'def f(x,y):\n dead=y.cos()\n return (x*1)*1',
            'def f(x,y):\n return x.sin()',
            'def f(x,y):\n return y.cos()',
            'def f(x,y):\n return x.sin()*False',
            'def f(x,y):\n return x.cos()*0',
            'def f(x,y):\n return -(-x)',
            'def f(x,y):\n return x*0+0'):
            self.rejected_history(source,
                [((2,), (1,2)), ((), (1,)), ((0,1), (1,3))], ((1.,), (2.,)))

    def test_direct_run_cannot_bypass_unequal_shapes_with_linear_addresses(self):
        for source in (CANCELLATION, 'def f(x,y):\n return x.sin()',
                       'def f(x,y):\n return (x*1)*1'):
            fn = program(source)
            compiled = native.compile(fn)
            x, y = self.upload([1.,2.], (2,)), self.upload([3.,4.], (2,))
            equal = compiled(x,y)
            selected = kernel(compiled)
            self.assertEqual(len(cache(compiled).graphs), 1)
            for shape in ((1,2), (1,1,2)):
                unequal_y = y.reshape(shape)
                with self.assertRaisesRegex(RuntimeError, BOUNDARY):
                    selected.run((x, unequal_y))
                with self.assertRaisesRegex(NotImplementedError, BOUNDARY):
                    compiled(x, unequal_y)
                self.assertEqual(len(cache(compiled).graphs), 1)
            # Failed admission neither poisons existing code nor publishes entries.
            result = self.without_replay(fn, compiled, (x,y))
            self.compare(result, self.torch.tensor(equal.cpu().tolist(), device='cuda:0'))
            native.compiler.reset()
            self.assertFalse(cache(compiled).graphs)
            self.assertEqual(selected.run((x,y))[0].cpu().tolist(), equal.cpu().tolist())
            compiled(x,y)
            self.assertIsNot(kernel(compiled), selected)
            native.compiler.reset()


del Hardware

if __name__ == '__main__':
    unittest.main()
