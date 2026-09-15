"""Persistent default-Inductor parity for the bounded broadcast expression domain."""
import math
import random
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend
from torch_rs import torch_rs as bridge
from test_compile_pointwise_jit import Hardware, available, cache, kernels, program


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class BroadcastPrimitives(unittest.TestCase):
    setUpClass = classmethod(Hardware.setUpClass.__func__)
    tearDown = Hardware.tearDown
    upload = Hardware.upload
    compare = Hardware.compare
    without_replay = Hardware.without_replay

    def inputs(self, shapes, changed):
        values = [0., -0., 1e-40, -1e-40, 1e-38, -1e-38, 2e38, -2e38,
                  float('inf'), -float('inf'), float('nan'), 4096., -4098.,
                  1.0000001192092896, -1.137, .713]
        args, refs, backing = [], [], []
        for index, shape in enumerate(shapes):
            data = [values[(i+index*7+changed*3) % len(values)]
                    for i in range(math.prod(shape))]
            if changed:
                data.reverse()
            # Every changed binding is an offset-contiguous view. Keep and check
            # the entire allocation, including both sentinels, after execution.
            for fw, destination in ((native,args), (self.torch,refs)):
                base = self.upload([91.]+data+[92.], (len(data)+2,), fw)
                destination.append(base[1:len(data)+1].reshape(shape))
                if fw is native:
                    backing.append((base, list([91.]+data+[92.])))
        return args, refs, backing

    def check_graph(self, source, patterns, bindings=None):
        bindings = bindings or {}
        fn, ref_fn = program(source, **bindings), program(source, self.torch, **bindings)
        compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
        for shapes in patterns:
            for changed in (0,1):
                args, refs, backing = self.inputs(shapes, changed)
                with self.subTest(source=source, shapes=shapes, changed=changed):
                    expected = reference(*refs)
                    actual = self.without_replay(fn, compiled, args)
                    self.compare(actual, expected)
                    with patch.object(frontend, 'analyze', side_effect=AssertionError('warm analysis')), \
                         patch.object(frontend, 'lower', side_effect=AssertionError('warm lowering')), \
                         patch.object(bridge, '_pointwise_compile', side_effect=AssertionError('warm compile')):
                        again = self.without_replay(fn, compiled, args)
                    self.compare(again, expected)
                    self.assertIsNot(again, actual)
                    if expected.numel():
                        self.assertNotEqual(actual.data_ptr(), again.data_ptr())
                        for arg in args:
                            self.assertNotEqual(actual.data_ptr(), arg.data_ptr())
                    for arg, ref in zip(args,refs):
                        self.compare(arg, ref, exact=True)
                    for base, data in backing:
                        self.compare(base, self.upload(data, (len(data),), self.torch), exact=True)
        # These sources contain no second arithmetic stage or competing product;
        # generated lowering must not introduce an FMA with another stage.
        for entry in kernels(compiled):
            self.assertNotIn('fmaf(', entry.plan(1))
            self.assertEqual(entry.ptx.count('.visible .entry'), 1)
        native.compiler.reset()
        self.torch.compiler.reset()

    def test_generated_primitives_relu_orders_ranks_and_ieee(self):
        rng = random.Random(194027)
        patterns = [((2,1),(1,2)), ((1,2),(2,1)), ((2,1),(2,2)),
                    ((3,1,2),(1,4,1)), ((1,2,1,3),(2,1,4,1)),
                    ((),(2,3)), ((0,1),(1,3))]
        # Generate independently composed depth-one expressions; every binary
        # operation and argument order is exercised with arbitrary ReLU chains.
        for operation in ('+', '-', '*'):
            for left, right in (('x','y'), ('y','x')):
                for _ in range(2):
                    lhs = left + '.relu()'*rng.randrange(3)
                    rhs = right + '.relu()'*rng.randrange(3)
                    expression = f'({lhs}{operation}{rhs})' + '.relu()'*rng.randrange(3)
                    self.check_graph('def f(x,y):\n return '+expression, patterns)

    def test_large_shapes_linear_maps_empty_scalar_and_repeated_inputs(self):
        patterns = [((257,1),(257,257)), ((257,257),(257,1)),
                    ((17,1),(17,33)), ((33,),(1,33)),
                    ((1,),( )), ((1,1,1),(0,2,3)), ((3,1),(1,5))]
        for expression in ('x+y', 'y-x', 'x*y', '(x.relu()*y.relu()).relu()',
                           'x*x', 'y-y', '-x.relu()', '(-y).relu()'):
            self.check_graph('def f(x,y):\n return '+expression, patterns)

    def test_captured_float_promotion_across_persistent_shape_history(self):
        shapes = [((2,1),(1,3)), ((1,2),(3,1)), ((),(2,3))]
        for expression in ('x+scale', 'scale-x', '(y.relu()*(-scale)).relu()'):
            source = 'def f(x,y):\n return '+expression
            fn, ref_fn = program(source, scale=1.137), program(source, self.torch, scale=1.137)
            compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
            for value in (1.137, -2.25, -0.0, 0.0, 1.137):
                fn.__globals__['scale'] = ref_fn.__globals__['scale'] = value
                for pattern in shapes:
                    args, refs, _ = self.inputs(pattern, value != 1.137)
                    with self.subTest(expression=expression, value=value, shapes=pattern):
                        self.compare(self.without_replay(fn, compiled, args), reference(*refs))
            self.assertTrue(any('float s0' in entry.source for entry in kernels(compiled)))
            self.assertLessEqual(len(cache(compiled).graphs), 6)
            native.compiler.reset()
            self.torch.compiler.reset()

    def test_scalar_kind_and_signed_zero_identity_rules(self):
        for expression in ('x+scale', 'x*scale', 'scale*x'):
            source = 'def f(x,y):\n return '+expression
            # bool/int values retain their kind, rather than sharing a promoted
            # float identity. One wrapper per framework sees the full sequence.
            fn, ref_fn = program(source, scale=False), program(source, self.torch, scale=False)
            compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
            for value in (False, 0, True, 1, 0.0, -0.0):
                fn.__globals__['scale'] = ref_fn.__globals__['scale'] = value
                args, refs, _ = self.inputs(((16,1),(1,2)), False)
                with self.subTest(expression=expression, kind=type(value), value=value):
                    self.compare(self.without_replay(fn, compiled, args), reference(*refs))
            native.compiler.reset()
            self.torch.compiler.reset()


del Hardware

if __name__ == '__main__':
    unittest.main()
