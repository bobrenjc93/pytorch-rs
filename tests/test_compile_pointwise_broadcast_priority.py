"""Broadcast load materialization must participate in default-Inductor FMA order."""
import math
import unittest

import torch_rs as native
from torch_rs import torch_rs as bridge
from test_compile_pointwise_jit import Hardware, available, cache, lower, program


class Metadata(unittest.TestCase):
    def test_broadcast_loads_change_competing_product_priority(self):
        graph = lower(program('def f(x, y):\n a=x+1.0\n return x*y+a*a'), 2)
        def source(shapes):
            return bridge._pointwise_source(graph.nodes, graph.output, 2, shapes)
        # The later broadcast load has an additional cache-policy materialization.
        self.assertIn('fmaf(v3, v3, v4)', source(((2, 1), (1, 2))))
        self.assertIn('fmaf(v3, v3, v4)', source(((2, 2), (1, 2))))
        for shapes in (((2, 2), (2, 2)), ((2, 1), (2, 2)),
                       ((2, 2), ()), ((), (2, 2)), ((2,), (1, 2))):
            with self.subTest(shapes=shapes):
                self.assertIn('fmaf(v0, v1, v5)', source(shapes))

    def test_dead_broadcast_load_cannot_change_live_ranks(self):
        source = 'def f(x, y):\n dead=y.relu()\n a=x+1.0\n return x*x+a*a'
        graph = lower(program(source), 2)
        code = bridge._pointwise_source(graph.nodes, graph.output, 2, ((2, 1), (1, 3)))
        self.assertNotIn('= x1[', code)
        self.assertEqual(code.count('fmaf('), 1)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class BroadcastPriority(unittest.TestCase):
    setUpClass = classmethod(Hardware.setUpClass.__func__)
    tearDown = Hardware.tearDown
    upload = Hardware.upload
    compare = Hardware.compare
    without_replay = Hardware.without_replay

    def check_graph(self, source, patterns, *, input_values=None):
        """Keep both wrappers alive across all shapes and changed bindings."""
        torch = self.torch
        fn = program(source)
        compiled, reference = native.compile(fn), torch.compile(program(source, torch))
        values = [2e38, -2e38, 0., -0., float('inf'), -float('inf'),
                  float('nan'), 1.0000001192092896, 1e-38, -1e-38, 1.137, -1.137]
        if input_values is None:
            input_values = (values, values)
        for shapes in patterns:
            for changed in (False, True):
                args, refs = [], []
                for shape, values in zip(shapes, input_values, strict=True):
                    data = [values[i % len(values)] for i in range(math.prod(shape))]
                    if changed:
                        data = [-v for v in data[::-1]]
                    for fw, destination in ((native, args), (torch, refs)):
                        if changed:
                            # Changed calls use fresh offset-contiguous bindings.
                            base = self.upload([91.] + data + [92.], (len(data) + 2,), fw)
                            destination.append(base[1:len(data) + 1].reshape(shape))
                        else:
                            destination.append(self.upload(data, shape, fw))
                with self.subTest(source=source, shapes=shapes, changed=changed):
                    expected = reference(*refs)
                    actual = self.without_replay(fn, compiled, args)
                    self.compare(actual, expected)
                    again = self.without_replay(fn, compiled, args)
                    self.compare(again, expected)
                    self.assertNotEqual(actual.data_ptr(), again.data_ptr())
                    for arg, ref in zip(args, refs):
                        self.assertNotEqual(actual.data_ptr(), arg.data_ptr())
                        self.compare(arg, ref, exact=True)
        for entry in cache(compiled).graphs.values():
            self.assertEqual(entry[1].ptx.count('.visible .entry'), 1)
        native.compiler.reset()
        torch.compiler.reset()

    def test_products_shapes_orders_and_changed_bindings(self):
        patterns = [((2, 1), (1, 2)), ((1, 2), (2, 1)),
                    ((2, 1), (2, 2)), ((2, 2), (1, 2)),
                    ((2, 2), (2, 2)), ((2,), (1, 2)),
                    ((), (2, 3, 2)), ((2, 3, 2), ())]
        for input_name in ('x', 'y'):
            for producer in (f'{input_name}+1.0', f'{input_name}.relu()',
                             f'{input_name}.relu()+{input_name}',
                             f'{input_name}.sin()+{input_name}'):
                for expression in ('x*y+a*a', 'a*a+x*y'):
                    self.check_graph(f'def f(x, y):\n a={producer}\n return {expression}', patterns)

    def test_interior_singletons_shared_dead_and_signed_products(self):
        patterns = [((3, 1, 2), (1, 2, 1)), ((1, 2, 1), (3, 1, 2)),
                    ((2, 1, 3, 1), (1, 2, 1, 2)),
                    ((17, 1), (1, 33)), ((1, 65), (17, 1))]
        for source in (
            'def f(x, y):\n dead=y.cos()\n a=x+1.0\n return x*y+a*a',
            'def f(x, y):\n a=y.relu()\n return a*a+x*y',
            'def f(x, y):\n a=x+1.0\n p=x*y\n return (p+a*a)-p',
            'def f(x, y):\n a=x+1.0\n return (x*y)*-1.0+a*a',
            'def f(x, y):\n a=x+1.0\n return a*a-x*y',
        ):
            self.check_graph(source, patterns)

    def test_finite_shape_history_without_reference_reset(self):
        self.check_graph('def f(x, y):\n a=x+1.0\n return x*y+a*a',
                         [((2, 1), (1, 2)), ((1, 2), (2, 1)),
                          ((2, 1), (2, 2)), ((3, 1), (1, 5))],
                         input_values=((4096.,), (-4098.,)))

    def test_ieee_shape_history_without_reference_reset(self):
        self.check_graph('def f(x, y):\n a=x+1.0\n return x*y+a*a',
                         [((2, 1), (1, 2)), ((1, 2), (2, 1)),
                          ((2, 1), (2, 2))],
                         input_values=((2e38, -2e38), (2e38, -2e38)))

    def test_fresh_large_broadcast_cancellation(self):
        for shapes in (((257, 1), (257, 257)),
                       ((257, 257), (257, 1)),
                       ((259, 1), (259, 263))):
            with self.subTest(shapes=shapes):
                self.check_graph('def f(x, y):\n a=x+1.0\n return x*y+a*a',
                                 [shapes], input_values=((4096.,), (-4098.,)))


del Hardware

if __name__ == '__main__':
    unittest.main()
