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

    def check_graph(self, source, patterns, *, finite_history=False):
        """Check cached native shapes against static IEEE or finite warm references."""
        torch = self.torch
        fn = program(source)
        compiled, reference = native.compile(fn), torch.compile(program(source, torch))
        values = [2e38, -2e38, 0., -0., float('inf'), -float('inf'),
                  float('nan'), 1.0000001192092896, 1e-38, -1e-38, 1.137, -1.137]
        if finite_history:
            values = [1.137, -1.137, 0., -0., 0.713, -0.337]
        for shapes in patterns:
            if not finite_history:
                # Native modules specialize concrete shapes. Reference automatic
                # symbolic recompilation can change FMA order after shape changes;
                # compare exceptional values with a fresh default specialization.
                # Keep the native wrapper alive to exercise its shape/code guards.
                torch.compiler.reset()
                reference = torch.compile(program(source, torch))
            for changed in (False, True):
                args, refs = [], []
                for shape in shapes:
                    data = [values[i % len(values)] for i in range(math.prod(shape))]
                    if changed:
                        data = [-v for v in data[::-1]]
                    for fw, destination in ((native, args), (torch, refs)):
                        # Changed calls use fresh offset-contiguous bindings.
                        base = self.upload([91.] + data + [92.], (len(data) + 2,), fw)
                        destination.append(base[1:len(data) + 1].reshape(shape))
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
                         finite_history=True)


del Hardware

if __name__ == '__main__':
    unittest.main()
