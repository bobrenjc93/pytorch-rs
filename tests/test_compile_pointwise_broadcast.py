"""Independent broadcast graphs for the ordinary native default CUDA compiler."""
import gc
import math
import os
import random
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import torch_rs as bridge
from torch_rs import _compile_pointwise as frontend
from test_compile_pointwise_jit import (
    Hardware, available, cache, kernel, kernels, legacy_kernel, lower, program, two_device_reservation,
)


class Metadata(unittest.TestCase):
    def test_codegen_uses_metadata_coordinates_and_keeps_numeric_lowering(self):
        graph = lower(program('def f(x, y):\n return x.relu() - y.relu()'), 2)
        source = bridge._pointwise_source(graph.nodes, graph.outputs, 2, ((3, 1), (2, 1, 5)))
        self.assertIn('x0[((i / 5ull) % 3ull) * 1ull]', source)
        self.assertIn('x1[((i / 15ull) % 2ull) * 5ull + ((i / 1ull) % 5ull) * 1ull]', source)
        self.assertIn('__fsub_rn(', source)
        self.assertNotIn('fmaf(', bridge._pointwise_plan(graph.nodes, graph.outputs, 2, ((3, 1), (2, 1, 5))))
        self.assertEqual(source.count('__global__'), 1)
        scalar = bridge._pointwise_source(graph.nodes, graph.outputs, 2, ((), (2, 3)))
        self.assertIn('x0[0]', scalar)
        self.assertIn('x1[i]', scalar)

    def test_dead_shape_errors_are_not_removed_and_empty_codegen_has_no_zero_divisor(self):
        graph = lower(program('def f(x, y):\n dead = x + y\n return -x'), 2)
        with self.assertRaisesRegex(RuntimeError, 'incompatible.*broadcast'):
            bridge._pointwise_source(graph.nodes, graph.outputs, 2, ((3,), (5,)))
        source = bridge._pointwise_source(graph.nodes, graph.outputs, 2, ((3, 1), (2, 1, 5)))
        self.assertIn('x0[i]', source)
        self.assertNotIn('= x1[', bridge._pointwise_plan(graph.nodes, graph.outputs, 2, ((3, 1), (2, 1, 5))))
        graph = lower(program('def f(x, y):\n return x + y'), 2)
        source = bridge._pointwise_source(graph.nodes, graph.outputs, 2, ((1, 0, 3), (2, 1, 1)))
        self.assertNotIn('/ 0ull', source)
        self.assertNotIn('% 0ull', source)
        for shapes in (((0,), (2,)), ((1 << 32, 1), (1, 1 << 32)), ((0, (1 << 64) - 1, 2), ())):
            with self.subTest(shapes=shapes), self.assertRaises(RuntimeError):
                bridge._pointwise_source(graph.nodes, graph.outputs, 2, shapes)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class Broadcast(unittest.TestCase):
    setUpClass = classmethod(Hardware.setUpClass.__func__)
    tearDown = Hardware.tearDown
    upload = Hardware.upload
    compare = Hardware.compare
    without_replay = Hardware.without_replay

    def test_generated_graphs_shapes_orders_values_and_offsets(self):
        rng = random.Random(8570123)
        for case in range(10):
            # Generate arbitrary ReLU placements around exactly one arithmetic
            # stage. Dead multi-stage/transcendental nodes still validate shapes.
            lines = ['def f(x, y):', ' dead = (x + y).sin() * 1.137']
            terms = []
            for name in ('x', 'y'):
                expression = name + '.relu()' * rng.randrange(4)
                lines.append(f' {name}r = {expression}')
                terms.append(f'{name}r')
            rng.shuffle(terms)
            expression = f'({terms[0]} {rng.choice(["+", "-", "*"])} {terms[1]})'
            lines.append(' return ' + expression + '.relu()' * rng.randrange(4))
            source = '\n'.join(lines)
            fn, ref_fn = program(source), program(source, self.torch)
            compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
            rank = rng.randrange(2, 6)
            dimensions = [rng.randrange(2, 5) for _ in range(rank)]
            left = tuple(d if rng.randrange(2) else 1 for d in dimensions)
            right = tuple(d if l == 1 or rng.randrange(2) else 1 for d, l in zip(dimensions, left))
            patterns = [(left, right), (right, left), ((), (2, 3, 1)), ((2, 3, 1), ()),
                        ((3, 1), (2, 1, 5)), ((1, 0, 3), (2, 1, 1)), ((), ())]
            for shapes in patterns:
                for changed in range(2):
                    args, refs = [], []
                    for shape in shapes:
                        values = [rng.uniform(-1.5, 1.5) for _ in range(math.prod(shape))]
                        # Every other binding is an offset-contiguous view.
                        for fw, destination in [(native, args), (self.torch, refs)]:
                            if changed:
                                base = self.upload([91., 92.] + values + [93.], (len(values) + 3,), fw)
                                destination.append(base[2:2 + len(values)].reshape(shape))
                            else:
                                destination.append(self.upload(values, shape, fw))
                    with self.subTest(case=case, shapes=shapes, changed=changed):
                        expected = reference(*refs)
                        actual = self.without_replay(fn, compiled, args)
                        self.compare(actual, expected)
                        again = self.without_replay(fn, compiled, args)
                        self.compare(again, expected)
                        if math.prod(actual.shape):
                            self.assertNotEqual(actual.data_ptr(), again.data_ptr())
                        for arg, ref in zip(args, refs):
                            self.compare(arg, ref, exact=True)
                            if math.prod(actual.shape) and math.prod(arg.shape):
                                self.assertNotEqual(actual.data_ptr(), arg.data_ptr())
            for entry in kernels(compiled):
                self.assertEqual(entry.ptx.count('.visible .entry'), 1)
            native.compiler.reset()
            self.torch.compiler.reset()

    def test_shape_guards_rebindings_warm_execution_and_module_lifetime(self):
        source = 'def f(x, y):\n return fw.relu(x) * scale'
        fn = program(source, scale=0.713)
        ref_fn = program(source, self.torch, scale=0.713)
        compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
        x, y = self.upload([1., 2., 3.], (3, 1)), self.upload([0.1, 0.2], (2,))
        tx, ty = self.upload([1., 2., 3.], (3, 1), self.torch), self.upload([0.1, 0.2], (2,), self.torch)
        first = compiled(x, y)
        old = kernel(compiled)
        old_prepared = next(reversed(cache(compiled).prepared.values()))[0]
        for scale in (0.713, -1.137, 0.337):
            fn.__globals__['scale'] = ref_fn.__globals__['scale'] = scale
            self.compare(compiled(x, y), reference(tx, ty))
        with patch.object(frontend, 'lower', side_effect=AssertionError('warm lowering')), \
             patch.object(bridge, '_pointwise_host_plan', side_effect=AssertionError('warm compile')):
            self.compare(self.without_replay(fn, compiled, (x, y)), reference(tx, ty))
        guarded = native.compile(program('def f(x, y):\n return x - y'))
        guarded(x, y)
        guarded_owner = kernel(guarded)
        legacy = legacy_kernel(guarded, (x, y))
        with self.assertRaisesRegex(RuntimeError, 'indexing guard'):
            legacy.run((y, x))
        self.assertIs(kernel(guarded), guarded_owner)
        fn.__globals__['fw'] = object()
        with self.assertRaises(NotImplementedError):
            compiled(x, y)
        fn.__globals__['fw'] = native
        native.compiler.reset()
        self.assertFalse(cache(compiled).graphs)
        self.assertTrue(old_prepared.belongs_to(old))
        self.compare(old_prepared.run((x, y))[0], tx.relu() * 0.713)
        del compiled, old, old_prepared, legacy, guarded, guarded_owner, x, y
        gc.collect()
        self.compare(first, tx.relu() * 0.713)

    def test_dead_expressions_unused_inputs_repetition_and_zero_tensor_shape(self):
        for source in ('def f(x, y):\n dead = x + y\n return -x',
                       'def f(x, y):\n dead = y.sin()\n return -x',
                       'def f(x, y):\n dead = x.sin()\n return -y',
                       'def f(x, y):\n return x + y * 0',
                       'def f(x, y):\n return (x + y) - (x + y)',
                       'def f(x, y):\n return -(x * y)'):
            compiled = native.compile(program(source))
            reference = self.torch.compile(program(source, self.torch))
            for shapes in [((3, 1), (2, 1, 5)), ((), (0, 7)), ((0, 7), ())]:
                args = [self.upload([0.] * math.prod(s), s) for s in shapes]
                refs = [self.upload([0.] * math.prod(s), s, self.torch) for s in shapes]
                with self.subTest(source=source, shapes=shapes):
                    if source.endswith(('x + y * 0', '(x + y) - (x + y)', '-(x * y)')):
                        with self.assertRaisesRegex(NotImplementedError, 'one arithmetic stage'):
                            compiled(*args)
                    else:
                        self.compare(compiled(*args), reference(*refs), exact=True)
        fn = program('def f(x, y):\n return x * 1.137 - y * 1.137')
        x = self.upload([1., -0., 3.], (3, 1))
        tx = self.upload([1., -0., 3.], (3, 1), self.torch)
        self.compare(native.compile(fn)(x, x), self.torch.compile(program('def f(x, y):\n return x * 1.137 - y * 1.137', self.torch))(tx, tx), exact=True)

    def test_unused_empty_offset_first_input_and_broadcast_recompile_limit(self):
        fn = program('def f(x, y):\n dead = x.sin()\n return -y')
        empty = self.upload([1., 2., 3.], (3,))[3:3]
        scalar = self.upload([2.], ())
        compiled = native.compile(fn)
        self.assertEqual(compiled(empty, scalar).cpu().item(), -2.)
        self.assertEqual(compiled(empty, scalar).cpu().item(), -2.)
        limited = native.compile(program('def f(x, y):\n return x + y'), recompile_limit=1)
        x = self.upload([1., 2., 3.], (3, 1))
        y = self.upload([4., 5.], (2,))
        limited(x, y)
        with self.assertRaisesRegex(NotImplementedError, 'recompile_limit'):
            limited(y, x)
        self.assertEqual(len(cache(limited).graphs), 1)
        limited(x, y)

    def test_ieee_broadcast_contraction(self):
        values = [0., -0., float('inf'), -float('inf'), float('nan'), 2e38, -2e38, 1e-38, -1e-38, 1.0000001192092896]
        for expression in ('x + y', 'x - y', 'y - x', 'x * y', '-x', '-y',
                           '(x * y).relu()', 'x * 2.0 - y * 2.0',
                           'y * 2.0 - x * 2.0', '-(x * y)',
                           'x.sin() - y.cos()', '(x + y) - (x + y)'):
            source = 'def f(x, y):\n return ' + expression
            compiled, reference = native.compile(program(source)), self.torch.compile(program(source, self.torch))
            for shapes in [((len(values), 1), (1, len(values))), ((1, len(values)), (len(values), 1))]:
                args = [self.upload(values, s) for s in shapes]
                refs = [self.upload(values, s, self.torch) for s in shapes]
                with self.subTest(expression=expression, shapes=shapes):
                    if expression in ('x * 2.0 - y * 2.0', 'y * 2.0 - x * 2.0',
                                      '-(x * y)', '(x + y) - (x + y)'):
                        with self.assertRaisesRegex(NotImplementedError, 'one arithmetic stage'):
                            compiled(*args)
                    else:
                        self.compare(compiled(*args), reference(*refs))

    def test_failures_before_nvrtc_and_retry(self):
        x = self.upload([1., 2., 3.], (3,))
        bad = self.upload([1., 2.], (2,))
        for source in ('def f(x, y):\n return x + y', 'def f(x, y):\n dead = x + y\n return -x',
                       'def f(x, y):\n return (x + y).sum()'):
            compiled = native.compile(program(source))
            with patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/broadcast-test'):
                with self.assertRaises(NotImplementedError):
                    compiled(x, bad)
            self.assertFalse(cache(compiled).graphs)
        compiled = native.compile(program('def f(x, y):\n return x + y'))
        y = self.upload([1.], ())
        with patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/broadcast-test'):
            with self.assertRaisesRegex(RuntimeError, 'NVRTC'):
                compiled(x, y)
        self.assertFalse(cache(compiled).graphs)
        self.assertEqual(compiled(x, y).cpu().tolist(), [2., 3., 4.])
        for invalid in [native.ones(1),
                        native.tensor([1.], requires_grad=True)]:
            with self.assertRaises(NotImplementedError):
                compiled(x, invalid)

    @unittest.skipUnless(two_device_reservation(), 'requires explicit reservation of two CUDA devices')
    def test_device_restoration_on_compile_run_failure_and_module_release(self):
        torch = self.torch
        compiled = native.compile(program('def f(x, y):\n return x.relu() - y'))
        x, y = self.upload([1., 2., 3.], (3, 1)), self.upload([0.1, 0.2], (2,))
        with torch.cuda.device(1):
            result = compiled(x, y)
            self.assertEqual(torch.cuda.current_device(), 1)
            self.assertEqual(tuple(compiled(x, y).shape), (3, 2))
            self.assertEqual(torch.cuda.current_device(), 1)
            with self.assertRaises(NotImplementedError):
                compiled(x, y.to('cuda:1'))
            native.compiler.reset()
            gc.collect()
            self.assertEqual(torch.cuda.current_device(), 1)
            with patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/broadcast-test'):
                with self.assertRaisesRegex(RuntimeError, 'NVRTC'):
                    compiled(x, y)
            self.assertEqual(torch.cuda.current_device(), 1)
        self.assertEqual(tuple(result.shape), (3, 2))


del Hardware

if __name__ == '__main__':
    unittest.main()
