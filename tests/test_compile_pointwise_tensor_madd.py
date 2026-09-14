"""Original tensor-leaf multiply-add admission and default-Inductor comparisons."""
import gc
import itertools
import math
import os
import struct
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend
from torch_rs import torch_rs as bridge
from test_compile_pointwise_jit import (
    Hardware, available, cache, lower, program, two_device_reservation,
)


# All ordered tensor ID triples in the existing two-tensor ABI, both add orders.
EXPRESSIONS = tuple(expression for a, b, c in itertools.product(('x', 'y'), repeat=3)
                    for expression in (f'{a}*{b}+{c}', f'{c}+{a}*{b}'))
NEAR_MISSES = (
    'x*y+x*y', 'x*x+y*y', 'x*y+(x+1.)*(x+1.)',
    '(x*y+x).relu()', '(x*y+x)+0', '(x*y+x)*1', '(x*y+x)*True',
    '(x*1)*y+x', 'x*(y+0)+x', 'x*y+(x*1)', 'x*y+x.relu()',
    'x.relu()*y+x', 'x*y-x', 'x-x*y', '-(x*y)+x', '(-x)*y+x',
    'x*(-y)+x', 'x*y+(-x)', '-(x*y+x)', '(x*y+x).sin()',
    'x.sin()*y+x', 'x*y+x.cos()', 'x*y+1.', 'x*y+1', 'x*y+True',
    'x*1.+y', 'x*1+y', 'x*True+y', 'x*y+0', 'x*y+False',
)


def dispatched(compiled):
    """Successful execution moves the dispatched module to the executor LRU end."""
    return next(reversed(cache(compiled).executors.values()))


def snapshot(compiled):
    state = cache(compiled)
    return (tuple(state.graphs.items()), tuple(state.executors.items()),
            tuple((id(entry), tuple(entry.lowerings.items()), dict(entry.observations))
                  for entry in state.graphs.values()))


class Metadata(unittest.TestCase):
    def source(self, body, shapes):
        graph = lower(program('def f(x,y):\n '+body), 2)
        return bridge._pointwise_source(graph.nodes, graph.outputs, 2, shapes)

    def test_all_ordered_tensor_leaves_single_fma_and_unused_arguments(self):
        for expression in EXPRESSIONS:
            for shapes in (((3,1),(2,1,5)), ((2,1,5),(3,1)), ((),(1,)),
                           ((0,1),(1,3)), ((7,),(1,7)), ((1,1),(1,))):
                with self.subTest(expression=expression, shapes=shapes):
                    source = self.source('return '+expression, shapes)
                    self.assertEqual(source.count('fmaf('), 1)
                    self.assertEqual(source.count('__global__'), 1)
                    self.assertNotIn('/ 0ull', source)
                    self.assertNotIn('% 0ull', source)
        source = self.source('dead=y.sin()*y+y\n return x*x+x', ((3,), (5,)))
        self.assertIn('x0[i]', source)
        self.assertNotIn('= x1[', source)
        self.assertNotIn('sinf(', source)

    def test_near_misses_remain_rejected_before_rewriting(self):
        for expression in NEAR_MISSES:
            for shapes in (((2,1),(1,3)), ((2,),(1,2)), ((),(1,)), ((0,),(1,0))):
                with self.subTest(expression=expression, shapes=shapes):
                    with self.assertRaisesRegex(RuntimeError, 'unequal input shapes'):
                        self.source('dead=x*y+x\n return '+expression, shapes)
            self.source('return '+expression, ((3,), (3,)))

    def test_runtime_scalar_leaves_and_all_dead_validation(self):
        for index in (0, 1, 2):
            for sign in (0, 1):
                nodes = [('input', 0, 0, 0), ('input', 1, 0, 0), ('input', 0, 0, 0),
                         ('mul', 0, 1, 0), ('add', 3, 2, 0)]
                nodes[index] = ('scalar', 0, sign, 0)
                with self.assertRaisesRegex(RuntimeError, 'unequal input shapes'):
                    bridge._pointwise_source(tuple(nodes), (4,), 2, ((2,), (1,2)))
        for shapes in (((3,),(5,)), ((1 << 32,1),(1,1 << 32)), ((0,),(2,))):
            with self.subTest(shapes=shapes), self.assertRaisesRegex(RuntimeError, 'broadcast'):
                self.source('dead=x+y\n return x*x+x', shapes)
        for shape in ((2**64-1, 2), (2**63-1,), (0, 2**64-1, 2)):
            with self.subTest(shape=shape), self.assertRaises(RuntimeError):
                self.source('return x*x+x', ((3,),shape))
        nodes = (('input',0,0,0), ('mul',0,0,0), ('add',1,0,0), ('add',0,99,0))
        with self.assertRaisesRegex(ValueError, 'earlier SSA node'):
            bridge._pointwise_source(nodes, (2,), 2, ((2,), (1,2)))


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class TensorMadd(unittest.TestCase):
    setUpClass = classmethod(Hardware.setUpClass.__func__)
    tearDown = Hardware.tearDown
    upload = Hardware.upload
    compare = Hardware.compare
    without_replay = Hardware.without_replay

    def inputs(self, shapes, changed=0, device=0):
        values = (0., -0., 1e-45, -1e-45, 1e-38, -1e-38,
                  2e38, -2e38, float('inf'), -float('inf'), float('nan'),
                  4096., -4098., 1.0000001192092896, -1.137, .713)
        args, refs, backing = [], [], []
        for index, shape in enumerate(shapes):
            data = [values[(i+index*7+changed*3) % len(values)] for i in range(math.prod(shape))]
            offset = 1+changed
            storage = [91.]*offset+data+[92.]
            for fw, destination in ((native,args), (self.torch,refs)):
                base = fw.tensor(storage, dtype=fw.float32).to(f'cuda:{device}')
                destination.append(base[offset:offset+len(data)].reshape(shape))
                backing.append((base, storage))
        return args, refs, backing

    def check_history(self, expression, shapes, device=0):
        fn = program('def f(x,y):\n return '+expression)
        reference = self.torch.compile(program('def f(x,y):\n return '+expression, self.torch))
        compiled = native.compile(fn)
        for pair in shapes:
            for changed in (0,1):
                args, refs, backing = self.inputs(pair, changed, device)
                with self.subTest(expression=expression, shapes=pair, changed=changed, device=device):
                    expected = reference(*refs)
                    caller_device = self.torch.cuda.current_device()
                    actual = self.without_replay(fn, compiled, args)
                    self.assertEqual(self.torch.cuda.current_device(), caller_device)
                    self.compare(actual, expected, exact=True)
                    selected = dispatched(compiled)
                    self.assertEqual(selected.source.count('fmaf('), 1)
                    self.assertEqual(selected.ptx.count('.visible .entry'), 1)
                    self.assertEqual(selected.ptx.count('fma.rn.f32'), 1)
                    with patch.object(frontend, 'analyze', side_effect=AssertionError('warm analysis')), \
                         patch.object(frontend, 'lower', side_effect=AssertionError('warm lowering')), \
                         patch.object(bridge, '_pointwise_compile', side_effect=AssertionError('warm compile')):
                        caller_device = self.torch.cuda.current_device()
                        again = self.without_replay(fn, compiled, args)
                        self.assertEqual(self.torch.cuda.current_device(), caller_device)
                    self.assertIs(dispatched(compiled), selected)
                    self.compare(again, expected, exact=True)
                    self.assertIsNot(actual, again)
                    if expected.numel():
                        self.assertNotEqual(actual.data_ptr(), again.data_ptr())
                        for arg in args:
                            self.assertNotEqual(actual.data_ptr(), arg.data_ptr())
                    for arg, ref in zip(args, refs):
                        self.compare(arg, ref, exact=True)
                    for base, data in backing:
                        actual_data = base.cpu().tolist()
                        for a, b in zip(actual_data, data):
                            if math.isnan(b):
                                self.assertTrue(math.isnan(a))
                            else:
                                # Storage values are rounded on upload.
                                self.assertEqual(struct.pack('f', a), struct.pack('f', b))
        native.compiler.reset()
        self.torch.compiler.reset()

    def test_all_leaf_ids_and_both_add_orders(self):
        for expression in EXPRESSIONS:
            self.check_history(expression, [((19,1),(1,23)), ((1,23),(19,1)),
                                            ((29,1),(1,31)), ((19,1),(1,23))])

    def test_rank_empty_singleton_and_block_boundary_histories(self):
        histories = [
            [((17,1,7),(1,11,1)), ((1,11,1),(17,1,7)), ((7,1),(3,1,13)), ((3,1,13),(7,1))],
            [((),(17,)), ((17,),()), ((1,),(1,1)), ((0,1),(1,13)), ((1,13),(0,1))],
            [((251,1),(251,263)), ((251,263),(251,1)), ((257,1),(257,269)), ((255,),(1,255)), ((256,),(1,256)), ((513,),(1,513))],
        ]
        for expression in ('x*y+x', 'x+x*y', 'x*x+y', 'y+x*x'):
            for shapes in histories:
                self.check_history(expression, shapes)

    def test_explicit_cancellation_overflow_and_rounding(self):
        # Fused cancellation distinguishes these from separately rounded eager ops.
        xvalues = [3e38, -3e38, 2e38, -2e38, 1.0000001192092896, -0., 0., 1e-45, -1e-45]
        yvalues = [-1.5, -1.5, -1.5, -1.5, -0.9999999403953552, 0., -0., -1., -1.]
        scenarios = [
            (('x*y+x', 'x+x*y'), xvalues, yvalues),
            (('y*x+y', 'y+y*x'), yvalues, xvalues),
            (('x*x+y', 'y+x*x'), [1.9e19,-1.9e19], [-3e38,-3e38]),
        ]
        for expressions, xv, yv in scenarios:
            for expression in expressions:
                fn = program('def f(x,y):\n return '+expression)
                reference = self.torch.compile(program('def f(x,y):\n return '+expression))
                compiled = native.compile(fn)
                size = len(xv)
                for shapes in (((size,), (1,size)), ((size,1),(1,size)), ((1,size),(size,1))):
                    args = [self.upload(v,s) for v,s in zip((xv,yv),shapes)]
                    refs = [self.upload(v,s,self.torch) for v,s in zip((xv,yv),shapes)]
                    expected = reference(*refs)
                    # The first products overflow float32 if rounded separately:
                    # ±3e38 * -1.5, or (±1.9e19)**2. Fused cancellation is finite.
                    self.assertTrue(self.torch.isfinite(expected.flatten()[:2]).all())
                    self.compare(self.without_replay(fn,compiled,args), expected, exact=True)
                native.compiler.reset()
                self.torch.compiler.reset()

    def test_identity_changes_unused_inputs_and_one_tensor(self):
        for expression in ('x*y+x', 'x+x*y', 'x*x+x', 'y+y*y'):
            fn = program('def f(x,y):\n return '+expression)
            reference = self.torch.compile(program('def f(x,y):\n return '+expression))
            compiled = native.compile(fn)
            for alias in (False, True, False, True):
                args, refs, _ = self.inputs(((17,), (17,)), int(alias))
                if alias:
                    args[1], refs[1] = args[0], refs[0]
                self.compare(self.without_replay(fn,compiled,args), reference(*refs), exact=True)
            native.compiler.reset()
            self.torch.compiler.reset()
        self.check_history('x*x+x', [((7,), (11,)), ((7,), (0,13)), ((7,), ())])
        fn = program('def f(x):\n return x*x+x')
        x = self.upload([1.,-0.,1e-40], (3,))
        tx = self.upload([1.,-0.,1e-40], (3,), self.torch)
        self.compare(native.compile(fn)(x), self.torch.compile(fn)(tx), exact=True)

    def test_direct_kernel_admission_address_guard_and_cache_failure_atomicity(self):
        for expression in ('x*y+x', 'x+x*y', *NEAR_MISSES):
            fn = program('def f(x,y):\n return '+expression)
            compiled = native.compile(fn)
            x, y = self.upload([1.,-2.], (2,)), self.upload([3.,4.], (2,))
            equal = compiled(x,y)
            selected = dispatched(compiled)  # Real NVRTC compilation and launch.
            before = snapshot(compiled)
            unequal = y.reshape(1,2)
            if expression in ('x*y+x', 'x+x*y'):
                reference = self.torch.compile(program('def f(x,y):\n return '+expression))
                expected = reference(self.upload([1.,-2.],(2,),self.torch),
                                     self.upload([3.,4.],(1,2),self.torch))
                self.compare(selected.run((x,unequal))[0], expected, exact=True)
                # Admission succeeds, but changed broadcast addressing still fails.
                with self.assertRaisesRegex(RuntimeError, 'indexing guard'):
                    selected.run((x.reshape(2,1),y.reshape(1,2)))
            else:
                with self.assertRaisesRegex(RuntimeError, 'unequal input shapes'):
                    selected.run((x,unequal))
                with self.assertRaisesRegex(NotImplementedError, 'unequal input shapes'):
                    compiled(x,unequal)
            self.assertEqual(snapshot(compiled), before)
            self.assertEqual(selected.run((x,y))[0].cpu().tolist(), equal.cpu().tolist())
            native.compiler.reset()
            self.assertFalse(cache(compiled).graphs)
            self.assertFalse(cache(compiled).executors)
            self.assertEqual(selected.run((x,y))[0].cpu().tolist(), equal.cpu().tolist())
            compiled(x,y)
            self.assertIsNot(dispatched(compiled), selected)
            native.compiler.reset()
            self.torch.compiler.reset()

    def test_unused_invalid_tensor_and_nvrtc_failure_publish_nothing(self):
        dead = native.compile(program('def f(x,y):\n dead=x+y\n return x*x+x'))
        dx, dy = self.upload([1.,2.], (2,)), self.upload([3.,4.], (2,))
        equal = dead(dx,dy)
        selected = dispatched(dead)
        before = snapshot(dead)
        # An unused argument still participates in the existing address ABI.
        with self.assertRaisesRegex(RuntimeError, 'indexing guard'):
            selected.run((dx,dy.reshape(1,2)))
        self.assertEqual(snapshot(dead), before)
        incompatible = self.upload([1.,2.,3.], (3,))
        with self.assertRaisesRegex(RuntimeError, 'incompatible.*broadcast'):
            selected.run((dx,incompatible))
        with self.assertRaisesRegex(NotImplementedError, 'incompatible.*broadcast'):
            dead(dx,incompatible)
        self.assertEqual(snapshot(dead), before)
        self.assertEqual(dead(dx,dy.reshape(1,2)).cpu().tolist(), equal.cpu().tolist())
        compiled = native.compile(program('def f(scale,x,unused):\n return x*x+x'))
        x = self.upload([1.,2.], (2,))
        unused = self.upload([3.], (1,))
        compiled(1.,x,unused)
        before = snapshot(compiled)
        for invalid in (native.ones(1), native.tensor([1.], requires_grad=True), self.upload([1.,2.,3.,4.],(2,2)).t()):
            with self.assertRaises(NotImplementedError):
                compiled(2.,x,invalid)
            self.assertEqual(snapshot(compiled), before)
        with patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/tensor-madd-test'):
            with self.assertRaisesRegex(RuntimeError, 'NVRTC'):
                compiled(2.,x.reshape(1,2),unused)
        self.assertEqual(snapshot(compiled), before)
        compiled(2.,x.reshape(1,2),unused)

    @unittest.skipUnless(two_device_reservation(), 'requires explicit reservation of two CUDA devices')
    def test_both_devices_restore_current_device_on_success_failure_and_reset(self):
        if self.torch.cuda.device_count() < 2:
            self.skipTest('requires two CUDA devices')
        for device in (0,1):
            with self.torch.cuda.device(1-device):
                self.check_history('x*y+x', [((7,1),(1,13)), ((1,13),(7,1))], device)
                self.assertEqual(self.torch.cuda.current_device(), 1-device)
                args, _, _ = self.inputs(((7,1),(1,13)), device=device)
                compiled = native.compile(program('def f(x,y):\n return x+y*x'))
                with patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/tensor-madd-device'):
                    with self.assertRaisesRegex(RuntimeError, 'NVRTC'):
                        compiled(*args)
                self.assertFalse(cache(compiled).graphs)
                self.assertEqual(self.torch.cuda.current_device(), 1-device)
                compiled(*args)
                selected = dispatched(compiled)
                with self.assertRaisesRegex(RuntimeError, 'same-device'):
                    selected.run((args[0],native.tensor([1.]).to(f'cuda:{1-device}')))
                native.compiler.reset()
                del selected, compiled
                gc.collect()
                self.assertEqual(self.torch.cuda.current_device(), 1-device)


del Hardware

if __name__ == '__main__':
    unittest.main()
