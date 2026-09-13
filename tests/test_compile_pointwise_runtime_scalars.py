"""Changed captured floats retain default Inductor's runtime rounding boundary."""
import struct
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import torch_rs as native
from torch_rs import _compile_pointwise as frontend
from torch_rs import torch_rs as bridge
from tests.test_compile_pointwise_jit import available, cache, kernels, program, two_device_reservation
from tests import test_compile_pointwise_jit as jit_tests


class RuntimeScalarAdmission(unittest.TestCase):
    def test_promotion_uses_successful_numeric_history_and_preserves_scalar_kind(self):
        fn = program('def f(x):\n return x + scale', scale=16777216.0)
        parsed = frontend.analyze(fn, 1)
        keys, values = frontend.resolve(fn, parsed)
        history = {(parsed.code, keys, (), (0,)): None}
        fn.__globals__['scale'] = 16777217.0
        changed, values = frontend.resolve(fn, parsed)
        dynamic, values, scalars = frontend.runtime_bindings(parsed, changed, values, history)
        self.assertEqual(scalars, (16777217.0,))
        graph = frontend.lower(parsed, values, 1)
        source = bridge._pointwise_source(graph.nodes, graph.output, 1)
        self.assertIn('float s0', source)
        self.assertIn('= s0;', source)
        history[(parsed.code, dynamic, (), (0,))] = None
        fn.__globals__['scale'] = 16777216.0
        keys, values = frontend.resolve(fn, parsed)
        self.assertEqual(frontend.runtime_bindings(parsed, keys, values, history)[0], dynamic)
        for scalar in (0, False):
            fn.__globals__['scale'] = scalar
            keys, values = frontend.resolve(fn, parsed)
            self.assertEqual(frontend.runtime_bindings(parsed, keys, values, history)[2], ())
        keys = (('tensor', 0), (float, struct.pack('=d', -0.0)))
        history = {(parsed.code, keys, (), (0,)): None}
        fn.__globals__['scale'] = 0.0
        keys, values = frontend.resolve(fn, parsed)
        self.assertEqual(frontend.runtime_bindings(parsed, keys, values, history)[2], ())

    def test_runtime_scalar_arity_and_outputs_are_validated_without_hardware(self):
        for nodes, output in [
            ((('input', 0, 0, 0), ('scalar', 64, 0, 0), ('add', 0, 1, 0)), 2),
            ((('input', 0, 0, 0), ('scalar', 0, 0, 0)), 1),
        ]:
            with self.assertRaises(ValueError):
                bridge._pointwise_source(nodes, output, 1)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class RuntimeScalarHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def pair(self, body, closure=False):
        if closure:
            def factory():
                scale = 16777217.0
                def f(x):
                    return ((x * 0) + scale) - 16777216.0
                def set_value(value):
                    nonlocal scale
                    scale = value
                return f, set_value
            fn, setter = factory()
            ref_fn, ref_setter = factory()
        else:
            fn, ref_fn = program(body, scale=16777217.0), program(body, scale=16777217.0)
            setter = lambda value: fn.__globals__.__setitem__('scale', value)
            ref_setter = lambda value: ref_fn.__globals__.__setitem__('scale', value)
        return native.compile(fn), self.torch.compile(ref_fn), setter, ref_setter

    def compare(self, compiled, reference, data):
        x, tx = native.tensor(data).to('cuda:0'), self.torch.tensor(data, device='cuda:0')
        result, expected = compiled(x), reference(tx).cpu()
        actual = self.torch.tensor(result.cpu().tolist())
        self.torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        mask = ~expected.isnan()
        self.assertTrue(self.torch.equal(actual.view(self.torch.int32)[mask],
                                        expected.view(self.torch.int32)[mask]))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.stride(), x.stride())
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device, x.device)
        self.assertNotEqual(result.data_ptr(), x.data_ptr())
        self.torch.testing.assert_close(self.torch.tensor(x.cpu().tolist()), tx.cpu(),
                                       rtol=0, atol=0, equal_nan=True)
        return actual

    def test_global_and_closure_transitions_reuse_runtime_code_and_reset_history(self):
        for closure in (False, True):
            with self.subTest(closure=closure):
                compiled, reference, setter, ref_setter = self.pair(
                    'def f(x):\n return ((x*0)+scale)-16777216.0', closure)
                outputs = []
                dynamic_kernel = None
                for value in (16777217.0, 16777216.0, 16777217.0, 16777218.0):
                    setter(value)
                    ref_setter(value)
                    outputs.append(self.compare(compiled, reference, [-0., 2., float('inf')])[0].item())
                    self.compare(compiled, reference, [float('nan'), -4., 0.])
                    entries = kernels(compiled)
                    if len(cache(compiled).graphs) == 2:
                        if dynamic_kernel is not None:
                            self.assertIs(entries[-1], dynamic_kernel)
                        dynamic_kernel = entries[-1]
                self.assertEqual(outputs, [1., 0., 0., 2.])
                entries = kernels(compiled)
                self.assertEqual(len(cache(compiled).graphs), 2)
                self.assertIn('float s0', entries[-1].source)
                native.compiler.reset()
                self.torch.compiler.reset()
                setter(16777217.0)
                ref_setter(16777217.0)
                self.assertEqual(self.compare(compiled, reference, [1., -1., 0.])[0].item(), 1.)
                self.assertEqual(len(cache(compiled).graphs), 1)
                self.assertNotIn('float s0', kernels(compiled)[0].source)
                native.compiler.reset()
                self.torch.compiler.reset()

    def test_runtime_scalar_negation_and_multiple_consumers(self):
        for body in ('(x*0)+(-scale)', 'x*scale + x*(-scale)',
                     '((x*0)+scale).relu() + (x*scale)'):
            compiled, reference, setter, ref_setter = self.pair('def f(x):\n return ' + body)
            for value in (1.137, 2.25, -0.0, 0.0, 1.137):
                setter(value)
                ref_setter(value)
                self.compare(compiled, reference, [-0., 0., 1., -1., 1e-38])
            self.assertEqual(len(cache(compiled).graphs), 2)
            native.compiler.reset()
            self.torch.compiler.reset()

    def test_multiple_captured_scalars_are_independent_runtime_parameters(self):
        source = 'def f(x):\n return (x*left)+((x*0)+right)'
        fn = program(source, left=1.125, right=16777217.0)
        ref_fn = program(source, left=1.125, right=16777217.0)
        compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
        dynamic_kernel = None
        for left, right in ((1.125, 16777217.0), (2.25, 16777216.0),
                            (-3.5, 16777218.0), (1.125, 16777217.0)):
            fn.__globals__.update(left=left, right=right)
            ref_fn.__globals__.update(left=left, right=right)
            self.compare(compiled, reference, [-0., 0., 1., -1., 1e10])
            entries = kernels(compiled)
            if len(cache(compiled).graphs) == 2:
                if dynamic_kernel is not None:
                    self.assertIs(entries[-1], dynamic_kernel)
                dynamic_kernel = entries[-1]
        self.assertEqual(len(cache(compiled).graphs), 2)
        self.assertIn('float s0, float s1', dynamic_kernel.source)

    def test_promoted_float_nonfinite_type_and_shape_transitions(self):
        fn = program('def f(x):\n return ((x*0)+scale)-16777216.0', scale=16777217.0)
        ref_fn = program('def f(x):\n return ((x*0)+scale)-16777216.0', scale=16777217.0)
        compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
        for value in (16777217.0, 16777218.0, float('inf'), float('-inf'),
                      float('nan'), 16777217, 16777217.0):
            fn.__globals__['scale'] = ref_fn.__globals__['scale'] = value
            self.compare(compiled, reference, [-0., 2., float('inf')])
        # The promoted parameter and its history remain usable on empty and
        # scalar-shaped tensors. New metadata must not restore static folding.
        for data in ([], 1.0):
            x, tx = native.tensor(data).to('cuda:0'), self.torch.tensor(data, device='cuda:0')
            actual, expected = compiled(x).cpu().tolist(), reference(tx).cpu()
            self.torch.testing.assert_close(self.torch.tensor(actual), expected, rtol=0, atol=0)

    def test_runtime_launch_values_are_not_shared_between_threads(self):
        fn = program('def f(x):\n return x*scale', scale=1.0)
        compiled = native.compile(fn)
        x = native.tensor([1., -2.]).to('cuda:0')
        compiled(x)
        fn.__globals__['scale'] = 2.0
        compiled(x)
        kernel = kernels(compiled)[-1]
        values = (1.25, -3.5, 8.0, 0.0) * 4
        def run(value):
            result = kernel.run((x,), (value,))
            return result.cpu().tolist()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, values))
        self.assertEqual(results, [[v, -2*v] for v in values])
        empty = native.tensor([]).to('cuda:0')
        with self.assertRaisesRegex(RuntimeError, 'scalar arity'):
            kernel.run((empty,), ())

    @unittest.skipUnless(two_device_reservation(),
                         'requires explicit two-device validation')
    def test_runtime_parameters_restore_current_device(self):
        if self.torch.cuda.device_count() < 2:
            self.skipTest('requires two CUDA devices')
        fn = program('def f(x):\n return ((x*0)+scale)-16777216.0', scale=16777217.0)
        compiled = native.compile(fn)
        for target in (0, 1):
            x = native.tensor([1.]).to(f'cuda:{target}')
            with self.torch.cuda.device(1-target):
                for scale in (16777217.0, 16777218.0, 16777217.0):
                    fn.__globals__['scale'] = scale
                    compiled(x)
                    self.assertEqual(self.torch.cuda.current_device(), 1-target)
        entries = kernels(compiled)
        kernel = next(entry for entry in entries if entry.device == 0 and 'float s0' in entry.source)
        with self.torch.cuda.device(1):
            with self.assertRaisesRegex(RuntimeError, 'device guard'):
                kernel.run((x,), (1.0,))
            native.compiler.reset()
            self.assertEqual(self.torch.cuda.current_device(), 1)

    def test_failed_compile_and_invalid_runtime_arguments_do_not_advance_history(self):
        fn = program('def f(x):\n return ((x*0)+scale)-16777216.0', scale=16777216.0)
        compiled = native.compile(fn)
        x = native.tensor([1.]).to('cuda:0')
        compiled(x)
        fn.__globals__['scale'] = 16777217.0
        with mock.patch.object(bridge, '_pointwise_compile', side_effect=RuntimeError('injected compilation failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected compilation failure'):
                compiled(x)
        self.assertEqual(len(cache(compiled).graphs), 1)
        compiled(x)
        kernel = kernels(compiled)[-1]
        callbacks = []
        class CustomFloat(float):
            def __float__(self):
                callbacks.append('float')
                return 1.0
        for values in ((), (1.0, 2.0), (CustomFloat(1.),), (1,)):
            with self.assertRaises((TypeError, RuntimeError)):
                kernel.run((x,), values)
        fn.__globals__['scale'] = CustomFloat(1.)
        with self.assertRaises(NotImplementedError):
            compiled(x)
        self.assertEqual(callbacks, [])
        self.assertEqual(len(cache(compiled).graphs), 2)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class PositionalScalarHardware(unittest.TestCase):
    setUpClass = classmethod(jit_tests.Hardware.setUpClass.__func__)
    tearDown = jit_tests.Hardware.tearDown
    upload = jit_tests.Hardware.upload
    compare = jit_tests.Hardware.compare
    without_replay = jit_tests.Hardware.without_replay

    def check(self, fn, compiled, reference, args, refs, exact=True):
        expected = reference(*refs)
        actual = self.without_replay(fn, compiled, args)
        self.compare(actual, expected, exact=exact)
        for x, tx in zip(args, refs):
            if type(x) is native.Tensor:
                self.compare(x, tx, exact=True)
                self.assertIsNot(actual, x)
                if x.numel() and actual.numel():
                    self.assertNotEqual(actual.data_ptr(), x.data_ptr())
        return actual

    def test_arbitrary_slots_one_two_tensors_mixed_captures_and_repeated_bindings(self):
        import itertools
        for names in (('x', 's', 'b'), ('x', 's', 'y', 'b')):
            for order in itertools.permutations(names):
                source = 'def f('+','.join(order)+'):\n return (x*s)+(y*b)+gain' if 'y' in names else (
                    'def f('+','.join(order)+'):\n a=x*s\n return a+a*b+gain')
                fn, rf = program(source, gain=0.3125), program(source, gain=0.3125)
                compiled, reference = native.compile(fn), self.torch.compile(rf)
                for s, b, gain in ((0.375, False, 0.3125), (-1.125, True, -0.625), (0.375, False, 0.3125)):
                    fn.__globals__['gain'] = rf.__globals__['gain'] = gain
                    data = {'s': s, 'b': b}
                    refs = dict(data)
                    for name, values in [('x', [-3., -0., 0., 1.25, 7.]), ('y', [2., 0., -0., -4., 0.25])]:
                        data[name] = self.upload(values, (5,))
                        refs[name] = self.upload(values, (5,), self.torch)
                    with self.subTest(order=order, s=s, b=b):
                        self.check(fn, compiled, reference, tuple(data[n] for n in order),
                                   tuple(refs[n] for n in order))
                native.compiler.reset()
                self.torch.compiler.reset()

    def test_float_boolean_nonfinite_overflow_histories_are_persistent(self):
        histories = (
            (16777217., 16777218., 16777217., float('inf'), float('nan'), -0., 0., 1e300),
            (-0., 0., -0., 1.25, 0., -0.), (0., -0., 0., -1.25, -0., 0.),
            (False, True, 0., False, 1., True),
            (float('inf'), 16777217., float('nan'), 16777217., 16777218.),
            (float('-inf'), 16777217., float('inf'), 16777217., -1e300),
        )
        for body in ('((x*0)+s)-16777216.0', 'x*s', '(x*s)+x*(-s)'):
            for history in histories:
                source = 'def f(s,x):\n return '+body
                fn, rf = program(source), program(source)
                compiled, reference = native.compile(fn), self.torch.compile(rf)
                for s in history:
                    for values in ([0., -0., 1., -1., 3e38, 1e-38],
                                   [float('inf'), float('-inf'), float('nan'), 2., -2., -1e-38]):
                        x, tx = self.upload(values, (6,)), self.upload(values, (6,), self.torch)
                        with self.subTest(body=body, history=history, s=s, values=values):
                            self.check(fn, compiled, reference, (s, x), (s, tx))
                # Reset both histories explicitly; neither is reset to mask a mismatch.
                native.compiler.reset()
                self.torch.compiler.reset()

    def test_shape_offset_identity_and_returned_scalar_value_history(self):
        source = 'def f(s,x,y):\n return (x*s)-(y*s)'
        fn, rf = program(source), program(source)
        compiled, reference = native.compile(fn), self.torch.compile(rf)
        for shape, offset, same, scalar in (((4,), 0, True, 16777217.),
                                            ((4,), 1, True, 16777217.),
                                            ((4,), 0, False, 16777218.),
                                            ((2, 2), 0, False, 16777217.),
                                            ((4,), 1, False, -0.),
                                            ((0,), 0, False, 1.25)):
            size = 0 if shape == (0,) else 4
            values = [1., -1., 0., -0., 0.125]
            x = self.upload(values[:size+offset], (size+offset,))[offset:].reshape(shape)
            tx = self.upload(values[:size+offset], (size+offset,), self.torch)[offset:].reshape(shape)
            y, ty = (x, tx) if same else (x[:], tx[:])
            with self.subTest(shape=shape, offset=offset, same=same, scalar=scalar):
                self.check(fn, compiled, reference, (scalar, x, y), (scalar, tx, ty))
        self.assertEqual({key[3] for key in cache(compiled).graphs}, {(0, 0), (0, 1)})

    def test_unused_and_overwritten_scalars_have_no_value_guards(self):
        for body in ('return x.sin()', 's=0.375\n return x*s'):
            source = 'def f(s,x):\n '+body
            fn, rf = program(source), program(source)
            compiled, reference = native.compile(fn, recompile_limit=1), self.torch.compile(rf)
            x, tx = self.upload([1., -1.], (2,)), self.upload([1., -1.], (2,), self.torch)
            for value in (0.375, False, 1.125, True, float('inf'), float('nan'), -0., 0.):
                self.check(fn, compiled, reference, (value, x), (value, tx), exact=False)
            self.assertEqual(len(cache(compiled).graphs), 1)
            from torch._dynamo.eval_frame import _debug_get_cache_entry_list
            self.assertEqual(len(_debug_get_cache_entry_list(rf)), 1)
            # Type admission still covers every public slot, even if unused.
            with self.assertRaises(NotImplementedError):
                compiled(1, x)

    def test_scalar_read_before_reassignment_keeps_its_source_history(self):
        source = 'def f(s,x,t):\n a=x*s\n s=t\n return a+x*s'
        fn, rf = program(source), program(source)
        compiled, reference = native.compile(fn), self.torch.compile(rf)
        x, tx = self.upload([1., -1., 0., -0.], (4,)), self.upload([1., -1., 0., -0.], (4,), self.torch)
        for s, t in ((16777217., 0.375), (16777218., 1.125), (16777217., 0.375)):
            self.check(fn, compiled, reference, (s, x, t), (s, tx, t))
        self.assertEqual(len(cache(compiled).graphs), 2)
        self.assertIn('float s1', kernels(compiled)[-1].source)

    def test_tensor_scalar_role_changes_keep_history_attached_to_public_slots(self):
        source = 'def f(a,b,c):\n return a*b+c'
        fn, rf = program(source), program(source)
        compiled, reference = native.compile(fn), self.torch.compile(rf)
        x, tx = self.upload([1., -1., 0., -0.], (4,)), self.upload([1., -1., 0., -0.], (4,), self.torch)
        for args, refs in (((x, 0.375, 1.125), (tx, 0.375, 1.125)),
                           ((0.375, x, 1.125), (0.375, tx, 1.125)),
                           ((0.75, x, 1.5), (0.75, tx, 1.5)),
                           ((x, 0.75, 1.5), (tx, 0.75, 1.5)),
                           ((x, x, 1.5), (tx, tx, 1.5)),
                           ((16777217., x, 16777218.), (16777217., tx, 16777218.)),
                           ((x, 16777217., 16777218.), (tx, 16777217., 16777218.))):
            self.check(fn, compiled, reference, args, refs)

    def test_distinct_equal_scalar_sources_and_closure_capture(self):
        def factory():
            gain = 16777217.
            def f(a, x, b):
                return ((x*0)+gain) - a + x*b - x*a
            def set_gain(value):
                nonlocal gain
                gain = value
            return f, set_gain
        fn, setter = factory()
        rf, ref_setter = factory()
        compiled, reference = native.compile(fn), self.torch.compile(rf)
        x, tx = self.upload([1., -1., 0., -0.], (4,)), self.upload([1., -1., 0., -0.], (4,), self.torch)
        for a, b, gain in ((16777217., 16777217., 16777217.),
                            (16777218., 16777217., 16777218.),
                            (16777217., 16777218., 16777217.),
                            (16777217., 16777217., 16777217.)):
            setter(gain)
            ref_setter(gain)
            self.check(fn, compiled, reference, (a, x, b), (a, tx, b))
        self.assertIn('float s2', kernels(compiled)[-1].source)

    def test_failed_calls_recompile_limit_reset_and_warm_no_replay(self):
        source = 'def f(s,x):\n return ((x*0)+s)-16777216.0'
        fn, rf = program(source), program(source)
        compiled, reference = native.compile(fn, recompile_limit=2), self.torch.compile(rf)
        x, tx = self.upload([1., -1.], (2,)), self.upload([1., -1.], (2,), self.torch)
        self.check(fn, compiled, reference, (16777217., x), (16777217., tx))
        before = dict(cache(compiled).graphs)
        with mock.patch.object(bridge, '_pointwise_compile', side_effect=RuntimeError('injected compile failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected compile failure'):
                compiled(16777218., x)
        self.assertEqual(cache(compiled).graphs, before)
        class FailedLaunch:
            def run(self, *args):
                raise RuntimeError('injected launch failure')
        with mock.patch.object(bridge, '_pointwise_compile', return_value=FailedLaunch()):
            with self.assertRaisesRegex(RuntimeError, 'injected launch failure'):
                compiled(16777218., x)
        self.assertEqual(cache(compiled).graphs, before)
        from tests.test_compile_pointwise_scalar_admission import callback_scalar
        effects = []
        for rejected in (1, 0, 2**64, object(), None, callback_scalar(effects)):
            with self.assertRaises(NotImplementedError):
                compiled(rejected, x)
            self.assertEqual(cache(compiled).graphs, before)
            self.assertEqual(effects, [])
        self.check(fn, compiled, reference, (16777218., x), (16777218., tx))
        with mock.patch.object(frontend, 'lower', side_effect=AssertionError('warm lowering')), \
             mock.patch.object(frontend, 'analyze', side_effect=AssertionError('warm analysis')), \
             mock.patch.object(bridge, '_pointwise_compile', side_effect=AssertionError('warm compilation')):
            self.check(fn, compiled, reference, (16777217., x), (16777217., tx))
        for args in ((True, x), (16777217., x[:1])):
            with self.assertRaisesRegex(NotImplementedError, 'recompile_limit=2'):
                compiled(*args)
        self.assertEqual(len(cache(compiled).graphs), 2)
        native.compiler.reset()
        self.torch.compiler.reset()
        result = self.check(fn, compiled, reference, (16777217., x), (16777217., tx))
        self.assertEqual(result.cpu().tolist(), [1., 1.])
        self.assertEqual(len(cache(compiled).graphs), 1)

    def test_combined_capture_parameter_scalar_abi_at_limit(self):
        names = [f'a{i}' for i in range(32)]
        captured = {f'g{i}': 0.125 for i in range(32)}
        source = 'def f(x,'+','.join(names)+'):\n return x+'+'+'.join(names+list(captured))
        fn, rf = program(source, **captured), program(source, **captured)
        compiled, reference = native.compile(fn), self.torch.compile(rf)
        x, tx = self.upload([1., -1.], (2,)), self.upload([1., -1.], (2,), self.torch)
        for scalar in (0.125, 0.25, 0.125):
            for name in captured:
                fn.__globals__[name] = rf.__globals__[name] = scalar
            self.check(fn, compiled, reference, (x,)+(scalar,)*32, (tx,)+(scalar,)*32)
        self.assertEqual(len(cache(compiled).graphs), 2)
        self.assertIn('float s63', kernels(compiled)[-1].source)

    def test_broadcasts_with_scalar_slots_use_only_tensor_shapes(self):
        for body in ('(x+y).relu()', '(x*s).relu()'):
            source = 'def f(s,x,flag,y):\n return '+body
            fn, rf = program(source), program(source)
            compiled, reference = native.compile(fn), self.torch.compile(rf)
            for rows, columns, scalar in ((7, 11, 0.375), (7, 11, -1.125), (3, 17, 0.375)):
                values = [i*0.25-1. for i in range(rows)]
                other = [i*0.125-0.75 for i in range(columns)]
                x, tx = self.upload(values, (rows, 1)), self.upload(values, (rows, 1), self.torch)
                y, ty = self.upload(other, (1, columns)), self.upload(other, (1, columns), self.torch)
                self.check(fn, compiled, reference, (scalar, x, False, y), (scalar, tx, False, ty))

    def test_positional_execution_without_installed_pytorch_or_original_body(self):
        import subprocess
        import sys
        script = '''
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('installed PyTorch production execution')
sys.meta_path.insert(0, Block())
import torch_rs as m
def f(s,x,flag,y):
    return x*s+y*flag
compiled=m.compile(f)
def forbid(frame,event,arg):
    if event == 'call' and frame.f_code is f.__code__:
        raise AssertionError('original body replay')
sys.setprofile(forbid)
for s,flag in ((0.375,False),(-1.125,True),(0.375,False)):
    x=m.tensor([1.,-2.]).to('cuda:0')
    y=m.tensor([3.,4.]).to('cuda:0')
    result=compiled(s,x,flag,y)
    assert result.cpu().tolist()==[s+3*flag,-2*s+4*flag]
    assert result.data_ptr()!=x.data_ptr()
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', script],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_unused_tensors_cannot_bypass_broadcast_boundary_and_one_tensor_is_unchanged(self):
        for body in ('x.sin()+s', 'x*s+s', '(x*False)+s'):
            fn = program('def f(s,x,unused,flag):\n return '+body)
            compiled = native.compile(fn)
            x = self.upload([0., -0., 1., -1.], (4,))
            for y in (x.reshape(1, 4), self.upload([2.], ())):
                with mock.patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/positional-boundary'):
                    with self.assertRaisesRegex(NotImplementedError, 'unequal input shapes'):
                        self.without_replay(fn, compiled, (0.375, x, y, False))
                self.assertFalse(cache(compiled).graphs)
            one_source = 'def f(s,x,flag):\n return '+body
            fn, rf = program(one_source), program(one_source)
            tx = self.upload([0., -0., 1., -1.], (4,), self.torch)
            self.check(fn, native.compile(fn), self.torch.compile(rf), (0.375, x, False), (0.375, tx, False), exact=False)

    @unittest.skipUnless(two_device_reservation(), 'requires explicit two-device reservation')
    def test_positional_runtime_parameters_restore_device_on_success_and_failure(self):
        if self.torch.cuda.device_count() < 2:
            self.skipTest('requires two CUDA devices')
        fn = program('def f(s,x):\n return x*s')
        compiled = native.compile(fn)
        x = native.tensor([1., -1.]).to('cuda:0')
        with self.torch.cuda.device(1):
            for value in (0.375, 0.75, -0.):
                self.without_replay(fn, compiled, (value, x))
                self.assertEqual(self.torch.cuda.current_device(), 1)
            with self.assertRaises(NotImplementedError):
                compiled(1, x)
            self.assertEqual(self.torch.cuda.current_device(), 1)
            native.compiler.reset()
            with mock.patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/positional-device'):
                with self.assertRaisesRegex(RuntimeError, 'NVRTC'):
                    compiled(1., x)
            self.assertEqual(self.torch.cuda.current_device(), 1)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class ScalarShapeSpecializationHardware(unittest.TestCase):
    setUpClass = classmethod(jit_tests.Hardware.setUpClass.__func__)
    tearDown = jit_tests.Hardware.tearDown
    upload = jit_tests.Hardware.upload
    without_replay = jit_tests.Hardware.without_replay
    maxDiff = None

    def binding_function(self, origin, tensor_count, broadcast):
        arguments = 'x,y' if tensor_count == 2 else 'x'
        if origin == 'parameter':
            arguments = 'scale,' + arguments
        # Two live tensor operands and a live scalar would require two arithmetic
        # stages, outside the unequal-shape surface. Keep actual broadcasting
        # one-stage; the unused-tensor histories observe the captured zero sign.
        body = 'discarded=x*scale\n return x*y' if broadcast else 'return x*scale'
        source = 'def f(' + arguments + '):\n ' + body
        if origin == 'closure':
            namespace = {}
            exec('def factory():\n scale=0.0\n ' + source.replace('\n', '\n ')
                 + '\n def set_scale(value):\n  nonlocal scale\n  scale=value'
                 + '\n return f,set_scale', namespace)
            return namespace['factory']()
        fn = program(source, scale=0.0)
        setter = None if origin == 'parameter' else lambda value: fn.__globals__.__setitem__('scale', value)
        return fn, setter

    def check_history(self, origin, initial, shapes, broadcast=False):
        # Each history starts clean on both sides. Neither wrapper nor either
        # compiler's history is reset between observations, including failures.
        native.compiler.reset()
        self.torch.compiler.reset()
        fn, setter = self.binding_function(origin, len(shapes[0]), broadcast)
        ref_fn, ref_setter = self.binding_function(origin, len(shapes[0]), broadcast)
        compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
        observations = []
        for step, input_shapes in enumerate(shapes):
            scalar = initial if step % 2 == 0 else -initial
            if setter is not None:
                setter(scalar)
                ref_setter(scalar)
            args, refs = [], []
            for position, shape in enumerate(input_shapes):
                size = 1
                for dimension in shape:
                    size *= dimension
                values = [scalar if broadcast and position == 1 else 1.0] * size
                args.append(self.upload(values, shape))
                refs.append(self.upload(values, shape, self.torch))
            if origin == 'parameter':
                args.insert(0, scalar)
                refs.insert(0, scalar)
            expected = reference(*refs)
            actual = self.without_replay(fn, compiled, tuple(args))
            actual_bits = self.torch.tensor(actual.cpu().tolist(), dtype=self.torch.float32)
            observations.append({
                'step': step, 'input_shapes': input_shapes, 'scalar': repr(scalar),
                'actual_shape': tuple(actual.shape), 'expected_shape': tuple(expected.shape),
                'actual_bits': actual_bits.reshape(-1).view(self.torch.int32).tolist(),
                'expected_bits': expected.cpu().reshape(-1).view(self.torch.int32).tolist(),
            })
        # Materialize the entire history before asserting, so a mismatch on the
        # first revisit cannot discard later shape/sign observations from logs.
        self.assertEqual(
            [(item['actual_shape'], item['actual_bits']) for item in observations],
            [(item['expected_shape'], item['expected_bits']) for item in observations],
            msg=f'origin={origin}, initial={initial!r}, observations={observations!r}')

    def check_origins(self, shapes, broadcast=False):
        for origin in ('parameter', 'global', 'closure'):
            for initial in (0.0, -0.0):
                with self.subTest(origin=origin, initial=repr(initial), broadcast=broadcast):
                    self.check_history(origin, initial, shapes, broadcast)

    def test_signed_zero_one_dimensional_revisit(self):
        self.check_origins((((2,),), ((3,),), ((2,),), ((4,),), ((2,),)))

    def test_signed_zero_rank_transitions(self):
        self.check_origins((((2,),), ((1, 2),), ((3,),), ((2,),),
                            ((2, 1),), ((1, 2),), ((2,),)))

    def test_signed_zero_singleton_empty_transitions(self):
        self.check_origins((((2,),), ((3,),), ((1,),), ((2,),),
                            ((0,),), ((3,),), ((),), ((2,),), ((1,),)))

    def test_signed_zero_unused_tensor_shape_transitions(self):
        self.check_origins((((2,), (2,)), ((3,), (1, 3)), ((2,), (2,)),
                            ((2,), (1, 2)), ((2,), (2,)), ((0,), (1,)),
                            ((2,), (2,))))

    def test_signed_zero_broadcast_shape_transitions(self):
        self.check_origins((((2, 1), (1, 3)), ((3, 1), (1, 4)),
                            ((2, 1), (1, 3)), ((1, 1), (1, 3)),
                            ((0, 1), (1, 3)), ((2, 1), (1, 3))), broadcast=True)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class PersistentSpecializationHardware(unittest.TestCase):
    setUpClass = classmethod(jit_tests.Hardware.setUpClass.__func__)
    tearDown = jit_tests.Hardware.tearDown
    upload = jit_tests.Hardware.upload
    without_replay = jit_tests.Hardware.without_replay

    def pair(self, source):
        fn, ref_fn = program(source), program(source)
        self.outputs = []
        return fn, native.compile(fn), self.torch.compile(ref_fn)

    def host(self, value):
        return self.torch.tensor(value.cpu().tolist(), dtype=self.torch.float32).reshape(tuple(value.shape))

    def assert_bits(self, actual, expected):
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        keep = ~expected.isnan()
        self.assertTrue(self.torch.equal(actual.view(self.torch.int32)[keep],
                                        expected.view(self.torch.int32)[keep]))

    def check(self, pair, args, refs):
        fn, compiled, reference = pair
        inputs = [(value, self.host(value)) for value in args if type(value) is native.Tensor]
        ref_inputs = [(value, value.cpu().clone()) for value in refs if isinstance(value, self.torch.Tensor)]
        expected = reference(*refs)
        actual = self.without_replay(fn, compiled, args)
        self.assert_bits(self.host(actual), expected.cpu())
        self.assertEqual(actual.stride(), expected.stride())
        for value, before in inputs:
            self.assert_bits(self.host(value), before)
            self.assertIsNot(actual, value)
            if actual.numel() and value.numel():
                self.assertNotEqual(actual.data_ptr(), value.data_ptr())
        for value, before in ref_inputs:
            self.assert_bits(value.cpu(), before)
        for previous in self.outputs:
            self.assertIsNot(actual, previous)
            if actual.numel() and previous.numel():
                self.assertNotEqual(actual.data_ptr(), previous.data_ptr())
        self.outputs.append(actual)
        return actual

    def test_older_static_precision_guard_survives_other_rank_runtime_graph(self):
        pair = self.pair('def f(s,x):\n return ((x*0)+s)-16777216.0')
        results = []
        for shape, scalar in (((2,), 16777217.), ((2, 2), 16777218.),
                              ((2,), 16777217.), ((3,), 16777217.), ((2,), 16777217.)):
            count = 1
            for size in shape:
                count *= size
            x, tx = self.upload([1.]*count, shape), self.upload([1.]*count, shape, self.torch)
            with self.subTest(shape=shape, scalar=scalar):
                results.append(self.check(pair, (scalar, x), (scalar, tx)).cpu().tolist())
        self.assertEqual(results[0], [1., 1.])
        self.assertEqual(results[2], [1., 1.])
        self.assertEqual(results[3], [0., 0., 0.])
        self.assertEqual(results[4], [0., 0.])

    def test_nonfinite_runtime_hit_and_cold_rank_specialization(self):
        pair = self.pair('def f(s,x):\n return x*s')
        compiled = pair[1]
        runtime = None
        for step, (scalar, shape) in enumerate(((1.25, (4,)), (2.25, (4,)),
                                               (float('inf'), (4,)), (float('inf'), (2, 2)),
                                               (float('nan'), (4,)), (float('nan'), (2, 2)),
                                               (1.25, (4,)))):
            values = [0., -0., 1., -1.]
            x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
            with self.subTest(step=step, scalar=scalar, shape=shape):
                self.check(pair, (scalar, x), (scalar, tx))
                selected = kernels(compiled)[-1]
                if step == 1:
                    runtime = selected
                elif step in (2, 4, 6):
                    self.assertIs(selected, runtime)
                elif step in (3, 5):
                    self.assertNotIn('float s0', selected.source)

    def test_tensor_float_roles_follow_reversed_alias_realization_order(self):
        pair = self.pair('def f(a,b):\n return b-a')
        x = self.upload([1., -1., 0., -0.], (4,))
        tx = self.upload([1., -1., 0., -0.], (4,), self.torch)
        # b is realized first even though a occupies the first public slot.
        for args, refs in (((x, x), (tx, tx)),
                           ((16777217., x), (16777217., tx)),
                           ((x, 16777217.), (tx, 16777217.)),
                           ((x, x[:]), (tx, tx[:])),
                           ((16777218., x), (16777218., tx)),
                           ((x, x), (tx, tx)),
                           ((16777217., x), (16777217., tx))):
            self.check(pair, args, refs)

    def test_unused_local_scalar_tensor_roles_reuse_logical_guard(self):
        pair = self.pair('def f(unused,x):\n local=unused\n return x*1.125')
        x = self.upload([1., -1., 0., -0.], (4,))
        tx = self.upload([1., -1., 0., -0.], (4,), self.torch)
        for unused, ref_unused in ((0.375, 0.375), (x, tx), (x[:], tx[:]),
                                   (False, False), (16777217., 16777217.), (x, tx)):
            self.check(pair, (unused, x), (ref_unused, tx))
            self.assertEqual(len(cache(pair[1]).graphs), 1)

    def test_contiguous_singleton_stride_history_keeps_selected_zero_sign(self):
        pair = self.pair('def f(s,x):\n return x*s')
        history = ((2, True, 0.), (3, True, -0.), (2, True, 0.),
                   (4, True, 0.), (2, False, 0.), (3, False, -0.),
                   (2, True, -0.), (3, True, 0.))
        for step, (count, transposed, scalar) in enumerate(history):
            args = []
            for framework in (native, self.torch):
                base = self.upload([1.]*count, (1, count), framework)
                value = base.transpose(0, 1) if transposed else base.reshape(count, 1)
                self.assertTrue(value.is_contiguous())
                self.assertEqual(tuple(value.shape), (count, 1))
                self.assertEqual(value.stride(), (1, count if transposed else 1))
                args.append(value)
            with self.subTest(step=step, shape=(count, 1), strides=args[0].stride(), scalar=repr(scalar)):
                actual = self.check(pair, (scalar, args[0]), (scalar, args[1]))
                if step in (2, 3):
                    # Generalizing size and singleton stride at -0.0 preserves
                    # that graph's sign on later +0.0 guard hits.
                    self.assertTrue(self.host(actual).signbit().all().item())

    def test_logical_hit_new_broadcast_executor_keeps_frozen_zero(self):
        pair = self.pair('def f(s,x,y):\n return x*s')
        selected = None
        for step, (count, scalar) in enumerate(((2, 0.), (3, -0.), (4, 0.), (5, -0.))):
            shape = (count,) if step == 0 else (1, count)
            x, tx = self.upload([1.]*count, (count,)), self.upload([1.]*count, (count,), self.torch)
            y, ty = self.upload([2.]*count, shape), self.upload([2.]*count, shape, self.torch)
            self.check(pair, (scalar, x, y), (scalar, tx, ty))
            current = next(reversed(cache(pair[1]).graphs.values()))
            if step == 1:
                selected = current
            elif step > 1:
                self.assertIs(current, selected)
                self.assertEqual(len(cache(pair[1]).graphs), 2)
                self.assertIsNot(kernels(pair[1])[-1], previous_executor)
            previous_executor = kernels(pair[1])[-1]

    def test_failed_compile_and_launch_preserve_both_cache_orders_and_history(self):
        pair = self.pair('def f(s,x,y):\n return x*s')
        compiled = pair[1]
        for count, scalar in ((2, 0.), (3, -0.)):
            x, tx = self.upload([1.]*count, (count,)), self.upload([1.]*count, (count,), self.torch)
            y, ty = self.upload([2.]*count, (1, count)), self.upload([2.]*count, (1, count), self.torch)
            self.check(pair, (scalar, x, y), (scalar, tx, ty))

        def snapshot():
            state = cache(compiled)
            return ([(key, id(entry), list(entry.lowerings.items())) for key, entry in state.graphs.items()],
                    list(state.executors.items()))

        class FailedLaunch:
            def run(self, *args):
                raise RuntimeError('injected transactional launch failure')

        x, tx = self.upload([1.]*4, (4,)), self.upload([1.]*4, (4,), self.torch)
        y, ty = self.upload([2.]*4, (1, 4)), self.upload([2.]*4, (1, 4), self.torch)
        before = snapshot()
        # Zero hits the dynamic shape guard but needs a new address formula;
        # True requires a new logical specialization as well as an executable.
        for scalar in (0., True):
            for failure in ('compile', 'launch'):
                effect = ({'side_effect': RuntimeError('injected transactional compile failure')}
                          if failure == 'compile' else {'return_value': FailedLaunch()})
                with self.subTest(scalar=scalar, failure=failure), \
                     mock.patch.object(bridge, '_pointwise_compile', **effect):
                    with self.assertRaisesRegex(RuntimeError, 'injected transactional'):
                        compiled(scalar, x, y)
                self.assertEqual(snapshot(), before)
        self.check(pair, (0., x, y), (0., tx, ty))
        self.check(pair, (True, x, y), (True, tx, ty))


if __name__ == '__main__':
    unittest.main()
