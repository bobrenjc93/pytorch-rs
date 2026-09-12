"""Changed captured floats retain default Inductor's runtime rounding boundary."""
import struct
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import torch_rs as native
from torch_rs import _compile_pointwise as frontend
from torch_rs import torch_rs as bridge
from tests.test_compile_pointwise_jit import available, cache, program


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
        keys = ((float, struct.pack('=d', -0.0)),)
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
                    entries = list(cache(compiled).graphs.values())
                    if len(entries) == 2:
                        if dynamic_kernel is not None:
                            self.assertIs(entries[-1][1], dynamic_kernel)
                        dynamic_kernel = entries[-1][1]
                self.assertEqual(outputs, [1., 0., 0., 2.])
                entries = list(cache(compiled).graphs.values())
                self.assertEqual(len(entries), 2)
                self.assertIn('float s0', entries[-1][1].source)
                native.compiler.reset()
                self.torch.compiler.reset()
                setter(16777217.0)
                ref_setter(16777217.0)
                self.assertEqual(self.compare(compiled, reference, [1., -1., 0.])[0].item(), 1.)
                self.assertEqual(len(cache(compiled).graphs), 1)
                self.assertNotIn('float s0', next(iter(cache(compiled).graphs.values()))[1].source)
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
            entries = list(cache(compiled).graphs.values())
            if len(entries) == 2:
                if dynamic_kernel is not None:
                    self.assertIs(entries[-1][1], dynamic_kernel)
                dynamic_kernel = entries[-1][1]
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
        kernel = list(cache(compiled).graphs.values())[-1][1]
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

    @unittest.skipUnless(os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1',
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
        entries = list(cache(compiled).graphs.values())
        kernel = next(entry[1] for entry in entries if entry[1].device == 0 and 'float s0' in entry[1].source)
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
        kernel = list(cache(compiled).graphs.values())[-1][1]
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


if __name__ == '__main__':
    unittest.main()
