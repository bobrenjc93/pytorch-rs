"""Callback-free globals lookup and reference-equivalent static float guards."""
import struct
import unittest
from unittest import mock

import torch_rs as native
from torch_rs import _compile_pointwise as frontend
from torch_rs import torch_rs as bridge
from tests.test_compile_pointwise_jit import available, cache, program


def install_key(fn, effects, string_subclass=False):
    class Key:
        def __hash__(self):
            effects.append('hash')
            return hash('scale')
        def __eq__(self, other):
            effects.append('equality')
            return other == 'scale'
        def __repr__(self):
            effects.append('repr')
            return 'key'
    class StringKey(str):
        __hash__ = Key.__hash__
        __eq__ = Key.__eq__
        __repr__ = Key.__repr__
    del fn.__globals__['scale']
    fn.__globals__[StringKey('scale') if string_subclass else Key()] = 2.0
    effects.clear()  # Fixture construction may invoke hash; admission must not.


class StaticGuardAdmission(unittest.TestCase):
    def test_globals_keys_rejected_before_lookup_callbacks(self):
        for body in ('x + scale', 'x.sum() + scale'):
            for subclass in (False, True):
                fn = program('def f(x):\n return '+body, scale=1.0)
                parsed = frontend.analyze(fn, 1)
                effects = []
                install_key(fn, effects, subclass)
                with self.subTest(body=body, string_subclass=subclass):
                    with self.assertRaises(NotImplementedError):
                        frontend.resolve(fn, parsed)
                    self.assertEqual(effects, [])

    def test_static_float_key_equates_zeros_without_losing_literal_bits(self):
        positive, pv = frontend.binding(0.0)
        negative, nv = frontend.binding(-0.0)
        self.assertEqual(positive, negative)
        self.assertNotEqual(struct.pack('=d', pv), struct.pack('=d', nv))
        self.assertNotEqual(frontend.scalar_bits(0.0), frontend.scalar_bits(-0.0))
        self.assertNotEqual(frontend.binding(False)[0], positive)
        self.assertNotEqual(frontend.binding(0)[0], positive)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class StaticGuardHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def test_public_cold_and_warm_globals_key_rejection_has_no_callbacks(self):
        for warm in (False, True):
            for invalid in (False, True):
                for subclass in (False, True):
                    fn = program('def f(x):\n return x + scale', scale=1.0)
                    compiled = native.compile(fn)
                    x = native.tensor([1., -1.]).to('cuda:0')
                    if warm:
                        compiled(x)
                    if invalid:
                        fn.__code__ = program('def f(x):\n return x.sum() + scale').__code__
                    effects = []
                    install_key(fn, effects, subclass)
                    before = len(cache(compiled).graphs)
                    with self.subTest(warm=warm, invalid=invalid, string_subclass=subclass):
                        with mock.patch.object(bridge, '_pointwise_compile', side_effect=AssertionError('codegen before admission')):
                            with self.assertRaises(NotImplementedError):
                                compiled(x)
                        self.assertEqual(effects, [])
                        self.assertEqual(len(cache(compiled).graphs), before)

    def pair(self, initial, closure):
        def make():
            if closure:
                scale = initial
                def fn(x):
                    return x * scale
                def setter(value):
                    nonlocal scale
                    scale = value
            else:
                fn = program('def f(x):\n return x*scale', scale=initial)
                def setter(value):
                    fn.__globals__['scale'] = value
            return fn, setter
        fn, setter = make()
        ref_fn, ref_setter = make()
        return native.compile(fn), self.torch.compile(ref_fn), setter, ref_setter

    def compare(self, compiled, reference, data):
        x = native.tensor(data).to('cuda:0')
        tx = self.torch.tensor(data, device='cuda:0')
        result = compiled(x)
        expected = reference(tx).cpu()
        actual = self.torch.tensor(result.cpu().tolist())
        self.assertTrue(self.torch.equal(actual.view(self.torch.int32), expected.view(self.torch.int32)))
        self.assertNotEqual(result.data_ptr(), x.data_ptr())
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device, x.device)
        self.assertEqual(x.cpu().tolist(), data)

    def test_static_zero_transitions_then_runtime_promotion(self):
        for initial in (0.0, -0.0):
            for closure in (False, True):
                compiled, reference, setter, ref_setter = self.pair(initial, closure)
                for step, value in enumerate((initial, -initial, initial, 1.25, -0.0, 0.0)):
                    setter(value)
                    ref_setter(value)
                    with self.subTest(initial=initial, closure=closure, step=step):
                        self.compare(compiled, reference, [1., -1.])
                        self.compare(compiled, reference, [-3., 4.])
                        self.assertEqual(len(cache(compiled).graphs), 1 if step < 3 else 2)
                native.compiler.reset()
                self.torch.compiler.reset()

    def test_static_zero_shape_change_and_reset_use_current_binding(self):
        for initial in (0.0, -0.0):
            compiled, reference, setter, ref_setter = self.pair(initial, False)
            self.compare(compiled, reference, [1., -1.])
            setter(-initial)
            ref_setter(-initial)
            self.compare(compiled, reference, [1., -1., 2.])
            native.compiler.reset()
            self.torch.compiler.reset()
            self.compare(compiled, reference, [1., -1.])
            self.assertEqual(len(cache(compiled).graphs), 1)
            native.compiler.reset()
            self.torch.compiler.reset()

    def test_literal_zero_signs_remain_distinct(self):
        for literal in ('0.0', '-0.0'):
            source = 'def f(x):\n return x*'+literal
            compiled = native.compile(program(source))
            reference = self.torch.compile(program(source))
            self.compare(compiled, reference, [1., -1.])


if __name__ == '__main__':
    unittest.main()
