"""Typed scalars, callback-free admission, and signed product provenance."""
import unittest

import torch_rs as native
from torch_rs import _compile_pointwise as frontend
from tests.test_compile_pointwise_jit import available, cache, lower, program


def callback_scalar(effects):
    class Meta(type):
        def __eq__(cls, other):
            effects.append('metaclass equality')
            return False
        __hash__ = type.__hash__

    class Scalar(metaclass=Meta):
        def __repr__(self):
            effects.append('constant repr')
            return 'custom scalar'
    return Scalar()


def replace_constant(fn, value):
    fn.__code__ = fn.__code__.replace(co_consts=(None, value))


class ScalarAdmission(unittest.TestCase):
    def test_boolean_constants_retain_their_ir_type(self):
        boolean = lower(program('def f(x):\n return x * False'))
        floating = lower(program('def f(x):\n return x * 0.0'))
        integer = lower(program('def f(x):\n return x * 0'))
        self.assertNotEqual(boolean, floating)
        self.assertNotEqual(integer, floating)

    def test_scalar_types_are_checked_by_identity(self):
        effects = []
        value = callback_scalar(effects)
        for check in (frontend.scalar_bits, frontend.binding):
            with self.subTest(check=check.__name__):
                with self.assertRaises(NotImplementedError):
                    check(value)
                self.assertEqual(effects, [])

    def test_constants_rejected_before_disassembly_repr(self):
        for nested in (False, True):
            effects = []
            value = callback_scalar(effects)
            fn = program('def f(x):\n return x + 1')
            replace_constant(fn, (value,) if nested else value)
            with self.subTest(nested=nested):
                with self.assertRaises(NotImplementedError):
                    frontend.analyze(fn, 1)
                self.assertEqual(effects, [])


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class ScalarHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def compare(self, actual, expected):
        torch = self.torch
        host = torch.tensor(actual.cpu().tolist(), dtype=torch.float32)
        expected = expected.cpu()
        torch.testing.assert_close(host, expected, rtol=0, atol=0, equal_nan=True)
        zeros = (host == 0) & (expected == 0)
        self.assertTrue(torch.equal(host.signbit()[zeros], expected.signbit()[zeros]))
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), 'cuda:0')
        self.assertFalse(actual.requires_grad)

    def test_global_scalar_cold_and_warm_rejection_never_calls_equality(self):
        x = native.tensor([2.]).to('cuda:0')
        for warmed in (False, True):
            effects = []
            fn = program('def f(x):\n return x + scale', scale=1.)
            compiled = native.compile(fn)
            if warmed:
                compiled(x)
            entries = len(cache(compiled).graphs)
            fn.__globals__['scale'] = callback_scalar(effects)
            with self.subTest(warmed=warmed):
                with self.assertRaises(NotImplementedError):
                    compiled(x)
                self.assertEqual(effects, [])
                self.assertEqual(len(cache(compiled).graphs), entries)
                fn.__globals__['scale'] = 1.
                self.assertEqual(compiled(x).cpu().tolist(), [3.])

    def test_custom_constants_cold_and_changed_code_never_call_repr(self):
        x = native.tensor([2.]).to('cuda:0')
        for warmed in (False, True):
            effects = []
            fn = program('def f(x):\n return x + 1')
            compiled = native.compile(fn)
            if warmed:
                compiled(x)
            old_code = fn.__code__
            entries = len(cache(compiled).graphs)
            replace_constant(fn, callback_scalar(effects))
            with self.subTest(warmed=warmed):
                with self.assertRaises(NotImplementedError):
                    compiled(x)
                self.assertEqual(effects, [])
                self.assertEqual(len(cache(compiled).graphs), entries)
                fn.__code__ = old_code
                self.assertEqual(compiled(x).cpu().tolist(), [3.])

    def test_boolean_multiplication_and_captured_type_changes(self):
        values = [float('inf'), -float('inf'), float('nan'), -3., 3., -0., 0., -1e-38]
        for body in ('x * scale', 'scale * x', '(x * scale) + x', '(x.sin() * scale).cos()'):
            source = 'def f(x):\n return ' + body
            fn, ref_fn = program(source, scale=False), program(source, scale=False)
            compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
            # Reuse one wrapper across changes that have identical float32 bits.
            for value in (False, 0.0, True, 1.0, 0, False):
                fn.__globals__['scale'] = ref_fn.__globals__['scale'] = value
                args = native.tensor(values).to('cuda:0')
                refs = self.torch.tensor(values, device='cuda:0')
                with self.subTest(body=body, scalar=value, scalar_type=type(value).__name__):
                    actual = compiled(args)
                    self.compare(actual, reference(refs))
                    self.compare(args, refs)
                    self.assertNotEqual(actual.data_ptr(), args.data_ptr())
            self.assertEqual(len(cache(compiled).graphs), 5)

    def test_literal_boolean_multiplication_in_both_orders(self):
        values = [float('inf'), -float('inf'), float('nan'), -3., 3., -0., 0.]
        x = native.tensor(values).to('cuda:0')
        tx = self.torch.tensor(values, device='cuda:0')
        for literal in ('False', 'True', '0.0', '1.0'):
            for expression in (f'x * {literal}', f'{literal} * x'):
                source = 'def f(x):\n return ' + expression
                compiled = native.compile(program(source))
                reference = self.torch.compile(program(source))
                with self.subTest(expression=expression):
                    result = compiled(x)
                    self.compare(result, reference(tx))
                    self.assertNotEqual(result.data_ptr(), x.data_ptr())

    def test_composed_zero_tensors_keep_float_semantics_and_zero_signs(self):
        values = [float('inf'), -float('inf'), float('nan'), -3., 3., -0., 0.]
        x = native.tensor(values).to('cuda:0')
        tx = self.torch.tensor(values, device='cuda:0')
        for expression in ('(x*False)*x', 'x*(x*False)', '-(x*False)',
                           '(x*False)+x', '(x*False).neg()+x',
                           '(x*False)+(x*False).neg()', '(x*False).neg()+(x*False)',
                           '(x*False).neg()+(x*False).neg()', 'x*0', 'x+0',
                           'x-False', 'False-x',
                           '(x+True)*x-(x+1.0)*x', '(x+0)*x-(x+0.0)*x'):
            source = 'def f(x):\n return ' + expression
            compiled = native.compile(program(source))
            reference = self.torch.compile(program(source))
            with self.subTest(expression=expression):
                self.compare(compiled(x), reference(tx))

    def test_signed_products_keep_rounding_and_expression_identity(self):
        values = [2e38, -2e38, 1e10, -1e10, 1.25, -1.25, 0., -0.,
                  float('inf'), -float('inf'), float('nan'), 1e-38, -1e-38]
        bodies = ['x * -2.0 - x * -1.0', 'x * 2.0 - x * 1.0',
                  'x * 2.0 + x * -2.0', 'x * -2.0 + x * 2.0',
                  'x * -2.0 - x * -2.0', 'x * 2.0 - x * 2.0',
                  'x * -2.0 + x', 'x + x * -2.0',
                  'a=x * 2.0\n return a + x * -2.0',
                  'a=x * -2.0\n return a - x * -1.0']
        for coefficient in (1.137, 3.713, 4.0, 0.5):
            bodies.extend([f'x * {-coefficient!r} - x * -1.0',
                           f'x * {coefficient!r} + x * {-coefficient!r}',
                           f'x * {-coefficient!r} + x * {coefficient!r}',
                           f'x * {-coefficient!r} - x * {-coefficient!r}'])
        for body in bodies:
            source = 'def f(x):\n ' + (body if body.startswith('a=') else 'return ' + body)
            compiled = native.compile(program(source))
            reference = self.torch.compile(program(source))
            for data in (values, list(reversed(values))):
                x = native.tensor(data).to('cuda:0')
                tx = self.torch.tensor(data, device='cuda:0')
                with self.subTest(body=body, reversed_values=data is not values):
                    result = compiled(x)
                    self.compare(result, reference(tx))
                    self.compare(x, tx)
                    self.assertNotEqual(result.data_ptr(), x.data_ptr())
            self.assertEqual(len(cache(compiled).graphs), 1)

    def test_negated_constant_tensor_arithmetic_preserves_zero_signs(self):
        values = [float('inf'), -float('inf'), float('nan'), -3., 3., -0., 0., 1e-38]
        # Include both operand orders, repeated nodes, and several arithmetic
        # stages. Floating zero controls must retain data-dependent IEEE values.
        bodies = ('-((x*scale)+0.0)', '-(0.0+(x*scale))',
                  '-((x*scale)+-0.0)', '-((x*scale)*2.0)',
                  '-(2.0*(x*scale))', '-((x*scale)-(x*scale))',
                  '-(((x*scale)+0.0)*3.713)',
                  '-(((x*scale)+1.25)-1.25)',
                  '-((x*scale)*-2.0)', '-((x*scale).neg()+-0.0)',
                  '-((x*scale).neg()-(x*scale))',
                  '-((x*scale)+(x*scale).neg())',
                  '-((x*scale).sin())', '-((x*scale).relu())',
                  '-((x*scale)*floatinf)')
        for body in bodies:
            source = 'def f(x):\n return ' + body
            fn = program(source, scale=0, floatinf=float('inf'))
            ref_fn = program(source, scale=0, floatinf=float('inf'))
            compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
            for scalar in (0, False, 0.0, 0):
                fn.__globals__['scale'] = ref_fn.__globals__['scale'] = scalar
                for data in (values, list(reversed(values))):
                    x = native.tensor(data).to('cuda:0')
                    tx = self.torch.tensor(data, device='cuda:0')
                    with self.subTest(body=body, kind=type(scalar).__name__, fresh=data is not values):
                        result = compiled(x)
                        expected = reference(tx)
                        self.compare(result, expected)
                        self.compare(x, tx)
                        self.assertNotEqual(result.data_ptr(), x.data_ptr())
                        if type(scalar) is not float and body in bodies[:8]:
                            host = self.torch.tensor(result.cpu().tolist())
                            self.assertTrue(self.torch.equal(host, self.torch.zeros_like(host)))
                            self.assertTrue(host.signbit().all().item())
            self.assertEqual(len(cache(compiled).graphs), 3)


if __name__ == '__main__':
    unittest.main()
