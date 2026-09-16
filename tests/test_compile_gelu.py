"""Bounded native functional GELU and exact public-default admission."""
from contextlib import contextmanager
import inspect
import math
import sys
import types
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests import test_compile_pointwise_jit as support
from tests.test_compile_pointwise_jit import available, cache, lower, program


@contextmanager
def no_gelu_replay(fn):
    previous = sys.getprofile()
    def reject(frame, event, arg):
        if event == 'call' and frame.f_code is fn.__code__:
            raise AssertionError('original body executed')
        if event == 'c_call' and arg is native.nn.functional.gelu:
            raise AssertionError('eager GELU replayed')
    sys.setprofile(reject)
    try:
        yield
    finally:
        sys.setprofile(previous)


class Admission(unittest.TestCase):
    def test_real_builtin_and_exact_positional_surface(self):
        gelu = native.nn.functional.gelu
        self.assertIs(type(gelu), types.BuiltinFunctionType)
        self.assertEqual(str(inspect.signature(gelu)), '(input, /)')
        with self.assertRaises(AttributeError):
            gelu.__code__ = (lambda x: x).__code__
        for owner in (native, native.Tensor):
            self.assertFalse(hasattr(owner, 'gelu'))
            self.assertFalse(hasattr(owner, 'erf'))
        x = native.tensor([1.])
        for args, kwargs in [((), {}), ((x, 'none'), {}), ((x,), {'approximate': 'none'}),
                             ((), {'input': x})]:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(TypeError):
                gelu(*args, **kwargs)
        with self.assertRaises(NotImplementedError):
            gelu(x)

    def test_subclass_rejected_without_protocol_execution(self):
        with self.assertRaises(TypeError):
            type('Subclass', (native.Tensor,), {})
        class ForeignTensor:
            @classmethod
            def __torch_function__(cls, *args, **kwargs):
                raise AssertionError('unsupported tensor protocol dispatched')
        value = ForeignTensor()
        with self.assertRaises(TypeError):
            native.nn.functional.gelu(value)

    def test_namespace_forms_have_same_ordinary_ir_and_scalar_kinds(self):
        graphs = [lower(program('def f(x):\n return ' + expression,
                                nn=native.nn, F=native.nn.functional,
                                gelu=native.nn.functional.gelu))
                  for expression in ('fw.nn.functional.gelu(x)', 'nn.functional.gelu(x)',
                                     'F.gelu(x)', 'gelu(x)')]
        self.assertTrue(all(g == graphs[0] for g in graphs))
        self.assertEqual(len(graphs[0].nodes), 9)  # Input plus eight decomposition nodes.
        self.assertEqual([n[0] for n in graphs[0].nodes].count('erf'), 1)
        self.assertEqual([n[0] for n in graphs[0].nodes].count('integer'), 1)
        self.assertNotIn('gelu', [n[0] for n in graphs[0].nodes])
        self.assertIn('torch_rs_erf', bridge._pointwise_source(graphs[0].nodes, graphs[0].outputs, 1))

    def test_unsupported_calls_helpers_and_original_shape_boundary(self):
        for expression in ('x.gelu()', 'x.erf()', 'fw.erf(x)', 'fw.gelu(x)',
                           'fw.nn.functional.gelu(input=x)', 'fw.nn.functional.gelu(x, "none")',
                           'fw.nn.functional.gelu(0.5)'):
            with self.subTest(expression=expression), self.assertRaises(NotImplementedError):
                lower(program('def f(x):\n return ' + expression))
        helper = program('def f(x):\n return fw.nn.functional.gelu(x)')
        with self.assertRaises(NotImplementedError):
            lower(program('def f(x):\n return helper(x)', helper=helper))
        helper = program('def f(x):\n return x*.5')
        lower(program('def f(x):\n return fw.nn.functional.gelu(helper(x))', helper=helper))
        for body in ('return fw.nn.functional.gelu(x)',
                     'return fw.nn.functional.gelu(x)*0'):
            graph = lower(program('def f(x,y):\n ' + body), 2)
            with self.assertRaisesRegex(RuntimeError, 'equal actual input shapes'):
                bridge._pointwise_plan(graph.nodes, graph.outputs, 2, [[2], [1, 2]])
        # A dead GELU does not change the accepted broadcast-trig output domain.
        graph = lower(program('def f(x,y):\n g=fw.nn.functional.gelu(x)\n return y.sin()'), 2)
        source = bridge._pointwise_source(graph.nodes, graph.outputs, 2, [[2], [1, 2]])
        self.assertNotIn('torch_rs_erf', source)

    def test_decomposition_obeys_existing_node_budget(self):
        fn = program('def f(x):\n' + ' x=fw.nn.functional.gelu(x)\n' * 512 + ' return x')
        with self.assertRaisesRegex(NotImplementedError, '4096-node'):
            lower(fn)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class FunctionalGelu(unittest.TestCase):
    setUpClass = classmethod(support.Hardware.setUpClass.__func__)
    tearDown = support.Hardware.tearDown
    upload = support.Hardware.upload
    compare = support.Hardware.compare

    def test_eager_metadata_offsets_empty_scalar_and_retained_storage(self):
        values = [-float('inf'), -5., -1., -0., 0., 0.375, 5., float('inf'), float('nan')]
        retained = []
        for shape in [(), (0,), (2, 0, 3), (9,), (3, 3), (1, 3, 3, 1)]:
            n = math.prod(shape)
            data = (values * (n // len(values) + 1))[:n]
            for offset in (False, True):
                pair = []
                for fw in (native, self.torch):
                    x = self.upload([99.] + data + [88.], (n + 2,), fw)[1:n+1].reshape(shape) if offset else self.upload(data, shape, fw)
                    pair.append(x)
                x, tx = pair
                before = x.cpu().tolist()
                actual, expected = native.nn.functional.gelu(x), self.torch.nn.functional.gelu(tx)
                self.compare(actual, expected)
                self.assertEqual(actual.storage_offset(), 0)
                self.assertTrue(actual.is_contiguous())
                self.assertIsNot(actual, x)
                if n:
                    self.assertNotEqual(actual.data_ptr(), x.data_ptr())
                self.torch.testing.assert_close(self.torch.tensor(x.cpu().tolist()), self.torch.tensor(before), equal_nan=True)
                for old, snapshot in retained:
                    self.torch.testing.assert_close(self.torch.tensor(old.cpu().tolist()), snapshot, equal_nan=True)
                retained.append((actual, self.torch.tensor(actual.cpu().tolist())))

    def test_public_default_persistent_shapes_and_no_eager_replay(self):
        expressions = ['fw.nn.functional.gelu(x)', 'fw.nn.functional.gelu(fw.nn.functional.gelu(x))',
                       'fw.nn.functional.gelu(x*.5-.25)*3.+.125',
                       'fw.nn.functional.gelu(x).sin()+x.cos()']
        for expression in expressions:
            fn = program('def f(x):\n return ' + expression)
            compiled = native.compile(fn)
            reference = self.torch.compile(program('def f(x):\n return ' + expression, self.torch))
            retained = []
            for shape in [(2,), (13,), (2,), (257,), (2,)]:
                values = [(-1 if i % 2 else 1) * (i % 23) / 4 for i in range(math.prod(shape))]
                args = self.upload(values, shape), self.upload(values, shape, self.torch)
                with no_gelu_replay(fn):
                    actual = compiled(args[0])
                self.compare(actual, reference(args[1]))
                for old, before in retained:
                    self.assertEqual(old.cpu().tolist(), before)
                retained.append((actual, actual.cpu().tolist()))
            for kernel in cache(compiled).executors.values():
                self.assertIn('--relocatable-device-code=true', kernel.options)
                self.assertIn('torch_rs_erf', kernel.source)

    def test_every_traversed_namespace_guard_on_warm_calls(self):
        bindings = dict(nn=native.nn, F=native.nn.functional, gelu=native.nn.functional.gelu)
        x = self.upload([-.75, .5], (2,))
        for expression, edges in [('fw.nn.functional.gelu(x)', [(native, 'nn'), (native.nn, 'functional'), (native.nn.functional, 'gelu')]),
                                  ('nn.functional.gelu(x)', [(native.nn, 'functional'), (native.nn.functional, 'gelu')]),
                                  ('F.gelu(x)', [(native.nn.functional, 'gelu')])]:
            fn = program('def f(x):\n return ' + expression, **bindings)
            compiled = native.compile(fn)
            compiled(x)
            before = (len(cache(compiled).graphs), len(cache(compiled).executors))
            for owner, attr in edges:
                with (patch.object(owner, attr, object()),
                      patch.object(frontend, 'lower', side_effect=AssertionError('warm lowering')),
                      self.assertRaisesRegex(NotImplementedError, 'namespace')):
                    compiled(x)
                self.assertEqual(before, (len(cache(compiled).graphs), len(cache(compiled).executors)))
            with patch.object(frontend, 'lower', side_effect=AssertionError('warm lowering')):
                compiled(x)
        # A saved builtin alias must remain independent of untraversed namespaces.
        fn = program('def f(x):\n return gelu(x)', **bindings)
        compiled = native.compile(fn)
        expected = compiled(x).cpu().tolist()
        with patch.object(native, 'nn', object()):
            self.assertEqual(compiled(x).cpu().tolist(), expected)
        with patch.object(native.nn.functional, 'gelu', object()):
            self.assertEqual(compiled(x).cpu().tolist(), expected)

    def test_namespace_keys_and_types_reject_without_callbacks(self):
        x = self.upload([1.], (1,))
        for owner, name, expression in [(native, 'nn', 'fw.nn.functional.gelu(x)'),
                                        (native.nn, 'functional', 'NN.functional.gelu(x)'),
                                        (native.nn.functional, 'gelu', 'F.gelu(x)')]:
            fn = program('def f(x):\n return ' + expression, NN=native.nn, F=native.nn.functional)
            compiled = native.compile(fn)
            compiled(x)
            namespace = owner.__dict__
            expected = namespace[name]
            armed = False
            class Key:
                def __hash__(self):
                    return hash(name)
                def __eq__(self, other):
                    if armed:
                        raise AssertionError('namespace equality callback')
                    return False
            key = Key()
            del namespace[name]
            namespace[key] = expected
            armed = True
            try:
                for call in (compiled, native.compile(fn)):
                    with self.assertRaisesRegex(NotImplementedError, 'keys'):
                        call(x)
            finally:
                armed = False
                del namespace[key]
                namespace[name] = expected
            class HostileModule(types.ModuleType):
                def __getattribute__(self, attr):
                    raise AssertionError('namespace attribute callback')
            original_type = type(owner)
            fresh = native.compile(fn)
            owner.__class__ = HostileModule
            try:
                for call in (compiled, fresh):
                    with self.assertRaisesRegex(NotImplementedError, 'type'):
                        call(x)
            finally:
                types.ModuleType.__setattr__(owner, '__class__', original_type)

    def test_native_eager_rejections_and_explicit_eager_backend(self):
        x = self.upload([1., 2., 3., 4.], (2, 2))
        with self.assertRaises(NotImplementedError):
            native.nn.functional.gelu(x.transpose(0, 1))
        # Public CUDA gradient creation remains unsupported before GELU.
        with self.assertRaises(NotImplementedError):
            x.requires_grad_(True)
        leaf = native.tensor([1.], requires_grad=True)
        for context in (native.enable_grad(), native.no_grad()):
            with context, self.assertRaises(NotImplementedError):
                native.nn.functional.gelu(leaf)
        fn = program('def f(x):\n return fw.nn.functional.gelu(x)')
        with self.assertRaises(NotImplementedError):
            native.compile(fn, backend='eager')(x)
        from torch_rs.overrides import TorchFunctionMode
        class Mode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise AssertionError('unsupported mode dispatched')
        with Mode(), self.assertRaises(TypeError):
            native.nn.functional.gelu(x)


if __name__ == '__main__':
    unittest.main()
