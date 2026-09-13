"""Typed scalars, callback-free admission, and signed product provenance."""
import unittest
from unittest import mock

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


class PositionalBindingAdmission(unittest.TestCase):
    def graph(self, source, args, **captures):
        fn = program(source, **captures)
        tensors, parameters = frontend.bind_arguments(args)
        parsed = frontend.analyze(fn, len(args))
        keys, values = frontend.resolve(fn, parsed, parameters)
        return parsed, keys, values, frontend.lower(parsed, values, len(tensors))

    def test_source_positions_are_separate_from_tensor_and_scalar_operands(self):
        from torch_rs import torch_rs as bridge
        x, y = native.tensor([1.]), native.tensor([2.])
        parsed, keys, values, graph = self.graph(
            'def f(a,x,b,y,c):\n return (x*a)+(y*b)+c+captured',
            (0.375, x, True, y, -0.75), captured=1.125)
        self.assertEqual(graph.inputs, 2)
        self.assertEqual(graph.nodes[:2], (('input', 0, 0, 0), ('input', 1, 0, 0)))
        self.assertEqual([(s.kind, s.name, s.position) for s in parsed.dependencies],
                         [('parameter', 'a', 0), ('parameter', 'x', 1),
                          ('parameter', 'b', 2), ('parameter', 'y', 3),
                          ('parameter', 'c', 4), ('LOAD_GLOBAL', 'captured', None)])
        self.assertIn('x1[i]', bridge._pointwise_source(graph.nodes, graph.output, 2))
        self.assertEqual(keys[1], ('tensor', 0))
        self.assertEqual(keys[3], ('tensor', 1))
        self.assertEqual(values[parsed.dependencies[3]], frontend.Value(1))

    def test_positional_rejection_never_invokes_object_or_type_hooks(self):
        effects = []
        class Trap(type(callback_scalar(effects))):
            def __float__(self):
                effects.append('float')
                return 1.0
            def __bool__(self):
                effects.append('bool')
                return True
            def __getattribute__(self, name):
                effects.append('attribute')
                raise AssertionError(name)
            def __eq__(self, other):
                effects.append('comparison')
                return False
        class Float(float):
            __float__ = Trap.__float__
        class Integer(int):
            __int__ = Trap.__float__
        x = native.tensor([1.])
        for value in (0, 1, 2**63, -2**100, 1+0j, None, [], {}, Trap(), Float(1), Integer(1)):
            for args in ((value, x), (x, value)):
                with self.assertRaises(NotImplementedError):
                    frontend.bind_arguments(args)
                self.assertEqual(effects, [])
        for args in ((), (1.0, True), (x, x, False, x)):
            with self.assertRaises(NotImplementedError):
                frontend.bind_arguments(args)

    def test_scalar_slots_preserve_original_ir_actual_shape_admission(self):
        from torch_rs import torch_rs as bridge
        x, y = native.tensor([1.]), native.tensor([2.])
        for body in ('x.sin()+scale', 'x*scale+scale', 'x*scale*False'):
            _, _, _, graph = self.graph('def f(scale,x,unused,flag):\n return '+body,
                                        (0.375, x, y, True))
            for shapes in (((5,), (1, 5)), ((5,), ()), ((0, 5), (1, 5))):
                with self.subTest(body=body, shapes=shapes):
                    with self.assertRaisesRegex(RuntimeError, 'arithmetic|sin/cos'):
                        bridge._pointwise_source(graph.nodes, graph.output, 2, shapes)
            # One tensor plus scalars retains the complete one-tensor language.
            _, _, _, graph = self.graph('def f(scale,x,flag):\n return '+body,
                                        (0.375, x, True))
            bridge._pointwise_source(graph.nodes, graph.output, 1, ((1, 5),))
        _, _, _, graph = self.graph('def f(a,x,b,y):\n return (x*a).relu()',
                                    (0.375, x, False, y))
        bridge._pointwise_source(graph.nodes, graph.output, 2, ((5,), (1, 5)))

    def test_captures_and_parameters_share_one_runtime_budget(self):
        from torch_rs import torch_rs as bridge
        x = native.tensor([1.])
        for total in (64, 65):
            parameters = [f'a{i}' for i in range(33)]
            captures = {f'g{i}': 1.25 for i in range(total-33)}
            fn = program('def f(x,'+','.join(parameters)+'):\n return x+'
                         +'+'.join(parameters+list(captures)), **captures)
            parsed = frontend.analyze(fn, 34)
            _, params = frontend.bind_arguments((x,)+(1.25,)*33)
            keys, values = frontend.resolve(fn, parsed, params)
            history = {(parsed.code, keys, (), (0,)): None}
            fn.__globals__.update({name: 2.5 for name in captures})
            _, params = frontend.bind_arguments((x,)+(2.5,)*33)
            keys, values = frontend.resolve(fn, parsed, params)
            if total == 65:
                with self.assertRaisesRegex(NotImplementedError, '64 runtime scalar'):
                    frontend.runtime_bindings(parsed, keys, values, history)
            else:
                _, values, scalars = frontend.runtime_bindings(parsed, keys, values, history)
                self.assertEqual(len(scalars), 64)
                graph = frontend.lower(parsed, values, 1)
                source = bridge._pointwise_source(graph.nodes, graph.output, 1)
                self.assertIn('float s63', source)
                self.assertNotIn('float s64', source)


class SharedCacheGuards(unittest.TestCase):
    """Exercise production cache selection without allocating synthetic shapes.

    Only the native metadata/compile/launch boundary is replaced. The returned
    IR and runtime arguments reveal which scalar specialization was dispatched;
    hardware tests remain responsible for numerical admission and execution.
    """

    def setUp(self):
        self.metadata = {}
        self.arguments = []
        self.launch_failure = None
        for name, replacement in (
                ('_compile_trace_tensor_metadata', lambda arg: self.metadata[id(arg)]),
                ('_pointwise_validate_inputs', lambda args: None)):
            patcher = mock.patch.object(frontend._native, name, side_effect=replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(frontend._native, '_pointwise_compile',
                                    side_effect=self.make_executor)
        self.compile_bridge = patcher.start()
        self.addCleanup(patcher.stop)

    def make_executor(self, tensors, nodes, output):
        executor = mock.Mock()
        if self.launch_failure is not None:
            executor.run.side_effect = self.launch_failure
        else:
            executor.run.side_effect = lambda args, scalars: (nodes, tuple(scalars))
        return executor

    def argument(self, shape, strides=None):
        if strides is None:
            stride, reversed_strides = 1, []
            for size in reversed(shape):
                reversed_strides.append(stride)
                stride *= max(size, 1)
            strides = tuple(reversed(reversed_strides))
        arg = native.tensor([1.])
        self.arguments.append(arg)  # Keep identities stable for metadata lookup.
        self.metadata[id(arg)] = (shape, strides, 'float32', False, 'cuda:0', 0)
        return arg

    def compiled(self, source='def f(scale,x):\n return x*scale'):
        return frontend.implementation(program(source), recompile_limit=8)

    def test_generalized_selection_keeps_zero_singleton_and_rank_zero_separate(self):
        compiled = self.compiled()
        positive = compiled(0., self.argument((2,)))
        negative = compiled(-0., self.argument((3,)))
        self.assertNotEqual(positive, negative)
        self.assertEqual(compiled(0., self.argument((2,))), negative)
        singleton = compiled(0., self.argument((1,)))
        empty = compiled(-0., self.argument((0,)))
        scalar = compiled(0., self.argument(()))
        self.assertEqual(len(cache(compiled).graphs), 5)
        self.assertEqual(compiled(0., self.argument((4,))), negative)
        self.assertEqual(compiled(-0., self.argument((1,))), singleton)
        self.assertEqual(compiled(0., self.argument((0,))), empty)
        self.assertEqual(compiled(-0., self.argument(())), scalar)
        self.assertEqual(len(cache(compiled).graphs), 5)

    def test_rank_history_generalizes_previously_unchanged_dimensions(self):
        compiled = self.compiled()
        compiled(0., self.argument((2, 3)))
        compiled(0., self.argument((7,)))
        after_rank_change = compiled(-0., self.argument((2, 4)))
        self.assertEqual(compiled(0., self.argument((9, 4))), after_rank_change)
        self.assertEqual(len(cache(compiled).graphs), 3)

    def test_role_history_stays_with_public_source_across_tensor_positions(self):
        compiled = self.compiled('def f(a,b):\n return a*b')
        compiled(0.375, self.argument((2, 3)))
        tensor_first = compiled(self.argument((2, 3)), 0.375)
        changed_shape = compiled(self.argument((4, 5)), 1.125)
        self.assertEqual(changed_shape[0], tensor_first[0])
        self.assertEqual(tensor_first[1], (0.375,))
        self.assertEqual(changed_shape[1], (1.125,))
        self.assertEqual(len(cache(compiled).graphs), 2)

    def test_contiguous_stride_relations_guard_noncanonical_singleton_strides(self):
        compiled = self.compiled()
        compiled(0., self.argument((2, 1, 3)))
        generalized = compiled(-0., self.argument((3, 1, 4)))
        self.assertEqual(compiled(0., self.argument((4, 1, 5))), generalized)
        # Both layouts are contiguous, but the prior graph inferred the middle
        # stride from the trailing dimension. The new layout must retrace.
        independent = compiled(0., self.argument((4, 1, 5), (5, 9, 1)))
        self.assertNotEqual(independent, generalized)
        self.assertEqual(len(cache(compiled).graphs), 3)

    def test_static_shape_history_promotes_only_independent_singleton_stride(self):
        compiled = self.compiled()
        compiled(0., self.argument((2, 1, 3)))
        generalized = compiled(-0., self.argument((2, 1, 3), (3, 9, 1)))
        self.assertEqual(compiled(0., self.argument((2, 1, 3), (3, 11, 1))), generalized)
        self.assertEqual(len(cache(compiled).graphs), 2)
        self.assertNotEqual(compiled(0., self.argument((2, 1, 3), (3, 1, 1))), generalized)
        self.assertEqual(len(cache(compiled).graphs), 3)

    def test_generalized_binary_shapes_preserve_broadcast_equalities(self):
        compiled = self.compiled('def f(scale,x,y):\n discarded=x*scale\n return x*y')
        compiled(0., self.argument((2, 3)), self.argument((2, 3)))
        generalized = compiled(-0., self.argument((4, 5)), self.argument((4, 5)))
        self.assertEqual(compiled(0., self.argument((6, 7)), self.argument((6, 7))), generalized)
        guards = next(reversed(cache(compiled).graphs))[2]
        x, y = self.argument((6, 7)), self.argument((6, 8))
        by_source = {frontend.BindingSource('parameter', name, position): self.metadata[id(arg)][:5]
                     for name, position, arg in (('x', 1, x), ('y', 2, y))}
        # Individually these shapes meet the rank/size guards. Their unequal
        # nonsingleton dimensions must fail before selecting this graph.
        self.assertFalse(guards.matches(by_source))
        self.assertEqual(len(cache(compiled).graphs), 2)

    def test_64bit_iteration_specialization_has_no_residual_or_inverse_32bit_bounds(self):
        compiled = self.compiled('def f(scale,x,y):\n discarded=x*scale\n return x*y')
        compiled(0., self.argument((2, 1)), self.argument((1, 2)))
        small = compiled(-0., self.argument((3, 1)), self.argument((1, 3)))
        # Each input fits 32-bit indexing, but their broadcast output does not.
        large = compiled(0., self.argument((65536, 1)), self.argument((1, 65536)))
        self.assertNotEqual(small, large)
        # A 64-bit graph has neither leftover bounds on individual inputs nor
        # an inverse bound that would prevent it from accepting small inputs.
        self.assertEqual(compiled(-0., self.argument((2**31, 1)), self.argument((1, 2))), large)
        self.assertEqual(compiled(-0., self.argument((2, 1)), self.argument((1, 2))), large)
        self.assertEqual(len(cache(compiled).graphs), 3)

    def test_32bit_index_limit_is_inclusive(self):
        compiled = self.compiled()
        compiled(0., self.argument((2,)))
        small = compiled(-0., self.argument((3,)))
        self.assertEqual(compiled(0., self.argument((2**31-1,))), small)
        self.assertEqual(len(cache(compiled).graphs), 2)
        large = compiled(0., self.argument((2**31,)))
        self.assertNotEqual(large, small)
        self.assertEqual(compiled(-0., self.argument((2,))), large)
        self.assertEqual(len(cache(compiled).graphs), 3)

    def test_logical_hit_abi_changes_keep_both_lrus_bounded_without_consuming_slots(self):
        fn = program('def f(a,x,b):\n local=a\n other=b\n return x*1.125')
        compiled = frontend.implementation(fn, recompile_limit=1)
        x, other = self.argument((4,)), self.argument((4,))
        history = ((0.375, x, False), (x, x, True), (other, x, False),
                   (True, x, other), (False, x, x), (True, x, -0.75))
        owner, entry = cache(compiled), None
        for cycle in range(2):
            for step, args in enumerate(history):
                with self.subTest(cycle=cycle, step=step):
                    _, scalars = compiled(*args)
                    self.assertEqual(scalars, ())
                    self.assertEqual(len(owner.graphs), 1)
                    current = next(iter(owner.graphs.values()))
                    if entry is None:
                        entry = current
                    self.assertIs(current, entry)
                    self.assertEqual(len(owner.executors), 1)
                    self.assertEqual(len(entry.lowerings), 1)
        # Revisited ABIs require rebuilding evicted executables while the
        # original logical specialization remains selected throughout.
        self.assertGreater(self.compile_bridge.call_count, len(history))

    def test_compile_and_launch_failures_do_not_publish_history_and_reset_clears_both_levels(self):
        compiled = self.compiled()
        original = compiled(0., self.argument((2,)))
        owner = cache(compiled)
        graphs, executors = dict(owner.graphs), dict(owner.executors)
        lowerings = [dict(entry.lowerings) for entry in owner.graphs.values()]
        self.compile_bridge.side_effect = RuntimeError('synthetic compile failure')
        with self.assertRaisesRegex(RuntimeError, 'synthetic compile failure'):
            compiled(-0., self.argument((3,)))
        self.compile_bridge.side_effect = self.make_executor
        self.launch_failure = RuntimeError('synthetic launch failure')
        with self.assertRaisesRegex(RuntimeError, 'synthetic launch failure'):
            compiled(-0., self.argument((3,)))
        self.assertEqual(owner.graphs, graphs)
        self.assertEqual(owner.executors, executors)
        self.assertEqual([entry.lowerings for entry in owner.graphs.values()], lowerings)
        self.launch_failure = None
        self.assertNotEqual(compiled(-0., self.argument((3,))), original)
        native.compiler.reset()
        self.assertFalse(owner.graphs)
        self.assertFalse(owner.executors)
        self.assertEqual(compiled(0., self.argument((2,))), original)
        self.assertEqual(len(owner.graphs), 1)


if __name__ == '__main__':
    unittest.main()
