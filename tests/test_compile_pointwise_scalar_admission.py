"""Typed scalars, callback-free admission, and signed product provenance."""
import struct
import unittest
from unittest import mock

import torch_rs as native
from torch_rs import _compile_pointwise as frontend
from tests.test_compile_pointwise_jit import available, cache, lower, mock_pointwise_executor, program


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


def scalar_source(origin):
    """Independent functions/setters for persistent scalar-source histories."""
    if origin == 'parameter':
        return program('def f(scale,x):\n return x*scale'), lambda value: None
    if origin == 'global':
        fn = program('def f(x):\n return x*scale', scale=0.)
        return fn, lambda value: fn.__globals__.__setitem__('scale', value)
    scale = 0.
    def fn(x):
        return x * scale
    def setter(value):
        nonlocal scale
        scale = value
    return fn, setter


def signed_payload_nans():
    return [struct.unpack('=d', struct.pack('=Q', sign | 0x7ff8000000000000 | payload))[0]
            for sign in (0, 1 << 63) for payload in range(1, 7)]


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
        return parsed, keys, values, frontend.lower(parsed, values, len(tensors)).graph

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
        self.assertIn('x1[i]', bridge._pointwise_source(graph.nodes, graph.outputs, 2))
        self.assertEqual(keys[parsed.dependencies[1]], ('tensor', 0))
        self.assertEqual(keys[parsed.dependencies[3]], ('tensor', 1))
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
        for value in (0, 1, 2**63, -2**100, 1+0j, None, Trap(), Float(1), Integer(1)):
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
                    if body == 'x.sin()+scale':
                        bridge._pointwise_source(graph.nodes, graph.outputs, 2, shapes)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'arithmetic|sin/cos'):
                            bridge._pointwise_source(graph.nodes, graph.outputs, 2, shapes)
            # One tensor plus scalars retains the complete one-tensor language.
            _, _, _, graph = self.graph('def f(scale,x,flag):\n return '+body,
                                        (0.375, x, True))
            bridge._pointwise_source(graph.nodes, graph.outputs, 1, ((1, 5),))
        _, _, _, graph = self.graph('def f(a,x,b,y):\n return (x*a).relu()',
                                    (0.375, x, False, y))
        bridge._pointwise_source(graph.nodes, graph.outputs, 2, ((5,), (1, 5)))

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
            history = {(parsed.code, tuple(keys.items()), (), (0,)): None}
            fn.__globals__.update({name: 2.5 for name in captures})
            _, params = frontend.bind_arguments((x,)+(2.5,)*33)
            keys, values = frontend.resolve(fn, parsed, params)
            if total == 65:
                with self.assertRaisesRegex(NotImplementedError, '64 runtime scalar'):
                    frontend.runtime_bindings(parsed, keys, values, history)
            else:
                _, values, scalars = frontend.runtime_bindings(parsed, keys, values, history)
                self.assertEqual(len(scalars), 64)
                graph = frontend.lower(parsed, values, 1).graph
                source = bridge._pointwise_source(graph.nodes, graph.outputs, 1)
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
        from tests.test_compile_pointwise_jit import mock_pointwise_host_plan
        patcher = mock.patch.object(frontend._native, '_pointwise_host_plan',
                                   side_effect=mock_pointwise_host_plan)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_executor(self, tensors, nodes, output):
        run = mock.Mock()
        if self.launch_failure is not None:
            run.side_effect = self.launch_failure
        else:
            run.side_effect = lambda args, scalars, numerical_hint, output_order: ((nodes, tuple(scalars)),)
        return mock_pointwise_executor(run)

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

    class MutationDict(dict):
        def __init__(self, contents):
            super().__init__(contents)
            self.mutations = []

        def pop(self, key, *default):
            self.mutations.append(('pop', key))
            return super().pop(key, *default)

        def __setitem__(self, key, value):
            self.mutations.append(('set', key))
            super().__setitem__(key, value)

        def __delitem__(self, key):
            self.mutations.append(('delete', key))
            super().__delitem__(key)

    def ordered_contents(self, owner):
        def contents(mapping):
            return tuple((key, id(value)) for key, value in mapping.items())
        return (contents(owner.graphs), contents(owner.executors),
                tuple(contents(entry.lowerings) for entry in owner.graphs.values()),
                contents(owner.prepared), owner.prepared_bytes)

    def test_newest_hits_do_not_mutate_any_recency_map_or_recompile(self):
        fn = program('def f(scale,x):\n return x*scale')
        compiled = frontend.implementation(fn, recompile_limit=8)
        x = self.argument((2,))
        expected = compiled(0.375, x)
        owner = cache(compiled)
        owner.graphs = self.MutationDict(owner.graphs)
        owner.executors = self.MutationDict(owner.executors)
        owner.prepared = self.MutationDict(owner.prepared)
        entry = next(iter(owner.graphs.values()))
        entry.lowerings = self.MutationDict(entry.lowerings)
        before = self.ordered_contents(owner)
        with mock.patch.object(frontend, 'analyze', side_effect=AssertionError('warm analyze')), \
                mock.patch.object(frontend, 'lower', side_effect=AssertionError('warm lower')):
            for _ in range(3):
                self.assertEqual(compiled(0.375, x), expected)
        self.assertEqual(self.compile_bridge.call_count, 1)
        self.assertEqual(self.ordered_contents(owner), before)
        for mapping in (owner.graphs, owner.executors, owner.prepared, entry.lowerings):
            self.assertEqual(mapping.mutations, [])
        # Code objects can compare equal while identity requires a new parsed
        # program. An equal newest key must not suppress replacing its entry.
        previous_code = fn.__code__
        fn.__code__ = previous_code.replace()
        self.assertIsNot(fn.__code__, previous_code)
        self.assertEqual(fn.__code__, previous_code)
        self.assertEqual(compiled(0.375, x), expected)
        self.assertIs(next(iter(owner.graphs))[0], fn.__code__)
        self.assertIsNot(next(iter(owner.graphs.values())), entry)
        self.assertEqual(len(owner.graphs), 1)
        with mock.patch.object(frontend, 'analyze', side_effect=AssertionError('warm analyze')), \
                mock.patch.object(frontend, 'lower', side_effect=AssertionError('warm lower')):
            self.assertEqual(compiled(0.375, x), expected)
        self.assertEqual(self.compile_bridge.call_count, 1)

    def test_older_logical_revisit_updates_recency_with_an_already_newest_shared_executor(self):
        compiled = self.compiled()
        x, singleton = self.argument((2,)), self.argument((1,))
        compiled(0.375, x)
        compiled(0.375, singleton)
        owner = cache(compiled)
        first, second = tuple(owner.graphs)
        entries = tuple(owner.graphs.values())
        graphs = [next(iter(entry.lowerings.values())) for entry in entries]
        self.assertIsNot(graphs[0], graphs[1])
        self.assertEqual(graphs[0], graphs[1])
        self.assertEqual(len(owner.executors), 1)
        owner.executors = self.MutationDict(owner.executors)
        lowerings = [tuple(entry.lowerings.items()) for entry in entries]
        compiled(0.375, x)
        self.assertEqual(tuple(owner.graphs), (second, first))
        self.assertEqual(owner.executors.mutations, [])
        self.assertEqual([tuple(entry.lowerings.items()) for entry in entries], lowerings)
        self.assertEqual(self.compile_bridge.call_count, 1)

    def test_lowering_recency_and_eviction_are_independent_of_shared_executor_recency(self):
        compiled = frontend.implementation(program(
            'def f(a,x,b):\n local=a\n other=b\n return x*1.125'), recompile_limit=2)
        x, other, singleton = (self.argument(shape) for shape in ((2,), (2,), (1,)))
        compiled(False, x, 0.)
        owner = cache(compiled)
        first = next(iter(owner.graphs))
        entry = owner.graphs[first]
        compiled(False, x, other)
        abi_a, abi_b = tuple(entry.lowerings)
        executable_a, executable_b = tuple(owner.executors)
        # Another logical specialization reuses A's executable, making it
        # newest without visiting the first specialization's lowering A.
        compiled(False, singleton, 0.)
        second = next(reversed(owner.graphs))
        self.assertEqual(tuple(entry.lowerings), (abi_a, abi_b))
        self.assertEqual(tuple(owner.executors), (executable_b, executable_a))
        owner.executors = self.MutationDict(owner.executors)
        compiled(False, x, 0.)
        self.assertEqual(tuple(owner.graphs), (second, first))
        self.assertEqual(tuple(entry.lowerings), (abi_b, abi_a))
        self.assertEqual(tuple(owner.executors), (executable_b, executable_a))
        self.assertEqual(owner.executors.mutations, [])
        compiled(other, x, False)
        abi_c, executable_c = next(reversed(entry.lowerings)), next(reversed(owner.executors))
        self.assertNotIn(abi_c, (abi_a, abi_b))
        self.assertNotIn(executable_c, (executable_a, executable_b))
        self.assertEqual(tuple(entry.lowerings), (abi_a, abi_c))
        self.assertEqual(tuple(owner.executors), (executable_a, executable_c))
        self.assertEqual(tuple(owner.graphs), (second, first))
        self.assertEqual(self.compile_bridge.call_count, 3)

    def test_graph_hash_collisions_keep_structural_equality_and_executor_sharing(self):
        compiled = self.compiled('def f(enabled,x):\n return x*enabled')
        x, singleton = self.argument((2,)), self.argument((1,))
        with mock.patch.object(frontend.Graph, '__hash__', return_value=7):
            disabled = compiled(False, x)
            enabled = compiled(True, x)
            self.assertNotEqual(disabled, enabled)
            self.assertEqual(compiled(False, singleton), disabled)
            owner = cache(compiled)
            graphs = [next(iter(entry.lowerings.values())) for entry in owner.graphs.values()]
            self.assertEqual([hash(lowering.graph) for lowering in graphs], [7, 7, 7])
            self.assertIsNot(graphs[0], graphs[2])
            self.assertEqual(graphs[0], graphs[2])
            self.assertNotEqual(graphs[0], graphs[1])
            self.assertEqual(len(owner.executors), 2)
            self.assertEqual(compiled(True, x), enabled)
            self.assertEqual(compiled(False, x), disabled)
            self.assertEqual(self.compile_bridge.call_count, 2)

    def test_unused_role_changes_validate_the_entire_filtered_tensor_abi(self):
        compiled = frontend.implementation(program(
            'def f(a,x,b):\n local=a\n other=b\n return -x'), recompile_limit=1)
        x, unused = self.argument((2,)), self.argument((2,))
        history = (((False, x, 0.), (x,), 0),
                   ((unused, x, False), (unused, x), 1),
                   ((True, x, unused), (x, unused), 0))
        with mock.patch.object(frontend._native, '_pointwise_validate_inputs') as validate:
            for args, tensors, input_index in history:
                for _ in range(2):
                    nodes, _ = compiled(*args)
                    validate.assert_called_with(tensors)
                    output = nodes[-1]
                    self.assertEqual(output[0], 'neg')
                    self.assertEqual(nodes[output[1]][:2], ('input', input_index))
                    executor = next(reversed(cache(compiled).executors.values()))
                    executor.run.assert_called_with(tensors, (), 2, (0,))
            before = self.ordered_contents(cache(compiled))
            validate.side_effect = RuntimeError('unused tensor admission failure')
            with self.assertRaisesRegex(RuntimeError, 'unused tensor admission failure'):
                compiled(unused, x, False)
            validate.assert_called_with((unused, x))
            self.assertEqual(self.ordered_contents(cache(compiled)), before)
        self.assertEqual(len(cache(compiled).graphs), 1)

    def test_failed_older_revisits_preserve_ordered_contents_before_reset(self):
        compiled = self.compiled('def f(a,x):\n local=a\n return x*1.125')
        x, unused, singleton = (self.argument(shape) for shape in ((2,), (2,), (1,)))
        compiled(False, x)
        compiled(False, singleton)
        owner = cache(compiled)
        before = self.ordered_contents(owner)
        executor = next(iter(owner.executors.values()))
        successful_run = executor.run.side_effect
        executor.run.side_effect = RuntimeError('cached launch failure')
        with self.assertRaisesRegex(RuntimeError, 'cached launch failure'):
            compiled(False, x)
        self.assertEqual(self.ordered_contents(owner), before)
        executor.run.side_effect = successful_run
        self.compile_bridge.side_effect = RuntimeError('new ABI compile failure')
        with self.assertRaisesRegex(RuntimeError, 'new ABI compile failure'):
            compiled(unused, x)
        self.assertEqual(self.ordered_contents(owner), before)
        self.compile_bridge.side_effect = self.make_executor
        self.launch_failure = RuntimeError('new ABI launch failure')
        with self.assertRaisesRegex(RuntimeError, 'new ABI launch failure'):
            compiled(unused, x)
        self.assertEqual(self.ordered_contents(owner), before)
        self.launch_failure = None
        compiled(unused, x)
        self.assertEqual(tuple(owner.graphs), tuple(reversed(tuple(key for key, _ in before[0]))))
        native.compiler.reset()
        self.assertEqual(self.ordered_contents(owner), ((), (), (), (), 0))
        compiled(False, x)
        self.assertEqual(len(owner.graphs), 1)
        self.assertEqual(len(owner.executors), 1)

    def test_nan_payloads_share_static_guard_without_rewriting_frozen_values(self):
        for origin in ('parameter', 'global', 'closure'):
            for reverse in (False, True):
                fn, setter = scalar_source(origin)
                compiled = native.compile(fn)
                values = signed_payload_nans()[::(-1 if reverse else 1)]
                x = self.argument((2,))
                initial = None
                for value in values:
                    with self.subTest(origin=origin, bits=frontend.scalar_bits(value)):
                        key, retained = frontend.binding(value)
                        self.assertEqual(key, frontend.binding(values[0])[0])
                        self.assertEqual(frontend.scalar_bits(retained), frontend.scalar_bits(value))
                        setter(value)
                        args = (value, x) if origin == 'parameter' else (x,)
                        cold = compiled(*args)
                        self.assertEqual(compiled(*args), cold)
                        if initial is None:
                            initial = cold
                        self.assertEqual(cold, initial)
                owner = cache(compiled)
                self.assertIn(('constant', 0, 0, frontend.scalar_bits(values[0])), initial[0])
                self.assertEqual(len(owner.graphs), 1)
                self.assertEqual(len(owner.executors), 1)
                frozen = next(iter(owner.graphs.values()))
                scalar = next(v for v in frozen.values.values() if type(v) is float)
                self.assertEqual(frontend.scalar_bits(scalar), frontend.scalar_bits(values[0]))
                # NaN history promotes the next finite binding. Later NaNs hit
                # that runtime graph, retaining their actual ABI bits.
                for value in (16777217., 16777218., *values):
                    setter(value)
                    args = (value, x) if origin == 'parameter' else (x,)
                    nodes, scalars = compiled(*args)
                    self.assertEqual(len(scalars), 1)
                    self.assertEqual(frontend.scalar_bits(scalars[0]), frontend.scalar_bits(value))
                    self.assertEqual(len(owner.graphs), 2)
                self.assertEqual(len(owner.executors), 2)
                native.compiler.reset()
                self.assertFalse(owner.graphs)
                self.assertFalse(owner.executors)

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
