"""Root literal-range normalization; mock cache tests make no CUDA claims."""
import builtins
import dis
import types
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests import test_compile_pointwise_helpers as helper_tests
from tests.test_compile_pointwise_helpers import no_bodies
from tests.test_compile_pointwise_jit import available, cache, lower, program, two_device_reservation


class LoopAdmission(unittest.TestCase):
    def test_range_forms_signed_steps_and_final_index(self):
        for args in ((0,), (-3,), (1,), (7,), (2, 2), (2, 3), (-4, 5),
                     (-5, 9, 3), (9, -4, -3), (-4, 9, -3), (4, 4, -1)):
            bounds = ', '.join(map(str, args))
            fn = program(f'def f(x):\n i = 13\n for i in alias({bounds}):\n  x = x + i\n return x * i', alias=range)
            lines = ['def f(x):', ' i = 13']
            for i in range(*args):
                lines.extend((f' i = {i}', ' x = x + i'))
            lines.append(' return x * i')
            with self.subTest(args=args), no_bodies(fn):
                self.assertEqual(lower(fn), lower(program('\n'.join(lines))))
                parsed = frontend.analyze(fn, 1)
                self.assertIs(parsed.code, fn.__code__)
                self.assertEqual(parsed.range_sources, (frontend.BindingSource('LOAD_GLOBAL', 'alias'),))

    def test_sequential_loops_carry_locals_and_overwrite_index(self):
        fn = program('def f(x):\n y = -x\n for i in range(2):\n  y = y+x\n  i = 7\n for j in range(4,0,-2):\n  y = y*j\n return y+i+j')
        inline = program('def f(x):\n y=-x\n y=y+x\n y=y+x\n y=y*4\n y=y*2\n return y+7+2')
        self.assertEqual(lower(fn), lower(inline))

    def test_jump_targets_on_noops_and_extended_arguments(self):
        for body in ('  x=-x\n', '  x=-x\n'*100):
            fn = program('def f(x):\n for i in range(2):\n'+body+' pass\n return x')
            expected = program('def f(x):\n'+body.replace('  ', ' ')*2+' return x')
            with no_bodies(fn): self.assertEqual(lower(fn), lower(expected))

    def test_zero_trip_preserves_initial_parameter_and_previous_index(self):
        for source, inline in (
            ('for x in range(0):\n  x = -x', ''),
            ('for i in range(0):\n  x = -x', ''),
            ('i=3\n for i in range(0):\n  x=-x\n x=x+i', 'x=x+3'),
            ('for i in range(1):\n  x=-x\n for i in range(0):\n  x=x+i\n x=x+i', 'x=-x\n x=x+0'),
        ):
            fn = program('def f(x):\n '+source+'\n return x+1')
            expected = program('def f(x):\n '+(inline+'\n ' if inline else '')+'return x+1')
            self.assertEqual(lower(fn), lower(expected))
        for name in ('i', 'missing'):
            fn = program(f'def f(x):\n for i in range(0):\n  missing=x\n return x+{name}')
            with self.assertRaisesRegex(NotImplementedError, 'unbound local'):
                lower(fn)
        with self.assertRaisesRegex(NotImplementedError, 'computed pointwise tensor'):
            lower(program('def f(x):\n for i in range(0):\n  x=-x\n return x'))

    def test_overwritten_parameter_is_not_an_initial_dependency(self):
        fn = program('def f(x, scale):\n for i in range(2):\n  scale=i\n return x*scale')
        parsed = frontend.analyze(fn, 2)
        self.assertEqual([s.name for s in parsed.dependencies], ['x'])
        fn = program('def f(x, scale):\n for i in range(0):\n  scale=i\n return x*scale')
        self.assertEqual([s.name for s in frontend.analyze(fn, 2).dependencies], ['x', 'scale'])

    def test_all_structure_rejected_before_expansion_even_zero_trip(self):
        bodies = ('if x:\n   x=-x', 'for j in range(2):\n   x=-x',
                  'return -x', 'break', 'x += x', 'x[0]=1', 'x=x/2')
        for body in bodies:
            fn = program('def f(x):\n for i in range(0):\n  '+body+'\n return -x')
            with self.subTest(body=body), self.assertRaises(NotImplementedError):
                lower(fn)
        helper = program('def f(x):\n for i in range(2):\n  x=-x\n return x')
        with self.assertRaises(NotImplementedError):
            lower(program('def f(x):\n return helper(x)', helper=helper))

    def test_unsupported_iterators_bounds_and_calls(self):
        for expr in ('range(n)', 'range(1,n)', 'range(1,4,n)', 'range(True)',
                     'range(2.0)', 'range(1,4,0)', 'range()', 'range(1,2,3,4)',
                     'iter(range(2))', 'reversed(range(2))', '(1,2)', 'iterator',
                     'factory()(2)', 'range(stop=2)'):
            fn = program(f'def f(x):\n for i in {expr}:\n  x=-x\n return x', n=2, iterator=object(), factory=lambda: range)
            with self.subTest(expr=expr), self.assertRaises(NotImplementedError):
                lower(fn)
        n = 2
        def closed(x):
            for i in range(n):
                x = -x
            return x
        with self.assertRaises(NotImplementedError):
            lower(closed)
        # Constant folding is bytecode semantics, not a source-spelling promise.
        self.assertEqual(lower(program('def f(x):\n for i in range(1+1):\n  x=-x\n return x')),
                         lower(program('def f(x):\n return -(-x)')))

    def test_zero_trip_body_operator_helper_and_data_admission(self):
        helpers = (
            program('def f(a):\n for j in range(2):\n  a=-a\n return a'),
            program('def f(a):\n return a.sum()'),
            program('def f(a, b):\n return a+b'),
        )
        for expression, bindings in (
            ('x.sum()', {}), ('fw.sum(x)', {'fw': native}), ('i+i', {}),
            ('helper(x)', {'helper': object()}),
            *(('helper(x)', {'helper': helper}) for helper in helpers),
            ('helper(x, helper)', {'helper': program('def f(a, ignored):\n return a')}),
            ('helper(x, fw)', {'helper': program('def f(a, ignored):\n return a'), 'fw': native}),
        ):
            fn = program('def f(x):\n for i in range(0):\n  x='+expression+'\n return -x', **bindings)
            bodies = tuple(v for v in bindings.values() if type(v) is types.FunctionType)
            with self.subTest(expression=expression), no_bodies(fn, *bodies):
                with self.assertRaises(NotImplementedError): lower(fn)

    def test_zero_trip_validation_does_not_change_locals_or_ir(self):
        helper = program('def f(a, ignored):\n return a.sin()')
        fn = program('def f(x, scale):\n saved=x\n for i in range(0):\n  x=helper(x, scale)\n  saved=x\n  scale=i\n for j in range(2):\n  saved=saved*scale\n return saved+x', helper=helper)
        expected = program('def f(x, scale):\n return x*scale*scale+x')
        with no_bodies(fn, helper):
            self.assertEqual(lower(fn, 2), lower(expected, 2))

    def test_hostile_values_containers_keys_and_constant_pool(self):
        effects = []
        class Hostile:
            def __call__(self, *args): effects.append('call'); return range(2)
            def __index__(self): effects.append('index'); return 2
            def __iter__(self): effects.append('iter'); return iter(())
            def __eq__(self, other): effects.append('eq'); return False
            def __repr__(self): effects.append('repr'); return 'hostile'
            def __hash__(self): return hash('alias')
        class Mapping(dict):
            def __getitem__(self, key): effects.append('getitem'); return super().__getitem__(key)
            def __iter__(self): effects.append('mapping iter'); return super().__iter__()
        fn = program('def f(x):\n for i in alias(0):\n  x=-x\n return -x', alias=range)
        for replacement in (Hostile(), 2, lambda n: range(n)):
            fn.__globals__['alias'] = replacement
            with self.assertRaises(NotImplementedError), no_bodies(fn): lower(fn)
        for expression in ('range(bound)', 'bound', 'bound(2)'):
            bad = program(f'def f(x):\n for i in {expression}:\n  x=-x\n return x', bound=Hostile())
            with self.assertRaises(NotImplementedError): lower(bad)
        fn.__globals__['alias'] = range
        namespaces = (Mapping(alias=range), {'alias': range, Hostile(): None})
        effects.clear()  # Constructing the deliberately colliding dict calls equality.
        for namespace in namespaces:
            for which in ('globals', 'builtins'):
                ns = namespace if which == 'globals' else {'alias': range, '__builtins__': namespace}
                bad = types.FunctionType(fn.__code__, ns)
                with patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('disassembly before containers')):
                    with self.assertRaises(NotImplementedError): lower(bad)
        fn.__code__ = fn.__code__.replace(co_consts=fn.__code__.co_consts + (Hostile(),))
        with patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('disassembly before constants')):
            with self.assertRaises(NotImplementedError): lower(fn)
        self.assertEqual(effects, [])

    def test_limits_precede_expansion_and_helpers_share_budget(self):
        fn = program('def f(x):\n for i in range(-9223372036854775808,18446744073709551615):\n  x=-x\n return x')
        with patch.object(dis.Instruction, '_replace', side_effect=AssertionError('expanded before bound')):
            with self.assertRaisesRegex(NotImplementedError, 'instruction limit'): lower(fn)
        # NOP is ignored by lowering, but every repeated occurrence is charged.
        fn = program('def f(x):\n for i in range(200):\n' + '  pass\n'*100 + '  x=-x\n return x')
        with self.assertRaisesRegex(NotImplementedError, 'instruction limit'): lower(fn)
        helper = program('def f(x):\n'+' y=x\n'*100+' return -x')
        for count, accepted in ((40, True), (90, False)):
            fn = program(f'def f(x):\n for i in range({count}):\n  x=helper(x)\n return x', helper=helper)
            if accepted: lower(fn)
            else:
                with self.assertRaisesRegex(NotImplementedError, 'instruction limit'): lower(fn)
        helper = program('def f(x):\n'+' x=-x\n'*700+' return x')
        fn = program('def f(x):\n for i in range(6):\n  x=helper(x)\n return x', helper=helper)
        with self.assertRaisesRegex(NotImplementedError, '4096-node limit'): lower(fn)

    def test_zero_trip_validation_shares_helper_and_node_budgets(self):
        helper = program('def f(x):\n'+' y=x\n'*100+' return -x')
        fn = program('def f(x):\n for i in range(0):\n'+'  x=helper(x)\n'*90+' return -x', helper=helper)
        with self.assertRaisesRegex(NotImplementedError, 'instruction limit'): lower(fn)
        helper = program('def f(x):\n'+' x=-x\n'*700+' return x')
        for second in ('for j in range(0):', 'for j in range(1):'):
            fn = program('def f(x):\n for i in range(0):\n'+'  x=helper(x)\n'*3+
                         ' '+second+'\n'+'  x=helper(x)\n'*3+' return -x', helper=helper)
            with self.subTest(second=second), self.assertRaisesRegex(NotImplementedError, '4096-node limit'):
                lower(fn)

    def test_malformed_exit_backedge_and_body_stack(self):
        fn = program('def f(x):\n for i in range(2):\n  x=-x\n return x')
        instructions = frontend.instructions_for(fn.__code__)
        for op, change in (('FOR_ITER', {'argval': -10}),
                           ('JUMP_BACKWARD', {'argval': -10}),
                           ('JUMP_ABSOLUTE', {'argval': -10}),
                           ('STORE_FAST', {'opname': 'COPY', 'arg': 2}),
                           ('END_FOR', {'opname': 'NOP'}),
                           ('POP_TOP', {'opname': 'NOP'}),
                           ('POP_ITER', {'opname': 'NOP'})):
            if not any(i.opname == op for i in instructions): continue
            replaced = tuple(i._replace(**change) if i.opname == op else i for i in instructions)
            with patch.object(frontend, 'instructions_for', return_value=replaced):
                with self.subTest(op=op), self.assertRaises(NotImplementedError): frontend.analyze(fn, 1)


class LoopCache(unittest.TestCase):
    setUp = helper_tests.HelperCache.setUp

    def test_zero_trip_cold_rejections_publish_nothing(self):
        helper = program('def f(a):\n for j in range(2):\n  a=-a\n return a')
        for expression in ('x.sum()', 'helper(x)'):
            fn = program('def f(x):\n for i in range(0):\n  x='+expression+'\n return -x', helper=helper)
            compiled = native.compile(fn)
            with (no_bodies(fn, helper), patch.object(bridge, '_pointwise_compile', side_effect=AssertionError('compile')),
                  self.assertRaises(NotImplementedError)):
                compiled(self.x)
            self.assertFalse(cache(compiled).graphs)
            self.assertFalse(cache(compiled).executors)

    def test_zero_trip_warm_helper_guards_rejection_and_recovery(self):
        effects = []
        class Hostile:
            def __call__(self, *args): effects.append('call')
            def __eq__(self, other): effects.append('eq'); return False
            def __repr__(self): effects.append('repr'); return 'hostile'
        helper = program('def f(a, ignored):\n return a.sin()')
        original = helper.__code__
        fn = program('def f(x):\n for i in range(0):\n  x=helper(x, captured)\n return -x', helper=helper, captured=0.5)
        compiled = native.compile(fn)
        with no_bodies(fn, helper): compiled(self.x)
        state = cache(compiled)
        def snapshot():
            return (list(state.graphs.items()), list(state.executors.items()),
                    [list(entry.lowerings.items()) for entry in state.graphs.values()])
        before = snapshot()
        for name, replacement in (('helper', Hostile()), ('captured', native.sin),
                                  ('captured', native), ('captured', helper),
                                  ('captured', Hostile()), ('captured', 1 << 80)):
            saved = fn.__globals__[name]
            fn.__globals__[name] = replacement
            with (patch.object(frontend, 'lower', side_effect=AssertionError('warm lower')),
                  patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('warm dis')),
                  patch.object(next(iter(state.executors.values())), 'run', side_effect=AssertionError('launch')),
                  self.assertRaises(NotImplementedError)):
                compiled(self.x)
            self.assertEqual(snapshot(), before)
            fn.__globals__[name] = saved
        for body in ('return a.sum()', 'for j in range(2):\n  a=-a\n return a'):
            helper.__code__ = program('def f(a, ignored):\n '+body).__code__
            with (no_bodies(fn, helper), patch.object(bridge, '_pointwise_compile', side_effect=AssertionError('compile')),
                  patch.object(next(iter(state.executors.values())), 'run', side_effect=AssertionError('launch')),
                  self.assertRaises(NotImplementedError)):
                compiled(self.x)
            self.assertEqual(snapshot(), before)
        helper.__code__ = original
        fn.__globals__['captured'] = 1.5  # Valid ignored data needs no value guard.
        with (no_bodies(fn, helper), patch.object(frontend, 'lower', side_effect=AssertionError('recovery lower')),
              patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('recovery dis'))):
            compiled(self.x)
        self.assertEqual(effects, [])
        native.compiler.reset()
        with no_bodies(fn, helper): compiled(self.x)

    def test_direct_closure_range_identity_and_empty_cell(self):
        alias = range
        def fn(x):
            for i in alias(2):
                x = -x
            return x
        compiled = native.compile(fn)
        compiled(self.x)
        for invalid in (lambda n: range(n), None):
            alias = invalid
            with self.assertRaises(NotImplementedError): compiled(self.x)
        del alias
        with self.assertRaises(NotImplementedError): compiled(self.x)
        alias = range
        with (patch.object(frontend, 'lower', side_effect=AssertionError('warm lower')),
              patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('warm dis'))):
            compiled(self.x)

    def test_warm_builtin_keys_and_unused_constant_pool_are_callback_free(self):
        effects = []
        class Key:
            def __hash__(self): return hash('range')
            def __eq__(self, other): effects.append('eq'); return False
            def __repr__(self): effects.append('repr'); return 'key'
        fn = program('def f(x):\n for i in range(0):\n  x=-x\n return -x',
                     __builtins__=dict(vars(builtins)))
        compiled = native.compile(fn)
        compiled(self.x)
        # A present global always shadows builtins, even if it is the frontend's
        # private missing sentinel. Lookup must test presence, not sentinel value.
        fn.__globals__['range'] = frontend._MISSING
        with self.assertRaises(NotImplementedError): compiled(self.x)
        del fn.__globals__['range']
        key = Key()
        fn.__builtins__[key] = None
        effects.clear()  # Fixture insertion intentionally collides.
        with patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('warm dis')):
            with self.assertRaises(NotImplementedError): compiled(self.x)
        self.assertEqual(effects, [])
        del fn.__builtins__[key]
        original = fn.__code__
        fn.__code__ = original.replace(co_consts=original.co_consts + (Key(),))
        effects.clear()
        with patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('unsafe dis')):
            with self.assertRaises(NotImplementedError): compiled(self.x)
        self.assertEqual(effects, [])
        fn.__code__ = original
        with patch.object(frontend, 'lower', side_effect=AssertionError('valid recovery lowered')):
            compiled(self.x)

    def test_warm_range_identity_rejection_recovery_and_atomicity(self):
        for trips in (0, 1, 4):
            for alias in (False, True):
                name = 'alias' if alias else 'range'
                ns = {'__builtins__': dict(vars(builtins))}
                if alias: ns[name] = range
                fn = program(f'def f(x):\n for i in {name}({trips}):\n  x=-x\n return x+1', **ns)
                compiled = native.compile(fn)
                with no_bodies(fn): compiled(self.x)
                state = cache(compiled)
                before = (list(state.graphs.items()), list(state.executors.items()),
                          [list(e.lowerings.items()) for e in state.graphs.values()])
                for where in (fn.__globals__, fn.__builtins__) if not alias else (fn.__globals__,):
                    saved = where.get(name)
                    where[name] = lambda n: range(n)
                    with (patch.object(frontend, 'lower', side_effect=AssertionError('lower')),
                          patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('dis')),
                          patch.object(next(iter(state.executors.values())), 'run', side_effect=AssertionError('launch')),
                          self.assertRaises(NotImplementedError)):
                        compiled(self.x)
                    self.assertEqual(before, (list(state.graphs.items()), list(state.executors.items()),
                                              [list(e.lowerings.items()) for e in state.graphs.values()]))
                    if saved is None: del where[name]
                    else: where[name] = saved
                with (patch.object(frontend, 'lower', side_effect=AssertionError('warm lower')),
                      patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('warm dis'))):
                    compiled(self.x)
                native.compiler.reset()
                self.assertFalse(state.graphs)
                compiled(self.x)

    def test_helper_code_freezing_new_abi_and_scalar_history(self):
        helper = program('def f(a, unused):\n return a')
        original = helper.__code__
        fn = program('def f(x, scale, unused):\n for i in range(2):\n  x=helper(x*scale, unused)\n return x', helper=helper)
        compiled = native.compile(fn)
        for scale, scalars in ((0.25, ()), (0.5, (0.5,)), (0.25, (0.25,))):
            with no_bodies(fn, helper): self.assertEqual(compiled(self.x, scale, 1.0)[2], scalars)
        helper.__code__ = program('def f(a, unused):\n return -a').__code__
        compiled(self.x, 0.25, 1.0)
        fn.__globals__['helper'] = types.FunctionType(original, {})
        result = compiled(self.x, 0.25, self.x)
        self.assertFalse(any(n[0] == 'neg' for n in result[0]))
        native.compiler.reset()
        self.assertEqual(compiled(self.x, 0.25, 1.0)[2], ())

    def test_ignored_helper_argument_remains_validated(self):
        helper = program('def f(a, ignored):\n return -a')
        fn = program('def f(x):\n for i in range(2):\n  x=helper(x, captured)\n return x', helper=helper, captured=0.5)
        compiled = native.compile(fn)
        compiled(self.x)
        fn.__globals__['captured'] = native.sin
        with patch.object(frontend, 'lower', side_effect=AssertionError('warm lower')):
            with self.assertRaises(NotImplementedError): compiled(self.x)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class LoopHardware(unittest.TestCase):
    def tearDown(self): native.compiler.reset()

    def test_zero_trip_helper_validation_and_warm_recovery(self):
        helper = program('def f(a):\n return a.sin()')
        original = helper.__code__
        fn = program('def f(x):\n for i in range(0):\n  x=helper(x)\n return -x', helper=helper)
        compiled = native.compile(fn)
        x = native.tensor([1.0, -2.0]).to('cuda:0')
        with no_bodies(fn, helper):
            self.assertEqual(compiled(x).cpu().tolist(), [-1.0, 2.0])
        helper.__code__ = program('def f(a):\n return a.sum()').__code__
        with no_bodies(fn, helper), self.assertRaises(NotImplementedError): compiled(x)
        fn.__globals__['helper'] = object()
        with self.assertRaises(NotImplementedError): compiled(x)
        helper.__code__ = original
        fn.__globals__['helper'] = helper
        with (no_bodies(fn, helper), patch.object(frontend, 'lower', side_effect=AssertionError('warm lower')),
              patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('warm dis'))):
            self.assertEqual(compiled(x).cpu().tolist(), [-1.0, 2.0])
        self.assertEqual(x.cpu().tolist(), [1.0, -2.0])

    def test_default_inductor_offsets_histories_ieee_and_inputs(self):
        import torch
        for source in (
            'def f(x):\n for i in range(3):\n  x=(x+0.125).relu()\n return x',
            'def f(x):\n for i in range(3,-2,-2):\n  x=x*i\n return x',
            'def f(x):\n for i in range(0):\n  x=-x\n return x+0.25',
        ):
            fn = program(source)
            compiled, reference = native.compile(fn), torch.compile(program(source, framework=torch))
            for size, offset in ((7, 1), (13, 2), (13, 3), (7, 2), (0, 1)):
                data = [float('nan'), float('inf'), -float('inf'), -0., 0., 0.25, -0.75]*4
                x = native.tensor(data, dtype=native.float32).to('cuda:0')[offset:offset+size]
                tx = torch.tensor(data, device='cuda:0')[offset:offset+size]
                before = x.cpu().tolist()
                with no_bodies(fn): actual = compiled(x)
                expected = reference(tx)
                torch.testing.assert_close(torch.tensor(actual.cpu().tolist()), expected.cpu(), equal_nan=True)
                torch.testing.assert_close(torch.tensor(x.cpu().tolist()), torch.tensor(before), equal_nan=True)
                self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                self.assertEqual(tuple(actual.stride()), tuple(expected.stride()))

    def test_original_ir_unequal_shapes_and_device_restoration(self):
        fn = program('def f(x,y):\n for i in range(2):\n  x=x+y\n return x')
        compiled = native.compile(fn)
        x = native.tensor([[1., 2.], [3., 4.]]).to('cuda:0')
        y = native.tensor([1., 2.]).to('cuda:0')
        with self.assertRaises(NotImplementedError): compiled(x,y)
        self.assertFalse(cache(compiled).graphs)
        if not two_device_reservation() or native.cuda.device_count() < 2:
            self.skipTest('requires two explicitly reserved GPUs for restoration')
        import torch
        original = torch.cuda.current_device()
        try:
            torch.cuda.set_device(1)
            compiled(x,x)
            self.assertEqual(torch.cuda.current_device(), 1)
            with self.assertRaises(NotImplementedError): compiled(x,y)
            self.assertEqual(torch.cuda.current_device(), 1)
            native.compiler.reset()
            self.assertEqual(torch.cuda.current_device(), 1)
        finally:
            torch.cuda.set_device(original)


if __name__ == '__main__': unittest.main()
