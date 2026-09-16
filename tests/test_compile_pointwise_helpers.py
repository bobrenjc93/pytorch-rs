"""Direct helper admission and frozen specialization semantics; no body replay."""
from contextlib import ExitStack, contextmanager, nullcontext
import dis
import gc
import sys
import types
import unittest
import weakref
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests.test_compile_pointwise_jit import available, cache, lower, mock_pointwise_executor, program, two_device_reservation


@contextmanager
def no_bodies(*functions):
    codes = tuple(fn.__code__ for fn in functions)
    previous = sys.getprofile()
    def forbid(frame, event, arg):
        if event == 'call' and any(frame.f_code is code for code in codes):
            raise AssertionError('original Python body executed')
    sys.setprofile(forbid)
    try:
        yield
    finally:
        sys.setprofile(previous)


def root(helper, expression='helper(x)', signature='x', **bindings):
    return program(f'def f({signature}):\n return {expression}', helper=helper, **bindings)


def check_ignored_capture_admission(test, x):
    """Run the same warm/fresh rejection history with mocked or real execution."""
    for closure in (False, True):
        helper = program('def f(a, ignored):\n return -a')
        identity = program('def f(a):\n return a')
        captured = 0.5
        if closure:
            def fn(x):
                saved = captured
                return helper(x, identity(saved))
            cell = dict(zip(fn.__code__.co_freevars, fn.__closure__))['captured']
            def replace(value):
                cell.cell_contents = value
        else:
            fn = root(helper, 'helper(x, captured)', captured=captured)
            def replace(value):
                fn.__globals__['captured'] = value
        compiled = native.compile(fn)
        with no_bodies(fn, helper, identity):
            compiled(x)
        state = cache(compiled)
        # Leave the selected entry older than a same-graph distinct code guard,
        # so failed admission must preserve recency as well as cache contents.
        original = helper.__code__
        helper.__code__ = original.replace()
        compiled(x)
        helper.__code__ = original
        executor = next(iter(state.executors.values()))
        def snapshot():
            return ([(key, id(entry), tuple(entry.lowerings.items()),
                      dict(entry.values), dict(entry.observations))
                     for key, entry in state.graphs.items()],
                    list(state.executors.items()))
        before = snapshot()
        for invalid in (identity, native, native.sin, 2**100, -(2**100)):
            replace(invalid)
            with test.subTest(closure=closure, invalid_type=type(invalid).__name__):
                fresh = native.compile(fn)
                with no_bodies(fn, helper, identity), test.assertRaises(NotImplementedError):
                    fresh(x)
                test.assertFalse(cache(fresh).graphs)
                test.assertFalse(cache(fresh).executors)
                with (patch.object(frontend, 'lower', side_effect=AssertionError('warm lowering')),
                      patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('warm parsing')),
                      patch.object(executor, 'run', side_effect=AssertionError('invalid launch'))
                      if type(executor) is types.SimpleNamespace else nullcontext(),
                      no_bodies(fn, helper, identity), test.assertRaises(NotImplementedError)):
                    compiled(x)
                test.assertEqual(snapshot(), before)
        for valid in (True, 7, 2**64-1, -2**63, -0.0, float('nan'), float('inf'), 0.75):
            replace(valid)
            with (test.subTest(closure=closure, valid=valid), no_bodies(fn, helper, identity),
                  patch.object(frontend, 'lower', side_effect=AssertionError('ignored value guard')),
                  patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('warm parsing'))):
                actual = compiled(x)
            if type(actual) is native.Tensor:
                test.assertEqual(actual.cpu().tolist(), [-v for v in x.cpu().tolist()])
            test.assertEqual(len(state.graphs), 2)
            test.assertEqual(len(state.executors), 1)
        source = next(s for s in next(iter(state.graphs.values())).data_sources if s.name == 'captured')
        for entry in state.graphs.values():
            test.assertNotIn(source, entry.values)
            test.assertNotIn(source, entry.observed)
            test.assertNotIn(source, entry.observations)


class HelperAdmission(unittest.TestCase):
    def test_repeated_multiple_and_composed_calls_match_inline_graph(self):
        helper = program('def f(a, b):\n c = a - b\n return c.relu()')
        other = program('def f(a):\n return a.sin()')
        fn = root(helper, 'other(helper(x, y)) + helper(y, x)', 'x, y', other=other)
        inline = program('def f(x, y):\n return (x-y).relu().sin() + (y-x).relu()')
        with no_bodies(fn, helper, other):
            self.assertEqual(lower(fn, 2), lower(inline, 2))

    def test_closure_binding_and_positional_only_helper(self):
        helper = program('def f(a, /, b):\n return a.__rsub__(b)')
        def fn(x, y):
            return helper(x, y)
        self.assertEqual(lower(fn, 2), lower(program('def f(x, y):\n return y-x'), 2))

    def test_native_functions_keep_priority_over_python_helpers(self):
        with patch.object(frontend, 'freeze_helper', side_effect=AssertionError('native operator treated as helper')):
            key, value = frontend.binding(native.sin)
        self.assertEqual(key[0], 'function')
        self.assertEqual(value, frontend.Call('sin'))

    def test_instruction_memo_still_charges_each_complete_frame(self):
        helper = program('def f(a):\n return a')
        fn = root(helper, 'helper(x) + helper(x)')
        parsed = frontend.analyze(fn, 1)
        _, values = frontend.resolve(fn, parsed)
        base = tuple(dis.get_instructions(helper))
        nop = types.SimpleNamespace(opname='NOP', argval=None)
        # Even unreachable admitted instructions after RETURN count in full.
        count = (16384 - len(parsed.instructions)) // 2
        for size, accepted in ((count, True), (count + 1, False)):
            instructions = base + (nop,) * (size - len(base))
            with patch.object(frontend, 'instructions_for', return_value=instructions) as parse:
                if accepted:
                    frontend.lower(parsed, values, 1)
                else:
                    with self.assertRaisesRegex(NotImplementedError, 'instruction limit'):
                        frontend.lower(parsed, values, 1)
                self.assertEqual(parse.call_count, 1)

    def test_identity_and_scalar_literal_returns_feed_tensor_operations(self):
        for body in ('return a', 'b = a\n return b', 'return 0.375', 'return True', 'return 2'):
            helper = program('def f(a):\n '+body)
            fn = root(helper, 'helper(x) + x')
            with no_bodies(fn, helper):
                graph = lower(fn)
            self.assertEqual(graph.nodes[-1][0], 'add')
        helper = program('def f(a):\n return a')
        self.assertEqual(lower(root(helper, 'helper(-x)')), lower(program('def f(x):\n return -x')))
        for helper in (helper, program('def f(a):\n return 1.0')):
            with self.assertRaisesRegex(NotImplementedError, 'computed pointwise tensor'):
                lower(root(helper))

    def test_return_opcodes_share_data_boundary(self):
        # Exercise both paths on every interpreter; on 3.12 real scalar helpers
        # additionally produce RETURN_CONST without any synthetic instructions.
        for opcode in ('RETURN_VALUE', 'RETURN_CONST'):
            for value in (None, 'metadata', frontend.Call('sin'), native,
                          frontend.Helper(program('def f(x):\n return -x').__code__)):
                fn = program('def f(x):\n return -x')
                parsed = frontend.analyze(fn, 1)
                instructions = ([types.SimpleNamespace(opname='LOAD_FAST', argval='x')]
                                if opcode == 'RETURN_VALUE' else [])
                instructions.append(types.SimpleNamespace(opname=opcode, argval=value))
                source = frontend.BindingSource('parameter', 'x', 0)
                parsed = frontend.Program(parsed.code, tuple(instructions), (source,))
                with self.subTest(opcode=opcode, kind=type(value).__name__), self.assertRaises(NotImplementedError):
                    frontend.lower(parsed, {source: value}, 1)
        if 'RETURN_CONST' in dis.opmap:
            helper = program('def f(a):\n return 0.375')
            self.assertIn('RETURN_CONST', [i.opname for i in dis.get_instructions(helper)])
            lower(root(helper, 'helper(x) * x'))

    def test_metadata_is_not_data_and_no_scalar_binary_arithmetic(self):
        for body in ('return None', 'return "text"', 'return a.sin',
                     'return a + 1.0'):
            helper = program('def f(a):\n '+body)
            arg = '1.0' if body == 'return a + 1.0' else 'x'
            with self.subTest(body=body), self.assertRaises(NotImplementedError):
                lower(root(helper, f'helper({arg}) + x'))

    def test_signature_and_unsupported_bytecode(self):
        sources = (
            'def f():\n return 1.0', 'def f(a, *, b):\n return a',
            'def f(a, *args):\n return a', 'def f(a, **kwargs):\n return a',
            'def f(a=1):\n return a', 'def f(a):\n yield a',
            'async def f(a):\n return a', 'def f(a):\n return -a if a else a',
            'def f(a):\n a += 1\n return a', 'def f(a):\n a.foo = 1\n return a',
            'def f(a):\n return fw.sin(a)', 'def f(a):\n return other(a)',
            'def f(a):\n try:\n  return -a\n except Exception:\n  return a',
            'def f(a):\n def inner():\n  return a\n return a',
        )
        for source in sources:
            with self.subTest(source=source), self.assertRaises(NotImplementedError):
                lower(root(program(source)))
        helper = program('def f(a, b):\n return a+b')
        for expression in ('helper(x)', 'helper(x, x, x)', 'helper(a=x, b=x)'):
            with self.subTest(expression=expression), self.assertRaises(NotImplementedError):
                lower(root(helper, expression))

    def test_higher_order_operands_rejected_even_if_ignored(self):
        helper = program('def f(a, ignored):\n return -a')
        for expression, bindings in (
            ('helper(x, helper)', {}), ('helper(x, fw)', {}),
            ('helper(x, op)', {'op': native.sin}), ('helper(x, x.sin)', {}),
        ):
            with self.subTest(expression=expression), self.assertRaises(NotImplementedError):
                lower(root(helper, expression, **bindings))
        class Callable:
            def __call__(self, x):
                raise AssertionError('callable executed')
        with self.assertRaises(NotImplementedError):
            lower(root(Callable()))

    def test_code_identity_not_function_or_structural_code_equality(self):
        fn = program('def f(a):\n return -a')
        same = types.FunctionType(fn.__code__, {})
        equal = types.FunctionType(fn.__code__.replace(), {})
        self.assertEqual(fn.__code__, equal.__code__)
        a, b, c = [frontend.binding(f)[1] for f in (fn, same, equal)]
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, c)
        self.assertEqual(hash(a), id(fn.__code__))
        class Other:
            def __getattribute__(self, name):
                raise AssertionError('foreign equality attribute lookup')
        self.assertFalse(a == Other())
        self.assertEqual(len({a, b, c}), 2)

    def test_all_constants_validated_before_disassembly_or_retention(self):
        effects = []
        class Constant:
            def __repr__(self):
                effects.append('repr')
                return 'constant'
            def __float__(self):
                effects.append('float')
                return 1.0
        for constant in (Constant(), (1,), 2**64, -2**63-1, object(), str.__new__(type('Text', (str,), {}), 's')):
            helper = program('def f(a):\n return -a')
            helper.__code__ = helper.__code__.replace(co_consts=helper.__code__.co_consts + (constant,))
            for expression in ('helper(x)', 'x + 1.0'):
                fn = root(helper, expression)
                if expression == 'x + 1.0':
                    fn = program('def f(x):\n ignored = helper\n return x+1.0', helper=helper)
                parsed = frontend.analyze(fn, 1)
                with patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('disassembly')):
                    with self.assertRaises(NotImplementedError):
                        frontend.resolve(fn, parsed)
            self.assertEqual(effects, [])

    def test_container_and_attribute_callbacks_never_run(self):
        effects = []
        class Defaults(tuple):
            def __bool__(self):
                effects.append('bool')
                return False
            def __iter__(self):
                effects.append('iter')
                return super().__iter__()
        class Attributes(dict):
            def __iter__(self):
                effects.append('attributes iter')
                return super().__iter__()
        class Key(str):
            def __hash__(self):
                return hash('_torchdynamo_inline')
            def __eq__(self, other):
                effects.append('key equality')
                return True
        for attribute, value in (('__defaults__', Defaults()), ('__defaults__', (1,)),
                                 ('__kwdefaults__', Attributes()), ('__kwdefaults__', {'a': 1}),
                                 ('__dict__', Attributes()), ('__dict__', {Key('key'): 0})):
            helper = program('def f(a):\n return -a')
            setattr(helper, attribute, value)
            effects.clear()
            with self.subTest(attribute=attribute), self.assertRaises(NotImplementedError):
                frontend.binding(helper)
            self.assertEqual(effects, [])
        captured = 1
        def closure(a):
            return a + captured
        for cells in (closure.__closure__, Defaults(closure.__closure__)):
            helper = types.FunctionType(closure.__code__, {}, closure=cells)
            with self.assertRaises(NotImplementedError):
                frontend.binding(helper)
        self.assertEqual(effects, [])
        helper = program('def f(a):\n return -a')
        helper.__defaults__, helper.__kwdefaults__ = (), {}
        frontend.binding(helper)

    def test_directive_presence_not_truthiness(self):
        class Directive:
            def __bool__(self):
                raise AssertionError('directive truthiness')
        for name in ('_torchdynamo_inline', '_dynamo_marked_constant', '_torchdynamo_disable'):
            for value in (False, None, Directive()):
                helper = program('def f(a):\n return -a')
                helper.__dict__[name] = value
                with self.subTest(name=name), self.assertRaisesRegex(NotImplementedError, 'directives'):
                    frontend.binding(helper)

    def test_lazy_inputs_shared_sources_and_returned_bound_values(self):
        helper = program('def f(a, b):\n ignored = b\n return a')
        fn = root(helper, 'helper(x, huge) + helper(x, scale)', huge=2**64-1, scale=0.375)
        parsed = frontend.analyze(fn, 1)
        _, values = frontend.resolve(fn, parsed)
        observed = []
        graph = frontend.lower(parsed, values, 1, observed=observed).graph
        self.assertEqual(graph.nodes, (('input', 0, 0, 0), ('add', 0, 0, 0)))
        self.assertEqual([s.name for s in observed], ['helper', 'x'])
        scalar_identity = program('def f(a):\n return a')
        fn = root(scalar_identity, 'x * helper(scale) + helper(scale)', scale=0.5)
        parsed = frontend.analyze(fn, 1)
        keys, values = frontend.resolve(fn, parsed)
        scale = next(s for s in parsed.dependencies if s.name == 'scale')
        values[scale] = frontend.RuntimeScalar(0)
        observed = []
        graph = frontend.lower(parsed, values, 1, observed=observed).graph
        self.assertEqual([node for node in graph.nodes if node[0] == 'scalar'], [('scalar', 0, 0, 0)])
        self.assertEqual(observed.count(scale), 1)

    def test_expanded_instruction_and_node_budgets(self):
        helper = program('def f(a):\n' + ' b = a\n' * 100 + ' return a')
        for calls, accepted in ((60, True), (90, False)):
            fn = program('def f(x):\n' + ' y = helper(x)\n' * calls + ' return -x', helper=helper)
            with self.subTest(calls=calls):
                if accepted:
                    lower(fn)
                else:
                    with self.assertRaisesRegex(NotImplementedError, 'instruction limit'):
                        lower(fn)
        helper = program('def f(a):\n' + ' a = -a\n' * 700 + ' return a')
        fn = program('def f(x):\n' + ' x = helper(x)\n' * 6 + ' return x', helper=helper)
        with self.assertRaisesRegex(NotImplementedError, '4096-node limit'):
            lower(fn)
        helper = program('def f(a):\n' + ' b = a\n' * 9000 + ' return a')
        with self.assertRaisesRegex(NotImplementedError, 'instruction limit'):
            lower(root(helper))


class HelperCache(unittest.TestCase):
    """Real frontend/cache with a fake native bridge; makes no CUDA claims."""
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(native.compiler.reset)
        metadata = bridge._compile_trace_tensor_metadata
        def fake_metadata(tensor):
            result = list(metadata(tensor))
            result[4] = 'cuda:0'
            return tuple(result)
        self.stack.enter_context(patch.object(bridge, '_compile_trace_tensor_metadata', fake_metadata))
        self.validate = self.stack.enter_context(patch.object(bridge, '_pointwise_validate_inputs'))
        def compile_(tensors, nodes, output):
            # Use the actual hardware-free native IR/codegen boundary.
            bridge._pointwise_source(nodes, output, len(tensors))
            return mock_pointwise_executor(lambda tensors, scalars, numerical_hint, output_order: ((nodes, output, scalars),))
        self.codegen = self.stack.enter_context(patch.object(bridge, '_pointwise_compile', side_effect=compile_))
        self.x = native.tensor([1.0, -2.0])

    def test_ignored_global_and_closure_arguments_revalidated_on_warm_hits(self):
        check_ignored_capture_admission(self, self.x)

    def test_frozen_code_survives_mutation_rebinding_and_absent_tensor_abi(self):
        f1 = program('def f(a):\n return -a')
        code_a = f1.__code__
        fn = root(f1, 'helper(x)', 'x, ignored')
        compiled = native.compile(fn)
        initial = compiled(self.x, 1.0)
        entry = next(iter(cache(compiled).graphs.values()))
        f1.__code__ = program('def f(a):\n return a.sin()').__code__
        changed = compiled(self.x, 1.0)
        self.assertNotEqual(initial[0], changed[0])
        f2 = types.FunctionType(code_a, {})
        fn.__globals__['helper'] = f2
        with no_bodies(fn, f1, f2):
            actual = compiled(self.x, self.x)  # New ABI, existing logical guard.
        self.assertEqual(actual[0][-1], ('neg', 0, 0, 0))
        self.assertIs(next(reversed(cache(compiled).graphs.values())), entry)
        self.assertEqual(len(entry.lowerings), 2)
        self.assertEqual(len(cache(compiled).graphs), 2)
        f1.__code__ = code_a
        fn.__globals__['helper'] = f1
        with patch.object(frontend, 'lower', side_effect=AssertionError('warm lower')):
            self.assertEqual(compiled(self.x, self.x), actual)

    def test_equal_distinct_code_guards_share_executor_and_same_code_warm_hit(self):
        helper = program('def f(a):\n return -a')
        fn = root(helper)
        compiled = native.compile(fn)
        compiled(self.x)
        fn.__globals__['helper'] = types.FunctionType(helper.__code__, {})
        with patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('warm parsing')):
            compiled(self.x)
        self.assertEqual(len(cache(compiled).graphs), 1)
        fn.__globals__['helper'] = types.FunctionType(helper.__code__.replace(), {})
        compiled(self.x)
        self.assertEqual(len(cache(compiled).graphs), 2)
        self.assertEqual(len(cache(compiled).executors), 1)

    def test_warm_mutation_revalidated_and_failure_atomic(self):
        helper = program('def f(a):\n return -a')
        fn = root(helper)
        compiled = native.compile(fn)
        compiled(self.x)
        original_code = helper.__code__
        for attribute, invalid in (('__defaults__', (1.0,)), ('__kwdefaults__', {'a': 1.0}),
                                   ('__dict__', {'_torchdynamo_disable': False}),
                                   ('__dict__', {1: None}),
                                   ('__code__', original_code.replace(co_consts=(None, object())))):
            before = list(cache(compiled).graphs.items()), list(cache(compiled).executors.items())
            old = getattr(helper, attribute)
            setattr(helper, attribute, invalid)
            with self.subTest(attribute=attribute), self.assertRaises(NotImplementedError):
                compiled(self.x)
            self.assertEqual(before, (list(cache(compiled).graphs.items()), list(cache(compiled).executors.items())))
            setattr(helper, attribute, old)
            compiled(self.x)
        value = 1.0
        def closed(a):
            return a + value
        fn.__globals__['helper'] = closed
        with self.assertRaises(NotImplementedError):
            compiled(self.x)
        fn.__globals__['helper'] = helper
        helper.__code__ = program('def f(a):\n return a.cos()').__code__
        for failure in ('compile', 'run'):
            before = list(cache(compiled).graphs.items()), list(cache(compiled).executors.items())
            executor = mock_pointwise_executor(lambda *args: (_ for _ in ()).throw(RuntimeError('run failure')))
            config = {'side_effect': RuntimeError('compile failure')} if failure == 'compile' else {'return_value': executor}
            with patch.object(bridge, '_pointwise_compile', **config), self.assertRaises(RuntimeError):
                compiled(self.x)
            self.assertEqual(before, (list(cache(compiled).graphs.items()), list(cache(compiled).executors.items())))
        compiled(self.x)
        self.assertEqual(len(cache(compiled).graphs), 2)

    def test_ignored_inputs_replacements_and_runtime_scalar_history(self):
        helper = program('def f(a, unused):\n return -a')
        fn = root(helper, 'helper(x, ignored)', 'x, ignored')
        compiled = native.compile(fn)
        for ignored in (0.0, 1.0, False, self.x):
            compiled(self.x, ignored)
        self.assertEqual(len(cache(compiled).graphs), 1)
        self.assertEqual(self.validate.call_count, 4)
        identity = program('def f(a):\n return a')
        fn = root(identity, '(x * helper(scale)) + helper(scale)', 'x, scale')
        compiled = native.compile(fn)
        outputs = [compiled(self.x, scale) for scale in (0.375, 0.5, 0.375, -0.0)]
        self.assertEqual([r[2] for r in outputs], [(), (0.5,), (0.375,), (-0.0,)])
        self.assertEqual(len(cache(compiled).graphs), 2)
        self.assertEqual(len([n for n in outputs[-1][0] if n[0] == 'scalar']), 1)
        native.compiler.reset()
        self.assertEqual(compiled(self.x, 0.375)[2], ())

    def test_ignored_helper_binding_replaced_on_new_abi(self):
        ignored = program('def f(a):\n return a')
        fn = program('def f(x, unused):\n saved = helper\n return -x', helper=ignored)
        compiled = native.compile(fn)
        compiled(self.x, 1.0)
        fn.__globals__['helper'] = 0.5
        compiled(self.x, self.x)
        self.assertEqual(len(cache(compiled).graphs), 1)
        # Ignored captures still undergo container and constants validation.
        ignored.__defaults__ = (1.0,)
        fn.__globals__['helper'] = ignored
        with self.assertRaises(NotImplementedError):
            compiled(self.x, 1.0)

    def test_native_input_validation_precedes_even_ignored_helper_arguments(self):
        helper = program('def f(a, b):\n return -a')
        fn = root(helper, 'helper(x, y)', 'x, y')
        compiled = native.compile(fn)
        compiled(self.x, self.x)
        before = self.codegen.call_count
        with patch.object(bridge, '_pointwise_validate_inputs', side_effect=ValueError('native invalid input')):
            with self.assertRaisesRegex(ValueError, 'native invalid input'):
                compiled(self.x, self.x)
        self.assertEqual(self.codegen.call_count, before)
        for invalid in (1, object(), helper):
            with self.assertRaises(NotImplementedError):
                compiled(self.x, invalid)

    def test_new_abi_failure_keeps_existing_lowerings_and_lru_order(self):
        helper = program('def f(a):\n return -a')
        fn = root(helper, 'helper(x)', 'x, unused')
        compiled = native.compile(fn)
        compiled(self.x, 1.0)
        state = cache(compiled)
        entry = next(iter(state.graphs.values()))
        before = (list(state.graphs.items()), list(state.executors.items()), list(entry.lowerings.items()))
        executor = mock_pointwise_executor(lambda *args: (_ for _ in ()).throw(RuntimeError('new ABI failure')))
        with patch.object(bridge, '_pointwise_compile', return_value=executor):
            with self.assertRaisesRegex(RuntimeError, 'new ABI failure'):
                compiled(self.x, self.x)
        self.assertEqual(before, (list(state.graphs.items()), list(state.executors.items()), list(entry.lowerings.items())))
        compiled(self.x, self.x)
        self.assertEqual(len(entry.lowerings), 2)
        self.assertEqual(len(state.graphs), 1)

    def test_recompile_limit_lowering_lru_and_reset_release_helper_code(self):
        helper = program('def f(a):\n return -a')
        fn = root(helper, 'helper(x)', 'x, unused')
        compiled = frontend.implementation(fn, 2)
        y = native.tensor([0.5, 0.75])
        for unused in (1.0, self.x, y):
            compiled(self.x, unused)
        state = cache(compiled)
        self.assertEqual(len(state.graphs), 1)
        self.assertEqual(len(next(iter(state.graphs.values())).lowerings), 2)
        self.assertEqual(len(state.executors), 2)
        helper.__code__ = helper.__code__.replace()
        compiled(self.x, y)
        self.assertEqual(len(state.graphs), 2)
        helper.__code__ = helper.__code__.replace()
        before = list(state.graphs.items()), list(state.executors.items())
        with self.assertRaisesRegex(NotImplementedError, 'recompile_limit=2'):
            compiled(self.x, y)
        self.assertEqual(before, (list(state.graphs.items()), list(state.executors.items())))
        native.compiler.reset()
        compiled(self.x, y)
        self.assertEqual(len(state.graphs), 1)

    def test_nonexecution_reset_and_no_retained_function_containers(self):
        class Payload:
            pass
        payload = Payload()
        helper = program('def f(a):\n return -a')
        helper.__dict__['annotation'] = payload
        helper.__globals__['unused'] = payload
        helper_ref, payload_ref = weakref.ref(helper), weakref.ref(payload)
        fn = root(helper)
        compiled = native.compile(fn)
        with no_bodies(fn, helper):
            compiled(self.x)
            compiled(self.x)
        code_ref = weakref.ref(helper.__code__)
        del fn.__globals__['helper'], helper, payload
        gc.collect()
        self.assertIsNone(helper_ref())
        self.assertIsNone(payload_ref())
        self.assertIsNotNone(code_ref())
        native.compiler.reset()
        gc.collect()
        self.assertIsNone(code_ref())
        self.assertFalse(cache(compiled).graphs)
        self.assertFalse(cache(compiled).executors)
        cache_ref = weakref.ref(cache(compiled))
        del compiled
        gc.collect()
        self.assertIsNone(cache_ref())


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class HelperHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def test_ignored_global_and_closure_arguments_revalidated_on_warm_hits(self):
        check_ignored_capture_admission(self, native.tensor([1.0, -2.0]).to('cuda:0'))

    def compare(self, actual, expected, inputs):
        torch = self.torch
        actual_cpu = torch.tensor(actual.cpu().tolist(), dtype=torch.float32).reshape(expected.shape)
        expected_cpu = expected.cpu()
        torch.testing.assert_close(actual_cpu, expected_cpu, rtol=1e-5, atol=1e-6, equal_nan=True)
        zeros = (actual_cpu == 0) & (expected_cpu == 0)
        self.assertTrue(torch.equal(actual_cpu.signbit()[zeros], expected_cpu.signbit()[zeros]))
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(str(actual.device), str(expected.device))
        self.assertEqual(actual.dtype, native.float32)
        self.assertFalse(actual.requires_grad)
        for value in inputs:
            if type(value) is native.Tensor and value.numel():
                self.assertNotEqual(actual.data_ptr(), value.data_ptr())

    def test_generic_helpers_fresh_persistent_shapes_scalars_aliases_and_ieee(self):
        torch = self.torch
        forms = (
            ('def f(a, b):\n return a-b', 'helper(x, y) + helper(y, x)'),
            ('def f(a):\n v = a.sin()\n return v * v', 'helper(helper(x)) + helper(y)'),
            ('def f(a):\n return a', 'helper(x) * helper(scale) + y'),
            ('def f(a):\n return 0.375', '(x - y) * helper(x)'),
            ('def f(a, b):\n return a*b', 'helper(x, scale) - y'),
        )
        for source, expression in forms:
            for persistent in (False, True):
                helper = program(source)
                fn = root(helper, expression, 'x, y, scale')
                ref_helper = program(source)
                ref_fn = root(ref_helper, expression, 'x, y, scale')
                compiled, reference = native.compile(fn), torch.compile(ref_fn)
                for step, (count, scale, alias) in enumerate(((12, 0.375, False), (17, 0.5, False),
                                                               (17, 0.375, True), (12, -0.0, False))):
                    if not persistent:
                        # Fresh wrappers on both sides, with ordinary compiler state.
                        compiled, reference = native.compile(fn), torch.compile(ref_fn)
                    values = [0., -0., float('inf'), -float('inf'), float('nan'),
                              1e-40, -1e-40, 1e30, -1e30, 1.25, -2.5, 0.375]
                    data = (values + [0.25] * count)[:count]
                    if step % 2:
                        data.reverse()
                    x = native.tensor(data).to('cuda:0')
                    tx = torch.tensor(data, device='cuda:0')
                    y = x if alias else native.tensor(list(reversed(data))).to('cuda:0')
                    ty = tx if alias else torch.tensor(list(reversed(data)), device='cuda:0')
                    args, refs = (x, y, scale), (tx, ty, scale)
                    with self.subTest(expression=expression, persistent=persistent, step=step):
                        for _ in range(5):
                            with no_bodies(fn, helper):
                                actual = compiled(*args)
                            expected = reference(*refs)
                        torch.cuda.synchronize()
                        self.compare(actual, expected, args)
                        torch.testing.assert_close(torch.tensor(x.cpu().tolist()), tx.cpu(),
                                                   rtol=0, atol=0, equal_nan=True)
                        executor = next(reversed(cache(compiled).executors.values()))
                        self.assertIn('torch_rs_pointwise', executor.ptx)
                        self.assertEqual(executor.ptx.count('.visible .entry'), 1)
                        self.assertNotIn('--use_fast_math', executor.options)

    def test_unchanged_frozen_helper_program(self):
        from scripts.torch_compile_default_corpus import CASES, build
        case = next(case for case in CASES if case.name == 'python_helper')
        for persistent in (False, True):
            candidate, reference_program = build(case, native, 'cuda'), build(case, self.torch, 'cuda')
            fn, ref_fn = candidate.function, reference_program.function
            helper = fn.__closure__[0].cell_contents
            compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
            for variant, sample in ((0, 0), (1, 0), (1, 1), (0, 1)):
                if not persistent:
                    compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
                args, refs = candidate.inputs(variant, sample), reference_program.inputs(variant, sample)
                with self.subTest(persistent=persistent, variant=variant, sample=sample):
                    for _ in range(5):
                        with no_bodies(fn, helper):
                            actual = compiled(*args)
                        expected = reference(*refs)
                    self.torch.cuda.synchronize()
                    self.compare(actual, expected, args)

    def test_original_ir_boundary_and_native_input_validation(self):
        helper = program('def f(a, b):\n return a*b+b')
        compiled = native.compile(root(helper, 'helper(x, y)', 'x, y'))
        x = native.ones(3, 4).to('cuda:0')
        y = native.ones(4).to('cuda:0')
        # Tensor-leaf multiply-add remains admitted through a helper.
        self.assertEqual(compiled(x, y).cpu().tolist(), [[2.] * 4] * 3)
        compiled = native.compile(root(program('def f(a, b):\n return a*b+0.5'),
                                       'helper(x, y)', 'x, y'))
        with self.assertRaises((ValueError, NotImplementedError)):
            compiled(x, y)
        self.assertFalse(cache(compiled).graphs)
        # The former one-stage trig rejection now follows the same native path
        # through helpers, with both Python bodies forbidden on cold/warm calls.
        helper = program('def f(a, b):\n return a.sin()+b')
        fn = root(helper, 'helper(x, y)', 'x, y')
        compiled = native.compile(fn)
        reference = self.torch.compile(root(program('def f(a, b):\n return a.sin()+b'),
                                            'helper(x, y)', 'x, y'))
        tx = self.torch.ones(3, 4, device='cuda:0')
        ty = self.torch.ones(4, device='cuda:0')
        for _ in range(2):
            with no_bodies(fn, helper):
                actual = compiled(x, y)
            self.compare(actual, reference(tx, ty), (x, y))
        helper = program('def f(a, ignored):\n return -a')
        compiled = native.compile(root(helper, 'helper(x, y)', 'x, y'))
        for invalid in (x.t(), native.ones(3, 4),
                        native.tensor([1., 2.], requires_grad=True)):
            with self.assertRaises((ValueError, NotImplementedError)):
                compiled(x, invalid)

    @unittest.skipUnless(two_device_reservation(), 'requires explicit two-device reservation')
    def test_two_h100_devices_restore_context_and_share_helper_semantics(self):
        if self.torch.cuda.device_count() < 2:
            self.skipTest('requires two CUDA devices')
        helper = program('def f(a):\n return a.sin() * 0.5')
        fn = root(helper, 'helper(x) + helper(x + 0.25)')
        ref_fn = root(program('def f(a):\n return a.sin() * 0.5'), 'helper(x) + helper(x + 0.25)')
        compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
        for device in (0, 1):
            x = native.tensor([0., -0., 1.25, -3.]).to(f'cuda:{device}')
            tx = self.torch.tensor([0., -0., 1.25, -3.], device=f'cuda:{device}')
            with self.torch.cuda.device(1-device), no_bodies(fn, helper):
                actual = compiled(x)
                self.assertEqual(self.torch.cuda.current_device(), 1-device)
            self.compare(actual, reference(tx), (x,))
        self.assertEqual({executor.device for executor in cache(compiled).executors.values()}, {0, 1})


if __name__ == '__main__':
    unittest.main()
