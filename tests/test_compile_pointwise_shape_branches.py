"""Source-bound shape branches; mock tests make no CUDA or performance claims."""
import math
import operator
import types
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests import test_compile_pointwise_helpers as helpers
from tests.test_compile_pointwise_helpers import no_bodies
from tests.test_compile_pointwise_jit import available, cache, lower, program


def shape_lower(fn, shapes=((3,),), arguments=None, **kwargs):
    if arguments is None:
        arguments = tuple(frontend.Value(i) for i in range(len(shapes)))
    parsed = frontend.analyze(fn, len(arguments))
    _, values = frontend.resolve(fn, parsed, arguments)
    metadata = []
    for shape in shapes:
        strides, stride = [], 1
        for size in reversed(shape):
            strides.append(stride)
            stride *= max(size, 1)
        metadata.append((shape, tuple(reversed(strides)), False, 'torch.float32', 'cuda:0'))
    return frontend.lower(parsed, values, len(shapes), metadata=tuple(metadata), **kwargs).graph


class ShapeBranchAdmission(unittest.TestCase):
    def test_all_comparisons_both_orders_and_outcomes(self):
        for spelling, comparison in (('<', operator.lt), ('<=', operator.le),
                                     ('==', operator.eq), ('!=', operator.ne),
                                     ('>=', operator.ge), ('>', operator.gt)):
            for reversed_ in (False, True):
                condition = f'4 {spelling} x.shape[0]' if reversed_ else f'x.shape[0] {spelling} 4'
                fn = program(f'def f(x):\n if {condition}:\n  return -x\n else:\n  return x.relu()')
                for size in (0, 1, 3, 4, 5):
                    result = comparison(4, size) if reversed_ else comparison(size, 4)
                    predicates = []
                    with self.subTest(condition=condition, size=size), no_bodies(fn):
                        actual = shape_lower(fn, ((size,),), predicates=predicates)
                    self.assertEqual(actual, lower(program('def f(x):\n return '+('-x' if result else 'x.relu()'))))
                    self.assertEqual(len(predicates), 1)
                    source = frontend.BindingSource('parameter', 'x', 0)
                    self.assertEqual(predicates[0].source, source)
                    self.assertTrue(predicates[0].matches({source: ((size,),)}))

    def test_aliases_negative_axis_and_rank_failures(self):
        fn = program('def f(x):\n alias=x\n axis=-1\n threshold=4\n if alias.shape[axis] >= threshold:\n  return -x\n return x.relu()')
        for shape in ((4,), (3, 4), (0, 4), (1, 4)):
            self.assertEqual(shape_lower(fn, (shape,)), lower(program('def f(x):\n return -x')))
        with self.assertRaisesRegex(NotImplementedError, 'axis.*range'):
            shape_lower(fn, ((),))
        for axis in (-3, 2):
            bad = program(f'def f(x):\n if x.shape[{axis}] == 1:\n  return -x\n return x.relu()')
            with self.assertRaisesRegex(NotImplementedError, 'axis.*range'):
                shape_lower(bad, ((2, 3),))

    def test_joins_early_returns_and_sequential_conditions(self):
        fn = program('def f(x):\n if x.shape[0] < 4:\n  y=-x\n else:\n  y=x.relu()\n if 6 <= x.shape[0]:\n  y=y+2\n return y*3')
        for size, expression in ((2, '-x*3'), (4, 'x.relu()*3'), (7, '(x.relu()+2)*3')):
            self.assertEqual(shape_lower(fn, ((size,),)), lower(program(f'def f(x):\n return {expression}')))
        fn = program('def f(x):\n if x.shape[0] < 4:\n  return -x\n y=x.relu()\n return y*3')
        for size, expression in ((2, '-x'), (7, 'x.relu()*3')):
            self.assertEqual(shape_lower(fn, ((size,),)), lower(program(f'def f(x):\n return {expression}')))

    def test_many_sequential_early_returns_are_bounded_without_python_recursion(self):
        source = 'def f(x):\n'+' if x.shape[0] > 1:\n  return -x\n'*500
        fn = program(source+' return x.relu()')
        for size, expression in ((3, '-x'), (1, 'x.relu()')):
            predicates, observed = [], []
            with no_bodies(fn):
                actual = shape_lower(fn, ((size,),), predicates=predicates, observed=observed)
            self.assertEqual(actual, lower(program(f'def f(x):\n return {expression}')))
            self.assertEqual(len(predicates), 1 if size == 3 else 500)
            self.assertEqual([s.name for s in observed], ['x'])
        invalid = program(source+' return x.sum()')
        with self.assertRaises(NotImplementedError), no_bodies(invalid):
            shape_lower(invalid)

    def test_root_loops_and_straight_line_helpers_compose(self):
        helper = program('def f(a):\n return a.sin()*0.5')
        fn = program('def f(x):\n original=x\n for i in range(2):\n  x=-x\n if original.shape[0] < 4:\n  y=helper(x)\n else:\n  y=x.relu()\n for j in range(2):\n  y=y+1\n return y', helper=helper)
        for size, expression in ((2, '(-(-x)).sin()*0.5+1+1'), (5, '(-(-x)).relu()+1+1')):
            with no_bodies(fn, helper):
                self.assertEqual(shape_lower(fn, ((size,),)), lower(program(f'def f(x):\n return {expression}')))

    def test_predicate_provenance_cannot_be_laundered(self):
        identity = program('def f(a):\n return a')
        literal = program('def f(a):\n return 4')
        prefixes_and_conditions = (
            ('', 'x.shape[0] < captured'), ('threshold=-captured', 'x.shape[0] < threshold'),
            ('threshold=-(-captured)', 'x.shape[0] < threshold'),
            ('threshold=helper(x)', 'x.shape[0] < threshold'),
            ('threshold=-helper(x)', 'x.shape[0] < threshold'),
            ('alias=identity(x)', 'alias.shape[0] < 4'),
            ('alias=x+1', 'alias.shape[0] < 4'),
            ('for i in range(1):\n  ignored=x', 'x.shape[0] < i'),
            ('for i in range(1):\n  ignored=x\n threshold=-i', 'x.shape[0] < threshold'),
            ('', 'x.shape[captured] < 4'), ('axis=-captured', 'x.shape[axis] < 4'),
            ('', 'x.shape[0]+1 < 4'), ('', 'x.shape[0] < 4.0'),
            ('', 'x.shape[0] < True'), ('', 'x'), ('', 'x.shape[0]'),
            ('', 'x.stride[0] < 4'), ('', 'x[0] < 4'),
        )
        for prefix, condition in prefixes_and_conditions:
            source = 'def f(x):\n '+(prefix+'\n ' if prefix else '')+f'if {condition}:\n  return -x\n return x.relu()'
            fn = program(source, captured=1, helper=literal, identity=identity)
            with self.subTest(source=source), no_bodies(fn, literal, identity), self.assertRaises(NotImplementedError):
                shape_lower(fn)

    def test_negated_boolean_literals_do_not_gain_predicate_provenance(self):
        cases = (
            ('axis=True\n axis=-axis', 'x.shape[axis] < 4', 'literal integer axis'),
            ('axis=False\n axis=-axis', 'x.shape[axis] < 4', 'literal integer axis'),
            ('threshold=True\n threshold=-threshold', 'x.shape[0] > threshold', 'exact integer literal'),
            ('threshold=False\n threshold=-(-threshold)', 'x.shape[0] > threshold', 'exact integer literal'),
        )
        for prefix, condition, diagnostic in cases:
            fn = program('def f(x):\n '+prefix+'\n if '+condition+':\n  return -x\n return x.relu()')
            with self.subTest(prefix=prefix), no_bodies(fn), self.assertRaisesRegex(NotImplementedError, diagnostic):
                shape_lower(fn)

    def test_exact_integer_negation_and_folded_integer_bytecode_remain_valid(self):
        fn = program('def f(x):\n axis=1\n axis=-axis\n threshold=4\n threshold=-(-threshold)\n if x.shape[axis]<threshold:\n  return -x\n return x.relu()')
        for size, expression in ((3, '-x'), (7, 'x.relu()')):
            with self.subTest(size=size), no_bodies(fn):
                self.assertEqual(shape_lower(fn, ((size,),)), lower(program('def f(x):\n return '+expression)))
        # CPython folds this spelling to an exact integer constant; admission
        # must not invent source distinctions absent from the bytecode.
        folded = program('def f(x):\n if x.shape[0] > -True:\n  return -x\n return x.relu()')
        with no_bodies(folded):
            self.assertEqual(shape_lower(folded), lower(program('def f(x):\n return -x')))

    def test_deferred_nested_regions_reject_even_in_inactive_arms(self):
        for body in ('if x.shape[0] < 2:\n   y=-x\n  else:\n   y=x.relu()',
                     'for i in range(0):\n   y=-x', 'y=x.sum()', 'y=x/2'):
            fn = program('def f(x):\n if x.shape[0] > 10:\n  '+body+'\n else:\n  y=-x\n return y')
            with self.subTest(body=body), self.assertRaises(NotImplementedError):
                shape_lower(fn)
        fn = program('def f(x):\n for i in range(0):\n  if x.shape[0] < 4:\n   x=-x\n return -x')
        with self.assertRaises(NotImplementedError): shape_lower(fn)
        helper = program('def f(a):\n if a.shape[0] < 4:\n  return -a\n return a.relu()')
        fn = program('def f(x):\n if x.shape[0] < 4:\n  return -x\n return helper(x)', helper=helper)
        with self.assertRaises(NotImplementedError): shape_lower(fn)

    def test_inactive_values_locals_ir_and_observations_are_isolated(self):
        fn = program('def f(x, scale):\n if x.shape[0] < 4:\n  scale=2\n  y=-x\n else:\n  y=x.sin()*scale\n return y*scale')
        observed, data_sources = [], set()
        actual = shape_lower(fn, arguments=(frontend.Value(0), 0.75), observed=observed, data_sources=data_sources)
        self.assertEqual(actual, lower(program('def f(x):\n return -x*2')))
        self.assertEqual([s.name for s in observed], ['x'])
        self.assertNotIn(frontend.BindingSource('parameter', 'scale', 1), data_sources)
        fn = program('def f(x):\n if x.shape[0] < 4:\n  y=-x\n else:\n  y=y+x\n return y')
        self.assertEqual(shape_lower(fn), lower(program('def f(x):\n return -x')))
        with self.assertRaisesRegex(NotImplementedError, 'unbound local'):
            shape_lower(fn, ((5,),))
        fn = program('def f(x):\n if x.shape[0] > 4:\n  y=-x\n return y+1')
        with self.assertRaisesRegex(NotImplementedError, 'unbound local'):
            shape_lower(fn)

    def test_both_arms_share_instruction_and_node_budgets(self):
        helper = program('def f(a):\n'+' a=-a\n'*700+' return a')
        fn = program('def f(x):\n if x.shape[0] < 4:\n'+'  x=helper(x)\n'*3+' else:\n'+'  x=helper(x)\n'*3+' return x', helper=helper)
        with self.assertRaisesRegex(NotImplementedError, '4096-node limit'):
            shape_lower(fn)
        helper = program('def f(a):\n'+' b=a\n'*100+' return -a')
        fn = program('def f(x):\n if x.shape[0] < 4:\n'+'  x=helper(x)\n'*45+' else:\n'+'  x=helper(x)\n'*45+' return x', helper=helper)
        with self.assertRaisesRegex(NotImplementedError, 'instruction limit'):
            shape_lower(fn)

    def test_portable_comparison_subscript_and_jump_spellings(self):
        fn = program('def f(x):\n if x.shape[0] > 4:\n  y=-x\n else:\n  y=x.relu()\n return y')
        original = frontend.instructions_for(fn.__code__)
        expected = shape_lower(fn)
        for jump in ('POP_JUMP_IF_FALSE', 'POP_JUMP_FORWARD_IF_FALSE'):
            rewritten = []
            for instruction in original:
                if instruction.opname == 'COMPARE_OP':
                    instruction = instruction._replace(argrepr='bool(>)')
                elif instruction.opname == 'BINARY_SUBSCR':
                    instruction = instruction._replace(opname='BINARY_OP', argrepr='[]')
                elif 'POP_JUMP' in instruction.opname:
                    instruction = instruction._replace(opname=jump)
                rewritten.append(instruction)
            with self.subTest(jump=jump), patch.object(frontend, 'instructions_for', return_value=tuple(rewritten)):
                self.assertEqual(shape_lower(fn), expected)


class ShapeBranchCache(unittest.TestCase):
    setUp = helpers.HelperCache.setUp

    def snapshot(self, compiled):
        state = cache(compiled)
        return ([(key, id(entry), tuple(entry.lowerings.items()), dict(entry.observations))
                 for key, entry in state.graphs.items()], list(state.executors.items()))

    def test_generalized_threshold_crossing_and_crossing_back(self):
        fn = program('def f(x):\n if x.shape[0] < 8:\n  return -x\n return x.relu()')
        compiled = native.compile(fn)
        counts = []
        with no_bodies(fn):
            for size in (3, 5, 10, 12, 3, 7, 10):
                result = compiled(native.ones(size))
                self.assertEqual(result[0][-1][0], 'neg' if size < 8 else 'relu')
                counts.append(len(cache(compiled).graphs))
        self.assertEqual(counts, [1, 2, 3, 3, 3, 3, 3])
        for key in cache(compiled).graphs:
            self.assertTrue(key[2].predicates)

    def test_metadata_only_sources_aliases_and_public_positions(self):
        fn = program('def f(unused, control, x):\n if control.shape[-1] < 8:\n  return -x\n return x.relu()')
        compiled = native.compile(fn)
        for size, alias, ignored in ((3, False, 0.5), (5, False, True), (9, False, False),
                                     (3, True, 1.0), (9, True, 0.0), (3, False, float('nan'))):
            control = native.ones(size)
            x = control if alias else native.ones(2)
            with self.subTest(size=size, alias=alias), no_bodies(fn):
                result = compiled(ignored, control, x)
            self.assertEqual(result[0][-1][0], 'neg' if size < 8 else 'relu')
            key = next(reversed(cache(compiled).graphs))
            self.assertEqual(key[2].predicates[0].source.position, 1)
        self.assertEqual(self.validate.call_count, 6)

    def test_ignored_source_kind_rebinding_preserves_predicate_input(self):
        fn = program('def f(ignored, x):\n if x.shape[0] < 8:\n  return -x\n return x.relu()')
        compiled = native.compile(fn)
        x = native.ones(3)
        self.assertEqual(compiled(0.5, x)[0][-1], ('neg', 0, 0, 0))
        # The predicate's public source remains x when a preceding ignored
        # source becomes native tensor slot zero on a new concrete ABI.
        for ignored in (native.ones(12), x, 0.75):
            result = compiled(ignored, x)
            self.assertEqual(result[0][-1][0], 'neg')
        self.assertEqual(len(cache(compiled).graphs), 1)
        self.assertEqual(compiled(native.ones(3), native.ones(12))[0][-1][0], 'relu')

    def test_inactive_scalar_does_not_promote_active_scalar_does(self):
        fn = program('def f(x, scale):\n if x.shape[0] < 8:\n  return -x\n return x*scale')
        compiled = native.compile(fn)
        for scale in (0.25, 0.5, float('nan'), -0.0):
            self.assertEqual(compiled(self.x, scale)[2], ())
        self.assertEqual(len(cache(compiled).graphs), 1)
        x = native.ones(10)
        self.assertEqual(compiled(x, 0.25)[2], ())
        self.assertEqual(compiled(x, 0.5)[2], (0.5,))
        actual = compiled(x, -0.0)[2]
        self.assertEqual(math.copysign(1, actual[0]), -1)
        self.assertTrue(math.isnan(compiled(x, float('nan'))[2][0]))

    def test_inactive_helper_mutation_warm_reuse_crossing_and_restoration(self):
        helper = program('def f(a):\n return a.relu()')
        fn = program('def f(x):\n if x.shape[0] < 8:\n  return -x\n return helper(x)', helper=helper)
        compiled = native.compile(fn)
        compiled(self.x)
        original = helper.__code__
        helper.__code__ = program('def f(a):\n return a.sum()').__code__
        with patch.object(frontend, 'lower', side_effect=AssertionError('same-arm reparse')):
            compiled(self.x)
        before = self.snapshot(compiled)
        with self.assertRaises(NotImplementedError): compiled(native.ones(10))
        self.assertEqual(self.snapshot(compiled), before)
        fresh = native.compile(fn)
        with self.assertRaises(NotImplementedError): fresh(self.x)
        self.assertFalse(cache(fresh).graphs)
        helper.__code__ = original
        self.assertEqual(compiled(native.ones(10))[0][-1][0], 'relu')
        helper.__code__ = program('def f(a):\n return a.sin()').__code__
        self.assertEqual(compiled(native.ones(10))[0][-1][0], 'sin')
        helper.__code__ = original
        with patch.object(frontend, 'lower', side_effect=AssertionError('restored code reparsed')):
            self.assertEqual(compiled(native.ones(10))[0][-1][0], 'relu')

    def test_new_abi_readmits_inactive_helper_body(self):
        helper = program('def f(a):\n return a.relu()')
        fn = program('def f(x, ignored):\n if x.shape[0] < 8:\n  return -x\n return helper(x)', helper=helper)
        compiled = native.compile(fn)
        compiled(self.x, 0.5)
        original = helper.__code__
        helper.__code__ = program('def f(a):\n return a.sum()').__code__
        with patch.object(frontend, 'lower', side_effect=AssertionError('warm helper body parse')):
            compiled(self.x, 0.75)
        before = self.snapshot(compiled)
        with self.assertRaises(NotImplementedError):
            compiled(self.x, self.x)
        self.assertEqual(self.snapshot(compiled), before)
        helper.__code__ = original
        self.assertEqual(compiled(self.x, self.x)[0][-1][0], 'neg')
        self.assertEqual(len(cache(compiled).graphs), 1)
        self.assertEqual(len(next(iter(cache(compiled).graphs.values())).lowerings), 2)

    def test_descriptor_code_mutation_failure_atomicity_and_reset(self):
        fn = program('def f(x):\n if x.shape[0] < 8:\n  return -x\n return x.relu()')
        compiled = native.compile(fn)
        compiled(self.x)
        before = self.snapshot(compiled)
        for cls in (native.Tensor, native.Tensor.__base__):
            with patch.object(cls, 'shape', property(lambda self: (_ for _ in ()).throw(AssertionError('descriptor invoked')))):
                with self.assertRaises(NotImplementedError): compiled(self.x)
            self.assertEqual(self.snapshot(compiled), before)
        for failure in ('compile', 'run'):
            executor = types.SimpleNamespace(run=lambda *args: (_ for _ in ()).throw(RuntimeError('run failure')))
            config = {'side_effect': RuntimeError('compile failure')} if failure == 'compile' else {'return_value': executor}
            with patch.object(bridge, '_pointwise_compile', **config), self.assertRaises(RuntimeError):
                compiled(native.ones(10))
            self.assertEqual(self.snapshot(compiled), before)
        old = fn.__code__
        fn.__code__ = program('def f(x):\n if x.shape[0] < 8:\n  return x.sin()\n return x.relu()').__code__
        self.assertEqual(compiled(self.x)[0][-1][0], 'sin')
        fn.__code__ = old
        self.assertEqual(compiled(self.x)[0][-1][0], 'neg')
        native.compiler.reset()
        self.assertFalse(cache(compiled).graphs)
        self.assertFalse(cache(compiled).executors)
        compiled(self.x)
        self.assertEqual(len(cache(compiled).graphs), 1)

    def test_boolean_predicate_rejection_preserves_caches_and_last_valid_code(self):
        fn = program('def f(x):\n if x.shape[0]<4:\n  return -x\n return x.relu()')
        compiled = native.compile(fn)
        expected = compiled(self.x)
        original = fn.__code__
        before, codegen_calls = self.snapshot(compiled), self.codegen.call_count
        cases = (
            ('axis=True\n axis=-axis', 'x.shape[axis]<4', 'literal integer axis'),
            ('axis=False\n axis=-axis', 'x.shape[axis]<4', 'literal integer axis'),
            ('threshold=True\n threshold=-threshold', 'x.shape[0]>threshold', 'exact integer literal'),
            ('threshold=False\n threshold=-(-threshold)', 'x.shape[0]>threshold', 'exact integer literal'),
        )
        try:
            for prefix, condition, diagnostic in cases:
                fn.__code__ = program('def f(x):\n '+prefix+'\n if '+condition+':\n  return -x\n return x.relu()').__code__
                with self.subTest(prefix=prefix), no_bodies(fn):
                    with self.assertRaisesRegex(NotImplementedError, diagnostic):
                        compiled(self.x)
                    self.assertEqual(self.snapshot(compiled), before)
                    fresh = native.compile(fn)
                    with self.assertRaisesRegex(NotImplementedError, diagnostic):
                        fresh(self.x)
                    self.assertFalse(cache(fresh).graphs)
                    self.assertFalse(cache(fresh).executors)
                    self.assertEqual(self.codegen.call_count, codegen_calls)
        finally:
            fn.__code__ = original
        with no_bodies(fn), patch.object(frontend, 'lower', side_effect=AssertionError('last-valid code reparsed')):
            self.assertEqual(compiled(self.x), expected)
        self.assertEqual(self.snapshot(compiled), before)
        self.assertEqual(self.codegen.call_count, codegen_calls)

    def test_unused_heavy_signature_guards_remain_observational(self):
        names = ', '.join(f'unused{i}' for i in range(120))
        fn = program(f'def f(x, {names}):\n if x.shape[0] < 8:\n  return -x\n return x.relu()')
        compiled = native.compile(fn)
        compiled(self.x, *([0.25]*120))
        with patch.object(frontend, 'lower', side_effect=AssertionError('unused source caused lowering')):
            compiled(self.x, *([float('nan')]*120))
        entry = next(iter(cache(compiled).graphs.values()))
        self.assertEqual([s.name for s in entry.observed], ['x'])
        self.assertEqual(len(cache(compiled).graphs), 1)

    def test_branch_caches_keep_independent_lru_bounds_and_reset(self):
        helper = program('def f(a):\n return -a')
        fn = program('def f(x, unused):\n if x.shape[0] < 8:\n  return helper(x)\n return x.relu()', helper=helper)
        compiled = frontend.implementation(fn, 2)
        y = native.tensor([0.5, 0.75])
        for unused in (1.0, self.x, y):
            compiled(self.x, unused)
        state = cache(compiled)
        self.assertEqual(len(state.graphs), 1)
        entry = next(iter(state.graphs.values()))
        self.assertEqual(len(entry.lowerings), 2)
        self.assertEqual(len(state.executors), 2)
        with patch.object(frontend, 'lower', wraps=frontend.lower) as lowering:
            compiled(self.x, 1.0)
        self.assertEqual(lowering.call_count, 1)  # The oldest concrete ABI was evicted.
        self.assertEqual(len(entry.lowerings), 2)
        self.assertEqual(len(state.executors), 2)
        helper.__code__ = helper.__code__.replace()
        compiled(self.x, 1.0)
        self.assertEqual(len(state.graphs), 2)
        before = self.snapshot(compiled)
        helper.__code__ = helper.__code__.replace()
        with self.assertRaisesRegex(NotImplementedError, 'recompile_limit=2'):
            compiled(self.x, 1.0)
        self.assertEqual(self.snapshot(compiled), before)
        native.compiler.reset()
        self.assertFalse(state.graphs)
        self.assertFalse(state.executors)
        compiled(self.x, 1.0)
        self.assertEqual(len(state.graphs), 1)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class ShapeBranchHardware(unittest.TestCase):
    setUpClass = classmethod(helpers.HelperHardware.setUpClass.__func__)
    tearDown = helpers.HelperHardware.tearDown
    compare = helpers.HelperHardware.compare

    def reference_evidence_start(self):
        from torch._dynamo.utils import counters
        from torch._inductor import metrics
        counters.clear()
        metrics.reset()

    def reference_evidence_check(self):
        from torch._dynamo.utils import counters
        from torch._inductor import metrics
        self.assertGreater(counters['stats']['unique_graphs'], 0)
        self.assertEqual(sum(counters['graph_break'].values()), 0)
        self.assertGreater(metrics.generated_kernel_count, 0)

    def test_persistent_default_wrappers_threshold_empty_and_ieee(self):
        torch = self.torch
        self.reference_evidence_start()
        source = 'def f(x):\n if x.shape[0] < 8:\n  return -x\n return x.relu()'
        for persistent in (True, False):
            fn, ref_fn = program(source), program(source, framework=torch)
            compiled, reference = native.compile(fn), torch.compile(ref_fn)
            for step, size in enumerate((3, 5, 10, 12, 3, 0, 1)):
                if not persistent:
                    compiled, reference = native.compile(fn), torch.compile(ref_fn)
                values = [0., -0., float('nan'), float('inf'), 1.25, -2.5, -float('inf'), 1e-40, -1e-40]
                pattern = values[step:] + values[:step]
                data = (pattern*2)[:size]
                if step % 2: data.reverse()
                x = native.tensor(data).to('cuda:0')
                tx = torch.tensor(data, dtype=torch.float32, device='cuda:0')
                with self.subTest(persistent=persistent, size=size), no_bodies(fn):
                    for _ in range(5):
                        actual, expected = compiled(x), reference(tx)
                    torch.cuda.synchronize()
                    self.compare(actual, expected, (x,))
                    torch.testing.assert_close(torch.tensor(x.cpu().tolist()), tx.cpu(), rtol=0, atol=0, equal_nan=True)
                    executor = next(reversed(cache(compiled).executors.values()))
                    self.assertIn('torch_rs_pointwise', executor.ptx)
                    self.assertEqual(executor.ptx.count('.visible .entry'), 1)
        self.reference_evidence_check()

    def test_persistent_helpers_loops_multiple_predicates_and_aliases(self):
        torch = self.torch
        self.reference_evidence_start()
        helper = program('def f(a):\n return a*0.5')
        source = 'def f(scale, control, x):\n if control.shape[-1] < 8:\n  y=helper(x)\n else:\n  y=-x\n if x.shape[0] == 1:\n  y=y.relu()\n for i in range(2):\n  y=y*scale\n return y'
        fn = program(source, helper=helper)
        ref_fn = program(source, framework=torch, helper=program('def f(a):\n return a*0.5'))
        compiled, reference = native.compile(fn), torch.compile(ref_fn)
        for size, scale, alias in ((3, 0.25, False), (5, 0.5, False), (10, 0.5, False), (3, -0.0, True), (1, 0.25, True)):
            data = [-float(i+1) for i in range(size)]
            x = native.tensor(data).to('cuda:0')
            tx = torch.tensor(data, device='cuda:0')
            control = x if alias else native.ones(size).to('cuda:0')
            tc = tx if alias else torch.ones(size, device='cuda:0')
            with self.subTest(size=size, scale=scale, alias=alias), no_bodies(fn, helper):
                for _ in range(5):
                    actual, expected = compiled(scale, control, x), reference(scale, tc, tx)
                torch.cuda.synchronize()
                self.compare(actual, expected, (control, x))
        self.reference_evidence_check()

    def test_native_original_inputs_and_ir_remain_admitted_before_launch(self):
        fn = program('def f(x, ignored):\n if x.shape[0] < 8:\n  return -x\n return x.relu()')
        compiled = native.compile(fn)
        x = native.ones(3, 4).to('cuda:0')
        compiled(x, x)
        state = cache(compiled)
        before = list(state.graphs.items()), list(state.executors.items())
        for invalid in (native.ones(3, 4), x.t(),
                        native.tensor([1., 2.], requires_grad=True)):
            with self.subTest(device=str(invalid.device), shape=invalid.shape), self.assertRaises((ValueError, NotImplementedError)):
                compiled(x, invalid)
            self.assertEqual((list(state.graphs.items()), list(state.executors.items())), before)
        bad = program('def f(x, ignored):\n if x.shape[0] < 8:\n  return x*x+1\n return x.relu()')
        compiled = native.compile(bad)
        with self.assertRaises((ValueError, NotImplementedError)):
            compiled(x, native.ones(4).to('cuda:0'))
        self.assertFalse(cache(compiled).graphs)
        self.assertFalse(cache(compiled).executors)
        scalar = native.tensor(1.0).to('cuda:0')
        compiled = native.compile(fn)
        with self.assertRaisesRegex(NotImplementedError, 'axis.*range'):
            compiled(scalar, scalar)
        self.assertFalse(cache(compiled).graphs)


if __name__ == '__main__':
    unittest.main()
