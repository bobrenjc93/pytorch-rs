"""Untimed native leading-sum checks against ordinary default Inductor."""
from contextlib import contextmanager
import gc
import json
import math
import sys
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, _compile_trace
from tests.test_compile_pointwise_jit import available, cache, program
from tests.test_compile_pointwise_method_guards import direct_binding, Poison


@contextmanager
def no_replay(*functions):
    """Call-event sentinel only: collects no durations or profile samples."""
    codes = tuple(fn.__code__ for fn in functions)
    previous = sys.getprofile()

    def forbidden(frame, event, arg):
        if event == 'call' and frame.f_code in codes:
            raise AssertionError('original Python body executed')
        if (event == 'c_call' and type(getattr(arg, '__self__', None)) is native.Tensor
                and getattr(arg, '__name__', '') in ('sum', '__truediv__')):
            raise AssertionError('eager Tensor method executed')

    sys.setprofile(forbidden)
    try:
        with patch.object(_compile_trace, '_execute_operation', side_effect=AssertionError('eager replay')):
            yield
    finally:
        sys.setprofile(previous)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class LeadingSumHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def upload(self, values, shape, framework=native):
        return framework.tensor(values, dtype=framework.float32).reshape(shape).to('cuda:0')

    def compare(self, actual, expected, exact=False):
        torch = self.torch
        # Compare the original GPU metadata before transfer can normalize it.
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), 0)
        self.assertEqual(expected.storage_offset(), 0)
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), str(expected.device))
        self.assertFalse(actual.requires_grad)
        self.assertFalse(expected.requires_grad)
        expected = expected.cpu()
        observed = torch.tensor(actual.cpu().tolist(), dtype=torch.float32).reshape(expected.shape)
        torch.testing.assert_close(observed, expected, rtol=0 if exact else 1e-5,
                                   atol=0 if exact else 1e-6, equal_nan=True)
        self.assertTrue(torch.equal(observed.isnan(), expected.isnan()))
        self.assertTrue(torch.equal(observed.isposinf(), expected.isposinf()))
        self.assertTrue(torch.equal(observed.isneginf(), expected.isneginf()))
        zeros = (expected == 0) & (observed == 0)
        self.assertTrue(torch.equal(observed.signbit()[zeros], expected.signbit()[zeros]))
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), 'cuda:0')
        self.assertFalse(actual.requires_grad)

    def pair(self, source, **bindings):
        fn, ref_fn = program(source, **bindings), program(source, self.torch, **bindings)
        return fn, ref_fn, native.compile(fn), self.torch.compile(ref_fn)

    def values(self, shape):
        return [float((i * 7) % 19 - 9)/8 for i in range(math.prod(shape))]

    def selected(self, compiled):
        entry = next(reversed(cache(compiled).graphs.values()))
        operation = next(reversed(entry.lowerings.values())).graph
        self.assertIs(type(operation), frontend.LeadingSum)
        return entry, operation

    def cache_state(self, compiled):
        state = cache(compiled)
        return (tuple((key, tuple(entry.lowerings)) for key, entry in state.graphs.items()),
                tuple(state.executors), tuple(state.prepared), state.prepared_bytes)

    def test_public_default_leading_sum_positive_capability(self):
        fn, _, compiled, reference = self.pair('def f(x):\n return x.sum(dim=0)')
        shape = (128, 256)
        x, tx = self.upload(self.values(shape), shape), self.upload(self.values(shape), shape, self.torch)
        with no_replay(fn):
            result = compiled(x)
        self.compare(result, reference(tx))
        self.assertNotEqual(result.data_ptr(), x.data_ptr())

    def test_withdrawn_small_empty_and_column_families_reject_without_publication(self):
        # These are admission negatives, not repairs of the retained K1 result.
        for shape in ((32, 1), (64, 256), (257, 256), (0, 256), (128, 0),
                      (128, 128), (128, 255), (128, 260)):
            for keepdim in (False, True):
                fn = program(f'def f(x):\n return x.sum(0, keepdim={keepdim})')
                compiled = native.compile(fn)
                values = [-0.] * math.prod(shape) if shape == (32, 1) else self.values(shape)
                x = self.upload(values, shape)
                before = self.cache_state(compiled)
                with self.subTest(shape=shape, keepdim=keepdim), no_replay(fn):
                    with self.assertRaises((RuntimeError, NotImplementedError, ValueError)):
                        compiled(x)
                self.assertEqual(self.cache_state(compiled), before)
        fn = program('def f(x):\n return x.sum(0)')
        compiled = native.compile(fn)
        x = self.upload(self.values((128, 252)), (128, 252))
        with no_replay(fn):
            compiled(x)
        before = self.cache_state(compiled)
        # Both extents individually fit. It is the generalized column guard
        # that is unsupported; no subsequent reference realignment is claimed.
        with no_replay(fn), self.assertRaises((RuntimeError, NotImplementedError, ValueError)):
            compiled(self.upload(self.values((128, 256)), (128, 256)))
        self.assertEqual(self.cache_state(compiled), before)

    def epilogue_metadata_history(self, source, history, **bindings):
        native.compiler.reset()
        self.torch.compiler.reset()
        fn, _, compiled, reference = self.pair(source, **bindings)
        preparations = []
        for shape, arguments in history:
            values = self.values(shape)
            x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
            with self.subTest(source=source, shape=shape, arguments=arguments, bindings=bindings):
                with no_replay(fn):
                    actual, prepared = compiled._torch_rs_pointwise_receipt(x, *arguments)
                expected = reference(tx, *arguments)
                self.compare(actual, expected)
                self.assertEqual(prepared.kind, 'leading_sum')
                preparations.append(prepared)
        return preparations

    def test_literal_and_captured_divisor_kinds_keepdim_and_revisits(self):
        history = tuple(((rows, 132), ()) for rows in (129, 128, 193, 129))
        for keepdim in (False, True):
            for literal in ('2.0', '1.0', '-2', '1', '0', 'True', 'False', '-0.0'):
                self.epilogue_metadata_history(
                    f'def f(x):\n return x.sum(0, keepdim={keepdim})/{literal}', history)
            for divisor in (2., 1, True, 0., False, -0., math.inf, -math.inf, math.nan):
                self.epilogue_metadata_history(
                    f'def f(x):\n return x.sum(0, keepdim={keepdim})/d', history, d=divisor)

    def test_static_divisor_nonfinite_zero_overflow_and_underflow_classification(self):
        shape = (128, 252)
        classes = (2.**-120, -2.**-120, 0., -0., math.inf, -math.inf, math.nan)
        values = [classes[column % len(classes)] if row == 0 else 0.
                  for row in range(shape[0]) for column in range(shape[1])]
        for keepdim in (False, True):
            for divisor in (0., -0., 2.**-130, 1e300, math.inf, -math.inf, math.nan):
                native.compiler.reset()
                self.torch.compiler.reset()
                fn, _, compiled, reference = self.pair(
                    f'def f(x):\n return x.sum(0, keepdim={keepdim})/d', d=divisor)
                x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
                with self.subTest(divisor=divisor, keepdim=keepdim), no_replay(fn):
                    actual = compiled(x)
                self.compare(actual, reference(tx), exact=True)

    def test_runtime_float_history_reads_current_slots_and_full_divide_source(self):
        shape = (128, 256)
        classes = (2.**-120, -2.**-120, 0., -0., 2.**120, -2.**120)
        values = [classes[column % len(classes)] if row == 0 else 0.
                  for row in range(shape[0]) for column in range(shape[1])]
        history = (2., 3., 2.**-130, 1e300, 0., -0., math.inf, math.nan, 2.)
        for keepdim in (False, True):
            native.compiler.reset()
            self.torch.compiler.reset()
            fn, _, compiled, reference = self.pair(
                f'def f(x,d):\n return x.sum(0, keepdim={keepdim})/d')
            sources = []
            for divisor in history:
                x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
                with self.subTest(divisor=divisor, keepdim=keepdim), no_replay(fn):
                    actual = compiled(x, divisor)
                self.compare(actual, reference(tx, divisor))
                prepared = next(reversed(cache(compiled).prepared.values()))[0]
                self.assertEqual(prepared.kind, 'leading_sum')
                sources.append((prepared.source, prepared.ptx))
            print(json.dumps({'test': self.id(), 'shape': shape, 'keepdim': keepdim,
                              'divisorHistoryHex': [value.hex() for value in history],
                              'reference': 'ordinary torch.compile, default options, persistent wrappers',
                              'sources': [{'source': source, 'ptx': ptx} for source, ptx in sources]},
                             allow_nan=False), flush=True)
            self.assertIn('div.full.f32', sources[1][0])
            self.assertIn('div.full.f32', sources[1][1])
            self.assertNotIn('div.full.ftz.f32', sources[1][1])

    def test_argument_bool_history_and_captured_kind_restoration(self):
        for keepdim in (False, True):
            history = tuple(((128, 252), (divisor,)) for divisor in (True, False, True))
            preparations = self.epilogue_metadata_history(
                f'def f(x,d):\n return x.sum(0, keepdim={keepdim})/d', history)
            self.assertTrue(all('div.full.f32' not in p.source for p in preparations))
        fn, ref_fn, compiled, reference = self.pair('def f(x):\n return x.sum(0)/divisor', divisor=2)
        shape = (128, 256)
        for divisor in (2, True, 2., 3., False, 2.):
            fn.__globals__['divisor'] = ref_fn.__globals__['divisor'] = divisor
            x, tx = self.upload(self.values(shape), shape), self.upload(self.values(shape), shape, self.torch)
            with self.subTest(kind=type(divisor).__name__, divisor=divisor), no_replay(fn):
                actual = compiled(x)
            self.compare(actual, reference(tx))

    def test_independent_dimension_histories_revisit_and_eviction(self):
        for keepdim in (False, True):
            for axis in (0, 1):
                native.compiler.reset()
                self.torch.compiler.reset()
                fn, _, compiled, reference = self.pair(
                    f'def f(x):\n return x.sum(0, keepdim={keepdim})/x.shape[{axis}]')
                preparations, operations = [], []
                for index, rows in enumerate((129, 128, 193, 129)):
                    if index == 3:
                        state = cache(compiled)
                        state.executors.clear()
                        state.prepared.clear()
                        state.prepared_bytes = 0
                        # Also force the selected logical entry to re-lower.
                        next(reversed(state.graphs.values())).lowerings.clear()
                    shape = (rows, 252)
                    values = self.values(shape)
                    x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
                    with self.subTest(axis=axis, keepdim=keepdim, rows=rows), no_replay(fn):
                        actual = compiled(x)
                    self.compare(actual, reference(tx))
                    entry, operation = self.selected(compiled)
                    operations.append(operation)
                    self.assertEqual(operation.row_hint, entry.numerical_hint)
                    self.assertEqual(operation.scalar_count, 0)
                    preparations.append(next(reversed(cache(compiled).prepared.values()))[0])
                self.assertEqual([op.row_hint for op in operations], [129, 128, 128, 128])
                self.assertEqual(preparations[1].executable_identity, preparations[3].executable_identity)
                self.assertEqual(operations[1], operations[3])
                if axis == 0:
                    self.assertIn('div.full.f32', preparations[3].source)
                else:
                    self.assertNotIn('div.full.f32', preparations[3].source)

    def test_bare_keepdim_cold_warm_prepared_reuse_and_fresh_ownership(self):
        for columns in (132, 252):
            for keepdim in (False, True):
                native.compiler.reset()
                self.torch.compiler.reset()
                fn, _, compiled, reference = self.pair(
                    f'def f(x):\n return x.sum(-2, keepdim={keepdim})')
                retained = []
                for rows in (65, 256, 65):
                    shape = (rows, columns)
                    values = self.values(shape)
                    x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
                    previous = None
                    for repeat in range(2):
                        with no_replay(fn):
                            actual, prepared = compiled._torch_rs_pointwise_receipt(x)
                        expected = reference(tx)
                        self.compare(actual, expected)
                        self.assertEqual(actual.shape, (1, columns) if keepdim else (columns,))
                        self.assertEqual(actual.stride(), (columns, 1) if keepdim else (1,))
                        self.assertTrue(actual.is_contiguous())
                        if previous is not None:
                            self.assertIs(prepared, previous)
                        previous = prepared
                        reused = prepared.run((x,))[0]
                        self.compare(reused, expected)
                        self.assertIsNot(reused, actual)
                        retained.extend((actual, reused))
                self.assertEqual(len({value.data_ptr() for value in retained}), len(retained))

    def test_dead_negation_full_scalar_abi_and_unread_control(self):
        for statement, expected_count in ((' unused=-d\n', 1), ('', 0)):
            native.compiler.reset()
            self.torch.compiler.reset()
            fn, _, compiled, reference = self.pair('def f(x,d):\n'+statement+' return x.sum(0)')
            shape = (128, 256)
            entries, preparations = [], []
            for divisor in (.5, .75, .5):
                values = self.values(shape)
                x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
                with no_replay(fn):
                    actual, prepared = compiled._torch_rs_pointwise_receipt(x, divisor)
                self.compare(actual, reference(tx, divisor))
                entry, operation = self.selected(compiled)
                entries.append(entry)
                preparations.append(prepared)
            self.assertEqual(operation.scalar_count, expected_count)
            self.assertIs(entries[1], entries[2])
            self.assertIs(preparations[1], preparations[2])
            self.assertEqual(len(entries[2].binding_checks), len(entries[1].binding_checks))

    def test_combined_shape_scalar_promotion_retains_hint_and_complete_abi(self):
        fn, _, compiled, reference = self.pair(
            'def f(x,divisor,dead):\n unused=-dead\n return x.sum(0)/divisor')
        entries, operations, preparations = [], [], []
        for rows, divisor, dead in ((129, .5, 2.), (128, .75, 3.), (193, .5, 2.)):
            shape = (rows, 252)
            values = self.values(shape)
            x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
            with no_replay(fn):
                actual, prepared = compiled._torch_rs_pointwise_receipt(x, divisor, dead)
            self.compare(actual, reference(tx, divisor, dead))
            entry, operation = self.selected(compiled)
            entries.append(entry)
            operations.append(operation)
            preparations.append(prepared)
        self.assertEqual(operations[1].row_hint, 128)
        self.assertEqual(operations[1].scalar_count, 2)
        self.assertEqual(operations[1].column_certificate, 252)
        self.assertIsNone(operations[1].row_certificate)
        self.assertEqual((operations[1].row_hint + 127)//128, 1)
        self.assertIs(entries[1], entries[2])
        self.assertEqual(operations[1], operations[2])
        self.assertEqual(preparations[1].executable_identity, preparations[2].executable_identity)

    def test_dead_runtime_slots_survive_epilogues_eviction_and_reset(self):
        for expression in ('x.sum(0)', 'x.sum(0)/2', 'x.sum(0)/x.shape[0]'):
            native.compiler.reset()
            self.torch.compiler.reset()
            fn, _, compiled, reference = self.pair(
                'def f(x,d):\n unused=-d\n return '+expression)
            operations, preparations = [], []
            for index, (rows, dead) in enumerate(((129, .5), (128, .75), (193, .5), (129, .75))):
                if index == 3:
                    state = cache(compiled)
                    state.executors.clear()
                    state.prepared.clear()
                    state.prepared_bytes = 0
                    next(reversed(state.graphs.values())).lowerings.clear()
                shape = (rows, 252)
                values = self.values(shape)
                x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
                with no_replay(fn):
                    actual, prepared = compiled._torch_rs_pointwise_receipt(x, dead)
                expected = reference(tx, dead)
                self.compare(actual, expected)
                operations.append(self.selected(compiled)[1])
                preparations.append(prepared)
            self.assertEqual([op.scalar_count for op in operations], [0, 1, 1, 1])
            self.assertEqual([op.row_hint for op in operations], [129, 128, 128, 128])
            self.assertEqual(operations[1], operations[3])
            self.assertEqual(preparations[1].executable_identity, preparations[3].executable_identity)
            native.compiler.reset()
            self.compare(prepared.run((x,), (dead,))[0], expected)
            for scalars in ((), (dead, 1.), (True,)):
                with self.assertRaises((RuntimeError, TypeError, ValueError)):
                    prepared.run((x,), scalars)

    def test_captured_dead_scalar_control(self):
        fn, ref_fn, compiled, reference = self.pair(
            'def f(x):\n unused=-d\n return x.sum(0)', d=.5)
        shape = (128, 256)
        entries = []
        for dead in (.5, .75, .5):
            fn.__globals__['d'] = ref_fn.__globals__['d'] = dead
            values = self.values(shape)
            x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
            with no_replay(fn):
                actual = compiled(x)
            self.compare(actual, reference(tx))
            entries.append(self.selected(compiled))
        self.assertEqual(entries[1][1].scalar_count, 1)
        self.assertIs(entries[1][0], entries[2][0])

    def test_live_divisor_dead_slot_order_and_nonfinite_runtime_hits(self):
        cases = (
            ('def f(x,divisor,dead):\n unused=-dead\n return x.sum(0)/divisor', 0),
            ('def f(x,dead,divisor):\n unused=-dead\n return x.sum(0)/divisor', 1),
            ('def f(x,divisor,dead):\n result=x.sum(0)/divisor\n unused=-dead\n return result', 0),
            ('def f(x,divisor,dead):\n unused=-dead\n return x.sum(0)/-divisor', 0),
        )
        for source, live_slot in cases:
            native.compiler.reset()
            self.torch.compiler.reset()
            fn, _, compiled, reference = self.pair(source)
            shape = (128, 256)
            selected = []
            for divisor, dead in ((.5, 2.), (.75, 3.), (.5, 2.), (math.inf, 4.), (math.nan, 5.)):
                arguments = (divisor, dead) if live_slot == 0 else (dead, divisor)
                values = self.values(shape)
                x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
                with self.subTest(source=source, divisor=divisor, dead=dead), no_replay(fn):
                    actual = compiled(x, *arguments)
                self.compare(actual, reference(tx, *arguments))
                selected.append(self.selected(compiled))
            for entry, operation in selected[1:]:
                self.assertIs(entry, selected[1][0])
                self.assertEqual(operation.scalar_count, 2)
                self.assertEqual(operation.divisor[1], live_slot)
                self.assertEqual(operation.divisor[2], '/-divisor' in source)

    def test_current_offset_fresh_outputs_aliases_receipts_and_reset(self):
        source = 'def f(data):\n x=data["x"]\n y=x.sum(0)\n return {"sum":[y,y],"input":x,"columns":x.shape[1]}'
        fn, _, compiled, reference = self.pair(source)
        retained = []
        for offset in (0, 1, 2):
            data = [float(i % 13) for i in range(128 * 256 + 2)]
            x = self.upload(data, (len(data),))[offset:offset+128*256].reshape((128, 256))
            tx = self.upload(data, (len(data),), self.torch)[offset:offset+128*256].reshape((128, 256))
            self.assertEqual(x.storage_offset(), offset)
            self.assertEqual(tx.storage_offset(), offset)
            expected = [sum(data[offset+row*256+column] for row in range(128))
                        for column in range(256)]
            with no_replay(fn):
                result, prepared = compiled._torch_rs_pointwise_receipt({'x': x})
            self.compare(result['sum'][0], reference({'x': tx})['sum'][0])
            self.assertEqual(result['sum'][0].cpu().tolist(), expected)
            self.assertIs(result['sum'][0], result['sum'][1])
            self.assertIs(result['input'], x)
            self.assertEqual(result['columns'], 256)
            self.assertEqual(prepared.kind, 'leading_sum')
            self.assertEqual(prepared.input_shapes, ((128, 256),))
            self.assertGreater(prepared.retained_bytes, 0)
            executor = next(reversed(cache(compiled).executors.values()))
            self.assertTrue(prepared.belongs_to(executor))
            with self.assertRaises((RuntimeError, ValueError)):
                executor.plan(1)
            for name in ('instruction_count', 'register_count'):
                with self.assertRaises((RuntimeError, ValueError)):
                    getattr(prepared, name)
            if retained:
                self.assertNotEqual(result['sum'][0].data_ptr(), retained[-1][0].data_ptr())
            retained.append((result['sum'][0], expected))
            with self.assertRaises((RuntimeError, TypeError)):
                prepared.run((x.reshape((256, 128)),))
            with self.assertRaises((RuntimeError, TypeError)):
                prepared.run((x,), (1.,))
            with self.assertRaises((RuntimeError, TypeError)):
                prepared.run((native.ones((128, 256)),))
        native.compiler.reset()
        self.assertFalse(cache(compiled).prepared)
        self.assertEqual(prepared.run((x,))[0].cpu().tolist(), expected)
        for tensor, old_expected in retained:
            self.assertEqual(tensor.cpu().tolist(), old_expected)

    def test_sum_method_mutation_and_invalid_inputs_do_not_publish(self):
        fn = program('def f(x):\n return x.sum(0)/2')
        compiled = native.compile(fn)
        x = self.upload(self.values((128, 256)), (128, 256))
        compiled(x)
        before = self.cache_state(compiled)
        for owner in (native.Tensor, native.Tensor.__base__):
            for name in ('sum', '__truediv__'):
                calls = []
                with direct_binding(owner, name, Poison(calls)):
                    with self.assertRaises(NotImplementedError):
                        compiled(x)
                    with self.assertRaises(NotImplementedError):
                        native.compile(fn)(x)
                self.assertEqual(calls, [])
                self.assertEqual(self.cache_state(compiled), before)
        for invalid in (native.ones((128, 256)), x.reshape((128*256,)), x.transpose(0, 1)):
            with self.subTest(shape=invalid.shape), self.assertRaises((NotImplementedError, RuntimeError)):
                compiled(invalid)
            self.assertEqual(self.cache_state(compiled), before)

    def test_receipt_and_reconstruction_failure_leave_all_cache_orders_unchanged(self):
        fn = program('def f(x):\n return x.sum(0)')
        compiled = native.compile(fn)
        x = self.upload(self.values((128, 256)), (128, 256))
        compiled(x)
        before = self.cache_state(compiled)
        changed = self.upload(self.values((193, 256)), (193, 256))
        for current in (x, changed):
            with patch.object(frontend, '_receipt', side_effect=RuntimeError('injected receipt failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected receipt failure'), no_replay(fn):
                    compiled._torch_rs_pointwise_receipt(current)
            self.assertEqual(self.cache_state(compiled), before)
            with patch.object(frontend.ResultSpec, 'reconstruct', side_effect=RuntimeError('injected reconstruction failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected reconstruction failure'), no_replay(fn):
                    compiled(current)
            self.assertEqual(self.cache_state(compiled), before)

    def test_inactive_new_sum_helper_preserves_warm_pointwise_only_reuse(self):
        helper = program('def f(a):\n return a.relu()')
        fn = program('def f(x,ignored):\n if x.shape[0]<8:\n  return -x\n return helper(x)', helper=helper)
        compiled = native.compile(fn)
        x = self.upload(list(range(12)), (3, 4))
        compiled(x, 0.5)
        original = helper.__code__
        helper.__code__ = program('def f(a):\n return a.sum(dim=0)').__code__
        with patch.object(frontend, 'lower', side_effect=AssertionError('warm inactive helper reparsed')):
            self.assertEqual(compiled(x, 0.75).cpu().tolist(), [[-float(i) for i in range(row*4, row*4+4)] for row in range(3)])
        state = cache(compiled)
        before = (tuple(state.graphs), tuple(state.executors), tuple(state.prepared), state.prepared_bytes)
        larger = self.upload(list(range(40)), (10, 4))
        for wrapper, args in ((compiled, (larger, 0.75)), (compiled, (x, x)),
                              (native.compile(fn), (x, 0.75))):
            with self.assertRaises(NotImplementedError):
                wrapper(*args)
        self.assertEqual((tuple(state.graphs), tuple(state.executors), tuple(state.prepared), state.prepared_bytes), before)
        helper.__code__ = original
        self.assertEqual(compiled(larger, 0.75).cpu().tolist(), larger.cpu().tolist())

    @unittest.skipUnless(sys.implementation.name == 'cpython', 'requires CPython reference ownership')
    def test_warm_calls_release_current_input_and_traceback_owners_without_gc(self):
        fn = program('def f(x):\n return x.sum(0)')
        compiled = native.compile(fn)
        x = self.upload(self.values((128, 256)), (128, 256))
        compiled(x)
        # Cold lowering has other helper lifetimes. Establish the observation
        # only after collecting setup and use a warm current input thereafter.
        gc.collect()
        enabled = gc.isenabled()
        gc.disable()
        try:
            baseline = sys.getrefcount(x)
            output = compiled(x)
            del output
            self.assertEqual(sys.getrefcount(x), baseline)
            traceback = None
            def fail_receipt(result, prepared):
                raise RuntimeError('retained traceback')
            with patch.object(frontend, '_receipt', fail_receipt):
                try:
                    compiled._torch_rs_pointwise_receipt(x)
                except RuntimeError as error:
                    traceback = error.__traceback__
            # The separately retained traceback legitimately keeps the call
            # alive; no Mock call history or exception instance owns a cycle.
            self.assertIsNotNone(traceback)
            self.assertGreater(sys.getrefcount(x), baseline)
            del traceback
            self.assertEqual(sys.getrefcount(x), baseline)
        finally:
            gc.collect()
            if enabled:
                gc.enable()


if __name__ == '__main__':
    unittest.main()
