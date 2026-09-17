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

    def test_public_default_leading_sum_positive_capability(self):
        # This control uses only pre-existing public APIs. The unchanged parent
        # rejects KW_NAMES at admission; missing private APIs cannot explain it.
        fn = program('def f(x):\n return x.sum(dim=0)')
        compiled = native.compile(fn)
        x = native.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=native.float32).to('cuda:0')
        result = compiled(x)
        self.assertEqual(result.cpu().tolist(), [5., 7., 9.])
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.stride(), (1,))
        self.assertEqual(str(result.device), 'cuda:0')
        self.assertNotEqual(result.data_ptr(), x.data_ptr())

    def test_source_derived_singleton_static_three_and_reciprocal_discriminators(self):
        cases = (
            ('x.sum(dim=0)', [-0.], (1, 1)),
            ('x.sum(dim=0, keepdim=True)', [-0.], (1, 1)),
            ('x.sum(-2)', [2.**24, 1., -2.**24], (3, 1)),
            ('x.sum(0)/tiny', [2.**-120], (1, 1)),
        )
        for expression, values, shape in cases:
            native.compiler.reset()
            self.torch.compiler.reset()
            with self.subTest(expression=expression):
                fn, _, compiled, reference = self.pair('def f(x):\n return '+expression, tiny=2.**-130)
                x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
                with no_replay(fn):
                    result = compiled(x)
                self.compare(result, reference(tx), exact=True)
                value = result.cpu().tolist()
                if expression.endswith('/tiny'):
                    self.assertEqual(value, [math.inf])
                elif shape == (3, 1):
                    self.assertEqual(value, [0.])
                else:
                    zero = value[0][0] if len(result.shape) == 2 else value[0]
                    self.assertEqual(math.copysign(1., zero), -1.)

    def test_static_divisor_nonfinite_zero_overflow_and_underflow_classification(self):
        for divisor in (0., -0., 2.**-130, 1e300, math.inf, -math.inf, math.nan):
            native.compiler.reset()
            self.torch.compiler.reset()
            with self.subTest(divisor=divisor):
                fn, _, compiled, reference = self.pair('def f(x):\n return x.sum(0)/d', d=divisor)
                values = [2.**-120, -2.**-120, 0., -0., math.inf, -math.inf, math.nan]
                x, tx = self.upload(values, (1, 7)), self.upload(values, (1, 7), self.torch)
                with no_replay(fn):
                    actual = compiled(x)
                self.compare(actual, reference(tx), exact=True)

    def test_runtime_float_history_reads_current_slots_and_full_divide_source(self):
        fn, _, compiled, reference = self.pair('def f(x,d):\n return x.sum(0)/d')
        values = [2.**-120, -2.**-120, 0., -0., 2.**120, -2.**120]
        sources = []
        history = (2., 3., 2.**-130, 1e300, 0., -0., math.inf, math.nan, 2.)
        for divisor in history:
            with self.subTest(divisor=divisor):
                x, tx = self.upload(values, (1, 6)), self.upload(values, (1, 6), self.torch)
                with no_replay(fn):
                    actual = compiled(x, divisor)
                self.compare(actual, reference(tx, divisor))
                prepared = next(reversed(cache(compiled).prepared.values()))[0]
                self.assertEqual(prepared.kind, 'leading_sum')
                sources.append((prepared.source, prepared.ptx))
        print(json.dumps({'test': self.id(), 'capture': 'native selected source and NVRTC PTX',
                          'shape': [1, 6], 'inputValuesHex': [value.hex() for value in values],
                          'divisorHistoryHex': [value.hex() for value in history],
                          'persistentPublicWrappers': True,
                          'reference': 'ordinary torch.compile, default options',
                          'sources': [{'source': source, 'ptx': ptx} for source, ptx in sources]},
                         allow_nan=False), flush=True)
        # On the second finite value, ordinary scalar history is promoted.
        self.assertIn('div.full.f32', sources[1][0])
        self.assertIn('div.full.f32', sources[1][1])
        self.assertNotIn('div.full.ftz.f32', sources[1][1])

    def test_independent_dimension_histories_small_revisit_and_eviction(self):
        for axis, shapes in (
                (0, ((3, 7), (5, 7), (3, 7), (3, 7))),
                (1, ((3, 7), (3, 9), (3, 7), (3, 7))),
                (1, ((3, 7), (5, 7), (3, 7), (3, 7)))):
            native.compiler.reset()
            self.torch.compiler.reset()
            fn, _, compiled, reference = self.pair(f'def f(x):\n return x.sum(0)/x.shape[{axis}]')
            preparations = []
            for index, shape in enumerate(shapes):
                # Evict only native executable/preparation storage, preserving
                # the selected logical history and the persistent reference.
                if index == 3:
                    cache(compiled).executors.clear()
                    cache(compiled).prepared.clear()
                    cache(compiled).prepared_bytes = 0
                values = [float((i * 7) % 19 - 9)/8 for i in range(math.prod(shape))]
                x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
                with self.subTest(axis=axis, shape=shape, index=index), no_replay(fn):
                    actual = compiled(x)
                self.compare(actual, reference(tx))
                preparations.append(next(reversed(cache(compiled).prepared.values()))[0])
            self.assertNotEqual(preparations[0].executable_identity, preparations[2].executable_identity)
            self.assertEqual(preparations[2].executable_identity, preparations[3].executable_identity)
            if axis == 0 or shapes[1][1] != shapes[0][1]:
                self.assertIn('div.full.f32', preparations[2].source)
            else:
                self.assertNotIn('div.full.f32', preparations[2].source)

    def test_generalized_small_cancellation_and_nonfinite_tree_history(self):
        fn, _, compiled, reference = self.pair('def f(x):\n return x.sum(0)')
        for rows in (3, 5, 3, 9):
            columns = [
                [2.**24, 1., -2.**24] + [0.] * (rows - 3),
                [math.inf, -math.inf] + [1.] * (rows - 2),
                [math.nan] + [1.] * (rows - 1),
                [-0.] * rows,
                [2.**-140] * rows,
            ]
            values = [columns[column][row] for row in range(rows) for column in range(5)]
            x, tx = self.upload(values, (rows, 5)), self.upload(values, (rows, 5), self.torch)
            with self.subTest(rows=rows), no_replay(fn):
                actual = compiled(x)
            self.compare(actual, reference(tx))

    def test_captured_scalar_kind_changes_and_restoration(self):
        fn, ref_fn, compiled, reference = self.pair('def f(x):\n return x.sum(0)/divisor', divisor=2)
        for divisor in (2, True, 2., 3., False, 2.):
            fn.__globals__['divisor'] = ref_fn.__globals__['divisor'] = divisor
            x, tx = self.upload([1., -1., 0., -0.], (1, 4)), self.upload([1., -1., 0., -0.], (1, 4), self.torch)
            with self.subTest(kind=type(divisor).__name__, divisor=divisor), no_replay(fn):
                actual = compiled(x)
            self.compare(actual, reference(tx))

    def test_empty_singleton_tails_and_held_out_tree_shapes(self):
        for keepdim in (False, True):
            native.compiler.reset()
            self.torch.compiler.reset()
            fn, _, compiled, reference = self.pair(f'def f(x):\n return x.sum(-2, keepdim={keepdim})')
            for shape in ((0, 5), (1, 5), (8, 33), (19, 67), (41, 129), (5, 0), (0, 0)):
                values = [float((i * 13) % 31 - 15)/16 for i in range(math.prod(shape))]
                x, tx = self.upload(values, shape), self.upload(values, shape, self.torch)
                with self.subTest(keepdim=keepdim, shape=shape):
                    with no_replay(fn):
                        actual = compiled(x)
                    expected = reference(tx)
                    try:
                        self.compare(actual, expected)
                    except AssertionError:
                        print(json.dumps({'test': self.id(), 'keepdim': keepdim,
                                          'inputShape': shape,
                                          'actualShape': actual.shape, 'actualStride': actual.stride(),
                                          'referenceShape': list(expected.shape),
                                          'referenceStride': expected.stride(),
                                          'reference': 'ordinary torch.compile, same persistent history'},
                                         allow_nan=False), flush=True)
                        raise

    def test_empty_keepdim_gpu_metadata_cold_warm_and_prepared_revisits(self):
        for rows in (0, 1, 5):
            for keepdim in (False, True):
                for epilogue in ('', '/2'):
                    native.compiler.reset()
                    self.torch.compiler.reset()
                    fn, _, compiled, reference = self.pair(
                        f'def f(x):\n return x.sum(0, keepdim={keepdim}){epilogue}')
                    retained = []
                    for columns in (0, 1, 0, 3, 0, 3):
                        shape = (rows, columns)
                        values = [float(i + 1) for i in range(rows * columns)]
                        x = self.upload(values, shape)
                        tx = self.upload(values, shape, self.torch)
                        previous = None
                        for repeat in range(2):
                            with self.subTest(rows=rows, columns=columns,
                                              keepdim=keepdim, epilogue=epilogue, repeat=repeat):
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
                    # Empty-pointer inequality is deliberately not an ownership
                    # oracle; the Rust companion checks actual storage owners.
                    self.assertEqual(len({id(value) for value in retained}), len(retained))

    def test_empty_rows_dimension_zero_divisor_writes_nan(self):
        fn, _, compiled, reference = self.pair('def f(x):\n return x.sum(0)/x.shape[0]')
        x, tx = self.upload([], (0, 5)), self.upload([], (0, 5), self.torch)
        with no_replay(fn):
            actual = compiled(x)
        self.compare(actual, reference(tx), exact=True)
        self.assertTrue(all(math.isnan(value) for value in actual.cpu().tolist()))

    def test_current_offset_fresh_outputs_aliases_receipts_and_reset(self):
        source = 'def f(data):\n x=data["x"]\n y=x.sum(0)\n return {"sum":[y,y],"input":x,"columns":x.shape[1]}'
        fn = program(source)
        compiled = native.compile(fn)
        retained = []
        for offset in (0, 1, 2):
            data = [float(i) for i in range(20)]
            x = self.upload(data, (20,))[offset:offset+12].reshape((3, 4))
            expected = [float(3*(offset+i)+12) for i in range(4)]
            with no_replay(fn):
                result, prepared = compiled._torch_rs_pointwise_receipt({'x': x})
            self.assertEqual(result['sum'][0].cpu().tolist(), expected)
            self.assertIs(result['sum'][0], result['sum'][1])
            self.assertIs(result['input'], x)
            self.assertEqual(result['columns'], 4)
            self.assertEqual(prepared.kind, 'leading_sum')
            self.assertEqual(prepared.input_shapes, ((3, 4),))
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
                prepared.run((x.reshape((2, 6)),))
            with self.assertRaises((RuntimeError, TypeError)):
                prepared.run((x,), (1.,))
            with self.assertRaises((RuntimeError, TypeError)):
                prepared.run((native.ones((3, 4)),))
        native.compiler.reset()
        self.assertFalse(cache(compiled).prepared)
        self.assertEqual(prepared.run((x,))[0].cpu().tolist(), expected)
        for tensor, old_expected in retained:
            self.assertEqual(tensor.cpu().tolist(), old_expected)

    def test_sum_method_mutation_and_invalid_inputs_do_not_publish(self):
        fn = program('def f(x):\n return x.sum(0)/2')
        compiled = native.compile(fn)
        x = self.upload(list(range(12)), (3, 4))
        compiled(x)
        state = cache(compiled)
        before = (tuple(state.graphs), tuple(state.executors), tuple(state.prepared), state.prepared_bytes)
        for owner in (native.Tensor, native.Tensor.__base__):
            for name in ('sum', '__truediv__'):
                calls = []
                with direct_binding(owner, name, Poison(calls)):
                    with self.assertRaises(NotImplementedError):
                        compiled(x)
                    with self.assertRaises(NotImplementedError):
                        native.compile(fn)(x)
                self.assertEqual(calls, [])
                self.assertEqual((tuple(state.graphs), tuple(state.executors), tuple(state.prepared), state.prepared_bytes), before)
        for invalid in (native.ones((3, 4)), x.reshape((12,)), x.transpose(0, 1)):
            with self.subTest(shape=invalid.shape), self.assertRaises((NotImplementedError, RuntimeError)):
                compiled(invalid)
            self.assertEqual((tuple(state.graphs), tuple(state.executors), tuple(state.prepared), state.prepared_bytes), before)

    def test_receipt_and_reconstruction_failure_leave_all_cache_orders_unchanged(self):
        fn = program('def f(x):\n return x.sum(0)')
        compiled = native.compile(fn)
        x = self.upload(list(range(12)), (3, 4))
        compiled(x)
        state = cache(compiled)
        before = (tuple(state.graphs), tuple(state.executors), tuple(state.prepared), state.prepared_bytes)
        for target in ('_receipt',):
            with patch.object(frontend, target, side_effect=RuntimeError('injected receipt failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected receipt failure'):
                    compiled._torch_rs_pointwise_receipt(x)
            self.assertEqual((tuple(state.graphs), tuple(state.executors), tuple(state.prepared), state.prepared_bytes), before)
        with patch.object(frontend.ResultSpec, 'reconstruct', side_effect=RuntimeError('injected reconstruction failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected reconstruction failure'):
                compiled(x)
        self.assertEqual((tuple(state.graphs), tuple(state.executors), tuple(state.prepared), state.prepared_bytes), before)

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
        x = self.upload(list(range(12)), (3, 4))
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
