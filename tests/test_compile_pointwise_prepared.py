"""Immutable preparation retention and current-input execution contracts.

Portable tests exercise the frontend transaction using explicit metadata/launch
mocks; they make no CUDA or performance claims. Hardware tests use real native
preparations and ordinary default torch.compile references.
"""
from contextlib import ExitStack
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests import test_compile_pointwise_jit as jit_tests
from tests import test_compile_pointwise_structured_outputs as structured_tests
from tests.test_compile_pointwise_jit import available, cache, program, two_device_reservation


def newest_prepared(compiled):
    return next(reversed(cache(compiled).prepared.values()))[0]


class PreparedExports(unittest.TestCase):
    def test_prepared_helper_is_private_to_the_native_bridge(self):
        namespace = {}
        exec('from torch_rs import *', namespace)
        for name in ('_PointwisePrepared', '_PointwiseHostPlan', '_PointwiseExecutable',
                     '_pointwise_host_plan'):
            with self.subTest(name=name):
                self.assertTrue(hasattr(bridge, name))
                self.assertNotIn(name, bridge.__all__)
                self.assertNotIn(name, native.__all__)
                self.assertFalse(hasattr(native, name))
                self.assertNotIn(name, namespace)


class PreparedCache(unittest.TestCase):
    def setUp(self):
        structured_tests.StructuredCache.setUp(self)
        compile_ = self.codegen.side_effect
        self.created = []

        def tracked_compile(*args):
            executor = compile_(*args)
            executor.run = Mock(wraps=executor.run)
            executor.prepare = Mock(wraps=executor.prepare)
            self.created.append(executor)
            return executor

        self.codegen.side_effect = tracked_compile

    snapshot = structured_tests.StructuredCache.snapshot

    def test_matching_warm_calls_do_not_prepare_or_reaccount(self):
        compiled = native.compile(program('def f(x):\n return [-x,x]'))
        x = native.ones(3)
        first = compiled(x)
        prepared = newest_prepared(compiled)
        current = native.ones(3)
        with patch.object(self.created[0], 'prepare', side_effect=AssertionError('warm preparation')), \
                patch.object(bridge, '_pointwise_host_plan', side_effect=AssertionError('warm planning/emission')), \
                patch.object(frontend, '_prepared_entry_bytes', side_effect=AssertionError('warm accounting')):
            second = compiled(current)
        self.assertIs(newest_prepared(compiled), prepared)
        self.assertIs(first[1], x)
        self.assertIs(second[1], current)
        self.assertIsNot(first, second)
        self.assertIsNot(first[0], second[0])
        self.created[0].run.assert_called_with((current,), (), 3, (0,))
        self.assertEqual(self.codegen.call_count, 1)

    def test_exact_shape_data_lru_does_not_consume_generalized_logical_slots(self):
        compiled = frontend.implementation(program('def f(x):\n return -x'), 2)
        for size in (3, 5, 7, 9, 5):
            compiled(native.ones(size))
            state = cache(compiled)
            self.assertLessEqual(len(state.prepared), 2)
            self.assertEqual(state.prepared_bytes, sum(item[1] for item in state.prepared.values()))
        self.assertEqual(len(cache(compiled).graphs), 2)
        self.assertEqual(len(cache(compiled).executors), 1)
        self.assertEqual(self.created[0].prepare.call_count, 5)
        self.assertEqual(self.host_plan.call_count, 5)
        self.assertEqual(self.codegen.call_count, 1)
        self.assertEqual([key[1] for key in cache(compiled).prepared], [((9,),), ((5,),)])
        self.assertEqual([key[2] for key in cache(compiled).prepared], [5, 5])

    def test_rank_zero_and_length_one_are_different_preparations(self):
        compiled = native.compile(program('def f(x):\n return -x'))
        scalar, vector = native.tensor(1.), native.ones(1)
        compiled(scalar)
        first = newest_prepared(compiled)
        compiled(vector)
        self.assertIsNot(newest_prepared(compiled), first)
        compiled(scalar)
        self.assertIs(newest_prepared(compiled), first)
        self.assertEqual(self.codegen.call_count, 1)
        self.assertEqual(self.created[0].prepare.call_count, 2)
        self.assertEqual({key[1] for key in cache(compiled).prepared}, {((),), ((1,),)})

    def test_revisited_shape_uses_the_selected_numerical_history(self):
        compiled = frontend.implementation(program('def f(x):\n p=x*x\n return (-p,p.sin())'), 3)
        preparations = []
        for count in (2, 13, 2, 2):
            compiled(native.ones(count))
            preparations.append(newest_prepared(compiled))
        self.assertEqual(self.hints, [2, 13, 13, 13])
        self.assertEqual(self.codegen.call_count, 1)
        self.assertEqual(self.created[0].prepare.call_count, 3)
        self.assertIsNot(preparations[0], preparations[2])
        self.assertIs(preparations[2], preparations[3])
        shape_two = [(key, value[0]) for key, value in cache(compiled).prepared.items()
                     if key[1] == ((2,),)]
        self.assertEqual({key[2] for key, _ in shape_two}, {2, 13})
        self.assertTrue(any(value is preparations[0] for _, value in shape_two))
        self.assertEqual(len(cache(compiled).executors), 1)

    def test_return_order_selects_data_but_cosmetic_containers_share_it(self):
        fn = program('def f(x):\n a=x*x\n b=-a\n return (a,b)')
        compiled = native.compile(fn)
        x = native.ones(13)
        compiled(x)
        first = newest_prepared(compiled)
        fn.__code__ = program('def f(x):\n a=x*x\n b=-a\n return (b,a)').__code__
        compiled(x)
        reverse = newest_prepared(compiled)
        self.assertIsNot(reverse, first)
        fn.__code__ = program('def f(x):\n a=x*x\n b=-a\n return {"b":[b,b],"a":a}').__code__
        result = compiled(x)
        self.assertIs(result['b'][0], result['b'][1])
        self.assertIs(newest_prepared(compiled), reverse)
        self.assertEqual(self.codegen.call_count, 1)
        self.assertEqual(self.created[0].prepare.call_count, 2)
        self.assertEqual({key[3] for key in cache(compiled).prepared}, {(0, 1), (1, 0)})

    def test_runtime_scalar_values_are_not_retained_plan_keys(self):
        compiled = native.compile(program('def f(s,x):\n return x*s'))
        x = native.ones(13)
        compiled(1.5, x)
        compiled(2.5, x)
        promoted = newest_prepared(compiled)
        executor = self.created[-1]
        compiled(-3.5, x)
        self.assertIs(newest_prepared(compiled), promoted)
        self.assertEqual(executor.prepare.call_count, 1)
        executor.run.assert_called_with((x,), (-3.5,), 13, (0,))

    def test_unused_input_shapes_still_bind_the_data_key(self):
        compiled = native.compile(program('def f(unused,x):\n return -x'))
        x = native.ones(3)
        for size in (1, 5, 1):
            compiled(native.ones(size), x)
        self.assertEqual({key[1] for key in cache(compiled).prepared},
                         {((1,), (3,)), ((5,), (3,))})
        self.assertEqual(sum(executor.prepare.call_count for executor in self.created), 2)

    def test_shape_snapshot_mismatch_is_not_run_or_published(self):
        compiled = native.compile(program('def f(x):\n return -x'))
        x = native.ones(3)
        compiled(x)
        before = self.snapshot(compiled)
        mismatched = types.SimpleNamespace(input_shapes=((99,),), retained_bytes=0,
                                           run=Mock(side_effect=AssertionError('mismatched run')))
        with patch.object(self.created[0], 'prepare', return_value=mismatched):
            with self.assertRaisesRegex(NotImplementedError, 'shapes changed'):
                compiled(native.ones(5))
        mismatched.run.assert_not_called()
        self.assertEqual(self.snapshot(compiled), before)

    def test_preparation_execution_and_reconstruction_failures_preserve_all_lrus(self):
        fn = program('def f(x):\n return -x')
        compiled = native.compile(fn)
        x = native.ones(3)
        compiled(x)
        compiled(native.ones(5))
        # Shape promotion changes the selected numerical hint. Warm this
        # shape under that hint before injecting a failure into its plan.
        compiled(x)
        older = newest_prepared(compiled)
        compiled(native.ones(5))
        before = self.snapshot(compiled)
        executor = self.created[0]
        for target, name, exception, args in (
                (bridge, '_pointwise_admit_inputs', RuntimeError('admission'), (x,)),
                (executor, 'prepare', RuntimeError('preparation'), (native.ones(7),)),
                (frontend, '_prepared_entry_bytes', MemoryError('accounting'), (native.ones(7),)),
                (older, 'run', RuntimeError('launch/conversion'), (x,)),
                (frontend.ResultSpec, 'reconstruct', MemoryError('reconstruction'), (x,))):
            with self.subTest(phase=str(exception)), patch.object(target, name, side_effect=exception):
                with self.assertRaises(type(exception)):
                    compiled(*args)
            self.assertEqual(self.snapshot(compiled), before)
        with patch.object(frontend.ResultSpec, 'reconstruct', side_effect=MemoryError('cold reconstruction')):
            cold = native.compile(fn)
            with self.assertRaises(MemoryError):
                cold(x)
            self.assertEqual(self.snapshot(cold), ([], [], [], 0))
        compiled(x)
        self.assertIs(newest_prepared(compiled), older)

    def test_byte_limit_and_count_limit_have_independent_success_only_lru_bounds(self):
        owner = frontend._state.NativeEagerCompileCache()
        keys = [(('code',), ((size,),), 3, (0,)) for size in (3, 5, 7)]
        owner.executors[('code',)] = object()
        executor = owner.executors[('code',)]
        values = [types.SimpleNamespace(belongs_to=lambda owner: owner is executor) for _ in keys]
        with owner.lock, patch.object(frontend, '_PREPARED_CACHE_BYTES', 128):
            for key, value in zip(keys, values):
                frontend._publish_preparation(owner, key, value, 64, 8, ('code',), executor, tuple(owner.prepared))
            self.assertEqual(list(owner.prepared), keys[1:])
            self.assertEqual(owner.prepared_bytes, 128)
            frontend._publish_preparation(owner, keys[1], values[1], 64, 8, ('code',), executor, tuple(owner.prepared))
            self.assertEqual(list(owner.prepared), [keys[2], keys[1]])
            frontend._publish_preparation(owner, keys[0], values[0], 64, 1, ('code',), executor, tuple(owner.prepared))
            self.assertEqual(list(owner.prepared), keys[:1])
            self.assertEqual(owner.prepared_bytes, 64)
        owner.clear()
        self.assertFalse(owner.prepared)
        self.assertEqual(owner.prepared_bytes, 0)

    def test_accounting_includes_native_capacity_and_python_key_storage(self):
        graph = jit_tests.lower(program('def f(x):\n return -x'))
        prepared = types.SimpleNamespace(retained_bytes=4096)
        code_key = (graph, 'cuda:0', None)
        key = (code_key, ((13,),), 13, (0,))
        actual = frontend._prepared_entry_bytes(key, prepared, code_key)
        self.assertGreater(actual, 4096 + sys.getsizeof(prepared) + sys.getsizeof(key) + 512)
        larger_key = (code_key, ((1, 1, 13),), 13, (0,))
        self.assertGreater(frontend._prepared_entry_bytes(larger_key, prepared, code_key), actual)
        prepared.retained_bytes += 8192
        self.assertEqual(frontend._prepared_entry_bytes(key, prepared, code_key), actual + 8192)
        # The actual executable key can be independent of the logical shape key
        # and contains exact Program bytes. Charge that retained referent too.
        actual_key = ('direct', bytes(8192))
        self.assertGreater(frontend._prepared_entry_bytes(key, prepared, actual_key),
                           frontend._prepared_entry_bytes(key, prepared, ('direct', b'')) + 8191)

    def test_oversize_is_ephemeral_and_cannot_keep_evicted_executors_alive_in_cache(self):
        compiled = frontend.implementation(program('def f(unused,x):\n return -x'), 2)
        x = native.ones(3)
        compiled(False, x)
        old_code_key = next(iter(cache(compiled).executors))
        self.assertEqual(len(cache(compiled).prepared), 1)
        compile_ = self.codegen.side_effect

        def oversized(*args):
            executor = compile_(*args)
            prepare = executor.prepare

            def prepare_oversized(*inputs):
                value = prepare(*inputs)
                value.retained_bytes = frontend._PREPARED_CACHE_BYTES + 1
                return value

            executor.prepare = Mock(side_effect=prepare_oversized)
            return executor

        self.codegen.side_effect = oversized
        compiled(native.ones(3), x)
        oversized_executor = self.created[-1]
        compiled(native.ones(3), x)
        self.assertEqual(oversized_executor.prepare.call_count, 2)
        self.assertEqual(len(cache(compiled).prepared), 1)
        compiled(native.ones(1), x)
        self.assertNotIn(old_code_key, cache(compiled).executors)
        self.assertFalse(cache(compiled).prepared)
        self.assertEqual(cache(compiled).prepared_bytes, 0)
        self.assertEqual(len(cache(compiled).graphs), 1)

    def test_reset_drops_all_prepared_entries_and_accounting(self):
        compiled = native.compile(program('def f(x):\n return -x'))
        x = native.ones(3)
        compiled(x)
        old = newest_prepared(compiled)
        native.compiler.reset()
        self.assertFalse(cache(compiled).graphs)
        self.assertFalse(cache(compiled).executors)
        self.assertFalse(cache(compiled).prepared)
        self.assertEqual(cache(compiled).prepared_bytes, 0)
        compiled(x)
        self.assertIsNot(newest_prepared(compiled), old)
        self.assertEqual(self.codegen.call_count, 2)

    def test_equal_code_key_does_not_certify_a_replaced_executor_owner(self):
        compiled = native.compile(program('def f(x):\n return -x'))
        x = native.ones(3)
        compiled(x)
        old = newest_prepared(compiled)
        state = cache(compiled)
        code_key, original = next(iter(state.executors.items()))
        replacement = jit_tests.mock_pointwise_executor(original.run)
        bind = replacement.bind

        def owned_bind(host):
            value = bind(host)
            value.belongs_to = lambda executor: executor is replacement
            return value

        replacement.bind = Mock(side_effect=owned_bind)
        state.executors[code_key] = replacement
        with patch.object(old, 'run', side_effect=AssertionError('stale owner executed')):
            compiled(x)
        current = newest_prepared(compiled)
        self.assertIsNot(current, old)
        self.assertTrue(current.belongs_to(replacement))
        self.assertFalse(current.belongs_to(original))
        replacement.bind.assert_called_once()
        self.assertEqual(self.codegen.call_count, 1)

    def test_missing_executor_rebuilds_and_replaces_stale_preparation(self):
        compiled = native.compile(program('def f(x):\n return -x'))
        x = native.ones(3)
        compiled(x)
        stale = newest_prepared(compiled)
        cache(compiled).executors.clear()
        with patch.object(stale, 'run', side_effect=AssertionError('unowned preparation executed')):
            compiled(x)
        self.assertIsNot(newest_prepared(compiled), stale)
        self.assertEqual(self.codegen.call_count, 2)
        self.assertEqual(cache(compiled).prepared_bytes,
                         sum(item[1] for item in cache(compiled).prepared.values()))

    def test_bind_owner_mismatch_and_query_errors_never_run_or_publish(self):
        fn = program('def f(x):\n return -x')
        warm = native.compile(fn)
        warm(native.ones(3))
        warm(native.ones(5))
        self.assertEqual(len(cache(warm).prepared), 2)
        build = self.host_plan.side_effect
        for cold in (False, True):
            for query_error in (False, True):
                compiled = native.compile(fn) if cold else warm
                x = native.ones(3 if cold else 7)
                before = self.snapshot(compiled)
                query = Mock(side_effect=RuntimeError('owner query')) if query_error else Mock(return_value=False)
                wrong = types.SimpleNamespace(input_shapes=(tuple(x.shape),), retained_bytes=0,
                                              belongs_to=query, run=Mock())
                executors = []

                def host_with_wrong_bind(*args):
                    host = build(*args)
                    compile_ = host.compile

                    def wrong_compile():
                        executor = compile_()
                        executor.bind = Mock(return_value=wrong)
                        executors.append(executor)
                        return executor

                    host.compile = wrong_compile
                    return host

                with self.subTest(cold=cold, query_error=query_error), ExitStack() as stack:
                    stack.enter_context(patch.object(bridge, '_pointwise_host_plan', host_with_wrong_bind))
                    if not cold:
                        executor = next(iter(cache(warm).executors.values()))
                        executors.append(executor)
                        stack.enter_context(patch.object(executor, 'bind', return_value=wrong))
                    accounting = stack.enter_context(patch.object(frontend, '_prepared_entry_bytes'))
                    receipt = stack.enter_context(patch.object(frontend, '_receipt'))
                    exception = RuntimeError if query_error else NotImplementedError
                    message = 'owner query' if query_error else 'prepared executable owner mismatch'
                    with self.assertRaisesRegex(exception, message):
                        compiled._torch_rs_pointwise_receipt(x)
                    query.assert_called_once_with(executors[0])
                    wrong.run.assert_not_called()
                    accounting.assert_not_called()
                    receipt.assert_not_called()
                    self.assertEqual(self.snapshot(compiled), before)

    def test_owner_admission_once_per_bind_and_never_on_hits_or_current_pruning(self):
        build = self.host_plan.side_effect
        queries = []

        def tracked_host(*args):
            host = build(*args)
            compile_ = host.compile

            def tracked_compile():
                executor = compile_()
                bind = executor.bind

                def tracked_bind(plan):
                    prepared = bind(plan)
                    query = Mock(wraps=prepared.belongs_to)
                    prepared.belongs_to = query
                    queries.append(query)
                    return prepared

                executor.bind = tracked_bind
                return executor

            host.compile = tracked_compile
            return host

        compiled = native.compile(program('def f(x):\n return -x'))
        x, y = native.ones(3), native.ones(5)
        with patch.object(bridge, '_pointwise_host_plan', tracked_host):
            for value in (x, y, x):
                compiled(value)
        self.assertEqual(len(cache(compiled).prepared), 3)
        self.assertEqual(len(queries), 3)
        for query in queries:
            self.assertEqual(query.call_count, 1)
            # Mutable stand-ins prove absence of queries, not native immutability.
            query.side_effect = AssertionError('repeated immutable owner query')
        with patch.object(bridge, '_pointwise_host_plan', side_effect=AssertionError('warm build')), \
                patch.object(frontend, '_prepared_entry_bytes', side_effect=AssertionError('warm accounting')):
            for value in (x, y, x, y):  # Newest, older, then recency changes.
                compiled(value)
        self.assertEqual([query.call_count for query in queries], [1, 1, 1])
        self.assertEqual(cache(compiled).prepared_bytes,
                         sum(entry[1] for entry in cache(compiled).prepared.values()))

    def test_nonselected_replaced_owner_prunes_all_its_preparations_only_after_success(self):
        fn = program('def f(x):\n return -x')
        compiled = native.compile(fn)
        x, y = native.ones(3), native.ones(5)
        compiled(x)
        compiled(y)
        state = cache(compiled)
        old_entries = list(state.prepared.values())
        self.assertEqual(len(old_entries), 2)
        old_key, old_executor = next(iter(state.executors.items()))
        fn.__code__ = program('def f(x):\n return x+x').__code__
        compiled(y)
        selected = newest_prepared(compiled)
        state.executors[old_key] = jit_tests.mock_pointwise_executor(old_executor.run)
        before = self.snapshot(compiled)
        before_bytes = state.prepared_bytes
        with ExitStack() as stack:
            for prepared, _, _, _ in old_entries:
                stack.enter_context(patch.object(prepared, 'run', side_effect=AssertionError('stale run')))
                stack.enter_context(patch.object(prepared, 'belongs_to', side_effect=AssertionError('prune queried owner')))
            for target, name in ((selected, 'run'), (frontend.ResultSpec, 'reconstruct')):
                with self.subTest(phase=name), patch.object(target, name, side_effect=RuntimeError(name)):
                    with self.assertRaisesRegex(RuntimeError, name):
                        compiled(y)
                    self.assertEqual(self.snapshot(compiled), before)
            compiled(y)
        self.assertEqual(len(state.prepared), 1)
        self.assertIs(newest_prepared(compiled), selected)
        self.assertEqual(state.prepared_bytes, before_bytes - sum(entry[1] for entry in old_entries))
        self.assertEqual(state.prepared_bytes, sum(entry[1] for entry in state.prepared.values()))

    def test_selected_replaced_owner_failures_leave_all_old_entries_until_success(self):
        compiled = native.compile(program('def f(x):\n return -x'))
        x, y = native.ones(3), native.ones(5)
        compiled(x)
        compiled(y)
        state = cache(compiled)
        old_entries = list(state.prepared.values())
        code_key, original = next(iter(state.executors.items()))
        replacement = jit_tests.mock_pointwise_executor(original.run)
        state.executors[code_key] = replacement
        before = self.snapshot(compiled)
        old_bytes = state.prepared_bytes
        with ExitStack() as stack:
            for prepared, _, _, _ in old_entries:
                stack.enter_context(patch.object(prepared, 'run', side_effect=AssertionError('stale run')))
            for target, name in ((replacement, 'bind'), (replacement, 'run'),
                                 (frontend.ResultSpec, 'reconstruct')):
                with self.subTest(phase=name), patch.object(target, name, side_effect=RuntimeError(name)):
                    with self.assertRaisesRegex(RuntimeError, name):
                        compiled(y)
                    self.assertEqual(self.snapshot(compiled), before)
            compiled(y)
        self.assertEqual(len(state.prepared), 1)
        entry = next(iter(state.prepared.values()))
        self.assertIs(entry[3], replacement)
        self.assertTrue(entry[0].belongs_to(replacement))
        self.assertTrue(all(entry[0] is not old[0] for old in old_entries))
        self.assertEqual(state.prepared_bytes, old_bytes - sum(old[1] for old in old_entries) + entry[1])
        self.assertEqual(state.prepared_bytes, sum(item[1] for item in state.prepared.values()))

    def test_actual_program_identity_controls_sharing_and_eviction(self):
        compiled = frontend.implementation(program('def f(x):\n return -x'), 2)
        build = self.host_plan.side_effect

        def explicit_program(*args):
            host = build(*args)
            # Model effective Programs, deliberately distinct from exact shape
            # and raw hint. Native tests establish actual planner equivalence.
            shape = tuple(args[0][0].shape)
            words = b'program-a' if shape[0] in (3, 5, 7) else bytes([shape[0]])
            host.executable_identity = (host.executable_identity, 'direct', words)
            return host

        self.host_plan.side_effect = explicit_program
        for size in (3, 5, 7, 5):
            compiled(native.ones(size))
        self.assertEqual(self.codegen.call_count, 1)
        first_key = next(iter(cache(compiled).executors))
        compiled(native.ones(9))
        self.assertEqual(self.codegen.call_count, 2)
        self.assertEqual(len(cache(compiled).executors), 2)
        compiled(native.ones(11))
        self.assertEqual(self.codegen.call_count, 3)
        self.assertNotIn(first_key, cache(compiled).executors)
        self.assertTrue(all(item[2] in cache(compiled).executors
                            for item in cache(compiled).prepared.values()))
        self.assertTrue(all(item[2] != first_key for item in cache(compiled).prepared.values()))

    def test_host_preparation_and_selected_compile_failures_never_publish(self):
        fn = program('def f(x):\n return -x')
        compiled = native.compile(fn)
        compiled(native.ones(3))
        before = self.snapshot(compiled)
        build = self.host_plan.side_effect
        for phase in ('planning', 'emission', 'compiler discovery', 'compiler create',
                      'compiler compile', 'module load', 'upload'):
            def failure(*args, phase=phase):
                if phase in ('planning', 'emission'):
                    raise RuntimeError(phase)
                host = build(*args)
                host.executable_identity = ('new-code', phase)
                if phase == 'upload':
                    executor = jit_tests.mock_pointwise_executor(Mock())
                    executor.bind = Mock(side_effect=RuntimeError(phase))
                    host.compile = Mock(return_value=executor)
                else:
                    host.compile = Mock(side_effect=RuntimeError(phase))
                return host

            with self.subTest(phase=phase), patch.object(bridge, '_pointwise_host_plan', side_effect=failure):
                with self.assertRaisesRegex(RuntimeError, phase):
                    compiled(native.ones(5))
                self.assertEqual(self.snapshot(compiled), before)
                cold = native.compile(fn)
                with self.assertRaisesRegex(RuntimeError, phase):
                    cold(native.ones(3))
                self.assertEqual(self.snapshot(cold), ([], [], [], 0))

    def test_receipt_allocation_failure_preserves_cold_and_warm_cache_state(self):
        fn = program('def f(x):\n return -x')
        compiled = native.compile(fn)
        x = native.ones(3)
        compiled(x)
        before = self.snapshot(compiled)
        with patch.object(frontend, '_receipt', side_effect=MemoryError('receipt allocation')):
            with self.assertRaisesRegex(MemoryError, 'receipt allocation'):
                compiled._torch_rs_pointwise_receipt(x)
            self.assertEqual(self.snapshot(compiled), before)
            with self.assertRaisesRegex(MemoryError, 'receipt allocation'):
                compiled._torch_rs_pointwise_receipt(native.ones(5))
            self.assertEqual(self.snapshot(compiled), before)
            cold = native.compile(fn)
            with self.assertRaisesRegex(MemoryError, 'receipt allocation'):
                cold._torch_rs_pointwise_receipt(x)
            self.assertEqual(self.snapshot(cold), ([], [], [], 0))

    def test_receipt_identifies_the_actual_warm_preparation(self):
        compiled = native.compile(program('def f(x):\n return [-x,x]'))
        first_input, second_input = native.ones(3), native.ones(5)
        compiled(first_input)
        compiled(second_input)
        # Promotion changes numerical history; retain both shapes under it.
        compiled(first_input)
        first = newest_prepared(compiled)
        compiled(second_input)
        second = newest_prepared(compiled)
        self.assertIsNot(first, second)
        with patch.object(bridge, '_pointwise_host_plan', side_effect=AssertionError('receipt rebuilt plan')):
            output, used = compiled._torch_rs_pointwise_receipt(first_input)
        self.assertIs(used, first)
        self.assertIs(output[1], first_input)
        self.assertIs(newest_prepared(compiled), first)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class PreparedHardware(unittest.TestCase):
    setUpClass = classmethod(jit_tests.Hardware.setUpClass.__func__)
    tearDown = jit_tests.Hardware.tearDown
    upload = jit_tests.Hardware.upload
    compare = jit_tests.Hardware.compare
    without_replay = jit_tests.Hardware.without_replay

    def setUp(self):
        from torch._inductor.utils import fresh_cache

        # Default Inductor may reuse a dynamic graph compiled by another test
        # at a different size, including its size-dependent numerical schedule.
        # Keep each test's persistent history intact without leaking its disk
        # cache to unrelated tests. Retain generated artifacts for diagnosis.
        directory = Path(__file__).resolve().parents[1] / 'target' / 'prepared-reference-caches'
        directory.mkdir(parents=True, exist_ok=True)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(fresh_cache(dir=str(directory), delete=False))

    def test_revisited_shape_matches_default_inductor_selected_numerical_history(self):
        source = 'def f(x):\n p=x*x\n return (-p,p.sin())'
        fn = program(source)
        compiled = native.compile(fn)
        reference = self.torch.compile(program(source, self.torch))
        preparations = []
        # Keep both wrappers and the old preparation alive across promotion.
        # Tiny positive products expose the selected plan's signed-zero policy.
        for count in (2, 13, 2, 2):
            values = [2.0 ** -80] * count
            x = self.upload(values, (count,))
            tx = self.upload(values, (count,), self.torch)
            outputs = self.without_replay(fn, compiled, (x,))
            expected = reference(tx)
            for actual, wanted in zip(outputs, expected):
                self.compare(actual, wanted)
                host = self.torch.tensor(actual.cpu().tolist(), dtype=self.torch.float32)
                self.assertTrue(self.torch.equal(host.view(self.torch.int32),
                                                wanted.cpu().view(self.torch.int32)))
            preparations.append(newest_prepared(compiled))
        self.assertIsNot(preparations[0], preparations[2])
        self.assertIs(preparations[2], preparations[3])
        self.assertEqual({key[2] for key in cache(compiled).prepared if key[1] == ((2,),)}, {2, 13})

    def test_actual_warm_preparation_reuse_and_retained_result_freshness(self):
        source = 'def f(x):\n a=x*x\n return (a,a+x)'
        fn = program(source)
        compiled = native.compile(fn)
        reference = self.torch.compile(program(source, self.torch))
        retained = []
        with patch.object(bridge, '_pointwise_host_plan', wraps=bridge._pointwise_host_plan) as build:
            for value in (0.25, -0.75, 1.125):
                x = self.upload([value] * 13, (13,))
                tx = self.upload([value] * 13, (13,), self.torch)
                outputs = self.without_replay(fn, compiled, (x,))
                expected = reference(tx)
                for actual, wanted in zip(outputs, expected):
                    self.compare(actual, wanted)
                    self.assertNotEqual(actual.data_ptr(), x.data_ptr())
                for earlier, earlier_expected in retained:
                    for actual, wanted in zip(earlier, earlier_expected):
                        self.compare(actual, wanted)
                    self.assertTrue(all(a.data_ptr() != b.data_ptr() for a in outputs for b in earlier))
                retained.append((outputs, expected))
        self.assertEqual(len(cache(compiled).executors), 1)
        self.assertEqual(build.call_count, 1)
        prepared = newest_prepared(compiled)
        self.assertEqual(prepared.input_shapes, ((13,),))
        self.assertGreater(prepared.retained_bytes, 0)

    def test_private_prepared_admission_cannot_bypass_graph_or_shape_checks(self):
        fn = program('def f(x,unused):\n return x.sin()')
        x = self.upload([1., 2.], (2,))
        compiled = native.compile(fn)
        compiled(x, x)
        executor = next(iter(cache(compiled).executors.values()))
        graph = jit_tests.lower(fn, 2)
        legacy = bridge._pointwise_compile((x, x), graph.nodes, graph.outputs)
        prepared = newest_prepared(compiled)
        for name in ('input_shapes', 'retained_bytes'):
            with self.assertRaises(AttributeError):
                setattr(prepared, name, None)
        reshaped = x.reshape((1, 2))
        with self.assertRaisesRegex(RuntimeError, 'shape guard'):
            prepared.run((x, reshaped))
        # Public admission now accepts this graph, but this old kernel has a
        # different address map for its unused argument. Neither guard changes.
        with self.assertRaisesRegex(RuntimeError, 'broadcast indexing guard'):
            legacy.prepare((x, reshaped))
        host = bridge._pointwise_host_plan((x, reshaped), graph.nodes, graph.outputs, 2, (0,))
        with self.assertRaises(RuntimeError):
            executor.bind(host)
        for inputs in ((x,), (x, native.ones(2)), (x, object())):
            with self.subTest(inputs=len(inputs)), self.assertRaises((RuntimeError, TypeError)):
                prepared.run(inputs)
        effects = []

        class Scalar:
            def __float__(self):
                effects.append('float')
                return 1.

        for scalar in (Scalar(), True, 1):
            with self.assertRaises(TypeError):
                prepared.run((x, x), (scalar,))
        self.assertEqual(effects, [])
        with self.assertRaises(RuntimeError):
            prepared.run((x, x), (1.,))
        expected = self.torch.compile(program('def f(x):\n return x.sin()', self.torch))(
            self.upload([1., 2.], (2,), self.torch))
        self.compare(prepared.run((x, x))[0], expected)
        self.compare(self.without_replay(fn, compiled, (x, reshaped)), expected)
        native.compiler.reset()
        self.assertFalse(cache(compiled).prepared)
        self.compare(prepared.run((x, x))[0], expected)

    def test_rank_zero_length_one_and_empty_broadcast_keep_numerical_domains(self):
        source = 'def f(x):\n p=x*x\n return (-p,p.sin())'
        compiled = native.compile(program(source))
        reference = self.torch.compile(program(source, self.torch))
        for shape in ((), (1,), ()):
            x = self.upload([1e-38], shape)
            tx = self.upload([1e-38], shape, self.torch)
            for actual, expected in zip(compiled(x), reference(tx)):
                self.compare(actual, expected, exact=True)
        self.assertEqual({key[1] for key in cache(compiled).prepared}, {((),), ((1,),)})
        source = 'def f(x,y):\n return x+y'
        compiled = native.compile(program(source))
        reference = self.torch.compile(program(source, self.torch))
        for rows in (0, 2, 0):
            x, y = self.upload([2.] * rows, (rows, 1)), self.upload([1., 2., 3.], (1, 3))
            tx = self.upload([2.] * rows, (rows, 1), self.torch)
            ty = self.upload([1., 2., 3.], (1, 3), self.torch)
            self.compare(compiled(x, y), reference(tx, ty), exact=True)
        self.assertEqual(len(cache(compiled).prepared), 2)

    @unittest.skipUnless(two_device_reservation(), 'requires explicit two-device reservation')
    def test_private_preparation_and_run_restore_device_and_reject_cross_device(self):
        if self.torch.cuda.device_count() < 2:
            self.skipTest('requires two CUDA devices')
        compiled = native.compile(program('def f(x):\n return x*2.0'))
        inputs = [native.tensor([1., 2.]).to(f'cuda:{device}') for device in (0, 1)]
        compiled(inputs[0])
        prepared = newest_prepared(compiled)
        executor = next(iter(cache(compiled).executors.values()))
        graph = jit_tests.lower(program('def f(x):\n return x*2.0'))
        legacy = bridge._pointwise_compile((inputs[0],), graph.nodes, graph.outputs)
        with self.torch.cuda.device(1):
            for action in (lambda: prepared.run((inputs[1],)), lambda: legacy.prepare((inputs[1],))):
                with self.assertRaises(RuntimeError):
                    action()
                self.assertEqual(self.torch.cuda.current_device(), 1)
            host = bridge._pointwise_host_plan((inputs[0],), graph.nodes, graph.outputs, 2, (0,))
            fresh = executor.bind(host)
            self.assertEqual(self.torch.cuda.current_device(), 1)
            self.assertEqual(fresh.run((inputs[0],))[0].cpu().tolist(), [2., 4.])
            self.assertEqual(self.torch.cuda.current_device(), 1)
        compiled(inputs[1])
        self.assertIsNot(newest_prepared(compiled), prepared)
        self.assertEqual(len(cache(compiled).executors), 2)
        compiled(inputs[0])
        self.assertIs(newest_prepared(compiled), prepared)


if __name__ == '__main__':
    unittest.main()
