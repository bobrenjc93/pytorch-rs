"""Portable checks for guard ownership and bounded prepared-owner cleanup."""
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend
from tests.test_compile_pointwise_jit import cache, program
from tests import test_compile_pointwise_prepared as prepared_tests


class WarmOverhead(unittest.TestCase):
    setUp = prepared_tests.PreparedCache.setUp
    snapshot = prepared_tests.PreparedCache.snapshot

    def test_all_owner_identities_reject_before_admission_cold_and_warm(self):
        # Derive the expected inventory independently of its grouped storage.
        owners = (native.Tensor, native.Tensor.__base__)
        expected = [(owner, name, owner.__dict__.get(name, frontend._MISSING))
                    for owner in owners for name in frontend._METHODS]
        self.assertEqual([(owner, name, value) for owner, checks in frontend._METHOD_GUARDS
                          for name, value in checks], expected)
        x = native.ones(3)
        for warm in (False, True):
            for owner, name, value in expected:
                # Adding a previously missing identity and deleting an existing
                # identity are both changes, even if inheritance resolves it.
                for delete in ((False, True) if value is not frontend._MISSING else (False,)):
                    with self.subTest(owner=owner, name=name, warm=warm, delete=delete):
                        compiled = native.compile(program('def f(x):\n return -x'))
                        if warm:
                            compiled(x)
                        before = self.snapshot(compiled)
                        self.admit.reset_mock()
                        try:
                            if delete:
                                delattr(owner, name)
                            else:
                                setattr(owner, name, lambda *args: None)
                            with self.assertRaises(NotImplementedError) as error:
                                compiled(x)
                        finally:
                            if value is frontend._MISSING:
                                delattr(owner, name)
                            else:
                                setattr(owner, name, value)
                        self.assertEqual(str(error.exception),
                                         'torch.compile(): native CUDA pointwise: '
                                         'patched Tensor operation binding: ' + name)
                        self.admit.assert_not_called()
                        self.assertEqual(self.snapshot(compiled), before)
                        compiled(x)
                        self.admit.assert_called_once_with((x,))

    def test_owner_and_method_rejection_order_is_unchanged(self):
        x = native.ones(3)
        compiled = native.compile(program('def f(x):\n return -x'))
        compiled(x)
        before = self.snapshot(compiled)
        public, base = native.Tensor, native.Tensor.__base__
        for replacements, first in (
                (((public, 'cos'), (public, 'sin')), 'sin'),
                (((base, 'neg'), (public, 'mul')), 'mul')):
            old = [(owner, name, owner.__dict__.get(name, frontend._MISSING))
                   for owner, name in replacements]
            self.admit.reset_mock()
            try:
                for owner, name in replacements:
                    setattr(owner, name, lambda *args: None)
                with self.assertRaisesRegex(NotImplementedError, 'binding: ' + first + '$'):
                    compiled(x)
            finally:
                for owner, name, value in old:
                    if value is frontend._MISSING:
                        delattr(owner, name)
                    else:
                        setattr(owner, name, value)
            self.admit.assert_not_called()
            self.assertEqual(self.snapshot(compiled), before)

    def test_each_invocation_reads_two_namespaces_and_every_identity(self):
        reads, checks_seen = [], []

        class Namespace:
            def __init__(self, owner):
                self.owner = owner

            def get(self, name, missing):
                checks_seen.append((self.owner, name))
                return self.owner.__dict__.get(name, missing)

        class Owner:
            def __init__(self, owner):
                self.owner = owner

            @property
            def __dict__(self):
                reads.append(self.owner)
                return Namespace(self.owner)

        guards = tuple((Owner(owner), checks) for owner, checks in frontend._METHOD_GUARDS)
        owners = [native.Tensor, native.Tensor.__base__]
        compiled = native.compile(program('def f(unused,x):\n return -x'))
        x = native.ones(3)
        with patch.object(frontend, '_METHOD_GUARDS', guards):
            for args in ((False, x), (False, x), (x, x), (x, x)):
                reads.clear()
                checks_seen.clear()
                self.admit.reset_mock()
                compiled(*args)
                self.assertEqual(reads, owners)
                self.assertEqual(checks_seen, [(owner, name) for owner in owners
                                               for name in frontend._METHODS])
                self.admit.assert_called_once_with(tuple(v for v in args if type(v) is native.Tensor))

    def test_recency_preserves_preparations_accounting_and_reset(self):
        # Same-key owner replacement requires a bounded scan even without
        # eviction. Preserve the recency/retention checks, not the old no-scan
        # assertion, under the native executable ownership contract.
        compiled = frontend.implementation(program('def f(x):\n return -x'), 8)
        for size in (3, 5, 3):
            compiled(native.ones(size))
        state = cache(compiled)
        for size in (3, 5, 3, 7):
            compiled(native.ones(size))
            self.assertEqual(state.prepared_bytes, sum(v[1] for v in state.prepared.values()))
        alternating, x, other = self.two_executors()
        values = dict(other.prepared.items())
        charge = other.prepared_bytes
        for unused in (False, native.ones(3), False, native.ones(3)):
            before = list(other.prepared.items())
            alternating(unused, x)
            selected = next(reversed(other.prepared))
            self.assertEqual(list(other.prepared.items()),
                             [(key, value) for key, value in before if key != selected]
                             + [(selected, values[selected])])
            for key, value in before:
                self.assertIs(other.prepared[key][0], value[0])
                self.assertEqual(other.prepared[key][1], value[1])
                if key != selected:
                    self.assertIs(other.prepared[key], value)
            self.assertEqual(other.prepared_bytes, charge)
        self.assertEqual(len(other.executors), 2)
        native.compiler.reset()
        self.assertEqual(state.prepared_bytes, 0)
        self.assertFalse(state.prepared)
        compiled(native.ones(3))

    def two_executors(self):
        compiled = frontend.implementation(program('def f(unused,x):\n return -x'), 2)
        x = native.ones(3)
        compiled(False, x)
        compiled(native.ones(3), x)
        return compiled, x, cache(compiled)

    def assert_retention(self, state, limit=2):
        self.assertLessEqual(len(state.executors), limit)
        self.assertLessEqual(len(state.prepared), limit)
        self.assertLessEqual(state.prepared_bytes, frontend._PREPARED_CACHE_BYTES)
        self.assertEqual(state.prepared_bytes, sum(value[1] for value in state.prepared.values()))
        self.assertTrue(all(state.executors.get(value[2]) is value[3] for value in state.prepared.values()))

    def test_recency_reinsertion_failure_drops_preparations_and_preserves_other_executors(self):
        compiled, x, state = self.two_executors()
        first, second = state.executors
        survivor = state.executors[second]
        failure = MemoryError('executor reinsertion')

        class FailReinsertion(dict):
            def __setitem__(self, key, value):
                if key == first:
                    raise failure
                super().__setitem__(key, value)

        state.executors = FailReinsertion(state.executors)
        with self.assertRaises(MemoryError) as error:
            compiled(False, x)
        self.assertIs(error.exception, failure)
        self.assertEqual(list(state.executors), [second])
        self.assertIs(state.executors[second], survivor)
        self.assertFalse(state.prepared)
        self.assertEqual(state.prepared_bytes, 0)
        self.assert_retention(state)
        # Calling the other survivor must not rebuild the lost key to mask an orphan.
        compiled(native.ones(3), x)
        self.assertEqual(self.codegen.call_count, 2)
        self.assertEqual(list(state.executors), [second])
        self.assert_retention(state)
        state.executors = dict(state.executors)
        compiled(False, x)
        self.assertEqual(self.codegen.call_count, 3)
        self.assert_retention(state)
        native.compiler.reset()
        self.assertFalse(state.graphs)
        self.assertFalse(state.executors)
        self.assertFalse(state.prepared)
        self.assertEqual(state.prepared_bytes, 0)
        compiled(False, x)
        self.assert_retention(state)

    def test_warm_owner_replacement_staging_failure_preserves_all_maps(self):
        compiled, x, state = self.two_executors()
        selected_key = next(iter(state.prepared))
        old_prepared, _, code_key, original = state.prepared[selected_key]
        replacement = prepared_tests.jit_tests.mock_pointwise_executor(original.run)
        state.executors[code_key] = replacement
        before = self.snapshot(compiled)
        failure = MemoryError('warm owner cleanup snapshot')

        class FailSnapshot(dict):
            def __iter__(self):
                raise failure

        state.prepared = FailSnapshot(state.prepared)
        with self.assertRaises(MemoryError) as raised:
            compiled(False, x)
        self.assertIs(raised.exception, failure)
        self.assertEqual(self.snapshot(compiled), before)
        self.assertIs(state.prepared[selected_key][0], old_prepared)
        self.assertIs(state.executors[code_key], replacement)
        state.prepared = dict(state.prepared.items())
        compiled(False, x)
        self.assertIsNot(state.prepared[selected_key][0], old_prepared)
        self.assertIs(state.prepared[selected_key][3], replacement)
        self.assert_retention(state)

    def test_repeated_failed_eviction_staging_preserves_capacity_and_survivors(self):
        for newest in (False, True):
            with self.subTest(newest=newest):
                initial_compiles = self.codegen.call_count
                compiled, x, state = self.two_executors()
                first, second = state.executors
                old_executors = list(state.executors.items())
                old_preparations = list(state.prepared.items())
                old_charge = state.prepared_bytes
                before = self.snapshot(compiled)
                failure = MemoryError('prepared key snapshot')

                class FailSnapshot(dict):
                    fail = True

                    def __iter__(self):
                        if self.fail:
                            raise failure
                        return super().__iter__()

                state.prepared = FailSnapshot(state.prepared)
                # The mock host assigns distinct executable keys to unused-input
                # shapes, unlike native direct-code sharing. This isolates
                # capacity-increasing publication misses under the same fault.
                misses = (1, 2, 4, 5, 6)
                for size in misses:
                    with self.subTest(size=size):
                        with self.assertRaises(MemoryError) as error:
                            compiled(native.ones(size), x)
                        self.assertIs(error.exception, failure)
                        self.assertEqual(self.snapshot(compiled), before)
                        self.assertLessEqual(len(state.executors), 2)
                        self.assertLessEqual(len(state.prepared), 2)
                        self.assertEqual(list(state.executors.items()), old_executors)
                        self.assertEqual(list(state.prepared.items()), old_preparations)
                        for key, value in old_executors:
                            self.assertIs(state.executors[key], value)
                        for key, value in old_preparations:
                            self.assertIs(state.prepared[key], value)
                            self.assertIs(state.executors[value[2]], value[3])
                        self.assertEqual(state.prepared_bytes, old_charge)
                        self.assertEqual(state.prepared_bytes,
                                         sum(value[1] for value in state.prepared.values()))
                        self.assertLessEqual(state.prepared_bytes, frontend._PREPARED_CACHE_BYTES)
                state.prepared.fail = False
                selected = second if newest else first
                survivor = state.executors[selected]
                compiled(native.ones(3) if newest else False, x)
                self.assertEqual(self.codegen.call_count, initial_compiles + 2 + len(misses))
                self.assertIs(next(reversed(state.executors.values())), survivor)
                self.assertEqual(list(state.executors), [first, second] if newest else [second, first])
                self.assert_retention(state)
                for key, value in old_preparations:
                    self.assertIs(state.prepared[key][0], value[0])
                    self.assertEqual(state.prepared[key][1], value[1])
                self.assertEqual(state.prepared_bytes, old_charge)
                # Only a successful later miss evicts the oldest retained owner.
                evicted = next(iter(state.executors))
                compiled(native.ones(1), x)
                third = next(reversed(state.executors))
                self.assertEqual(list(state.executors), [selected, third])
                self.assertNotIn(evicted, state.executors)
                self.assertIs(state.executors[selected], survivor)
                selected_key, selected_value = old_preparations[1 if newest else 0]
                self.assertEqual(next(iter(state.prepared)), selected_key)
                self.assertIs(state.prepared[selected_key][0], selected_value[0])
                self.assert_retention(state)
                native.compiler.reset()
                self.assertFalse(state.graphs)
                self.assertFalse(state.prepared)
                self.assertFalse(state.executors)
                self.assertEqual(state.prepared_bytes, 0)
                compiled(False, x)
                self.assert_retention(state)

    def test_eviction_with_no_retained_preparations(self):
        compiled, x, state = self.two_executors()
        first = next(iter(state.executors))
        state.prepared.clear()
        state.prepared_bytes = 0
        with patch.object(frontend, '_prepared_entry_bytes', return_value=frontend._PREPARED_CACHE_BYTES + 1):
            compiled(native.ones(1), x)
        self.assertNotIn(first, state.executors)
        self.assertFalse(state.prepared)
        self.assert_retention(state)

    def test_executor_eviction_preserves_surviving_preparation_order_and_charge(self):
        compiled = frontend.implementation(program('def f(unused,x):\n return -x'), 2)
        x = native.ones(3)
        compiled(False, x)
        state = cache(compiled)
        first_key = next(iter(state.prepared))
        compiled(native.ones(3), x)
        second_key = next(reversed(state.prepared))
        compiled(False, x)
        self.assertEqual(list(state.prepared), [second_key, first_key])
        first_value = state.prepared[first_key]
        compiled(native.ones(1), x)
        third_key = next(reversed(state.prepared))
        self.assertEqual(list(state.prepared), [first_key, third_key])
        self.assertIs(state.prepared[first_key], first_value)
        self.assertEqual(list(state.executors), [first_value[2], state.prepared[third_key][2]])
        self.assertEqual(state.prepared_bytes, first_value[1] + state.prepared[third_key][1])

    def test_executor_eviction_removes_all_its_preparations_only_after_success(self):
        fn = program('def f(unused,x):\n return -x')
        compiled = frontend.implementation(fn, 3)
        x = native.ones(3)
        for size in (3, 5, 3):
            compiled(False, native.ones(size))
        state = cache(compiled)
        evicted = next(iter(state.executors))
        self.assertEqual(len(state.prepared), 3)
        # Other executors are deliberately oversized: retain the three old
        # plans until executor eviction, without prepared count/byte eviction.
        with patch.object(frontend, '_prepared_entry_bytes', return_value=frontend._PREPARED_CACHE_BYTES + 1):
            for unused in (native.ones(3), native.ones(1)):
                compiled(unused, x)
            before = self.snapshot(compiled)
            unused = native.tensor(1.)
            with patch.object(frontend.ResultSpec, 'reconstruct', side_effect=MemoryError('reconstruct')):
                with self.assertRaises(MemoryError):
                    compiled(unused, x)
            self.assertEqual(self.snapshot(compiled), before)
            compiled(unused, x)
        self.assertNotIn(evicted, state.executors)
        self.assertFalse(state.prepared)
        self.assertEqual(state.prepared_bytes, 0)
        self.assertEqual(len(state.executors), 3)
        compiled(unused, x)
        self.assertEqual(len(state.prepared), 1)
        self.assertEqual(state.prepared_bytes, sum(v[1] for v in state.prepared.values()))


@unittest.skipUnless(prepared_tests.available(), 'requires native CUDA and reference PyTorch CUDA')
class WarmOverheadHardware(unittest.TestCase):
    setUpClass = classmethod(prepared_tests.PreparedHardware.setUpClass.__func__)
    setUp = prepared_tests.PreparedHardware.setUp
    tearDown = prepared_tests.PreparedHardware.tearDown
    upload = prepared_tests.PreparedHardware.upload
    compare = prepared_tests.PreparedHardware.compare
    without_replay = prepared_tests.PreparedHardware.without_replay

    def test_public_call_eviction_preserves_current_inputs_outputs_and_reset(self):
        fn = program('def f(y,x):\n return [y-x,x]')
        compiled = native.compile(fn)
        reference = self.torch.compile(program('def f(y,x):\n return [y-x,x]', self.torch))
        retained = []
        old_key = None
        distinct_keys = set()
        # Changing an ignored shape cannot force native executable eviction:
        # direct code shares identical Programs. These used broadcast operands
        # change the actual address formula while retaining two logical entries.
        for index, size in enumerate((1, *range(2, 11), 1)):
            values = [0.25 + index, -0.5 - index, 1. + index]
            x = self.upload(values, (3, 1))
            tx = self.upload(values, (3, 1), self.torch)
            y = self.upload([0.] * size, (size,))
            ty = self.upload([0.] * size, (size,), self.torch)
            output = self.without_replay(fn, compiled, (y, x))
            expected = reference(ty, tx)
            self.compare(output[0], expected[0])
            self.assertIs(output[1], x)
            self.assertTrue(all(output[0].data_ptr() != earlier[0].data_ptr()
                                for _, earlier in retained))
            retained.append((size, output))
            for earlier_index, (earlier_size, earlier) in enumerate(retained):
                self.assertEqual(earlier[0].cpu().tolist(),
                                 [[value] * earlier_size for value in
                                  (-0.25 - earlier_index, 0.5 + earlier_index, -1. - earlier_index)])
            state = cache(compiled)
            selected_key = next(reversed(state.executors))
            if old_key is None:
                old_key = selected_key
            if index < 10:
                self.assertNotIn(selected_key, distinct_keys)
                distinct_keys.add(selected_key)
            if size == 10:
                self.assertNotIn(old_key, state.executors)
                self.assertEqual(len(state.executors), 8)
            self.assertLessEqual(len(state.graphs), 2)
            self.assertLessEqual(len(state.executors), 8)
            self.assertLessEqual(len(state.prepared), 8)
            self.assertTrue(all(state.executors.get(value[2]) is value[3]
                                for value in state.prepared.values()))
            self.assertEqual(state.prepared_bytes, sum(value[1] for value in state.prepared.values()))
        self.assertEqual(len(distinct_keys), 10)
        native.compiler.reset()
        self.assertFalse(state.graphs)
        self.assertFalse(state.executors)
        self.assertFalse(state.prepared)
        self.assertEqual(state.prepared_bytes, 0)
        self.compare(compiled(y, x)[0], reference(ty, tx)[0])
        for index, (size, earlier) in enumerate(retained):
            self.assertEqual(earlier[0].cpu().tolist(),
                             [[value] * size for value in (-0.25 - index, 0.5 + index, -1. - index)])


if __name__ == '__main__':
    unittest.main()
