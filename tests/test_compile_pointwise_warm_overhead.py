"""Portable checks for guard ownership and event-local prepared-plan cleanup."""
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

    def test_no_preparation_scan_without_executor_eviction(self):
        class NoScan(dict):
            def __iter__(self):
                raise AssertionError('prepared scan without executor eviction')

        class NoMembership(dict):
            def __contains__(self, key):
                raise AssertionError('executor membership scan')

        compiled = frontend.implementation(program('def f(x):\n return -x'), 8)
        for size in (3, 5, 3):
            compiled(native.ones(size))
        state = cache(compiled)
        state.prepared = NoScan(state.prepared)
        state.executors = NoMembership(state.executors)
        for size in (3, 5, 3, 7):
            compiled(native.ones(size))
            self.assertEqual(state.prepared_bytes, sum(v[1] for v in state.prepared.values()))
        native.compiler.reset()
        self.assertEqual(state.prepared_bytes, 0)
        self.assertFalse(state.prepared)
        compiled(native.ones(3))

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
        self.assertEqual(list(state.executors), [first_key[0], third_key[0]])
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
        fn = program('def f(unused,x):\n return [-x,x]')
        compiled = native.compile(fn)
        reference = self.torch.compile(program('def f(unused,x):\n return [-x,x]', self.torch))
        retained = []
        old_key = None
        # The ignored input changes the native ABI/address formula while the
        # used input keeps the same logical guard. Cross the default count bound.
        for size in (None, *range(1, 10), None):
            x = self.upload([0.25, -0.5, 1.], (3,))
            tx = self.upload([0.25, -0.5, 1.], (3,), self.torch)
            unused = False if size is None else self.upload([1.] * size, (size,))
            tu = False if size is None else self.upload([1.] * size, (size,), self.torch)
            output = self.without_replay(fn, compiled, (unused, x))
            expected = reference(tu, tx)
            self.compare(output[0], expected[0])
            self.assertIs(output[1], x)
            self.assertTrue(all(output[0].data_ptr() != earlier[0].data_ptr() for earlier in retained))
            retained.append(output)
            state = cache(compiled)
            if old_key is None:
                old_key = next(iter(state.executors))
            if size == 9:
                self.assertNotIn(old_key, state.executors)
            self.assertTrue(all(key[0] in state.executors for key in state.prepared))
            self.assertEqual(state.prepared_bytes, sum(value[1] for value in state.prepared.values()))
        native.compiler.reset()
        self.assertFalse(state.executors)
        self.assertFalse(state.prepared)
        self.assertEqual(state.prepared_bytes, 0)
        self.compare(compiled(False, x)[0], reference(False, tx)[0])


if __name__ == '__main__':
    unittest.main()
