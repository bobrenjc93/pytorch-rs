"""Failure injection at the logical commit and derived ownership boundary."""
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend
from tests.test_compile_pointwise_jit import cache, program, available
from tests import test_compile_pointwise_prepared as prepared_tests
from tests import test_compile_pointwise_warm_overhead as warm_tests


class PublicationInterrupt(BaseException):
    pass


class StageRecent(unittest.TestCase):
    def test_identical_newest_replacement_preserves_old_map_and_key_order(self):
        first, newest = object(), object()
        first_value, old, replacement = object(), object(), object()
        mapping = {first: first_value, newest: old}
        staged = frontend._stage_recent(mapping, newest, replacement, 2)
        self.assertIsNot(staged, mapping)
        for actual, expected in zip(staged, (first, newest)):
            self.assertIs(actual, expected)
        self.assertEqual(tuple(mapping), (first, newest))
        self.assertIs(mapping[newest], old)
        self.assertIs(mapping[first], first_value)
        self.assertIs(staged[first], first_value)
        self.assertIs(staged[newest], replacement)

    def test_identical_newest_replacement_enforces_capacity_even_when_over_limit(self):
        keys = tuple(object() for _ in range(4))
        values = tuple(object() for _ in keys)
        mapping = dict(zip(keys, values))
        replacement = object()
        for limit in (4, 2, 1, 0):
            with self.subTest(limit=limit):
                staged = frontend._stage_recent(mapping, keys[-1], replacement, limit)
                self.assertIsNot(staged, mapping)
                self.assertEqual(tuple(staged), keys[-limit:] if limit else ())
                self.assertEqual(tuple(mapping), keys)
                for key, value in zip(keys, values):
                    self.assertIs(mapping[key], value)
                if limit:
                    self.assertIs(next(reversed(staged)), keys[-1])
                    self.assertIs(staged[keys[-1]], replacement)

    def test_older_promotion_insertion_eviction_and_unchanged_reuse(self):
        first, newest, new = object(), object(), object()
        value = object()
        mapping = {first: value, newest: value}
        # Even an over-limit unchanged hit must retain the original map.
        self.assertIs(frontend._stage_recent(mapping, newest, value, 1), mapping)
        promoted = frontend._stage_recent(mapping, first, value, 2)
        self.assertIsNot(promoted, mapping)
        self.assertEqual(tuple(promoted), (newest, first))
        inserted = frontend._stage_recent(mapping, new, value, 2)
        self.assertEqual(tuple(inserted), (newest, new))
        self.assertIs(inserted[new], value)
        self.assertEqual(tuple(mapping), (first, newest))

    def test_only_identical_newest_skips_pop(self):
        class CountPop(dict):
            def __init__(self, values=()):
                super().__init__(values)
                self.pops = []

            def copy(self):
                return CountPop(self)

            def pop(self, key, *default):
                self.pops.append(key)
                return super().pop(key, *default)

        first, key = object(), (1, (2,))
        equal = tuple([1, tuple([2])])
        self.assertIsNot(key, equal)
        old, replacement = object(), object()
        mapping = CountPop({first: old, key: old})
        for incoming, expected_pops in ((key, 0), (equal, 1), (first, 1), (None, 1)):
            with self.subTest(incoming=incoming):
                staged = frontend._stage_recent(mapping, incoming, replacement, 2)
                self.assertEqual(len(staged.pops), expected_pops)
                self.assertIs(next(reversed(staged)), incoming)
                self.assertIs(staged[incoming], replacement)
        # None is also a valid key: an empty map cannot qualify by sentinel identity.
        empty = CountPop()
        staged = frontend._stage_recent(empty, None, replacement, 1)
        self.assertEqual(staged.pops, [None])
        self.assertIs(staged[None], replacement)
        self.assertFalse(empty)
        self.assertEqual(mapping.pops, [])


def logical_snapshot(owner):
    return (id(owner.graphs), tuple(
        (key, id(key), id(entry), id(entry.lowerings),
         tuple((abi, id(abi), id(value)) for abi, value in entry.lowerings.items()),
         id(entry.payload),
         tuple((name, id(getattr(entry.payload, name))) for name in entry.payload._fields),
         tuple((source, id(value), repr(value)) for source, value in entry.payload.values.items()),
         tuple(entry.payload.observations.items()))
        for key, entry in owner.graphs.items()))


def snapshot(owner):
    return (logical_snapshot(owner), id(owner.executors),
            tuple((key, id(value)) for key, value in owner.executors.items()),
            id(owner.prepared), tuple((key, id(value), id(value[0]), value[1], id(value[3]))
                                      for key, value in owner.prepared.items()), owner.prepared_bytes)


class FaultDict(dict):
    """Fail before or after a chosen actual dict mutation; cleanup must bypass us."""
    def __init__(self, values, operation, failure, after, predicate=lambda key: True):
        super().__init__(values)
        self.operation, self.failure, self.after = operation, failure, after
        self.predicate, self.fired = predicate, 0

    def mutate(self, operation, key, action):
        if operation == self.operation and self.predicate(key):
            if self.after:
                action()
            self.fired += 1
            raise self.failure
        return action()

    def pop(self, key, *default):
        return self.mutate('pop', key, lambda: dict.pop(self, key, *default))

    def __setitem__(self, key, value):
        return self.mutate('set', key, lambda: dict.__setitem__(self, key, value))

    def __delitem__(self, key):
        return self.mutate('delete', key, lambda: dict.__delitem__(self, key))

    def clear(self):
        raise AssertionError('recovery must use base dict.clear')


class Publication(unittest.TestCase):
    setUp = prepared_tests.PreparedCache.setUp
    two_executors = warm_tests.WarmOverhead.two_executors
    assert_retention = warm_tests.WarmOverhead.assert_retention

    def unlocked(self, owner):
        self.assertTrue(owner.lock.acquire(blocking=False))
        owner.lock.release()

    def test_single_owner_records_defaults_and_replacement_identity(self):
        names = ('values', 'observed', 'observations', 'binding_checks',
                 'tensor_sources', 'data_sources', 'numerical_hint')
        self.assertEqual(frontend.SpecializationPayload._fields, names)
        self.assertEqual(frontend.Specialization._fields, ('payload', 'lowerings'))
        default = frontend.SpecializationPayload({}, (), {})
        self.assertEqual(default._field_defaults, dict(
            binding_checks=(), tensor_sources=(), data_sources=(), numerical_hint=1))
        for name, expected in default._field_defaults.items():
            self.assertEqual(getattr(default, name), expected)
        fields = tuple(object() for _ in names)
        payload = frontend.SpecializationPayload(*fields)
        old_lowerings, new_lowerings = {}, {}
        entry = frontend.Specialization(payload, old_lowerings)
        replacement = frontend.Specialization(payload, new_lowerings)
        from tests.test_compile_pointwise_helpers import assert_replaced_shell
        assert_replaced_shell(self, entry, replacement)
        self.assertIs(entry.lowerings, old_lowerings)
        self.assertIs(replacement.lowerings, new_lowerings)
        for name, value in zip(names, fields):
            self.assertIs(getattr(payload, name), value)
            self.assertFalse(hasattr(entry, name))
        for record in (default, payload, entry, replacement):
            self.assertFalse(hasattr(record, '__dict__'))
            for name in record._fields:
                with self.assertRaises(AttributeError):
                    setattr(record, name, object())
        # Freezing is shallow: construction retains the caller's field objects.
        values, observations = {}, {}
        shallow = frontend.SpecializationPayload(values, (), observations)
        values['value'] = object()
        observations['observation'] = object()
        self.assertIs(shallow.values, values)
        self.assertIs(shallow.observations, observations)

    def test_cold_payload_failure_with_existing_history_does_not_publish(self):
        for exception in (MemoryError, PublicationInterrupt):
            with self.subTest(exception=exception):
                other, _, other_owner = self.two_executors()
                other_before = snapshot(other_owner)
                compiled = frontend.implementation(program('def f(x):\n return -x'), 2)
                compiled(native.ones(3))
                owner = cache(compiled)
                before = snapshot(owner)
                failure = exception('cold payload')
                with patch.object(frontend, 'SpecializationPayload', side_effect=failure) as fault, \
                        patch.object(frontend, '_publish_preparation') as publish:
                    with self.assertRaises(exception) as raised:
                        compiled(native.ones(5))
                self.assertIs(raised.exception, failure)
                fault.assert_called_once()
                publish.assert_not_called()
                self.assertEqual(snapshot(owner), before)
                self.assertEqual(snapshot(other_owner), other_before)
                self.unlocked(owner)
                compiled(native.ones(5))
                self.assertEqual(len(owner.graphs), 2)
                self.assert_retention(owner)

    def test_each_logical_staging_failure_preserves_every_identity(self):
        for phase in ('lowerings', 'shell', 'graphs'):
            for exception in (MemoryError, PublicationInterrupt):
                with self.subTest(phase=phase, exception=exception):
                    other, _, other_owner = self.two_executors()
                    other_before = snapshot(other_owner)
                    compiled, x, owner = self.two_executors()
                    before = snapshot(owner)
                    failure = exception(phase)
                    stage = frontend._stage_recent
                    def staging(mapping, *args):
                        if (mapping is owner.graphs) == (phase == 'graphs'):
                            raise failure
                        return stage(mapping, *args)
                    target, name = frontend, 'Specialization' if phase == 'shell' else '_stage_recent'
                    # A third ABI changes an existing logical entry, not its frozen fields.
                    with patch.object(target, name, side_effect=failure if phase == 'shell' else staging) as fault:
                        with self.assertRaises(exception) as raised:
                            compiled(x, x)
                    self.assertIs(raised.exception, failure)
                    self.assertGreater(fault.call_count, 0)
                    self.assertEqual(snapshot(owner), before)
                    self.unlocked(owner)
                    self.assertEqual(snapshot(other_owner), other_before)
                    compiled(x, x)
                    self.assert_retention(owner)

    def test_newest_equal_key_hits_do_not_copy_or_replace(self):
        compiled, x, owner = self.two_executors()
        compiled(False, x)
        before = snapshot(owner)
        class NoCopy(dict):
            def copy(self):
                raise AssertionError('newest hit copied')
        entry = next(reversed(owner.graphs.values()))
        owner.graphs = NoCopy(owner.graphs)
        key = next(reversed(owner.graphs))
        owner.graphs[key] = frontend.Specialization(entry.payload, NoCopy(entry.lowerings))
        before = snapshot(owner)
        with patch.object(frontend, 'Specialization', side_effect=AssertionError('newest shell')), \
                patch.object(frontend, 'SpecializationPayload', side_effect=AssertionError('newest payload')):
            for _ in range(3):
                compiled(False, x)  # execute constructs fresh, equal ABI and graph keys.
        self.assertEqual(snapshot(owner), before)

    def test_equal_key_replacement_is_staged_without_mutating_old_map(self):
        key, equal = (1, (2,)), tuple([1, tuple([2])])
        self.assertIsNot(key, equal)
        old, replacement = object(), object()
        mapping = {key: old}
        self.assertIs(frontend._stage_recent(mapping, equal, old, 2), mapping)
        staged = frontend._stage_recent(mapping, equal, replacement, 2)
        self.assertIsNot(staged, mapping)
        self.assertIs(mapping[key], old)
        self.assertIs(staged[key], replacement)
        self.assertIs(next(iter(staged)), equal)

    def test_derived_mutation_failures_clear_only_derived_ownership(self):
        phases = ('executor_pop', 'executor_set', 'executor_evict',
                  'prepared_prune', 'prepared_pop', 'prepared_set',
                  'prepared_count', 'prepared_bytes', 'charge_add', 'charge_sub', 'charge_prune')
        for phase in phases:
            for after in (False, True):
                for exception in (MemoryError, PublicationInterrupt):
                    with self.subTest(phase=phase, after=after, exception=exception):
                        other, _, other_owner = self.two_executors()
                        other_before = snapshot(other_owner)
                        failure = exception(phase)
                        with patch.object(frontend, '_prepared_entry_bytes', return_value=64):
                            compiled, x, owner = self.two_executors()
                            args = (False, x)
                            byte_limit = frontend._PREPARED_CACHE_BYTES
                            mapping, operation, predicate = 'executors', 'pop', lambda key: True
                            if phase == 'executor_set':
                                operation = 'set'
                            elif phase == 'executor_evict':
                                operation = 'delete'
                                args = (native.ones(5), x)
                            elif phase in ('prepared_prune', 'charge_prune'):
                                mapping = 'prepared'
                                key = next(iter(owner.executors))
                                from tests.test_compile_pointwise_jit import mock_pointwise_executor
                                owner.executors[key] = mock_pointwise_executor(owner.executors[key].run)
                            elif phase in ('prepared_pop', 'prepared_set', 'charge_add', 'charge_sub'):
                                mapping = 'prepared'
                                operation = 'set' if phase in ('prepared_set', 'charge_add') else 'pop'
                            elif phase in ('prepared_count', 'prepared_bytes'):
                                # Exact-shape data misses can reuse a single executor.
                                limit = 4 if phase == 'prepared_bytes' else 2
                                compiled = frontend.implementation(program('def f(x):\n return -x'), limit)
                                for size in (3, 5, 3):
                                    compiled(native.ones(size))
                                owner = cache(compiled)
                                args = (native.ones(7),)
                                mapping = 'prepared'
                                old_key = next(iter(owner.prepared))
                                predicate = lambda key: key == old_key
                                if phase == 'prepared_bytes':
                                    byte_limit = owner.prepared_bytes
                                    self.assertLess(len(owner.prepared) + 1, limit + 1)
                                    self.assertGreater(owner.prepared_bytes + 64, byte_limit)
                            if phase.startswith('charge_'):
                                original = type(owner).__setattr__
                                fired = []
                                def set_charge(instance, name, value):
                                    if instance is owner and name == 'prepared_bytes' and not fired:
                                        is_add = value > instance.prepared_bytes
                                        if is_add == (phase == 'charge_add'):
                                            if after:
                                                original(instance, name, value)
                                            fired.append(value)
                                            raise failure
                                    original(instance, name, value)
                                injection = patch.object(type(owner), '__setattr__', set_charge)
                            else:
                                broken = FaultDict(getattr(owner, mapping), operation, failure, after, predicate)
                                setattr(owner, mapping, broken)
                                injection = patch.object(frontend, '_PREPARED_CACHE_BYTES', byte_limit)
                            history = logical_snapshot(owner)
                            with injection:
                                with self.assertRaises(exception) as raised:
                                    compiled(*args)
                            self.assertIs(raised.exception, failure)
                            self.assertTrue(fired if phase.startswith('charge_') else broken.fired)
                            self.assertEqual(logical_snapshot(owner), history)
                            self.assertFalse(owner.executors)
                            self.assertFalse(owner.prepared)
                            self.assertEqual(owner.prepared_bytes, 0)
                            self.assertEqual(snapshot(other_owner), other_before)
                            self.unlocked(owner)
                            # Remove the injector, then continue from committed history.
                            owner.executors = dict(owner.executors)
                            owner.prepared = dict(owner.prepared)
                            compiled(*args)
                            self.assert_retention(owner, limit=4 if phase == 'prepared_bytes' else 2)

    def test_repeated_publication_failure_stays_bounded_and_reset_clears_history(self):
        compiled, x, owner = self.two_executors()
        history = logical_snapshot(owner)
        failure = PublicationInterrupt('repeat')
        with patch.object(frontend, '_publish_preparation', side_effect=failure) as fault:
            for _ in range(8):
                with self.assertRaises(PublicationInterrupt) as raised:
                    compiled(False, x)
                self.assertIs(raised.exception, failure)
                self.assertEqual(logical_snapshot(owner), history)
                self.assert_retention(owner)
                self.assertFalse(owner.executors)
                self.assertFalse(owner.prepared)
                self.unlocked(owner)
        self.assertEqual(fault.call_count, 8)
        compiled(False, x)
        owner.executors = FaultDict(owner.executors, '', failure, False)
        owner.prepared = FaultDict(owner.prepared, '', failure, False)
        owner.graphs = FaultDict(owner.graphs, '', failure, False)
        owner.clear()
        self.assertFalse(owner.graphs)
        self.assertFalse(owner.executors)
        self.assertFalse(owner.prepared)
        self.assertEqual(owner.prepared_bytes, 0)
        compiled(False, x)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class PublicationHardware(unittest.TestCase):
    setUpClass = classmethod(prepared_tests.PreparedHardware.setUpClass.__func__)
    setUp = prepared_tests.PreparedHardware.setUp
    tearDown = prepared_tests.PreparedHardware.tearDown
    upload = prepared_tests.PreparedHardware.upload
    without_replay = prepared_tests.PreparedHardware.without_replay

    def bits(self, actual, expected):
        actual = self.torch.tensor(actual.cpu().tolist(), dtype=self.torch.float32)
        self.assertTrue(self.torch.equal(actual.view(self.torch.int32), expected.cpu().view(self.torch.int32)))

    def test_retained_numerical_hint_survives_failure_with_persistent_reference(self):
        source = 'def f(x):\n p=x*x\n return (-p,p.sin())'
        fn = program(source)
        compiled, reference = native.compile(fn), self.torch.compile(program(source, self.torch))
        for step, count in enumerate((2, 13, 2, 2, 13, 2)):
            x = self.upload([2.**-80]*count, (count,))
            tx = self.upload([2.**-80]*count, (count,), self.torch)
            expected = reference(tx)
            if step == 2:
                history = logical_snapshot(cache(compiled))
                failure = MemoryError('after preparation publication')
                publish = frontend._publish_preparation
                def fail(*args):
                    publish(*args)
                    raise failure
                with patch.object(frontend, '_publish_preparation', side_effect=fail) as fault:
                    with self.assertRaises(MemoryError) as raised:
                        self.without_replay(fn, compiled, (x,))
                self.assertIs(raised.exception, failure)
                fault.assert_called_once()
                self.assertEqual(logical_snapshot(cache(compiled)), history)
                self.assertFalse(cache(compiled).prepared)
                self.assertFalse(cache(compiled).executors)
                self.assertEqual(cache(compiled).prepared_bytes, 0)
            actual = self.without_replay(fn, compiled, (x,))
            for a, e in zip(actual, expected):
                self.bits(a, e)

    def test_signed_zero_history_survives_failure_with_same_default_reference(self):
        for initial in (0., -0.):
            for phase, origin in ((phase, origin) for phase in ('graphs', 'derived')
                                  for origin in ('parameter', 'global', 'closure')):
                with self.subTest(initial=repr(initial), phase=phase, origin=origin):
                    # Isolate independent histories, never reset after a failure.
                    native.compiler.reset()
                    self.torch.compiler.reset()
                    from tests.test_compile_pointwise_runtime_scalars import ScalarShapeSpecializationHardware
                    binding = ScalarShapeSpecializationHardware.binding_function
                    fn, setter = binding(self, origin, 1, False)
                    ref_fn, ref_setter = binding(self, origin, 1, False)
                    compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
                    for step, (count, scalar) in enumerate(((2, initial), (13, -initial), (2, -initial), (13, initial), (2, initial))):
                        x = self.upload([1.]*count, (count,))
                        tx = self.upload([1.]*count, (count,), self.torch)
                        if setter is not None:
                            setter(scalar)
                            ref_setter(scalar)
                        args = (scalar, x) if origin == 'parameter' else (x,)
                        refs = (scalar, tx) if origin == 'parameter' else (tx,)
                        expected = reference(*refs)
                        if step == 2:
                            owner = cache(compiled)
                            before = snapshot(owner)
                            failure = PublicationInterrupt('signed zero')
                            stage, publish = frontend._stage_recent, frontend._publish_preparation
                            def fail_stage(mapping, *args):
                                if mapping is owner.graphs:
                                    raise failure
                                return stage(mapping, *args)
                            def fail_publish(*args):
                                publish(*args)
                                raise failure
                            target = '_stage_recent' if phase == 'graphs' else '_publish_preparation'
                            with patch.object(frontend, target, side_effect=fail_stage if phase == 'graphs' else fail_publish) as fault:
                                with self.assertRaises(PublicationInterrupt) as raised:
                                    self.without_replay(fn, compiled, args)
                            self.assertIs(raised.exception, failure)
                            self.assertGreater(fault.call_count, 0)
                            self.assertEqual(logical_snapshot(owner), before[0])
                            if phase == 'graphs':
                                self.assertEqual(snapshot(owner), before)
                            else:
                                self.assertFalse(owner.executors)
                                self.assertFalse(owner.prepared)
                                self.assertEqual(owner.prepared_bytes, 0)
                        self.bits(self.without_replay(fn, compiled, args), expected)

    def test_alias_only_failure_keeps_exactly_once_storage_effects(self):
        from tests.test_compile_pointwise_helpers import no_bodies
        fn = program('def f(x):\n x.add_(1)\n return x.view(-1)')
        compiled = native.compile(fn)
        x = self.upload([0.]*3, (3,))
        with no_bodies(fn):
            compiled(x)
        owner = cache(compiled)
        before = snapshot(owner)
        stage = frontend._stage_recent
        failure = MemoryError('alias graph staging')
        def fail(mapping, *args):
            if mapping is owner.graphs:
                raise failure
            return stage(mapping, *args)
        with patch.object(frontend, '_stage_recent', side_effect=fail) as fault, \
                patch.object(frontend, '_publish_preparation', side_effect=AssertionError('alias numerical publication')):
            with self.assertRaises(MemoryError) as raised:
                with no_bodies(fn):
                    compiled(x)
        self.assertIs(raised.exception, failure)
        self.assertEqual(fault.call_count, 2)
        self.assertEqual(x.cpu().tolist(), [2.]*3)
        self.assertEqual(snapshot(owner), before)
        with no_bodies(fn):
            actual = compiled(x)
        self.assertEqual(x.cpu().tolist(), [3.]*3)
        self.assertEqual(actual.cpu().tolist(), [3.]*3)
        self.assertEqual(snapshot(owner), before)
