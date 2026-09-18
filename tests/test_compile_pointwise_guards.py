"""Stateless namespace admission and broadcast-fold equivalence.

Cache tests use the existing portable launch fixture, not CUDA emulation or
performance evidence. Native execution remains covered by the GPU suites.
"""
import gc
import itertools
import math
import types
import unittest
import weakref
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests.test_compile_pointwise_helpers import no_bodies
from tests.test_compile_pointwise_jit import cache, program
from tests import test_compile_pointwise_structured_outputs as structured_tests


def hostile_key(target, effects, string_subclass=False):
    class Key:
        def __hash__(self):
            effects.append('hash')
            return hash(target)

        def __eq__(self, other):
            effects.append('equality')
            raise AssertionError('namespace lookup compared a rejected key')

        def __repr__(self):
            effects.append('repr')
            raise AssertionError('namespace admission formatted a rejected key')

    class StringKey(str):
        __hash__ = Key.__hash__
        __eq__ = Key.__eq__
        __repr__ = Key.__repr__

    return StringKey(target) if string_subclass else Key()


class NamespacePredicate(unittest.TestCase):
    def test_sparse_large_unicode_namespace_and_late_rejection(self):
        predicate = bridge._pointwise_namespace_keys_exact
        namespace = {f'key_{i}_\N{SNOWMAN}': object() for i in range(2048)}
        for i in range(0, 2048, 2):
            del namespace[f'key_{i}_\N{SNOWMAN}']
        self.assertIs(predicate(namespace), True)
        effects = []
        key = hostile_key('missing', effects, string_subclass=True)
        namespace[key] = None
        effects.clear()
        self.assertIs(predicate(namespace), False)
        self.assertEqual(effects, [])
        del namespace[key]
        self.assertIs(predicate(namespace), True)
        namespace.clear()
        namespace[0] = None
        namespace.update({str(i): None for i in range(2048)})
        self.assertIs(predicate(namespace), False)

    def test_scan_does_not_retain_values_or_rejection_keys(self):
        class Value:
            pass

        value = Value()
        key = hostile_key('late', [])
        value_ref, key_ref = weakref.ref(value), weakref.ref(key)
        namespace = {'first': value, key: value}
        self.assertIs(bridge._pointwise_namespace_keys_exact(namespace), False)
        namespace.clear()
        del value, key
        gc.collect()
        self.assertIsNone(value_ref())
        self.assertIsNone(key_ref())

    def test_exact_types_and_private_exports(self):
        predicate = bridge._pointwise_namespace_keys_exact
        for value in ({}, {'': object(), 'unicode \N{SNOWMAN}': None}):
            self.assertIs(predicate(value), True)
        for value in (None, [], (), 1, 'dict', {1: None}, {b'name': None}):
            self.assertIs(predicate(value), False)
        name = '_pointwise_namespace_keys_exact'
        self.assertNotIn(name, bridge.__all__)
        self.assertNotIn(name, native.__all__)
        self.assertFalse(hasattr(native, name))
        namespace = {}
        exec('from torch_rs import *', namespace)
        self.assertNotIn(name, namespace)

    def test_rejected_objects_and_keys_have_no_callbacks(self):
        effects = []

        class Mapping(dict):
            def __iter__(self):
                effects.append('iteration')
                raise AssertionError('custom iteration')

            def keys(self):
                effects.append('keys')
                raise AssertionError('custom keys')

            def __len__(self):
                effects.append('length')
                raise AssertionError('custom length')

        for value in (Mapping(), Mapping(valid=None)):
            self.assertIs(bridge._pointwise_namespace_keys_exact(value), False)
        self.assertEqual(effects, [])
        for subclass in (False, True):
            key = hostile_key('captured', effects, subclass)
            value = {'unrelated': None, key: None}
            effects.clear()  # Insertion itself necessarily hashes the key.
            for _ in range(2):
                self.assertIs(bridge._pointwise_namespace_keys_exact(value), False)
            self.assertEqual(effects, [])

    def test_function_builtins_are_not_replaced_by_globals_entry(self):
        actual = {}
        fn = types.FunctionType((lambda x: x).__code__, {'__builtins__': actual})
        # A replacement with invalid keys must not override the frozen table.
        fn.__globals__['__builtins__'] = {1: None}
        self.assertIs(fn.__builtins__, actual)
        frontend.validate_namespaces(fn)
        actual[2] = None
        fn.__globals__['__builtins__'] = {}
        with self.assertRaisesRegex(NotImplementedError, 'builtins keys must be exact strings'):
            frontend.validate_namespaces(fn)
        del actual[2]
        frontend.validate_namespaces(fn)

    def test_globals_errors_precede_builtins_errors(self):
        class Mapping(dict):
            pass

        fn = types.FunctionType((lambda x: x).__code__, Mapping(__builtins__={1: None}))
        with self.assertRaisesRegex(NotImplementedError, 'globals must be an exact dict'):
            frontend.validate_namespaces(fn)
        fn = types.FunctionType((lambda x: x).__code__, {'__builtins__': Mapping(), 1: None})
        with self.assertRaisesRegex(NotImplementedError, 'globals keys must be exact strings'):
            frontend.validate_namespaces(fn)
        del fn.__globals__[1]
        with self.assertRaisesRegex(NotImplementedError, 'builtins must be an exact dict'):
            frontend.validate_namespaces(fn)


class NamespaceCache(unittest.TestCase):
    setUp = structured_tests.StructuredCache.setUp
    snapshot = structured_tests.StructuredCache.snapshot

    def test_all_method_guards_still_reject_unused_shadowing_and_recover(self):
        fn = program('def f(x):\n return -x')
        compiled = native.compile(fn)
        x = native.ones(3)
        compiled(x)
        before = self.snapshot(compiled)
        launches = len(self.launches)
        # Include aliases and inherited metadata access, even though the body
        # only uses negation. Each public shadow must reject before execution.
        names = ('neg', 'negative', '__neg__', 'relu', 'sin', 'cos',
                 'add', '__add__', '__radd__', 'sub', 'subtract', '__sub__',
                 '__rsub__', 'mul', 'multiply', '__mul__', '__rmul__',
                 '__getattribute__', 'shape')
        for name in names:
            with self.subTest(name=name), patch.object(native.Tensor, name, object()):
                with self.assertRaisesRegex(NotImplementedError, 'patched Tensor operation binding'):
                    compiled(x)
                self.assertEqual(self.snapshot(compiled), before)
                self.assertEqual(len(self.launches), launches)
        with no_bodies(fn):
            compiled(x)
        self.assertEqual(len(self.launches), launches + 1)
        self.assertEqual(self.snapshot(compiled), before)

    def test_warm_namespace_rejection_deletion_recovery_and_retention(self):
        for which, target in (('globals', 'helper'), ('builtins', 'range')):
            for subclass in (False, True):
                with self.subTest(namespace=which, string_subclass=subclass):
                    helper = program('def f(a):\n return -a')
                    fn = program('def f(x):\n for i in range(1):\n  x=helper(x)\n return (-x,x)',
                                 helper=helper, __builtins__={'range': range})
                    compiled = native.compile(fn)
                    x = native.ones(3)
                    with no_bodies(fn, helper):
                        compiled(x)
                    before = self.snapshot(compiled)
                    launches = len(self.launches)
                    namespace = getattr(fn, '__' + which + '__')
                    saved = namespace.pop(target)
                    effects = []
                    key = hostile_key(target, effects, subclass)
                    reference = weakref.ref(key)
                    namespace[key] = None
                    effects.clear()
                    executor = next(iter(cache(compiled).executors.values()))
                    with (no_bodies(fn, helper),
                          patch.object(frontend, 'validate_ranges', side_effect=AssertionError('range lookup')),
                          patch.object(frontend, 'freeze_helper', side_effect=AssertionError('helper admission')),
                          patch.object(executor, 'run', side_effect=AssertionError('native launch'))):
                        # Retain the real exception: assertRaises clears its traceback.
                        try:
                            compiled(x)
                        except NotImplementedError as error:
                            caught = error
                        else:
                            self.fail('warm namespace mutation was accepted')
                    self.assertIn(which + ' keys must be exact strings', str(caught))
                    self.assertEqual(effects, [])
                    self.assertEqual(self.snapshot(compiled), before)
                    self.assertEqual(len(self.launches), launches)
                    del namespace[key]
                    del key
                    gc.collect()
                    self.assertIsNotNone(caught.__traceback__)
                    self.assertIsNone(reference())
                    # Removing the hostile key must not cache a success verdict
                    # or silently substitute the now-missing captured binding.
                    with no_bodies(fn, helper), self.assertRaises(NotImplementedError):
                        compiled(x)
                    self.assertEqual(self.snapshot(compiled), before)
                    self.assertEqual(len(self.launches), launches)
                    namespace[target] = saved
                    with (no_bodies(fn, helper),
                          patch.object(frontend, 'lower', side_effect=AssertionError('warm lowering'))):
                        compiled(x)
                    self.assertEqual(self.snapshot(compiled), before)
                    self.assertEqual(len(self.launches), launches + 1)


def original_broadcast_elements(shapes):
    """Frozen pre-repair arithmetic, independent of the production helper."""
    result = ()
    for shape in shapes:
        rank = max(len(result), len(shape))
        left = (1,) * (rank - len(result)) + result
        right = (1,) * (rank - len(shape)) + shape
        if any(a != b and a != 1 and b != 1 for a, b in zip(left, right)):
            return -1
        result = tuple(b if a == 1 else a for a, b in zip(left, right))
    return math.prod(result)


class BroadcastIdentity(unittest.TestCase):
    def test_original_fold_equivalence_for_shape_sequences_and_generators(self):
        shapes = [()] + [(d,) for d in (0, 1, 2, 3)]
        shapes += list(itertools.product((0, 1, 2, 3), repeat=2))
        # All short sequences include incompatible dimensions after zero products;
        # an early zero return would incorrectly accept some of these cases.
        for count in range(4):
            for sequence in itertools.product(shapes, repeat=count):
                expected = original_broadcast_elements(sequence)
                self.assertEqual(frontend._broadcast_elements(sequence), expected, sequence)
                self.assertEqual(frontend._broadcast_elements(s for s in sequence), expected, sequence)

    def test_integer_bounds_rank_zero_and_unequal_ranks(self):
        cases = (
            ((), 1), (((),), 1), (((0,), (1,)), 0),
            (((0, 2), (1, 3)), -1), (((2, 1, 3), (4, 1)), 24),
            (((2147483647,),), 2147483647),
            (((2147483648,),), 2147483648),
            (((65536, 65536),), 4294967296),
            (((2 ** 70, 1), (1, 2 ** 70)), 2 ** 140),
        )
        for shapes, expected in cases:
            with self.subTest(shapes=shapes):
                self.assertEqual(original_broadcast_elements(shapes), expected)
                self.assertEqual(frontend._broadcast_elements(iter(shapes)), expected)
