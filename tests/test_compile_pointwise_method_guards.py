"""Direct-owner method admission; existing launch mocks make no GPU claims."""
from contextlib import ExitStack, contextmanager
import gc
import types
import unittest
from unittest.mock import patch
import weakref

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests import test_compile_pointwise_structured_outputs as structured
from tests.test_compile_pointwise_jit import cache, program


# Independently enumerate the contract: deriving this from _METHODS or the new
# grouped table would let an omitted guard silently remove its own regression.
NAMES = (
    'neg', 'negative', '__neg__', 'relu', 'sin', 'cos',
    'add', '__add__', '__radd__', 'sub', 'subtract', '__sub__', '__rsub__',
    'mul', 'multiply', '__mul__', '__rmul__', '__getattribute__', 'shape',
)
OWNERS = (native.Tensor, native.Tensor.__base__)
ABSENT = object()
ORIGINAL = tuple(tuple(owner.__dict__.get(name, ABSENT) for name in NAMES)
                 for owner in OWNERS)
PREFIX = 'torch.compile(): native CUDA pointwise: '


@contextmanager
def direct_binding(owner, name, value):
    """Restore the exact direct namespace, including inherited-only bindings."""
    previous = owner.__dict__.get(name, ABSENT)
    try:
        if value is ABSENT:
            delattr(owner, name)
        else:
            setattr(owner, name, value)
        yield
    finally:
        if previous is ABSENT:
            if name in owner.__dict__:
                delattr(owner, name)
        else:
            setattr(owner, name, previous)


class Poison:
    def __init__(self, calls):
        self.calls = calls

    def forbidden(self, *args, **kwargs):
        self.calls.append('callback')
        raise AssertionError('method guard executed a user callback')

    __get__ = __call__ = __eq__ = __bool__ = __repr__ = forbidden


class MethodGuardContract(unittest.TestCase):
    def test_original_owners_order_and_all_expected_identities(self):
        self.assertEqual(len(NAMES), 19)
        self.assertIs(frontend._TENSOR_TYPE, OWNERS[0])
        self.assertIs(OWNERS[0].__base__, OWNERS[1])
        self.assertIs(type(frontend._METHOD_GUARDS), tuple)
        self.assertEqual(len(frontend._METHOD_GUARDS), 2)
        for index, (owner, guards) in enumerate(frontend._METHOD_GUARDS):
            self.assertIs(owner, OWNERS[index])
            self.assertIs(type(guards), tuple)
            self.assertEqual(tuple(name for name, _ in guards), NAMES)
            for (name, expected), original in zip(guards, ORIGINAL[index]):
                with self.subTest(owner=index, name=name):
                    self.assertIs(expected, frontend._MISSING if original is ABSENT else original)
                    self.assertIs(owner.__dict__.get(name, ABSENT), original)

    def test_actual_owners_have_ordinary_metaclasses_and_live_namespaces(self):
        # Do not generalize the optimization to custom metaclasses. Both real
        # native owners must support live proxies and every enumerated mutation.
        for index, owner in enumerate(OWNERS):
            self.assertIs(type(owner), type)
            namespace = owner.__dict__
            self.assertIs(type(namespace), types.MappingProxyType)
            for name, original in zip(NAMES, ORIGINAL[index]):
                with self.subTest(owner=index, name=name):
                    token = object()
                    with direct_binding(owner, name, token):
                        self.assertIs(namespace[name], token)
                    self.assertIs(namespace.get(name, ABSENT), original)
                    with direct_binding(owner, name, None):
                        self.assertIn(name, namespace)
                        self.assertIsNone(namespace[name])
                    if original is not ABSENT:
                        with direct_binding(owner, name, ABSENT):
                            self.assertNotIn(name, namespace)
                    self.assertIs(namespace.get(name, ABSENT), original)
        # There are no expected immutable/unexercised pairs. A TypeError above
        # identifies the individual owner/name as a failure, never a broad skip.


class MethodGuardAdmission(unittest.TestCase):
    setUp = structured.StructuredCache.setUp
    snapshot = structured.StructuredCache.snapshot

    def assert_rejected_before_execution(self, compiled, x, name):
        before = self.snapshot(compiled)
        counters = (self.codegen.call_count, len(self.launches), len(self.hints), len(self.orders))
        with ExitStack() as stack:
            for owner, attribute in (
                (bridge, '_compile_trace_tensor_metadata'),
                (bridge, '_pointwise_validate_inputs'),
                (frontend, 'analyze'), (frontend, 'lower'),
            ):
                stack.enter_context(patch.object(owner, attribute,
                    side_effect=AssertionError('guard rejection reached ' + attribute)))
            # Even a warm prepared entry must not be touched or reordered. This
            # covers preparation/run too: both are downstream of this lock.
            lock = stack.enter_context(patch.object(cache(compiled), 'lock'))
            lock.__enter__.side_effect = AssertionError('guard reached cache lock')
            with self.assertRaises(NotImplementedError) as caught:
                compiled(x)
        self.assertEqual(str(caught.exception), PREFIX + 'patched Tensor operation binding: ' + name)
        self.assertEqual(self.snapshot(compiled), before)
        self.assertEqual((self.codegen.call_count, len(self.launches), len(self.hints), len(self.orders)), counters)

    def test_every_owner_name_mutation_cold_warm_created_patched_reset_recovery(self):
        calls = []
        x, other = native.ones(3), native.ones(8)
        fn = program('def f(x):\n if x.shape[0]<4:\n  return -x\n return x.relu()')
        exercised = set()
        for index, owner in enumerate(OWNERS):
            for name, original in zip(NAMES, ORIGINAL[index]):
                actions = [('replace' if original is not ABSENT else 'add', Poison(calls)),
                           ('none', None)]
                if original is not ABSENT:
                    actions.append(('delete', ABSENT))
                for action, value in actions:
                    for state in ('cold', 'warm', 'created-patched'):
                        with self.subTest(owner=index, name=name, action=action, state=state):
                            compiled = None if state == 'created-patched' else native.compile(fn)
                            if state == 'warm':
                                compiled(x)
                                compiled(other)
                                compiled(x)  # Multiple entries and observable LRU order.
                            with direct_binding(owner, name, value):
                                if compiled is None:
                                    compiled = native.compile(fn)
                                self.assert_rejected_before_execution(compiled, x, name)
                                native.compiler.reset()
                                self.assertEqual(self.snapshot(compiled), ([], [], [], 0))
                                self.assert_rejected_before_execution(compiled, x, name)
                            self.assertIs(owner.__dict__.get(name, ABSENT), original)
                            compiled(x)
                            self.assertTrue(cache(compiled).graphs)
                            self.assertTrue(cache(compiled).executors)
                            self.assertTrue(cache(compiled).prepared)
                            exercised.add((index, name))
                            self.assertEqual(calls, [])
        self.assertEqual(exercised, {(index, name) for index in range(2) for name in NAMES})

    def test_class_major_immediate_error_order_even_for_unused_operations(self):
        x = native.ones(3)
        compiled = native.compile(program('def f(x):\n return -x'))
        compiled(x)
        cases = (
            ((0, 'shape'), (0, 'sin'), 'sin'),
            ((0, 'shape'), (1, 'neg'), 'shape'),
            ((0, '__getattribute__'), (1, 'neg'), '__getattribute__'),
            ((1, 'shape'), (1, '__getattribute__'), '__getattribute__'),
        )
        calls = []
        for first, second, expected in cases:
            with self.subTest(first=first, second=second), ExitStack() as stack:
                for index, name in (first, second):
                    stack.enter_context(direct_binding(OWNERS[index], name, Poison(calls)))
                self.assert_rejected_before_execution(compiled, x, expected)
            compiled(x)
        self.assertEqual(calls, [])

    def test_argument_and_function_mode_errors_keep_precedence(self):
        x = native.ones(3)
        compiled = native.compile(program('def f(x):\n return -x'))
        with direct_binding(OWNERS[0], 'neg', None):
            with self.assertRaises(NotImplementedError) as caught:
                compiled(x=x)
            self.assertEqual(str(caught.exception), PREFIX + 'expected positional arguments without keywords')
            with self.assertRaises(NotImplementedError) as caught:
                compiled()
            self.assertEqual(str(caught.exception), PREFIX + 'expected one or two positional tensor occurrences')
            with patch.object(native.overrides, '_get_current_function_mode', return_value=object()):
                with self.assertRaises(NotImplementedError) as caught:
                    compiled(x)
                self.assertEqual(str(caught.exception), PREFIX + 'active __torch_function__ mode')
                with self.assertRaises(NotImplementedError) as caught:
                    compiled()
                self.assertEqual(str(caught.exception), PREFIX + 'expected one or two positional tensor occurrences')
        compiled(x)

    def test_public_tensor_rebinding_does_not_replace_original_guard_owners(self):
        x = native.ones(3)
        compiled = native.compile(program('def f(x):\n return -x'))
        compiled(x)
        with patch.object(native, 'Tensor', object()):
            compiled(x)
            with direct_binding(OWNERS[1], 'shape', None):
                self.assert_rejected_before_execution(compiled, x, 'shape')
            compiled(x)

    def test_retained_rejection_traceback_does_not_retain_removed_poison(self):
        x = native.ones(3)
        compiled = native.compile(program('def f(x):\n return -x'))
        compiled(x)
        calls = []

        def reject(owner, name):
            poison = Poison(calls)
            reference = weakref.ref(poison)
            previous = owner.__dict__.get(name, ABSENT)
            setattr(owner, name, poison)
            retained = None
            try:
                compiled(x)
            except NotImplementedError as error:
                retained = error  # Keep the real traceback and compiled frame.
            finally:
                if previous is ABSENT:
                    delattr(owner, name)
                else:
                    setattr(owner, name, previous)
                del poison
            return retained, reference

        for index, owner in enumerate(OWNERS):
            for name in NAMES:
                with self.subTest(owner=index, name=name):
                    error, reference = reject(owner, name)
                    self.assertIsNotNone(error)
                    self.assertIsNotNone(error.__traceback__)
                    self.assertEqual(str(error), PREFIX + 'patched Tensor operation binding: ' + name)
                    gc.collect()
                    self.assertIsNone(reference())
                    compiled(x)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
