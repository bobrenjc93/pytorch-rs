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
from tests.test_compile_pointwise_jit import available, cache, program


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


class NativeMethodIdentityBoundary(unittest.TestCase):
    def check(self, table, missing=ABSENT):
        return bridge._pointwise_method_identity_mismatch(table, missing)

    def assert_invalid(self, table):
        with self.assertRaises(TypeError) as caught:
            self.check(table)
        self.assertEqual(str(caught.exception),
                         'expected native owner and exact method identity tuples')

    def test_malformed_tuples_and_names_do_not_invoke_protocols(self):
        calls = []
        def forbidden(*args, **kwargs):
            calls.append('callback')
            raise AssertionError('native boundary invoked a protocol')

        class Tuple(tuple):
            __iter__ = __len__ = __getitem__ = __bool__ = __eq__ = __repr__ = forbidden

        class Name(str):
            __hash__ = __eq__ = __bool__ = __repr__ = __str__ = forbidden

        class Proxy:
            __iter__ = __len__ = __getitem__ = __bool__ = __eq__ = __repr__ = forbidden

        owner, expected = OWNERS[0], object()
        binding = ('neg', expected)
        row = (owner, (binding,))
        malformed = (
            None, [], Proxy(), Tuple((row,)),
            (None,), ([],), (Proxy(),), (Tuple(row),),
            ((),), ((owner,),), ((owner, (), None),),
            ((owner, []),), ((owner, Proxy()),), ((owner, Tuple((binding,))),),
            ((owner, (None,)),), ((owner, ([],)),), ((owner, (Proxy(),)),),
            ((owner, (Tuple(binding),)),), ((owner, ((),)),),
            ((owner, (('neg',),)),), ((owner, (('neg', expected, None),)),),
            ((owner, ((None, expected),)),), ((owner, ((b'neg', expected),)),),
            ((owner, ((Name('neg'), expected),)),),
            ((owner, ((Proxy(), expected),)),),
        )
        for index, table in enumerate(malformed):
            with self.subTest(index=index):
                self.assert_invalid(table)
                self.assertEqual(calls, [])
        # Validate later rows even if the first valid row would mismatch.
        for table in malformed[4:]:
            self.assert_invalid((row,) + table)
        self.assertEqual(calls, [])

    def test_foreign_owners_are_rejected_before_namespace_access(self):
        calls = []
        def forbidden(*args, **kwargs):
            calls.append('namespace/protocol')
            raise AssertionError('foreign owner was inspected')

        class Heap:
            pass

        class TensorBaseChild(OWNERS[1]):
            pass

        class Meta(type):
            __getattribute__ = __eq__ = __bool__ = __repr__ = forbidden

        class Custom(metaclass=Meta):
            pass

        class Owner:
            __getattribute__ = __eq__ = __bool__ = __repr__ = forbidden

        class Namespace:
            get = __getitem__ = __iter__ = __bool__ = __repr__ = forbidden

        # Static builtins must not reach GenericGetDict: their storage differs
        # from these original PyO3 heap owners. Empty bindings still validate.
        for index, owner in enumerate((Heap, TensorBaseChild, Custom, Owner(), Namespace(),
                                       object, type, int, str, tuple, dict, None)):
            for bindings in ((), (('neg', object()),)):
                with self.subTest(index=index, empty=not bindings):
                    row = (owner, bindings)
                    self.assert_invalid((row,))
                    self.assert_invalid(((OWNERS[0], (('neg', object()),)), row))
                    self.assertEqual(calls, [])

    def test_missing_sentinel_and_present_none_use_literal_identity(self):
        calls = []
        name = '_native_method_guard_identity_test'
        sentinel, other = object(), object()
        poison = Poison(calls)
        for index, owner in enumerate(OWNERS):
            self.assertNotIn(name, owner.__dict__)
            for actual in (ABSENT, sentinel, None, poison):
                with ExitStack() as stack:
                    if actual is not ABSENT:
                        stack.enter_context(direct_binding(owner, name, actual))
                    for missing in (sentinel, other, None, poison):
                        for expected in (sentinel, other, None, poison):
                            with self.subTest(owner=index):
                                result = self.check(((owner, ((name, expected),)),), missing)
                                if owner.__dict__.get(name, missing) is expected:
                                    self.assertIsNone(result)
                                else:
                                    self.assertIs(result, name)
                                self.assertEqual(calls, [])

    def test_supplied_table_and_unicode_names_define_order(self):
        first, second, third = '_guard_\u03bb', '_guard_\ud800', ''
        expected, replacement = object(), object()
        with ExitStack() as stack:
            for owner, name in ((OWNERS[1], first), (OWNERS[1], second), (OWNERS[0], third)):
                stack.enter_context(direct_binding(owner, name, expected))
            table = ((OWNERS[1], ((second, expected), (first, expected))),
                     (OWNERS[0], ((third, expected),)))
            self.assertIsNone(self.check(()))
            self.assertIsNone(self.check(((OWNERS[0], ()),)))
            self.assertIsNone(self.check(table))
            with direct_binding(OWNERS[0], third, replacement):
                self.assertIs(self.check(table), third)
                with direct_binding(OWNERS[1], first, replacement):
                    self.assertIs(self.check(table), first)
                    with direct_binding(OWNERS[1], second, replacement):
                        self.assertIs(self.check(table), second)
            self.assertIsNone(self.check(table))

    def test_original_owners_survive_public_rebinding_without_exports(self):
        name = '_pointwise_method_identity_mismatch'
        self.assertNotIn(name, bridge.__all__)
        self.assertNotIn(name, native.__all__)
        self.assertNotIn(name, native.__dict__)
        with patch.object(native, 'Tensor', object()):
            self.assertIsNone(self.check(frontend._METHOD_GUARDS, frontend._MISSING))
            with direct_binding(OWNERS[1], 'shape', None):
                self.assertEqual(self.check(frontend._METHOD_GUARDS, frontend._MISSING), 'shape')


class MethodGuardAdmission(unittest.TestCase):
    setUp = structured.StructuredCache.setUp
    snapshot = structured.StructuredCache.snapshot

    def assert_rejected_before_execution(self, compiled, x, name):
        before = self.snapshot(compiled)
        counters = (self.codegen.call_count, self.host_plan.call_count,
                    len(self.launches), len(self.hints), len(self.orders))
        with ExitStack() as stack:
            for owner, attribute in (
                (bridge, '_compile_trace_tensor_metadata'),
                (bridge, '_pointwise_validate_inputs'),
                (frontend, 'analyze'), (frontend, 'lower'),
                (frontend, '_receipt'),
            ):
                stack.enter_context(patch.object(owner, attribute,
                    side_effect=AssertionError('guard rejection reached ' + attribute)))
            # Even a warm prepared entry must not be touched or reordered. This
            # covers preparation/run too: both are downstream of this lock.
            lock = stack.enter_context(patch.object(cache(compiled), 'lock'))
            lock.__enter__.side_effect = AssertionError('guard reached cache lock')
            for invoke in (compiled, compiled._torch_rs_pointwise_receipt):
                with self.assertRaises(NotImplementedError) as caught:
                    invoke(x)
                self.assertEqual(str(caught.exception),
                                 PREFIX + 'patched Tensor operation binding: ' + name)
        self.assertEqual(self.snapshot(compiled), before)
        self.assertEqual((self.codegen.call_count, self.host_plan.call_count,
                          len(self.launches), len(self.hints), len(self.orders)), counters)

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

    def test_real_checks_repeat_on_ordinary_and_receipt_calls_across_abis(self):
        x = native.ones(3)
        compiled = native.compile(program('def f(unused,x):\n return -x'))
        calls = []
        for invoke in (compiled, compiled._torch_rs_pointwise_receipt):
            for args in ((False, x), (x, x), (False, x), (x, x)):
                invoke(*args)
                before = self.snapshot(compiled)
                launches = len(self.launches)
                self.admit.reset_mock()
                with direct_binding(OWNERS[1], 'cos', Poison(calls)):
                    with self.assertRaises(NotImplementedError) as caught:
                        invoke(*args)
                    self.assertEqual(str(caught.exception), PREFIX + 'patched Tensor operation binding: cos')
                    self.admit.assert_not_called()
                    self.assertEqual(self.snapshot(compiled), before)
                    self.assertEqual(len(self.launches), launches)
                invoke(*args)
                self.assertEqual(len(self.launches), launches + 1)
        self.assertEqual(len(cache(compiled).graphs), 1)
        entry = next(iter(cache(compiled).graphs.values()))
        self.assertEqual(len(entry.lowerings), 2)
        self.assertEqual(calls, [])


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class NativeMethodGuardAdmission(unittest.TestCase):
    """Real cold/warm execution and retained owners, separate from timing."""
    snapshot = structured.StructuredCache.snapshot

    def tearDown(self):
        native.compiler.reset()

    def test_cold_and_warm_rejection_preserve_real_owners_receipts_and_outputs(self):
        absent = next((index, name) for index in range(2)
                      for name, original in zip(NAMES, ORIGINAL[index])
                      if original is ABSENT)
        present = next((index, name) for index in range(2)
                       for name, original in zip(NAMES, ORIGINAL[index])
                       if original is not ABSENT)
        calls = []
        cases = ((0, 'neg', Poison(calls)), (0, 'shape', Poison(calls)),
                 (1, '__getattribute__', Poison(calls)),
                 (*absent, Poison(calls)), (*present, ABSENT))
        for state in ('cold', 'warm'):
            for index, name, value in cases:
                with self.subTest(state=state, owner=index, name=name):
                    native.compiler.reset()
                    compiled = native.compile(program('def f(x):\n return -x'))
                    x = native.tensor([1., 2., 3.]).to('cuda:0')
                    retained = []
                    if state == 'warm':
                        for size in (3, 5, 3):
                            current = native.ones(size).to('cuda:0')
                            out, receipt = compiled._torch_rs_pointwise_receipt(current)
                            retained.append((out, receipt, [-1.] * size))
                        self.assertGreaterEqual(len(cache(compiled).prepared), 2)
                    before = self.snapshot(compiled)
                    with direct_binding(OWNERS[index], name, value), ExitStack() as stack:
                        for owner, attribute in (
                            (bridge, '_compile_trace_tensor_metadata'),
                            (bridge, '_pointwise_validate_inputs'),
                            (bridge, '_pointwise_host_plan'),
                            (frontend, '_receipt'),
                        ):
                            stack.enter_context(patch.object(owner, attribute,
                                side_effect=AssertionError('guard reached ' + attribute)))
                        lock = stack.enter_context(patch.object(cache(compiled), 'lock'))
                        lock.__enter__.side_effect = AssertionError('guard reached cache lock')
                        for invoke in (compiled, compiled._torch_rs_pointwise_receipt):
                            with self.assertRaises(NotImplementedError) as caught:
                                invoke(x)
                            self.assertEqual(str(caught.exception),
                                             PREFIX + 'patched Tensor operation binding: ' + name)
                        self.assertEqual(self.snapshot(compiled), before)
                    self.assertIs(OWNERS[index].__dict__.get(name, ABSENT),
                                  ORIGINAL[index][NAMES.index(name)])
                    result, receipt = compiled._torch_rs_pointwise_receipt(x)
                    self.assertEqual(result.cpu().tolist(), [-1., -2., -3.])
                    owner = next(reversed(cache(compiled).executors.values()))
                    self.assertTrue(receipt.belongs_to(owner))
                    for out, old_receipt, expected in retained:
                        self.assertEqual(out.cpu().tolist(), expected)
                        self.assertTrue(old_receipt.belongs_to(owner))
                        self.assertIsNot(result, out)
                    self.assertEqual(cache(compiled).prepared_bytes,
                                     sum(item[1] for item in cache(compiled).prepared.values()))
                    self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
