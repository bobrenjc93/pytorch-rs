import re
import unittest
from unittest.mock import patch

import numpy as np
import torch_rs as torch
from tests import test_unflatten_reference as method_tests

reference_torch = method_tests.reference_torch


class TopLevelUnflattenReferenceTests(method_tests.UnflattenReferenceTests):
    # Exercise the same contiguous/transposed/offset/empty view and weighted
    # leaf-gradient cases through the public function and native view engine.
    @staticmethod
    def unflatten(module, source, dim, sizes):
        return module.unflatten(source, dim, sizes)

    def test_replaced_method_preserves_native_views_and_l1_gradients(self):
        # Compose the two new CPU surfaces with offset/transposed views and
        # unequal upstream weights, all the way back to both original leaves.
        def contract(module):
            left = module.tensor([[0., 1., 2., 3., 4., 5.],
                                  [6., 7., 8., 9., 10., 11.]], requires_grad=True)
            right = module.tensor([[9., 3., 2., 4., 0., 9.],
                                   [0., 8., 1., 9., 15., 2.]], requires_grad=True)
            source, target = left.transpose(0, 1)[1:5], right.transpose(0, 1)[1:5]
            with patch.object(module.Tensor, 'unflatten', side_effect=AssertionError('method lookup')) as replaced:
                result = module.unflatten(source, 0, (2, 2))
                other = module.unflatten(input=target, dim=0, sizes=(2, -1))
                loss = module.nn.functional.l1_loss(result, other, reduction='none')
                weights = module.tensor([[[1., -2.], [3., -4.]], [[5., -6.], [7., -8.]]])
                (loss * weights).sum().backward()
                replaced.assert_not_called()
            metadata = (tuple(result.shape), result.stride(), result.storage_offset(),
                        result.data_ptr() == source.data_ptr(), result.requires_grad, result.is_leaf)
            gradients = left.grad.tolist(), right.grad.tolist()
            del source, target, left, right
            return metadata, result.tolist(), loss.tolist(), gradients

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_replaced_method_preserves_dispatch_errors_and_mode_restoration(self):
        # The public operation still dispatches legitimately; forwarding and
        # exceptions must restore both nested modes without a method lookup.
        def contract(module):
            events = []
            source = module.ones((6,))

            class Mode(module.overrides.TorchFunctionMode):
                def __init__(self, label):
                    self.label = label

                def __torch_function__(self, func, types, args=(), kwargs=None):
                    events.append((self.label, func is module.unflatten))
                    return func(*args, **(kwargs or {}))

            with patch.object(module.Tensor, 'unflatten', side_effect=AssertionError('method lookup')) as replaced:
                with Mode('outer'), Mode('inner'):
                    for dim, sizes in ((99, ()), (0, ()), (0, (2, 4))):
                        with self.assertRaises(Exception) as raised:
                            module.unflatten(source, dim, sizes)
                        events.append((type(raised.exception).__name__, str(raised.exception)))
                        self.assertEqual(len(module.overrides._get_current_function_mode_stack()), 2)
                    result = module.unflatten(source, 0, (2, 3))
                replaced.assert_not_called()
            self.assertEqual(module.overrides._get_current_function_mode_stack(), [])
            return result.tolist(), events

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_positional_keyword_and_integer_forms(self):
        def contract(module):
            source = module.tensor(np.arange(12, dtype=np.float32).reshape(2, 6).tolist())
            results = [
                module.unflatten(source, 1, (2, 3)),
                module.unflatten(source, -1, sizes=[2, -1]),
                module.unflatten(source, sizes=(2, 3), dim=1),
                module.unflatten(sizes=module.Size([2, 3]), input=source, dim=1),
                module.unflatten(source, np.int64(1), (np.int32(2), 3)),
            ]
            for alias in ('input', 'x', 'a', 'x1'):
                results.append(module.unflatten(**{alias: source, 'dim': 1, 'sizes': (2, 3)}))
            return results
        for actual, expected in zip(contract(torch), contract(reference_torch), strict=True):
            self.assert_matches(actual, expected)

    def assert_call_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual:
            actual_call()
        with self.assertRaises(Exception) as expected:
            expected_call()
        self.assertIs(type(actual.exception), type(expected.exception))
        def diagnostic(error):
            return str(error).split('\nException raised from ', 1)[0].rstrip('"\n')
        self.assertEqual(diagnostic(actual.exception), diagnostic(expected.exception))

    def test_sizes_subclasses_use_stored_contents_without_python_hooks(self):
        actual = torch.tensor(np.arange(12, dtype=np.float32).reshape(2, 6).tolist())
        expected = reference_torch.tensor(actual.tolist())
        for base in (list, tuple):
            events = []

            class Sizes(base):
                def __iter__(self):
                    events.append('iter')
                    return iter((3, 2))

                def __len__(self):
                    events.append('len')
                    return 17

                def __getitem__(self, key):
                    events.append('getitem')
                    return 99

            for values in ((2, 3), (2, -1)):
                for keyword in (False, True):
                    with self.subTest(base=base, values=values, keyword=keyword):
                        sizes = Sizes(values)
                        def call(module, source):
                            if keyword:
                                return module.unflatten(input=source, dim=1, sizes=sizes)
                            return module.unflatten(source, 1, sizes)
                        self.assert_matches(call(torch, actual), call(reference_torch, expected))
            for values in ((True, 6), (2, 4), (), (2, 3.0)):
                with self.subTest(base=base, invalid_values=values):
                    sizes = Sizes(values)
                    self.assert_call_error_matches(
                        lambda: torch.unflatten(actual, 1, sizes),
                        lambda: reference_torch.unflatten(expected, 1, sizes),
                    )
            self.assertEqual(events, [])

    def test_native_dim_and_sizes_types_precede_their_own_overrides(self):
        def unexpected_override(*args, **kwargs):
            raise AssertionError('native schema arguments must not dispatch their own handlers')

        for integer_base in (int, np.int64):
            class Dim(integer_base):
                __torch_function__ = classmethod(unexpected_override)

            for sizes_base in (list, tuple):
                class Sizes(sizes_base):
                    __torch_function__ = classmethod(unexpected_override)

                for keyword in (False, True):
                    with self.subTest(integer_base=integer_base, sizes_base=sizes_base,
                                      keyword=keyword):
                        def call(module):
                            source = module.ones((2, 6))
                            if keyword:
                                return module.unflatten(input=source, dim=Dim(1), sizes=Sizes((2, 3)))
                            return module.unflatten(source, Dim(1), Sizes((2, 3)))
                        self.assert_matches(call(torch), call(reference_torch))

    def test_sizes_are_converted_before_dimension_side_effects(self):
        for keyword in (False, True):
            for replacement in ([3, 2], []):
                with self.subTest(keyword=keyword, replacement=replacement):
                    def contract(module):
                        sizes, events = [2, 3], []

                        class Dim(np.int64):
                            def __index__(self):
                                events.append('dim')
                                sizes[:] = replacement
                                return 0

                        source = module.tensor([0., 1., 2., 3., 4., 5.])
                        if keyword:
                            result = module.unflatten(input=source, dim=Dim(0), sizes=sizes)
                        else:
                            result = module.unflatten(source, Dim(0), sizes)
                        self.assertEqual(tuple(result.shape), (2, 3))
                        self.assertEqual(events, ['dim'])
                        self.assertEqual(sizes, replacement)
                        self.assertEqual(result.data_ptr(), source.data_ptr())
                        return result

                    self.assert_matches(contract(torch), contract(reference_torch))

    def test_sizes_conversion_errors_precede_dimension_conversion(self):
        for keyword in (False, True):
            for sizes in ((2, 3.0), (2, 2**100), (2, object())):
                for dim_value in (2**100, -(2**100), 'side_effect'):
                    with self.subTest(keyword=keyword, sizes=sizes, dim=dim_value):
                        events = []

                        class Dim(np.int64):
                            def __index__(self):
                                events.append('dim')
                                return 0

                        def call(module):
                            dim = Dim(0) if dim_value == 'side_effect' else dim_value
                            source = module.ones(6)
                            if keyword:
                                return module.unflatten(input=source, dim=dim, sizes=sizes)
                            return module.unflatten(source, dim, sizes)

                        self.assert_call_error_matches(
                            lambda: call(torch), lambda: call(reference_torch),
                        )
                        self.assertEqual(events, [])

    def test_sizes_subclasses_still_dispatch_stored_element_overrides(self):
        def contract(module, base, position):
            events = []
            marker = object()
            source = module.ones((2, 6))

            class Element:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append((func is module.unflatten, types == (Element,),
                                   args[0] is source, args[2] is sizes))
                    return marker

            class Sizes(base):
                @classmethod
                def __torch_function__(cls, *args, **kwargs):
                    raise AssertionError('container override must be ignored')

                def __iter__(self):
                    raise AssertionError('must read stored elements')

            values = [2, 3]
            values[position] = Element()
            sizes = Sizes(values)
            self.assertIs(module.unflatten(source, 1, sizes), marker)
            return events

        for base in (list, tuple):
            for position in (0, 1):
                with self.subTest(base=base, position=position):
                    self.assertEqual(contract(torch, base, position),
                                     contract(reference_torch, base, position))

    def test_arbitrary_indexable_dimensions_fail_before_operand_dispatch(self):
        events = []

        class Indexable:
            def __index__(self):
                events.append('index')
                return 1

        class Override:
            @classmethod
            def __torch_function__(cls, *args, **kwargs):
                raise AssertionError('invalid dimension must be rejected before dispatch')

        for overridden in (False, True):
            actual = Override() if overridden else torch.ones((2, 6))
            expected = Override() if overridden else reference_torch.ones((2, 6))
            for keyword in (False, True):
                with self.subTest(overridden=overridden, keyword=keyword):
                    def call(module, source):
                        if keyword:
                            return module.unflatten(input=source, dim=Indexable(), sizes=(2, 3))
                        return module.unflatten(source, Indexable(), (2, 3))
                    self.assert_call_error_matches(
                        lambda: call(torch, actual), lambda: call(reference_torch, expected),
                    )
        self.assertEqual(events, [])

    def test_binding_dimension_product_and_ambiguous_inference_errors(self):
        actual, expected = torch.zeros((2, 6)), reference_torch.zeros((2, 6))
        calls = (
            ((), {}), ((1,), {}),
            ((1, (2, 3), 0), {}), ((1, (2, 3)), {'dim': 1}),
            ((1, (2, 3)), {'sizes': (2, 3)}),
            ((1, (2, 3)), {'extra': 0}),
            ((), {'dim': 1}), ((), {'sizes': (2, 3)}),
            ((-3, (2, 3)), {}), ((2, (2, 3)), {}),
            ((2**100, (2, 3)), {}), ((True, (2, 3)), {}),
            ((None, (2, 3)), {}), ((1.0, (2, 3)), {}), (('named', (2, 3)), {}),
            ((1, (2, 4)), {}), ((-1, (4, -1)), {}),
            ((1, (-1, -1)), {}), ((1, (-2, 3)), {}), ((1, (0, -1)), {}),
            ((99, ()), {}), ((1, []), {}), ((1, None), {}),
            ((1, 6), {}), ((1, range(1, 3)), {}),
            ((1, (True, 6)), {}), ((1, (2.0, 3)), {}), ((1, (2**100,)), {}),
            ((1, (2, 3.0)), {}), ((1, (2, object())), {}),
            ((), {'dim': True, 'sizes': (2, 3)}),
            ((), {'dim': 1, 'sizes': (True, 6)}),
            ((), {'dim': 1, 'sizes': None}),
        )
        for args, kwargs in calls:
            with self.subTest(args=args, kwargs=kwargs):
                self.assert_call_error_matches(
                    lambda: torch.unflatten(actual, *args, **kwargs),
                    lambda: reference_torch.unflatten(expected, *args, **kwargs),
                )
        for module_source in (False, True):
            for kwargs in ({}, {'dim': 1, 'sizes': (2, 3)}, {'tensor': 0, 'dim': 1, 'sizes': (2, 3)}):
                with self.subTest(source=module_source, kwargs=kwargs):
                    self.assert_call_error_matches(
                        lambda: torch.unflatten(*((None,) if module_source else ()), **kwargs),
                        lambda: reference_torch.unflatten(*((None,) if module_source else ()), **kwargs),
                    )
        self.assert_call_error_matches(
            lambda: torch.unflatten(actual, 1, (2, 3), input=actual),
            lambda: reference_torch.unflatten(expected, 1, (2, 3), input=expected),
        )
        for shape, dim, sizes in (
            ((), 0, (1,)), ((), -1, (1,)), ((), 1, (1,)),
            ((), 0, ()), ((), 99, ()), ((2, 6), -99, ()),
            ((0, 6), 0, (0, -1)), ((0, 6), 0, (-1, 0)),
            ((0, 6), 1, (4, 2)), ((0, 6), 1, (4, -1)),
        ):
            with self.subTest(shape=shape, dim=dim, sizes=sizes):
                self.assert_call_error_matches(
                    lambda: torch.unflatten(torch.zeros(shape), dim, sizes),
                    lambda: reference_torch.unflatten(reference_torch.zeros(shape), dim, sizes),
                )

    def test_shared_storage_updates_and_lifetime(self):
        def contract(module):
            leaf = module.ones((2, 6), requires_grad=True)
            (leaf * 4).sum().backward()
            source = leaf.grad.transpose(0, 1)[1:5]
            result = module.unflatten(source, 0, (2, 2))
            metadata = result.storage_offset(), result.stride(), result.data_ptr() == source.data_ptr()
            (leaf * 5).sum().backward()
            del source, leaf
            return metadata, result.tolist(), result.sum().item()
        self.assertEqual(contract(torch), contract(reference_torch))

    def override_contract(self, module):
        events = []
        marker = object()
        source = module.ones((2, 6))

        def record(label, func, types, args, kwargs):
            def describe(value):
                if value is source:
                    return 'source'
                if isinstance(value, Override):
                    return type(value).__name__
                if isinstance(value, (tuple, list)):
                    return tuple(describe(item) for item in value)
                return value
            events.append((label, func is module.unflatten,
                           tuple(t.__name__ for t in types), describe(args),
                           None if kwargs is None else {k: describe(v) for k, v in kwargs.items()}))

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                record(cls.__name__, func, types, args, kwargs)
                return marker

        class Subclass(Override):
            pass

        for operand in (Override(), Subclass()):
            self.assertIs(module.unflatten(operand, 1, (2, 3)), marker)
            self.assertIs(module.unflatten(input=operand, dim=1, sizes=(2, 3)), marker)
            self.assertIs(module.unflatten(operand, dim=1, sizes=(2, 3)), marker)
            self.assertIs(module.unflatten(operand, 99, (-1, -1)), marker)
        self.assertIs(module.unflatten(source, Override(), (2, 3)), marker)
        self.assertIs(module.unflatten(source, 1, Override()), marker)
        self.assertIs(module.unflatten(Override(), 1, (Subclass(), 3)), marker)
        self.assertIs(module.unflatten(Override(), 1, (Override(), 3)), marker)
        return events

    def test_public_override_identity_call_layout_and_precedence(self):
        self.assertEqual(self.override_contract(torch), self.override_contract(reference_torch))

    def test_schema_errors_precede_override_dispatch(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise AssertionError('schema error must precede dispatch')
        for args, kwargs in (
            ((Override(),), {}),
            ((Override(), True, (2, 3)), {}),
            ((Override(), 1, None), {}),
            ((Override(), 1, (True, 6)), {}),
            ((Override(), 1, (2, 3)), {'dim': 1}),
            ((Override(), 1, (2, 3)), {'extra': 1}),
        ):
            with self.subTest(args=args, kwargs=kwargs):
                self.assert_call_error_matches(
                    lambda: torch.unflatten(*args, **kwargs),
                    lambda: reference_torch.unflatten(*args, **kwargs),
                )

    def mode_contract(self, module):
        events = []
        source = module.ones((2, 6))
        marker = object()

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self, label, result):
                self.label, self.result = label, result

            def __repr__(self):
                return f'Mode({self.label})'

            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append((self.label, func is module.unflatten,
                               tuple(t.__name__ for t in types), len(args),
                               None if kwargs is None else tuple(kwargs)))
                if self.result == 'forward':
                    return func(*args, **(kwargs or {}))
                return self.result

        with Mode('accept', marker):
            accepted = module.unflatten(input=source, dim=1, sizes=(2, 3)) is marker
        with Mode('lower', 'forward'), Mode('upper', 'forward'):
            forwarded = module.unflatten(source, 1, (2, 3))
        with Mode('decline', NotImplemented), self.assertRaises(TypeError) as raised:
            module.unflatten(source, 1, (2, 3))
        message = str(raised.exception)
        self.assertEqual(module.overrides._get_current_function_mode_stack(), [])
        return accepted, forwarded.tolist(), events, message

    def test_accepting_forwarding_nested_and_declining_modes(self):
        self.assertEqual(self.mode_contract(torch), self.mode_contract(reference_torch))

    def test_mode_decline_falls_back_to_operand(self):
        def contract(module):
            events = []
            class Override:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append(('operand', func is module.unflatten))
                    return NotImplemented
            class Mode(module.overrides.TorchFunctionMode):
                def __repr__(self):
                    return 'Mode()'
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    events.append(('mode', func is module.unflatten))
                    return NotImplemented
            with Mode(), self.assertRaises(TypeError) as raised:
                module.unflatten(Override(), 0, (1,))
            self.assertEqual(module.overrides._get_current_function_mode_stack(), [])
            return events, re.sub(r'0x[0-9a-f]+', '0x<address>', str(raised.exception))
        self.assertEqual(contract(torch), contract(reference_torch))


if __name__ == '__main__':
    unittest.main()
