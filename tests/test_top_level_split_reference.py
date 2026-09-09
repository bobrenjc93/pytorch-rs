import inspect
import re
import unittest

import torch_rs as torch
from tests import test_split_reference as method_tests

reference_torch = method_tests.reference_torch


class TopLevelSplitReferenceTests(method_tests.TensorSplitReferenceTests):
    # Run the method's layout, aliasing, empty, no_grad, and selected/repeated/
    # nested-output backward differentials through the public function as well.
    @staticmethod
    def split(module, source, size, dim=0):
        return module.split(source, split_size_or_sections=size, dim=dim)

    @staticmethod
    def section_call_forms(module, source, sections):
        return (module.split(source, sections), module.split(source, sections, -1),
                module.split(source, split_size_or_sections=sections),
                module.split(dim=-1, split_size_or_sections=sections, tensor=source))

    def error_contract(self, module):
        source = module.zeros((2, 3))
        scalar = module.tensor(1.)
        calls = [lambda: module.split(), lambda: module.split(source),
                 lambda: module.split(split_size_or_sections=2),
                 lambda: module.split(source, 2, 0, 0),
                 lambda: module.split(source, 2, tensor=source),
                 lambda: module.split(source, 2, split_size_or_sections=2),
                 lambda: module.split(source, 2, 0, dim=0),
                 lambda: module.split(source, split_size=2),
                 lambda: module.split(input=source, split_size_or_sections=2),
                 lambda: module.split(source, 2, extra=0),
                 lambda: module.split(source, -1), lambda: module.split(source, 0),
                 lambda: module.split(source, -1, 10), lambda: module.split(source, 0, 10),
                 lambda: module.split(source, 1, -3), lambda: module.split(source, 1, 2),
                 lambda: module.split(source, 2**100), lambda: module.split(source, 1, 2**100),
                 lambda: module.split(scalar, 1), lambda: module.split(scalar, -1)]
        errors = []
        for call in calls:
            with self.assertRaises(Exception) as raised:
                call()
            errors.append((type(raised.exception).__name__, str(raised.exception)))
        return errors

    def test_public_binding_and_signature_match(self):
        actual = inspect.signature(torch.split)
        expected = inspect.signature(reference_torch.split)
        self.assertEqual(str(actual).replace('torch_rs.', 'torch.'), str(expected))
        for module in (torch, reference_torch):
            source = module.tensor([float(i) for i in range(7)])
            for call in (
                lambda: module.split(source, 3),
                lambda: module.split(source, 3, -1),
                lambda: module.split(source, split_size_or_sections=3),
                lambda: module.split(tensor=source, dim=0, split_size_or_sections=3),
            ):
                self.assertEqual([o.tolist() for o in call()],
                                 [[0., 1., 2.], [3., 4., 5.], [6.]])

    def test_storage_aliasing_and_lifetime_match(self):
        def contract(module):
            base = module.tensor([float(i) for i in range(30)])
            source = base.reshape(2, 3, 5)[1].t()
            outputs = module.split(source, 2)
            result = [(o.is_set_to(source.narrow(0, start, o.shape[0])),
                       o.data_ptr() == base.data_ptr() + (15 + start) * base.element_size())
                      for start, o in zip((0, 2, 4), outputs, strict=True)]
            del source, base
            return result, [o.tolist() for o in outputs]
        self.assertEqual(contract(torch), contract(reference_torch))

    def override_contract(self, module):
        events = []
        marker = object()

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append((func is module.split,
                               tuple(t.__name__ for t in types),
                               args[0] is source, args[1] is size, kwargs))
                return marker

        class Subclass(Override):
            pass

        results = []
        for source in (Override(), Subclass()):
            # Dispatch happens before split size and dimension validation.
            for size in (2, [1, 1], object()):
                for call in (
                    lambda: module.split(source, size),
                    lambda: module.split(source, size, -1),
                    lambda: module.split(tensor=source, split_size_or_sections=size, dim=-1),
                ):
                    results.append(call() is marker)
        return results, events

    def test_operand_overrides_match(self):
        self.assertEqual(self.override_contract(torch), self.override_contract(reference_torch))

    def mode_contract(self, module):
        events = []
        source = module.ones((5,))
        marker = object()

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self, label, result):
                self.label, self.result = label, result

            def __repr__(self):
                return f'Mode({self.label})'

            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append((self.label, func is module.split,
                               tuple(t.__name__ for t in types),
                               args[0] is source, args[1], kwargs))
                if self.result == 'forward':
                    return func(*args, **kwargs)
                return self.result

        with Mode('accept', marker):
            accepted = module.split(tensor=source, split_size_or_sections=2) is marker
        with Mode('lower', 'forward'), Mode('upper', 'forward'):
            forwarded = module.split(source, 2, -1)
        with Mode('decline', NotImplemented), self.assertRaises(TypeError) as raised:
            module.split(source, 2)
        declined = str(raised.exception).replace('torch_rs.functional', 'torch.functional')
        self.assertEqual(module.overrides._get_current_function_mode_stack(), [])
        return accepted, [o.tolist() for o in forwarded], declined, events

    def test_accepting_forwarding_nested_and_declining_modes_match(self):
        self.assertEqual(self.mode_contract(torch), self.mode_contract(reference_torch))

    def test_declining_override_and_mode_fallback_match(self):
        def contract(module):
            events = []

            class Override:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append('override')
                    return NotImplemented

            class Mode(module.overrides.TorchFunctionMode):
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    events.append('mode')
                    return NotImplemented

            with Mode(), self.assertRaises(TypeError) as raised:
                module.split(Override(), 2)
            self.assertEqual(module.overrides._get_current_function_mode_stack(), [])
            message = str(raised.exception).replace('torch_rs.functional', 'torch.functional')
            return events, re.sub(r'0x[0-9a-f]+', '0x<address>', message)
        self.assertEqual(contract(torch), contract(reference_torch))


if __name__ == '__main__':
    unittest.main()
