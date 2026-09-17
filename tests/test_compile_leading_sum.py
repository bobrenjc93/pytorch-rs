"""Portable leading-sum language and strict native descriptor contracts.

Synthetic metadata here exercises lowering only, never CUDA execution. The
hardware companion proves the public default-compile capability independently.
"""
import unittest

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests.test_compile_pointwise_helpers import no_bodies
from tests.test_compile_pointwise_jit import program


def semantic(fn, shape=(3, 5), arguments=None):
    arguments = (frontend.Value(0),) if arguments is None else arguments
    parsed = frontend.analyze(fn, len(arguments))
    _, values = frontend.resolve(fn, parsed, arguments)
    metadata = ((shape, (max(shape[-1], 1), 1), False, 'torch.float32', 'cuda:0'),)
    with no_bodies(fn):
        return frontend.lower(parsed, values, 1, metadata=metadata)


class LeadingSumLanguage(unittest.TestCase):
    def test_axes_keepdim_and_keyword_order_share_typed_semantics(self):
        for call, axis, keepdim in (
                ('x.sum(0)', 0, False), ('x.sum(-2)', -2, False),
                ('x.sum(dim=0)', 0, False), ('x.sum(0, True)', 0, True),
                ('x.sum(-2, keepdim=False)', -2, False),
                ('x.sum(dim=-2, keepdim=True)', -2, True),
                ('x.sum(keepdim=True, dim=0)', 0, True)):
            with self.subTest(call=call):
                result = semantic(program('def f(x):\n return ' + call))
                self.assertIs(type(result.graph), frontend.LeadingSum)
                self.assertFalse(hasattr(result.graph, 'nodes'))
                self.assertEqual(result.graph.axis % 2, axis % 2)
                self.assertEqual(result.graph.keepdim, keepdim)
                self.assertEqual(result.graph.divisor, ('none',))

    def test_single_result_aliases_containers_helpers_and_input_dimensions(self):
        helper = program('def f(a):\n return a.sum(0)')
        for body in ('y=x.sum(0)\nreturn {"sum":[y,y],"x":x,"shape":x.shape[1]}',
                     'y=helper(x)\nreturn y',
                     'y=x.sum(0)/x.shape[-1]\nreturn (y,y)',
                     'y=x.sum(0)/x.shape[-2]\nreturn y'):
            with self.subTest(body=body), no_bodies(helper):
                result = semantic(program('def f(x):\n '+body.replace('\n', '\n '), helper=helper))
                self.assertIs(type(result.graph), frontend.LeadingSum)

    def test_divisor_exact_scalar_kinds_and_current_argument_leaves(self):
        for value in (False, True, 0, -2, 0., -0., 2.**-130, float('inf'), float('nan')):
            with self.subTest(kind=type(value).__name__, value=value):
                result = semantic(program('def f(x):\n return x.sum(0)/divisor', divisor=value))
                self.assertEqual(result.graph.divisor[0], 'constant')
                self.assertEqual(result.graph.divisor[1], type(value).__name__)
        for value in (False, True, 2.5):
            with self.subTest(argument=value):
                result = semantic(program('def f(x,d):\n return x.sum(0)/d'),
                                  arguments=(frontend.Value(0), value))
                self.assertIs(type(result.graph), frontend.LeadingSum)
        with self.assertRaises(NotImplementedError):
            frontend.bind_arguments((native.ones((2, 3)), 2))

    def test_unsupported_reductions_options_and_result_combinations(self):
        bodies = (
            'return x.sum()', 'return x.sum(1)', 'return x.sum(-1)',
            'return x.sum(True)', 'return x.sum(0, 1)', 'return x.sum(dim=(0,))',
            'return x.sum(0, dim=0)', 'return x.sum(dim=0, dtype=fw.float32)',
            'return x.sum(dim=0, out=x)', 'return x.sum(axis=0)',
            'return x.sum(**{"dim":0})', 'return fw.sum(x, dim=0)',
            'return x.mean(0)', 'return x.sum(0).div(2)',
            'return (-x).sum(0)', 'return (x+x).sum(0)',
            'return x.sum(0)+1', 'return x.sum(0)/2/3',
            'return 2/x.sum(0)', 'return x.sum(0)/x',
            'return x.sum(0)/(x.shape[0]+1)',
            'return (x.sum(0), x.sum(0))',
            'y=x.sum(0)\nreturn (y,y/2)',
            'y=x.sum(0)\nreturn (y/2,y/2)',
        )
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(NotImplementedError):
                semantic(program('def f(x):\n '+body.replace('\n', '\n ')))

    def test_control_flow_rejects_reductions_in_active_inactive_and_helper_paths(self):
        helper = program('def f(a):\n return a.sum(0)')
        bodies = (
            'if x.shape[0]>2:\n return x.sum(0)\nreturn -x',
            'if x.shape[0]>2:\n return -x\nreturn x.sum(0)',
            'if x.shape[0]>2:\n ignored=1\nreturn x.sum(0)',
            'for i in range(0):\n ignored=x.sum(0)\nreturn -x',
            'for i in range(1):\n ignored=1\nreturn x.sum(0)',
            'if x.shape[0]>2:\n return -x\nreturn helper(x)',
        )
        for body in bodies:
            with self.subTest(body=body), no_bodies(helper), self.assertRaises(NotImplementedError):
                semantic(program('def f(x):\n '+body.replace('\n', '\n '), helper=helper))


class LeadingSumDescriptor(unittest.TestCase):
    def test_malformed_descriptors_reject_before_tensor_admission_or_callbacks(self):
        calls = []

        class Poison:
            def forbidden(self, *args):
                calls.append('callback')
                raise AssertionError('descriptor invoked user code')
            __int__ = __index__ = __float__ = __bool__ = __iter__ = __eq__ = forbidden

        class Tuple(tuple):
            pass

        descriptor = ('leading_sum', 0, False, 3, ('none',))
        invalid = [Poison(), list(descriptor), Tuple(descriptor), (),
                   ('pointwise', 0, False, 3, ('none',)),
                   ('leading_sum', True, False, 3, ('none',)),
                   ('leading_sum', 1, False, 3, ('none',)),
                   ('leading_sum', 0, 0, 3, ('none',)),
                   ('leading_sum', 0, False, True, ('none',)),
                   ('leading_sum', 0, False, -1, ('none',)),
                   ('leading_sum', 0, False, 2**64, ('none',))]
        for divisor in (('none', 1), ('constant', 'complex', 0),
                        ('constant', 'float', Poison()), ('runtime', True, False),
                        ('runtime', 2**64, False), ('runtime', 0, 0),
                        ('dimension', 2, None), ('dimension', 0, True),
                        ('dimension', 0, 2**64)):
            invalid.append(('leading_sum', 0, False, 3, divisor))
        for value in invalid:
            with self.subTest(kind=type(value).__name__), self.assertRaises((TypeError, ValueError, OverflowError)):
                bridge._leading_sum_host_plan((), value)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
