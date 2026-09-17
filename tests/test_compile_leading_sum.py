"""Portable leading-sum language and strict native descriptor contracts.

Synthetic metadata here exercises lowering only, never CUDA execution. The
hardware companion proves the public default-compile capability independently.
"""
from dataclasses import replace
import unittest

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests.test_compile_pointwise_helpers import no_bodies
from tests.test_compile_pointwise_jit import program


def semantic(fn, shape=(128, 256), arguments=None):
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
    def test_selected_history_and_full_scalar_abi_finalize_before_identity(self):
        fn = program('def f(x):\n return x.sum(0)/x.shape[0]')
        lowering = semantic(fn)
        source = lowering.graph.input
        properties = (False, 'torch.float32', 'cuda:0')
        metadata = (((193, 252), (252, 1), *properties),)
        parsed = frontend.analyze(fn, 1)
        guards, observations, hint = frontend._shape_guards(
            parsed, lowering.graph, {source: 0}, metadata, {})
        self.assertEqual(hint, 193)
        self.assertEqual(observations[source], metadata[0])
        cold = frontend._finalize_leading_sum(lowering, guards, hint, 0).graph
        self.assertEqual(cold.descriptor,
                         ('leading_sum', 0, False, 193, ('dimension', 0, 193), 0, 252, 193))
        generalized = frontend.ShapeGuards(
            ((source, frontend.TensorGuard((None, 252), (252, 1), properties)),), (), ())
        selected = frontend._finalize_leading_sum(lowering, generalized, 128, 2).graph
        self.assertEqual(selected.descriptor,
                         ('leading_sum', 0, False, None, ('dimension', 0, None), 2, 252, 128))
        # Re-finalization on a selected hit retains its H and all dead/live slots.
        self.assertEqual(frontend._finalize_leading_sum(
            replace(lowering, graph=selected), generalized, 128, 2).graph, selected)
        identities = {selected, replace(selected, scalar_count=1),
                      replace(selected, row_hint=193), replace(selected, column_certificate=256)}
        self.assertEqual(len(identities), 4)
        columns_changed = frontend.ShapeGuards(
            ((source, frontend.TensorGuard((None, None), (None, 1), properties)),), (), ())
        with self.assertRaisesRegex(NotImplementedError, 'exact column guard'):
            frontend._finalize_leading_sum(lowering, columns_changed, 193, 2)

    def test_full_scalar_counts_and_divisor_slots_reach_input_admission(self):
        for count in (0, 1, 64):
            divisors = [('none',), ('constant', 'float', frontend.scalar_bits(2.)),
                        ('dimension', 0, 128), ('dimension', 1, 256)]
            if count:
                divisors.extend((('runtime', 0, False), ('runtime', count-1, True)))
            for divisor in divisors:
                with self.subTest(count=count, divisor=divisor):
                    descriptor = ('leading_sum', 0, False, 128, divisor, count, 256, 128)
                    # Descriptor acceptance is observed without CUDA discovery:
                    # the empty input tuple then fails whole-input admission.
                    with self.assertRaisesRegex(RuntimeError, 'missing tensor input'):
                        bridge._leading_sum_host_plan((), descriptor)

    def test_malformed_descriptors_reject_before_tensor_admission_or_callbacks(self):
        calls = []

        class Poison:
            def forbidden(self, *args):
                calls.append('callback')
                raise AssertionError('descriptor invoked user code')
            __int__ = __index__ = __float__ = __bool__ = __iter__ = __eq__ = forbidden

        class Tuple(tuple):
            pass

        class Integer(int):
            pass

        class String(str):
            pass

        descriptor = ('leading_sum', 0, False, 128, ('none',), 0, 256, 128)
        invalid = [Poison(), list(descriptor), Tuple(descriptor), (),
                   descriptor[:5], descriptor[:6], descriptor[:7], descriptor+(0,)]
        replacements = {
            0: (Poison(), 'pointwise', String('leading_sum')),
            1: (Poison(), True, Integer(0), 1, 2**64),
            2: (Poison(), 0),
            3: (Poison(), True, Integer(128), -1, 2**64, 64, 257, 129),
            5: (Poison(), True, Integer(0), -1, 2**64, 65),
            6: (Poison(), None, True, Integer(256), -1, 2**64, 128, 255, 260),
            7: (Poison(), None, True, Integer(128), -1, 2**64, 64, 257, 129),
        }
        for index, values in replacements.items():
            for value in values:
                invalid.append(descriptor[:index]+(value,)+descriptor[index+1:])
        for divisor in (('none', 1), ('constant', 'complex', 0),
                        ('constant', 'float', Poison()), ('runtime', True, False),
                        ('runtime', 2**64, False), ('runtime', 0, 0),
                        ('runtime', 0, False), ('runtime', Integer(0), False),
                        Tuple(('none',)), ('constant', String('float'), 0),
                        ('dimension', 2, None), ('dimension', 0, True),
                        ('dimension', 0, 2**64), ('dimension', 0, None),
                        ('dimension', 1, None), ('dimension', 1, 252)):
            invalid.append(descriptor[:4]+(divisor,)+descriptor[5:])
        for value in invalid:
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises((TypeError, ValueError, OverflowError, RuntimeError)) as error:
                    bridge._leading_sum_host_plan((), value)
                self.assertNotIn('missing tensor input', str(error.exception))
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
