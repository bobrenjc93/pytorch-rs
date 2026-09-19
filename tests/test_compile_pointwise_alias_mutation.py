"""Bounded default alias mutation: admission and real public default comparisons."""
import dataclasses
import dis
from contextlib import nullcontext
import gc
import os
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests.test_compile_pointwise_helpers import no_bodies
from tests.test_compile_pointwise_jit import available, cache, program
from tests.test_compile_pointwise_structured_outputs import StructuredCache
from tests.test_cuda_contiguous import read_bits, write_bits


def lowered(source, **bindings):
    fn = program(source, **bindings)
    parsed = frontend.analyze(fn, 1)
    _, values = frontend.resolve(fn, parsed)
    metadata = (((2, 3), (3, 1), False, 'torch.float32', 'cuda:0', 0),)
    with no_bodies(fn):
        result = frontend.lower(parsed, values, 1, metadata=metadata)
        frontend.preflight_operations(result, values, metadata)
        return result


class AliasMutationAdmission(unittest.TestCase):
    def test_cpu_and_gradient_calls_reject_without_mutation(self):
        for grad in (False, True):
            x = native.tensor([1.0, -2.0], requires_grad=grad)
            compiled = native.compile(program('def f(x):\n return x.add_(1)'))
            with self.assertRaises(NotImplementedError):
                compiled(x)
            self.assertEqual(x.tolist(), [1.0, -2.0])
            self.assertFalse(cache(compiled).graphs)

    def test_alias_only_and_plain_views_keep_numerical_graph_separate(self):
        for body in ('return x.add_(1)',
                     'v=x.view((3,2)); return v.add_(other=0.5,alpha=1)',
                     'v=x.view(3,2).transpose(0,1); v.add_(True); return (v,x)',
                     'x.add_(1); x.add_(-0.5); return x'):
            with self.subTest(body=body):
                self.assertIsNone(lowered('def f(x):\n '+body).graph)
        self.assertIsNotNone(lowered('def f(x):\n return (x.view(3,2),-x)').graph)

    def test_numerical_operations_reject_even_unused_or_inactive(self):
        bodies = (
            'y=-x\n x.add_(1)\n return x',
            'x.add_(1)\n y=x+2\n return x',
            'for i in range(0):\n  y=x.sin()\n x.add_(1)\n return x',
            'if x.shape[0]<4:\n  x.add_(1)\n  return x\n return -x',
            'if x.shape[0]<4:\n  return -x\n x.add_(1)\n return x',
        )
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(NotImplementedError):
                lowered('def f(x):\n '+body)
        helper = program('def f(x):\n unused=x.cos()\n return x')
        with self.assertRaises(NotImplementedError):
            lowered('def f(x):\n helper(x)\n return x.add_(1)', helper=helper)

    def test_hostile_shape_constants_reject_before_callbacks(self):
        calls = []
        class Dimension(int):
            def __repr__(self):
                calls.append('repr')
                return '3'
            def __index__(self):
                calls.append('index')
                return 3
        for shape in ((Dimension(3),2), (2**100,2)):
            fn = program('def f(x):\n x.add_(1)\n return x.view((3,2))')
            fn.__code__ = fn.__code__.replace(co_consts=tuple(
                shape if type(value) is tuple and value == (3,2) else value
                for value in fn.__code__.co_consts))
            with self.assertRaises(NotImplementedError):
                frontend.analyze(fn,1)
        self.assertEqual(calls,[])

    def test_python310_stack_keyword_form_admission_loop_and_helper(self):
        # This is a synthetic opcode fixture, not execution on Python 3.10.
        # Keep real offsets/control flow and replace only keyword-call spelling.
        original = dis.get_instructions
        seen = []
        def stack_keywords(code, *args, **kwargs):
            pending = False
            for instruction in original(code, *args, **kwargs):
                if instruction.opname == 'KW_NAMES':
                    pending = True
                    yield instruction._replace(opname='LOAD_CONST',
                        argval=code.co_consts[instruction.arg])
                elif instruction.opname == 'CALL_KW' or (
                        pending and instruction.opname in ('CALL', 'CALL_FUNCTION', 'CALL_METHOD')):
                    pending = False
                    seen.append('CALL_FUNCTION_KW')
                    yield instruction._replace(opname='CALL_FUNCTION_KW')
                else:
                    if instruction.opname == 'CALL_FUNCTION_KW':
                        seen.append(instruction.opname)
                    yield instruction
        helper = program('def f(x):\n return x.add_(other=0.137,alpha=1)')
        cases = (
            'def f(x):\n return x.add_(other=0.137,alpha=1)',
            'def f(x):\n for i in range(2):\n  x.add_(other=0.137,alpha=1)\n return x',
            'def f(x):\n for i in range(2):\n  helper(x)\n return x',
        )
        for source in cases:
            with self.subTest(source=source), patch.object(frontend.dis, 'get_instructions', stack_keywords):
                seen.clear()
                result = lowered(source, helper=helper)
                self.assertIsNone(result.graph)
                self.assertTrue(seen)
        for expression in ('x.add_(x2=1)', 'x.add_(1,other=2)',
                           'x.add_(1,alpha=True)', 'x.view(shape=(3,2))',
                           'helper(x=x)'):
            with self.subTest(expression=expression), patch.object(frontend.dis, 'get_instructions', stack_keywords):
                seen.clear()
                error = RuntimeError if expression == 'x.add_(1,alpha=True)' else NotImplementedError
                with self.assertRaises(error):
                    lowered('def f(x):\n return '+expression, helper=helper)
                self.assertTrue(seen)

    def test_view_and_mutation_language_remains_bounded(self):
        for expression in ('x.reshape(3,2)', 'x.t()', 'x.contiguous()',
                           'x.view([3,2])', 'x.view(True,6)',
                           'x.add_(x)', 'x.add_(1,2)', 'x.add_(x2=1)',
                           'x.view(3,2)+1'):
            # A numerical consumer of an admitted input alias is still outside
            # the default numerical planner, even without a mutation.
            with self.subTest(expression=expression), self.assertRaises(NotImplementedError):
                lowered('def f(x):\n return '+expression)
        for alpha in ('True','False','0.5','1.0000000000009095'):
            with self.subTest(alpha=alpha), self.assertRaises((NotImplementedError,RuntimeError)):
                lowered('def f(x):\n return x.add_(1,alpha='+alpha+')')


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class AliasMutationHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def pair(self, source, **bindings):
        fn = program(source, **bindings)
        reference_fn = program(source, self.torch, **bindings)
        return fn, native.compile(fn), self.torch.compile(reference_fn)

    def inputs(self, shift=0, offset=0):
        values = np.arange(6 + offset + 2, dtype=np.float32) * 0.25 + shift
        a = native.tensor(values).to('cuda:0')
        b = self.torch.tensor(values, device='cuda:0')
        return a, b, a[offset:offset+6].view(2,3), b[offset:offset+6].view(2,3)

    def compare(self, a, b):
        if isinstance(b, self.torch.Tensor):
            self.assertEqual((tuple(a.shape), a.stride(), a.storage_offset(),
                              str(a.dtype), str(a.device), a.requires_grad),
                             (tuple(b.shape), b.stride(), b.storage_offset(),
                              str(b.dtype), str(b.device), b.requires_grad))
            np.testing.assert_allclose(np.asarray(a.cpu().tolist()),
                                       np.asarray(b.cpu().tolist()),
                                       rtol=1e-5, atol=1e-6, equal_nan=True)
        elif isinstance(b, (tuple, list)):
            self.assertIs(type(a), type(b))
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.compare(left, right)
        else:
            self.assertEqual(a, b)

    def compare_bits(self, a, b):
        actual, expected = read_bits(a), read_bits(b)
        nan = np.isnan(expected.view(np.float32))
        np.testing.assert_array_equal(np.isnan(actual.view(np.float32)), nan)
        np.testing.assert_array_equal(actual[~nan], expected[~nan])

    def test_exact_receivers_aliases_fresh_warm_offsets_and_storage_lifetime(self):
        source = ('def f(x,s):\n v=x.view((3,2)).transpose(0,1)\n'
                  ' r=v.add_(other=s,alpha=1)\n w=x.view(3,2)\n return (r,v,v,w,x)')
        fn, compiled, reference = self.pair(source)
        retained = []
        for step, scalar in enumerate((True, 2.0, -0.5, 1.25)):
            a, b, x, tx = self.inputs(step, step % 2)
            for _ in range(2):
                with no_bodies(fn):
                    out = compiled(x, scalar)
                wanted = reference(tx, scalar)
                self.compare(out, wanted)
                self.compare_bits(a, b)
                self.assertIs(out[0], out[1])
                self.assertIs(out[1], out[2])
                self.assertIsNot(out[1], out[3])
                self.assertIs(out[-1], x)
                self.assertEqual(out[0].data_ptr(), x.data_ptr())
                self.assertFalse(cache(compiled).executors)
                self.assertFalse(cache(compiled).prepared)
            retained.append((out, wanted))
            del a, b, x, tx
            gc.collect()
        for out, wanted in retained:
            self.compare(out, wanted)

    def test_discarded_calls_and_overlapping_original_inputs_execute_in_order(self):
        fn, compiled, reference = self.pair(
            'def f(x,y):\n x.add_(16777216.)\n y.add_(-16777216.)\n x.add_(1.)\n return (x,y)')
        for same in (True, False):
            a, b, x, tx = self.inputs()
            y, ty = (x, tx) if same else (a[1:7].view(2,3), b[1:7].view(2,3))
            with no_bodies(fn):
                out = compiled(x,y)
            expected = reference(tx,ty)
            self.compare(out, expected)
            self.compare_bits(a,b)
            self.assertIs(out[0],x)
            self.assertIs(out[1],y)

    def test_nested_effect_receivers_match_eager_identity_semantics(self):
        # Eager PyTorch is an independent semantic oracle here, not a compiler
        # performance/parity comparison. The developer default-compile capture
        # separately preserves a reference wrapper-identity discrepancy.
        source = ('def f(x,s):\n flat=x.view(-1)\n column=flat.view(-1,1)\n'
                  ' row=column.transpose(0,1)\n result=flat.add_(other=s,alpha=1)\n'
                  ' row.add_(0.137)\n return (result,row,flat,x)')
        fn = program(source)
        compiled = native.compile(fn)
        eager = program(source, self.torch)
        for step, scalar in enumerate((0.217, -0.319, 0.217)):
            a, b, x, tx = self.inputs(step, step % 2)
            with no_bodies(fn):
                out = compiled(x, scalar)
            expected = eager(tx, scalar)
            self.compare(out, expected)
            self.compare_bits(a, b)
            for result, original in ((out, x), (expected, tx)):
                self.assertIs(result[0], result[2])
                self.assertIs(result[3], original)
                self.assertIsNot(result[0], result[1])
                self.assertEqual(result[0].data_ptr(), original.data_ptr())
            x.add_(-0.173)
            tx.add_(-0.173)
            self.compare(out, expected)
            self.compare_bits(a, b)

    def test_scalar_empty_singleton_and_shape_history(self):
        fn, compiled, reference = self.pair(
            'def f(x):\n v=x.view(-1)\n v.add_(0.5)\n return (v,x)')
        for shape in ((), (0,), (1,), (0,3), (3,1), (1,)):
            count = int(np.prod(shape,dtype=np.int64))
            values = np.arange(count+2,dtype=np.float32)
            a = native.tensor(values).to('cuda:0')
            b = self.torch.tensor(values,device='cuda:0')
            x,tx = a[1:count+1].reshape(shape), b[1:count+1].reshape(shape)
            with no_bodies(fn):
                out = compiled(x)
            expected = reference(tx)
            self.compare(out,expected)
            self.compare_bits(a,b)
            self.assertIs(out[1],x)
            self.assertFalse(cache(compiled).prepared)

    def test_literal_loops_shape_branches_and_helpers(self):
        helper = program('def f(x,s):\n return x.add_(s)')
        source = ('def f(x,s):\n v=x.view(3,2)\n for i in range(2):\n'
                  '  helper(v,s)\n if x.shape[0]<4:\n  v.add_(1)\n'
                  ' else:\n  v.add_(2)\n return (v,x)')
        fn, compiled, reference = self.pair(source, helper=helper)
        for scalar in (0.5, -1.25):
            a,b,x,tx = self.inputs()
            with no_bodies(fn, helper):
                out = compiled(x,scalar)
            self.compare(out, reference(tx,scalar))
            self.compare_bits(a,b)

    def test_selected_tree_scalars_are_current_and_preserve_original_alias(self):
        fn, compiled, reference = self.pair(
            'def f(p):\n x=p["x"]\n x.add_(p["values"][0])\n return x')
        for scalar in (True, False, 3.0, -2.0, 0.25):
            a,b,x,tx = self.inputs()
            with no_bodies(fn):
                result = compiled({'x':x,'values':[scalar]})
            expected = reference({'x':tx,'values':[scalar]})
            self.assertIs(result,x)
            self.assertIs(expected,tx)
            self.compare_bits(a,b)

    def test_plain_view_can_mix_with_independent_numerical_output(self):
        fn, compiled, reference = self.pair('def f(x):\n return (x.view(3,2),-x,x)')
        a,b,x,tx = self.inputs(offset=1)
        with no_bodies(fn):
            out = compiled(x)
        expected = reference(tx)
        self.compare(out, expected)
        out[0].add_(2)
        expected[0].add_(2)
        self.compare(out, expected)
        self.assertNotEqual(out[1].data_ptr(),x.data_ptr())

    def test_exceptional_bits_zero_add_and_runtime_scalar_values(self):
        fn, compiled, reference = self.pair('def f(x,s):\n x.add_(s)\n return x')
        bits = np.array([0,0x80000000,1,0x80000001,0x7f800000,0xff800000,
                         0x7fc12345,0x3f800000,0x4b800000],dtype=np.uint32)
        for scalar in (0.0, -0.0, 1.0, -1.0, float('nan'), float('inf')):
            # Keep native history/current-value checks, but use a fresh default
            # reference: PyTorch 2.13's +0/-0 scalar guard can reuse +0 for -0.
            self.torch.compiler.reset()
            reference = self.torch.compile(program(
                'def f(x,s):\n x.add_(s)\n return x', self.torch))
            x = native.zeros(len(bits)).to('cuda:0')
            tx = self.torch.zeros(len(bits),device='cuda:0')
            write_bits(x,bits); write_bits(tx,bits)
            with no_bodies(fn):
                result = compiled(x,scalar)
            expected = reference(tx,scalar)
            self.assertIs(result,x)
            self.assertIs(expected,tx)
            self.compare_bits(x,tx)

    def test_late_inactive_invalid_admission_never_writes_or_publishes(self):
        tails = ('unused=x.view(4,2)',
                 'unused=x.transpose(0,9)',
                 'unused=x.add_(1,alpha=False)',
                 'unused=-x',
                 'for i in range(0):\n  unused=x.view(4,2)',
                 'if x.shape[0]>4:\n  unused=x.transpose(0,9)')
        for tail in tails:
            fn = program('def f(x):\n x.add_(1)\n '+tail+'\n return x')
            compiled = native.compile(fn)
            a,_,x,_ = self.inputs(offset=1)
            before = read_bits(a)
            for _ in range(2):
                with self.subTest(tail=tail), self.assertRaises((NotImplementedError,ValueError,IndexError,RuntimeError)):
                    compiled(x)
                np.testing.assert_array_equal(read_bits(a), before)
                self.assertFalse(cache(compiled).graphs)
                self.assertFalse(cache(compiled).prepared)
                self.assertFalse(cache(compiled).executors)

    def test_invalid_current_inputs_and_scalars_leave_warm_state_usable(self):
        fn, compiled, _ = self.pair('def f(x,s,a):\n return x.add_(s,alpha=a)')
        _,_,x,_ = self.inputs()
        compiled(x,0.5,1.0)
        before = StructuredCache.snapshot(self, compiled)
        bits = read_bits(x)
        for scalar,alpha in ((x,1.0),(1.0,True),(1.0,False),(1.0,1+2**-40),(1.0,0.0)):
            with self.assertRaises((NotImplementedError,TypeError,RuntimeError)):
                compiled(x,scalar,alpha)
            np.testing.assert_array_equal(read_bits(x),bits)
            self.assertEqual(StructuredCache.snapshot(self,compiled),before)
        for invalid in (native.ones(2,3), x.transpose(0,1),
                        self.torch.ones(2,3,dtype=self.torch.float64,device='cuda:0'),
                        native.ones(2,3,requires_grad=True)):
            with self.assertRaises((NotImplementedError,RuntimeError)):
                compiled(invalid,1.0,1.0)
            self.assertEqual(StructuredCache.snapshot(self,compiled),before)
        self.assertIs(compiled(x,1.0,1.0),x)

    def test_reconstruction_failure_cannot_publish_or_reorder_cache(self):
        fn = program('def f(x):\n v=x.view(3,2)\n v.add_(1)\n return (v,x)')
        compiled = native.compile(fn)
        _,_,x,_ = self.inputs()
        compiled(x)
        before = StructuredCache.snapshot(self,compiled)
        with patch.object(frontend.ResultSpec, 'reconstruct',
                          side_effect=MemoryError('reconstruction')):
            for current in (compiled,native.compile(fn)):
                snapshot = StructuredCache.snapshot(self,current)
                with self.assertRaisesRegex(MemoryError,'reconstruction'):
                    current(x)
                self.assertEqual(StructuredCache.snapshot(self,current),snapshot)
        self.assertEqual(StructuredCache.snapshot(self,compiled),before)
        # Failure after a native write is not a rollback guarantee. Recovery
        # uses fresh inputs and compares the actual next public invocation.
        _,_,fresh,reference_input = self.inputs()
        reference = self.torch.compile(program(
            'def f(x):\n v=x.view(3,2)\n v.add_(1)\n return (v,x)',self.torch))
        self.compare(compiled(fresh),reference(reference_input))

    def test_required_operation_bindings_and_hostile_scalars_never_callback(self):
        fn,compiled,_ = self.pair(
            'def f(x,s):\n v=x.view(3,2).transpose(0,1)\n return v.add_(s)')
        _,_,x,_ = self.inputs()
        compiled(x,1.0)
        before = StructuredCache.snapshot(self,compiled)
        bits = read_bits(x)
        calls = []
        def forbidden(*args,**kwargs):
            calls.append('callback')
            raise AssertionError('executed a rejected operation or scalar callback')
        for name in ('view','transpose','add_'):
            with patch.object(native.Tensor,name,forbidden):
                with self.assertRaises(NotImplementedError):
                    compiled(x,1.0)
                with self.assertRaises(NotImplementedError):
                    native.compile(fn)(x,1.0)
                # Operations unused by an old numerical program stay unguarded.
                native.compile(program('def f(x):\n return -x'))(x)
            self.assertEqual(StructuredCache.snapshot(self,compiled),before)
            np.testing.assert_array_equal(read_bits(x),bits)
        class Scalar(float):
            __float__ = __int__ = __index__ = forbidden
        for value in (Scalar(1),object()):
            with self.assertRaises(NotImplementedError):
                compiled(x,value)
            self.assertEqual(StructuredCache.snapshot(self,compiled),before)
            np.testing.assert_array_equal(read_bits(x),bits)
        self.assertEqual(calls,[])

    def test_changed_inactive_helper_is_rechecked_only_when_required(self):
        helper = program('def f(x):\n return x.add_(2)')
        original = helper.__code__
        fn = program('def f(x):\n if x.shape[0]<4:\n  return x.add_(1)\n return helper(x)',
                     helper=helper)
        compiled = native.compile(fn)
        _,_,x,_ = self.inputs()
        compiled(x)
        helper.__code__ = program('def f(x):\n return x.sin()').__code__
        with no_bodies(fn,helper):
            compiled(x)
        large = native.ones(5,3).to('cuda:0')
        bits = read_bits(large)
        before = StructuredCache.snapshot(self,compiled)
        with self.assertRaises(NotImplementedError):
            compiled(large)
        np.testing.assert_array_equal(read_bits(large),bits)
        self.assertEqual(StructuredCache.snapshot(self,compiled),before)
        helper.__code__ = original
        self.assertIs(compiled(large),large)
        np.testing.assert_array_equal(read_bits(large).view(np.float32),np.full(15,3,dtype=np.float32))

    def test_unary_negated_current_scalars_through_helpers_and_global_changes(self):
        helper = program('def f(value):\n return -value')
        source = 'def f(x,s):\n return x.add_(helper(s))'
        fn,compiled,reference = self.pair(source,helper=helper)
        _,_,x,tx = self.inputs()
        for scalar in (0.25,-1.5,True,False,-0.0,0.25):
            with no_bodies(fn,helper):
                actual = compiled(x,scalar)
            expected = reference(tx,scalar)
            self.assertIs(actual,x)
            self.assertIs(expected,tx)
            self.compare_bits(x,tx)
        source = 'def f(x):\n return x.add_(helper(shift))'
        fn = program(source,helper=helper,shift=0.5)
        reference_fn = program(source,self.torch,helper=helper,shift=0.5)
        compiled,reference = native.compile(fn),self.torch.compile(reference_fn)
        for scalar in (0.5,-0.75,True,False,0.5):
            fn.__globals__['shift'] = scalar
            reference_fn.__globals__['shift'] = scalar
            _,_,x,tx = self.inputs()
            with no_bodies(fn,helper):
                actual = compiled(x)
            self.compare(actual,reference(tx))
            self.compare_bits(x,tx)

    def test_retained_unary_alpha_uses_current_scalar_type_before_any_write(self):
        # The scalar is used only by a retained inactive body. Its old type is
        # not an active specialization guard, and warm preflight must reapply
        # Python unary conversion to the current exact scalar.
        for mode in ('parameter', 'parameter_zero_trip', 'tree_zero_trip', 'closure'):
            for initial, sequence in ((True, ((1.5, False), (1.0, True), (True, True), (False, False), (True, True))),
                                      (1.0, ((True, True), (1.5, False), (True, True)))):
                helper = program('def f(x,s):\n return x.add_(0.137,alpha=-(-s))')
                if mode == 'parameter':
                    fn = program('def f(x,s):\n if x.shape[0]<4:\n  return x.add_(1)\n return helper(x,s)', helper=helper)
                    arguments = lambda x, scalar: (x, scalar)
                elif mode == 'parameter_zero_trip':
                    fn = program('def f(x,s):\n for i in range(0):\n  helper(x,s)\n return x.add_(1)', helper=helper)
                    arguments = lambda x, scalar: (x, scalar)
                elif mode == 'tree_zero_trip':
                    fn = program('def f(p):\n x=p["x"]\n for i in range(0):\n  helper(x,p["s"])\n return x.add_(1)', helper=helper)
                    arguments = lambda x, scalar: ({'x': x, 's': scalar},)
                else:
                    # Keep the helper global: a branch target at closure-call
                    # PUSH_NULL is outside the existing normalized boundary.
                    namespace = {'helper': helper}
                    exec('def factory(captured):\n def f(x):\n  if x.shape[0]<4:\n'
                         '   return x.add_(1)\n  return helper(x,captured)\n return f', namespace)
                    fn = namespace['factory'](initial)
                    cell = dict(zip(fn.__code__.co_freevars, fn.__closure__))['captured']
                    def arguments(x, scalar):
                        cell.cell_contents = scalar
                        return (x,)
                compiled = native.compile(fn)
                x = native.ones(2,3).to('cuda:0')
                compiled(*arguments(x, initial))
                if mode in ('parameter', 'closure'):
                    # A different predicate arm leaves the original entry older;
                    # failed small-arm preflight must not reorder either entry.
                    compiled(*arguments(native.ones(5,3).to('cuda:0'), initial))
                for scalar, valid in sequence:
                    with self.subTest(mode=mode, initial=initial, scalar=scalar):
                        before = StructuredCache.snapshot(self, compiled)
                        bits = read_bits(x)
                        # Zero-trip admission retains its existing observed
                        # bindings; source changes may require a new lowering.
                        warm = (nullcontext() if mode.endswith('zero_trip') else
                                patch.object(frontend, 'lower', side_effect=AssertionError(
                                    'retained warm recipe was lowered again')))
                        with no_bodies(fn, helper), warm:
                            if valid:
                                self.assertIs(compiled(*arguments(x, scalar)), x)
                            else:
                                with self.assertRaises((NotImplementedError, RuntimeError)):
                                    compiled(*arguments(x, scalar))
                        if valid:
                            np.testing.assert_array_equal(read_bits(x).view(np.float32),
                                                          bits.view(np.float32) + np.float32(1))
                        else:
                            np.testing.assert_array_equal(read_bits(x), bits)
                            self.assertEqual(StructuredCache.snapshot(self, compiled), before)

    def test_retained_alpha_no_unary_and_odd_unary_do_not_coerce_floats(self):
        for expression, initial, sequence in (
                ('s', 1.0, ((True, False), (1.5, False), (1.0, True))),
                ('-s', -1.0, ((True, False), (-1.5, False), (-1.0, True)))):
            fn = program('def f(x,s):\n if x.shape[0]<4:\n  return x.add_(1)\n return x.add_(0.137,alpha='+expression+')')
            compiled = native.compile(fn)
            x = native.ones(2,3).to('cuda:0')
            compiled(x, initial)
            for scalar, valid in sequence:
                before, bits = StructuredCache.snapshot(self, compiled), read_bits(x)
                with self.subTest(expression=expression, scalar=scalar), no_bodies(fn), patch.object(
                        frontend, 'lower', side_effect=AssertionError('unexpected warm lowering')):
                    if valid:
                        self.assertIs(compiled(x, scalar), x)
                    else:
                        with self.assertRaises((NotImplementedError, RuntimeError)):
                            compiled(x, scalar)
                if not valid:
                    np.testing.assert_array_equal(read_bits(x), bits)
                    self.assertEqual(StructuredCache.snapshot(self, compiled), before)

    def test_inactive_effect_sequence_paths_rebind_after_structure_changes(self):
        helper = program('def f(x,s):\n return x.add_(0.137,alpha=s[-1])')
        for mode in ('direct', 'helper', 'zero_trip'):
            source = ('def f(x,s):\n if x.shape[0]<4:\n  return x.add_(1)\n'
                      ' return x.add_(0.137,alpha=s[-1])')
            if mode == 'helper':
                source = source.replace('x.add_(0.137,alpha=s[-1])', 'helper(x,s["items"])')
            elif mode == 'zero_trip':
                source = ('def f(x,s):\n for i in range(0):\n  helper(x,s)\n return x.add_(1)')
            fn = program(source, helper=helper)
            wrap = (lambda s: {'items': s}) if mode == 'helper' else (lambda s: s)
            compiled = native.compile(fn)
            x = native.ones(2,3).to('cuda:0')
            with self.subTest(mode=mode), no_bodies(fn, helper):
                self.assertIs(compiled(x,wrap([1.0])),x)
                # Keep an older matching small-arm specialization: rejected
                # current admission must not publish or change its recency.
                compiled(native.ones(5,3).to('cuda:0'),wrap([1.0]))
                for invalid in ([1.0,1.5], [1.0,True], [1.0,1.0,1.5]):
                    before, bits = StructuredCache.snapshot(self,compiled), read_bits(x)
                    with self.assertRaises((NotImplementedError,RuntimeError)):
                        compiled(x,wrap(invalid))
                    np.testing.assert_array_equal(read_bits(x),bits)
                    self.assertEqual(StructuredCache.snapshot(self,compiled),before)
                    cold = native.compile(fn)
                    with self.assertRaises((NotImplementedError,RuntimeError)):
                        cold(x,wrap(invalid))
                    np.testing.assert_array_equal(read_bits(x),bits)
                    self.assertFalse(cache(cold).graphs)
                # Growth, shrinkage and list/tuple changes all select the
                # current last item, including new Tensor owners and offsets.
                for items in ([1.5,1.0], [1.0], (1.5,1.0), (1.0,)):
                    owner = native.ones(9).to('cuda:0')
                    current = owner[1:7].view(2,3)
                    self.assertIs(compiled(current,wrap(items)),current)
                    np.testing.assert_array_equal(read_bits(current).view(np.float32),
                                                  np.full(6,2,dtype=np.float32))
                    self.assertFalse(cache(compiled).executors)
                    self.assertFalse(cache(compiled).prepared)

    def test_warm_inactive_nested_effect_projects_current_scalars_before_writes(self):
        helper = program('def f(x,p):\n return x.add_(0.137,alpha=p["items"][-1])')
        fn = program('def f(x,p):\n if x.shape[0]<4:\n'
                     '  return x.add_(1)\n return helper(x,p)', helper=helper)
        compiled = native.compile(fn)
        x = native.ones(2,3).to('cuda:0')
        compiled(x, {'items': [1.0]})
        entry = next(iter(cache(compiled).graphs.values()))
        retained = next(iter(entry.lowerings.values()))
        for scalar, valid in ((1.0, True), (1.5, False), (True, False), (1.0, True)):
            # Fresh storage and reordered/unobserved dict keys still project
            # the retained inactive path; preflight sees its current scalar.
            current = native.ones(2,3).to('cuda:0')
            before = StructuredCache.snapshot(self, compiled)
            bits = read_bits(current)
            with self.subTest(scalar=scalar), no_bodies(fn, helper), \
                    patch.object(frontend.BindingSource, 'child', side_effect=AssertionError('new source')), \
                    patch.object(frontend._BindingResolution, 'complete', side_effect=AssertionError('full walk')), \
                    patch.object(frontend, 'lower', side_effect=AssertionError('warm lowering')):
                if valid:
                    self.assertIs(compiled(current, {'unused': False, 'items': [scalar]}), current)
                else:
                    with self.assertRaises((NotImplementedError, RuntimeError)):
                        compiled(current, {'unused': False, 'items': [scalar]})
            expected = bits.view(np.float32) + np.float32(1) if valid else bits.view(np.float32)
            np.testing.assert_array_equal(read_bits(current).view(np.float32), expected)
            self.assertEqual(StructuredCache.snapshot(self, compiled), before)
            self.assertIs(next(iter(entry.lowerings.values())), retained)
        bits = read_bits(current)
        complete = frontend._BindingResolution.complete
        with no_bodies(fn, helper), patch.object(frontend._BindingResolution, 'complete',
                autospec=True, side_effect=complete) as expansion:
            self.assertIs(compiled(current, {'items': [False, 1.0]}), current)
        expansion.assert_called_once()
        np.testing.assert_array_equal(read_bits(current).view(np.float32),
                                      bits.view(np.float32) + np.float32(1))
        from tests.test_compile_pointwise_helpers import assert_replaced_shell
        current = next(iter(cache(compiled).graphs.values()))
        assert_replaced_shell(self, entry, current)
        self.assertEqual(tuple(current.lowerings), tuple(entry.lowerings))
        self.assertIs(next(iter(entry.lowerings.values())), retained)
        self.assertIsNot(next(iter(current.lowerings.values())), retained)

    def test_inactive_unpack_keeps_whole_program_length_admission(self):
        fn = program('def f(x,s):\n if x.shape[0]<4:\n  return x.add_(1)\n'
                     ' a,b=s\n return x.add_(a,alpha=b)')
        compiled = native.compile(fn)
        x = native.ones(2,3).to('cuda:0')
        with no_bodies(fn):
            compiled(x,[0.137,1.0])
            before,bits = StructuredCache.snapshot(self,compiled),read_bits(x)
            with self.assertRaises(NotImplementedError):
                compiled(x,[0.137,1.0,1.0])
            np.testing.assert_array_equal(read_bits(x),bits)
            self.assertEqual(StructuredCache.snapshot(self,compiled),before)
            self.assertIs(compiled(x,[0.137,1.0]),x)

    def test_later_abi_retains_its_own_inactive_sequence_admission(self):
        helper = program('def f(a,s):\n return a.add_(1)')
        fn = program('def f(x,s,ignored):\n if x.shape[0]<4:\n'
                     '  return x.add_(1)\n return helper(x,s)', helper=helper)
        compiled = native.compile(fn)
        x = native.ones(2,3).to('cuda:0')
        y = native.ones(2,3).to('cuda:0')
        with no_bodies(fn,helper):
            self.assertIs(compiled(x,[1.0],False),x)
        entry = next(iter(cache(compiled).graphs.values()))
        old_abi, old_lowering = next(iter(entry.lowerings.items()))
        helper.__code__ = program('def f(a,s):\n return a.add_(0.137,alpha=s[-1])').__code__
        with no_bodies(fn,helper):
            self.assertIs(compiled(x,[1.0],y),x)
        self.assertEqual(len(cache(compiled).graphs),1)
        from tests.test_compile_pointwise_helpers import assert_replaced_shell
        current = next(iter(cache(compiled).graphs.values()))
        assert_replaced_shell(self, entry, current)
        self.assertEqual(tuple(entry.lowerings), (old_abi,))
        self.assertIs(current.lowerings[old_abi], old_lowering)
        entry = current
        new_abi = next(key for key in entry.lowerings if key != old_abi)
        for invalid in ([1.0,1.5], [1.0,True], (1.0,1.5)):
            before,bits = StructuredCache.snapshot(self,compiled),read_bits(x)
            with no_bodies(fn,helper), self.assertRaises((NotImplementedError,RuntimeError)):
                compiled(x,invalid,y)
            np.testing.assert_array_equal(read_bits(x),bits)
            self.assertEqual(StructuredCache.snapshot(self,compiled),before)
        for items in ([1.5,1.0], [1.0], (1.5,1.0), (1.0,)):
            with no_bodies(fn,helper):
                self.assertIs(compiled(x,items,y),x)
            entry = next(iter(cache(compiled).graphs.values()))
            self.assertIs(entry.lowerings[old_abi],old_lowering)
            # Later-ABI evidence must not be unioned into the original logical
            # entry: its already-admitted helper never selected anything in s.
            with no_bodies(fn,helper), patch.object(frontend,'lower',side_effect=AssertionError('old ABI reparsed')):
                self.assertIs(compiled(x,[1.5],False),x)
            entry = next(iter(cache(compiled).graphs.values()))
            self.assertIs(entry.lowerings[old_abi],old_lowering)
        current = entry.lowerings[new_abi]
        helper.__code__ = program('def f(a,s):\n return a.sin()').__code__
        with no_bodies(fn,helper), patch.object(frontend,'lower',side_effect=AssertionError('valid retained ABI reparsed')):
            self.assertIs(compiled(x,(1.0,),y),x)
        entry = next(iter(cache(compiled).graphs.values()))
        self.assertIs(entry.lowerings[new_abi],current)
        self.assertFalse(cache(compiled).executors)
        self.assertFalse(cache(compiled).prepared)

    def test_later_abi_selected_dict_paths_and_unpack_preflight(self):
        for selection in ('s["selected"]["items"][-1]', 's["selected"]["alpha"]', 'unpack'):
            helper = program('def f(a,s):\n return a.add_(1)')
            fn = program('def f(x,s,ignored):\n if x.shape[0]<4:\n'
                         '  return x.add_(1)\n return helper(x,s)', helper=helper)
            compiled = native.compile(fn)
            x,y = native.ones(2,3).to('cuda:0'),native.ones(2,3).to('cuda:0')
            compiled(x,False,False)
            if selection == 'unpack':
                helper.__code__ = program('def f(a,s):\n other,alpha=s\n return a.add_(other,alpha=alpha)').__code__
                valid = [0.137,1.0]
                invalid = ([0.137], [0.137,1.0,1.0])
            else:
                helper.__code__ = program('def f(a,s):\n return a.add_(0.137,alpha='+selection+')').__code__
                if selection.endswith('["alpha"]'):
                    valid = {'selected': {'alpha': 1.0}}
                    # Both traversed dict signatures still match; only the
                    # selected scalar leaf disappears from the current tree.
                    invalid = ({'selected': {'other': 1.0}}, {'selected': {}})
                else:
                    valid = {'selected': {'items': [1.0]}}
                    invalid = ({'other': {'items': [1.0]}}, {'selected': {'other': [1.0]}})
            with no_bodies(fn,helper):
                compiled(x,valid,y)
            for items in invalid:
                before,bits = StructuredCache.snapshot(self,compiled),read_bits(x)
                with self.subTest(selection=selection,items=items), no_bodies(fn,helper), self.assertRaises(NotImplementedError):
                    compiled(x,items,y)
                np.testing.assert_array_equal(read_bits(x),bits)
                self.assertEqual(StructuredCache.snapshot(self,compiled),before)
                with no_bodies(fn,helper):
                    self.assertIs(compiled(x,valid,y),x)
            if selection != 'unpack':
                helper.__code__ = program('def f(a,s):\n return a.sin()').__code__
                # Keyed lookup observes presence, not all keys or insertion order.
                for items in ({'extra': False,'selected': {'extra': True,**valid['selected']}},
                              {'selected': {**valid['selected'],'extra': False},'extra': True}):
                    with no_bodies(fn,helper), patch.object(frontend,'lower',side_effect=AssertionError('unrelated key guard')):
                        self.assertIs(compiled(x,items,y),x)

    def test_retained_structure_replacement_preserves_frozen_scalar_bits(self):
        helper = program('def f(s):\n return s[-1].view(3,2)')
        fn = program('def f(x,s,gain):\n if x.shape[0]<4:\n'
                     '  return x*gain\n return helper(s)',helper=helper)
        nan_values = np.array([0x7ff8000000001234,0xfff8000000004321],dtype=np.uint64).view(np.float64)
        for first,current in ((-0.0,0.0), (float(nan_values[0]),float(nan_values[1]))):
            compiled = native.compile(fn)
            x = native.tensor([1.,-1.,2.,-2.,3.,-3.]).to('cuda:0').view(2,3)
            with no_bodies(fn,helper):
                expected = read_bits(compiled(x,[x],first))
            entry = next(iter(cache(compiled).graphs.values()))
            for items in ([False,x], (x,), [x]):
                with no_bodies(fn,helper):
                    actual = compiled(x,items,current)
                np.testing.assert_array_equal(read_bits(actual),expected)
                self.assertEqual(len(cache(compiled).graphs),1)
                from tests.test_compile_pointwise_helpers import assert_replaced_shell
                committed = next(iter(cache(compiled).graphs.values()))
                assert_replaced_shell(self, entry, committed)
                self.assertEqual(tuple(committed.lowerings), tuple(entry.lowerings))
                entry = committed
            before,bits = StructuredCache.snapshot(self,compiled),read_bits(x)
            with no_bodies(fn,helper), self.assertRaises(NotImplementedError):
                compiled(x,[x,False],current)
            np.testing.assert_array_equal(read_bits(x),bits)
            self.assertEqual(StructuredCache.snapshot(self,compiled),before)

    def test_scalar_promotion_lowering_retains_inactive_structure(self):
        helper = program('def f(s):\n return s[-1].view(3,2)')
        fn = program('def f(x,s,gain):\n if x.shape[0]<4:\n'
                     '  return x*gain\n return helper(s)',helper=helper)
        compiled = native.compile(fn)
        x = native.ones(2,3).to('cuda:0')
        with no_bodies(fn,helper):
            compiled(x,[x],1.0)
            np.testing.assert_array_equal(read_bits(compiled(x,[x],2.0)).view(np.float32),np.full(6,2,dtype=np.float32))
        entry = next(reversed(cache(compiled).graphs.values()))
        self.assertTrue(any(type(value) is frontend.RuntimeScalar for value in entry.payload.values.values()))
        before,bits = StructuredCache.snapshot(self,compiled),read_bits(x)
        with no_bodies(fn,helper), self.assertRaises(NotImplementedError):
            compiled(x,[x,False],3.0)
        np.testing.assert_array_equal(read_bits(x),bits)
        self.assertEqual(StructuredCache.snapshot(self,compiled),before)
        with no_bodies(fn,helper):
            actual = compiled(x,(False,x),3.0)
        np.testing.assert_array_equal(read_bits(actual).view(np.float32),np.full(6,3,dtype=np.float32))
        from tests.test_compile_pointwise_helpers import assert_replaced_shell
        committed = next(reversed(cache(compiled).graphs.values()))
        assert_replaced_shell(self, entry, committed)
        self.assertEqual(tuple(committed.lowerings), tuple(entry.lowerings))

    def test_retained_structure_does_not_own_caller_containers_or_tensors(self):
        helper = program('def f(a,s):\n return a.add_(0.137,alpha=s["items"][-1])')
        fn = program('def f(x,s,ignored):\n if x.shape[0]<4:\n'
                     '  return x.add_(1)\n return helper(x,s)',helper=helper)
        compiled = native.compile(fn)
        inputs = []
        for length in (1,2):
            x,y = native.ones(2,3).to('cuda:0'),native.ones(2,3).to('cuda:0')
            tree = {'items': [1.0]*length}
            with no_bodies(fn,helper):
                compiled(x,tree,False)
                compiled(x,tree,y)
            inputs.extend((x,y,tree,tree['items']))
        forbidden = {id(value) for value in inputs}
        pending,seen = [cache(compiled).graphs],set()
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            self.assertNotIn(id(value),forbidden)
            self.assertIsNot(type(value),frontend.InputTree)
            self.assertIsNot(type(value),native.Tensor)
            if type(value) is dict:
                pending.extend(value.keys())
                pending.extend(value.values())
            elif type(value) in (frontend.Specialization, frontend.SpecializationPayload):
                pending.extend(getattr(value, name) for name in value._fields)
            elif type(value) in (list,tuple):
                pending.extend(value)
            elif dataclasses.is_dataclass(value) or type(value) is types.SimpleNamespace:
                pending.extend(vars(value).values())
            elif type(value) is types.FunctionType and value.__closure__:
                pending.extend(cell.cell_contents for cell in value.__closure__)
        self.assertTrue(cache(compiled).graphs)
        for entry in cache(compiled).graphs.values():
            self.assertIn(id(entry), seen)
            for record in (entry, entry.payload):
                for name in record._fields:
                    self.assertIn(id(getattr(record, name)), seen, name)

    def test_literal_effect_results_inactive_calls_and_reset(self):
        for result in ('None','7','(7,9)'):
            source = ('def f(x):\n for i in range(0):\n  x.add_(100)\n'
                      ' if x.shape[0]<4:\n  x.add_(1)\n else:\n  x.add_(2)\n'
                      ' return '+result)
            fn,compiled,reference = self.pair(source)
            for rows in (2,5,2):
                x = native.ones(rows,3).to('cuda:0')
                tx = self.torch.ones(rows,3,device='cuda:0')
                with no_bodies(fn):
                    actual = compiled(x)
                self.assertEqual(actual,reference(tx))
                self.compare_bits(x,tx)
                np.testing.assert_array_equal(read_bits(x).view(np.float32),
                    np.full(rows*3,2 if rows<4 else 3,dtype=np.float32))
                self.assertFalse(cache(compiled).executors)
                self.assertFalse(cache(compiled).prepared)
            native.compiler.reset()
            self.assertEqual(StructuredCache.snapshot(self,compiled),([],[],[],0))
            with no_bodies(fn):
                actual = compiled(x)
            self.assertEqual(actual,reference(tx))
            self.compare_bits(x,tx)

    def test_unused_higher_rank_input_keeps_original_admission(self):
        fn, compiled, reference = self.pair('def f(x,y):\n return x.add_(0.137)')
        x = native.ones(3).to('cuda:0')
        tx = self.torch.ones(3,device='cuda:0')
        for shape in ((2,3,4), (1,2,1,3)):
            y = native.ones(*shape).to('cuda:0')
            ty = self.torch.ones(shape,device='cuda:0')
            before = read_bits(y)
            with no_bodies(fn):
                out = compiled(x,y)
            self.assertIs(out,x)
            self.compare(out,reference(tx,ty))
            np.testing.assert_array_equal(read_bits(y),before)

    def test_boolean_negation_preserves_integer_alpha_conversion(self):
        fn, compiled, reference = self.pair(
            'def f(x,enabled):\n return x.add_(0.137,alpha=-(-enabled))')
        x = native.ones(3).to('cuda:0')
        tx = self.torch.ones(3,device='cuda:0')
        for _ in range(2):
            with no_bodies(fn):
                out = compiled(x,True)
            self.assertIs(out,x)
            self.compare(out,reference(tx,True))
        bits = read_bits(x)
        with self.assertRaises(NotImplementedError):
            compiled(x,False)
        np.testing.assert_array_equal(read_bits(x),bits)

    def test_constant_tuple_identity_survives_calls_reset_and_code_change(self):
        helper = program('def f(unused):\n return (7,9)')
        fn = program('def f(x):\n x.add_(0.137)\n a=(2,3)\n b=(2,3)\n return (a,b,helper(x),[a,x])', helper=helper)
        compiled = native.compile(fn)
        constant = next(value for value in fn.__code__.co_consts if type(value) is tuple)
        helper_constant = next(value for value in helper.__code__.co_consts if type(value) is tuple)
        retained = []
        for reset in (False, False, True):
            if reset:
                native.compiler.reset()
            x = native.ones(2,3).to('cuda:0')
            for _ in range(2):
                with no_bodies(fn, helper):
                    out = compiled(x)
                self.assertIs(out[0], constant)
                self.assertIs(out[1], constant)
                self.assertIs(out[2], helper_constant)
                self.assertIs(out[3][0], constant)
                self.assertIs(out[3][1], x)
                for previous in retained:
                    self.assertIsNot(out, previous)
                    self.assertIsNot(out[3], previous[3])
                    self.assertIs(out[0], previous[0])
                retained.append(out)
        replacement = tuple([11,13])
        fn.__code__ = fn.__code__.replace(co_consts=tuple(
            replacement if value is constant else value for value in fn.__code__.co_consts))
        with no_bodies(fn, helper):
            changed = compiled(native.ones(2,3).to('cuda:0'))
        self.assertIs(changed[0], replacement)
        self.assertIs(changed[1], replacement)
        self.assertIs(changed[2], helper_constant)
        self.assertIs(retained[0][0], constant)

    def test_alias_owner_lifetime_and_recompile_budget_before_writes(self):
        fn, compiled, reference = self.pair(
            'def f(x):\n v=x.view(-1)\n v.add_(0.137)\n return v')
        x = native.ones(2,3).to('cuda:0')
        tx = self.torch.ones(2,3,device='cuda:0')
        with no_bodies(fn):
            out = compiled(x)
        expected = reference(tx)
        pointer = x.data_ptr()
        del x, tx
        gc.collect()
        self.assertEqual(out.data_ptr(),pointer)
        out.add_(0.217)
        expected.add_(0.217)
        self.compare(out,expected)

        fn = program('def f(x):\n return x.add_(shift)',shift=0)
        compiled = native.compile(fn)
        x = native.ones(2,3).to('cuda:0')
        limit = native._COMPILE_DEFAULT_RECOMPILE_LIMIT
        for scalar in range(limit):
            fn.__globals__['shift'] = scalar
            self.assertIs(compiled(x),x)
        before = StructuredCache.snapshot(self,compiled)
        bits = read_bits(x)
        fn.__globals__['shift'] = limit
        with self.assertRaisesRegex(NotImplementedError,'recompile_limit'):
            compiled(x)
        np.testing.assert_array_equal(read_bits(x),bits)
        self.assertEqual(StructuredCache.snapshot(self,compiled),before)
        native.compiler.reset()
        self.assertIs(compiled(x),x)
        self.assertFalse(cache(compiled).executors)
        self.assertFalse(cache(compiled).prepared)

    def test_native_only_effects_without_reference_import_or_body_replay(self):
        script = '''
import sys
class BlockTorch:
 def find_spec(self, name, *args):
  if name == 'torch' or name.startswith('torch.'):
   raise AssertionError('compiled alias mutation imported PyTorch')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as native
def helper(value):
 return -value
def f(x, scalar):
 v=x.view(3,2).transpose(0,1)
 same=v.add_(other=helper(scalar),alpha=1)
 x.add_(0.5)
 return (same,v,x)
compiled=native.compile(f)
x=native.ones(2,3).to('cuda:0')
pointer=x.data_ptr()
codes=(f.__code__,helper.__code__)
def forbid(frame,event,arg):
 if event == 'call' and frame.f_code in codes:
  raise AssertionError('original Python body executed')
sys.setprofile(forbid)
try:
 first=compiled(x,0.25)
 second=compiled(x,0.5)
finally:
 sys.setprofile(None)
assert first[0] is first[1] and second[0] is second[1]
assert first[0] is not second[0]
assert first[2] is x and second[2] is x
assert first[0].data_ptr()==pointer==second[0].data_ptr()
assert x.cpu().tolist()==[[1.25,1.25,1.25],[1.25,1.25,1.25]]
assert not compiled._torch_rs_pointwise_cache.executors
assert not compiled._torch_rs_pointwise_cache.prepared
assert 'torch' not in sys.modules
native.compiler.reset()
assert first[0].cpu().tolist()==[[1.25,1.25,1.25],[1.25,1.25,1.25]]
print('native-only compiled alias mutation passed')
'''
        result = subprocess.run([sys.executable,'-B','-c',script],text=True,capture_output=True,
            env={**os.environ,'TORCH_RS_NVRTC':native.__file__+'.missing-nvrtc'})
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('native-only compiled alias mutation passed',result.stdout)


if __name__ == '__main__':
    unittest.main()
