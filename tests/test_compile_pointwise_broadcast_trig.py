"""One-stage original-IR trig admission and ordinary default CUDA execution."""
import math
from contextlib import ExitStack
from types import SimpleNamespace
import struct
import unittest
from unittest.mock import Mock, patch

import torch_rs as native
from torch_rs import torch_rs as bridge
from test_compile_pointwise_jit import Hardware, available, cache, lower, program

# Retain the four former broadcast-priority trig rejections as positives.
FORMER_REJECTIONS = ('x.sin()', 'y.cos()', 'x.sin()*False', 'x.cos()*0')
EXPRESSIONS = FORMER_REJECTIONS + (
    'x.sin().cos()', '(x+y).sin()', '(y-x).cos()', '(x*y).sin().cos()',
    'x.sin()+y.cos()', 'y.sin()-x.cos()', 'x.sin()*y.cos()',
    '(-x).sin()', '-x.cos()', '(x.relu()+y.sin()).cos().relu()',
)
SHAPES = (((2, 1), (1, 3)), ((1, 3), (2, 1)), ((3,), (1, 3)),
          ((), (1,)), ((0, 1), (1, 3)))


class Metadata(unittest.TestCase):
    def test_unary_and_one_stage_trig_before_and_after_arithmetic(self):
        for expression in EXPRESSIONS + ('(x*0.0).cos()', '(x*True).sin()'):
            graph = lower(program('def f(x,y):\n return '+expression), 2)
            for shapes in SHAPES:
                with self.subTest(expression=expression, shapes=shapes):
                    self.assertIn('out0[i]', bridge._pointwise_plan(
                        graph.nodes, graph.outputs, 2, shapes))

    def test_mixed_madd_trig_rejects_in_both_orders_even_after_identities(self):
        for trig in ('p.sin()', 'p.cos()', 'x.sin()*False', 'x.cos()*0'):
            for roots in (f'(p+x,{trig})', f'({trig},p+x)'):
                graph = lower(program('def f(x,y):\n p=x*y\n return '+roots), 2)
                for shapes in SHAPES:
                    with self.subTest(roots=roots, shapes=shapes):
                        with self.assertRaisesRegex(RuntimeError, 'arithmetic stage'):
                            bridge._pointwise_plan(graph.nodes, graph.outputs, 2, shapes)
                # The equal-shape domain still accepts these graphs.
                bridge._pointwise_plan(graph.nodes, graph.outputs, 2, ((3,), (3,)))


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class BroadcastTrig(unittest.TestCase):
    setUpClass = classmethod(Hardware.setUpClass.__func__)
    tearDown = Hardware.tearDown
    upload = Hardware.upload
    compare = Hardware.compare
    without_replay = Hardware.without_replay

    def host(self, value):
        return self.torch.tensor(value.cpu().tolist(), dtype=self.torch.float32).reshape(tuple(value.shape))

    def assert_bits(self, actual, expected):
        self.torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        keep = ~expected.isnan()
        self.assertTrue(self.torch.equal(actual.view(self.torch.int32)[keep],
                                        expected.view(self.torch.int32)[keep]))

    def args(self, shapes, values, framework, offset=False):
        result = []
        for shape in shapes:
            data = [values[i % len(values)] for i in range(math.prod(shape))]
            if offset:
                value = self.upload([99.] + data + [88.], (len(data)+2,), framework)
                value = value[1:len(data)+1].reshape(shape)
            else:
                value = self.upload(data, shape, framework)
            result.append(value)
        return tuple(result)

    def check(self, fn, compiled, reference, args, refs, retained, exact=False):
        inputs = [(x, self.host(x)) for x in args if type(x) is native.Tensor]
        expected = reference(*refs)
        actual = self.without_replay(fn, compiled, args)
        self.compare(actual, expected, exact=exact)
        if exact:
            self.assert_bits(self.host(actual), expected.cpu())
        for value, before in inputs:
            self.assert_bits(self.host(value), before)
            self.assertIsNot(actual, value)
            if actual.numel() and value.numel():
                self.assertNotEqual(actual.data_ptr(), value.data_ptr())
        for value, before in retained:
            self.assert_bits(self.host(value), before)
            self.assertIsNot(actual, value)
            if actual.numel() and value.numel():
                self.assertNotEqual(actual.data_ptr(), value.data_ptr())
        retained.append((actual, self.host(actual)))
        return actual

    def test_former_rejections_keep_the_original_shapes_and_values(self):
        for expression in FORMER_REJECTIONS:
            native.compiler.reset()
            self.torch.compiler.reset()
            source = 'def f(x,y):\n return '+expression
            fn = program(source)
            compiled, reference = native.compile(fn), self.torch.compile(program(source))
            retained = []
            for shapes in (((2,), (1, 2)), ((), (1,)), ((0, 1), (1, 3))):
                args, refs = [], []
                for shape, value in zip(shapes, (1., 2.)):
                    data = [value] * math.prod(shape)
                    args.append(self.upload(data, shape))
                    refs.append(self.upload(data, shape, self.torch))
                for _ in range(2):
                    with self.subTest(expression=expression, shapes=shapes):
                        self.check(fn, compiled, reference, tuple(args), tuple(refs), retained)

    def test_reference_shape_value_offset_and_retained_output_histories(self):
        for expression in EXPRESSIONS:
            # Independent expressions start fresh; shape/value revisits retain
            # both ordinary wrappers and stock compiler limits throughout.
            native.compiler.reset()
            self.torch.compiler.reset()
            source = 'def f(x,y):\n return '+expression
            fn = program(source)
            compiled, reference = native.compile(fn), self.torch.compile(program(source))
            retained = []
            for shapes in SHAPES + SHAPES[:1]:
                for warm in (False, True):
                    values = ([0., -0., 0.375, -1.25, 3.] if not warm else
                              [1.25, -0.375, -0., 0., -3.])
                    args = self.args(shapes, values, native, offset=warm)
                    refs = self.args(shapes, values, self.torch, offset=warm)
                    with self.subTest(expression=expression, shapes=shapes, warm=warm):
                        self.check(fn, compiled, reference, args, refs, retained)

    def test_scalar_kind_zero_trig_witness_and_runtime_promotion(self):
        # Integer/bool zero folds to a constant tensor; float zero retains
        # runtime multiplication. cos().sin() additionally distinguishes the
        # constant-only binary64 sine of one from runtime float32 libdevice.
        values = [float('nan'), float('inf'), -float('inf'), -1., -0., 0., 1.]
        for literal in ('0', 'False', '0.0', '-0.0', 'True'):
            for trig in ('sin', 'cos', 'cos().sin'):
                native.compiler.reset()
                self.torch.compiler.reset()
                source = f'def f(x,unused):\n return (x*{literal}).{trig}()'
                fn = program(source)
                compiled, reference = native.compile(fn), self.torch.compile(program(source))
                retained = []
                shapes = ((7,), (1, 7))
                for data in (values, values[::-1]):
                    self.check(fn, compiled, reference, self.args(shapes, data, native),
                               self.args(shapes, data, self.torch), retained, exact=True)
        for trig in ('sin', 'cos', 'cos().sin'):
            native.compiler.reset()
            self.torch.compiler.reset()
            source = f'def f(s,x,unused):\n return (x*s).{trig}()'
            fn = program(source)
            compiled, reference = native.compile(fn), self.torch.compile(program(source))
            retained = []
            for count, scalar in ((7, 0.0), (9, -0.0), (7, 0.5), (11, 0.0),
                                  (7, -0.0), (7, False), (7, True), (7, 0.0)):
                shapes = ((count,), (1, count))
                self.check(fn, compiled, reference,
                           (scalar, *self.args(shapes, values, native)),
                           (scalar, *self.args(shapes, values, self.torch)), retained, exact=True)
            self.assertTrue(any('float s0' in entry.source
                                for entry in cache(compiled).executors.values()))

    def test_subnormal_amplification_and_nested_runtime_trig_exact(self):
        values = [struct.unpack('=f', struct.pack('=I', b))[0]
                  for b in (1, 0x007fffff, 0x00800000, 0x80000001, 0x807fffff, 0x80800000)]
        values += [0., -0., 0.375, -1.25]
        for expression in ('x.sin()*1e38', 'x.sin().cos()', 'x.cos().sin()',
                           'x.sin().sin()*1e38'):
            native.compiler.reset()
            self.torch.compiler.reset()
            fn = program('def f(x,unused):\n return '+expression)
            reference = self.torch.compile(program('def f(x,unused):\n return '+expression))
            compiled, retained = native.compile(fn), []
            for count in (10, 13, 10):
                shapes = ((count,), (1, count))
                self.check(fn, compiled, reference, self.args(shapes, values, native),
                           self.args(shapes, values, self.torch), retained, exact=True)

    def test_cached_native_execution_and_structured_aliases(self):
        source = "def f(x,y):\n p=(x+y).sin()\n q=p.cos()\n return {'out':(p,p,q),'input':x}"
        fn = program(source)
        compiled, reference = native.compile(fn), self.torch.compile(program(source))
        shapes = ((2, 1), (1, 3))
        retained = []
        executors = []
        compile_native = bridge._pointwise_compile

        def observe_compile(*args):
            executor = SimpleNamespace(prepare=Mock(wraps=compile_native(*args).prepare))
            executors.append(executor)
            return executor

        for warm in range(3):
            args = self.args(shapes, [0.375+warm, -1.25], native, offset=True)
            refs = self.args(shapes, [0.375+warm, -1.25], self.torch, offset=True)
            before = [self.host(x) for x in args]
            expected = reference(*refs)
            # After the cold call, both compilation and preparing instructions
            # are forbidden: the existing immutable native preparation executes.
            with ExitStack() as guards:
                if warm:
                    guards.enter_context(patch.object(bridge, '_pointwise_compile',
                                                       side_effect=AssertionError('recompile')))
                    guards.enter_context(patch.object(executors[0], 'prepare',
                                                       side_effect=AssertionError('prepare')))
                else:
                    guards.enter_context(patch.object(bridge, '_pointwise_compile', observe_compile))
                actual = self.without_replay(fn, compiled, args)
            self.assertIs(actual['out'][0], actual['out'][1])
            self.assertIs(actual['input'], args[0])
            self.compare(actual['out'][0], expected['out'][0])
            self.compare(actual['out'][2], expected['out'][2])
            self.assertNotEqual(actual['out'][0].data_ptr(), actual['out'][2].data_ptr())
            for x, snapshot in zip(args, before):
                self.assert_bits(self.host(x), snapshot)
            for old, snapshot in retained:
                self.assert_bits(self.host(old), snapshot)
                for result in (actual['out'][0], actual['out'][2]):
                    self.assertNotEqual(old.data_ptr(), result.data_ptr())
            retained.extend((result, self.host(result))
                            for result in (actual['out'][0], actual['out'][2]))
            self.assertEqual(len(cache(compiled).prepared), 1)
        self.assertEqual(executors[0].prepare.call_count, 1)


del Hardware

if __name__ == '__main__':
    unittest.main()
