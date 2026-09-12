"""Alias-only CUDA view capture: public differentials and fail-closed planning."""
import ctypes
from dataclasses import replace
import dis
import gc
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch_rs as native
from torch_rs import _compile_bytecode, _compile_trace as trace
from tests.test_compile_cuda_boundary import compile_with_cache
from tests.test_compile_cuda_mul_scalar import call_without_python, make_program, POLICIES
from tests.test_cuda_add import available, torch
from tests.test_cuda_contiguous import metadata, read_bits, write_bits


def graphlet(shape):
    return make_program(f'''def program(x):
    shape = {shape!r}
    a = x.view(shape)
    b = x.view(size={shape!r})
    shared = [a, a, b, a.contiguous()]
    return x, shared, shared
''')


class ViewMetadataTests(unittest.TestCase):
    def test_shared_native_planner_without_hardware(self):
        for shape, strides, requested, expected, offset in (
            ((), (), (1, 1), (1, 1), 7),
            ((6,), (3,), (2, 3), (9, 3), 7),
            ((0, 7), (1, 9), (0, 7), (1, 9), 7),
            ((0, 7), (1, 9), (7, 0), (1, 1), 7),
        ):
            recorder = trace.CompileTraceRecorder()
            x = recorder.input(shape=shape, stride=strides, device='cuda:0', storage_offset=7)
            out = x.view(requested)
            self.assertEqual(out.metadata.stride, expected)
            self.assertEqual(out.metadata.storage_offset, offset)
            self.assertEqual(recorder._operations[0].shape, requested)
            self.assertIsNone(recorder._operations[0].axes)
        for options in ({'device': 'cpu'}, {'requires_grad': True},
                        {'shape': (1, 2, 3)}, {'dtype': trace.CompileTraceDType('torch.float64')}):
            recorder = trace.CompileTraceRecorder()
            with self.assertRaises(NotImplementedError):
                recorder.input(**({'shape': (3, 7), 'device': 'cuda:0'} | options)).view(-1)
            self.assertEqual(recorder._operations, [])

    def test_exact_dimensions_and_no_index_conversion(self):
        class Index:
            def __index__(self):
                raise AssertionError('conversion')
        class Integer(int):
            pass
        recorder = trace.CompileTraceRecorder()
        x = recorder.input(shape=(1,), device='cuda:0')
        for shape in ((Index(),), (Integer(1),), (np.int64(1),), (1, True)):
            with self.assertRaises(NotImplementedError):
                x.view(shape)
        for shape in ((True,), (1.,), (1, 1.), (2**63,), (-2**63-1,)):
            with self.assertRaises(TypeError):
                x.view(shape)
        self.assertEqual(recorder._operations, [])

    def test_alias_only_planner_and_binding_order_without_hardware(self):
        recorder = trace.CompileTraceRecorder()
        x = recorder.input(shape=(3, 2), stride=(1, 3), device='cuda:0', storage_offset=5)
        with self.assertRaisesRegex(RuntimeError, 'view size is not compatible'):
            x.view(-1)
        self.assertEqual(recorder._operations, [])
        self.assertEqual(x.reshape(-1).metadata.storage_offset, 0)
        for args, kwargs in (((2**63,), {'foo': 1}), (((True,),), {'foo': 1}),
                             ((), {'size': (2**63,), 'foo': 1}), ((1,), {'size': (1,)})):
            with self.assertRaisesRegex(TypeError, 'invalid combination'):
                trace._bind_view_shape(args, kwargs)
        for dtype in (native.float32,):
            for args, kwargs in (((dtype,), {}), ((), {'dtype': dtype})):
                with self.assertRaises(trace.CompileTraceUnsupportedError):
                    trace._bind_view_shape(args, kwargs)
        class Tuple(tuple):
            def __iter__(self):
                raise AssertionError('tuple conversion')
        for dims in (Tuple((1,)), [1]):
            with self.assertRaises(trace.CompileTraceUnsupportedError):
                trace._bind_view_shape((dims,), {})
        for shape, stride, requested, expected in (
            ((1, 3), (37, 2), (3, 1), (2, 2)),
            ((3, 1), (2, 37), (3,), (2,)),
        ):
            x = trace.CompileTraceRecorder().input(shape=shape, stride=stride, device='cuda:0', storage_offset=9)
            self.assertEqual(x.view(requested).metadata.stride, expected)

    def test_known_shape_errors_before_capture_admission_without_hardware(self):
        class Index:
            def __index__(self):
                raise AssertionError('conversion')
            def __repr__(self):
                raise AssertionError('repr')
        x = trace.CompileTraceRecorder().input(shape=(1,), device='cuda:0')
        for shape, message in (((-2, True), 'invalid shape dimension -2'),
                               ((-2, Index()), 'invalid shape dimension -2'),
                               ((-2, trace._RESHAPE_UNKNOWN_DIMENSION), 'invalid shape dimension -2'),
                               ((-1, -1, 1), 'only one dimension can be inferred'),
                               ((-1, -1, -2), 'only one dimension can be inferred'),
                               ((-2, -1, -1), 'invalid shape dimension -2'),
                               ((-1, True, -1), 'only one dimension can be inferred')):
            with self.subTest(shape_type=tuple(type(d) for d in shape)):
                with self.assertRaisesRegex(RuntimeError, message) as caught:
                    x.view(shape)
                self.assertIs(type(caught.exception), RuntimeError)
        for shape in ((-2, True, 1.), (-1, -1, 2**63)):
            with self.assertRaises(TypeError) as caught:
                x.view(shape)
            self.assertIs(type(caught.exception), TypeError)
        for shape in ((1, True), (1, Index()), (1, 1, 1)):
            with self.assertRaises(trace.CompileTraceUnsupportedError):
                x.view(shape)

    def test_unsupported_expression_validation_without_hardware(self):
        metadata = trace.CompileTraceRecorder().input(shape=(1,), device='cuda:0').metadata
        for expression, expected in (('([1,2,3], foo=x)', TypeError),
                                     ('(2**63, k//1)', TypeError),
                                     ('(2**63, (k//1)+1)', TypeError),
                                     ('(2**63, (k//1)*x)', TypeError),
                                     ('(2**63, -k)', TypeError),
                                     ('(-2, -k)', RuntimeError),
                                     ('(0, True)', RuntimeError),
                                     ('((2,1,1))', RuntimeError),
                                     ('(-2, True)', RuntimeError),
                                     ('((-1,-1,1))', RuntimeError)):
            fn = make_program(f'def program(x):\n    k = 1\n    a = -x\n    return a.view{expression}\n')
            with self.subTest(expression=expression), \
                 patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')):
                with self.assertRaises(expected) as caught:
                    _compile_bytecode.lower_compile_graph(fn, (metadata,))
                self.assertIs(type(caught.exception), expected)

        class Poison:
            def __neg__(self):
                raise AssertionError('user negation')
            def __index__(self):
                raise AssertionError('user conversion')
        fn = make_program('def program(x):\n    return -x\n')
        instruction = next(i for i in dis.get_instructions(fn) if i.opname == 'UNARY_NEGATIVE')
        state = _compile_bytecode._LoweringState(fn)
        stack = [_compile_bytecode._BytecodeConstant(Poison())]
        _compile_bytecode._handle_unary_neg(None, {}, stack, fn, instruction, state, ())
        self.assertIs(stack[0].value, trace._RESHAPE_UNKNOWN_DIMENSION)
        self.assertIsInstance(state.local_constant_error, trace.CompileTraceUnsupportedError)

    def test_known_element_counts_before_capture_admission_without_hardware(self):
        for method in ('view', 'reshape'):
            for size, shape, message in ((1, (0, True), 'invalid for input of size 1'),
                                         (1, (2, 1, 1), 'invalid for input of size 1'),
                                         (0, (0, -1, 1), 'ambiguous'),
                                         (0, (-1, False), 'ambiguous')):
                recorder = trace.CompileTraceRecorder()
                x = recorder.input(shape=(size,), device='cuda:0')
                with self.subTest(method=method, size=size, shape=shape):
                    with self.assertRaisesRegex(RuntimeError, message) as caught:
                        getattr(x, method)(shape)
                    self.assertIs(type(caught.exception), RuntimeError)
                    self.assertEqual(recorder._operations, [])
            for size, shape in ((1, (1, True)), (1, (1, 1, 1)),
                                (0, (0, True)), (0, (0, 1, 1))):
                x = trace.CompileTraceRecorder().input(shape=(size,), device='cuda:0')
                with self.assertRaises(trace.CompileTraceUnsupportedError):
                    getattr(x, method)(shape)
        for planner in (trace._native._compile_trace_cuda_view_metadata,
                        trace._native._compile_trace_cuda_reshape_metadata):
            with self.assertRaisesRegex(RuntimeError, 'invalid for input of size 1') as caught:
                planner((1,), (1,), 0, (2, 1, 1))
            self.assertIs(type(caught.exception), RuntimeError)
            with self.assertRaises(NotImplementedError):
                planner((1,), (1,), 0, (1, 1, 1))
            with self.assertRaises(TypeError):
                planner((1,), (1,), 0, (0, True))



@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class CompileCudaViewTests(unittest.TestCase):
    def assert_pair(self, actual, expected):
        self.assertEqual(metadata(actual), metadata(expected))
        np.testing.assert_array_equal(read_bits(actual.contiguous()), read_bits(expected.contiguous()))

    def test_seeded_layouts_shapes_policies_identity_and_cache(self):
        rng = np.random.default_rng(19771978)
        layouts = (lambda x: x, lambda x: x.t(), lambda x: x[1:][:, 1:-1],
                   lambda x: x[1:][:, 1:-1].t(), lambda x: x.select(1, 1)[1:],
                   lambda x: x.select(0, 1).select(0, 2), lambda x: x[:1],
                   lambda x: x[:1][:, :1], lambda x: x[:0], lambda x: x[:, :0])
        for size in ((3, 7), (17, 31), tuple(map(int, rng.integers(33, 65, 2)))):
            data = rng.normal(size=size).astype(np.float32)
            for layout in layouts:
                x = layout(native.tensor(data).to('cuda:0'))
                y = layout(torch.tensor(data, device='cuda:0'))
                shapes = [tuple(x.shape), (-1,), (1, -1), (-1, 1)]
                if x.numel() == 1:
                    shapes.append(())
                for shape in shapes:
                    for fullgraph, dynamic in POLICIES:
                        with self.subTest(size=size, input=metadata(x), shape=shape, policy=(fullgraph, dynamic)):
                            torch._dynamo.reset()
                            fn = graphlet(shape)
                            try:
                                eager = fn(y)
                            except RuntimeError as expected_error:
                                with self.assertRaises(RuntimeError) as eager_error:
                                    fn(x)
                                self.assertEqual(str(eager_error.exception), str(expected_error))
                                with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                                    with self.assertRaises(RuntimeError) as caught:
                                        compile_with_cache(fn, fullgraph, dynamic)[0](x)
                                    self.assertIs(type(caught.exception), RuntimeError)
                                    self.assertEqual(str(caught.exception), str(expected_error))
                                continue
                            self.assert_pair(fn(x)[1][0], eager[1][0])
                            compiled, cache = compile_with_cache(fn, fullgraph, dynamic)
                            reference = torch.compile(fn, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
                            for warm in (False, True):
                                expected = reference(y)
                                with patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                                    if warm:
                                        with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')):
                                            out = call_without_python(compiled, {fn.__code__}, x)
                                    else:
                                        out = call_without_python(compiled, {fn.__code__}, x)
                                self.assertIs(out[0], x)
                                self.assertIs(out[1], out[2])
                                a, repeated, b, packed = out[1]
                                self.assertIs(a, repeated)
                                self.assertIsNot(a, x)
                                self.assertIsNot(a, b)
                                self.assertEqual(packed is a, expected[1][3] is expected[1][0])
                                for actual, ref in zip(out[1], expected[1]):
                                    self.assert_pair(actual, ref)
                                    if x.numel():
                                        self.assertEqual(actual.data_ptr() == x.data_ptr(), ref.data_ptr() == y.data_ptr())
                            self.assertEqual(len(cache.graphs), 1)

    def test_large_valid_empty_singleton_scalar(self):
        for shape, requested in (((), ()), ((), (1, 1)), ((1,), ()),
                                 ((0, 2**61), (0, 2**61)), ((0, 7), (2**61, 0)),
                                 ((0,), (-1,)), ((1, 1), (1,)), ((1031, 1033), (-1,))):
            x = native.zeros(shape).to('cuda:0')
            y = torch.zeros(shape, device='cuda:0')
            if len(shape) == 2 and shape != (1031, 1033):
                x, y = x.t(), y.t()
            for dynamic in (False, True):
                torch._dynamo.reset()
                fn = graphlet(requested)
                compiled, cache = compile_with_cache(fn, dynamic=dynamic)
                reference = torch.compile(fn, backend='eager', fullgraph=True, dynamic=dynamic)
                for _ in range(2):
                    self.assert_pair(compiled(x)[1][0], reference(y)[1][0])
                self.assertEqual(len(cache.graphs), 1)

    def test_bits_shared_mutation_and_lifetimes(self):
        bits = np.array([0, 0x80000000, 0x7f800001, 0xff800001, 0x7fc12345,
                         0xffc54321, 1, 0x80000001, 0x7f800000, 0xff800000,
                         0x3f800000, 0xbf800000], dtype=np.uint32)
        for layout in (lambda x: x, lambda x: x.t(), lambda x: x[:, 1:3],
                       lambda x: x.select(1, 1), lambda x: x.select(0, 1).select(0, 1)):
            x, y = native.zeros((3, 4), device='cuda:0'), torch.zeros((3, 4), device='cuda:0')
            write_bits(x, bits); write_bits(y, bits)
            a, b = layout(x), layout(y)
            fn = graphlet(tuple(a.shape))
            compiled, _ = compile_with_cache(fn)
            torch._dynamo.reset()
            reference = torch.compile(fn, backend='eager', fullgraph=True)
            out, expected = compiled(a)[1], reference(b)[1]
            self.assert_pair(out[0], expected[0])
            write_bits(x, bits[::-1].copy()); write_bits(y, bits[::-1].copy())
            self.assert_pair(out[0], expected[0])
            if out[0].is_contiguous():
                changed = np.full(out[0].numel(), 0x80000000, dtype=np.uint32)
                write_bits(out[0], changed); write_bits(expected[0], changed)
                self.assert_pair(x, y)
                self.assert_pair(out[2], expected[2])
            kept = out[0]
            del x, a, out, compiled
            gc.collect()
            self.assert_pair(kept, expected[0])

    def test_composition_and_dynamic_compatibility_recovery(self):
        fn = make_program('''def program(x, y):
    a = x.transpose(0, 1).reshape(-1).view(5, 7)
    b = y.t().view((7, 3))
    c = (-(a @ b.contiguous()) * 0.5).relu()
    return c.view(-1), c.sum(1).view(1, -1), a + a, a * 1.25
''')
        rng = np.random.default_rng(19771979)
        for fullgraph, dynamic in POLICIES:
            torch._dynamo.reset()
            compiled, _ = compile_with_cache(fn, fullgraph, dynamic)
            reference = torch.compile(fn, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
            data = [rng.normal(size=s).astype(np.float32) for s in ((7, 5), (3, 7))]
            args = [native.tensor(v).to('cuda:0') for v in data]
            refs = [torch.tensor(v, device='cuda:0') for v in data]
            for _ in range(2):
                for a, b in zip(call_without_python(compiled, {fn.__code__}, *args), reference(*refs)):
                    self.assertEqual(metadata(a), metadata(b))
                    np.testing.assert_allclose(a.cpu().tolist(), b.cpu().numpy(), rtol=2e-5, atol=2e-5)
        fn = make_program('def program(x):\n    early = x.contiguous()\n    return x.view(-1)\n')
        base = native.ones((5, 7)).to('cuda:0')
        ref = torch.ones((5, 7), device='cuda:0')
        for dynamic in (False, True):
            compiled, cache = compile_with_cache(fn, dynamic=dynamic)
            for rows in (1, 5, 0, 1):
                x, y = base[:rows][:, :3], ref[:rows][:, :3]
                if rows == 5:
                    with self.assertRaises(RuntimeError):
                        y.view(-1)
                    with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('partial execution')):
                        with self.assertRaisesRegex(RuntimeError, 'view size is not compatible') as caught:
                            compiled(x)
                        self.assertIs(type(caught.exception), RuntimeError)
                    continue
                out, expected = compiled(x), y.view(-1)
                self.assert_pair(out, expected)
                if x.numel():
                    self.assertEqual(out.data_ptr() == x.data_ptr(), rows == 1)
            self.assertEqual(len(cache.graphs), 1 if dynamic else 2)
        # Same rank/stride/offset, but the fixed shape becomes invalid on a cache hit.
        fn = make_program('def program(x):\n    a = x.contiguous()\n    return a.view(7)\n')
        compiled, _ = compile_with_cache(fn, dynamic=True)
        compiled(base[:1])
        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
            with self.assertRaises(RuntimeError) as caught:
                compiled(base)
            self.assertIs(type(caught.exception), RuntimeError)

    def test_tensor_valued_invalid_binding_before_execution(self):
        x, y = native.ones((1,)).to('cuda:0'), torch.ones((1,), device='cuda:0')
        for expression in ('(1, size=x)', '(1, foo=x)', '((1,), x)',
                           '(size=(1,), foo=x)', '(x, size=(1,))',
                           '(x, foo=1)', '(foo=x)'):
            for prefix in ('a = x', 'a = -x'):
                fn = make_program(f'def program(x):\n    {prefix}\n    return a.view{expression}\n')
                with self.subTest(expression=expression, prefix=prefix):
                    with self.assertRaises(TypeError):
                        fn(y)
                    for fullgraph, dynamic in POLICIES:
                        compiled, cache = compile_with_cache(fn, fullgraph, dynamic)
                        for attempt in range(2):
                            with self.subTest(policy=(fullgraph, dynamic), attempt=attempt), \
                                 patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')), \
                                 patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                                with self.assertRaises(TypeError):
                                    call_without_python(compiled, {fn.__code__}, x)
                                self.assertEqual(len(cache.graphs), 0)
        # Structurally valid nonconstant dimensions remain unsupported capture.
        for expression in ('(x)', '(1, x)', '(size=x)'):
            fn = make_program(f'def program(x):\n    return x.view{expression}\n')
            for fullgraph, dynamic in POLICIES:
                with self.subTest(unsupported=expression, policy=(fullgraph, dynamic)), \
                     patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')), \
                     patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                    with self.assertRaises(trace.CompileTraceUnsupportedError):
                        call_without_python(compile_with_cache(fn, fullgraph, dynamic)[0], {fn.__code__}, x)

    def test_partial_shapes_validate_known_errors_before_unsupported_capture(self):
        x, y = native.ones((1,)).to('cuda:0'), torch.ones((1,), device='cuda:0')
        errors = ('(True, x)', '(1.0, x)', '(2**63, x)', '(-2**63-1, x)',
                  '((1, x), foo=1)', '((1, x), size=(1,))', '((1, x), 1)',
                  '((True, x),)', '((1.0, x),)', '((2**63, x),)',
                  '(1, x, 1.0)', '(x, 1.0)', '((x, 1.0),)',
                  '(size=(1, x), foo=1)', '(shape, foo=1)', '(shape, size=(1,))')
        for expression in errors:
            for prefix in ('a = x', 'a = -x'):
                local = '    shape = (1, x)\n' if expression.startswith('(shape,') else ''
                fn = make_program(f'def program(x):\n    {prefix}\n{local}    return a.view{expression}\n')
                with self.subTest(expression=expression, prefix=prefix):
                    with self.assertRaises(TypeError):
                        fn(y)
                    for fullgraph, dynamic in POLICIES:
                        compiled, cache = compile_with_cache(fn, fullgraph, dynamic)
                        for attempt in range(2):
                            with self.subTest(policy=(fullgraph, dynamic), attempt=attempt), \
                                 patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')), \
                                 patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                                with self.assertRaises(TypeError):
                                    call_without_python(compiled, {fn.__code__}, x)
                                self.assertEqual(len(cache.graphs), 0)
        # Retaining a tuple for call validation must not admit nonconstant shapes,
        # mixed output pytrees, tuple helper arguments or tuple arithmetic.
        for body in ('return x.view((1, x))', 'return x.view(size=(1, x))',
                     'shape = (1, x)\n    return x.view(shape)',
                     'return (1, x)', 'out = (1, x)\n    return out',
                     'unused = (1, x)\n    return x',
                     'unused = (1, x)\n    unused = x\n    return x',
                     'return ((1, x), x)', 'return [(1, x), x]',
                     'return helper((1, x))', 'return x + (1, x)'):
            fn = make_program(f'def helper(x):\n    return -x\ndef program(x):\n    {body}\n')
            for fullgraph, dynamic in POLICIES:
                with self.subTest(unsupported=body, policy=(fullgraph, dynamic)), \
                     patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')), \
                     patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                    with self.assertRaises(trace.CompileTraceUnsupportedError):
                        call_without_python(compile_with_cache(fn, fullgraph, dynamic)[0], {fn.__code__}, x)

    def test_local_nonnumeric_constants_preserve_public_binding_errors(self):
        x, y = native.ones((1,)).to('cuda:0'), torch.ones((1,), device='cuda:0')
        for literal in ('None', "'invalid'", "b'invalid'", '1j', '...'):
            for expression in ('(d)', '(1, d)', '((1, d),)', '(1, foo=d)',
                               '(1, size=d)', '((1,), d)', '(size=d)'):
                for prefix in ('a = x', 'a = -x'):
                    local = make_program(f'def program(x):\n    {prefix}\n    d = {literal}\n    return a.view{expression}\n')
                    inline = make_program(f'def program(x):\n    {prefix}\n    return a.view{expression.replace("d", literal)}\n')
                    with self.subTest(literal=literal, expression=expression, prefix=prefix):
                        with self.assertRaises(TypeError) as reference_local:
                            local(y)
                        with self.assertRaises(TypeError) as reference_inline:
                            inline(y)
                        self.assertEqual(str(reference_local.exception), str(reference_inline.exception))
                        for fullgraph, dynamic in POLICIES:
                            compiled, cache = compile_with_cache(local, fullgraph, dynamic)
                            inline_compiled, _ = compile_with_cache(inline, fullgraph, dynamic)
                            for attempt in range(2):
                                with self.subTest(policy=(fullgraph, dynamic), attempt=attempt), \
                                     patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')), \
                                     patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                                    with self.assertRaises(TypeError) as actual:
                                        call_without_python(compiled, {local.__code__}, x)
                                    with self.assertRaises(TypeError) as inline_error:
                                        call_without_python(inline_compiled, {inline.__code__}, x)
                                    self.assertEqual(str(actual.exception), str(inline_error.exception))
                                    self.assertEqual(len(cache.graphs), 0)
        # Deferring a local's rejection must not accept otherwise unsupported
        # graphs, even if the local is unused, overwritten, or precedes view.
        for literal in ('None', "'invalid'", "b'invalid'", '1j', '...'):
            for body in ('return x', 'd = x\n    return d', 'return x.view(1)',
                         'return d', 'return (x, d)', 'return x * d'):
                fn = make_program(f'def program(x):\n    d = {literal}\n    {body}\n')
                for fullgraph, dynamic in POLICIES:
                    compiled, cache = compile_with_cache(fn, fullgraph, dynamic)
                    for attempt in range(2):
                        with self.subTest(literal=literal, unsupported=body, policy=(fullgraph, dynamic), attempt=attempt), \
                             patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')), \
                             patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                            with self.assertRaises(trace.CompileTraceUnsupportedError):
                                call_without_python(compiled, {fn.__code__}, x)
                            self.assertEqual(len(cache.graphs), 0)

    def test_validation_only_lists_and_computed_integer_dimensions(self):
        x, y = native.ones((1,)).to('cuda:0'), torch.ones((1,), device='cuda:0')
        for expression in ('([1], foo=x)', '(size=[1], foo=x)', '([1], x)',
                           '(1, [1], 1.)', '((1, [x]),)', '(2**63, n+1)',
                           '(-2, n+1)', '(2, n*n)', '(size=(n+1,))',
                           '(n-2, -1)', '(d, foo=x)', '(1, size=d)'):
            fn = make_program(f'def program(x):\n    n = 1\n    d = [1]\n    a = -x\n    return a.view{expression}\n')
            try:
                fn(y)
            except (TypeError, RuntimeError) as error:
                expected = type(error)
            else:
                self.fail(expression)
            for fullgraph, dynamic in POLICIES:
                compiled, cache = compile_with_cache(fn, fullgraph, dynamic)
                with self.subTest(expression=expression, policy=(fullgraph, dynamic)), \
                     patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')), \
                     patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                    with self.assertRaises(expected) as caught:
                        call_without_python(compiled, {fn.__code__}, x)
                    self.assertIs(type(caught.exception), expected)
                    self.assertFalse(cache.graphs)
        for body in ('return x.view([1])', 'return x.view(size=[1])',
                     'd = [1]\n    return x', 'd = [1]\n    d = x\n    return d',
                     'd = [1]\n    return (x, d)', 'return helper([1])',
                     'n = 1\n    return x.view(n+0)', 'n = 1\n    return x.view(n*1)',
                     'n = 1\n    return x.view(size=(n-0,))',
                     'n = 1\n    d = n+1\n    return x',
                     'n = 1\n    d = n+1\n    d = x\n    return d'):
            fn = make_program(f'def helper(x):\n    return x\ndef program(x):\n    {body}\n')
            with self.subTest(unsupported=body), \
                 patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')), \
                 patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                with self.assertRaises(trace.CompileTraceUnsupportedError):
                    call_without_python(compile_with_cache(fn)[0], {fn.__code__}, x)

    def test_unsupported_argument_expressions_preserve_public_errors(self):
        x, y = native.ones((1,)).to('cuda:0'), torch.ones((1,), device='cuda:0')
        cases = (
            ('([1,2,3], foo=x)', TypeError, 'invalid combination'),
            ('(d, foo=x)', TypeError, 'invalid combination'),
            ('(size=[1,2,3], foo=x)', TypeError, 'invalid combination'),
            ('([1,2,3], x)', TypeError, 'invalid combination'),
            ('(2**63, k//1)', TypeError, 'overflows signed int64'),
            ('(size=(2**63, k//1))', TypeError, 'overflows signed int64'),
            ('(k//1, 2**63)', TypeError, 'overflows signed int64'),
            ('(2**63, (k//1)+1)', TypeError, 'overflows signed int64'),
            ('(2**63, (k//1)*1)', TypeError, 'overflows signed int64'),
            ('(2**63, (k//1)+k)', TypeError, 'overflows signed int64'),
            ('(2**63, (k//1)*x)', TypeError, 'overflows signed int64'),
            ('(2**63, k+1.0)', TypeError, 'overflows signed int64'),
            ('(2**63, -k)', TypeError, 'overflows signed int64'),
            ('(-k, 2**63)', TypeError, 'overflows signed int64'),
            ('(2**63, -(k//1))', TypeError, 'overflows signed int64'),
            ('(2**63, -(-k)+1)', TypeError, 'overflows signed int64'),
            ('(0, -k)', RuntimeError, 'invalid for input of size 1'),
            ('(-2, -k)', RuntimeError, 'invalid shape dimension -2'),
            ('(-2, k//1)', RuntimeError, 'invalid shape dimension -2'),
            ('(-2, True)', RuntimeError, 'invalid shape dimension -2'),
            ('(size=(-2, True))', RuntimeError, 'invalid shape dimension -2'),
            ('((-1,-1,1))', RuntimeError, 'only one dimension can be inferred'),
            ('((-1,True,-1))', RuntimeError, 'only one dimension can be inferred'),
            ('((-2,1,1))', RuntimeError, 'invalid shape dimension -2'),
            ('((-1,-1,1.))', TypeError, 'must be int'),
            ('(-2, True, 1.)', TypeError, 'must be int'),
        )
        for expression, expected, message in cases:
            fn = make_program(f'def program(x):\n    k = 1\n    d = [1,2,3]\n    a = -x\n    return a.view{expression}\n')
            for eager_input in (x, y):
                with self.assertRaises(expected) as caught:
                    fn(eager_input)
                self.assertIs(type(caught.exception), expected)
            for fullgraph, dynamic in POLICIES:
                compiled, cache = compile_with_cache(fn, fullgraph, dynamic)
                for attempt in range(2):
                    with self.subTest(expression=expression, policy=(fullgraph, dynamic), attempt=attempt), \
                         patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')), \
                         patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                        with self.assertRaisesRegex(expected, message) as caught:
                            call_without_python(compiled, {fn.__code__}, x)
                        self.assertIs(type(caught.exception), expected)
                        self.assertFalse(cache.graphs)
        # Opaque expressions must never be executed or become accepted shapes,
        # even if discarded. This includes division by zero and enormous shifts.
        for expression in ('k//1', 'k/1', 'k%1', 'k**1', 'k<<1', 'k>>1',
                           'k&1', 'k|1', 'k^1', 'k//0', 'k<<2**63', '[1,2,3]', '[*x]',
                           '(k//1)+1', '(k//1)*x', '(k//1)@x', 'k+1.0',
                           '-k', '-(-k)', '-(k//1)', '-(k//1)+1'):
            for body in (f'return x.view({expression})',
                         f'd = {expression}\n    return x',
                         f'd = {expression}\n    d = x\n    return d'):
                fn = make_program(f'def program(x):\n    k = 1\n    {body}\n')
                for fullgraph, dynamic in POLICIES:
                    with self.subTest(unsupported=body, policy=(fullgraph, dynamic)), \
                         patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')), \
                         patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                        compiled, cache = compile_with_cache(fn, fullgraph, dynamic)
                        with self.assertRaises(trace.CompileTraceUnsupportedError):
                            call_without_python(compiled, {fn.__code__}, x)
                        self.assertFalse(cache.graphs)

    def test_known_element_count_errors_before_unsupported_capture(self):
        invalid = ((1, '(0, True)'), (1, '(1, False)'), (1, '(-1, False)'),
                   (1, '((2, 1, 1))'), (1, '(size=(2, 1, 1))'),
                   (0, '((0, -1, 1))'), (0, '(size=(0, -1, 1))'),
                   (0, '((-1, 0, True))'), (6, '((4, -1, True))'))
        for size, expression in invalid:
            x, y = native.ones((size,)).to('cuda:0'), torch.ones((size,), device='cuda:0')
            for prefix in ('a = x', 'a = -x'):
                fn = make_program(f'def program(x):\n    {prefix}\n    return a.view{expression}\n')
                with self.assertRaises(RuntimeError) as native_error:
                    fn(x)
                with self.assertRaises(RuntimeError) as reference_error:
                    fn(y)
                self.assertIs(type(native_error.exception), RuntimeError)
                self.assertIs(type(reference_error.exception), RuntimeError)
                self.assertEqual(str(native_error.exception), str(reference_error.exception))
                for fullgraph, dynamic in POLICIES:
                    compiled, cache = compile_with_cache(fn, fullgraph, dynamic)
                    for attempt in range(2):
                        with self.subTest(size=size, expression=expression, prefix=prefix, policy=(fullgraph, dynamic), attempt=attempt), \
                             patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')), \
                             patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                            with self.assertRaises(RuntimeError) as caught:
                                call_without_python(compiled, {fn.__code__}, x)
                            self.assertIs(type(caught.exception), RuntimeError)
                            self.assertEqual(str(caught.exception), str(reference_error.exception))
                            self.assertFalse(cache.graphs)
        for size, shape in ((1, (1, True)), (1, (1, 1, 1)), (0, (0, True)),
                            (0, (0, 1, 1)), (6, (2, -1, True))):
            x, y = native.ones((size,)).to('cuda:0'), torch.ones((size,), device='cuda:0')
            for expression in (f'({shape!r})', f'(size={shape!r})'):
                fn = make_program(f'def program(x):\n    return x.view{expression}\n')
                self.assert_pair(fn(x), fn(y))
                for fullgraph, dynamic in POLICIES:
                    compiled, cache = compile_with_cache(fn, fullgraph, dynamic)
                    with self.subTest(valid_out_of_scope=expression, size=size, policy=(fullgraph, dynamic)), \
                         patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')):
                        with self.assertRaises(trace.CompileTraceUnsupportedError):
                            call_without_python(compiled, {fn.__code__}, x)
                        self.assertFalse(cache.graphs)

    def test_public_binding_order_and_unsupported_forms(self):
        x, y = native.ones((1,)).to('cuda:0'), torch.ones((1,), device='cuda:0')
        errors = ['()', '(True,)', '((True,),)', '(1.,)', '(1, 1.)',
                  '(None,)', '(2**63,)', '(-2**63-1,)', '(-2,)', '(-1,-1)', '(2,)',
                  '((1,),1)', '(1,size=(1,))', '(1,foo=0)', '(foo=0)',
                  '(dtype=1)', '(shape=(1,))', '(True,foo=0)', '((1.,),foo=0)', '(2**63,foo=0)', '(size=1)',
                  '(-2,1.)', '(2**63,1.)']
        for expression in errors:
            fn = make_program(f'def program(x):\n    a = -x\n    return a.view{expression}\n')
            with self.subTest(expression=expression):
                try:
                    fn(y)
                except Exception as error:
                    expected = type(error)
                else:
                    self.fail('reference must reject')
                with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')), \
                     patch.object(trace, '_execute_operation', side_effect=AssertionError('early execution')):
                    with self.assertRaises(expected) as caught:
                        compile_with_cache(fn)[0](x)
                    self.assertIs(type(caught.exception), expected)
        for expression in ('(1,True)', '((1,True),)', '([1],)', '((1,1,1),)', '(x.shape)', '(x.numel())'):
            fn = make_program(f'def program(x):\n    return x.view{expression}\n')
            with self.subTest(unsupported=expression), self.assertRaises(NotImplementedError):
                compile_with_cache(fn)[0](x)
        for expression in ('(size=(1,))', '(1,1)', '((1,1),)', '(())'):
            fn = make_program(f'def program(x):\n    return x.view{expression}\n')
            self.assert_pair(compile_with_cache(fn)[0](x), fn(y))
        for shape in ((0, -1), (-1, 0)):
            fn = graphlet(shape)
            with self.assertRaises(RuntimeError) as caught, patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                compile_with_cache(fn)[0](native.zeros((0,), device='cuda:0'))
            self.assertIs(type(caught.exception), RuntimeError)
        for text in ('def program(x, n):\n    return x.view(n)\n',
                     'def program(x):\n    return x.view_as(x)\n',
                     'def program(x):\n    return x.reshape_as(x)\n',
                     'def program(x):\n    return m.reshape(x, shape=(1,))\n'):
            fn = make_program(text, module=native)
            with self.assertRaises(TypeError if 'x, n' in text else NotImplementedError):
                compile_with_cache(fn)[0](*((x, 1) if 'x, n' in text else (x,)))
        # A compatible view may remain strided; arithmetic still needs packing.
        for expression in ('-x.view(-1)', 'x.view(-1) * 2.'):
            fn = make_program(f'def program(x):\n    return {expression}\n')
            with self.assertRaises(NotImplementedError):
                compile_with_cache(fn)[0](native.ones((3, 7)).to('cuda:0').select(1, 1))

    def test_live_globals_locals_and_helpers(self):
        x = native.ones((3, 7)).to('cuda:0')
        for form in ('ROWS, -1', '(ROWS, -1)', 'size=(ROWS, -1)'):
            fn = make_program(f'def program(x):\n    return x.view({form})\n', ROWS=1)
            for dynamic in (False, True):
                compiled, cache = compile_with_cache(fn, dynamic=dynamic)
                for rows in (1, 3, 7):
                    fn.__globals__['ROWS'] = rows
                    self.assertEqual(tuple(compiled(x).shape), (rows, 21 // rows))
                self.assertEqual(len(cache.graphs), 3)
                for rows, error in ((True, TypeError), (1., TypeError), (2**63, TypeError),
                                    (2, RuntimeError), (np.int64(1), NotImplementedError)):
                    fn.__globals__['ROWS'] = rows
                    with self.assertRaises(error) as caught, patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                        compiled(x)
                    if error in (TypeError, RuntimeError):
                        self.assertIs(type(caught.exception), error)
        fn = make_program('''def helper(x):
    rows = 1
    dims = (rows, -1)
    return x.view(dims)
def program(x):
    return helper(x) + bias.view(1, -1)
''', bias=x)
        compiled, _ = compile_with_cache(fn)
        self.assertEqual(call_without_python(compiled, {fn.__code__, fn.__globals__['helper'].__code__}, x).cpu().tolist(), [[2.] * 21])
        fn.__globals__['bias'] = native.full((3, 7), 3.).to('cuda:0')
        self.assertEqual(compiled(x).cpu().tolist(), [[4.] * 21])
        fn.__globals__['bias'] = x.cpu()
        with self.assertRaises(NotImplementedError):
            compiled(x)

    def test_cached_graph_early_late_and_repeated_output_metadata(self):
        x = native.ones((1, 1)).to('cuda:0')
        fn = make_program('''def program(x):
    a = x.view(1, 1)
    b = a.contiguous().view(1, 1)
    out = [b]
    return out, out
''')
        for dynamic in (False, True):
            compiled, cache = compile_with_cache(fn, dynamic=dynamic)
            compiled(x)
            key, graph = next(iter(cache.graphs.items()))
            bad = [replace(graph, dynamic=1)]
            for i in (0, len(graph.operations)-1):
                for change in ({'shape': None}, {'shape': (True, 1)}, {'shape': (1., 1)},
                               {'shape': (2, 1)}, {'shape': [1, 1]}, {'shape': (-1, -1)},
                               {'shape': (1, 1, 1)}, {'shape': (2**63,)},
                               {'scalar': 1}, {'axes': (0, 1)}, {'reduction': (1, False)},
                               {'inputs': ()}, {'inputs': ('arg0', 'arg0')}, {'inputs': ['arg0']},
                               {'op': 'call_reduction'}, {'target': True}, {'name': 0}):
                    ops = list(graph.operations); ops[i] = replace(ops[i], **change)
                    bad.append(replace(graph, operations=tuple(ops)))
            ops = list(graph.operations); ops[1] = replace(ops[1], shape=(1, 1))
            bad.append(replace(graph, operations=tuple(ops)))
            for change in ({'shape': (True, 1)}, {'shape': (1., 1)}, {'stride': (1, False)},
                           {'storage_offset': 0.}, {'storage_offset': 1}, {'requires_grad': 0},
                           {'requires_grad': True}, {'device': 'cuda:1'}, {'shape': (2, 1)},
                           {'dtype': trace.CompileTraceDType('torch.float64')}):
                for i, op in enumerate(graph.operations):
                    ops = list(graph.operations); ops[i] = replace(op, metadata=replace(op.metadata, **change))
                    bad.append(replace(graph, operations=tuple(ops)))
                first, second = graph.output_metadata.elements
                leaf = replace(second.elements[0], **change)
                bad.append(replace(graph, output_metadata=replace(graph.output_metadata,
                    elements=(first, replace(second, elements=(leaf,))))))
            cpu = x.cpu()
            bad.append(replace(graph, captures=(trace.CompileTraceCapture('unused', cpu, trace._metadata_from_native_tensor(cpu)),)))
            with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')), \
                 patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                for invalid in bad:
                    cache.graphs[key] = invalid
                    with self.assertRaises((NotImplementedError, ValueError, TypeError, RuntimeError)):
                        compiled(x)
            cache.graphs[key] = graph
            self.assertIs(compiled(x)[0][0].__class__, native.Tensor)
            out = compiled(x)
            self.assertIs(out[0], out[1])

    def test_same_device_context_and_unused_actual_inputs(self):
        driver = ctypes.CDLL('libcuda.so.1')
        driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        def context():
            pointer = ctypes.c_void_p()
            self.assertEqual(driver.cuCtxGetCurrent(ctypes.byref(pointer)), 0)
            return pointer.value
        x = native.ones((3, 7)).to('cuda:0')
        torch.empty(1, device='cuda:0')
        before = context()
        fn = make_program('def program(x, unused):\n    return x.view(size=(3, -1))\n')
        compiled, _ = compile_with_cache(fn)
        for _ in range(2):
            self.assertEqual(compiled(x, x).data_ptr(), x.data_ptr())
            self.assertEqual(context(), before)
        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
            with self.assertRaises(NotImplementedError):
                compiled(x, x.cpu())
            bad = make_program('def program(x):\n    a = -x\n    return a.t().view(-1)\n')
            with self.assertRaisesRegex(RuntimeError, 'view size is not compatible') as caught:
                compile_with_cache(bad)[0](x)
            self.assertIs(type(caught.exception), RuntimeError)
        self.assertEqual(context(), before)

    def test_guards_methods_and_blocked_reference_import(self):
        fn = make_program('def program(x):\n    return x.t().contiguous().view(-1)\n')
        x = native.ones((3, 7)).to('cuda:0')
        compiled, cache = compile_with_cache(fn)
        compiled(x)
        graph = next(iter(cache.graphs.values()))
        for wrong in (x.cpu(), x.t(), x[1:], x[:, 1:], native.ones((1, 3, 7)).to('cuda:0'),
                      native.ones((3, 7), requires_grad=True)):
            with self.assertRaises((ValueError, NotImplementedError)), patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                graph.forward(wrong)
        with patch.object(native.Tensor, 'view', side_effect=AssertionError('method replay')), \
             patch.object(native.Tensor, 'contiguous', side_effect=AssertionError('packing replay')):
            self.assertEqual(graph.forward(x).cpu().tolist(), [1.] * 21)
        def reject(*args):
            raise AssertionError('method or descriptor replay')
        for replacement in (reject, property(reject)):
            with patch.object(native.Tensor, 'view', replacement):
                for wrapper in (compiled, compile_with_cache(fn)[0]):
                    with self.assertRaises(NotImplementedError):
                        wrapper(x)
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('reference import')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as n
def f(x):
    a = x.view(-1)
    b = x.t().contiguous().view(-1)
    return a, a, b
def reject(frame, event, arg):
    if event == 'call' and frame.f_code is f.__code__:
        raise AssertionError('Python replay')
x = n.tensor([[1.,2.,3.],[4.,5.,6.]]).to('cuda:0')
f_compiled = n.compile(f, backend='eager', fullgraph=True)
sys.setprofile(reject)
for _ in range(3):
    a, repeated, b = f_compiled(x)
    assert a is repeated and a is not x and b is not x
    assert a.data_ptr() == x.data_ptr() != b.data_ptr()
    assert b.cpu().tolist() == [1.,4.,2.,5.,3.,6.]
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class ViewDeviceTests(unittest.TestCase):
    def test_context_restoration_and_unused_mixed_capture(self):
        driver = ctypes.CDLL('libcuda.so.1')
        driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        def context():
            pointer = ctypes.c_void_p()
            self.assertEqual(driver.cuCtxGetCurrent(ctypes.byref(pointer)), 0)
            return pointer.value
        previous = torch.cuda.current_device()
        try:
            for ordinal in (0, 1):
                x = native.ones((3, 7)).to(f'cuda:{ordinal}')
                torch.cuda.set_device(1 - ordinal)
                torch.empty(1, device=f'cuda:{1-ordinal}')
                before = context()
                fn = make_program('def program(x):\n    return x.view(-1), x.t().view(7, 3)\n')
                compiled, cache = compile_with_cache(fn)
                for _ in range(2):
                    for out in compiled(x):
                        self.assertEqual(str(out.device), f'cuda:{ordinal}')
                        self.assertEqual(out.cpu().reshape(-1).tolist(), [1.] * 21)
                    self.assertEqual(torch.cuda.current_device(), 1 - ordinal)
                    self.assertEqual(context(), before)
                key, graph = next(iter(cache.graphs.items()))
                other = native.ones((3, 7)).to(f'cuda:{1-ordinal}')
                cache.graphs[key] = replace(graph, captures=(trace.CompileTraceCapture('unused', other, trace._metadata_from_native_tensor(other)),))
                with self.assertRaises(NotImplementedError), patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                    compiled(x)
                self.assertEqual(context(), before)
                unused = make_program('def program(x, unused):\n    return x.view(-1)\n')
                with self.assertRaises(NotImplementedError), patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                    compile_with_cache(unused)[0](x, other)
                self.assertEqual(context(), before)
        finally:
            torch.cuda.set_device(previous)
