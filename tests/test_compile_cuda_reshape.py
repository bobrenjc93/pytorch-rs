"""Non-scoring bounded CUDA reshape capture; no evaluator workloads are changed."""
import ctypes
from dataclasses import replace
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
    a = x.reshape(shape)
    b = x.reshape({shape!r})
    shared = [a, a, b, a.contiguous()]
    return x, shared, shared
''')


class ReshapeMetadataTests(unittest.TestCase):
    def test_shared_native_planner_without_hardware(self):
        for shape, strides, requested, expected, offset in (
            ((), (), (1, 1), (1, 1), 7),
            ((6,), (3,), (2, 3), (9, 3), 7),
            ((3, 2), (1, 3), (6,), (1,), 0),
            ((0, 7), (1, 9), (0, 7), (1, 9), 7),
            ((0, 7), (1, 9), (7, 0), (1, 1), 7),
        ):
            recorder = trace.CompileTraceRecorder()
            x = recorder.input(shape=shape, stride=strides, device='cuda:0', storage_offset=7)
            out = x.reshape(requested)
            self.assertEqual(out.metadata.stride, expected)
            self.assertEqual(out.metadata.storage_offset, offset)
            self.assertEqual(recorder._operations[0].shape, requested)
            self.assertIsNone(recorder._operations[0].axes)
        for options in ({'device': 'cpu'}, {'requires_grad': True},
                        {'shape': (1, 2, 3)}, {'dtype': trace.CompileTraceDType('torch.float64')}):
            recorder = trace.CompileTraceRecorder()
            with self.assertRaises(NotImplementedError):
                recorder.input(**({'shape': (3, 7), 'device': 'cuda:0'} | options)).reshape(-1)
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
                x.reshape(shape)
        for shape in ((True,), (1.,), (1, 1.), (2**63,), (-2**63-1,)):
            with self.assertRaises(TypeError):
                x.reshape(shape)
        self.assertEqual(recorder._operations, [])


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class CompileCudaReshapeTests(unittest.TestCase):
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

    def test_large_valid_empty_singleton_scalar_and_pack(self):
        for shape, requested in (((), ()), ((), (1, 1)), ((1,), ()),
                                 ((0, 2**61), (0, 2**61)), ((0, 7), (2**61, 0)),
                                 ((0,), (-1,)), ((1, 1), (1,)), ((1031, 1033), (-1,))):
            x = native.zeros(shape).to('cuda:0')
            y = torch.zeros(shape, device='cuda:0')
            if len(shape) == 2:
                x, y = x.t(), y.t()
            for dynamic in (False, True):
                torch._dynamo.reset()
                fn = graphlet(requested)
                compiled, cache = compile_with_cache(fn, dynamic=dynamic)
                reference = torch.compile(fn, backend='eager', fullgraph=True, dynamic=dynamic)
                for _ in range(2):
                    self.assert_pair(compiled(x)[1][0], reference(y)[1][0])
                self.assertEqual(len(cache.graphs), 1)

    def test_bits_mutation_independence_and_lifetimes(self):
        bits = np.array([0, 0x80000000, 0x7f800001, 0xff800001, 0x7fc12345,
                         0xffc54321, 1, 0x80000001, 0x7f800000, 0xff800000,
                         0x3f800000, 0xbf800000], dtype=np.uint32)
        for layout in (lambda x: x, lambda x: x.t(), lambda x: x[:, 1:3],
                       lambda x: x.select(1, 1), lambda x: x.select(0, 1).select(0, 1)):
            x, y = native.zeros((3, 4), device='cuda:0'), torch.zeros((3, 4), device='cuda:0')
            write_bits(x, bits); write_bits(y, bits)
            a, b = layout(x), layout(y)
            fn = graphlet((-1,))
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

    def test_composition_and_dynamic_alias_pack_transition(self):
        fn = make_program('''def program(x, y):
    a = x.transpose(0, 1).reshape(-1).reshape(5, 7)
    b = y.t().reshape((7, 3))
    c = -(a @ b.contiguous()) * 0.5
    return c.reshape(-1), c.sum(1).reshape(1, -1), a + a, a * 1.25
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
        fn = make_program('def program(x):\n    return x.reshape(-1)\n')
        base = native.ones((5, 7)).to('cuda:0')
        ref = torch.ones((5, 7), device='cuda:0')
        for dynamic in (False, True):
            compiled, cache = compile_with_cache(fn, dynamic=dynamic)
            for rows in (1, 5, 0, 1):
                x, y = base[:rows][:, :3], ref[:rows][:, :3]
                out, expected = compiled(x), y.reshape(-1)
                self.assert_pair(out, expected)
                if x.numel():
                    self.assertEqual(out.data_ptr() == x.data_ptr(), rows == 1)
            self.assertEqual(len(cache.graphs), 1 if dynamic else 3)
        # Same rank/stride/offset, but the fixed shape becomes invalid on a cache hit.
        fn = make_program('def program(x):\n    a = x.contiguous()\n    return a.reshape(7)\n')
        compiled, _ = compile_with_cache(fn, dynamic=True)
        compiled(base[:1])
        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
            with self.assertRaises(RuntimeError):
                compiled(base)

    def test_tensor_valued_invalid_binding_before_execution(self):
        x, y = native.ones((1,)).to('cuda:0'), torch.ones((1,), device='cuda:0')
        for expression in ('(1, shape=x)', '(1, foo=x)', '((1,), x)',
                           '(shape=(1,), foo=x)', '(x, shape=(1,))',
                           '(x, foo=1)', '(foo=x)'):
            for prefix in ('a = x', 'a = -x'):
                fn = make_program(f'def program(x):\n    {prefix}\n    return a.reshape{expression}\n')
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
        for expression in ('(x)', '(1, x)', '(shape=x)'):
            fn = make_program(f'def program(x):\n    return x.reshape{expression}\n')
            for fullgraph, dynamic in POLICIES:
                with self.subTest(unsupported=expression, policy=(fullgraph, dynamic)), \
                     patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native execution')), \
                     patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                    with self.assertRaises(trace.CompileTraceUnsupportedError):
                        call_without_python(compile_with_cache(fn, fullgraph, dynamic)[0], {fn.__code__}, x)

    def test_partial_shapes_validate_known_errors_before_unsupported_capture(self):
        x, y = native.ones((1,)).to('cuda:0'), torch.ones((1,), device='cuda:0')
        errors = ('(True, x)', '(1.0, x)', '(2**63, x)', '(-2**63-1, x)',
                  '((1, x), foo=1)', '((1, x), shape=(1,))', '((1, x), 1)',
                  '((True, x),)', '((1.0, x),)', '((2**63, x),)',
                  '(1, x, 1.0)', '(x, 1.0)', '((x, 1.0),)',
                  '(shape=(1, x), foo=1)', '(shape, foo=1)', '(shape, shape=(1,))')
        for expression in errors:
            for prefix in ('a = x', 'a = -x'):
                local = '    shape = (1, x)\n' if expression.startswith('(shape,') else ''
                fn = make_program(f'def program(x):\n    {prefix}\n{local}    return a.reshape{expression}\n')
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
        for body in ('return x.reshape((1, x))', 'return x.reshape(shape=(1, x))',
                     'shape = (1, x)\n    return x.reshape(shape)',
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
                               '(1, shape=d)', '((1,), d)', '(shape=d)'):
                for prefix in ('a = x', 'a = -x'):
                    local = make_program(f'def program(x):\n    {prefix}\n    d = {literal}\n    return a.reshape{expression}\n')
                    inline = make_program(f'def program(x):\n    {prefix}\n    return a.reshape{expression.replace("d", literal)}\n')
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
        # graphs, even if the local is unused, overwritten, or precedes reshape.
        for literal in ('None', "'invalid'", "b'invalid'", '1j', '...'):
            for body in ('return x', 'd = x\n    return d', 'return x.reshape(1)',
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

    def test_public_binding_order_and_unsupported_forms(self):
        x, y = native.ones((1,)).to('cuda:0'), torch.ones((1,), device='cuda:0')
        errors = ['()', '(True,)', '((True,),)', '(1.,)', '(1, 1.)',
                  '(None,)', '(2**63,)', '(-2**63-1,)', '(-2,)', '(-1,-1)', '(2,)',
                  '((1,),1)', '(1,shape=(1,))', '(1,foo=0)', '(foo=0)',
                  '(True,foo=0)', '((1.,),foo=0)', '(2**63,foo=0)', '(shape=1)',
                  '(-2,1.)', '(2**63,1.)']
        for expression in errors:
            fn = make_program(f'def program(x):\n    a = -x\n    return a.reshape{expression}\n')
            with self.subTest(expression=expression):
                try:
                    fn(y)
                except Exception as error:
                    expected = type(error)
                else:
                    self.fail('reference must reject')
                with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')), \
                     patch.object(trace, '_execute_operation', side_effect=AssertionError('early execution')):
                    with self.assertRaises(expected):
                        compile_with_cache(fn)[0](x)
        for expression in ('(1,True)', '((1,True),)', '([1],)', '((1,1,1),)', '(x.shape)', '(x.numel())'):
            fn = make_program(f'def program(x):\n    return x.reshape{expression}\n')
            with self.subTest(unsupported=expression), self.assertRaises(NotImplementedError):
                compile_with_cache(fn)[0](x)
        for expression in ('(shape=(1,))', '(1,1)', '((1,1),)', '(())'):
            fn = make_program(f'def program(x):\n    return x.reshape{expression}\n')
            self.assert_pair(compile_with_cache(fn)[0](x), fn(y))
        for shape in ((0, -1), (-1, 0)):
            fn = graphlet(shape)
            with self.assertRaises(RuntimeError), patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                compile_with_cache(fn)[0](native.zeros((0,), device='cuda:0'))
        for text in ('def program(x, n):\n    return x.reshape(n)\n',
                     'def program(x):\n    return x.view(-1)\n',
                     'def program(x):\n    return x.reshape_as(x)\n',
                     'def program(x):\n    return module.reshape(x, (1,))\n'):
            fn = make_program(text, module=native)
            with self.assertRaises(TypeError if 'x, n' in text else NotImplementedError):
                compile_with_cache(fn)[0](*((x, 1) if 'x, n' in text else (x,)))
        # View-compatible reshape may remain strided; arithmetic still needs packing.
        for expression in ('-x.reshape(-1)', 'x.reshape(-1) * 2.'):
            fn = make_program(f'def program(x):\n    return {expression}\n')
            with self.assertRaises(NotImplementedError):
                compile_with_cache(fn)[0](native.ones((3, 7)).to('cuda:0').select(1, 1))

    def test_live_globals_locals_and_helpers(self):
        x = native.ones((3, 7)).to('cuda:0')
        for form in ('ROWS, -1', '(ROWS, -1)'):
            fn = make_program(f'def program(x):\n    return x.reshape({form})\n', ROWS=1)
            for dynamic in (False, True):
                compiled, cache = compile_with_cache(fn, dynamic=dynamic)
                for rows in (1, 3, 7):
                    fn.__globals__['ROWS'] = rows
                    self.assertEqual(tuple(compiled(x).shape), (rows, 21 // rows))
                self.assertEqual(len(cache.graphs), 3)
                for rows, error in ((True, TypeError), (1., TypeError), (2**63, TypeError),
                                    (2, RuntimeError), (np.int64(1), NotImplementedError)):
                    fn.__globals__['ROWS'] = rows
                    with self.assertRaises(error), patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                        compiled(x)
        fn = make_program('''def helper(x):
    rows = 1
    dims = (rows, -1)
    return x.reshape(dims)
def program(x):
    return helper(x) + bias.reshape(1, -1)
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
    a = x.reshape(1, 1)
    b = a.contiguous().reshape(1, 1)
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

    def test_guards_methods_and_blocked_reference_import(self):
        fn = make_program('def program(x):\n    return x.t().reshape(-1)\n')
        x = native.ones((3, 7)).to('cuda:0')
        compiled, cache = compile_with_cache(fn)
        compiled(x)
        graph = next(iter(cache.graphs.values()))
        for wrong in (x.cpu(), x.t(), x[1:], x[:, 1:], native.ones((1, 3, 7)).to('cuda:0'),
                      native.ones((3, 7), requires_grad=True)):
            with self.assertRaises((ValueError, NotImplementedError)), patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                graph.forward(wrong)
        with patch.object(native.Tensor, 'reshape', side_effect=AssertionError('method replay')), \
             patch.object(native.Tensor, 'contiguous', side_effect=AssertionError('packing replay')):
            self.assertEqual(graph.forward(x).cpu().tolist(), [1.] * 21)
        def reject(*args):
            raise AssertionError('method or descriptor replay')
        for replacement in (reject, property(reject)):
            with patch.object(native.Tensor, 'reshape', replacement):
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
    a = x.reshape(-1)
    b = x.t().reshape(-1)
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
class ReshapeDeviceTests(unittest.TestCase):
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
                fn = make_program('def program(x):\n    return x.reshape(-1), x.t().reshape(-1)\n')
                compiled, cache = compile_with_cache(fn)
                for _ in range(2):
                    for out in compiled(x):
                        self.assertEqual(str(out.device), f'cuda:{ordinal}')
                        self.assertEqual(out.cpu().tolist(), [1.] * 21)
                    self.assertEqual(torch.cuda.current_device(), 1 - ordinal)
                    self.assertEqual(context(), before)
                key, graph = next(iter(cache.graphs.items()))
                other = native.ones((3, 7)).to(f'cuda:{1-ordinal}')
                cache.graphs[key] = replace(graph, captures=(trace.CompileTraceCapture('unused', other, trace._metadata_from_native_tensor(other)),))
                with self.assertRaises(NotImplementedError), patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                    compiled(x)
                self.assertEqual(context(), before)
        finally:
            torch.cuda.set_device(previous)
