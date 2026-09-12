"""Non-scoring no-argument squeeze graphlets; scoring corpora stay unchanged."""
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


def squeeze(x):
    return x.squeeze()


def nested(x):
    a = x.squeeze()
    b = x.squeeze()
    shared = [a, (b, a.squeeze())]
    return x, shared, shared, a


def composition(x, y):
    a = x.squeeze().t().transpose(0, 1).contiguous()
    b = y.squeeze().contiguous()
    c = (a @ b).relu()
    d = c.sum(1, keepdim=True).squeeze()
    return d, -(d.view(1, -1).reshape(-1).squeeze() * 0.5) + d


class SqueezeMetadataTests(unittest.TestCase):
    def test_consumer_declarations_and_runtime_ranks_are_independent(self):
        for expression in (lambda x: -x, lambda x: x.relu(), lambda x: x + x):
            recorder = trace.CompileTraceRecorder()
            x = recorder.input(shape=(1, 7), device='cuda:0').squeeze()
            graph = recorder.finish(expression(x))
            op = graph.operations[-1]
            declared = {x.name: x.metadata}
            for shape in ((), (7,), (3, 7), (0, 7)):
                current = replace(x.metadata, shape=shape, stride=trace._contiguous_stride(shape))
                expected = trace._expected_operation_metadata(
                    op, {x.name: current}, grad_enabled=False, declared_values=declared,
                )
                self.assertEqual(expected.shape, shape)
                self.assertEqual(expected.stride, current.stride)
                if len(shape) != len(x.metadata.shape):
                    bad = replace(op, metadata=expected)
                    with self.assertRaises(trace.CompileTraceUnsupportedError) as caught:
                        trace._expected_operation_metadata(
                            bad, {x.name: current}, grad_enabled=False, declared_values=declared,
                        )
                    self.assertIs(type(caught.exception), trace.CompileTraceUnsupportedError)

    def test_layouts_and_boundaries_without_hardware(self):
        for shape, stride in (((), ()), ((1,), (19,)), ((7,), (3,)),
                              ((1, 7), (123, 3)), ((7, 1), (3, 99)),
                              ((3, 7), (1, 9)), ((1, 0), (19, 7)),
                              ((0, 2**32), (2**32, 1))):
            recorder = trace.CompileTraceRecorder()
            x = recorder.input(shape=shape, stride=stride, device='cuda:0', storage_offset=11)
            y = x.squeeze()
            self.assertIsNot(x, y)
            self.assertEqual(y.metadata.shape, tuple(n for n in shape if n != 1))
            self.assertEqual(y.metadata.stride, tuple(s for n, s in zip(shape, stride) if n != 1))
            self.assertEqual(y.metadata.storage_offset, 11)
            self.assertEqual(recorder.finish(y).operations[0].target, 'squeeze')
        for change in ({'device': 'cpu'}, {'requires_grad': True}, {'shape': (1, 2, 1)},
                       {'dtype': trace.CompileTraceDType('torch.float64')}):
            recorder = trace.CompileTraceRecorder()
            with self.assertRaises(trace.CompileTraceUnsupportedError) as caught:
                recorder.input(**({'shape': (1, 7), 'device': 'cuda:0'} | change)).squeeze()
            self.assertIs(type(caught.exception), trace.CompileTraceUnsupportedError)
            self.assertEqual(recorder._operations, [])


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class CompileCudaSqueezeTests(unittest.TestCase):
    def assert_pair(self, actual, expected):
        self.assertEqual(metadata(actual), metadata(expected))
        np.testing.assert_array_equal(read_bits(actual.contiguous()), read_bits(expected.contiguous()))

    def test_seeded_layouts_fresh_wrappers_and_nested_identity(self):
        rng = np.random.default_rng(198001)
        shapes = [(3, 7), (17, 31)] + [tuple(map(int, rng.integers(3, 30, 2))) for _ in range(2)]
        layouts = (lambda x: x, lambda x: x.t(), lambda x: x[:1], lambda x: x[:, :1],
                   lambda x: x[:1].t(), lambda x: x[1:2][:, 1:],
                   lambda x: x[:, 1:2], lambda x: x.select(1, 1)[1:],
                   lambda x: x.select(0, 1).select(0, 1), lambda x: x[:1][:, :1],
                   lambda x: x[:0], lambda x: x[:1][:, :0], lambda x: x[:0][:, :1])
        for shape in shapes:
            for layout in layouts:
                for fullgraph, dynamic in POLICIES:
                    torch._dynamo.reset()
                    compiled, cache = compile_with_cache(nested, fullgraph, dynamic)
                    reference = torch.compile(nested, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
                    previous = None
                    for warm in (False, True):
                        data = rng.normal(size=shape).astype(np.float32)
                        x = layout(native.tensor(data).to('cuda:0'))
                        y = layout(torch.tensor(data, device='cuda:0'))
                        self.assertEqual(metadata(x), metadata(y))
                        expected = reference(y)
                        with patch.object(trace, '_execute_operation', side_effect=AssertionError('Python node replay')):
                            if warm:
                                with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')):
                                    actual = call_without_python(compiled, {nested.__code__}, x)
                            else:
                                actual = call_without_python(compiled, {nested.__code__}, x)
                        self.assertIs(actual[0], x)
                        self.assertIs(actual[1], actual[2])
                        self.assertIs(actual[1][0], actual[3])
                        leaves = [actual[3], *actual[1][1]]
                        self.assertEqual(len({id(v) for v in leaves}), 3)
                        for a, b in zip(leaves, [expected[3], *expected[1][1]]):
                            self.assertIsNot(a, x)
                            self.assertIsNot(a, previous)
                            self.assertEqual(a.data_ptr(), x.data_ptr())
                            self.assert_pair(a, b)
                        previous = actual[3]
                    self.assertEqual(len(cache.graphs), 1)

    def test_scalar_vector_and_empty_constructors(self):
        for shape in ((), (1,), (7,), (0,), (1, 0), (0, 1), (0, 2**32)):
            for fullgraph, dynamic in POLICIES:
                torch._dynamo.reset()
                compiled, _ = compile_with_cache(squeeze, fullgraph, dynamic)
                reference = torch.compile(squeeze, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
                # Equal canonical layouts, including empty inputs.
                x = native.zeros(shape).to('cuda:0')
                y = torch.zeros(shape, device='cuda:0')
                self.assertEqual(metadata(x), metadata(y))
                for _ in range(2):
                    out = call_without_python(compiled, {squeeze.__code__}, x)
                    self.assertIsNot(out, x)
                    self.assert_pair(out, reference(y))

    def test_dynamic_singletons_disappear_and_reappear(self):
        data = np.arange(35, dtype=np.float32).reshape(5, 7)
        a, b = native.tensor(data).to('cuda:0'), torch.tensor(data, device='cuda:0')
        compiled, cache = compile_with_cache(nested, dynamic=True)
        torch._dynamo.reset()
        reference = torch.compile(nested, backend='eager', fullgraph=True, dynamic=True)
        for rows, cols in ((1, 1), (5, 7), (1, 7), (5, 1), (0, 1), (1, 0), (1, 1)):
            x, y = a[:rows][:, :cols], b[:rows][:, :cols]
            self.assertEqual(metadata(x), metadata(y))
            if cache.graphs:
                with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')):
                    out = compiled(x)
            else:
                out = compiled(x)
            self.assert_pair(out[3], reference(y)[3])
        self.assertEqual(len(cache.graphs), 1)

    def test_compositions_all_policies(self):
        rng = np.random.default_rng(198002)
        for fullgraph, dynamic in POLICIES:
            torch._dynamo.reset()
            compiled, _ = compile_with_cache(composition, fullgraph, dynamic)
            reference = torch.compile(composition, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
            for _ in range(2):
                data = [rng.normal(size=s).astype(np.float32) for s in ((7, 5), (3, 7))]
                args = [native.tensor(v).to('cuda:0').t() for v in data]
                refs = [torch.tensor(v, device='cuda:0').t() for v in data]
                for x, y in zip(args, refs):
                    self.assertEqual(metadata(x), metadata(y))
                with patch.object(trace, '_execute_operation', side_effect=AssertionError('Python node replay')):
                    out = call_without_python(compiled, {composition.__code__}, *args)
                for a, b in zip(out, reference(*refs)):
                    self.assertEqual(metadata(a), metadata(b))
                    np.testing.assert_allclose(a.cpu().tolist(), b.cpu().numpy(), rtol=2e-5, atol=2e-5)

    def test_dynamic_arithmetic_consumers_replan_ranks(self):
        rng = np.random.default_rng(198003)
        for packed in (False, True):
            value = 'x.squeeze()' + ('.contiguous()' if packed else '')
            for expression in (f'-{value}', f'{value}.relu()', f'{value} + {value}',
                               f'(-{value} + {value}.relu()) * 0.5'):
                fn = make_program('def program(x):\n    return ' + expression + '\n')
                # All slices retain rank 2, strides (7, 1) and offset 0.
                # Packing also admits surviving noncontiguous column strides.
                shapes = [(1, 1), (3, 7), (1, 7), (0, 7), (1, 0), (1, 1)]
                if packed:
                    shapes[3:3] = [(3, 1), (0, 1)]
                for sizes in (shapes, shapes[1:] + shapes[:1]):
                    with self.subTest(packed=packed, expression=expression, sizes=sizes):
                        torch._dynamo.reset()
                        compiled, cache = compile_with_cache(fn, dynamic=True)
                        reference = torch.compile(fn, backend='eager', fullgraph=True, dynamic=True)
                        previous = None
                        for rows, cols in sizes:
                            data = rng.normal(size=(3, 7)).astype(np.float32)
                            x = native.tensor(data).to('cuda:0')[:rows][:, :cols]
                            y = torch.tensor(data, device='cuda:0')[:rows][:, :cols]
                            self.assertEqual(metadata(x), metadata(y))
                            expected = reference(y)
                            for _ in range(2):
                                with patch.object(trace, '_execute_operation', side_effect=AssertionError('Python node replay')):
                                    if cache.graphs:
                                        with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')):
                                            actual = call_without_python(compiled, {fn.__code__}, x)
                                    else:
                                        actual = call_without_python(compiled, {fn.__code__}, x)
                                self.assertIsNot(actual, previous)
                                self.assert_pair(actual, expected)
                                previous = actual
                        self.assertEqual(len(cache.graphs), 1)

    def test_dynamic_consumer_declarations_still_prevalidated(self):
        x = native.ones((3, 7)).to('cuda:0')
        for expression in ('-a', 'a.relu()', 'a + a'):
            fn = make_program('def program(x):\n    a = x.squeeze().contiguous()\n    return ' + expression + '\n')
            compiled, cache = compile_with_cache(fn, dynamic=True)
            compiled(x[:1])
            key, graph = next(iter(cache.graphs.items()))
            op = graph.operations[-1]
            runtime_metadata = replace(op.metadata, shape=(3, 7), stride=(7, 1))
            # Even a declaration matching the new runtime result must be
            # rejected if it disagrees with the original captured inputs.
            for bad, error in (
                (replace(graph, operations=(*graph.operations[:-1], replace(op, metadata=runtime_metadata))),
                 trace.CompileTraceUnsupportedError),
                (replace(graph, output_metadata=runtime_metadata), ValueError),
            ):
                cache.graphs[key] = bad
                with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')), \
                     patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early native operation')), \
                     patch.object(trace, '_execute_operation', side_effect=AssertionError('Python node replay')):
                    with self.assertRaises(error) as caught:
                        compiled(x)
                    self.assertIs(type(caught.exception), error)
            cache.graphs[key] = graph
            self.assert_pair(compiled(x), fn(x))

    def test_dynamic_consumers_keep_strided_arithmetic_unsupported(self):
        x = native.ones((3, 7)).to('cuda:0')
        for expression in ('-x.squeeze()', 'x.squeeze().relu()', 'x.squeeze() + x.squeeze()'):
            fn = make_program('def program(x):\n    return ' + expression + '\n')
            compiled, cache = compile_with_cache(fn, dynamic=True)
            compiled(x[:1])
            with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')), \
                 patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early native operation')):
                with self.assertRaises(trace.CompileTraceUnsupportedError) as caught:
                    compiled(x[:, :1])
                self.assertIs(type(caught.exception), trace.CompileTraceUnsupportedError)
            self.assertEqual(len(cache.graphs), 1)

    def test_shared_raw_bits_mutation_and_storage_lifetime(self):
        bits = np.array([0, 0x80000000, 0x7f800001, 0xff800001, 0x7fc12345,
                         0xffc54321, 1, 0x80000001, 0x7f800000, 0xff800000,
                         0x3f800000, 0xbf800000], dtype=np.uint32)
        for fullgraph, dynamic in POLICIES:
            for view in (lambda x: x[:1], lambda x: x[:, 1:2], lambda x: x[1:2][:, 1:],
                         lambda x: x.t(), lambda x: x.select(1, 1),
                         lambda x: x.select(0, 1).select(0, 1)):
                a, b = native.zeros((3, 4), device='cuda:0'), torch.zeros((3, 4), device='cuda:0')
                write_bits(a, bits); write_bits(b, bits)
                x, y = view(a), view(b)
                self.assertEqual(metadata(x), metadata(y))
                compiled, _ = compile_with_cache(squeeze, fullgraph, dynamic)
                torch._dynamo.reset()
                reference = torch.compile(squeeze, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
                out, expected = compiled(x), reference(y)
                self.assert_pair(out, expected)
                write_bits(a, bits[::-1].copy()); write_bits(b, bits[::-1].copy())
                self.assert_pair(out, expected)
                # A scalar selected from even a strided output mutates its base.
                u, v = out, expected
                while u.dim():
                    u, v = u.select(0, 0), v.select(0, 0)
                write_bits(u, np.array([0x80000000], dtype=np.uint32))
                write_bits(v, np.array([0x80000000], dtype=np.uint32))
                self.assert_pair(a, b)
                del a, x, u, compiled
                gc.collect()
                self.assert_pair(out, expected)

    def test_scope_and_method_identity_guards_without_user_hooks(self):
        x = native.ones((3, 7)).to('cuda:0')
        class Hook:
            def __index__(self):
                raise AssertionError('index hook')
        expressions = ('x.squeeze(0)', 'x.squeeze((0,))', 'x.squeeze([0])', 'x.squeeze(0, 1)',
                       'x.squeeze(dim=0)', 'x.squeeze(other=0)', 'x.squeeze(hook)',
                       'm.squeeze(x, dim=0)', 'x.squeeze_()', 'x.unsqueeze(0)', 'x.flatten()',
                       '-x.t().squeeze()', 'x.t().squeeze() + x.t().squeeze()',
                       'x.t().squeeze() * 2', 'x.t().squeeze().relu()',
                       'x.t().squeeze().sum(1)', 'x.t().squeeze() @ x')
        for fullgraph, dynamic in POLICIES:
            for expression in expressions:
                fn = make_program('def program(x):\n    return ' + expression + '\n', hook=Hook())
                with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early native operation')):
                    with self.assertRaises(trace.CompileTraceUnsupportedError) as caught:
                        compile_with_cache(fn, fullgraph, dynamic)[0](x)
                    self.assertIs(type(caught.exception), trace.CompileTraceUnsupportedError)
        for bad in (x.cpu(), native.ones((1, 3, 7)).to('cuda:0')):
            with self.assertRaises(trace.CompileTraceUnsupportedError) as caught:
                compile_with_cache(squeeze)[0](bad)
            self.assertIs(type(caught.exception), trace.CompileTraceUnsupportedError)
        compiled, cache = compile_with_cache(squeeze)
        compiled(x)
        graph = next(iter(cache.graphs.values()))
        def reject(*args):
            raise AssertionError('method hook')
        for warm in (False, True):
            wrapper = compiled if warm else compile_with_cache(squeeze)[0]
            for replacement in (reject, property(reject)):
                with patch.object(native.Tensor, 'squeeze', replacement):
                    with self.assertRaises(trace.CompileTraceUnsupportedError) as caught:
                        wrapper(x)
                    self.assertIs(type(caught.exception), trace.CompileTraceUnsupportedError)
                    self.assert_pair(graph.forward(x), x)

    def test_whole_graph_validation_on_cache_hits(self):
        fn = make_program('def program(x):\n    a = x.squeeze()\n    b = -a.contiguous()\n    return b.squeeze()\n')
        x = native.ones((1, 7)).to('cuda:0')
        for dynamic in (False, True):
            compiled, cache = compile_with_cache(fn, dynamic=dynamic)
            compiled(x)
            key, graph = next(iter(cache.graphs.items()))
            invalid = []
            for index in (0, len(graph.operations) - 1):
                op = graph.operations[index]
                for change in ({'inputs': ()}, {'inputs': ('missing',)}, {'inputs': ('arg0', 'arg0')},
                               {'scalar': 0}, {'shape': ()}, {'axes': (0, 0)},
                               {'reduction': (1, False)}, {'op': 'bad'}, {'target': 'flatten'},
                               {'metadata': None}):
                    ops = list(graph.operations); ops[index] = replace(op, **change)
                    invalid.append(replace(graph, operations=tuple(ops)))
                for change in ({'shape': (True,)}, {'shape': (7.,)}, {'stride': (True,)},
                               {'stride': (1.,)}, {'stride': (2,)}, {'storage_offset': False},
                               {'storage_offset': 1}, {'requires_grad': 0}, {'requires_grad': True},
                               {'device': 'cuda:1'}, {'dtype': trace.CompileTraceDType('torch.float64')}):
                    ops = list(graph.operations); ops[index] = replace(op, metadata=replace(op.metadata, **change))
                    invalid.append(replace(graph, operations=tuple(ops)))
            invalid += [replace(graph, output='missing'), replace(graph, output_metadata=None),
                        replace(graph, output_metadata=replace(graph.output_metadata, shape=(True,)))]
            cpu = x.cpu()
            invalid.append(replace(graph, captures=(trace.CompileTraceCapture('unused', cpu, trace._metadata_from_native_tensor(cpu)),)))
            with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')), \
                 patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early native operation')), \
                 patch.object(trace, '_execute_operation', side_effect=AssertionError('Python node replay')):
                for bad in invalid:
                    cache.graphs[key] = bad
                    with self.assertRaises(trace.CompileTraceUnsupportedError) as caught:
                        compiled(x)
                    self.assertIs(type(caught.exception), trace.CompileTraceUnsupportedError)
            cache.graphs[key] = graph
            self.assertEqual(compiled(x).cpu().tolist(), [-1.] * 7)

    def test_repeated_output_metadata_is_prevalidated(self):
        x = native.ones((1, 7)).to('cuda:0')
        for dynamic in (False, True):
            compiled, cache = compile_with_cache(nested, dynamic=dynamic)
            compiled(x)
            key, graph = next(iter(cache.graphs.items()))
            a, first, second, last = graph.output_metadata.elements
            leaf, pair = second.elements
            bad = replace(second, elements=(replace(leaf, stride=(True,)), pair))
            cache.graphs[key] = replace(graph, output_metadata=replace(graph.output_metadata, elements=(a, first, bad, last)))
            with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early native operation')):
                with self.assertRaises(trace.CompileTraceUnsupportedError) as caught:
                    compiled(x)
                self.assertIs(type(caught.exception), trace.CompileTraceUnsupportedError)

    def test_reference_import_is_not_needed(self):
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('reference import')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as n
from torch_rs import _compile_trace as trace
from unittest.mock import patch
def f(x):
    a = x.squeeze()
    return a, a, x.squeeze()
x = n.tensor([[1., 2., 3.]]).to('cuda:0')
c = n.compile(f, backend='eager', fullgraph=True)
def reject(frame, event, arg):
    if event == 'call' and frame.f_code is f.__code__:
        raise AssertionError('Python body replay')
sys.setprofile(reject)
with patch.object(trace, '_execute_operation', side_effect=AssertionError('Python node replay')):
    for _ in range(2):
        a, b, d = c(x)
        assert a is b and a is not d and a is not x
        assert a.cpu().tolist() == [1., 2., 3.]
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class SqueezeDeviceTests(unittest.TestCase):
    def test_restoration_and_unused_input_capture_device_guards(self):
        driver = ctypes.CDLL('libcuda.so.1')
        driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        def context():
            value = ctypes.c_void_p()
            self.assertEqual(driver.cuCtxGetCurrent(ctypes.byref(value)), 0)
            return value.value
        previous = torch.cuda.current_device()
        fn = make_program('def program(x, y):\n    return x.squeeze().contiguous()\n')
        try:
            for ordinal in (0, 1):
                x = native.ones((3, 1)).to(f'cuda:{ordinal}')
                y = native.ones((3, 1)).to(f'cuda:{1-ordinal}')
                torch.cuda.set_device(1-ordinal)
                torch.empty(1, device=f'cuda:{1-ordinal}')
                before = context()
                compiled, cache = compile_with_cache(fn)
                for _ in range(2):
                    self.assertEqual(compiled(x, x).cpu().tolist(), [1.] * 3)
                    self.assertEqual(context(), before)
                    self.assertEqual(torch.cuda.current_device(), 1-ordinal)
                key, graph = next(iter(cache.graphs.items()))
                with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early operation')):
                    with self.assertRaises(trace.CompileTraceUnsupportedError) as caught:
                        compiled(x, y)
                    self.assertIs(type(caught.exception), trace.CompileTraceUnsupportedError)
                    cache.graphs[key] = replace(graph, captures=(trace.CompileTraceCapture('unused', y, trace._metadata_from_native_tensor(y)),))
                    with self.assertRaises(trace.CompileTraceUnsupportedError) as caught:
                        compiled(x, x)
                    self.assertIs(type(caught.exception), trace.CompileTraceUnsupportedError)
                self.assertEqual(context(), before)
        finally:
            torch.cuda.set_device(previous)
