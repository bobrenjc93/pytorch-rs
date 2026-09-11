"""Non-scoring Tensor.t graphlets; the frozen 38-case corpus is unchanged."""
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


def transpose(x):
    return x.t()


def views(x):
    y = x.t()
    z = x.t()
    return [x, y, (y, z, y.t()), y.contiguous()]


def arithmetic(x, y):
    a = x.t().contiguous()
    b = y.t().contiguous()
    c = -(a @ b) * 0.5
    return c + c, c.sum(1), a + a, a * 1.25


class TMetadataTests(unittest.TestCase):
    def test_bounded_metadata(self):
        for shape, stride in (((), ()), ((7,), (3,)), ((3, 7), (9, 1)),
                              ((0, 2**32 + 1), (2**32 + 1, 1))):
            recorder = trace.CompileTraceRecorder()
            x = recorder.input(shape=shape, stride=stride, device='cuda:0', storage_offset=2)
            y = x.t()
            self.assertIsNot(y, x)
            self.assertEqual(y.metadata.shape, shape[::-1])
            self.assertEqual(y.metadata.stride, stride[::-1])
            self.assertEqual(y.metadata.storage_offset, 2)
        for options in ({'device': 'cpu'}, {'requires_grad': True},
                        {'shape': (1, 2, 3)}, {'dtype': trace.CompileTraceDType('torch.float64')}):
            recorder = trace.CompileTraceRecorder()
            with self.assertRaises(NotImplementedError):
                recorder.input(**({'shape': (3, 7), 'device': 'cuda:0'} | options)).t()
            self.assertEqual(recorder._operations, [])


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class CompileCudaTTests(unittest.TestCase):
    def assert_pair(self, a, b):
        self.assertEqual(metadata(a), metadata(b))
        np.testing.assert_array_equal(read_bits(a.contiguous()), read_bits(b.contiguous()))

    def test_seeded_layouts_identity_and_cache(self):
        rng = np.random.default_rng(19751976)
        shapes = [(3, 7), (17, 31), (257, 263), (1031, 1033)]
        shapes += [tuple(map(int, rng.integers(3, 65, 2))) for _ in range(3)]
        layouts = (lambda x: x, lambda x: x.t(), lambda x: x[1:][:, 1:-1],
                   lambda x: x[1:][:, 1:-1].t(), lambda x: x.select(1, 1)[1:],
                   lambda x: x.select(0, 1).select(0, 2), lambda x: x[:1],
                   lambda x: x[:1][:, :1], lambda x: x[:0], lambda x: x[:, :0])
        for shape in shapes:
            for view in layouts:
                for fullgraph, dynamic in POLICIES:
                    torch._dynamo.reset()
                    compiled, cache = compile_with_cache(views, fullgraph, dynamic)
                    reference = torch.compile(views, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
                    previous = None
                    for warm in (False, True):
                        data = rng.normal(size=shape).astype(np.float32)
                        a, b = native.tensor(data).to('cuda:0'), torch.tensor(data, device='cuda:0')
                        x, y = view(a), view(b)
                        expected = reference(y)
                        with patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                            if warm:
                                with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')):
                                    out = call_without_python(compiled, {views.__code__}, x)
                            else:
                                out = call_without_python(compiled, {views.__code__}, x)
                        self.assertIs(out[0], x)
                        self.assertIs(out[1], out[2][0])
                        for result in (out[1], out[2][1], out[2][2]):
                            self.assertIsNot(result, x)
                            self.assertEqual(result.data_ptr(), x.data_ptr())
                        self.assertIsNot(out[1], out[2][1])
                        self.assertIsNot(out[2][2], out[1])
                        self.assertIsNot(out[1], previous)
                        self.assertEqual(out[3] is out[1], expected[3] is expected[1])
                        self.assert_pair(out[1], expected[1])
                        self.assert_pair(out[2][2], expected[2][2])
                        self.assert_pair(out[3], expected[3])
                        np.testing.assert_array_equal(read_bits(a), data.ravel().view(np.uint32))
                        previous = out[1]
                    self.assertEqual(len(cache.graphs), 1)

    def test_single_t_large_empty_extents(self):
        for shape in ((), (0,), (1,), (0, 2**32), (2**32 + 1, 0), (0, 2**61)):
            for fullgraph, dynamic in POLICIES:
                torch._dynamo.reset()
                compiled, cache = compile_with_cache(transpose, fullgraph, dynamic)
                reference = torch.compile(transpose, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
                a, b = native.zeros(shape).to('cuda:0'), torch.zeros(shape, device='cuda:0')
                for _ in range(2):
                    out, expected = compiled(a), reference(b)
                    self.assertIsNot(out, a)
                    self.assertIsNot(expected, b)
                    self.assert_pair(out, expected)
                    self.assertEqual(out.data_ptr(), a.data_ptr())
                self.assertEqual(len(cache.graphs), 1)

    def test_packed_arithmetic_and_transposed_input_becomes_contiguous(self):
        rng = np.random.default_rng(19751977)
        for dynamic in (False, True):
            for pretranspose in (False, True):
                torch._dynamo.reset()
                compiled, cache = compile_with_cache(arithmetic, dynamic=dynamic)
                reference = torch.compile(arithmetic, backend='eager', fullgraph=True, dynamic=dynamic)
                for _ in range(2):
                    data = [rng.normal(size=s).astype(np.float32) for s in ((7, 5), (3, 7))]
                    if pretranspose:
                        data = [v.T.copy() for v in data]
                    args = [native.tensor(v).to('cuda:0') for v in data]
                    refs = [torch.tensor(v, device='cuda:0') for v in data]
                    if pretranspose:
                        args = [a.t() for a in args]; refs = [b.t() for b in refs]
                    actual = call_without_python(compiled, {arithmetic.__code__}, *args)
                    expected = reference(*refs)
                    for a, b in zip(actual, expected):
                        self.assertEqual(metadata(a), metadata(b))
                        np.testing.assert_allclose(np.array(a.cpu().tolist()), b.cpu().numpy(), rtol=2e-5, atol=2e-5)
                self.assertEqual(len(cache.graphs), 1)

    def test_raw_bits_shared_mutation_and_lifetimes(self):
        bits = np.array([0, 0x80000000, 0x7f800001, 0xff800001, 0x7fc12345,
                         0xffc54321, 1, 0x80000001, 0x7f800000, 0xff800000,
                         0x3f800000, 0xbf800000], dtype=np.uint32)
        for view in (lambda x: x, lambda x: x[:, 1:3], lambda x: x.select(1, 1),
                     lambda x: x.select(0, 1).select(0, 1)):
            a, b = native.zeros((3, 4), device='cuda:0'), torch.zeros((3, 4), device='cuda:0')
            write_bits(a, bits); write_bits(b, bits)
            x, y = view(a), view(b)
            compiled, _ = compile_with_cache(views)
            torch._dynamo.reset()
            reference = torch.compile(views, backend='eager', fullgraph=True)
            out, expected = compiled(x), reference(y)
            self.assert_pair(out[1], expected[1])
            write_bits(a, bits[::-1].copy()); write_bits(b, bits[::-1].copy())
            self.assert_pair(out[1], expected[1])
            self.assert_pair(out[3], expected[3])
            # Mutate through a contiguous alias produced by t().t().
            restored = out[2][2]
            if restored.is_contiguous():
                changed = np.full(restored.numel(), 0x80000000, dtype=np.uint32)
                write_bits(restored, changed); write_bits(expected[2][2], changed)
                self.assert_pair(a, b)
            kept = out[1]
            del a, x, out, compiled
            gc.collect()
            self.assert_pair(kept, expected[1])

    def test_dynamic_guards_and_alias_pack_transition(self):
        a = native.tensor(np.arange(35, dtype=np.float32).reshape(5, 7)).to('cuda:0')
        fn = make_program('def program(x):\n    return x.t().contiguous()\n')
        compiled, cache = compile_with_cache(fn, dynamic=True)
        for x in (a[:1], a, a[:0], a[:1]):
            out = compiled(x)
            self.assertEqual(out.cpu().tolist(), x.t().cpu().tolist())
            self.assertIsNot(out, x)
            self.assertEqual(out.data_ptr() == x.data_ptr(), x.t().is_contiguous())
        self.assertEqual(len(cache.graphs), 1)
        compiled, cache = compile_with_cache(transpose, dynamic=False)
        compiled(a)
        graph = next(iter(cache.graphs.values()))
        for wrong in (a.t(), a[1:], a[:, 1:], a.cpu(), native.ones((2, 3, 4)).to('cuda:0')):
            with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early native operation')):
                with self.assertRaises((ValueError, NotImplementedError)):
                    graph.forward(wrong)

    def test_invalid_cached_metadata_types_early_late_and_outputs(self):
        x = native.ones((1, 1)).to('cuda:0')
        fn = make_program('def program(x):\n    a = x.t()\n    b = -a.contiguous()\n    return b.t()\n')
        for dynamic in (False, True):
            compiled, cache = compile_with_cache(fn, dynamic=dynamic)
            compiled(x)
            key, graph = next(iter(cache.graphs.items()))
            bad = []
            changes = ({'shape': (True, 1)}, {'shape': (1., 1)}, {'stride': (1, True)},
                       {'stride': (1., 1)}, {'storage_offset': False}, {'storage_offset': 0.},
                       {'requires_grad': 0}, {'requires_grad': 0.})
            device = trace.CompileTraceDevice('cuda', 0)
            object.__setattr__(device, 'index', False)
            changes += ({'device': device},)
            for change in changes:
                for index, op in enumerate(graph.operations):
                    ops = list(graph.operations)
                    ops[index] = replace(op, metadata=replace(op.metadata, **change))
                    bad.append(replace(graph, operations=tuple(ops)))
                bad.append(replace(graph, output_metadata=replace(graph.output_metadata, **change)))
                item = graph.inputs[0]
                bad.append(replace(graph, inputs=(replace(item, metadata=replace(item.metadata, **change)),)))
            for index in (0, len(graph.operations)-1):
                for change in ({'inputs': ()}, {'inputs': ('missing',)}, {'scalar': 1},
                               {'inputs': ('arg0', 'arg0')}, {'reduction': (1, False)}, {'metadata': None},
                               {'op': 'invalid'}, {'target': 'transpose'}):
                    ops = list(graph.operations); ops[index] = replace(ops[index], **change)
                    bad.append(replace(graph, operations=tuple(ops)))
            for change in ({'device': 'cuda:1'}, {'requires_grad': True}, {'storage_offset': 1},
                           {'dtype': trace.CompileTraceDType('torch.float64')},
                           {'shape': (2, 1)}, {'stride': (2, 1)}):
                last = graph.operations[-1]
                bad.append(replace(graph, operations=(*graph.operations[:-1], replace(last, metadata=replace(last.metadata, **change)))))
                bad.append(replace(graph, output_metadata=replace(graph.output_metadata, **change)))
            cpu = native.ones((1, 1))
            bad.append(replace(graph, captures=(trace.CompileTraceCapture('unused', cpu, trace._metadata_from_native_tensor(cpu)),)))
            with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')), \
                 patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early graph operation')), \
                 patch.object(trace, '_execute_operation', side_effect=AssertionError('early single operation')):
                for invalid in bad:
                    cache.graphs[key] = invalid
                    with self.assertRaises((ValueError, NotImplementedError)):
                        compiled(x)
            cache.graphs[key] = graph
            self.assertEqual(compiled(x).cpu().tolist(), [[-1.]])

    def test_repeated_containers_validate_each_metadata_pair_on_cache_hits(self):
        def replace_leaf(spec, change):
            if isinstance(spec, trace.CompileTraceTensorMetadata):
                return replace(spec, **change)
            return replace(spec, elements=tuple(replace_leaf(child, change) for child in spec.elements))

        def leaf(value):
            while isinstance(value, (tuple, list)):
                value = value[0]
            return value

        x = native.ones((1, 1)).to('cuda:0')
        for expression in ('[x.t()]', '(x.t(),)', '([x.t()],)', '[(x.t(),)]'):
            fn = make_program(f'def program(x):\n    y = {expression}\n    return y, y\n')
            for dynamic in (False, True):
                with self.subTest(expression=expression, dynamic=dynamic):
                    torch._dynamo.reset()
                    compiled, cache = compile_with_cache(fn, dynamic=dynamic)
                    reference = torch.compile(fn, backend='eager', fullgraph=True, dynamic=dynamic)
                    actual, expected = compiled(x), reference(torch.ones((1, 1), device='cuda:0'))
                    self.assertIs(actual[0], actual[1])
                    self.assertIs(expected[0], expected[1])
                    self.assertIsNot(leaf(actual), x)
                    self.assert_pair(leaf(actual), leaf(expected))
                    key, graph = next(iter(cache.graphs.items()))
                    self.assertIs(graph.output.elements[0], graph.output.elements[1])
                    first, second = graph.output_metadata.elements

                    # An equal, independently constructed metadata tree must
                    # still preserve the returned container and tensor aliases.
                    equivalent = replace_leaf(second, {})
                    self.assertIsNot(equivalent, first)
                    metadata = replace(graph.output_metadata, elements=(first, equivalent))
                    cache.graphs[key] = replace(graph, output_metadata=metadata)
                    with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')):
                        actual = compiled(x)
                    self.assertIs(actual[0], actual[1])
                    self.assertIs(leaf(actual[0]), leaf(actual[1]))

                    for change in ({'shape': (True, 1)}, {'shape': (1., 1)},
                                   {'stride': (True, 1)}, {'storage_offset': 0.},
                                   {'requires_grad': 0}, {'device': 'cuda:1'}):
                        invalid = replace_leaf(second, change)
                        metadata = replace(graph.output_metadata, elements=(first, invalid))
                        cache.graphs[key] = replace(graph, output_metadata=metadata)
                        with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')), \
                             patch.object(trace._native, '_compile_trace_cuda_graph', wraps=trace._native._compile_trace_cuda_graph) as native_calls, \
                             patch.object(trace, '_execute_operation', side_effect=AssertionError('Python operation')):
                            with self.assertRaises((ValueError, NotImplementedError)):
                                compiled(x)
                            native_calls.assert_not_called()
                    cache.graphs[key] = graph
                    actual = compiled(x)
                    self.assertIs(actual[0], actual[1])
                    self.assertEqual(leaf(actual).cpu().tolist(), [[1.]])

    def test_strided_arithmetic_and_public_scope_remain_rejected(self):
        x = native.ones((3, 7)).to('cuda:0')
        expressions = ('-a', 'a * 2', 'a + a', 'a @ a', 'a.sum(1)',
                       'x.t(0)', 'x.t(dim=0)', 'x.t(foo=True)', 'x.t().reshape(-1)',
                       'x.swapdims(0, 1)', 'x.permute(1, 0)', 'x.T', 'x.mT', 'm.t(x)')
        for dynamic in (False, True):
            for expression in expressions:
                fn = make_program(f'def program(x):\n    a = x.t()\n    b = a.contiguous()\n    return {expression}\n')
                with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early operation')):
                    with self.assertRaises(NotImplementedError):
                        compile_with_cache(fn, dynamic=dynamic)[0](x)
        for x in (x.cpu(), native.ones((1, 3, 7)).to('cuda:0')):
            with self.assertRaises(NotImplementedError):
                compile_with_cache(transpose)[0](x)

    def test_callable_global_cache_and_no_redispatch(self):
        x = native.ones((3, 7)).to('cuda:0')
        compiled, cache = compile_with_cache(transpose)
        compiled(x)
        for cold in (False, True):
            wrapper = compile_with_cache(transpose)[0] if cold else compiled
            def reject(*args):
                raise AssertionError('Tensor method redispatch')
            for replacement in (reject, property(reject)):
                with patch.object(native.Tensor, 't', replacement):
                    with self.assertRaises(NotImplementedError):
                        wrapper(x)
        graph = next(iter(cache.graphs.values()))
        with patch.object(native.Tensor, 't', side_effect=AssertionError('Tensor method redispatch')), \
             patch.object(native.Tensor, 'contiguous', side_effect=AssertionError('packing redispatch')):
            self.assertEqual(graph.forward(x).cpu().tolist(), [[1.] * 3] * 7)
        fn = make_program('def helper(x):\n    return x.t().contiguous()\ndef program(x):\n    return helper(x) + bias.t().contiguous()\n', bias=x)
        compiled, cache = compile_with_cache(fn)
        self.assertEqual(call_without_python(compiled, {fn.__code__, fn.__globals__['helper'].__code__}, x).cpu().tolist(), [[2.] * 3] * 7)
        fn.__globals__['bias'] = native.full((3, 7), 3.).to('cuda:0')
        self.assertEqual(compiled(x).cpu().tolist(), [[4.] * 3] * 7)
        fn.__globals__['bias'] = x.cpu()
        with self.assertRaises(NotImplementedError):
            compiled(x)
        limited, _ = compile_with_cache(transpose, limit=1)
        limited(x)
        with self.assertRaises(NotImplementedError):
            limited(x.t())

    def test_reference_import_blocked(self):
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('reference import')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as n
def f(x):
    y = x.t()
    return y, y, y.contiguous(), y.t()
def reject(frame, event, arg):
    if event == 'call' and frame.f_code is f.__code__:
        raise AssertionError('Python replay')
x = n.tensor([[1., 2., 3.], [4., 5., 6.]]).to('cuda:0')
compiled = n.compile(f, backend='eager', fullgraph=True)
sys.setprofile(reject)
for _ in range(3):
    out = compiled(x)
    assert out[0] is out[1] and out[0] is not x and out[3] is not x
    assert out[0].data_ptr() == out[3].data_ptr() == x.data_ptr()
    assert out[2].cpu().tolist() == [[1., 4.], [2., 5.], [3., 6.]]
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class CompileTDeviceTests(unittest.TestCase):
    def test_device_context_and_unused_capture(self):
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
                compiled, cache = compile_with_cache(views)
                for _ in range(2):
                    out = compiled(x)
                    self.assertEqual(str(out[1].device), f'cuda:{ordinal}')
                    self.assertEqual(out[1].cpu().tolist(), [[1.] * 3] * 7)
                    self.assertEqual(torch.cuda.current_device(), 1 - ordinal)
                    self.assertEqual(context(), before)
                key, graph = next(iter(cache.graphs.items()))
                other = native.ones((3, 7)).to(f'cuda:{1-ordinal}')
                cache.graphs[key] = replace(graph, captures=(trace.CompileTraceCapture('unused', other, trace._metadata_from_native_tensor(other)),))
                with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early operation')):
                    with self.assertRaises(NotImplementedError):
                        compiled(x)
                self.assertEqual(context(), before)
        finally:
            torch.cuda.set_device(previous)
