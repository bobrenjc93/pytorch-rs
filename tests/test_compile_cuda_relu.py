"""Bounded Tensor.relu capture, independent of all scoring corpora."""
from concurrent.futures import ThreadPoolExecutor
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
from tests.test_compile_cuda_mul_scalar import call_without_python, make_program
from tests.test_cuda_add import Comparison, available, runtime, torch, upload
from tests.test_cuda_relu import EDGE_BITS, copy_bits, download_bits, upload_bits

CAPTURE = None


def relu(x):
    return x.relu()


def helper(x):
    return x.relu()


def captured(x):
    return helper(x) + CAPTURE.relu()


def nested(x):
    a = x.relu()
    b = (-a * -0.5).relu()
    out = [a, (b, x, a)]
    return out, out


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class CompileCudaReluTests(Comparison, unittest.TestCase):
    def test_seeded_shapes_composition_and_real_cache_values(self):
        rng = np.random.default_rng(19782026)
        cases = [(relu, [s]) for s in [(), (0,), (2, 0, 3), (0, 2**40), (1,),
                 (257,), (65539,), (1048589,), (3, 7), (2, 1, 3, 4, 2)]]
        programs = [
            ('(-x).relu() + (x * 0.5).relu()', [(3, 7)]),
            ('(x * 0.5).relu().sum(1)', [(7, 1031)]),
            ('(x * 0.5).relu().sum(1)', [(0, 7)]),
            ('(x * 0.5).relu().sum(1)', [(3, 0)]),
            ('(x * 0.5).relu().sum(1)', [(1, 1)]),
            ('(x @ y).relu()', [(3, 7), (7, 5)]),
            ('(x @ y).relu()', [(3, 0), (0, 5)]),
            ('(x @ y).relu()', [(0, 7), (7, 5)]),
            ('((x @ y).relu() + (x @ y)).relu().sum(-1, keepdim=True)', [(3, 7), (7, 5)]),
            ('(x + y).relu()', [(7, 5), (5,)]),
            ('x.transpose(0, 1).contiguous().relu().reshape(-1)', [(3, 7)]),
            ('x.t().reshape(-1).relu()', [(3, 7)]),
            ('x.reshape(7, 3).relu().t().contiguous()', [(3, 7)]),
        ]
        for expression, shapes in programs:
            parameters = 'x,y' if len(shapes) == 2 else 'x'
            cases.append((make_program(f'def program({parameters}):\n    return {expression}\n'), shapes))
        for program, shapes in cases:
            for fullgraph in (True, False):
                compiled, cache = compile_with_cache(program, fullgraph)
                torch._dynamo.reset()
                reference = torch.compile(program, backend='eager', fullgraph=fullgraph)
                for _ in range(2):
                    # Binary fractions keep unfused matmul sums exactly representable.
                    values = [rng.integers(-16, 17, size=int(np.prod(s))).astype(np.float32) / 8 for s in shapes]
                    args = [upload(native, a, s) for a, s in zip(values, shapes)]
                    refs = [upload(torch, a, s) for a, s in zip(values, shapes)]
                    expected = reference(*refs)
                    torch.testing.assert_close(expected, program(*refs), rtol=0, atol=0)
                    with patch.object(trace, '_execute_operation', side_effect=AssertionError('per-node Python')):
                        actual = call_without_python(compiled, {program.__code__}, *args)
                    self.compare(actual, expected)
                    self.compare(program(*args), expected)
                    for x, tx in zip(args, refs):
                        self.compare(x, tx)
                    self.assertEqual(actual.storage_offset(), 0)
                self.assertEqual(len(cache.graphs), 1)

    def test_ieee_bits_offsets_strides_and_dynamic_guards(self):
        bits = np.tile(EDGE_BITS, 30)
        base, reference = [upload_bits(m, bits) for m in (native, torch)]
        views = (lambda x: x, lambda x: x[1:258], lambda x: x[2:259],
                 lambda x: x.select(0, 1), lambda x: x[660:].reshape(2, 0, 3),
                 lambda x: x.reshape(1, 30, 22).transpose(0, 1))
        compiled, cache = compile_with_cache(relu)
        ref = torch.compile(relu, backend='eager', fullgraph=True)
        for view in views:
            x, tx = view(base), view(reference)
            actual, expected = compiled(x), ref(tx)
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), 0)
            np.testing.assert_array_equal(download_bits(actual), download_bits(expected))
            np.testing.assert_array_equal(download_bits(actual), download_bits(x.relu()))
        self.assertEqual(len(cache.graphs), len(views))
        graph = list(cache.graphs.values())[1]
        for wrong in (base[2:259], base, base.cpu(), base[:256].reshape(16, 16).t()):
            with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                with self.assertRaises((ValueError, NotImplementedError)):
                    graph.forward(wrong)
        singleton = list(cache.graphs.values())[-1]
        with self.assertRaisesRegex(ValueError, 'stride'):
            singleton.forward(base.reshape(30, 1, 22))
        dynamic, cache = compile_with_cache(relu, dynamic=True)
        dynamic(base[1:258])
        before = dict(cache.graphs)
        self.compare(dynamic(base[1:4]), reference[1:4].relu())
        self.assertEqual(cache.graphs, before)
        dynamic(base[2:])
        self.assertEqual(len(cache.graphs), 2)

    def test_cpu_cuda_separation_capture_helpers_and_nested_lifetimes(self):
        cpu = native.tensor([-2., 3.])
        cuda = cpu.to('cuda:0')
        for order in ((cpu, cuda), (cuda, cpu)):
            compiled, cache = compile_with_cache(relu)
            for x in order:
                self.assertEqual(compiled(x).cpu().tolist(), [0., 3.])
            with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')):
                self.assertEqual(compiled(native.tensor([4., -5.]).to('cuda:0')).cpu().tolist(), [4., 0.])
            self.assertEqual(len(cache.graphs), 2)
        compiled, cache = compile_with_cache(captured)
        for value in (cuda, native.tensor([7., -1.]).to('cuda:0')):
            with patch(__name__ + '.CAPTURE', value):
                with patch.object(trace, '_execute_operation', side_effect=AssertionError('Python execution')):
                    out = call_without_python(compiled, {captured.__code__, helper.__code__}, cuda)
                self.assertEqual(out.cpu().tolist(), [max(0., v) + r for v, r in zip(value.cpu().tolist(), [0., 3.])])
                before = dict(cache.graphs)
                with self.assertRaisesRegex(NotImplementedError, 'matching devices'):
                    compiled(cpu)
                self.assertEqual(cache.graphs, before)
        self.assertEqual(len(cache.graphs), 2)
        with patch(__name__ + '.CAPTURE', cuda), patch(__name__ + '.helper', lambda x: x.abs()), \
             patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
            before = dict(cache.graphs)
            with self.assertRaises(NotImplementedError):
                compiled(cuda)
            self.assertEqual(cache.graphs, before)
        compiled, _ = compile_with_cache(nested)
        out = compiled(cuda)
        self.assertIs(out[0], out[1])
        self.assertIs(out[0][0], out[0][1][2])
        self.assertIs(out[0][1][1], cuda)
        self.assertNotEqual(out[0][0].data_ptr(), cuda.data_ptr())
        del cuda
        gc.collect()
        self.assertEqual(out[0][0].cpu().tolist(), [0., 3.])
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda v: compiled(native.full((257,), v).to('cuda:0')), range(-3, 6)))
        for v, out in zip(range(-3, 6), results):
            self.assertEqual(out[0][1][0].cpu().tolist(), [max(0., v) * .5] * 257)

    def test_compiled_stream_completion_and_input_mutation(self):
        values = np.arange(65539, dtype=np.float32) / 16 - 7
        x, tx = [upload(m, values, values.shape) for m in (native, torch)]
        compiled, _ = compile_with_cache(relu)
        expected = torch.compile(relu, backend='eager', fullgraph=True)(tx)
        lib, stream = runtime(), torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            out = call_without_python(compiled, {relu.__code__}, x)
            self.assertEqual(lib.cudaStreamQuery(ctypes.c_void_p(1)), 0)
            copied = torch.empty_like(tx)
            self.assertEqual(lib.cudaMemcpyAsync(copied.data_ptr(), out.data_ptr(), values.nbytes,
                                                3, stream.cuda_stream), 0)
        stream.synchronize()
        torch.testing.assert_close(copied, expected, rtol=0, atol=0)
        zeros = np.zeros_like(values)
        copy_bits(x.data_ptr(), zeros.ctypes.data, zeros.nbytes, 1)
        self.compare(out, expected)
        del x
        gc.collect()
        self.compare(out, expected)

    def test_malformed_early_late_and_repeated_output_metadata_zero_execution(self):
        x = native.ones((1, 1)).to('cuda:0')
        for dynamic in (False, True):
            compiled, cache = compile_with_cache(nested, dynamic=dynamic)
            compiled(x)
            key, graph = next(iter(cache.graphs.items()))
            bad = [replace(graph, dynamic=1), replace(graph, output='missing'), replace(graph, output_metadata=None)]
            for i in (0, len(graph.operations)-1):
                op = graph.operations[i]
                for change in ({'inputs': ()}, {'inputs': ('arg0', 'arg0')}, {'inputs': ['arg0']},
                               {'inputs': ('missing',)}, {'scalar': 1}, {'shape': (1, 1)},
                               {'axes': (0, 1)}, {'reduction': (1, False)}, {'target': True},
                               {'target': 'abs'}, {'op': 'call_reduction'}, {'name': 0}):
                    ops = list(graph.operations); ops[i] = replace(op, **change)
                    bad.append(replace(graph, operations=tuple(ops)))
                for change in ({'shape': (True, 1)}, {'shape': (1., 1)}, {'shape': (2, 1)},
                               {'stride': (1, False)}, {'stride': (2, 1)}, {'storage_offset': 0.},
                               {'storage_offset': 1}, {'requires_grad': 0}, {'requires_grad': True},
                               {'device': 'cuda:1'}, {'dtype': trace.CompileTraceDType('torch.float64')}):
                    ops = list(graph.operations); ops[i] = replace(op, metadata=replace(op.metadata, **change))
                    bad.append(replace(graph, operations=tuple(ops)))
                    first, second = graph.output_metadata.elements
                    leaf = replace(second.elements[0], **change)
                    bad.append(replace(graph, output_metadata=replace(graph.output_metadata,
                        elements=(first, replace(second, elements=(leaf, second.elements[1]))))))
            cpu = x.cpu()
            for capture_value in (cpu, x):
                meta = trace._metadata_from_native_tensor(capture_value)
                if capture_value is x:
                    meta = replace(meta, device=trace.CompileTraceDevice('cuda', 1))
                bad.append(replace(graph, captures=(trace.CompileTraceCapture('unused', capture_value, meta),)))
            with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')), \
                 patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early native execution')):
                for invalid in bad:
                    cache.graphs[key] = invalid
                    with self.subTest(invalid=invalid), self.assertRaises((NotImplementedError, ValueError, TypeError, RuntimeError)):
                        compiled(x)
            cache.graphs[key] = graph
            self.assertEqual(compiled(x)[0][0].cpu().tolist(), [[1.]])

    def test_single_node_output_prevalidation_and_native_payload_types(self):
        x = native.ones((3, 7)).to('cuda:0')
        compiled, cache = compile_with_cache(relu)
        compiled(x)
        graph = next(iter(cache.graphs.values()))
        bad = [replace(graph, output_metadata=None), replace(graph, output='missing'),
               replace(graph, output_metadata=replace(graph.output_metadata, shape=(7, 3)))]
        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early native execution')):
            for invalid in bad:
                with self.assertRaises((ValueError, NotImplementedError, TypeError)):
                    invalid.forward(x)
        bridge = trace._native._compile_trace_cuda_graph
        valid = ('relu', (0,), None, (3, 7), (7, 1))
        bad = [('relu', indices, payload, shape, strides)
               for indices, payload, shape, strides in (
                   ((), None, (3, 7), (7, 1)), ((0, 0), None, (3, 7), (7, 1)),
                   ((True,), None, (3, 7), (7, 1)), ({0}, None, (3, 7), (7, 1)),
                   ((0.,), None, (3, 7), (7, 1)), ((99,), None, (3, 7), (7, 1)),
                   ((0,), False, (3, 7), (7, 1)), ((0,), 1., (3, 7), (7, 1)),
                   ((0,), (), (3, 7), (7, 1)), ((0,), None, (3., 7), (7, 1)),
                   ((0,), None, (3, 7), (True, 3)), ((0,), None, (7, 3), (3, 1)),
               )]
        for invalid in bad:
            for nodes in ([invalid, valid], [valid, invalid]):
                with self.assertRaises((ValueError, NotImplementedError, TypeError, OverflowError, RuntimeError)):
                    bridge((x,), nodes)
        self.assertEqual(bridge((x,), [valid])[0].cpu().tolist(), [[1.] * 7] * 3)

    def test_method_identity_strict_binding_and_native_compatibility(self):
        x = native.ones((3, 7)).to('cuda:0')
        compiled, cache = compile_with_cache(relu)
        compiled(x)
        def reject(*args):
            raise AssertionError('method/descriptor replay')
        for replacement in (reject, property(reject)):
            with patch.object(native.Tensor, 'relu', replacement), \
                 patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early native execution')):
                for wrapper in (compiled, compile_with_cache(relu)[0]):
                    with self.assertRaises(NotImplementedError):
                        wrapper(x)
        for expression in ('x.relu(1)', 'x.relu(inplace=False)', 'x.relu(input=x)',
                           'm.relu(x, out=None)', 'x.relu_()', 'x.relu().abs()'):
            program = make_program(f'def program(x):\n    return {expression}\n')
            wrapper, rejected_cache = compile_with_cache(program)
            with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                with self.assertRaises((TypeError, NotImplementedError)):
                    wrapper(x)
            self.assertEqual(rejected_cache.graphs, {})
        self.compare(trace._native._compile_trace_unary(x, 'relu'), torch.ones((3, 7), device='cuda:0'))
        with self.assertRaises(NotImplementedError):
            trace._native._compile_trace_unary(x.t(), 'relu')
        metadata = trace._metadata_from_native_tensor(x)
        for change in ({'requires_grad': True}, {'stride': (1, 3)},
                       {'dtype': trace.CompileTraceDType('torch.float64')}):
            with self.assertRaises(NotImplementedError):
                trace._unary_output_metadata(replace(metadata, **change), 'relu')
        fresh, fresh_cache = compile_with_cache(relu)
        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=RuntimeError('launch failed')):
            with self.assertRaisesRegex(RuntimeError, 'launch failed'):
                fresh(x)
        self.assertEqual(fresh_cache.graphs, {})
        self.assertEqual(fresh(x).cpu().tolist(), [[1.] * 7] * 3)

    def test_block_reference_import_and_python_execution_in_fresh_process(self):
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('reference import')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
from torch_rs import _compile_trace as t
def program(x):
    return (x * .5).relu().sum(1)
def reject(*args):
    raise AssertionError('per-node Python execution')
t._execute_operation = reject
f = m.compile(program, backend='eager', fullgraph=True)
def profile(frame,event,arg):
    if event == 'call' and frame.f_code is program.__code__:
        raise AssertionError('original body')
sys.setprofile(profile)
for v in (-2., 3.):
    assert f(m.full((3,7),v).to('cuda:0')).cpu().tolist() == [max(0.,v)*3.5]*3
sys.setprofile(None)
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class CompileCudaReluDeviceTests(Comparison, unittest.TestCase):
    def test_context_captures_mixed_devices_and_restoration(self):
        compiled, cache = compile_with_cache(captured)
        reference = torch.compile(captured, backend='eager', fullgraph=True)
        previous, lib = torch.cuda.current_device(), runtime()
        try:
            for shape in ((), (0,), (3, 7)):
                inputs = [native.full(shape, 1.25).to(f'cuda:{i}') for i in (0, 1)]
                refs = [torch.full(shape, 1.25, device=f'cuda:{i}') for i in (0, 1)]
                for current, target in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    torch.cuda.set_device(current)
                    with patch(__name__ + '.CAPTURE', inputs[target]):
                        result = compiled(inputs[target])
                        before = dict(cache.graphs)
                        with self.assertRaises(NotImplementedError):
                            compiled(inputs[1-target])
                        self.assertEqual(cache.graphs, before)
                    with patch(__name__ + '.CAPTURE', refs[target]):
                        self.compare(result, reference(refs[target]))
                    del result
                    gc.collect()
                    ordinal = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
        finally:
            torch.cuda.set_device(previous)
