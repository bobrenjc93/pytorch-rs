"""Non-scoring bounded Tensor.transpose capture and composition differentials."""
from dataclasses import replace
import gc
import itertools
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


def views(dim0, dim1):
    # Freeze literal axes explicitly; do not infer support for shape expressions.
    return make_program(f'''def program(x):
    a = x.transpose({dim0}, {dim1})
    b = x.transpose(dim1={dim1}, dim0={dim0})
    shared = [a, b, a.transpose({dim0}, dim1={dim1}), a.t().t(), a.contiguous()]
    return x, shared, shared
''')


class TransposeMetadataTests(unittest.TestCase):
    def test_proxy_and_exact_axis_types(self):
        class Integer(int):
            pass
        class Index:
            def __index__(self):
                raise AssertionError('must not invoke user conversion')
        for rank in range(3):
            recorder = trace.CompileTraceRecorder()
            x = recorder.input(shape=(3, 7)[:rank], device='cuda:0')
            for a, b in itertools.product(range(-max(rank, 1), max(rank, 1)), repeat=2):
                y = x.transpose(dim1=b, dim0=a)
                self.assertIsNot(y, x)
                expected = tuple(reversed(x.metadata.shape)) if rank == 2 and a % 2 != b % 2 else x.metadata.shape
                self.assertEqual(y.metadata.shape, expected)
            for axis, error in ((True, TypeError), (0., TypeError), (None, TypeError),
                                (Integer(0), NotImplementedError), (Index(), NotImplementedError),
                                (np.int64(0), NotImplementedError), (2**63, ValueError),
                                (-2**63-1, ValueError), (max(rank, 1), IndexError)):
                before = len(recorder._operations)
                with self.assertRaises(error):
                    x.transpose(axis, 0)
                self.assertEqual(len(recorder._operations), before)
        for options in ({'device': 'cpu'}, {'requires_grad': True}, {'shape': (1, 2, 3)}):
            recorder = trace.CompileTraceRecorder()
            x = recorder.input(**({'shape': (3, 7), 'device': 'cuda:0'} | options))
            with self.assertRaises(NotImplementedError):
                x.transpose(0, 1)
            self.assertEqual(recorder._operations, [])


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class CompileCudaTransposeTests(unittest.TestCase):
    def assert_pair(self, actual, expected):
        self.assertEqual(metadata(actual), metadata(expected))
        np.testing.assert_array_equal(read_bits(actual.contiguous()), read_bits(expected.contiguous()))

    def test_every_axis_layout_policy_identity_and_cache(self):
        rng = np.random.default_rng(19761977)
        layouts = (lambda x: x, lambda x: x.t(), lambda x: x[1:][:, 1:-1],
                   lambda x: x[1:][:, 1:-1].t(), lambda x: x.select(1, 1)[1:],
                   lambda x: x.select(0, 1).select(0, 2), lambda x: x[:1],
                   lambda x: x[:1][:, :1], lambda x: x[:0], lambda x: x[:, :0])
        for shape in ((3, 7), (17, 31), tuple(map(int, rng.integers(33, 64, 2)))):
            data = rng.normal(size=shape).astype(np.float32)
            for layout in layouts:
                x = layout(native.tensor(data).to('cuda:0'))
                y = layout(torch.tensor(data, device='cuda:0'))
                rank = max(len(x.shape), 1)
                for axes in itertools.product(range(-rank, rank), repeat=2):
                    for fullgraph, dynamic in POLICIES:
                        with self.subTest(shape=shape, axes=axes, dynamic=dynamic):
                            torch._dynamo.reset()
                            fn = views(*axes)
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
                                a, b, double, mixed, packed = out[1]
                                self.assertIsNot(a, b)
                                for view in (a, b, double, mixed):
                                    self.assertIsNot(view, x)
                                    self.assertEqual(view.data_ptr(), x.data_ptr())
                                self.assertIsNot(double, a)
                                self.assertEqual(packed is a, expected[1][4] is expected[1][0])
                                for actual, ref in zip(out[1], expected[1]):
                                    self.assert_pair(actual, ref)
                            self.assertEqual(len(cache.graphs), 1)

    def test_large_empty_and_large_nonsquare(self):
        for shape in ((), (0,), (1,), (0, 2**32), (2**32+1, 0), (0, 2**61), (1031, 1033)):
            x = native.zeros(shape).to('cuda:0')
            y = torch.zeros(shape, device='cuda:0')
            rank = max(len(shape), 1)
            for a, b in itertools.product(range(-rank, rank), repeat=2):
                fn = make_program(f'def program(x):\n    return x.transpose({a}, {b})\n')
                for dynamic in (False, True):
                    torch._dynamo.reset()
                    compiled, _ = compile_with_cache(fn, dynamic=dynamic)
                    reference = torch.compile(fn, backend='eager', fullgraph=True, dynamic=dynamic)
                    for _ in range(2):
                        actual, expected = compiled(x), reference(y)
                        self.assertIsNot(actual, x)
                        self.assertEqual(actual.data_ptr(), x.data_ptr())
                        self.assert_pair(actual, expected)

    def test_bits_mutation_and_lifetime(self):
        bits = np.array([0, 0x80000000, 0x7f800001, 0xff800001, 0x7fc12345,
                         0xffc54321, 1, 0x80000001, 0x7f800000, 0xff800000,
                         0x3f800000, 0xbf800000], dtype=np.uint32)
        for layout in (lambda x: x, lambda x: x[:, 1:3], lambda x: x.select(1, 1),
                       lambda x: x.select(0, 1).select(0, 1)):
            x, y = native.zeros((3, 4), device='cuda:0'), torch.zeros((3, 4), device='cuda:0')
            write_bits(x, bits); write_bits(y, bits)
            a, b = layout(x), layout(y)
            rank = max(len(a.shape), 1)
            fn = views(0, rank-1)
            compiled, _ = compile_with_cache(fn)
            torch._dynamo.reset()
            reference = torch.compile(fn, backend='eager', fullgraph=True)
            out, expected = compiled(a), reference(b)
            for v, ref in zip(out[1], expected[1]):
                self.assert_pair(v, ref)
            write_bits(x, bits[::-1].copy()); write_bits(y, bits[::-1].copy())
            for v, ref in zip(out[1], expected[1]):
                self.assert_pair(v, ref)
            restored = out[1][2]
            if restored.is_contiguous():
                write_bits(restored, np.zeros(restored.numel(), dtype=np.uint32))
                write_bits(expected[1][2], np.zeros(restored.numel(), dtype=np.uint32))
                self.assert_pair(x, y)
            kept = out[1][0]
            del x, a, out, restored, compiled
            gc.collect()
            self.assert_pair(kept, expected[1][0])

    def test_composed_arithmetic_and_dynamic_pack_transition(self):
        fn = make_program('''def program(x, y):
    a = x.transpose(dim0=-1, dim1=-2).contiguous()
    b = y.t().transpose(0, 0).contiguous()
    c = -(a @ b) * 0.5
    return c + c, c.sum(1), a + a, a * 1.25
''')
        rng = np.random.default_rng(19761978)
        for dynamic in (False, True):
            for pretranspose in (False, True):
                torch._dynamo.reset()
                compiled, cache = compile_with_cache(fn, dynamic=dynamic)
                reference = torch.compile(fn, backend='eager', fullgraph=True, dynamic=dynamic)
                for _ in range(2):
                    data = [rng.normal(size=s).astype(np.float32) for s in ((7, 5), (3, 7))]
                    if pretranspose:
                        data = [v.T.copy() for v in data]
                    args = [native.tensor(v).to('cuda:0') for v in data]
                    refs = [torch.tensor(v, device='cuda:0') for v in data]
                    if pretranspose:
                        args = [a.t() for a in args]; refs = [b.t() for b in refs]
                    actual = call_without_python(compiled, {fn.__code__}, *args)
                    for a, b in zip(actual, reference(*refs)):
                        self.assertEqual(metadata(a), metadata(b))
                        np.testing.assert_allclose(a.cpu().tolist(), b.cpu().numpy(), rtol=2e-5, atol=2e-5)
                self.assertEqual(len(cache.graphs), 1)
        fn = make_program('def program(x):\n    dim0 = 0\n    return x.transpose(dim0, 1).contiguous()\n')
        x = native.ones((5, 7)).to('cuda:0')
        compiled, cache = compile_with_cache(fn, dynamic=True)
        for a in (x[:1], x, x[:0], x[:1]):
            out = compiled(a)
            self.assertEqual(out.cpu().tolist(), a.transpose(0, 1).cpu().tolist())
            self.assertEqual(out.data_ptr() == a.data_ptr(), a.transpose(0, 1).is_contiguous())
        self.assertEqual(len(cache.graphs), 1)

    def test_public_argument_errors_and_unsupported_forms_before_execution(self):
        x = native.ones((3, 7)).to('cuda:0')
        ref = torch.ones((3, 7), device='cuda:0')
        for args in ('', '0', '0, 1, 2', '0, dim0=0, dim1=1', '0, 1, foo=0',
                     'dim1=1', 'True, 1', '0., 1', 'None, 1', "'axis', 1",
                     '2, 0', '-3, 1', '2, True', '2, 0.', "'a', 'b'",
                     f'{2**63}, True', f'2, {2**63}', f'{2**63}, 0', f'{-2**63-1}, 0'):
            fn = make_program(f'def program(x):\n    a = -x\n    return a.transpose({args})\n')
            try:
                fn(ref)
            except Exception as error:
                expected = type(error)
            else:
                self.fail(f'reference accepted {args}')
            with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                with self.assertRaises(expected, msg=args):
                    compile_with_cache(fn)[0](x)
        for expr in ('x.transpose(x, 1)', 'x.transpose(x.shape[0], 1)', 'x.transpose(x.dim()-1, 0)',
                     'x.swapdims(0, 1)', 'm.transpose(x, 0, dim1=1)', 'x.permute(1, 0)', 'x.T',
                     '-a', 'a * 2', 'a + a', 'a @ a', 'a.sum(1)'):
            fn = make_program(f'def program(x):\n    a = x.transpose(0, 1)\n    b = a.contiguous()\n    return {expr}\n')
            with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                with self.assertRaises(NotImplementedError, msg=expr):
                    compile_with_cache(fn)[0](x)

    def test_cached_nodes_outputs_and_repeated_metadata_pairings(self):
        x = native.ones((1, 1)).to('cuda:0')
        fn = make_program('''def program(x):
    a = x.transpose(0, 1)
    b = -a.contiguous()
    c = [b.transpose(-1, -2)]
    return c, c
''')
        for dynamic in (False, True):
            compiled, cache = compile_with_cache(fn, dynamic=dynamic)
            compiled(x)
            key, graph = next(iter(cache.graphs.items()))
            bad = [replace(graph, dynamic=1), replace(graph, dynamic=None)]
            for i in (0, len(graph.operations)-1):
                for change in ({'axes': None}, {'axes': (True, 1)}, {'axes': (0., 1)},
                               {'axes': (0, 2)}, {'axes': [0, 1]}, {'axes': (0,)},
                               {'scalar': 1}, {'reduction': (1, False)}, {'inputs': ()},
                               {'inputs': ('arg0', 'arg0')}, {'inputs': ['arg0']},
                               {'op': 'call_reduction'}, {'target': True}, {'name': 0}):
                    ops = list(graph.operations); ops[i] = replace(ops[i], **change)
                    bad.append(replace(graph, operations=tuple(ops)))
            for change in ({'shape': (True, 1)}, {'shape': (1., 1)}, {'stride': (1, False)},
                           {'storage_offset': 0.}, {'requires_grad': 0}, {'requires_grad': True}, {'device': 'cuda:1'},
                           {'shape': (2, 1)}, {'dtype': trace.CompileTraceDType('torch.float64')}):
                for i, op in enumerate(graph.operations):
                    ops = list(graph.operations); ops[i] = replace(op, metadata=replace(op.metadata, **change))
                    bad.append(replace(graph, operations=tuple(ops)))
                first, second = graph.output_metadata.elements
                leaf = replace(second.elements[0], **change)
                bad.append(replace(graph, output_metadata=replace(graph.output_metadata,
                    elements=(first, replace(second, elements=(leaf,))))))
            cpu = native.ones((1, 1))
            bad.append(replace(graph, captures=(trace.CompileTraceCapture('unused', cpu, trace._metadata_from_native_tensor(cpu)),)))
            with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')), \
                 patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                for invalid in bad:
                    cache.graphs[key] = invalid
                    with self.assertRaises((NotImplementedError, ValueError, TypeError, IndexError)):
                        compiled(x)
            cache.graphs[key] = graph
            actual = compiled(x)
            self.assertIs(actual[0], actual[1])

    def test_guards_and_no_method_redispatch(self):
        fn = make_program('def program(x):\n    return x.transpose(0, 1)\n')
        x = native.ones((3, 7)).to('cuda:0')
        compiled, cache = compile_with_cache(fn)
        compiled(x)
        graph = next(iter(cache.graphs.values()))
        for wrong in (x.cpu(), x.t(), x[1:], x[:, 1:], native.ones((1, 3, 7)).to('cuda:0'),
                      native.ones((3, 7), requires_grad=True)):
            with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                with self.assertRaises((ValueError, NotImplementedError)):
                    graph.forward(wrong)
        with patch.object(native.Tensor, 'transpose', side_effect=AssertionError('method replay')):
            self.assertEqual(graph.forward(x).cpu().tolist(), [[1.] * 3] * 7)
        def reject(*args):
            raise AssertionError('Tensor method redispatch')
        for replacement in (reject, property(reject)):
            with patch.object(native.Tensor, 'transpose', replacement):
                for wrapper in (compiled, compile_with_cache(fn)[0]):
                    with self.assertRaises(NotImplementedError):
                        wrapper(x)
        fn = make_program('def helper(x):\n    return x.transpose(0, 1).contiguous()\ndef program(x):\n    return helper(x) + bias.transpose(0, 1).contiguous()\n', bias=x)
        compiled, _ = compile_with_cache(fn)
        self.assertEqual(call_without_python(compiled, {fn.__code__, fn.__globals__['helper'].__code__}, x).cpu().tolist(), [[2.] * 3] * 7)
        fn.__globals__['bias'] = native.full((3, 7), 3.).to('cuda:0')
        self.assertEqual(compiled(x).cpu().tolist(), [[4.] * 3] * 7)
        fn.__globals__['bias'] = x.cpu()
        with self.assertRaises(NotImplementedError):
            compiled(x)

    def test_global_axis_type_and_value_guards(self):
        fn = make_program('def program(x):\n    return x.transpose(AXIS, 1)\n', AXIS=0)
        x = native.ones((3, 7)).to('cuda:0')
        for dynamic in (False, True):
            compiled, cache = compile_with_cache(fn, dynamic=dynamic)
            for axis in (0, 1, -2, -1):
                fn.__globals__['AXIS'] = axis
                out = compiled(x)
                self.assertIsNot(out, x)
                self.assertEqual(tuple(out.shape), (7, 3) if axis % 2 == 0 else (3, 7))
            self.assertEqual(len(cache.graphs), 4)
            for axis, error in ((True, TypeError), (0., TypeError), (2, IndexError),
                                (np.int64(0), NotImplementedError)):
                fn.__globals__['AXIS'] = axis
                with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early execution')):
                    with self.assertRaises(error):
                        compiled(x)

    def test_blocked_reference_import(self):
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('reference import')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as n
def f(x):
    y = x.transpose(dim1=1, dim0=0)
    return y, y, y.contiguous(), y.transpose(-1, -2)
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
class TransposeDeviceTests(unittest.TestCase):
    def test_restore_device_context_and_unused_capture(self):
        # Reuse the established context/unused-capture assertions with transpose graphlets.
        from tests import test_compile_cuda_t as t_tests
        fn = make_program('def program(x):\n    return x, x.transpose(0, 1)\n')
        with patch.object(t_tests, 'views', fn):
            t_tests.CompileTDeviceTests('test_device_context_and_unused_capture').test_device_context_and_unused_capture()
