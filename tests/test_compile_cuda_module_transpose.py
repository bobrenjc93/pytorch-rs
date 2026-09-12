"""Native positional transpose capture; non-scoring source-bound regressions."""
from contextlib import ExitStack
from dataclasses import replace
import ctypes
import gc
import itertools
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch_rs as native
from torch_rs import transpose as genuine_transpose
from torch_rs import _compile_bytecode as bytecode, _compile_trace as trace
from tests import test_compile_cuda_module_squeeze as squeeze_tests
from tests.test_compile_cuda_boundary import compile_with_cache
from tests.test_compile_cuda_module_arithmetic import ExactErrors, Hostile
from tests.test_compile_cuda_mul_scalar import POLICIES, call_without_python
from tests.test_cuda_add import Comparison, available, runtime, torch, upload
from tests.test_cuda_contiguous import write_bits


def program(source, module=native, **bindings):
    namespace = {'__name__': __name__, 'm': module,
                 'tr': genuine_transpose if module is native else module.transpose, **bindings}
    exec(source, namespace)
    return namespace['program']


def expression(text, arity=1, module=native, **bindings):
    return program(f'def program({"x" if arity == 1 else "x, y"}):\n    return {text}\n', module, **bindings)


def nested(a, b, module=native):
    return program(f"""def program(x):
    a = m.transpose(x, {a}, {b})
    b = tr(x, {a}, {b})
    shared = [a, (b, tr(a, {a}, {b}), m.t(a))]
    return x, shared, shared, a
""", module)


# Fresh processes exercise joint export mutation before either lazy frontend loads.
_STARTUP = r'''
import sys
import torch_rs as m
from torch_rs import transpose as good
mode, mutation = sys.argv[1:]
assert 'torch_rs._compile_bytecode' not in sys.modules
assert 'torch_rs._compile_trace' not in sys.modules
assert good is m._C.transpose
assert not hasattr(m._C._VariableFunctionsClass, 'transpose')
callbacks = []
def fake(x, dim0, dim1):
    callbacks.append('body')
    return x
fake.__module__ = 'untrusted_import'
class Hostile:
    def __getattribute__(self, name):
        callbacks.append(name)
        raise AssertionError('attribute callback')
    def fail(self, *args, **kwargs):
        callbacks.append('call/equality/conversion/descriptor')
        raise AssertionError('callback')
    __call__ = __eq__ = __hash__ = __bool__ = __float__ = __index__ = __get__ = fail
bad = fake if mutation == 'fake' else Hostile()
for module in (m, m._C):
    if mutation == 'deleted': del module.transpose
    else: module.transpose = bad
bad_alias = bad
def direct(x): return good(x, 0, 1)
def module_call(x): return m.transpose(x, 0, 1)
def counterfeit(x): return bad_alias(x, 0, 1)
def control(x): return m.relu(m.add(m.neg(x), x))
def profile(frame, event, argument):
    if event == 'call' and frame.f_code in (direct.__code__, module_call.__code__,
                                          counterfeit.__code__, control.__code__, fake.__code__):
        raise AssertionError('original Python body')
def reject(callback):
    try:
        callback()
    except Exception as error:
        from torch_rs._compile_trace import CompileTraceUnsupportedError
        assert type(error) is CompileTraceUnsupportedError, type(error)
    else:
        raise AssertionError('accepted counterfeit')
    assert not callbacks, callbacks
sys.setprofile(profile)
assert 'torch_rs._compile_bytecode' not in sys.modules
if mode == 'frontend':
    from torch_rs import _compile_bytecode as b, _compile_trace as t
    meta = (t.CompileTraceTensorMetadata((1, 2), (2, 1), t.float32, 'cuda:0', False, 0),)
    for fn in (module_call, counterfeit):
        for _ in range(2): reject(lambda: b.lower_compile_graph(fn, meta))
    assert b._builtin_target(good).target == 'transpose'
    assert b._builtin_target(bad) is None
    assert [op.target for op in b.lower_compile_graph(direct, meta).operations] == ['transpose']
    assert [op.target for op in b.lower_compile_graph(control, meta).operations] == ['neg', 'add', 'relu']
    m.transpose = m._C.transpose = good
    assert b.lower_compile_graph(module_call, meta).operations[0].target == 'transpose'
else:
    from torch_rs import _compiler_state as state
    x = m.tensor([[-2., 3.]]).to('cuda:0')
    for fullgraph, dynamic in ((True, None), (True, False), (True, True), (False, None)):
        if mutation == 'deleted':
            vars(m).pop('transpose', None); vars(m._C).pop('transpose', None)
        else:
            m.transpose = m._C.transpose = bad
        previous = set(state.native_eager_compile_caches)
        f = m.compile(module_call, backend='eager', fullgraph=fullgraph, dynamic=dynamic, recompile_limit=1)
        cache, = set(state.native_eager_compile_caches) - previous
        for _ in range(2):
            reject(lambda: f(x))
            assert not cache.graphs
        for fn, expected in ((direct, [[-2.], [3.]]), (control, [[0., 0.]])):
            g = m.compile(fn, backend='eager', fullgraph=fullgraph, dynamic=dynamic, recompile_limit=1)
            for _ in range(2):
                out = g(x)
                assert out.cpu().tolist() == expected
                if fn is direct:
                    assert out is not x and out.data_ptr() == x.data_ptr()
        previous = set(state.native_eager_compile_caches)
        g = m.compile(counterfeit, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
        bad_cache, = set(state.native_eager_compile_caches) - previous
        for _ in range(2):
            reject(lambda: g(x))
            assert not bad_cache.graphs
        m.transpose = m._C.transpose = good
        outputs = [f(x), f(x)]
        assert all(v.cpu().tolist() == [[-2.], [3.]] for v in outputs)
        assert outputs[0] is not outputs[1]
        assert all(v.data_ptr() == x.data_ptr() for v in outputs)
        assert len(cache.graphs) == 1
        assert x.cpu().tolist() == [[-2., 3.]]
assert not callbacks, callbacks
'''

def startup(test, mode):
    for mutation in ('fake', 'hostile', 'deleted'):
        with test.subTest(mutation=mutation):
            result = subprocess.run([sys.executable, '-c', _STARTUP, mode, mutation],
                                    capture_output=True, text=True, timeout=60)
            test.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ModuleTransposeFrontendTests(ExactErrors, unittest.TestCase):
    def test_startup_before_frontend_import(self):
        startup(self, 'frontend')

    def test_identity_layouts_boundaries_and_exact_errors(self):
        meta = trace.CompileTraceTensorMetadata((3, 7), (19, 2), trace.float32, 'cuda:0', False, 11)
        self.assertEqual(bytecode._builtin_target(genuine_transpose), bytecode._BytecodeBuiltin('transpose', 3))
        for text in ('m.transpose(x, 0, 1)', 'tr(x, -1, -2)'):
            graph = bytecode.lower_compile_graph(expression(text), (meta,))
            self.assertEqual(graph.operations[0].target, 'transpose')
            self.assertEqual(graph.output_metadata, replace(meta, shape=(7, 3), stride=(2, 19)))
        for text in ('m.transpose()', 'tr(x)', 'tr(x, 0)', 'tr(x, 0, 1, 2)',
                     'tr(1, 0, 1)', 'tr(input=x, dim0=0, dim1=1)', 'tr(x, 0, dim1=1)',
                     'm._C.transpose(x, 0, 1)', 'm.nn.functional.transpose(x, 0, 1)',
                     'tr(x, x, 0)', 'tr(x, x.shape[0], 0)', 'tr(x, x.dim()-1, 0)',
                     'x.transpose_(0, 1)', 'x.swapdims(0, 1)', 'x.swapaxes(0, 1)', 'x.permute(1, 0)', 'x.T',
                     '-tr(x, 0, 1)', 'm.relu(tr(x, 0, 1))', 'tr(x, 0, 1) + tr(x, 0, 1)',
                     'm.mul(tr(x, 0, 1), 2)', 'tr(x, 0, 1).sum(1)'):
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.lower_compile_graph(expression(text), (meta,)))
        for change in ({'device': 'cpu'}, {'requires_grad': True}, {'shape': (1, 2, 3), 'stride': (6, 3, 1)},
                       {'dtype': trace.CompileTraceDType('torch.float64')}):
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.lower_compile_graph(expression('tr(x, 0, 1)'), (replace(meta, **change),)))
        for axes, error in (('True, 1', TypeError), ('0., 1', TypeError), ('None, 1', TypeError),
                            ('2, True', TypeError), ('2, 0.', TypeError), ('2, 0', IndexError),
                            ('-3, 1', IndexError), (f'{2**63}, True', TypeError),
                            (f'2, {2**63}', ValueError), (f'{-2**63-1}, 0', ValueError)):
            self.exact_error(error, lambda: bytecode.lower_compile_graph(expression(f'tr(x, {axes})'), (meta,)))
        class Integer(int): pass
        for axis in (Hostile(), Integer(0), np.int64(0)):
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.lower_compile_graph(expression('tr(x, AXIS, 1)', AXIS=axis), (meta,)))
        for policy in POLICIES:
            f, cache = compile_with_cache(expression('tr(x, 0, 1)'), *policy, limit=1)
            for _ in range(2):
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(native.ones((3, 7))))
                self.assertEqual(cache.graphs, {})

    def test_wrong_namespaces_descriptors_and_deleted_fields(self):
        meta = (trace.CompileTraceTensorMetadata((3, 7), (7, 1), trace.float32, 'cuda:0', False, 0),)
        fake = types.ModuleType('torch_rs'); fake.__dict__.update(vars(native))
        class ModuleSubclass(types.ModuleType):
            __getattribute__ = Hostile.fail
        for value in (fake, ModuleSubclass('torch_rs'), Hostile(), types.SimpleNamespace(transpose=genuine_transpose)):
            fn = expression('m.transpose(x, 0, 1)'); fn.__globals__['m'] = value
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.prepare_compile_cache_request(fn, meta))
        for value in (Hostile(), property(Hostile.fail), native.abs):
            fn = expression('tr(x, 0, 1)'); fn.__globals__['tr'] = value
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.prepare_compile_cache_request(fn, meta))
        with patch.object(native, '__getattr__', Hostile.fail, create=True):
            saved = vars(native).pop('transpose')
            try:
                self.exact_error(trace.CompileTraceUnsupportedError,
                                 lambda: bytecode.prepare_compile_cache_request(expression('m.transpose(x, 0, 1)'), meta))
            finally:
                native.transpose = saved


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class ModuleTransposeCudaTests(ExactErrors, squeeze_tests.BitComparison, unittest.TestCase):
    # Shared assertions require cold/warm/lowered-graph execution, immutable inputs,
    # fresh outputs and zero original-body/per-node-Python execution.
    checked = squeeze_tests.ModuleSqueezeCudaTests.checked

    def test_startup_before_first_compile(self):
        startup(self, 'cuda')

    def test_original_96_cells_inductor_all_policies(self):
        rng = np.random.default_rng(198601)
        layouts = (((), 0, 0), ((), -1, 0), ((7,), 0, 0), ((7,), -1, -1),
                   ((3, 5), 0, 1), ((3, 5), -1, -2), ((3, 5), 0, 0), ((3, 5), 1, 1),
                   ((0, 1), 0, 1), ((0, 1), -1, -2), ((1, 7), 0, 1), ((1, 7), -1, -2))
        for template in ('m.transpose(x, {a}, {b})', 'tr(x, {a}, {b})',
                         'm.transpose(x + x, {a}, {b})', 'tr(x + x, {a}, {b})',
                         'm.relu(m.transpose(x, {a}, {b}).contiguous())',
                         'm.relu(tr(x, {a}, {b}).contiguous())',
                         'm.transpose(m.relu(x), {a}, {b})', 'tr(m.relu(x), {a}, {b})'):
            for shape, a, b in layouts:
                text = template.format(a=a, b=b)
                values = rng.integers(-31, 32, size=shape).astype(np.float32) * .125
                args, refs = [upload(native, values.reshape(-1), shape)], [upload(torch, values.reshape(-1), shape)]
                for policy in POLICIES:
                    with self.subTest(text=text, shape=shape, policy=policy):
                        torch._dynamo.reset()
                        self.checked(expression(text), expression(text, module=torch), args, refs, policy,
                                     'inductor', aliases=template in ('m.transpose(x, {a}, {b})', 'tr(x, {a}, {b})'))

    def test_layout_axes_nested_identity_and_lifetime(self):
        rng = np.random.default_rng(198602)
        layouts = (lambda x: x, lambda x: x.t(), lambda x: x[1:][:, 1:],
                   lambda x: x[1:][:, 1:].t(), lambda x: x.select(1, 1)[1:],
                   lambda x: x.select(0, 1).select(0, 1), lambda x: x[:1],
                   lambda x: x[:1][:, :1], lambda x: x[:0], lambda x: x[:, :0])
        for shape in ((3, 7), (17, 31)):
            for layout in layouts:
                rank = max(len(layout(torch.zeros(shape)).shape), 1)
                for a, b in itertools.product(range(-rank, rank), repeat=2):
                    for policy in POLICIES:
                        torch._dynamo.reset()
                        fn, ref = nested(a, b), nested(a, b, torch)
                        f, cache = compile_with_cache(fn, *policy)
                        reference = torch.compile(ref, backend='eager', fullgraph=policy[0], dynamic=policy[1])
                        previous = None
                        for warm in (False, True):
                            data = rng.normal(size=shape).astype(np.float32)
                            x = layout(native.tensor(data).to('cuda:0'))
                            y = layout(torch.tensor(data, device='cuda:0'))
                            with ExitStack() as stack:
                                stack.enter_context(patch.object(trace, '_execute_operation', side_effect=AssertionError('Python node')))
                                if warm:
                                    stack.enter_context(patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')))
                                out = call_without_python(f, {fn.__code__}, x)
                            expected = reference(y)
                            self.assertIs(out[0], x)
                            self.assertIs(out[1], out[2]); self.assertIs(out[3], out[1][0])
                            leaves, refs = [out[3], *out[1][1]], [expected[3], *expected[1][1]]
                            self.assertEqual(len({id(v) for v in leaves}), 4)
                            for value, ref_value in zip(leaves, refs):
                                self.assertIsNot(value, x); self.assertIsNot(value, previous)
                                self.assertEqual(value.data_ptr(), x.data_ptr())
                                self.compare(value, ref_value)
                            self.compare(x, y)
                            previous = out[3]
                        self.assertEqual(len(cache.graphs), 1)
                        del x, out, leaves, f, cache
                        gc.collect()
                        self.compare(previous, expected[3])
        for shape in ((), (0,), (1,), (257,), (0, 2**32), (2**32+1, 0)):
            rank = max(len(shape), 1)
            for policy in POLICIES:
                torch._dynamo.reset()
                text = f'tr(x, 0, {rank-1})'
                self.checked(expression(text), expression(text, module=torch),
                             [native.zeros(shape).to('cuda:0')], [torch.zeros(shape, device='cuda:0')], policy, aliases=True)

    def test_shared_raw_bits_mutation_and_parent_deletion(self):
        bits = np.array([0, 0x80000000, 0x7f800001, 0xff800001, 0x7fc12345, 0xffc54321,
                         1, 0x80000001, 0x7f800000, 0xff800000, 0x3f800000, 0xbf800000], dtype=np.uint32)
        for policy in POLICIES:
            for layout in (lambda x: x, lambda x: x[1:][:, 1:], lambda x: x.t(),
                           lambda x: x.select(1, 1), lambda x: x.select(0, 1).select(0, 1)):
                a, b = native.zeros((3, 4), device='cuda:0'), torch.zeros((3, 4), device='cuda:0')
                write_bits(a, bits); write_bits(b, bits)
                x, y = layout(a), layout(b)
                for axes in ((0, max(x.dim(), 1)-1), (-1, -1)):
                    fn, ref = nested(*axes), nested(*axes, torch)
                    f, cache = compile_with_cache(fn, *policy)
                    outputs = [call_without_python(f, {fn.__code__}, x)[3] for _ in range(2)]
                    expected = ref(y)[3]
                    self.assertIsNot(outputs[0], outputs[1])
                    for out in outputs: self.compare(out, expected)
                    self.compare(a, b)
                    write_bits(a, bits[::-1].copy()); write_bits(b, bits[::-1].copy())
                    for out in outputs: self.compare(out, expected)
                    u, v = outputs[0], expected
                    while u.dim(): u, v = u.select(0, 0), v.select(0, 0)
                    write_bits(u, np.array([0x80000000], dtype=np.uint32))
                    write_bits(v, np.array([0x80000000], dtype=np.uint32))
                    self.compare(a, b)
                del a, x, u, f, cache
                gc.collect()
                for out in outputs: self.compare(out, expected)

    def test_compositions_helpers_and_dynamic_rectangles(self):
        texts = ('tr(m.add(x, y), 0, 1)', 'm.add(tr(x, 0, 1).contiguous(), tr(y, 0, 1).contiguous())',
                 'm.neg(tr(x, 0, 1).contiguous())', 'tr(m.neg(x), 0, 1)',
                 'm.multiply(tr(x, 0, 1).contiguous(), -.5)', 'tr(m.mul(x, -.5), 0, 1)',
                 'tr(x, 0, 1).contiguous().view(-1)', 'tr(x, 0, 1).reshape(-1)',
                 'tr(x, 0, 1).contiguous().sum(1)', 'tr(x.sum(1, keepdim=True), 0, 1)',
                 'tr(x, 0, 1).t().transpose(0, 0)', 'm.t(tr(x, 0, 1))',
                 'tr(m.t(x), -1, -2)', 'm.squeeze(tr(x, 0, 1))', 'tr(x.squeeze(), 0, 1)',
                 'tr(x, 0, 1).contiguous().matmul(y)', 'tr(m.matmul(x, y.t().contiguous()), 0, 1)')
        data = np.arange(15, dtype=np.float32).reshape(3, 5) * .125 - 1
        args, refs = [native.tensor(data).to('cuda:0')] * 2, [torch.tensor(data, device='cuda:0')] * 2
        sources = [f'def program(x, y):\n    return {text}\n' for text in texts]
        sources += ['def helper(x):\n    return m.transpose(x, 0, 1)\ndef program(x, y):\n    return m.relu(helper(x).contiguous()) + tr(y, 0, 1).contiguous()\n']
        for source in sources:
            for policy in POLICIES:
                torch._dynamo.reset()
                self.checked(program(source), program(source, torch), args, refs, policy)
        for text in ('tr(x, 0, 1)', 'm.relu(tr(x, 0, 1).contiguous())',
                     'tr(x, 0, 1).t()', 'tr(tr(x, -1, -2), 0, 1)',
                     'm.neg(tr(x, 0, 1).contiguous())', 'm.mul(tr(x, 0, 1).contiguous(), .5)'):
            for policy in POLICIES:
                torch._dynamo.reset()
                fn, ref = expression(text), expression(text, module=torch)
                f, cache = compile_with_cache(fn, *policy)
                reference = torch.compile(ref, backend='eager', fullgraph=policy[0], dynamic=policy[1])
                base, tbase = native.ones((7, 11)).to('cuda:0'), torch.ones((7, 11), device='cuda:0')
                for rows, cols in ((3, 7), (5, 9), (1, 1), (1, 7), (3, 1), (0, 1), (1, 0), (3, 7)):
                    x, y = base[:rows][:, :cols], tbase[:rows][:, :cols]
                    for _ in range(2):
                        with ExitStack() as stack:
                            if policy[1] and cache.graphs:
                                stack.enter_context(patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')))
                            self.compare(call_without_python(f, {fn.__code__}, x), reference(y))
                        self.compare(x, y)
                self.assertEqual(len(cache.graphs), 1 if policy[1] else 7)
        # A warm dynamic graph must reject newly strided arithmetic before native work.
        fn = expression('m.relu(tr(x, 0, 1))'); f, cache = compile_with_cache(fn, dynamic=True)
        x = native.ones((3, 7)).to('cuda:0'); f(x[:1])
        with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')), \
             patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native work')):
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
        self.assertEqual(len(cache.graphs), 1)

    def test_per_used_field_guards_aliases_and_recovery(self):
        x, tx = native.tensor([[-2., 3.]]).to('cuda:0'), torch.tensor([[-2., 3.]], device='cuda:0')
        controls = ('m.t(x)', 'm.squeeze(x)', 'm.relu(x)', 'm.add(x, x)', 'm.neg(x)',
                    'm.mul(x, 2)', 'm.matmul(x, x.t().contiguous())', 'tr(x, 0, 1)')
        sources = [f'def program(x):\n    return {text}\n' for text in controls]
        sources += ['def helper(x):\n    return tr(x, 0, 1)\ndef program(x):\n    return m.relu(helper(x).contiguous())\n']
        for source in sources:
            for policy in POLICIES:
                fn, ref = program(source), program(source, torch)
                f, cache = compile_with_cache(fn, *policy, limit=1)
                self.compare(f(x), ref(tx)); graph = next(iter(cache.graphs.values()))
                for deleted in (False, True):
                    with ExitStack() as stack:
                        for module in (native, native._C):
                            stack.enter_context(patch.object(module, 'transpose', Hostile()))
                            if deleted: del module.transpose
                            stack.enter_context(patch.object(module, '__getattr__', Hostile.fail, create=True))
                        with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('unused field cache miss')):
                            self.compare(f(x), ref(tx))
                        self.compare(compile_with_cache(fn, *policy, limit=1)[0](x), ref(tx))
                        self.compare(graph.forward(x), ref(tx))
                self.assertEqual(len(cache.graphs), 1)
        sources = [('def program(x):\n    return m.transpose(x, 0, 1)\n', 'transpose'),
                   ('def program(x):\n    return tr(x, 0, 1)\n', 'tr'),
                   ('def helper(x):\n    return m.transpose(x, 0, 1)\ndef program(x):\n    return m.relu(helper(x).contiguous())\n', 'transpose')]
        for source, field in sources:
            for policy in POLICIES:
                fn, ref = program(source), program(source, torch)
                bindings = vars(native) if field == 'transpose' else fn.__globals__
                f, cache = compile_with_cache(fn, *policy, limit=1)
                self.compare(f(x), ref(tx)); graph = next(iter(cache.graphs.values()))
                # Canonical one/two-argument operations still reject this three-argument call.
                for value in (Hostile(), native.mul, native.t, native.neg, native.squeeze, property(Hostile.fail)):
                    with patch.dict(bindings, {field: value}):
                        for _ in range(2): self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
                        self.compare(graph.forward(x), ref(tx))
                        failed, empty = compile_with_cache(fn, *policy, limit=1)
                        for _ in range(2):
                            self.exact_error(trace.CompileTraceUnsupportedError, lambda: failed(x))
                            self.assertEqual(empty.graphs, {})
                    self.compare(failed(x), ref(tx))
                    self.assertEqual(len(cache.graphs), 1)
                saved = bindings.pop(field)
                try: self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
                finally: bindings[field] = saved
                self.compare(f(x), ref(tx))
        # The genuine callable works under other supported field names; spelling grants no credit.
        for field in ('t', 'squeeze', 'relu', 'add', 'neg', 'mul', 'matmul'):
            fn = expression(f'm.{field}(x, 0, 1)'); f, cache = compile_with_cache(fn, limit=1)
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
            self.assertEqual(cache.graphs, {})
            with patch.object(native, field, genuine_transpose):
                for _ in range(2): self.compare(f(x), tx.transpose(0, 1))
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
            self.assertEqual(len(cache.graphs), 1)
        for replacement, expected in ((native.t, tx.t()), (native.squeeze, tx.squeeze()),
                                      (native.relu, tx.relu()), (native.neg, -tx)):
            for policy in POLICIES:
                f, cache = compile_with_cache(expression('m.transpose(x)'), *policy, limit=1)
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
                self.assertEqual(cache.graphs, {})
                with patch.object(native, 'transpose', replacement):
                    for _ in range(2): self.compare(f(x), expected)
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
                self.assertEqual(len(cache.graphs), 1)
        fn = expression('tr(x, 0, 1)')
        other = types.FunctionType(fn.__code__, {**fn.__globals__, 'tr': Hostile()})
        good, _ = compile_with_cache(fn); bad, cache = compile_with_cache(other)
        self.compare(good(x), tx.t())
        self.exact_error(trace.CompileTraceUnsupportedError, lambda: bad(x)); self.assertEqual(cache.graphs, {})
        other.__globals__['tr'] = genuine_transpose
        self.compare(bad(x), tx.t())
        fn = expression('m.transpose(x, 0, 1)'); f, cache = compile_with_cache(fn, limit=1); f(x)
        fake = types.ModuleType('torch_rs'); fake.__dict__.update(vars(native))
        for module in (fake, types.SimpleNamespace(transpose=genuine_transpose), Hostile()):
            with patch.dict(fn.__globals__, {'m': module}):
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
        self.compare(f(x), tx.t()); self.assertEqual(len(cache.graphs), 1)

    def test_axis_globals_error_precedence_and_failed_cache_recovery(self):
        x, tx = native.ones((3, 7)).to('cuda:0'), torch.ones((3, 7), device='cuda:0')
        for policy in POLICIES:
            fn = program('def program(x):\n    b = 1\n    return tr(m.neg(x), AXIS, b)\n', AXIS=0)
            f, cache = compile_with_cache(fn, *policy)
            for axis in (0, 1, -2, -1):
                fn.__globals__['AXIS'] = axis
                for _ in range(2): self.compare(f(x), (-tx).transpose(axis, 1))
            self.assertEqual(len(cache.graphs), 4)
            for axis, error in ((True, TypeError), (0., TypeError), (2, IndexError),
                                (2**63, ValueError), (Hostile(), trace.CompileTraceUnsupportedError),
                                (np.int64(0), trace.CompileTraceUnsupportedError), (x, trace.CompileTraceUnsupportedError)):
                fn.__globals__['AXIS'] = axis
                with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native work')):
                    self.exact_error(error, lambda: f(x))
                self.assertEqual(len(cache.graphs), 4)
            fn.__globals__['AXIS'] = 0
            self.compare(f(x), (-tx).t())
            for axes in ('True, 1', '0., 1', 'None, 1', '2, 0', '2, True', '2, 0.',
                         f'{2**63}, True', f'2, {2**63}', f'{-2**63-1}, 0'):
                ref = expression(f'm.transpose(x, {axes})', module=torch)
                try: ref(tx)
                except Exception as error: kind = type(error)
                else: self.fail(axes)
                fn = expression(f'tr(m.neg(x), {axes})'); f, cache = compile_with_cache(fn, *policy)
                with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native work')):
                    for _ in range(2): self.exact_error(kind, lambda: f(x))
                self.assertEqual(cache.graphs, {})

    def test_whole_graph_prevalidation_and_unused_inputs(self):
        x = native.ones((3, 7)).to('cuda:0')
        fn = expression('tr(m.relu(tr(x, 0, 1).contiguous()), -1, -2)', 2)
        for policy in POLICIES:
            f, cache = compile_with_cache(fn, *policy); f(x, x)
            key, graph = next(iter(cache.graphs.items())); node = graph.operations[-1]
            bad = [replace(graph, output='missing'), replace(graph, output_metadata=None)]
            bad += [replace(graph, operations=(*graph.operations[:-1], replace(node, **change))) for change in
                    ({'inputs': ('missing',)}, {'inputs': ('arg0', 'arg0')}, {'scalar': 1}, {'shape': ()},
                     {'axes': None}, {'axes': [0, 1]},
                     {'target': 'flatten'}, {'metadata': None},
                     {'metadata': replace(node.metadata, storage_offset=1)},
                     {'metadata': replace(node.metadata, requires_grad=True)},
                     {'metadata': replace(node.metadata, stride=(True, 1))},
                     {'metadata': replace(node.metadata, device='cuda:1')})]
            bad = [(trace.CompileTraceUnsupportedError, invalid) for invalid in bad]
            bad += [(error, replace(graph, operations=(*graph.operations[:-1], replace(node, axes=axes))))
                    for axes, error in (((True, 1), TypeError), ((0, 2), IndexError))]
            with ExitStack() as stack:
                for owner, name in ((trace, '_execute_operation'), (trace._native, '_compile_trace_cuda_graph'),
                                    (trace._native, '_compile_trace_unary'), (trace._native, '_compile_trace_binary')):
                    stack.enter_context(patch.object(owner, name, side_effect=AssertionError('early execution')))
                for error, invalid in bad:
                    cache.graphs[key] = invalid
                    self.exact_error(error, lambda: f(x, x))
                cache.graphs[key] = graph
                for wrong in (x.cpu(), native.ones((1, 3, 7)).to('cuda:0'), native.ones((3, 7), requires_grad=True)):
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(wrong, x))
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x, x.cpu()))
                self.exact_error(TypeError, lambda: f(x, Hostile()))
            self.assertEqual(len(cache.graphs), 1)
            self.compare(f(x, x), torch.ones((7, 3), device='cuda:0').t())
        f, cache = compile_with_cache(fn)
        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=RuntimeError('launch failed')):
            self.exact_error(RuntimeError, lambda: f(x, x))
        self.assertEqual(cache.graphs, {})
        f(x, x)
        f, cache = compile_with_cache(nested(0, 1)); f(x); key, graph = next(iter(cache.graphs.items()))
        a, first, second, last = graph.output_metadata.elements
        leaf, pair = second.elements
        invalid = replace(second, elements=(replace(leaf, stride=(True, 1)), pair))
        cache.graphs[key] = replace(graph, output_metadata=replace(graph.output_metadata, elements=(a, first, invalid, last)))
        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native work')):
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
        cache.graphs[key] = graph; self.compare(f(x)[3], torch.ones((3, 7), device='cuda:0').t())

    def test_blocked_reference_import_and_helper_bodies(self):
        source = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('reference import')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
from torch_rs import transpose as tr
def helper(x): return m.transpose(x, 0, 1)
def f(x):
    a = tr(helper(x), -1, -2)
    return a, a, tr(x, 0, 1)
from torch_rs import _compile_trace as t
def reject(*args): raise AssertionError('Python node')
t._execute_operation = reject
def profile(frame, event, arg):
    if event == 'call' and frame.f_code in (f.__code__, helper.__code__):
        raise AssertionError('original body')
sys.setprofile(profile)
g = m.compile(f, backend='eager', fullgraph=True)
for value in (-2., 3.):
    x = m.full((3, 7), value).to('cuda:0')
    a, b, c = g(x)
    assert a is b and a is not c and a is not x
    assert a.data_ptr() == c.data_ptr() == x.data_ptr()
    assert a.cpu().tolist() == [[value] * 7] * 3
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class ModuleTransposeDeviceTests(ExactErrors, Comparison, unittest.TestCase):
    def test_restoration_unused_inputs_captures_and_lifetime(self):
        lib, previous = runtime(), torch.cuda.current_device()
        driver = ctypes.CDLL('libcuda.so.1')
        driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        def context():
            value = ctypes.c_void_p()
            self.assertEqual(driver.cuCtxGetCurrent(ctypes.byref(value)), 0)
            return value.value
        try:
            for policy in POLICIES:
                f, cache = compile_with_cache(expression('m.transpose(x, 0, 1)', 2), *policy)
                for current, target in ((1, 0), (0, 1), (1, 0)):
                    torch.cuda.set_device(current)
                    torch.empty(1, device=f'cuda:{current}')
                    before = context()
                    x = native.full((1, 7), 2.).to(f'cuda:{target}')
                    out = f(x, x)
                    self.assertEqual(context(), before)
                    self.assertEqual(torch.cuda.current_device(), current)
                    self.assertEqual(out.data_ptr(), x.data_ptr())
                    expected = torch.full((1, 7), 2., device=f'cuda:{target}').transpose(0, 1)
                    self.compare(out, expected)
                    wrong = native.ones((1, 7)).to(f'cuda:{current}')
                    with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('launched')):
                        self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x, wrong))
                        key, graph = next((k, g) for k, g in cache.graphs.items() if g.operations[0].metadata.device == f'cuda:{target}')
                        bad = replace(graph, captures=(trace.CompileTraceCapture('unused', wrong, trace._metadata_from_native_tensor(wrong)),))
                        self.exact_error(trace.CompileTraceUnsupportedError, lambda: bad.forward(x, x))
                    del x, wrong
                    gc.collect()
                    self.compare(out, expected)
                    ordinal = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
                    self.assertEqual(context(), before)
                self.assertEqual(len(cache.graphs), 2)
        finally:
            torch.cuda.set_device(previous)
