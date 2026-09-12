"""Trusted t spelling/view regressions; independent of scoring corpora."""
from contextlib import ExitStack
import ctypes
from dataclasses import replace
import gc
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch_rs as native
from torch_rs import t as genuine_t
from torch_rs import _compile_bytecode as bytecode, _compile_trace as trace
from tests.test_compile_cuda_boundary import compile_with_cache
from tests.test_compile_cuda_module_arithmetic import ExactErrors, Hostile
from tests.test_compile_cuda_module_squeeze import BitComparison
from tests.test_compile_cuda_mul_scalar import POLICIES, call_without_python
from tests.test_cuda_add import Comparison, available, runtime, torch, upload
from tests.test_cuda_contiguous import metadata, read_bits, write_bits


def program(source, module=native):
    namespace = {'__name__': __name__, 'm': module,
                 's': genuine_t if module is native else module.t}
    exec(source, namespace)
    return namespace['program']


def expression(text, arity=1, module=native):
    return program(f'def program({"x" if arity == 1 else "x, y"}):\n    return {text}\n', module)


NESTED = '''def program(x):
    a = m.t(x)
    b = s(x)
    shared = [a, (b, s(a))]
    return x, shared, shared, a
'''


_STARTUP = r'''
import sys
import torch_rs as m
from torch_rs import t as good
mode, mutation = sys.argv[1:]
assert 'torch_rs._compile_bytecode' not in sys.modules
assert 'torch_rs._compile_trace' not in sys.modules
assert good is m._C.t
assert not hasattr(m._C._VariableFunctionsClass, 't')
callbacks = []
def fake(x):
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
    if mutation == 'deleted': del module.t
    else: module.t = bad
bad_alias = bad
def direct(x): return good(x)
def module_call(x): return m.t(x)
def counterfeit(x): return bad_alias(x)
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
    assert b._builtin_target(good).target == 't'
    assert b._builtin_target(bad) is None
    assert [op.target for op in b.lower_compile_graph(direct, meta).operations] == ['t']
    assert [op.target for op in b.lower_compile_graph(control, meta).operations] == ['neg', 'add', 'relu']
    m.t = m._C.t = good
    assert b.lower_compile_graph(module_call, meta).operations[0].target == 't'
else:
    from torch_rs import _compiler_state as state
    x = m.tensor([[-2., 3.]]).to('cuda:0')
    for fullgraph, dynamic in ((True, None), (True, False), (True, True), (False, None)):
        if mutation == 'deleted':
            vars(m).pop('t', None); vars(m._C).pop('t', None)
        else:
            m.t = m._C.t = bad
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
        m.t = m._C.t = good
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


class ModuleTFrontendTests(ExactErrors, unittest.TestCase):
    def test_startup_before_frontend_import(self):
        startup(self, 'frontend')

    def test_identity_arity_layouts_and_unchanged_boundaries(self):
        meta = trace.CompileTraceTensorMetadata((1, 7), (19, 3), trace.float32, 'cuda:0', False, 11)
        self.assertEqual(bytecode._builtin_target(genuine_t), bytecode._BytecodeBuiltin('t', 1))
        for text in ('m.t(x)', 's(x)', 'm.t(s(x))'):
            graph = bytecode.lower_compile_graph(expression(text), (meta,))
            self.assertEqual(graph.operations[-1].target, 't')
            self.assertEqual(graph.output_metadata, meta if text == 'm.t(s(x))' else replace(meta, shape=(7, 1), stride=(3, 19)))
        for text in ('m.t()', 's(x, 0)', 's(x, (0,))', 's(x, [0])', 's(x, 0, 1)',
                     'm.t(1)', 's(input=x)', 'm.t(x, dim=0)', 'm.t(x, out=None)',
                     'm._C.t(x)', 'm.nn.functional.t(x)', 'x.t_()', 'x.flatten()',
                     'x.unsqueeze(0)', 'm.t(x).relu()', 'm.neg(s(x))', 's(x) + s(x)',
                     'm.multiply(s(x), 2)', 's(x).sum(1)'):
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.lower_compile_graph(expression(text), (meta,)))
        for change in ({'device': 'cpu'}, {'requires_grad': True}, {'shape': (1, 2, 1), 'stride': (2, 1, 1)},
                       {'dtype': trace.CompileTraceDType('torch.float64')}):
            for text in ('m.t(x)', 's(x)'):
                self.exact_error(trace.CompileTraceUnsupportedError,
                                 lambda: bytecode.lower_compile_graph(expression(text), (replace(meta, **change),)))
        for policy in POLICIES:
            fn = expression('m.t(x)')
            f, cache = compile_with_cache(fn, *policy, limit=1)
            for _ in range(2):
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(native.ones((1, 7))))
                self.assertEqual(cache.graphs, {})

    def test_wrong_namespaces_descriptors_and_deleted_fields(self):
        meta = (trace.CompileTraceTensorMetadata((1, 7), (7, 1), trace.float32, 'cuda:0', False, 0),)
        fake = types.ModuleType('torch_rs'); fake.__dict__.update(vars(native))
        class ModuleSubclass(types.ModuleType):
            __getattribute__ = Hostile.fail
        for value in (fake, ModuleSubclass('torch_rs'), Hostile(), types.SimpleNamespace(t=genuine_t)):
            fn = expression('m.t(x)'); fn.__globals__['m'] = value
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.prepare_compile_cache_request(fn, meta))
        for value in (Hostile(), property(Hostile.fail), native.abs):
            fn = expression('s(x)'); fn.__globals__['s'] = value
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.prepare_compile_cache_request(fn, meta))
        fn = expression('m.t(x)')
        with patch.object(native, '__getattr__', Hostile.fail, create=True):
            saved = vars(native).pop('t')
            try:
                self.exact_error(trace.CompileTraceUnsupportedError,
                                 lambda: bytecode.prepare_compile_cache_request(fn, meta))
            finally:
                native.t = saved


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class ModuleTCudaTests(ExactErrors, BitComparison, unittest.TestCase):
    def test_startup_before_first_compile(self):
        startup(self, 'cuda')

    def checked(self, fn, ref, args, refs, policy, backend='eager', aliases=False):
        f, cache = compile_with_cache(fn, *policy)
        reference = torch.compile(ref, backend=backend, fullgraph=policy[0], dynamic=policy[1])
        codes = {v.__code__ for v in fn.__globals__.values() if type(v) is types.FunctionType}
        retained = []
        for warm in (False, True):
            with ExitStack() as stack:
                stack.enter_context(patch.object(trace, '_execute_operation', side_effect=AssertionError('Python node')))
                if warm:
                    stack.enter_context(patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')))
                out = call_without_python(f, codes, *args)
            self.compare(out, reference(*refs))
            for x, tx in zip(args, refs): self.compare(x, tx)
            self.assertIsNot(out, args[0])
            if aliases: self.assertEqual(out.data_ptr(), args[0].data_ptr())
            retained.append(out)
        self.assertIsNot(retained[0], retained[1])
        self.assertEqual(len(cache.graphs), 1)
        graph = next(iter(cache.graphs.values()))
        self.compare(graph.forward(*args), ref(*refs))
        return f, cache

    def test_seeded_original_32_cells_inductor_all_policies(self):
        rng = np.random.default_rng(198501)
        for text in ('m.t(x)', 's(x)', 'm.t(x + x)', 's(x + x)',
                     'm.relu(m.t(x).contiguous())', 'm.relu(s(x).contiguous())', 'm.t(m.relu(x))', 's(m.relu(x))'):
            for shape in ((), (7,), (3, 5), (0, 1)):
                values = rng.integers(-31, 32, size=int(np.prod(shape))).astype(np.float32) * .125
                args, refs = [upload(native, values, shape)], [upload(torch, values, shape)]
                for policy in POLICIES:
                    with self.subTest(text=text, shape=shape, policy=policy):
                        torch._dynamo.reset()
                        self.checked(expression(text), expression(text, module=torch), args, refs,
                                     policy, 'inductor', aliases=text in ('m.t(x)', 's(x)'))

    def test_seeded_layouts_nested_identity_and_lifetime(self):
        rng = np.random.default_rng(198502)
        shapes = [(3, 7), (17, 31)] + [tuple(map(int, rng.integers(3, 30, 2))) for _ in range(2)]
        layouts = (lambda x: x, lambda x: x.t(), lambda x: x[:1], lambda x: x[:, :1],
                   lambda x: x[:1].t(), lambda x: x[1:2][:, 1:], lambda x: x[:, 1:2],
                   lambda x: x.select(1, 1)[1:], lambda x: x.select(0, 1).select(0, 1),
                   lambda x: x[:1][:, :1], lambda x: x[:0], lambda x: x[:1][:, :0],
                   lambda x: x[:0][:, :1])
        for shape in shapes:
            for layout in layouts:
                for policy in POLICIES:
                    torch._dynamo.reset()
                    fn, ref = program(NESTED), program(NESTED, torch)
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
                            actual = call_without_python(f, {fn.__code__}, x)
                        expected = reference(y)
                        self.assertIs(actual[0], x)
                        self.assertIs(actual[1], actual[2])
                        self.assertIs(actual[1][0], actual[3])
                        leaves = [actual[3], *actual[1][1]]
                        self.assertEqual(len({id(v) for v in leaves}), 3)
                        for a, b in zip(leaves, [expected[3], *expected[1][1]]):
                            self.assertIsNot(a, x)
                            self.assertIsNot(a, previous)
                            self.assertEqual(a.data_ptr(), x.data_ptr())
                            self.compare(a, b)
                        self.compare(x, y)
                        previous = actual[3]
                    self.assertEqual(len(cache.graphs), 1)
                    del x, actual, f, cache
                    gc.collect()
                    self.compare(previous, expected[3])
        for shape in ((), (0,), (1,), (7,), (257,), (1, 0), (0, 1), (0, 2**32)):
            for policy in POLICIES:
                torch._dynamo.reset()
                self.checked(expression('s(x)'), expression('s(x)', module=torch),
                             [native.zeros(shape).to('cuda:0')], [torch.zeros(shape, device='cuda:0')],
                             policy, aliases=True)

    def test_shared_ieee_bits_mutation_parent_deletion(self):
        bits = np.array([0, 0x80000000, 0x7f800001, 0xff800001, 0x7fc12345, 0xffc54321,
                         1, 0x80000001, 0x7f800000, 0xff800000, 0x3f800000, 0xbf800000], dtype=np.uint32)
        for policy in POLICIES:
            for view in (lambda x: x[:1], lambda x: x[:, 1:2], lambda x: x[1:2][:, 1:],
                         lambda x: x.t(), lambda x: x.select(1, 1),
                         lambda x: x.select(0, 1).select(0, 1)):
                a, b = native.zeros((3, 4), device='cuda:0'), torch.zeros((3, 4), device='cuda:0')
                write_bits(a, bits); write_bits(b, bits)
                x, y = view(a), view(b)
                fn, ref = program(NESTED), program(NESTED, torch)
                f, cache = compile_with_cache(fn, *policy)
                expected = ref(y)
                outputs = [call_without_python(f, {fn.__code__}, x)[3] for _ in range(2)]
                self.compare(a, b)
                for out in outputs:
                    self.assertEqual(out.data_ptr(), x.data_ptr())
                    self.compare(out, expected[3])
                write_bits(a, bits[::-1].copy()); write_bits(b, bits[::-1].copy())
                for out in outputs: self.compare(out, expected[3])
                u, v = outputs[0], expected[3]
                while u.dim(): u, v = u.select(0, 0), v.select(0, 0)
                write_bits(u, np.array([0x80000000], dtype=np.uint32))
                write_bits(v, np.array([0x80000000], dtype=np.uint32))
                self.compare(a, b)
                del a, x, u, f, cache
                gc.collect()
                for out in outputs: self.compare(out, expected[3])

    def test_compositions_helpers_all_policies(self):
        rng = np.random.default_rng(198503)
        texts = ('m.t(m.add(x, y))', 'm.add(s(x).contiguous(), s(y).contiguous())',
                 'm.neg(s(x).contiguous())', 's(m.neg(x))',
                 'm.multiply(s(x).contiguous(), -.5)', 's(m.mul(x, -.5))',
                 's(x).contiguous().view(-1)', 's(x.view(-1))', 's(x).reshape(5, 3)',
                 's(x.reshape(5, 3))', 's(x).contiguous().sum(1)', 's(x.sum(1, keepdim=True))',
                 's(x).t().transpose(0, 1).contiguous()', 's(x.t().contiguous())',
                 's(x.matmul(y.t().contiguous()))', 's(x).contiguous().matmul(y)',
                 's(x).squeeze()', 'm.t(x.squeeze())', 'm.squeeze(s(x))',
                 'm.t(s(m.t(s(x))))')
        values = rng.integers(-31, 32, size=15).astype(np.float32) * .125
        args, refs = [upload(native, values, (3, 5))] * 2, [upload(torch, values, (3, 5))] * 2
        sources = [f'def program(x, y):\n    return {text}\n' for text in texts]
        sources += ['def helper(x):\n    return m.t(x)\ndef program(x, y):\n    return m.relu(helper(x).contiguous()) + s(y).contiguous()\n']
        for source in sources:
            for policy in POLICIES:
                torch._dynamo.reset()
                self.checked(program(source), program(source, torch), args, refs, policy)

    def test_dynamic_rectangles_singletons_and_consumers(self):
        rng = np.random.default_rng(198504)
        for text in ('s(x)', 'm.relu(s(x).contiguous())', 'm.neg(s(x).contiguous())',
                     'm.add(s(x).contiguous(), s(x).contiguous())',
                     'm.multiply(s(x).contiguous(), .5)', 's(m.relu(x.contiguous()))'):
            for policy in POLICIES:
                torch._dynamo.reset()
                fn, ref = expression(text), expression(text, module=torch)
                f, cache = compile_with_cache(fn, *policy)
                reference = torch.compile(ref, backend='eager', fullgraph=policy[0], dynamic=policy[1])
                for rows, cols in ((1, 1), (3, 7), (1, 7), (3, 1), (0, 1), (1, 0), (1, 1)):
                    data = rng.integers(-31, 32, size=(3, 7)).astype(np.float32) * .125
                    x = native.tensor(data).to('cuda:0')[:rows][:, :cols]
                    y = torch.tensor(data, device='cuda:0')[:rows][:, :cols]
                    for _ in range(2):
                        with ExitStack() as stack:
                            if policy[1] and cache.graphs:
                                stack.enter_context(patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')))
                            self.compare(call_without_python(f, {fn.__code__}, x), reference(y))
                        self.compare(x, y)
                self.assertEqual(len(cache.graphs), 1 if policy[1] else 6)
        # A formerly contiguous singleton t becomes strided as rows grow.
        x = native.ones((3, 7)).to('cuda:0')
        for text in ('m.neg(s(x))', 'm.relu(s(x))', 's(x) + s(x)', 'm.mul(s(x), 2)',
                     's(x).sum(1)', 's(x).matmul(x)', 's(x).view(-1)'):
            fn = expression(text)
            f, cache = compile_with_cache(fn, dynamic=True)
            f(x[:1])
            with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')), \
                 patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native work')):
                self.exact_error(RuntimeError if '.view(' in text else trace.CompileTraceUnsupportedError,
                                 lambda: f(x))
            self.assertEqual(len(cache.graphs), 1)
            self.compare(f(x[:1]), expression(text, module=torch)(torch.ones((1, 7), device='cuda:0')))

    def test_used_field_guards_canonical_substitutions_and_recovery(self):
        x, tx = native.tensor([[-2., 3.]]).to('cuda:0'), torch.tensor([[-2., 3.]], device='cuda:0')
        sources = [f'def program(x):\n    return {text}\n' for text in
                   ('m.squeeze(x)', 'm.relu(x)', 'm.add(x, x)', 'm.neg(x)', 'm.mul(x, 2)',
                    'm.matmul(x, x.t().contiguous())', 's(x)')]
        sources += ['def helper(x):\n    return s(x)\ndef program(x):\n    return m.relu(helper(x).contiguous())\n']
        for source in sources:
            fn, ref = program(source), program(source, torch)
            for policy in POLICIES:
                f, cache = compile_with_cache(fn, *policy, limit=1)
                self.compare(f(x), ref(tx))
                graph = next(iter(cache.graphs.values()))
                for deleted in (False, True):
                    with ExitStack() as stack:
                        for module in (native, native._C):
                            stack.enter_context(patch.object(module, 't', Hostile()))
                            if deleted: del module.t
                            stack.enter_context(patch.object(module, '__getattr__', Hostile.fail, create=True))
                        with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('unused field cache miss')):
                            self.compare(f(x), ref(tx))
                        cold, _ = compile_with_cache(fn, *policy, limit=1)
                        self.compare(cold(x), ref(tx))
                        self.compare(graph.forward(x), ref(tx))
                self.assertEqual(len(cache.graphs), 1)
        sources = [('def program(x):\n    return m.t(x)\n', 't'),
                   ('def program(x):\n    return s(x)\n', 's'),
                   ('def helper(x):\n    return m.t(x)\ndef program(x):\n    return m.relu(helper(x).contiguous())\n', 't')]
        for source, field in sources:
            for policy in POLICIES:
                fn, ref = program(source), program(source, torch)
                bindings = vars(native) if field == 't' else fn.__globals__
                f, cache = compile_with_cache(fn, *policy)
                self.compare(f(x), ref(tx))
                for replacement, expected in ((native.neg, -tx), (native.relu, tx.relu()), (native.squeeze, tx.squeeze())):
                    if 'helper' in source: expected = expected.relu()
                    with patch.dict(bindings, {field: replacement}):
                        self.compare(f(x), expected)
                self.assertEqual(len(cache.graphs), 4)
                limited, one = compile_with_cache(fn, *policy, limit=1); limited(x)
                graph = next(iter(one.graphs.values()))
                for value in (Hostile(), native.mul, native.neg, property(Hostile.fail)):
                    with patch.dict(bindings, {field: value}):
                        self.exact_error(trace.CompileTraceUnsupportedError, lambda: limited(x))
                        # A lowered graph retains its captured operation semantics.
                        self.compare(graph.forward(x), ref(tx))
                    self.assertEqual(len(one.graphs), 1)
                with patch.dict(bindings, {field: Hostile()}):
                    failed, empty = compile_with_cache(fn, *policy)
                    for _ in range(2):
                        self.exact_error(trace.CompileTraceUnsupportedError, lambda: failed(x))
                        self.assertEqual(empty.graphs, {})
                self.compare(failed(x), ref(tx))
                saved = bindings.pop(field)
                try:
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: limited(x))
                finally:
                    bindings[field] = saved
                self.compare(limited(x), ref(tx))
                self.assertEqual(len(one.graphs), 1)
        # Recognize the operation's identity even under another canonical name.
        for text, field in (('m.neg(x)', 'neg'), ('m.relu(x)', 'relu')):
            f, cache = compile_with_cache(expression(text))
            f(x)
            with patch.object(native, field, genuine_t):
                out = f(x)
                self.compare(out, tx.t())
                self.assertEqual(out.data_ptr(), x.data_ptr())
            self.assertEqual(len(cache.graphs), 2)
        fn = expression('s(x)')
        other = types.FunctionType(fn.__code__, {**fn.__globals__, 's': native.neg})
        a, _ = compile_with_cache(fn); b, _ = compile_with_cache(other)
        self.compare(a(x), tx.t()); self.compare(b(x), -tx)
        fn.__globals__['s'] = Hostile()
        self.exact_error(trace.CompileTraceUnsupportedError, lambda: a(x))
        self.compare(b(x), -tx)
        fn = expression('m.t(x)')
        f, cache = compile_with_cache(fn, limit=1); f(x)
        fake = types.ModuleType('torch_rs'); fake.__dict__.update(vars(native))
        for module in (fake, types.SimpleNamespace(t=genuine_t), Hostile()):
            with patch.dict(fn.__globals__, {'m': module}):
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
        self.compare(f(x), tx.t())
        self.assertEqual(len(cache.graphs), 1)

    def test_shared_dispatch_arity_and_tensor_globals(self):
        x = native.tensor([[1., -2., 3.]]).to('cuda:0')
        tx = torch.tensor([[1., -2., 3.]], device='cuda:0')
        for policy in POLICIES:
            for text, field in (('m.t(x, x)', 't'), ('s(x, x)', 's')):
                fn = expression(text)
                bindings = vars(native) if field == 't' else fn.__globals__
                f, cache = compile_with_cache(fn, *policy, limit=1)
                for _ in range(2):
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
                    self.assertEqual(cache.graphs, {})
                with patch.dict(bindings, {field: native.add}):
                    for _ in range(2): self.compare(call_without_python(f, {fn.__code__}, x), tx + tx)
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
                self.assertEqual(len(cache.graphs), 1)
            fn = program('def helper(x):\n    return m.t(bias)\ndef program(x):\n    return helper(x), s(x)\n')
            fn.__globals__['bias'] = x
            f, cache = compile_with_cache(fn, *policy)
            codes = {fn.__code__, fn.__globals__['helper'].__code__}
            for value in (2., -3., 2.):
                bias = native.full((1, 3), value).to('cuda:0')
                fn.__globals__['bias'] = bias
                for _ in range(2):
                    a, b = call_without_python(f, codes, x)
                    self.compare(a, torch.full((1, 3), value, device='cuda:0').t())
                    self.compare(b, tx.t())
                    self.assertEqual(a.data_ptr(), bias.data_ptr())
            count = len(cache.graphs)
            fn.__globals__['bias'] = x.cpu()
            with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native work')):
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
            self.assertEqual(len(cache.graphs), count)
            fn.__globals__['bias'] = x
            self.compare(f(x)[0], tx.t())

    def test_whole_graph_validation_and_failure_cache_cleanliness(self):
        x = native.ones((1, 7)).to('cuda:0')
        fn = expression('s(m.relu(s(x).contiguous()))', 2)
        for policy in POLICIES:
            f, cache = compile_with_cache(fn, *policy)
            f(x, x); key, graph = next(iter(cache.graphs.items()))
            node = graph.operations[-1]
            malformed = [replace(graph, output='missing'), replace(graph, output_metadata=None)]
            malformed += [replace(graph, operations=(*graph.operations[:-1], replace(node, **change)))
                          for change in ({'inputs': ('missing',)}, {'inputs': ('arg0', 'arg0')},
                                         {'scalar': 1}, {'shape': ()}, {'axes': (0, 0)},
                                         {'target': 'flatten'}, {'metadata': None},
                                         {'metadata': replace(node.metadata, requires_grad=True)},
                                         {'metadata': replace(node.metadata, storage_offset=1)},
                                         {'metadata': replace(node.metadata, stride=(2,))},
                                         {'metadata': replace(node.metadata, device='cuda:1')})]
            with ExitStack() as stack:
                for owner, name in ((trace, '_execute_operation'), (trace._native, '_compile_trace_cuda_graph'),
                                    (trace._native, '_compile_trace_unary'), (trace._native, '_compile_trace_binary')):
                    stack.enter_context(patch.object(owner, name, side_effect=AssertionError('execution before validation')))
                for bad in malformed:
                    cache.graphs[key] = bad
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x, x))
                cache.graphs[key] = graph
                for bad in (x.cpu(), native.ones((1, 1, 7)).to('cuda:0')):
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(bad, x))
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x, x.cpu()))
                self.exact_error(TypeError, lambda: f(x, Hostile()))
            self.assertEqual(len(cache.graphs), 1)
            self.compare(f(x, x), expression('s(m.relu(s(x).contiguous()))', module=torch)(torch.ones((1, 7), device='cuda:0')))
        f, cache = compile_with_cache(fn)
        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=RuntimeError('launch failed')):
            self.exact_error(RuntimeError, lambda: f(x, x))
        self.assertEqual(cache.graphs, {})
        self.compare(f(x, x), expression('s(m.relu(s(x).contiguous()))', module=torch)(torch.ones((1, 7), device='cuda:0')))
        # Late output declarations must also be checked on every repeated alias.
        fn = program(NESTED); f, cache = compile_with_cache(fn)
        f(x); key, graph = next(iter(cache.graphs.items()))
        a, first, second, last = graph.output_metadata.elements
        leaf, pair = second.elements
        bad = replace(second, elements=(replace(leaf, stride=(True,)), pair))
        cache.graphs[key] = replace(graph, output_metadata=replace(graph.output_metadata, elements=(a, first, bad, last)))
        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native work')):
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
        cache.graphs[key] = graph
        self.compare(f(x)[3], torch.ones((1, 7), device='cuda:0').t())

    def test_dynamic_consumer_declarations_are_not_runtime_declarations(self):
        x = native.ones((3, 7)).to('cuda:0')
        for text in ('m.neg(s(x).contiguous())', 'm.relu(s(x).contiguous())',
                     'm.add(s(x).contiguous(), s(x).contiguous())'):
            fn = expression(text); f, cache = compile_with_cache(fn, dynamic=True)
            f(x[:1]); key, graph = next(iter(cache.graphs.items()))
            op = graph.operations[-1]
            current = replace(op.metadata, shape=(7, 3), stride=(3, 1))
            for bad, error in (
                (replace(graph, operations=(*graph.operations[:-1], replace(op, metadata=current))),
                 ValueError),
                (replace(graph, output_metadata=current), ValueError),
            ):
                cache.graphs[key] = bad
                with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')), \
                     patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native work')):
                    self.exact_error(error, lambda: f(x))
            cache.graphs[key] = graph
            self.compare(f(x), expression(text, module=torch)(torch.ones((3, 7), device='cuda:0')))

    def test_reference_import_and_original_helper_bodies_are_not_needed(self):
        source = """
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('reference import')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
from torch_rs import t as s
def helper(x): return m.t(x)
def f(x):
    a = s(helper(x))
    return a, a, s(x)
from torch_rs import _compile_trace as t
def reject(*args): raise AssertionError('Python node')
t._execute_operation = reject
def profile(frame, event, arg):
    if event == 'call' and frame.f_code in (f.__code__, helper.__code__):
        raise AssertionError('original body')
sys.setprofile(profile)
g = m.compile(f, backend='eager', fullgraph=True)
for value in (-2., 3.):
    x = m.full((1, 7), value).to('cuda:0')
    a, b, c = g(x)
    assert a is b and a is not c and a is not x
    assert a.data_ptr() == c.data_ptr() == x.data_ptr()
    assert a.cpu().tolist() == [[value] * 7]
assert 'torch' not in sys.modules
"""
        result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class ModuleTDeviceTests(ExactErrors, Comparison, unittest.TestCase):
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
                f, cache = compile_with_cache(expression('m.t(x)', 2), *policy)
                for current, target in ((1, 0), (0, 1), (1, 0)):
                    torch.cuda.set_device(current)
                    torch.empty(1, device=f'cuda:{current}')
                    before = context()
                    x = native.full((1, 7), 2.).to(f'cuda:{target}')
                    out = f(x, x)
                    self.assertEqual(context(), before)
                    self.assertEqual(torch.cuda.current_device(), current)
                    self.assertEqual(out.data_ptr(), x.data_ptr())
                    expected = torch.full((1, 7), 2., device=f'cuda:{target}').t()
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
