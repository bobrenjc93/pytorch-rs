"""Trusted native ReLU spellings; non-scoring CUDA/frontend regressions."""
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
from torch_rs import relu as genuine_relu
from torch_rs import _compile_bytecode as bytecode, _compile_trace as trace
from tests.test_compile_cuda_boundary import compile_with_cache
from tests.test_compile_cuda_module_arithmetic import ExactErrors, Hostile
from tests.test_compile_cuda_mul_scalar import POLICIES, call_without_python
from tests.test_cuda_add import Comparison, available, runtime, torch, upload
from tests.test_cuda_relu import EDGE_BITS, download_bits, upload_bits


def program(source, module=native):
    namespace = {'__name__': __name__, 'm': module,
                 'r': genuine_relu if module is native else module.relu}
    exec(source, namespace)
    return namespace['program']


def expression(text, arity=1, module=native):
    return program(f'def program({"x" if arity == 1 else "x, y"}):\n    return {text}\n', module)


_STARTUP = r'''
import sys
import types
import torch_rs as m
from torch_rs import relu as good
mode, mutation = sys.argv[1:]
assert 'torch_rs._compile_bytecode' not in sys.modules
assert 'torch_rs._compile_trace' not in sys.modules
assert good is m._C.relu
assert not hasattr(m._C._VariableFunctionsClass, 'relu')
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
        callbacks.append('call/equality/conversion')
        raise AssertionError('callback')
    __call__ = __eq__ = __hash__ = __bool__ = __float__ = __index__ = fail
bad = fake if mutation == 'fake' else Hostile()
for module in (m, m._C):
    if mutation == 'deleted':
        del module.relu
    else:
        module.relu = bad
bad_alias = bad

def direct(x): return good(x)
def module_call(x): return m.relu(x)
def counterfeit(x): return bad_alias(x)
def control(x): return m.add(m.neg(x), x)

def reject(callback):
    try:
        callback()
    except Exception as error:
        from torch_rs._compile_trace import CompileTraceUnsupportedError
        assert type(error) is CompileTraceUnsupportedError, type(error)
    else:
        raise AssertionError('accepted counterfeit')
    assert not callbacks, callbacks

assert 'torch_rs._compile_bytecode' not in sys.modules
if mode == 'frontend':
    from torch_rs import _compile_bytecode as b, _compile_trace as t
    meta = (t.CompileTraceTensorMetadata((2,), (1,), t.float32, 'cuda:0', False, 0),)
    for fn in (module_call, counterfeit):
        for _ in range(2): reject(lambda: b.lower_compile_graph(fn, meta))
    assert b._builtin_target(good).target == 'relu'
    assert b._builtin_target(bad) is None
    assert [op.target for op in b.lower_compile_graph(direct, meta).operations] == ['relu']
    assert [op.target for op in b.lower_compile_graph(control, meta).operations] == ['neg', 'add']
    m.relu = m._C.relu = good
    assert b.lower_compile_graph(module_call, meta).operations[0].target == 'relu'
else:
    from torch_rs import _compiler_state as s
    x = m.tensor([-2., 3.]).to('cuda:0')
    for policy in ((True, None), (True, False), (True, True), (False, None)):
        if mutation == 'deleted':
            vars(m).pop('relu', None); vars(m._C).pop('relu', None)
        else:
            m.relu = m._C.relu = bad
        old = set(s.native_eager_compile_caches)
        f = m.compile(module_call, backend='eager', fullgraph=policy[0], dynamic=policy[1], recompile_limit=1)
        cache, = set(s.native_eager_compile_caches) - old
        for _ in range(2):
            reject(lambda: f(x))
            assert not cache.graphs
        for fn, expected in ((direct, [0., 3.]), (control, [0., 0.])):
            g = m.compile(fn, backend='eager', fullgraph=policy[0], dynamic=policy[1], recompile_limit=1)
            for _ in range(2): assert g(x).cpu().tolist() == expected
        g = m.compile(counterfeit, backend='eager', fullgraph=policy[0], dynamic=policy[1])
        for _ in range(2): reject(lambda: g(x))
        m.relu = m._C.relu = good
        outputs = [f(x), f(x)]
        assert all(v.cpu().tolist() == [0., 3.] for v in outputs)
        assert outputs[0].data_ptr() != outputs[1].data_ptr()
        assert len(cache.graphs) == 1
        assert x.cpu().tolist() == [-2., 3.]
assert not callbacks, callbacks
'''


def startup(test, mode):
    for mutation in ('fake', 'hostile', 'deleted'):
        with test.subTest(mutation=mutation):
            result = subprocess.run([sys.executable, '-c', _STARTUP, mode, mutation],
                                    capture_output=True, text=True, timeout=60)
            test.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ModuleReluFrontendTests(ExactErrors, unittest.TestCase):
    def test_startup_before_frontend_import(self):
        startup(self, 'frontend')

    def test_identity_registry_arity_and_unchanged_cpu_boundary(self):
        meta = trace.CompileTraceTensorMetadata((3, 5), (5, 1), trace.float32, 'cuda:0', False, 0)
        self.assertEqual(bytecode._builtin_target(genuine_relu).target, 'relu')
        self.assertEqual(bytecode._builtin_target(genuine_relu).arity, 1)
        for text in ('m.relu(x)', 'r(x)', 'm.relu(r(x))', 'r(x + x)'):
            fn = expression(text)
            graph = bytecode.lower_compile_graph(fn, (meta,))
            self.assertEqual(graph.operations[-1].target, 'relu')
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.lower_compile_graph(fn, (replace(meta, device='cpu'),)))
        for text in ('m.relu()', 'r(x, x)', 'm.relu(1)', 'r(input=x)', 'm.relu(x, out=None)',
                     'm.relu(x, inplace=False)', 'm.nn.functional.relu(x)', 'x.relu_()'):
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.lower_compile_graph(expression(text), (meta,)))
        for bad in (replace(meta, requires_grad=True), replace(meta, stride=(1, 3)),
                    replace(meta, dtype=trace.CompileTraceDType('torch.float64'))):
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.lower_compile_graph(expression('m.relu(x)'), (bad,)))
        # CPU method capture/backward retains its already-supported semantics.
        fn = expression('x.relu()')
        x = native.tensor([-2., 3.], requires_grad=True)
        out = native.compile(fn, backend='eager', fullgraph=True)(x)
        self.assertEqual(out.tolist(), [0., 3.])
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [0., 1.])

    def test_hostile_namespaces_and_deleted_fields_without_callbacks(self):
        meta = (trace.CompileTraceTensorMetadata((3,), (1,), trace.float32, 'cuda:0', False, 0),)
        fake = types.ModuleType('torch_rs'); fake.__dict__.update(vars(native))
        class ModuleSubclass(types.ModuleType):
            __getattribute__ = Hostile.fail
        for value in (fake, ModuleSubclass('torch_rs'), Hostile(), types.SimpleNamespace(relu=genuine_relu)):
            fn = expression('m.relu(x)'); fn.__globals__['m'] = value
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.prepare_compile_cache_request(fn, meta))
        fn = expression('r(x)')
        for value in (Hostile(), property(Hostile.fail), native.abs):
            fn.__globals__['r'] = value
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.prepare_compile_cache_request(fn, meta))
        fn = expression('m.relu(x)')
        with patch.object(native, '__getattr__', Hostile.fail, create=True):
            saved = vars(native).pop('relu')
            try:
                self.exact_error(trace.CompileTraceUnsupportedError,
                                 lambda: bytecode.prepare_compile_cache_request(fn, meta))
            finally:
                native.relu = saved


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class ModuleReluCudaTests(ExactErrors, Comparison, unittest.TestCase):
    def test_startup_before_first_compile(self):
        startup(self, 'cuda')

    def checked(self, fn, ref_fn, args, refs, policy, backend='eager'):
        compiled, cache = compile_with_cache(fn, *policy)
        reference = torch.compile(ref_fn, backend=backend, fullgraph=policy[0], dynamic=policy[1])
        codes = {v.__code__ for v in fn.__globals__.values() if type(v) is types.FunctionType}
        retained = []
        for _ in range(2):
            with patch.object(trace, '_execute_operation', side_effect=AssertionError('per-node Python')):
                result = call_without_python(compiled, codes, *args)
            expected = ref_fn(*refs)
            self.compare(result, expected)
            torch.testing.assert_close(reference(*refs), expected, rtol=0, atol=0)
            for x, tx in zip(args, refs):
                self.compare(x, tx)
                self.assertIsNot(result, x)
                if result.numel() and x.numel(): self.assertNotEqual(result.data_ptr(), x.data_ptr())
            retained.append(result)
        self.assertIsNot(retained[0], retained[1])
        if retained[0].numel(): self.assertNotEqual(retained[0].data_ptr(), retained[1].data_ptr())
        self.assertEqual(len(cache.graphs), 1)
        with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('warm cache miss')):
            self.compare(compiled(*args), ref_fn(*refs))
        graph = next(iter(cache.graphs.values()))
        self.compare(graph.forward(*args), ref_fn(*refs))
        return compiled, cache

    def test_seeded_spelling_matrix_inductor_all_policies(self):
        rng = np.random.default_rng(198301)
        for text, arity in (('m.relu(x)', 1), ('r(x)', 1), ('m.relu(x + y)', 2),
                            ('r(x + y)', 2), ('-m.relu(x)', 1), ('-r(x)', 1),
                            ('m.relu(m.relu(x))', 1), ('r(r(x))', 1)):
            for shape in ((), (7,), (3, 5), (0,)):
                values = [rng.integers(-31, 32, size=int(np.prod(shape))).astype(np.float32) * .125 for _ in range(arity)]
                args = [upload(native, v, shape) for v in values]
                refs = [upload(torch, v, shape) for v in values]
                for policy in POLICIES:
                    with self.subTest(text=text, shape=shape, policy=policy):
                        torch._dynamo.reset()
                        self.checked(expression(text, arity), expression(text, arity, torch), args, refs, policy, 'inductor')

    def test_generated_compositions_layouts_helpers_and_ieee(self):
        rng = np.random.default_rng(198305)
        sources = []
        for length in (3, 11, 23):
            lines, values = ['def program(x, y):'], ['x', 'y']
            for i in range(length):
                a, b = rng.choice(values, 2)
                form = rng.choice([f'm.relu({a})', f'r({a})', f'm.add({a}, {b})', f'm.neg({a})', f'm.mul({a}, .5)'])
                lines.append(f'    v{i} = {form}'); values.append(f'v{i}')
            sources.append('\n'.join(lines + [f'    return r({values[-1]})']) + '\n')
        sources.append('def helper(x):\n    return r(x)\ndef program(x, y):\n    return m.relu(helper(x) + y)\n')
        views = (lambda x: x, lambda x: x[1:258], lambda x: x.select(0, 1),
                 lambda x: x[420:].reshape(2, 0, 3),
                 lambda x: x.reshape(1, 7, 60).transpose(0, 1), lambda x: x[1:2])
        values = rng.integers(-31, 32, size=420).astype(np.float32) * .125
        for source in sources:
            for view in views:
                args = [view(upload(native, values, (420,)))] * 2
                refs = [view(upload(torch, values, (420,)))] * 2
                for policy in POLICIES:
                    self.checked(program(source), program(source, torch), args, refs, policy)
        bits = np.tile(EDGE_BITS, 12)[:257]
        for text in ('m.relu(x)', 'r(x)'):
            for policy in POLICIES:
                x, tx = [upload_bits(m, bits) for m in (native, torch)]
                fn = expression(text); f, _ = compile_with_cache(fn, *policy)
                for _ in range(2):
                    np.testing.assert_array_equal(download_bits(f(x)), download_bits(tx.relu()))
                np.testing.assert_array_equal(download_bits(x), bits)

    def test_compositions_and_dynamic_squeeze_rank_changes(self):
        texts = ('m.relu(m.add(x, y))', 'm.add(r(x), r(y))', 'm.neg(r(x))', 'r(m.neg(x))',
                 'm.mul(r(x), -.5)', 'r(m.mul(x, -.5))', 'r(x).view(-1)', 'r(x.view(-1))',
                 'r(x).reshape(5, 3)', 'r(x.reshape(5, 3))', 'r(x).sum(1)', 'r(x.sum(1))',
                 'r(x).transpose(0, 1).contiguous()', 'r(x.transpose(0, 1).contiguous())',
                 'r(m.matmul(x, y.t().contiguous()))', 'm.matmul(r(x), r(y).t().contiguous())')
        values = np.arange(15, dtype=np.float32) * .125 - 1
        args = [upload(native, values, (3, 5))] * 2
        refs = [upload(torch, values, (3, 5))] * 2
        for text in texts:
            for policy in POLICIES:
                self.checked(expression(text, 2), expression(text, 2, torch), args, refs, policy)
        for text in ('r(x.squeeze())', 'r(x).squeeze()', 'm.relu(r(x).squeeze())'):
            for policy in POLICIES:
                fn, ref = expression(text), expression(text, module=torch)
                f, cache = compile_with_cache(fn, *policy)
                for shape in ((1, 1), (5, 1), (0, 1), (1, 1)):
                    for value in (-2., 3.):
                        x = native.full(shape, value).to('cuda:0')
                        tx = torch.full(shape, value, device='cuda:0')
                        self.compare(call_without_python(f, {fn.__code__}, x), ref(tx))
                self.assertEqual(len(cache.graphs), 1 if policy[1] else 3)

    def test_used_field_guards_substitutions_and_recovery(self):
        x = native.tensor([-2., 3.]).to('cuda:0')
        tx = torch.tensor([-2., 3.], device='cuda:0')
        for text in ('m.mul(x, 2)', 'm.add(x, x)', 'm.neg(x)', 'm.matmul(x.reshape(1, 2), x.reshape(2, 1))', 'r(x)'):
            fn, ref = expression(text), expression(text, module=torch)
            for policy in POLICIES:
                f, cache = compile_with_cache(fn, *policy, limit=1)
                self.compare(f(x), ref(tx))
                for deleted in (False, True):
                    with ExitStack() as stack:
                        for module in (native, native._C):
                            stack.enter_context(patch.object(module, 'relu', Hostile()))
                            if deleted: del module.relu
                            stack.enter_context(patch.object(module, '__getattr__', Hostile.fail, create=True))
                        with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('unused field cache miss')):
                            self.compare(f(x), ref(tx))
                        cold, _ = compile_with_cache(fn, *policy, limit=1)
                        self.compare(cold(x), ref(tx))
                self.assertEqual(len(cache.graphs), 1)
        for text, field in (('m.relu(x)', 'relu'), ('r(x)', 'r')):
            fn = expression(text)
            bindings = vars(native) if field == 'relu' else fn.__globals__
            f, cache = compile_with_cache(fn)
            self.compare(f(x), tx.relu())
            with patch.dict(bindings, {field: native.neg}):
                self.compare(f(x), -tx)
            self.assertEqual(len(cache.graphs), 2)
            limited, one = compile_with_cache(fn, limit=1); limited(x)
            for value in (Hostile(), native.mul, native.neg, property(Hostile.fail)):
                with patch.dict(bindings, {field: value}):
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: limited(x))
                self.assertEqual(len(one.graphs), 1)
            with patch.dict(bindings, {field: Hostile()}):
                failed, empty = compile_with_cache(fn)
                for _ in range(2):
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: failed(x))
                    self.assertEqual(empty.graphs, {})
            self.compare(failed(x), tx.relu())
            saved = bindings.pop(field)
            try:
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: limited(x))
                self.assertEqual(len(one.graphs), 1)
            finally:
                bindings[field] = saved
            self.compare(limited(x), tx.relu())
        fn = expression('m.relu(x)')
        f, cache = compile_with_cache(fn, limit=1)
        self.compare(f(x), tx.relu())
        fake_module = types.ModuleType('torch_rs')
        fake_module.__dict__.update(vars(native))
        for module in (fake_module, types.SimpleNamespace(relu=genuine_relu), Hostile()):
            with patch.dict(fn.__globals__, {'m': module}):
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
        self.compare(f(x), tx.relu())
        self.assertEqual(len(cache.graphs), 1)
        # A canonical unary substitution in either direction follows identity.
        fn = expression('m.neg(x)')
        f, cache = compile_with_cache(fn)
        self.compare(f(x), -tx)
        with patch.object(native, 'neg', genuine_relu):
            self.compare(f(x), tx.relu())
        self.assertEqual(len(cache.graphs), 2)
        # Identical code with separate global dictionaries never shares bindings.
        first = expression('r(x)')
        second = types.FunctionType(first.__code__, {**first.__globals__, 'r': native.neg})
        a, _ = compile_with_cache(first); b, _ = compile_with_cache(second)
        self.compare(a(x), tx.relu()); self.compare(b(x), -tx)
        first.__globals__['r'] = Hostile()
        self.exact_error(trace.CompileTraceUnsupportedError, lambda: a(x))
        self.compare(b(x), -tx)

    def test_nested_outputs_lifetime_dynamic_and_changed_inputs(self):
        source = 'def program(x, y):\n    a = r(x + y)\n    shared = [a, x]\n    return shared, (a, shared, m.relu(y))\n'
        for policy in POLICIES:
            fn = program(source); f, cache = compile_with_cache(fn, *policy)
            retained = []
            for size, value in ((5, -2.), (11, 3.), (257, -7.), (5, 13.)):
                x = native.full((size,), value).to('cuda:0')
                out = call_without_python(f, {fn.__code__}, x, x)
                self.assertIs(out[0], out[1][1]); self.assertIs(out[0][0], out[1][0])
                self.assertIs(out[0][1], x)
                self.assertNotEqual(out[0][0].data_ptr(), x.data_ptr())
                self.assertEqual(x.cpu().tolist(), [value] * size)
                retained.append((out[0][0], size, max(0., value * 2)))
            self.assertEqual(len(cache.graphs), 1 if policy[1] else 3)
            del x, out, f, cache
            gc.collect()
            for output, size, value in retained: self.assertEqual(output.cpu().tolist(), [value] * size)

    def test_prevalidation_unused_inputs_and_execution_errors(self):
        x = native.ones((3, 5)).to('cuda:0')
        for text in ('m.relu(x)', 'r(m.relu(x) + y)'):
            fn = expression(text, 2); f, cache = compile_with_cache(fn, dynamic=True)
            f(x, x); graph = next(iter(cache.graphs.values()))
            node = graph.operations[-1]
            malformed = [replace(graph, output='missing'), replace(graph, output_metadata=None)]
            malformed += [replace(graph, operations=(*graph.operations[:-1], replace(node, **change)))
                          for change in ({'inputs': ('missing',)}, {'scalar': 1}, {'target': 'abs'},
                                         {'metadata': replace(node.metadata, requires_grad=True)},
                                         {'metadata': replace(node.metadata, device='cuda:1')})]
            with ExitStack() as stack:
                for owner, name in ((trace, '_execute_operation'), (trace._native, '_compile_trace_cuda_graph'),
                                    (trace._native, '_compile_trace_unary'), (trace._native, '_compile_trace_binary')):
                    stack.enter_context(patch.object(owner, name, side_effect=AssertionError('execution before validation')))
                for bad in malformed:
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: bad.forward(x, x))
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x, x.cpu()))
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x.t(), x.t()))
                self.exact_error(TypeError, lambda: f(x, Hostile()))
            self.assertEqual(len(cache.graphs), 1)
        fn = expression('r(x + x)'); f, cache = compile_with_cache(fn)
        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=RuntimeError('launch failed')):
            self.exact_error(RuntimeError, lambda: f(x))
        self.assertEqual(cache.graphs, {})
        self.assertEqual(f(x).cpu().tolist(), [[2.] * 5] * 3)

    def test_no_reference_import_or_python_body(self):
        source = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('reference import')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
from torch_rs import relu as r
def helper(x): return r(x)
def f(x): return m.relu(helper(x) + x)
from torch_rs import _compile_trace as t
def reject(*args): raise AssertionError('per-node Python')
t._execute_operation = reject
def profile(frame, event, arg):
    if event == 'call' and frame.f_code in (f.__code__, helper.__code__):
        raise AssertionError('original body')
sys.setprofile(profile)
g = m.compile(f, backend='eager', fullgraph=True)
for value in (-2., 3.):
    assert g(m.full((7,), value).to('cuda:0')).cpu().tolist() == [max(0., value * 2)] * 7
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class ModuleReluDeviceTests(ExactErrors, Comparison, unittest.TestCase):
    def test_restoration_success_rejection_unused_inputs_and_lifetime(self):
        lib, previous = runtime(), torch.cuda.current_device()
        try:
            for policy in POLICIES:
                f, cache = compile_with_cache(expression('r(x + y)', 2), *policy)
                unused, _ = compile_with_cache(expression('m.relu(x)', 2), *policy)
                for current, target in ((1, 0), (0, 1), (1, 0)):
                    torch.cuda.set_device(current)
                    x = native.full((3, 5), 2.).to(f'cuda:{target}')
                    out = f(x, x)
                    self.compare(out, torch.full((3, 5), 4., device=f'cuda:{target}'))
                    wrong = native.ones((3, 5)).to(f'cuda:{current}')
                    with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('launched')):
                        self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x, wrong))
                        self.exact_error(trace.CompileTraceUnsupportedError, lambda: unused(x, wrong))
                    del out, x, wrong
                    gc.collect()
                    ordinal = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
                self.assertEqual(len(cache.graphs), 2)
        finally:
            torch.cuda.set_device(previous)
