"""Bounded top-level CUDA reshape; non-scoring implementation regressions."""
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
from torch_rs import reshape as genuine_reshape
from torch_rs import _compile_bytecode as bytecode, _compile_trace as trace
from tests.test_compile_cuda_boundary import compile_with_cache
from tests.test_compile_cuda_module_arithmetic import ExactErrors, Hostile
from tests.test_compile_cuda_module_squeeze import BitComparison
from tests.test_compile_cuda_mul_scalar import POLICIES, call_without_python
from tests.test_cuda_add import Comparison, available, runtime, torch, upload
from tests.test_cuda_contiguous import metadata, read_bits, write_bits


def program(source, module=native):
    namespace = {'__name__': __name__, 'm': module,
                 'r': genuine_reshape if module is native else module.reshape}
    exec(source, namespace)
    return namespace['program']


def expression(text, module=native, arity=1):
    return program(f'def program({"x" if arity == 1 else "x, y"}):\n    return {text}\n', module)


# Frozen baseline's complete matrix, including the two zero-credit diagnostics.
LAYOUTS = (((), ()), ((), (1,)), ((), (1, 1)), ((6,), (2, 3)),
           ((6,), (-1, 2)), ((6,), (6,)), ((2, 3), (3, 2)),
           ((2, 3), (6,)), ((2, 3), (-1,)), ((0, 3), (0, 3)),
           ((0, 3), (3, 0)), ((0, 3), (0,)))
SPELLINGS = ('m.reshape(x, {t})', 'r(x, {t})', 'm.reshape(x, {l})', 'r(x, {l})',
             'm.reshape(x + x, {t})', 'r(x + x, {t})',
             'm.relu(m.reshape(x, {t}))', 'm.relu(r(x, {t}))',
             'x.reshape({t})', 'm.transpose(x, 0, 0)', 'm.t(x)', 'm.squeeze(x)')


_STARTUP = r'''
import sys
import torch_rs as m
from torch_rs import reshape as good
mode, mutation = sys.argv[1:]
assert 'torch_rs._compile_bytecode' not in sys.modules
assert 'torch_rs._compile_trace' not in sys.modules
owner = m._C._VariableFunctionsClass
assert good is owner.reshape is m._C.reshape
try: owner.reshape = lambda *args: None
except TypeError: pass
else: raise AssertionError('mutable owner')
calls = []
def fake(*args):
    calls.append('body')
    raise AssertionError('fake body')
fake.__module__ = 'untrusted_import'
class Hostile:
    def fail(self, *args):
        calls.append('hook')
        raise AssertionError('hook')
    __call__ = __eq__ = __hash__ = __bool__ = __float__ = __index__ = __get__ = __getattribute__ = fail
bad = fake if mutation == 'fake' else Hostile()
for module in (m, m._C):
    if mutation == 'deleted': del module.reshape
    else: module.reshape = bad
# The owner export itself is writable; the retained immutable owner is trusted.
m._C._VariableFunctionsClass = bad
bad_alias = bad
def direct(x): return good(x, [-1])
def module_call(x): return m.reshape(x, (-1,))
def counterfeit(x): return bad_alias(x, (-1,))
def control(x): return m.relu(m.add(m.neg(x), x))
def profile(frame, event, arg):
    if event == 'call' and frame.f_code in (direct.__code__, module_call.__code__, counterfeit.__code__, control.__code__, fake.__code__):
        raise AssertionError('original body')
def reject(callback):
    try: callback()
    except Exception as e:
        from torch_rs._compile_trace import CompileTraceUnsupportedError
        assert type(e) is CompileTraceUnsupportedError, type(e)
    else: raise AssertionError('accepted fake')
    assert not calls
sys.setprofile(profile)
if mode == 'frontend':
    from torch_rs import _compile_bytecode as b, _compile_trace as t
    meta = (t.CompileTraceTensorMetadata((1, 2), (2, 1), t.float32, 'cuda:0', False, 0),)
    for fn in (module_call, counterfeit):
        for _ in range(2): reject(lambda: b.lower_compile_graph(fn, meta))
    assert b._builtin_target(good).target == 'reshape'
    assert b._builtin_target(bad) is None
    assert b.lower_compile_graph(direct, meta).operations[0].target == 'reshape'
    assert len(b.lower_compile_graph(control, meta).operations) == 3
    m.reshape = m._C.reshape = good
    assert b.lower_compile_graph(module_call, meta).operations[0].target == 'reshape'
else:
    from torch_rs import _compiler_state as state
    x = m.tensor([[-2., 3.]]).to('cuda:0')
    for fullgraph, dynamic in ((True, None), (True, False), (True, True), (False, None)):
        if mutation == 'deleted': vars(m).pop('reshape', None)
        else: m.reshape = bad
        previous = set(state.native_eager_compile_caches)
        f = m.compile(module_call, backend='eager', fullgraph=fullgraph, dynamic=dynamic, recompile_limit=1)
        cache, = set(state.native_eager_compile_caches) - previous
        for _ in range(2):
            reject(lambda: f(x)); assert not cache.graphs
        for fn, expected in ((direct, [-2., 3.]), (control, [[0., 0.]])):
            g = m.compile(fn, backend='eager', fullgraph=fullgraph, dynamic=dynamic, recompile_limit=1)
            for _ in range(2): assert g(x).cpu().tolist() == expected
        g = m.compile(counterfeit, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
        for _ in range(2): reject(lambda: g(x))
        m.reshape = m._C.reshape = good
        a, b = f(x), f(x)
        assert a.cpu().tolist() == b.cpu().tolist() == [-2., 3.]
        assert a is not b and a is not x
        assert a.data_ptr() == b.data_ptr() == x.data_ptr()
        assert len(cache.graphs) == 1
assert not calls
'''


def startup(test, mode):
    for mutation in ('fake', 'hostile', 'deleted'):
        with test.subTest(mutation=mutation):
            result = subprocess.run([sys.executable, '-c', _STARTUP, mode, mutation],
                                    capture_output=True, text=True, timeout=60)
            test.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ModuleReshapeFrontendTests(ExactErrors, unittest.TestCase):
    def test_startup_before_frontend_import(self):
        startup(self, 'frontend')

    def test_signature_boundaries_and_exact_error_precedence(self):
        meta = trace.CompileTraceTensorMetadata((1,), (3,), trace.float32, 'cuda:0', False, 7)
        self.assertEqual(bytecode._builtin_target(genuine_reshape), bytecode._BytecodeBuiltin('reshape', 2))
        for text in ('m.reshape(x, (1, 1))', 'r(x, [1, 1])'):
            graph = bytecode.lower_compile_graph(expression(text), (meta,))
            self.assertEqual(graph.operations[0].target, 'reshape')
            self.assertEqual(graph.output_metadata, replace(meta, shape=(1, 1), stride=(3, 3)))
        unsupported = ('m.reshape()', 'r(x)', 'r(x, 1, 1)', 'r(x, shape=(1,))',
                       'r(input=x, shape=(1,))', 'r(x, (1,), out=x)', 'm._C.reshape(x, (1,))',
                       'm.nn.functional.reshape(x, (1,))', 'r(x, x.shape)', 'r(x, (x.numel(),))',
                       'r(x, (1, True))', 'r(x, (x,))', 'r(x, (1,1,1))')
        for text in unsupported:
            with self.subTest(text=text):
                self.exact_error(trace.CompileTraceUnsupportedError,
                                 lambda: bytecode.lower_compile_graph(expression(text), (meta,)))
        # A singleton's stride does not make it noncontiguous. Use a vector
        # with multiple elements to exercise the existing arithmetic boundary.
        for text in ('r(x, [-1]).relu()', 'm.neg(r(x, [-1]))', 'r(x, [-1]) + x', 'r(x, [-1]) * 2'):
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.lower_compile_graph(expression(text), (replace(meta, shape=(7,)),)))
        for shape, error in (('1', TypeError), ('None', TypeError), ('1.', TypeError),
                             ('True', TypeError), ('(True,)', TypeError), ('[1.]', TypeError),
                             ('(x, 1.)', TypeError), ('(-2, 1.)', TypeError),
                             ('(2**63,)', TypeError), ('(-2,)', RuntimeError),
                             ('(-1, -1)', RuntimeError), ('(2,)', RuntimeError)):
            fn = expression(f'r(-x.contiguous(), {shape})')
            with self.subTest(shape=shape):
                self.exact_error(error, lambda: bytecode.lower_compile_graph(fn, (meta,)))
        for shape in ('(0, -1)', '[-1, 0]'):
            self.exact_error(RuntimeError, lambda: bytecode.lower_compile_graph(
                expression(f'r(x, {shape})'), (replace(meta, shape=(0,)),)))
        for change in ({'device': 'cpu'}, {'requires_grad': True},
                       {'shape': (1, 1, 1), 'stride': (1, 1, 1)},
                       {'dtype': trace.CompileTraceDType('torch.float64')}):
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.lower_compile_graph(expression('r(x, [-1])'), (replace(meta, **change),)))
        for policy in POLICIES:
            f, cache = compile_with_cache(expression('r(x, [-1])'), *policy)
            for _ in range(2):
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(native.ones((1,))))
                self.assertFalse(cache.graphs)

    def test_constant_lists_require_the_supported_consumer(self):
        meta = (trace.CompileTraceTensorMetadata((1,), (1,), trace.float32, 'cuda:0', False, 0),)
        for body in ('shape = [1]\n    return x',
                     'shape = [1]\n    shape = (1,)\n    return r(x, shape)',
                     'shape = [1]\n    return x.reshape(shape)',
                     'shape = [1]\n    return r(x, shape), shape',
                     'n = 1\n    shape = [n + 0]\n    return r(x, shape)',
                     'shape = [1]\n    return shape'):
            fn = program('def program(x):\n    ' + body + '\n')
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: bytecode.lower_compile_graph(fn, meta))
        fn = program('def program(x):\n    shape = [1]\n    alias = shape\n    return r(x, shape), m.reshape(x, alias)\n')
        graph = bytecode.lower_compile_graph(fn, meta)
        self.assertEqual([op.target for op in graph.operations], ['reshape', 'reshape'])

    def test_wrong_namespaces_shape_globals_and_hooks(self):
        meta = (trace.CompileTraceTensorMetadata((1,), (1,), trace.float32, 'cuda:0', False, 0),)
        fake = types.ModuleType('torch_rs'); fake.__dict__.update(vars(native))
        class ModuleSubclass(types.ModuleType):
            __getattribute__ = Hostile.fail
        for value in (fake, ModuleSubclass('torch_rs'), Hostile(), types.SimpleNamespace(reshape=genuine_reshape)):
            fn = expression('m.reshape(x, [-1])'); fn.__globals__['m'] = value
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: bytecode.lower_compile_graph(fn, meta))
        for value in (Hostile(), property(Hostile.fail), native.abs):
            fn = expression('r(x, [-1])'); fn.__globals__['r'] = value
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: bytecode.lower_compile_graph(fn, meta))
        for shape in ((1,), [1], Hostile(), (Hostile(),), np.int64(1)):
            fn = expression('r(x, shape)'); fn.__globals__['shape'] = shape
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: bytecode.lower_compile_graph(fn, meta))
        fn = expression('m.reshape(x, [-1])')
        with patch.object(native, '__getattr__', Hostile.fail, create=True):
            saved = vars(native).pop('reshape')
            try:
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: bytecode.lower_compile_graph(fn, meta))
            finally:
                native.reshape = saved


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class ModuleReshapeCudaTests(ExactErrors, BitComparison, unittest.TestCase):
    def test_startup_before_first_compile(self):
        startup(self, 'cuda')

    def test_frozen_144_cells_all_policies(self):
        rng = np.random.default_rng(198701)
        excluded = 0
        for index, template in enumerate(SPELLINGS):
            for shape, target in LAYOUTS:
                text = template.format(t=repr(target), l=repr(list(target)))
                data = np.asarray(rng.integers(-31, 32, size=shape) * .125, dtype=np.float32)
                x, y = upload(native, data.reshape(-1), shape), upload(torch, data.reshape(-1), shape)
                fn, ref = expression(text), expression(text, torch)
                eager = ref(y); self.compare(fn(x), eager)
                ineligible = index in (4, 5) and shape == (0, 3) and target == (3, 0)
                excluded += ineligible
                for policy in POLICIES:
                    with self.subTest(text=text, shape=shape, policy=policy):
                        torch._dynamo.reset()
                        f, cache = compile_with_cache(fn, *policy)
                        reference = torch.compile(ref, backend='inductor', fullgraph=policy[0], dynamic=policy[1])
                        retained = []
                        for warm in (False, True):
                            expected = reference(y)
                            if ineligible:
                                self.assertEqual(expected.stride(), (0, 0))
                                self.assertEqual(eager.stride(), (1, 1))
                                a, b = metadata(expected), metadata(eager)
                                self.assertEqual(a[:1] + a[2:], b[:1] + b[2:])
                            else: self.assertEqual(metadata(expected), metadata(eager))
                            torch.testing.assert_close(expected, eager, rtol=0, atol=0)
                            with ExitStack() as stack:
                                stack.enter_context(patch.object(trace, '_execute_operation', side_effect=AssertionError('Python node')))
                                if warm: stack.enter_context(patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')))
                                out = call_without_python(f, {fn.__code__}, x)
                            self.compare(out, eager)
                            self.assertIsNot(out, x)
                            if index in (0, 1, 2, 3, 8, 9, 10, 11): self.assertEqual(out.data_ptr(), x.data_ptr())
                            retained.append(out)
                        self.assertIsNot(*retained)
                        self.assertEqual(len(cache.graphs), 1)
                        self.compare(next(iter(cache.graphs.values())).forward(x), eager)
                        self.compare(x, y)
        self.assertEqual(excluded, 2)  # Retained diagnostics, zero strict-parity credit.

    def test_layouts_bits_alias_pack_identities_and_lifetime(self):
        bits = np.array([0, 0x80000000, 0x7f800001, 0xff800001, 0x7fc12345, 0xffc54321,
                         1, 0x80000001, 0x7f800000, 0xff800000, 0x3f800000, 0xbf800000], dtype=np.uint32)
        layouts = (lambda x: x, lambda x: x.t(), lambda x: x[1:][:, 1:3],
                   lambda x: x.select(1, 1), lambda x: x.select(0, 1).select(0, 1),
                   lambda x: x[:1], lambda x: x[:0], lambda x: x[:, :0])
        for policy in POLICIES:
            for layout in layouts:
                for target in ((-1,), (1, -1)):
                    a, b = native.zeros((3, 4)).to('cuda:0'), torch.zeros((3, 4), device='cuda:0')
                    write_bits(a, bits); write_bits(b, bits)
                    x, y = layout(a), layout(b)
                    source = f'''def program(x):
    a = m.reshape(x, {target!r})
    b = r(x, {list(target)!r})
    shared = [a, a, b]
    return x, shared, shared
'''
                    fn, ref = program(source), program(source, torch)
                    f, cache = compile_with_cache(fn, *policy)
                    previous = None
                    for _ in range(2):
                        out, expected = call_without_python(f, {fn.__code__}, x), ref(y)
                        self.assertIs(out[0], x); self.assertIs(out[1], out[2])
                        self.assertIs(out[1][0], out[1][1]); self.assertIsNot(out[1][0], out[1][2])
                        self.assertIsNot(out[1][0], previous); self.assertIsNot(out[1][0], x)
                        for actual, value in zip(out[1], expected[1]):
                            self.compare(actual, value)
                            if x.numel(): self.assertEqual(actual.data_ptr() == x.data_ptr(), value.data_ptr() == y.data_ptr())
                        previous = out[1][0]
                    write_bits(a, bits[::-1].copy()); write_bits(b, bits[::-1].copy())
                    for actual, value in zip(out[1], expected[1]): self.compare(actual, value)
                    if previous.numel():
                        u, v = previous, expected[1][0]
                        while u.dim(): u, v = u.select(0, 0), v.select(0, 0)
                        write_bits(u, np.array([0x80000000], dtype=np.uint32)); write_bits(v, np.array([0x80000000], dtype=np.uint32))
                        self.compare(a, b); self.compare(out[1][2], expected[1][2])
                        del u
                    del x, a, out, f, cache
                    gc.collect()
                    self.compare(previous, expected[1][0])

    def test_dynamic_specializations_helpers_and_consumers(self):
        texts = ('r(x, [-1])', 'm.relu(r(x, [-1]).contiguous())', 'm.neg(r(x, [-1]).contiguous())',
                 'r(x, [-1]).contiguous() + r(x, [-1]).contiguous()',
                 'm.mul(r(x, [-1]).contiguous(), .5)', 'r(x, [1, -1]).sum(1)',
                 'r(x, [1, -1]).t().transpose(0, 1).contiguous().view(-1)')
        for text in texts:
            for policy in POLICIES:
                fn, ref = expression(text), expression(text, torch)
                f, cache = compile_with_cache(fn, *policy)
                for rows in (1, 5, 0, 1):
                    x = native.ones((5, 7)).to('cuda:0')[:rows][:, :3]
                    y = torch.ones((5, 7), device='cuda:0')[:rows][:, :3]
                    for _ in range(2):
                        with ExitStack() as stack:
                            if policy[1] and cache.graphs: stack.enter_context(patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')))
                            self.compare(call_without_python(f, {fn.__code__}, x), ref(y))
                self.assertEqual(len(cache.graphs), 1 if policy[1] else 3)
        for shape in ((3, 7), (17, 31), (1, 1), (), (0, 7)):
            x, y = native.ones(shape).to('cuda:0'), torch.ones(shape, device='cuda:0')
            source = 'def helper(x):\n    shape = [1, -1]\n    return m.reshape(x, shape)\ndef program(x, y):\n    return r(m.add(helper(x), r(y, [1, -1])), (-1,))\n'
            fn, ref = program(source), program(source, torch)
            for policy in POLICIES:
                f, _ = compile_with_cache(fn, *policy)
                self.compare(call_without_python(f, {fn.__code__, fn.__globals__['helper'].__code__}, x, x), ref(y, y))
        # No widening of strided arithmetic after a view-compatible reshape.
        x = native.ones((3, 7)).to('cuda:0').select(1, 1)
        for text in ('-r(x, [-1])', 'm.relu(r(x, [-1]))', 'r(x, [-1]) * 2', 'r(x, [-1]) + x'):
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: compile_with_cache(expression(text))[0](x))

    def test_per_field_guards_rebinding_lowered_reuse_and_recovery(self):
        x, y = native.tensor([[-2., 3.]]).to('cuda:0'), torch.tensor([[-2., 3.]], device='cuda:0')
        controls = ('m.transpose(x, 0, 0)', 'm.t(x)', 'm.squeeze(x)', 'm.relu(x)',
                    'm.add(x, x)', 'm.neg(x)', 'm.mul(x, 2)', 'r(x, [-1])')
        for text in controls:
            for policy in POLICIES:
                fn, ref = expression(text), expression(text, torch)
                f, cache = compile_with_cache(fn, *policy, limit=1); self.compare(f(x), ref(y))
                graph = next(iter(cache.graphs.values()))
                for deleted in (False, True):
                    with ExitStack() as stack:
                        for module in (native, native._C):
                            stack.enter_context(patch.object(module, 'reshape', Hostile()))
                            if deleted: del module.reshape
                            stack.enter_context(patch.object(module, '__getattr__', Hostile.fail, create=True))
                        with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('unrelated cache miss')):
                            self.compare(f(x), ref(y))
                        self.compare(compile_with_cache(fn, *policy, limit=1)[0](x), ref(y))
                        self.compare(graph.forward(x), ref(y))
                self.assertEqual(len(cache.graphs), 1)
        for source, field in (('def program(x):\n    return m.reshape(x, [-1])\n', 'reshape'),
                              ('def program(x):\n    return r(x, [-1])\n', 'r'),
                              ('def helper(x):\n    return m.reshape(x, [-1])\ndef program(x):\n    return m.relu(helper(x))\n', 'reshape')):
            for policy in POLICIES:
                fn, ref = program(source), program(source, torch)
                bindings = vars(native) if field == 'reshape' else fn.__globals__
                f, cache = compile_with_cache(fn, *policy, limit=1); f(x)
                graph = next(iter(cache.graphs.values()))
                for bad in (Hostile(), property(Hostile.fail), native.neg, native.add):
                    with patch.dict(bindings, {field: bad}):
                        for _ in range(2): self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
                        self.compare(graph.forward(x), ref(y))
                        failed, empty = compile_with_cache(fn, *policy)
                        self.exact_error(trace.CompileTraceUnsupportedError, lambda: failed(x))
                        self.assertFalse(empty.graphs)
                    self.compare(failed(x), ref(y))
                    self.assertEqual(len(cache.graphs), 1)
                saved = bindings.pop(field)
                try: self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
                finally: bindings[field] = saved
                self.compare(f(x), ref(y))
        # Arity and semantics follow genuine callable identity, not export name.
        for text, field in (('m.add(x, [-1])', 'add'), ('m.reshape(x, [-1])', 'reshape')):
            with patch.object(native, field, genuine_reshape):
                self.compare(compile_with_cache(expression(text))[0](x), y.reshape(-1))
        fn = expression('m.reshape(x, x)', arity=1)
        with patch.object(native, 'reshape', native.add):
            self.compare(compile_with_cache(fn)[0](x), y + y)
        fn = expression('r(x, [-1])')
        other = types.FunctionType(fn.__code__, {**fn.__globals__, 'r': Hostile()})
        f, cache = compile_with_cache(fn, limit=1); f(x)
        self.exact_error(trace.CompileTraceUnsupportedError, lambda: compile_with_cache(other)[0](x))
        with patch.dict(fn.__globals__, {'r': Hostile()}):
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
        self.compare(f(x), y.reshape(-1)); self.assertEqual(len(cache.graphs), 1)

    def test_prevalidation_invalid_shapes_outputs_and_failure_cache(self):
        x = native.ones((1, 7)).to('cuda:0')
        for policy in POLICIES:
            fn = expression('r(m.relu(r(x, [-1])), [1, -1])', arity=2)
            f, cache = compile_with_cache(fn, *policy); f(x, x)
            key, graph = next(iter(cache.graphs.items())); node = graph.operations[-1]
            malformed = [replace(graph, output='missing'), replace(graph, output_metadata=None)]
            malformed += [replace(graph, operations=(*graph.operations[:-1], replace(node, **change)))
                          for change in ({'inputs': ('missing',)}, {'inputs': ('arg0', 'arg0')},
                                         {'shape': (-1, -1)}, {'shape': (True,)}, {'shape': None},
                                         {'axes': (0, 0)}, {'scalar': 1}, {'metadata': None},
                                         {'metadata': replace(node.metadata, storage_offset=1)},
                                         {'metadata': replace(node.metadata, device='cuda:1')})]
            with ExitStack() as stack:
                for owner, name in ((trace, '_execute_operation'), (trace._native, '_compile_trace_cuda_graph'),
                                    (trace._native, '_compile_trace_unary'), (trace._native, '_compile_trace_binary')):
                    stack.enter_context(patch.object(owner, name, side_effect=AssertionError('native work')))
                for bad in malformed:
                    cache.graphs[key] = bad
                    with self.assertRaises((trace.CompileTraceUnsupportedError, RuntimeError, TypeError)):
                        f(x, x)
                cache.graphs[key] = graph
                for a, b in ((x.cpu(), x), (x, x.cpu()), (native.ones((1, 1, 7)).to('cuda:0'), x)):
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(a, b))
                self.exact_error(TypeError, lambda: f(x, Hostile()))
            self.compare(f(x, x), torch.ones((1, 7), device='cuda:0'))
            for shape, kind in (('(True,)', TypeError), ('[1.]', TypeError), ('(-1,-1)', RuntimeError),
                                ('(8,)', RuntimeError), ('(x, 1.)', TypeError), ('(x,)', trace.CompileTraceUnsupportedError)):
                bad, empty = compile_with_cache(expression(f'r(-x, {shape})'), *policy)
                with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native work')):
                    for _ in range(2): self.exact_error(kind, lambda: bad(x))
                self.assertFalse(empty.graphs)
        f, cache = compile_with_cache(expression('r(x, [-1])'))
        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=RuntimeError('launch failed')):
            self.exact_error(RuntimeError, lambda: f(x))
        self.assertFalse(cache.graphs); self.compare(f(x), torch.ones(7, device='cuda:0'))
        f, cache = compile_with_cache(expression('r(x.contiguous(), [7])'), dynamic=True)
        f(x)
        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native work')):
            self.exact_error(RuntimeError, lambda: f(native.ones((2, 7)).to('cuda:0')))
        self.compare(f(x), torch.ones(7, device='cuda:0')); self.assertEqual(len(cache.graphs), 1)

    def test_no_reference_import_or_python_bodies(self):
        source = r'''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'): raise AssertionError('reference import')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
from torch_rs import reshape as r
def helper(x): return m.reshape(x, [-1])
def f(x):
    a = r(helper(x), (1, -1))
    return a, a, r(x, [1, -1])
from torch_rs import _compile_trace as t
def reject(*args): raise AssertionError('Python node')
t._execute_operation = reject
def profile(frame, event, arg):
    if event == 'call' and frame.f_code in (f.__code__, helper.__code__): raise AssertionError('body')
sys.setprofile(profile)
g = m.compile(f, backend='eager', fullgraph=True)
for value in (-2., 3.):
    x = m.full((1, 7), value).to('cuda:0')
    a, b, c = g(x)
    assert a is b and a is not c and a is not x
    assert a.data_ptr() == c.data_ptr() == x.data_ptr()
    assert a.cpu().tolist() == [[value] * 7]
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class ModuleReshapeDeviceTests(ExactErrors, Comparison, unittest.TestCase):
    def test_view_pack_restoration_unused_inputs_and_captures(self):
        lib, previous = runtime(), torch.cuda.current_device()
        driver = ctypes.CDLL('libcuda.so.1')
        driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        def context():
            value = ctypes.c_void_p()
            self.assertEqual(driver.cuCtxGetCurrent(ctypes.byref(value)), 0)
            return value.value
        try:
            for policy in POLICIES:
                for text in ('m.reshape(x, [-1])', 'r(x.t(), (-1,))'):
                    f, cache = compile_with_cache(expression(text, arity=2), *policy)
                    for current, target in ((1, 0), (0, 1), (1, 0)):
                        torch.cuda.set_device(current); torch.empty(1, device=f'cuda:{current}')
                        before = context()
                        x = native.ones((3, 7)).to(f'cuda:{target}')
                        out = f(x, x)
                        self.assertEqual(context(), before); self.assertEqual(torch.cuda.current_device(), current)
                        self.assertEqual(out.data_ptr() == x.data_ptr(), 'x.t()' not in text)
                        expected = torch.ones(21, device=f'cuda:{target}')
                        self.compare(out, expected)
                        wrong = native.ones((3, 7)).to(f'cuda:{current}')
                        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native work')):
                            self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x, wrong))
                            graph = next(g for g in cache.graphs.values() if g.operations[0].metadata.device == f'cuda:{target}')
                            bad = replace(graph, captures=(trace.CompileTraceCapture('unused', wrong, trace._metadata_from_native_tensor(wrong)),))
                            self.exact_error(trace.CompileTraceUnsupportedError, lambda: bad.forward(x, x))
                        del x, wrong
                        gc.collect(); self.compare(out, expected)
                        ordinal = ctypes.c_int(); self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                        self.assertEqual(ordinal.value, current); self.assertEqual(context(), before)
                    self.assertEqual(len(cache.graphs), 2)
        finally:
            torch.cuda.set_device(previous)
