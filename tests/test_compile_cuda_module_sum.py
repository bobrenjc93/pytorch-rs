"""Positional native row sums; non-scoring frontend and H100 regressions."""
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
from torch_rs import sum as genuine_sum
from torch_rs import _compile_bytecode as bytecode, _compile_trace as trace
from tests.test_compile_cuda_boundary import compile_with_cache
from tests.test_compile_cuda_module_arithmetic import ExactErrors, Hostile
from tests.test_compile_cuda_mul_scalar import POLICIES, call_without_python
from tests.test_cuda_add import Comparison, available, runtime, torch, upload
from tests import test_cuda_sum_rows as eager_tests


def program(source, module=native):
    namespace = {'__name__': __name__, 'm': module,
                 's': genuine_sum if module is native else module.sum}
    exec(source, namespace)
    return namespace['program']


def expression(text, module=native, arity=1):
    return program(f'def program({"x" if arity == 1 else "x, y"}):\n    return {text}\n', module)


SHAPES = ((0, 0), (0, 3), (1, 0), (1, 1), (1, 3), (2, 1),
          (2, 3), (3, 2), (3, 7), (7, 3), (17, 65), (32, 1025))
SPELLINGS = ('m.sum(x, 1)', 's(x, 1)', 'm.sum(x, -1)', 's(x, -1)',
             'm.sum(x + x, 1)', 's(x + x, 1)',
             'm.relu(m.sum(x, 1))', 'm.relu(s(x, 1))')
NESTED = '''def program(x):
    a = m.sum(x, 1)
    b = s(x, -1)
    shared = [a, (b, a.view(-1, 1))]
    return x, shared, shared, a
'''

_STARTUP = r'''
import sys
import torch_rs as m
from torch_rs import sum as good
mode, mutation = sys.argv[1:]
assert 'torch_rs._compile_bytecode' not in sys.modules
assert 'torch_rs._compile_trace' not in sys.modules
owner = m._C._VariableFunctionsClass
assert good is owner.sum is m._C.sum
try: owner.sum = lambda *args: None
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
    if mutation == 'deleted': del module.sum
    else: module.sum = bad
m._C._VariableFunctionsClass = bad
bad_alias = bad
def direct(x): return good(x, -1)
def module_call(x): return m.sum(x, 1)
def counterfeit(x): return bad_alias(x, 1)
def control(x): return m.relu(m.add(m.neg(x), x)).sum(dim=1)
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
    assert b._builtin_target(good).target == 'sum'
    assert b._builtin_target(bad) is None
    assert b.lower_compile_graph(direct, meta).operations[0].reduction == (1, False)
    assert len(b.lower_compile_graph(control, meta).operations) == 4
    m.sum = m._C.sum = good
    assert b.lower_compile_graph(module_call, meta).operations[0].target == 'sum'
else:
    from torch_rs import _compiler_state as state
    x = m.tensor([[-2., 3.]]).to('cuda:0')
    for fullgraph, dynamic in ((True, None), (True, False), (True, True), (False, None)):
        if mutation == 'deleted': vars(m).pop('sum', None)
        else: m.sum = bad
        previous = set(state.native_eager_compile_caches)
        f = m.compile(module_call, backend='eager', fullgraph=fullgraph, dynamic=dynamic, recompile_limit=1)
        cache, = set(state.native_eager_compile_caches) - previous
        for _ in range(2):
            reject(lambda: f(x)); assert not cache.graphs
        for fn, expected in ((direct, [1.]), (control, [0.])):
            g = m.compile(fn, backend='eager', fullgraph=fullgraph, dynamic=dynamic, recompile_limit=1)
            for _ in range(2): assert g(x).cpu().tolist() == expected
        g = m.compile(counterfeit, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
        for _ in range(2): reject(lambda: g(x))
        m.sum = m._C.sum = good
        a, b = f(x), f(x)
        assert a.cpu().tolist() == b.cpu().tolist() == [1.]
        assert a is not b and a is not x
        assert len({a.data_ptr(), b.data_ptr(), x.data_ptr()}) == 3
        assert len(cache.graphs) == 1
assert not calls
'''


def startup(test, mode):
    for mutation in ('fake', 'hostile', 'deleted'):
        with test.subTest(mutation=mutation):
            result = subprocess.run([sys.executable, '-c', _STARTUP, mode, mutation],
                                    capture_output=True, text=True, timeout=60)
            test.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ModuleSumFrontendTests(ExactErrors, unittest.TestCase):
    def test_startup_before_frontend_import(self):
        startup(self, 'frontend')

    def test_identity_arguments_and_metadata_boundaries(self):
        meta = trace.CompileTraceTensorMetadata((3, 7), (7, 1), trace.float32, 'cuda:0', False, 9)
        self.assertEqual(bytecode._builtin_target(genuine_sum), bytecode._BytecodeBuiltin('sum', 2))
        for text in SPELLINGS:
            graph = bytecode.lower_compile_graph(expression(text), (meta,))
            self.assertEqual(graph.output_metadata, replace(meta, shape=(3,), stride=(1,), storage_offset=0))
            node = next(n for n in graph.operations if n.target == 'sum')
            self.assertEqual((node.op, node.reduction, node.scalar), ('call_reduction', (1, False), None))
        for text in ('s()', 's(x)', 's(x, 1, False)', 's(x, dim=1)',
                     's(input=x, dim=1)', 's(x, 1, keepdim=False)', 's(x, 1, dtype=None)',
                     's(x, 1, out=None)', 's(x, 1, unknown=0)', 's(x, True)', 's(x, 1.)',
                     's(x, None)', 's(x, 0)', 's(x, -2)', 's(x, (1,))', 's(x, [1])',
                     's(x, x)', 's(1, 1)', 's(*(x, 1))', 's(x, **{"dim": 1})',
                     'm._C.sum(x, 1)', 'm.nn.functional.sum(x, 1)'):
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.lower_compile_graph(expression(text), (meta,)))
        for change in ({'device': 'cpu'}, {'requires_grad': True}, {'stride': (1, 3)},
                       {'shape': (21,), 'stride': (1,)}, {'shape': (1, 3, 7), 'stride': (21, 7, 1)},
                       {'dtype': trace.CompileTraceDType('torch.float64')}):
            for text in ('m.sum(x, 1)', 's(x, -1)'):
                self.exact_error(trace.CompileTraceUnsupportedError,
                                 lambda: bytecode.lower_compile_graph(expression(text), (replace(meta, **change),)))
        fn = program('def program(x):\n    dim = -1\n    return s(x, dim)\n')
        self.assertEqual(bytecode.lower_compile_graph(fn, (meta,)).operations[0].reduction, (1, False))
        fn = expression('s(x, DIM)'); fn.__globals__['DIM'] = 1
        first = bytecode.prepare_compile_cache_request(fn, (meta,)).key
        fn.__globals__['DIM'] = -1
        self.assertNotEqual(first, bytecode.prepare_compile_cache_request(fn, (meta,)).key)
        for bad in (Hostile(), np.int64(1), native.tensor(1.), True, 1.):
            fn.__globals__['DIM'] = bad
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: bytecode.lower_compile_graph(fn, (meta,)))

    def test_wrong_namespaces_and_descriptors_without_hooks(self):
        meta = (trace.CompileTraceTensorMetadata((3, 7), (7, 1), trace.float32, 'cuda:0', False, 0),)
        fake = types.ModuleType('torch_rs'); fake.__dict__.update(vars(native))
        class ModuleSubclass(types.ModuleType):
            __getattribute__ = Hostile.fail
        for value in (fake, ModuleSubclass('torch_rs'), Hostile(), types.SimpleNamespace(sum=genuine_sum)):
            fn = expression('m.sum(x, 1)'); fn.__globals__['m'] = value
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: bytecode.prepare_compile_cache_request(fn, meta))
        for value in (Hostile(), property(Hostile.fail), native.abs):
            fn = expression('s(x, 1)'); fn.__globals__['s'] = value
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: bytecode.prepare_compile_cache_request(fn, meta))
        with patch.object(native, '__getattr__', Hostile.fail, create=True):
            saved = vars(native).pop('sum')
            try:
                self.exact_error(trace.CompileTraceUnsupportedError,
                                 lambda: bytecode.prepare_compile_cache_request(expression('m.sum(x, 1)'), meta))
            finally: native.sum = saved

    def test_cpu_rejection_does_not_publish_specialization(self):
        for policy in POLICIES:
            for text in ('m.sum(x, 1)', 's(x, -1)'):
                f, cache = compile_with_cache(expression(text), *policy, limit=1)
                for _ in range(2):
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(native.ones((3, 7))))
                    self.assertFalse(cache.graphs)


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class ModuleSumCudaTests(ExactErrors, Comparison, unittest.TestCase):
    compare_sum = eager_tests.CudaSumRowsTests.compare_sum

    def test_startup_before_first_compile(self):
        startup(self, 'cuda')

    def checked(self, fn, ref, args, refs, policy, backend='eager'):
        f, cache = compile_with_cache(fn, *policy)
        reference = torch.compile(ref, backend=backend, fullgraph=policy[0], dynamic=policy[1])
        codes = {v.__code__ for v in fn.__globals__.values() if type(v) is types.FunctionType}
        retained = []
        for warm in (False, True):
            with ExitStack() as stack:
                stack.enter_context(patch.object(trace, '_execute_operation', side_effect=AssertionError('Python node')))
                if warm: stack.enter_context(patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')))
                out = call_without_python(f, codes, *args)
            self.compare(out, reference(*refs))
            self.compare(out, ref(*refs))
            for x, tx in zip(args, refs): self.compare(x, tx)
            self.assertIsNot(out, args[0])
            if out.numel(): self.assertNotEqual(out.data_ptr(), args[0].data_ptr())
            retained.append(out)
        self.assertIsNot(*retained)
        if out.numel(): self.assertNotEqual(retained[0].data_ptr(), retained[1].data_ptr())
        self.assertEqual(len(cache.graphs), 1)
        self.compare(next(iter(cache.graphs.values())).forward(*args), ref(*refs))

    def test_frozen_96_gap_cells_all_policies_inductor(self):
        rng = np.random.default_rng(198801)
        for text in SPELLINGS:
            for shape in SHAPES:
                data = (rng.integers(-31, 32, size=shape) * .125).astype(np.float32)
                args, refs = [upload(m, data.ravel(), shape) for m in (native, torch)]
                for policy in POLICIES:
                    with self.subTest(text=text, shape=shape, policy=policy):
                        torch._dynamo.reset()
                        self.checked(expression(text), expression(text, torch), [args], [refs], policy, 'inductor')
                if text in SPELLINGS[:4]:
                    self.compare(args.sum(1), refs.sum(1))

    def test_offsets_singletons_nested_identity_changed_inputs_and_lifetime(self):
        rng = np.random.default_rng(198802)
        layouts = (lambda x: x, lambda x: x[1:], lambda x: x[:1].t(),
                   lambda x: x[:0].t(), lambda x: x[:1][:, :1])
        for shape in ((3, 7), (2, 1025), (3, 0), (0, 3), (1, 1)):
            for layout in layouts:
                for policy in POLICIES:
                    fn = program(NESTED); f, cache = compile_with_cache(fn, *policy)
                    retained = []
                    for warm in (False, True):
                        data = (rng.integers(-31, 32, size=shape) * .125).astype(np.float32)
                        x, tx = [layout(upload(m, data.ravel(), shape)) for m in (native, torch)]
                        with ExitStack() as stack:
                            stack.enter_context(patch.object(trace, '_execute_operation', side_effect=AssertionError('Python node')))
                            if warm: stack.enter_context(patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')))
                            out = call_without_python(f, {fn.__code__}, x)
                        expected = tx.sum(1)
                        self.assertIs(out[0], x); self.assertIs(out[1], out[2]); self.assertIs(out[1][0], out[3])
                        a, (b, view) = out[1]
                        self.assertIsNot(a, b); self.assertIsNot(a, view)
                        self.compare(a, expected); self.compare(b, expected)
                        self.compare(view, expected.view(-1, 1)); self.compare(x, tx)
                        self.assertEqual(view.data_ptr(), a.data_ptr())
                        if a.numel():
                            self.assertNotEqual(a.data_ptr(), b.data_ptr())
                            self.assertNotEqual(a.data_ptr(), x.data_ptr())
                        for previous, _ in retained:
                            self.assertIsNot(a, previous)
                            if a.numel(): self.assertNotEqual(a.data_ptr(), previous.data_ptr())
                        retained.append((a, expected))
                    self.assertEqual(len(cache.graphs), 1)
                    del x, out, f, cache, a, b, view
                    gc.collect()
                    for actual, expected in retained: self.compare(actual, expected)

    def test_consumers_helpers_and_special_values(self):
        texts = ('s(x, 1).view(-1, 1)', 'm.reshape(s(x, -1), (3, 1))',
                 'm.t(s(x, 1))', 'm.squeeze(s(x, 1))', 'm.neg(s(x, 1))',
                 'm.mul(s(x, 1), .5)', 's(x, 1) + y.sum(1)', 'm.sum(x + y, -1)',
                 's(m.relu(x), 1)', 's(x.transpose(0, 1).contiguous(), 1)')
        sources = [f'def program(x, y):\n    return {text}\n' for text in texts]
        sources += ['def helper(x):\n    return m.sum(x, -1)\ndef program(x, y):\n    return m.relu(helper(x)) + s(y, 1)\n']
        data = np.arange(-10, 11, dtype=np.float32).reshape(3, 7) * .125
        x, tx = [upload(m, data.ravel(), data.shape) for m in (native, torch)]
        for source in sources:
            for policy in POLICIES:
                torch._dynamo.reset()
                self.checked(program(source), program(source, torch), [x, x], [tx, tx], policy)
        patterns = [[1e8, 1, 1, -1e8], [3e38, 0, 3e38, -3e38],
                    [float('inf'), -float('inf'), 0, 1], [float('nan'), 1, 2, 3],
                    [-0., 0., -0., 0.], [1e-44, -1e-44, 1e-44, 1e-44]]
        for data in (np.array(patterns, dtype=np.float32), np.full((2, 65539), .1, dtype=np.float32)):
            x, tx = [upload(m, data.ravel(), data.shape) for m in (native, torch)]
            for text in SPELLINGS[:4]:
                for policy in POLICIES:
                    f, _ = compile_with_cache(expression(text), *policy)
                    actual = f(x)
                    self.compare(actual, x.sum(1))
                    self.compare_sum(actual, tx.sum(1))

    def test_per_used_field_guards_aliases_substitutions_and_recovery(self):
        x = native.tensor([[-2., 3.]]).to('cuda:0')
        tx = torch.tensor([[-2., 3.]], device='cuda:0')
        controls = ('m.add(x, x)', 'm.neg(x)', 'm.mul(x, 2)', 'm.reshape(x, (2, 1))',
                    'm.transpose(x, 0, 1)', 'm.t(x)', 'm.squeeze(x)', 'm.relu(x)',
                    'x.sum(dim=-1, keepdim=True)', 's(x, 1)')
        for text in controls:
            for policy in POLICIES:
                fn, ref = expression(text), expression(text, torch)
                f, cache = compile_with_cache(fn, *policy, limit=1); self.compare(f(x), ref(tx))
                graph = next(iter(cache.graphs.values()))
                for deleted in (False, True):
                    with ExitStack() as stack:
                        for module in (native, native._C):
                            stack.enter_context(patch.object(module, 'sum', Hostile()))
                            if deleted: del module.sum
                            stack.enter_context(patch.object(module, '__getattr__', Hostile.fail, create=True))
                        with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('unrelated cache miss')):
                            self.compare(f(x), ref(tx))
                        self.compare(compile_with_cache(fn, *policy, limit=1)[0](x), ref(tx))
                        self.compare(graph.forward(x), ref(tx))
                self.assertEqual(len(cache.graphs), 1)
        for source, field in (('def program(x):\n    return m.sum(x, 1)\n', 'sum'),
                              ('def program(x):\n    return s(x, -1)\n', 's'),
                              ('def helper(x):\n    return m.sum(x, 1)\ndef program(x):\n    return m.relu(helper(x))\n', 'sum')):
            for policy in POLICIES:
                fn, ref = program(source), program(source, torch)
                bindings = vars(native) if field == 'sum' else fn.__globals__
                f, cache = compile_with_cache(fn, *policy, limit=1); f(x)
                graph = next(iter(cache.graphs.values()))
                for bad in (Hostile(), property(Hostile.fail), native.neg, native.add):
                    with patch.dict(bindings, {field: bad}):
                        for _ in range(2): self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
                        self.compare(graph.forward(x), ref(tx))
                        failed, empty = compile_with_cache(fn, *policy)
                        for _ in range(2): self.exact_error(trace.CompileTraceUnsupportedError, lambda: failed(x))
                        self.assertFalse(empty.graphs)
                    self.compare(failed(x), ref(tx)); self.assertEqual(len(cache.graphs), 1)
                saved = bindings.pop(field)
                try: self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
                finally: bindings[field] = saved
                self.compare(f(x), ref(tx))
        # Canonical substitutions use identity and the substituted schema.
        for text, field in (('m.add(x, 1)', 'add'), ('m.reshape(x, 1)', 'reshape')):
            with patch.object(native, field, genuine_sum):
                self.compare(compile_with_cache(expression(text))[0](x), tx.sum(1))
        fn = expression('m.sum(x, 1)'); f, cache = compile_with_cache(fn)
        self.compare(f(x), tx.sum(1))
        with patch.object(native, 'sum', native.mul): self.compare(f(x), tx * 1)
        self.compare(f(x), tx.sum(1)); self.assertEqual(len(cache.graphs), 2)
        fn = expression('s(x, 1)')
        other = types.FunctionType(fn.__code__, {**fn.__globals__, 's': native.mul})
        a, cache = compile_with_cache(fn, limit=1); b, _ = compile_with_cache(other)
        self.compare(a(x), tx.sum(1)); self.compare(b(x), tx)
        with patch.dict(fn.__globals__, {'s': Hostile()}):
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: a(x))
            self.compare(b(x), tx)
        self.compare(a(x), tx.sum(1)); self.assertEqual(len(cache.graphs), 1)
        fn = expression('s(x, DIM)'); fn.__globals__['DIM'] = 1
        f, cache = compile_with_cache(fn); f(x)
        for bad in (True, 1., Hostile(), native.tensor(1.), 0):
            fn.__globals__['DIM'] = bad
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
        self.assertEqual(len(cache.graphs), 1)
        fn.__globals__['DIM'] = -1
        self.compare(f(x), tx.sum(1)); self.assertEqual(len(cache.graphs), 2)

    def test_live_rejections_and_restored_module_namespace(self):
        x = native.ones((3, 7)).to('cuda:0')
        for policy in POLICIES:
            for text in ('s(x)', 's(x, 1, False)', 'm.sum(x, dim=1)',
                         's(x, 1, dtype=None)', 's(x, 1, out=x)', 's(x, 1, unknown=0)',
                         's(x, True)', 's(x, 1.)', 's(x, x)', 's(x, (1,))',
                         's(x, [1])', 's(x, None)', 's(x, 0)', 's(x, -2)',
                         's(*(x, 1))', 's(x, **{"dim": 1})'):
                f, cache = compile_with_cache(expression(text), *policy, limit=1)
                with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('launched')):
                    for _ in range(2): self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
                self.assertFalse(cache.graphs)
            fn = expression('m.sum(x, 1)'); f, cache = compile_with_cache(fn, *policy, limit=1)
            f(x)
            fake = types.ModuleType('torch_rs'); fake.__dict__.update(vars(native))
            for module in (fake, types.SimpleNamespace(sum=genuine_sum), Hostile()):
                with patch.dict(fn.__globals__, {'m': module}):
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
            self.compare(f(x), torch.full((3,), 7., device='cuda:0'))
            self.assertEqual(len(cache.graphs), 1)

    def test_prevalidation_late_declarations_outputs_and_failure_cache(self):
        x = native.ones((3, 7)).to('cuda:0')
        for policy in POLICIES:
            f, cache = compile_with_cache(expression('s(m.relu(x), 1)', arity=2), *policy)
            f(x, x); key, graph = next(iter(cache.graphs.items())); node = graph.operations[-1]
            malformed = [replace(graph, output='missing'), replace(graph, output_metadata=None),
                         replace(graph, output_metadata=replace(node.metadata, shape=(4,)))]
            malformed += [replace(graph, operations=(*graph.operations[:-1], replace(node, **change)))
                          for change in ({'inputs': ('missing',)}, {'inputs': ('arg0', 'arg0')},
                                         {'reduction': None}, {'reduction': (0, False)},
                                         {'reduction': (True, False)}, {'reduction': (1, 1)},
                                         {'shape': (3,)}, {'axes': (0, 0)}, {'scalar': 1},
                                         {'metadata': None}, {'metadata': replace(node.metadata, storage_offset=1)},
                                         {'metadata': replace(node.metadata, device='cuda:1')})]
            with ExitStack() as stack:
                for owner, name in ((trace, '_execute_operation'), (trace._native, '_compile_trace_cuda_graph'),
                                    (trace._native, '_compile_trace_reduction'), (trace._native, '_compile_trace_unary')):
                    stack.enter_context(patch.object(owner, name, side_effect=AssertionError('execution before validation')))
                for bad in malformed:
                    cache.graphs[key] = bad
                    with self.assertRaises((trace.CompileTraceUnsupportedError, TypeError, ValueError)):
                        f(x, x)
                cache.graphs[key] = graph
                for args in ((x.cpu(), x), (x, x.cpu()), (x.t(), x),
                             (x.reshape(21), x), (x.reshape(1, 3, 7), x),
                             (native.ones((3, 7), requires_grad=True), x)):
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(*args))
                self.exact_error(TypeError, lambda: f(x, Hostile()))
                for text in ('s(m.relu(x), 0)', 's(s(x, 1), 1)', 's(x, 1) + x', 's(x, 1, False)'):
                    failed, empty = compile_with_cache(expression(text), *policy)
                    for _ in range(2): self.exact_error(trace.CompileTraceUnsupportedError, lambda: failed(x))
                    self.assertFalse(empty.graphs)
            self.compare(f(x, x), torch.full((3,), 7., device='cuda:0'))
            failed, empty = compile_with_cache(expression('m.sum(x, 1)'), *policy)
            with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=RuntimeError('launch failed')):
                self.exact_error(RuntimeError, lambda: failed(x))
            self.assertFalse(empty.graphs)
            self.compare(failed(x), torch.full((3,), 7., device='cuda:0'))
        f, cache = compile_with_cache(program(NESTED)); f(x)
        key, graph = next(iter(cache.graphs.items()))
        a, first, second, last = graph.output_metadata.elements
        leaf, pair = second.elements
        bad = replace(second, elements=(replace(leaf, stride=(True,)), pair))
        cache.graphs[key] = replace(graph, output_metadata=replace(graph.output_metadata, elements=(a, first, bad, last)))
        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native work')):
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x))
        cache.graphs[key] = graph
        self.compare(f(x)[3], torch.full((3,), 7., device='cuda:0'))
        # Dynamic cache hits still validate a later incompatible consumer.
        f, cache = compile_with_cache(expression('s(x, 1) + y', arity=2), dynamic=True)
        y = native.ones((3,)).to('cuda:0'); f(x, y)
        with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')), \
                patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('native work')):
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(native.ones((4, 7)).to('cuda:0'), y))
        self.compare(f(x, y), torch.full((3,), 8., device='cuda:0'))
        self.assertEqual(len(cache.graphs), 1)

    def test_no_reference_import_or_python_bodies(self):
        source = r'''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'): raise AssertionError('reference import')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
from torch_rs import sum as s
def helper(x): return m.sum(x, 1)
def f(x):
    a = helper(x)
    return a, a, s(x, -1)
from torch_rs import _compile_trace as t
def reject(*args): raise AssertionError('Python node')
t._execute_operation = reject
def profile(frame, event, arg):
    if event == 'call' and frame.f_code in (f.__code__, helper.__code__): raise AssertionError('body')
sys.setprofile(profile)
g = m.compile(f, backend='eager', fullgraph=True)
for value in (-2., 3.):
    x = m.full((3, 7), value).to('cuda:0')
    a, b, c = g(x)
    assert a is b and a is not c and a is not x
    assert len({a.data_ptr(), c.data_ptr(), x.data_ptr()}) == 3
    assert a.cpu().tolist() == c.cpu().tolist() == [value * 7] * 3
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class ModuleSumDeviceTests(ExactErrors, Comparison, unittest.TestCase):
    def test_restoration_unused_inputs_exception_paths_and_lifetime(self):
        lib, previous = runtime(), torch.cuda.current_device()
        driver = ctypes.CDLL('libcuda.so.1')
        driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        def context():
            value = ctypes.c_void_p()
            self.assertEqual(driver.cuCtxGetCurrent(ctypes.byref(value)), 0)
            return value.value
        try:
            for text in ('m.sum(x, 1)', 'm.relu(s(x, -1))'):
                for policy in POLICIES:
                    f, cache = compile_with_cache(expression(text, arity=2), *policy)
                    for current, target in ((1, 0), (0, 1), (1, 0)):
                        torch.cuda.set_device(current)
                        torch.empty(1, device=f'cuda:{current}')
                        before = context()
                        x = native.full((3, 7), 2.).to(f'cuda:{target}')
                        out = f(x, x)
                        self.assertEqual(context(), before)
                        self.assertEqual(torch.cuda.current_device(), current)
                        expected = torch.full((3,), 14., device=f'cuda:{target}')
                        self.compare(out, expected)
                        wrong = native.ones((3, 7)).to(f'cuda:{current}')
                        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('launched')):
                            self.exact_error(trace.CompileTraceUnsupportedError, lambda: f(x, wrong))
                            graph = next(g for g in cache.graphs.values() if g.operations[0].metadata.device == f'cuda:{target}')
                            bad = replace(graph, captures=(trace.CompileTraceCapture('unused', wrong, trace._metadata_from_native_tensor(wrong)),))
                            self.exact_error(trace.CompileTraceUnsupportedError, lambda: bad.forward(x, x))
                        failed, empty = compile_with_cache(expression(text), *policy)
                        with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=RuntimeError('launch failed')):
                            self.exact_error(RuntimeError, lambda: failed(x))
                        self.assertFalse(empty.graphs)
                        self.compare(failed(x), expected)
                        del x, wrong, failed
                        gc.collect()
                        self.compare(out, expected)
                        ordinal = ctypes.c_int()
                        self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                        self.assertEqual(ordinal.value, current)
                        self.assertEqual(context(), before)
                    self.assertEqual(len(cache.graphs), 2)
        finally: torch.cuda.set_device(previous)
