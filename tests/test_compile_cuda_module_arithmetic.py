"""Public native call spellings and precise guards; no scoring-corpus changes."""
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
from torch_rs import _compile_bytecode as bytecode, _compile_trace as trace
from tests.test_compile_cuda_boundary import compile_with_cache
from tests.test_compile_cuda_mul_scalar import POLICIES, call_without_python
from tests.test_cuda_add import Comparison, available, runtime, torch, upload


def program(source, module=native):
    namespace = {'__name__': __name__, 'm': module, 'alias': module,
                 'a': module.add, 'n': module.neg, 'negative': module.negative}
    exec(source, namespace)
    return namespace['program']


def expression(text, arity=2, module=native):
    return program(f'def program({"x" if arity == 1 else "x, y"}):\n    return {text}\n', module)


class ExactErrors:
    def exact_error(self, kind, callback):
        with self.assertRaises(kind) as raised:
            callback()
        self.assertIs(type(raised.exception), kind)


class Hostile:
    def fail(self, *args, **kwargs):
        raise AssertionError('user callback executed')
    __call__ = __eq__ = __hash__ = __bool__ = __float__ = __index__ = fail
    __getattr__ = fail


# Each subprocess starts before the frontend import; parent-process patching
# cannot exercise the lazy-initialization boundary.
_OWNER_STARTUP_PROBE = r'''
import sys
import types
import torch_rs as m

mode, replacement = sys.argv[1:]
assert 'torch_rs._compile_bytecode' not in sys.modules
assert 'torch_rs._compile_trace' not in sys.modules
owner = m._C._VariableFunctionsClass
canonical = owner.neg
callbacks = []
def counterfeit(x):
    callbacks.append('counterfeit body')
    return x
# This is an imported counterfeit, not a supported same-module helper.
counterfeit.__module__ = 'untrusted_native_replacement'
class Hostile:
    def __getattribute__(self, name):
        callbacks.append(name)
        raise AssertionError('owner callback: ' + name)
if replacement == 'namespace':
    m._C._VariableFunctionsClass = types.SimpleNamespace(**{
        name: getattr(owner, name) for name in ('mul', 'multiply', 'matmul', 'add', 'neg', 'negative')
    })
    m._C._VariableFunctionsClass.neg = counterfeit
elif replacement == 'hostile':
    m._C._VariableFunctionsClass = Hostile()
else:
    assert replacement == 'deleted'
    del m._C._VariableFunctionsClass
m.neg = counterfeit
bad_alias = m.neg

def program(x): return m.neg(x)
def imported_bad(x): return bad_alias(x)
def imported_good(x): return canonical(x)
def untouched(x): return m.add(canonical(x), x)

def rejected(callback):
    try:
        callback()
    except Exception as error:
        from torch_rs._compile_trace import CompileTraceUnsupportedError
        assert type(error) is CompileTraceUnsupportedError, type(error)
    else:
        raise AssertionError('counterfeit accepted')
    assert not callbacks, callbacks

assert 'torch_rs._compile_bytecode' not in sys.modules
if mode == 'frontend':
    from torch_rs import _compile_bytecode as frontend, _compile_trace as trace
    metadata = (trace.CompileTraceTensorMetadata((2,), (1,), trace.float32, 'cuda:0', False, 0),)
    for fn in (program, imported_bad):
        for _ in range(2):
            rejected(lambda: frontend.lower_compile_graph(fn, metadata))
    assert frontend._builtin_target(counterfeit) is None
    for name in ('add', 'neg', 'negative', 'mul', 'multiply', 'matmul'):
        assert frontend._builtin_target(getattr(owner, name)) is not None
    assert [op.target for op in frontend.lower_compile_graph(untouched, metadata).operations] == ['neg', 'add']
    m.neg = canonical
    assert frontend.lower_compile_graph(program, metadata).operations[0].target == 'neg'
else:
    assert mode == 'cuda'
    from torch_rs import _compiler_state as state
    x = m.tensor([1., -3.]).to('cuda:0')
    for fullgraph, dynamic in ((True, None), (True, False), (True, True), (False, None)):
        m.neg = counterfeit
        previous = set(state.native_eager_compile_caches)
        compiled = m.compile(program, backend='eager', fullgraph=fullgraph, dynamic=dynamic, recompile_limit=1)
        cache, = set(state.native_eager_compile_caches) - previous
        for _ in range(2):
            rejected(lambda: compiled(x))
            assert not cache.graphs
        # A direct canonical import and unused fields remain valid while the
        # public neg binding and exported owner are still malformed.
        for fn, expected in ((imported_good, [-1., 3.]), (untouched, [0., 0.])):
            control = m.compile(fn, backend='eager', fullgraph=fullgraph, dynamic=dynamic, recompile_limit=1)
            for _ in range(2): assert control(x).cpu().tolist() == expected
        invalid = m.compile(imported_bad, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
        for _ in range(2): rejected(lambda: invalid(x))
        m.neg = canonical
        retained = []
        for _ in range(2):
            output = compiled(x)
            assert output.cpu().tolist() == [-1., 3.]
            retained.append(output)
        assert retained[0].data_ptr() != retained[1].data_ptr()
        assert len(cache.graphs) == 1
        before = dict(cache.graphs)
        m.neg = counterfeit
        for _ in range(2): rejected(lambda: compiled(x))
        assert cache.graphs == before
        m.neg = canonical
        assert compiled(x).cpu().tolist() == [-1., 3.]
        assert x.cpu().tolist() == [1., -3.]
assert not callbacks, callbacks
'''


def check_owner_startup(test, mode):
    for replacement in ('namespace', 'hostile', 'deleted'):
        with test.subTest(replacement=replacement):
            result = subprocess.run([sys.executable, '-c', _OWNER_STARTUP_PROBE, mode, replacement],
                                    capture_output=True, text=True, timeout=60)
            test.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ModuleArithmeticFrontendTests(ExactErrors, unittest.TestCase):
    def test_identity_arity_registry_and_cpu_boundary_without_gpu(self):
        cuda = trace.CompileTraceTensorMetadata((3, 5), (5, 1), trace.float32, 'cuda:0', False, 0)
        cpu = replace(cuda, device='cpu')
        for text, arity, targets in (
            ('m.add(x, y)', 2, ['add']), ('a(x, y)', 2, ['add']),
            ('m.neg(x)', 1, ['neg']), ('n(x)', 1, ['neg']),
            ('alias.negative(x)', 1, ['neg']), ('negative(x)', 1, ['neg']),
            ('m.add(m.neg(x), alias.negative(y))', 2, ['neg', 'neg', 'add']),
        ):
            fn = expression(text, arity)
            graph = bytecode.lower_compile_graph(fn, (cuda,) * arity)
            self.assertEqual([op.target for op in graph.operations], targets)
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.lower_compile_graph(fn, (cpu,) * arity))
        for text in ('m.add(x)', 'm.add(x, y, x)', 'm.neg()', 'm.neg(x, y)',
                     'negative(x, y)', 'm.add(x, 2)', 'm.add(2, x)', 'n(2)',
                     'm.add(input=x, other=y)', 'm.add(x, y, alpha=1)',
                     'm.add(x, y, out=None)', 'n(input=x)', 'm.negative(x, out=None)'):
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.lower_compile_graph(expression(text), (cuda, cuda)))
        for metadata in (replace(cuda, requires_grad=True),
                         replace(cuda, dtype=trace.CompileTraceDType('torch.float64')),
                         replace(cuda, stride=(1, 3))):
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.lower_compile_graph(expression('m.add(n(x), y)'), (metadata, metadata)))
        # Existing CPU operator semantics, including gradients, are unchanged.
        for text in ('x + y', '-x'):
            fn = expression(text)
            x = native.tensor([1., -2.], requires_grad=True)
            result = native.compile(fn, backend='eager', fullgraph=True)(x, x)
            self.assertEqual(result.tolist(), [2., -4.] if text == 'x + y' else [-1., 2.])
            result.sum().backward()
            self.assertEqual(x.grad.tolist(), [2., 2.] if text == 'x + y' else [-1., -1.])

    def test_module_and_direct_guards_do_not_invoke_callbacks(self):
        metadata = (trace.CompileTraceTensorMetadata((5,), (1,), trace.float32, 'cuda:0', False, 0),)
        fake = types.ModuleType('torch_rs')
        fake.__dict__.update(vars(native))
        class ModuleSubclass(types.ModuleType):
            __getattribute__ = Hostile.fail
        for value in (Hostile(), fake, ModuleSubclass('torch_rs'), object()):
            fn = expression('m.neg(x)', 1)
            fn.__globals__['m'] = value
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.prepare_compile_cache_request(fn, metadata))
        fn = expression('n(x)', 1)
        for value in (Hostile(), property(Hostile.fail), native.abs):
            fn.__globals__['n'] = value
            self.exact_error(trace.CompileTraceUnsupportedError,
                             lambda: bytecode.prepare_compile_cache_request(fn, metadata))
        # Attribute deletion must use the module dictionary, never __getattr__.
        fn = expression('m.neg(x)', 1)
        saved = native.neg
        try:
            del native.neg
            with patch.object(native, '__getattr__', Hostile.fail, create=True):
                self.exact_error(trace.CompileTraceUnsupportedError,
                                 lambda: bytecode.prepare_compile_cache_request(fn, metadata))
        finally:
            native.neg = saved

    def test_exported_owner_rebinding_cannot_supply_callable_identities(self):
        class Owner:
            def __getattribute__(self, name):
                raise AssertionError('owner callback: ' + name)
        # The immutable class object, not its mutable exported module slot,
        # owns the recognized builtins throughout this frontend's lifetime.
        with patch.object(trace._native, '_VariableFunctionsClass', Owner()):
            self.assertEqual(bytecode._builtin_target(native.neg).target, 'neg')
            self.assertEqual(bytecode._builtin_target(native.add).arity, 2)
            self.assertIsNone(bytecode._builtin_target(Hostile()))

    def test_owner_rebinding_before_first_frontend_import(self):
        check_owner_startup(self, "frontend")

    def test_native_planner_rejects_declarations_without_device(self):
        hook = trace._native._compile_trace_cuda_graph
        self.exact_error(TypeError, lambda: hook((object(),), []))
        self.exact_error(NotImplementedError, lambda: hook((native.ones(3),), [('neg', (0,), None, (3,), (1,))]))


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class ModuleArithmeticCudaTests(ExactErrors, Comparison, unittest.TestCase):
    def test_owner_rebinding_before_first_public_compiled_call(self):
        check_owner_startup(self, "cuda")

    def checked(self, fn, ref_fn, args, refs, fullgraph, dynamic):
        compiled, cache = compile_with_cache(fn, fullgraph, dynamic)
        reference = torch.compile(ref_fn, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
        codes = {fn.__code__}
        codes.update(v.__code__ for v in fn.__globals__.values() if type(v) is types.FunctionType)
        outputs = []
        for _ in range(2):
            actual = call_without_python(compiled, codes, *args)
            expected = reference(*refs)
            self.compare(actual, expected)
            outputs.append(actual)
            for value, ref in zip(args, refs):
                self.compare(value, ref)
                self.assertIsNot(actual, value)
                if actual.numel() and value.numel():
                    self.assertNotEqual(actual.data_ptr(), value.data_ptr())
        self.assertIsNot(outputs[0], outputs[1])
        if outputs[0].numel():
            self.assertNotEqual(outputs[0].data_ptr(), outputs[1].data_ptr())
        self.assertEqual(len(cache.graphs), 1)
        with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')):
            self.compare(compiled(*args), reference(*refs))
        return compiled, cache

    def test_seeded_calls_generated_compositions_and_all_policies(self):
        rng = np.random.default_rng(198301)
        forms = [('m.add(x, y)', 2), ('a(x, y)', 2), ('m.neg(x)', 1),
                 ('n(x)', 1), ('alias.negative(x)', 1), ('negative(x)', 1),
                 ('m.add(m.neg(x), y)', 2), ('a(negative(x), y)', 2),
                 ('m.add(x, x)', 1)]
        sources = [(f'def program({"x" if arity == 1 else "x, y"}):\n    return {expr}\n', arity)
                   for expr, arity in forms]
        for length in (3, 11, 23):
            lines, values = ['def program(x, y):'], ['x', 'y']
            for i in range(length):
                a, b = rng.choice(values, 2)
                forms = [f'm.add({a}, {b})', f'a({a}, {b})', f'm.neg({a})',
                         f'negative({a})', f'alias.negative({a})', f'n({a})']
                lines.append(f'    v{i} = {rng.choice(forms)}')
                values.append(f'v{i}')
            sources.append(('\n'.join(lines + [f'    return {values[-1]}']) + '\n', 2))
        for source, arity in sources:
            for policy in POLICIES:
                for shape in ((), (0,), (2, 0, 3), (1,), (257,), (65539,), (3, 5)):
                    with self.subTest(source=source, policy=policy, shape=shape):
                        values = [rng.integers(-31, 32, size=int(np.prod(shape))).astype(np.float32) * .125 for _ in range(arity)]
                        args = [upload(native, v, shape) for v in values]
                        refs = [upload(torch, v, shape) for v in values]
                        self.checked(program(source), program(source, torch), args, refs, *policy)

    def test_inductor_reference_public_spelling_matrix(self):
        rng = np.random.default_rng(198304)
        for text, arity in (('m.add(x, y)', 2), ('a(x, y)', 2), ('m.neg(x)', 1),
                            ('n(x)', 1), ('m.negative(x)', 1), ('negative(x)', 1),
                            ('m.add(m.neg(x), y)', 2), ('a(negative(x), y)', 2)):
            for shape in ((), (7,), (3, 5), (0,)):
                with self.subTest(text=text, shape=shape):
                    values = [rng.integers(-31, 32, size=int(np.prod(shape))).astype(np.float32) * .125
                              for _ in range(arity)]
                    args = [upload(native, v, shape) for v in values]
                    refs = [upload(torch, v, shape) for v in values]
                    fn, ref_fn = expression(text, arity), expression(text, arity, torch)
                    compiled, cache = compile_with_cache(fn, True, False)
                    torch._dynamo.reset()
                    reference = torch.compile(ref_fn, backend='inductor', fullgraph=True, dynamic=False)
                    for _ in range(2):
                        actual = call_without_python(compiled, {fn.__code__}, *args)
                        expected = reference(*refs)
                        # Inductor may turn neg(+0) into +0. Native must retain
                        # eager PyTorch's -0 bits; the frozen gap probe compares
                        # Inductor numerically with zero tolerance as well.
                        self.compare(actual, ref_fn(*refs))
                        torch.testing.assert_close(expected, ref_fn(*refs), rtol=0, atol=0)
                        np.testing.assert_allclose(np.asarray(actual.cpu().tolist(), dtype=np.float32),
                                                   expected.cpu().numpy(), rtol=0, atol=0)
                    self.assertEqual(len(cache.graphs), 1)

    def test_layouts_broadcast_offsets_and_ieee_bits(self):
        rng = np.random.default_rng(198302)
        data = rng.normal(size=420).astype(np.float32)
        views = [lambda x: x[7:414].reshape(11, 37), lambda x: x.select(0, 13),
                 lambda x: x.reshape(1, 7, 60).transpose(0, 1),
                 lambda x: x.reshape(7, 60)[7:7][:, 60:60],
                 lambda x: x[420:].reshape(2, 0, 3)]
        for view in views:
            for policy in POLICIES:
                args = [view(upload(native, data, (420,)))] * 2
                refs = [view(upload(torch, data, (420,)))] * 2
                self.checked(expression('m.add(n(x), y)'), expression('m.add(n(x), y)', module=torch), args, refs, *policy)
        for shape in ((3, 5), (1, 5), (0, 5), (3, 0)):
            matrix, vector = [rng.normal(size=int(np.prod(s))).astype(np.float32) for s in (shape, shape[1:])]
            for reverse in (False, True):
                for policy in POLICIES:
                    text = 'a(y, n(x))' if reverse else 'a(n(x), y)'
                    self.checked(expression(text), expression(text, module=torch),
                                 [upload(native, matrix, shape), upload(native, vector, shape[1:])],
                                 [upload(torch, matrix, shape), upload(torch, vector, shape[1:])], *policy)
        bits = np.array([0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff,
                         0x7f800000, 0xff800000, 0x7fc00000, 0xffc12345], dtype=np.uint32)
        for text, arity in (('n(x)', 1), ('a(x, y)', 2)):
            data = bits.view(np.float32)
            self.checked(expression(text, arity), expression(text, arity, torch),
                         [upload(native, data, data.shape)] * arity,
                         [upload(torch, data, data.shape)] * arity, True, None)

    def test_compositions_helpers_dynamic_squeeze_and_no_replay(self):
        texts = ['m.add(n(x), y).view(-1).neg()', 'n(m.add(x.view(3, 5), y.reshape(3, 5)))',
                 'm.add(x, y).transpose(0, 1).contiguous().negative()',
                 'n(x.transpose(0, 1).contiguous()).transpose(0, 1).contiguous()',
                 'm.add(n(x).relu(), y).sum(1)', 'n(m.add(x, y).relu().sum(1))',
                 'm.add(n(x) @ y.transpose(0, 1).contiguous(), (x @ y.transpose(0, 1).contiguous()).relu())',
                 'n(m.matmul(m.add(x, y), y.transpose(0, 1).contiguous()))']
        rng = np.random.default_rng(198303)
        data = [rng.integers(-4, 5, size=15).astype(np.float32) for _ in range(2)]
        args = [upload(native, v, (3, 5)) for v in data]
        refs = [upload(torch, v, (3, 5)) for v in data]
        for text in texts:
            for policy in POLICIES:
                fn, ref = expression(text), expression(text, module=torch)
                with patch.object(trace, '_execute_operation', side_effect=AssertionError('Python per-node replay')):
                    self.checked(fn, ref, args, refs, *policy)
        source = 'def helper(x):\n    return alias.negative(x)\ndef program(x, y):\n    return m.add(helper(x), n(y))\n'
        for policy in POLICIES:
            with patch.object(trace, '_execute_operation', side_effect=AssertionError('Python replay')):
                self.checked(program(source), program(source, torch), args, refs, *policy)
        # Constant stride, same input rank, changing squeezed output rank.
        for text in ('n(x.squeeze())', 'm.add(x.squeeze(), x.squeeze())',
                     'm.add(n(x).squeeze(), x.squeeze()).relu()'):
            fn = expression(text, 1)
            compiled, cache = compile_with_cache(fn, dynamic=True)
            for shape in ((1, 1), (5, 1), (0, 1), (1, 1)):
                x = native.full(shape, -2.).to('cuda:0')
                tx = torch.full(shape, -2., device='cuda:0')
                self.compare(call_without_python(compiled, {fn.__code__}, x), expression(text, 1, torch)(tx))
            self.assertEqual(len(cache.graphs), 1)

    def test_precise_guards_rebindings_failed_attempts_and_limits(self):
        x = native.full((3, 3), 2.).to('cuda:0')
        tx = torch.full((3, 3), 2., device='cuda:0')
        for text in ('m.mul(x, 2)', 'm.matmul(x, x)', 'm.neg(x)', 'alias.negative(x)', 'm.add(x, x)', 'n(x)'):
            used = {'m.neg(x)': 'neg', 'alias.negative(x)': 'negative', 'm.add(x, x)': 'add'}.get(text)
            for policy in POLICIES:
                fn = expression(text, 1)
                compiled, cache = compile_with_cache(fn, *policy, limit=1)
                expected = expression(text, 1, torch)(tx)
                self.compare(compiled(x), expected)
                for attribute in ('add', 'neg', 'negative'):
                    if attribute == used:
                        continue
                    with patch.object(native, attribute, Hostile()):
                        with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('irrelevant cache miss')):
                            self.compare(compiled(x), expected)
                        cold, cold_cache = compile_with_cache(fn, *policy, limit=1)
                        self.compare(cold(x), expected)
                        self.assertEqual(len(cold_cache.graphs), 1)
                    saved = vars(native).pop(attribute)
                    try:
                        with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('deleted unused field')):
                            self.compare(compiled(x), expected)
                        cold, _ = compile_with_cache(fn, *policy, limit=1)
                        self.compare(cold(x), expected)
                    finally:
                        setattr(native, attribute, saved)
                self.assertEqual(len(cache.graphs), 1)
                if used:
                    before = dict(cache.graphs)
                    with patch.object(native, used, Hostile()), patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('launched')):
                        self.exact_error(trace.CompileTraceUnsupportedError, lambda: compiled(x))
                    self.assertEqual(cache.graphs, before)
                    self.compare(compiled(x), expected)
        # A live canonical operation replacement changes semantics and key.
        for text, owner, name in (('m.add(x, y)', native, 'add'), ('a(x, y)', None, 'a')):
            fn = expression(text)
            compiled, cache = compile_with_cache(fn)
            self.compare(compiled(x, x), tx + tx)
            replacement = patch.object(owner, name, native.matmul) if owner else patch.dict(fn.__globals__, {name: native.matmul})
            with replacement:
                self.compare(compiled(x, x), tx @ tx)
            self.assertEqual(len(cache.graphs), 2)
            limited, variants = compile_with_cache(fn, limit=1)
            limited(x, x)
            replacement = patch.object(owner, name, native.matmul) if owner else patch.dict(fn.__globals__, {name: native.matmul})
            with replacement:
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: limited(x, x))
            self.assertEqual(len(variants.graphs), 1)
        fn = expression('m.add(m.neg(x), alias.negative(y))')
        compiled, cache = compile_with_cache(fn, limit=1)
        compiled(x, x)
        for attribute in ('add', 'neg', 'negative'):
            for value in (Hostile(), native.mul):
                with patch.object(native, attribute, value):
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: compiled(x, x))
            saved = vars(native).pop(attribute)
            try:
                with patch.object(native, '__getattr__', Hostile.fail, create=True):
                    self.exact_error(trace.CompileTraceUnsupportedError, lambda: compiled(x, x))
            finally:
                setattr(native, attribute, saved)
            self.compare(compiled(x, x), -tx + -tx)
        self.assertEqual(len(cache.graphs), 1)
        # Separate global dictionaries with the same code and names stay isolated.
        first = expression('n(x)', 1)
        second = types.FunctionType(first.__code__, {**first.__globals__, 'n': native.negative})
        one, _ = compile_with_cache(first)
        two, _ = compile_with_cache(second)
        one(x); two(x)
        first.__globals__['n'] = Hostile()
        self.exact_error(trace.CompileTraceUnsupportedError, lambda: one(x))
        self.compare(two(x), -tx)

    def test_dynamic_reuse_nested_outputs_and_lifetime(self):
        source = 'def program(x, y):\n    a = m.add(n(x), y)\n    shared = [a, x]\n    return shared, (a, shared, negative(y))\n'
        for policy in POLICIES:
            fn = program(source)
            compiled, cache = compile_with_cache(fn, *policy)
            retained = []
            for size, value in ((5, 2.), (11, -3.), (257, 7.), (5, 13.)):
                x = native.full((size,), value).to('cuda:0')
                y = native.full((size,), value * 2).to('cuda:0')
                result = call_without_python(compiled, {fn.__code__}, x, y)
                self.assertIs(result[0], result[1][1])
                self.assertIs(result[0][0], result[1][0])
                self.assertIs(result[0][1], x)
                self.assertIsNot(result[0][0], x)
                self.assertEqual(result[0][0].cpu().tolist(), [value] * size)
                retained.append((result[0][0], size, value))
            self.assertEqual(len(cache.graphs), 1 if policy[1] else 3)
            del x, y, result, compiled, cache
            gc.collect()
            for value, size, expected in retained:
                self.assertEqual(value.cpu().tolist(), [expected] * size)

    def test_prevalidation_exact_errors_unused_inputs_and_hooks(self):
        x = native.ones((3, 5)).to('cuda:0')
        fn = expression('m.add(n(x), y)')
        compiled, cache = compile_with_cache(fn, dynamic=True)
        compiled(x, x)
        graph = next(iter(cache.graphs.values()))
        bad_graphs = [replace(graph, output='missing'), replace(graph, output_metadata=None)]
        node = graph.operations[-1]
        bad_graphs += [replace(graph, operations=(*graph.operations[:-1], replace(node, **change)))
                       for change in ({'target': 'subtract'}, {'inputs': ('missing',)},
                                      {'metadata': replace(node.metadata, requires_grad=True)},
                                      {'metadata': replace(node.metadata, device='cuda:1')})]
        # Include single-node outputs: validation must precede direct native hooks too.
        single = bytecode.lower_compile_graph(expression('n(x)', 1), (trace._metadata_from_native_tensor(x),))
        bad_graphs.append(replace(single, output='missing'))
        with ExitStack() as stack:
            for owner, name in ((trace, '_execute_operation'), (trace._native, '_compile_trace_cuda_graph'),
                                (trace._native, '_compile_trace_binary'), (trace._native, '_compile_trace_unary')):
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError('native execution before validation')))
            for malformed in bad_graphs:
                self.exact_error(trace.CompileTraceUnsupportedError,
                                 lambda: malformed.forward(*((x,) * len(malformed.inputs))))
            for args in ((x, x.t()), (x, x.cpu()), (x, x[:, 1:4])):
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: compiled(*args))
            unused = native.compile(expression('n(x)'), backend='eager', fullgraph=True)
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: unused(x, x.cpu()))
            for text in ('m.add(x)', 'n(x, y)', 'm.add(x, 1)', 'n(1)', 'm.add(x, y, alpha=1)',
                         'm.neg(input=x)', 'm.add(x, y, out=None)'):
                rejected, empty = compile_with_cache(expression(text))
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: rejected(x, x))
                self.assertEqual(empty.graphs, {})
            self.exact_error(TypeError, lambda: compiled(x, 2))
        self.assertEqual(len(cache.graphs), 1)
        from torch_rs.overrides import TorchFunctionMode
        class Mode(TorchFunctionMode):
            __torch_function__ = Hostile.fail
        with Mode():
            self.exact_error(trace.CompileTraceUnsupportedError, lambda: compiled(x, x))
        # Native Tensor is sealed; capture does not introduce subclass support.
        self.exact_error(TypeError, lambda: type('TensorSubclass', (native.Tensor,), {}))
        self.exact_error(TypeError, lambda: compiled(Hostile(), x))
        for text in ('m.add(x, y)', 'n(x)'):
            fn = expression(text)
            rejected, empty = compile_with_cache(fn)
            with patch.object(trace, '_execute_operation', side_effect=AssertionError('CPU execution')):
                self.exact_error(trace.CompileTraceUnsupportedError,
                                 lambda: rejected(native.ones(3, requires_grad=True), native.ones(3)))
            self.assertEqual(empty.graphs, {})
        with patch.object(native.Tensor, '__torch_function__', Hostile.fail, create=True):
            self.assertEqual(compiled(x, x).cpu().tolist(), [[0.] * 5] * 3)
        for attribute in ('neg', 'negative', 'add'):
            with patch.object(native.Tensor, attribute, Hostile.fail):
                self.exact_error(trace.CompileTraceUnsupportedError, lambda: compiled(x, x))
        # No installed reference import or original Python execution is needed.
        source = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('imported PyTorch')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
from torch_rs import neg as n
def helper(x):
    return n(x)
def f(x):
    return m.add(helper(x), m.negative(x))
x = m.tensor([1., -3.]).to('cuda:0')
assert m.compile(f, backend='eager', fullgraph=True)(x).cpu().tolist() == [-2., 6.]
'''
        result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class ModuleArithmeticDeviceTests(ExactErrors, Comparison, unittest.TestCase):
    def test_restoration_success_rejection_unused_input_and_lifetime(self):
        lib, previous = runtime(), torch.cuda.current_device()
        try:
            for policy in POLICIES:
                compiled, cache = compile_with_cache(expression('m.add(n(x), y)'), *policy)
                for current, target in ((1, 0), (0, 1), (1, 0)):
                    torch.cuda.set_device(current)
                    x = native.full((3, 5), -2.).to(f'cuda:{target}')
                    result = compiled(x, x)
                    self.compare(result, torch.full((3, 5), 0., device=f'cuda:{target}'))
                    ordinal = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
                    wrong = native.ones((3, 5)).to(f'cuda:{current}')
                    before = dict(cache.graphs)
                    with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('launched')):
                        self.exact_error(trace.CompileTraceUnsupportedError, lambda: compiled(x, wrong))
                        unused = native.compile(expression('n(x)'), backend='eager', fullgraph=True)
                        self.exact_error(trace.CompileTraceUnsupportedError, lambda: unused(x, wrong))
                        with patch.object(native, 'add', Hostile()):
                            self.exact_error(trace.CompileTraceUnsupportedError, lambda: compiled(x, x))
                    self.assertEqual(cache.graphs, before)
                    del result, wrong, x
                    gc.collect()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
                self.assertEqual(len(cache.graphs), 2)
        finally:
            torch.cuda.set_device(previous)
