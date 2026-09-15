"""Independent expression generation and native default-JIT regression evidence."""
import ast
import concurrent.futures
from contextlib import ExitStack
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
import weakref
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, _compile_trace
from torch_rs import torch_rs as bridge


def program(source, framework=native, **bindings):
    namespace = {"fw": framework, **bindings}
    exec(source, namespace)
    return namespace["f"]


def lower(fn, arity=1):
    descriptor = frontend.analyze(fn, arity)
    _, values = frontend.resolve(fn, descriptor)
    return frontend.lower(descriptor, values, arity).graph


def cache(compiled):
    return compiled._torch_rs_pointwise_cache


def kernels(compiled):
    return list(cache(compiled).executors.values())


def kernel(compiled):
    return next(iter(cache(compiled).executors.values()))


def mock_pointwise_executor(run, metadata=None, retained_bytes=0):
    """Adapt frontend-only launch mocks to immutable preparation, not CUDA.

    The fake retains checked metadata only; preparation does not retain tensors.
    Forward through executor.run so existing injected launch failures still test
    the real frontend's post-run/reconstruction publication boundary.
    """
    executor = types.SimpleNamespace(run=run)

    def prepare(tensors, numerical_hint, output_order):
        read_metadata = metadata or bridge._compile_trace_tensor_metadata
        shapes = tuple(read_metadata(tensor)[0] for tensor in tensors)
        return types.SimpleNamespace(
            input_shapes=shapes,
            retained_bytes=retained_bytes,
            run=lambda inputs, scalars: executor.run(inputs, scalars, numerical_hint, output_order),
        )

    executor.prepare = prepare
    return executor


def diagnostic_capture_selectors():
    # Load only the capture accessor; importing a capture script launches CUDA
    # and writes artifacts. These are the exact maintained script definitions.
    root = Path(__file__).resolve().parents[1]
    for relative in ('docs/diagnostics/compile-pointwise-broadcast/capture_bounded.py',
                     'docs/diagnostics/compile-pointwise-jit/capture.py'):
        source = ast.parse((root / relative).read_text(), filename=relative)
        selected = [node for node in source.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'dispatched_kernel']
        assert len(selected) == 1
        namespace = {}
        exec(compile(ast.Module(body=selected, type_ignores=[]), relative, 'exec'), namespace)
        yield relative, namespace['dispatched_kernel']


def gpu_dispatch_diagnostic():
    root = Path(__file__).resolve().parents[1]
    path = root / 'docs/diagnostics/compile-pointwise-positional/warm-dispatch/gpu-dispatch.py'
    spec = importlib.util.spec_from_file_location('_synthetic_gpu_dispatch_check', path)
    diagnostic = importlib.util.module_from_spec(spec)
    # The stdlib-only consumer has no import-time GPU execution. Keep the
    # isolated module out of sys.modules and do not replace production owners.
    spec.loader.exec_module(diagnostic)
    return diagnostic, path


def custom_globals_program(source):
    effects = []
    class Globals(dict):
        def __contains__(self, key):
            effects.append(('contains', key))
            return super().__contains__(key)
        def __getitem__(self, key):
            effects.append(('getitem', key))
            return super().__getitem__(key)
    namespace = Globals(scale=0.713)
    exec(source, namespace)
    fn = dict.__getitem__(namespace, 'f')
    effects.clear()
    return fn, effects


def custom_closure_program():
    effects = []
    class Closure(tuple):
        def __bool__(self):
            effects.append('closure truthiness')
            return True
        def __iter__(self):
            effects.append('closure iteration')
            return super().__iter__()
    scale = 0.713
    def fn(x):
        return x * scale
    return types.FunctionType(fn.__code__, fn.__globals__, closure=Closure(fn.__closure__)), effects


def two_device_reservation():
    # Visible device ordinals are remapped to 0 and 1 inside the process, even
    # when the caller reserves another physical pair or selects GPU UUIDs.
    selected = tuple(part.strip() for part in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    return len(selected) == 2 and len(set(selected)) == 2 and all(part and part != '-1' for part in selected)


class Admission(unittest.TestCase):
    def tearDown(self):
        native.compiler.reset()

    def test_diagnostic_capture_selectors_follow_successful_executor_lru(self):
        actual_cache = frontend._state.NativeEagerCompileCache()
        compiled = types.SimpleNamespace(_torch_rs_pointwise_cache=actual_cache)
        # Logical entries deliberately have the current non-tuple value layout.
        actual_cache.graphs[object()] = frontend.Specialization({}, (), {}, {})
        first, second, third = object(), object(), object()
        actual_cache.executors.update(first=first, second=second, third=third)
        for path, select in diagnostic_capture_selectors():
            with self.subTest(capture=path, call='new module'):
                self.assertIs(select(compiled), third)
        # Successful warm use moves an older module to the end. The selected
        # module is neither the first entry nor the most recently created one.
        actual_cache.executors['second'] = actual_cache.executors.pop('second')
        for path, select in diagnostic_capture_selectors():
            with self.subTest(capture=path, call='warm revisit'):
                self.assertIs(select(compiled), second)
        actual_cache.clear()
        self.assertFalse(actual_cache.executors)

    def _verify_synthetic_gpu_dispatch_reports(self, mismatches):
        root = Path(__file__).resolve().parents[1]
        diagnostic, path = gpu_dispatch_diagnostic()
        script_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        target = root / 'target'
        target.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='dispatch-verifier-', dir=target) as temporary:
            directory = Path(temporary)
            reports = []
            for index, (build, implementation) in enumerate(diagnostic.ORDER):
                cases = []
                for name, program_text, _ in diagnostic.CASES:
                    traversals = []
                    for phase, count in (('setup', 2), ('warmup', 5), ('sample', 17)):
                        for traversal_index in range(count):
                            traversals.append({'phase': phase, 'index': traversal_index,
                                               'outputs': [{'shape': [1], 'stride': [1],
                                                            'dtype': 'torch.float32', 'device': 'cuda:0',
                                                            'requiresGrad': False, 'values': [float(1.0).hex()]}]})
                    cases.append({'name': name, 'program': program_text,
                                  'inputs': [{'synthetic': True}], 'states': [{}],
                                  'traversals': traversals, 'medianNsPerCall': 1})
                for leg_index, case_index, traversal_index in mismatches:
                    if index == leg_index:
                        cases[case_index]['traversals'][traversal_index]['outputs'][0]['values'] = [float(2.0).hex()]
                report = {'passed': True, 'legIndex': index, 'buildLabel': build,
                          'implementation': implementation, 'scriptSha256': script_hash,
                          'diagnosticVersion': diagnostic.VERSION,
                          'synchronization': {'path': 'synthetic-runtime', 'sha256': 'a' * 64,
                                              'version': 13000, 'api': 'cudaDeviceSynchronize',
                                              'logicalDevice': 0},
                          'gpuUuid': 'GPU-synthetic-fixture', 'source': {'commit': build},
                          'startedAt': f'{index:02d}:00', 'finishedAt': f'{index:02d}:01',
                          'cases': cases}
                output = directory / f'leg-{index}.json'
                output.write_text(json.dumps(report))
                reports.append(output)
            record = {'scriptSha256': script_hash}
            args = types.SimpleNamespace(reports=reports)
            if mismatches:
                with self.assertRaisesRegex(AssertionError, 'retained comparison/contract failures'):
                    diagnostic.verify(args, record, directory)
            else:
                diagnostic.verify(args, record, directory)
            self.assertEqual(len(record['reports']), 16)
            self.assertEqual([item['sha256'] for item in record['reports']],
                             [hashlib.sha256(path.read_bytes()).hexdigest() for path in reports])
            self.assertEqual(len(record['pairs']), 56)
            self.assertEqual(sum(row['outputsCompared'] for row in record['pairs']), 56 * 24)
            self.assertTrue(record['pairs'][-1]['passed'])
            return record

    def test_gpu_dispatch_verifier_preserves_ieee_values_and_jit_tolerance(self):
        diagnostic, _ = gpu_dispatch_diagnostic()
        def observation(values):
            return {'shape': [len(values)], 'stride': [1], 'dtype': 'torch.float32',
                    'device': 'cuda:0', 'requiresGrad': False,
                    'values': diagnostic.encoded_values(values)}
        values = [0.0, -0.0, float('nan'), float('inf'), -float('inf')]
        record = observation(values)
        restored = json.loads(json.dumps(record, allow_nan=False))
        self.assertEqual(restored, record)
        self.assertEqual(restored['values'][:2], ['0x0.0p+0', '-0x0.0p+0'])
        diagnostic.close_outputs(restored, record)
        for actual, expected in ((0.0, -0.0), (-0.0, 0.0), (float('inf'), -float('inf')),
                                 (float('nan'), 1.0), (1.0, float('nan')), (1.0, float('inf'))):
            with self.subTest(actual=actual, expected=expected), self.assertRaises(AssertionError):
                diagnostic.close_outputs(observation([actual]), observation([expected]))
        # This lies above max(atol, rtol*abs(ref)) but below the existing JIT
        # atol+rtol*abs(ref) bound. A nearby value above that sum must fail.
        diagnostic.close_outputs(observation([1.0 + 1.05e-5]), observation([1.0]))
        with self.assertRaises(AssertionError):
            diagnostic.close_outputs(observation([1.0 + 1.11e-5]), observation([1.0]))
        changed_metadata = dict(record, stride=[2])
        with self.assertRaises(AssertionError):
            diagnostic.close_outputs(changed_metadata, record)

    def test_gpu_dispatch_verifier_checks_all_synthetic_pairs(self):
        record = self._verify_synthetic_gpu_dispatch_reports(())
        self.assertTrue(record['passed'])
        self.assertEqual(record['comparisonFailures'], [])
        self.assertTrue(all(row['passed'] for row in record['pairs']))

    def test_gpu_dispatch_verifier_retains_all_synthetic_numerical_failures(self):
        # Two mismatches in the first history and one in a different pair must
        # not hide later outputs, histories or a final passing comparison.
        record = self._verify_synthetic_gpu_dispatch_reports(((0, 0, 7), (0, 0, 8), (5, 3, 9)))
        self.assertFalse(record['passed'])
        self.assertEqual(len(record['comparisonFailures']), 3)
        self.assertEqual(sum(not row['passed'] for row in record['pairs']), 2)
        locations = [failure['location'] for failure in record['comparisonFailures']]
        self.assertIn('pair[0,1].literal_unary.traversal[7].output[0]', locations)
        self.assertIn('pair[0,1].literal_unary.traversal[8].output[0]', locations)
        self.assertIn('pair[4,5].repeated_alias.traversal[9].output[0]', locations)

    def test_gpu_dispatch_synchronizer_propagates_runtime_errors(self):
        diagnostic, _ = gpu_dispatch_diagnostic()
        synchronizer = object.__new__(diagnostic.CudaSynchronizer)
        from unittest.mock import Mock
        synchronizer.runtime = types.SimpleNamespace(cudaSetDevice=Mock(return_value=0),
                                                     cudaDeviceSynchronize=Mock(return_value=0))
        synchronizer()
        synchronizer.runtime.cudaSetDevice.assert_called_once_with(0)
        synchronizer.runtime.cudaDeviceSynchronize.assert_called_once_with()
        synchronizer.runtime.cudaSetDevice.return_value = 10
        with self.assertRaisesRegex(RuntimeError, 'synchronization error 10'):
            synchronizer()
        # A failed device selection must not synchronize another device.
        self.assertEqual(synchronizer.runtime.cudaDeviceSynchronize.call_count, 1)
        synchronizer.runtime.cudaSetDevice.return_value = 0
        synchronizer.runtime.cudaDeviceSynchronize.return_value = 700
        with self.assertRaisesRegex(RuntimeError, 'synchronization error 700'):
            synchronizer()

    def test_two_device_reservation_accepts_explicit_distinct_pairs(self):
        for mask, expected in (('0,1', True), ('6,7', True), ('GPU-a,GPU-b', True),
                               ('', False), ('0', False), ('0,1,2', False),
                               ('0,0', False), ('0,', False), (',1', False), ('-1,1', False)):
            with self.subTest(mask=mask), patch.dict(os.environ, CUDA_VISIBLE_DEVICES=mask):
                self.assertEqual(two_device_reservation(), expected)
        with patch.dict(os.environ):
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            self.assertFalse(two_device_reservation())

    def test_typed_ir_retains_reused_nodes_and_rounds_constants(self):
        fn = program('def f(x, y):\n a = fw.sin(x)\n b = a * a\n return (b - y.cos()) + 0.10000000000001')
        graph = lower(fn, 2)
        source = bridge._pointwise_plan(graph.nodes, graph.outputs, graph.inputs)
        self.assertEqual(source.count('sinf('), 1)
        self.assertIn('fmaf(v2, v2, -v4)', source)
        self.assertIn('cosf(', source)
        self.assertNotIn('__sinf', source)
        self.assertIn('0x3dcccccdu', source)
        self.assertEqual(frontend.scalar_bits(-0.0), 0x8000000000000000)

    def test_rejects_whole_graph_without_executing_user_objects(self):
        effects = []
        class Trap:
            def __getattr__(self, name):
                effects.append(name)
                raise AssertionError('executed user attribute')
            def __float__(self):
                effects.append('float')
                raise AssertionError('executed user conversion')
        sources = [
            'def f(x):\n effects.append(1)\n return -x',
            'def f(x):\n return x if x else -x',
            'def f(x):\n return x.sum()', 'def f(x):\n return x.reshape(2, 2)',
            'def f(x):\n return x + trap', 'def f(x):\n return trap.sin(x)',
            'def f(x):\n x += x\n return x', 'def f(x):\n return x, x',
            'def f(x):\n return x / 2', 'def f(x):\n return x',
        ]
        for source in sources:
            with self.subTest(source=source), self.assertRaises(NotImplementedError):
                lower(program(source, effects=effects, trap=Trap()))
        self.assertEqual(effects, [])

    def test_native_ir_validation_is_hardware_free(self):
        for nodes, output in [((), 0), ((("neg", 0, 0, 0),), 0),
            ((("input", 2, 0, 0),), 0), ((("input", 0, 0, 0), ("wat", 0, 0, 0)), 1),
            ((("constant", 0, 0, 0), ("sin", 0, 0, 0)), 1),
            ((("input", 0, 0, 0), ("neg", 0, 0, 0)), 7)]:
            with self.subTest(nodes=nodes), self.assertRaises((ValueError, NotImplementedError)):
                bridge._pointwise_source(nodes, (output,), 1)
        with self.assertRaises(TypeError):
            bridge._pointwise_source((("input", True, 0, 0),), (0,), 1)

    def test_custom_globals_rejected_without_lookup_hooks(self):
        for expression in ('x.sum() + scale', 'x * scale', '-x'):
            fn, effects = custom_globals_program('def f(x):\n return ' + expression)
            for _ in range(2):
                with self.subTest(expression=expression), self.assertRaisesRegex(
                        NotImplementedError, 'globals must be an exact dict'):
                    lower(fn)
                self.assertEqual(effects, [])

    def test_rejected_namespace_key_is_not_retained_by_traceback(self):
        class Key:
            pass

        for which in ('globals', 'builtins'):
            with self.subTest(namespace=which):
                fn = types.FunctionType((lambda x: x).__code__, {'__builtins__': {}})
                namespace = getattr(fn, '__' + which + '__')
                key = Key()
                reference = weakref.ref(key)
                namespace[key] = None
                # assertRaises clears tracebacks; deliberately retain this one.
                try:
                    frontend.validate_namespaces(fn)
                except NotImplementedError as error:
                    caught = error
                else:
                    self.fail('non-string namespace key was accepted')
                self.assertIn(which + ' keys must be exact strings', str(caught))
                namespace.clear()
                del key
                gc.collect()
                self.assertIsNotNone(caught.__traceback__)
                self.assertIsNone(reference())

    def test_custom_closure_rejected_without_container_hooks(self):
        fn, effects = custom_closure_program()
        with self.assertRaisesRegex(NotImplementedError, 'closure must be an exact tuple'):
            lower(fn)
        self.assertEqual(effects, [])

    def test_positional_operator_spellings_and_integer_bytecode(self):
        fn = program('def f(x, y):\n a = b = fw.add(x, y)\n return fw.subtract(a, -3).mul(2) + b.__rsub__(0.713)')
        graph = lower(fn, 2)
        source = bridge._pointwise_plan(graph.nodes, graph.outputs, 2)
        self.assertEqual(sum(node[0] == 'add' for node in graph.nodes), 2)
        self.assertEqual(sum(node[0] == 'sub' for node in graph.nodes), 2)
        self.assertIn('0x40000000u', source)

    def test_defaults_disable_backend_resolution_and_cpu_rejection(self):
        fn = program('def f(x):\n return -x')
        self.assertIs(native.compile(fn, disable=True), fn)
        compiled = native.compile(fn)
        self.assertEqual(compiled._torch_rs_compile_backend, 'inductor')
        with self.assertRaisesRegex(NotImplementedError, 'default backend does not compile CPU tensors'):
            compiled(native.tensor([1.]))
        with self.assertRaisesRegex(NotImplementedError, 'requires exact native CUDA float32 Tensor inputs'):
            compiled(1)
        self.assertEqual(native.compile(fn, backend='eager')(native.tensor([2.])).tolist(), [-2.])

    def test_default_rejections_explain_supported_backend_without_executing_user_code(self):
        effects = []
        class Trap:
            def __repr__(self):
                effects.append('repr')
                return 'trap'
            def __torch_function__(self, *args, **kwargs):
                effects.append('dispatch')
                raise AssertionError('unsupported input must not dispatch')
        fn = program('def f(x):\n effects.append("body")\n return -x', effects=effects)
        compiled = native.compile(fn)
        for value in (native.tensor([1.]), Trap(), 1, None):
            with self.subTest(kind=type(value).__name__), self.assertRaises(NotImplementedError) as caught:
                compiled(value)
            message = str(caught.exception)
            self.assertTrue(message.startswith('torch.compile():'), message)
            self.assertIn('default backend', message)
            self.assertIn('docs/compile-pointwise-jit.md', message)
            self.assertNotIn("only backend='eager'", message)
            self.assertNotIn('CUDA compilation', message)
        self.assertEqual(effects, [])
        self.assertEqual(len(cache(compiled).graphs), 0)


def available():
    try:
        import torch
        return native.cuda.is_available() and torch.cuda.is_available()
    except ImportError:
        return False


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class Hardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def upload(self, values, shape, fw=native):
        return fw.tensor(values, dtype=fw.float32).reshape(shape).to('cuda:0')

    def compare(self, actual, expected, exact=False, zero_sign=True):
        torch = self.torch
        host = torch.tensor(actual.cpu().tolist(), dtype=torch.float32).reshape(tuple(expected.shape))
        torch.testing.assert_close(host, expected.cpu(), rtol=0 if exact else 1e-5,
                                   atol=0 if exact else 1e-6, equal_nan=True)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), str(expected.device))
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        # IEEE zeros have observable sign even when numerical closeness succeeds.
        zeros = (expected.cpu() == 0) & (host == 0)
        if zero_sign:
            self.assertTrue(torch.equal(torch.signbit(host)[zeros], torch.signbit(expected.cpu())[zeros]))

    def without_replay(self, fn, compiled, args):
        def reject(frame, event, arg):
            if event == 'call' and frame.f_code is fn.__code__:
                raise AssertionError('original Python body executed')
            if event == 'c_call' and getattr(arg, '__name__', '') in ('sin', 'cos', 'relu', '__add__', '__sub__', '__mul__', '__neg__'):
                raise AssertionError('eager tensor kernel called')
        old = sys.getprofile()
        try:
            sys.setprofile(reject)
            with ExitStack() as guards:
                guards.enter_context(patch.object(_compile_trace, '_execute_operation', side_effect=AssertionError('eager graph replay')))
                guards.enter_context(patch.object(native, '_execute_native_eager_compile_graph', side_effect=AssertionError('eager backend')))
                for name in ('_compile_trace_cuda_graph', '_compile_trace_unary', '_compile_trace_binary', '_compile_trace_scalar'):
                    guards.enter_context(patch.object(bridge, name, side_effect=AssertionError('eager native chain')))
                return compiled(*args)
        finally:
            sys.setprofile(old)

    def test_gpu_dispatch_real_native_and_default_histories(self):
        diagnostic, path = gpu_dispatch_diagnostic()
        root = Path(__file__).resolve().parents[1]
        # Preserve all observations, including failures, beneath the worktree.
        (root / 'target').mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix='dispatch-smoke-', dir=root / 'target'))
        script = r"""
import importlib, importlib.util, sys, traceback
from pathlib import Path
path, directory, implementation = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
spec = importlib.util.spec_from_file_location('dispatch_smoke', path)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
directory.mkdir()
diagnostic.environment(diagnostic.ROOT, directory)
fw = importlib.import_module('torch_rs' if implementation == 'native' else 'torch')
fw.set_num_threads(1)
barrier = diagnostic.CudaSynchronizer()
record = {'implementation': implementation, 'scriptSha256': diagnostic.sha(path.read_bytes()),
          'synchronization': barrier.metadata, 'gpuBefore': diagnostic.inventory(),
          'cases': [], 'frameworkPath': fw.__file__, 'executable': sys.executable}
calls = 0
def synchronize():
    global calls
    calls += 1
    barrier()
for name, source, kind in diagnostic.CASES:
    row = {'name': name, 'passed': False}
    record['cases'].append(row)
    try:
        diagnostic.history(fw, source, kind, row, synchronize)
        row['passed'] = True
    except Exception:
        row['failure'] = traceback.format_exc()
record.update(synchronizationCalls=calls, torchImported='torch' in sys.modules,
              gpuAfter=diagnostic.inventory(), runtime=diagnostic.runtime_identity())
diagnostic.write(directory/'report.json.gz', record)
"""
        reports = []
        for implementation in ('native', 'reference'):
            child = directory / implementation
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            env.pop('PYTHONHOME', None)
            result = subprocess.run([sys.executable, '-I', '-B', '-c', script,
                                     str(path), str(child), implementation], env=env,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (directory / (implementation + '.log')).write_text(result.stdout)
            self.assertEqual(result.returncode, 0, result.stdout)
            reports.append(diagnostic.read(child / 'report.json.gz'))
        print(f'Diagnostic smoke observations retained at {directory}', flush=True)
        self.assertFalse(reports[0]['torchImported'])
        self.assertEqual(reports[0]['synchronization'], reports[1]['synchronization'])
        for report in reports:
            self.assertEqual(report['synchronizationCalls'], 7 * 24 * 2)
            self.assertEqual(len(report['cases']), 7)
            for row in report['cases']:
                with self.subTest(implementation=report['implementation'], history=row['name']):
                    self.assertTrue(row['passed'], row.get('failure'))
                    self.assertEqual(len(row['traversals']), 24)
                    self.assertEqual(len(row['sampleNs']), 17)
        for left, right in zip(reports[0]['cases'], reports[1]['cases']):
            self.assertEqual(left['inputs'], right['inputs'])
            self.assertEqual(left['states'], right['states'])
            self.assertEqual(len(left['traversals']), len(right['traversals']))
            for a, b in zip(left['traversals'], right['traversals']):
                self.assertEqual((a['phase'], a['index']), (b['phase'], b['index']))
                self.assertEqual(len(a['outputs']), len(b['outputs']))
                for actual, expected in zip(a['outputs'], b['outputs']):
                    diagnostic.close_outputs(actual, expected)

    def test_diagnostic_capture_selectors_follow_actual_shape_revisit(self):
        fn = program('def f(x, y):\n return (x.relu() - y.relu()).relu()')
        compiled = native.compile(fn)
        dispatched = []
        for rows, columns in ((7, 11), (3, 17), (7, 11)):
            x = self.upload([1.] * rows, (rows, 1))
            y = self.upload([0.25] * columns, (columns,))
            actual = self.without_replay(fn, compiled, (x, y))
            self.assertEqual(actual.cpu().tolist(), [[0.75] * columns] * rows)
            selected = [select(compiled) for _, select in diagnostic_capture_selectors()]
            self.assertIs(selected[0], selected[1])
            dispatched.append(selected[0])
        self.assertIsNot(dispatched[0], dispatched[1])
        self.assertIs(dispatched[0], dispatched[2])
        self.assertEqual(len(cache(compiled).executors), 2)

    def test_generated_expression_trees_default_inductor_and_fresh_data(self):
        rng = random.Random(926317)
        sources = []
        for _ in range(12):
            statements = ['def f(x, y):', ' a = fw.sin(x)', ' b = a * a']
            terms = ['x', 'y', 'a', 'b']
            for index in range(5):
                left = rng.choice(terms)
                if rng.randrange(2):
                    expression = f'{left} {rng.choice(["+", "-", "*"])} {rng.choice(terms + [repr(rng.uniform(-1.5, 1.5))])}'
                else:
                    expression = f'{left}.{rng.choice(["sin", "cos", "relu", "neg"])}()'
                term = f'v{index}'
                statements.append(f' {term} = {expression}')
                terms.append(term)
            statements.append(' return v4 + b * 0.137219')
            sources.append('\n'.join(statements))
        for index, source in enumerate(sources):
            fn = program(source)
            reference_fn = program(source, self.torch)
            compiled, reference = native.compile(fn), self.torch.compile(reference_fn)
            for shape in [(37,), (3, 19), (2, 3, 5), (11, 263), (), (0, 7)]:
                for changed in range(2):
                    count = math.prod(shape)
                    values = [[rng.uniform(-3, 3) for _ in range(count)] for _ in range(2)]
                    args = [self.upload(v, shape) for v in values]
                    refs = [self.upload(v, shape, self.torch) for v in values]
                    expected = reference(*refs)
                    actual = self.without_replay(fn, compiled, args)
                    with self.subTest(index=index, shape=shape, changed=changed):
                        self.compare(actual, expected)
                        self.compare(actual, reference_fn(*refs))
                        for arg, ref in zip(args, refs):
                            self.compare(arg, ref, exact=True)
                            self.assertIsNot(actual, arg)
                            if count:
                                self.assertNotEqual(actual.data_ptr(), arg.data_ptr())
            # The generalized rank-two guard accepts the later (11, 263)
            # shape. Compare logical entries with this persistent reference,
            # rather than counting every distinct concrete metadata tuple.
            reference_entries = self.torch._dynamo.eval_frame._debug_get_cache_entry_list(reference_fn.__code__)
            self.assertEqual(len(reference_entries), 5)
            self.assertEqual(len(cache(compiled).graphs), len(reference_entries))
            self.assertEqual(len({id(entry) for entry in kernels(compiled)}), 1)
            generated = kernel(compiled)
            self.assertIn('torch_rs_pointwise', generated.ptx)
            self.assertEqual(generated.ptx.count('.visible .entry'), 1)
            self.assertIn('--fmad=true', generated.options)
            self.assertNotIn('--use_fast_math', generated.options)
            self.assertGreaterEqual(generated.nvrtc_version[0], 12)

    def test_ieee_values_and_scalar_rounding(self):
        values = [0., -0., float('inf'), -float('inf'), float('nan'), 1e30, -1e30,
                  1e-40, -1e-40, 16777216., -16777216., 1.25]
        for expression in ['-x', 'x.relu()', 'x.sin()', 'x.cos()', 'x * -0.0',
                           'x + 0.10000000000001', '1.0000000596046448 - x', 'x * 1.0000000596046448']:
            fn = program('def f(x):\n return ' + expression)
            ref_fn = program('def f(x):\n return ' + expression, self.torch)
            with self.subTest(expression=expression):
                actual = native.compile(fn)(self.upload(values, (len(values),)))
                tx = self.upload(values, (len(values),), self.torch)
                expected = self.torch.compile(ref_fn)(tx)
                eager = ref_fn(tx)
                self.compare(actual, expected)
                # Default Inductor and eager disagree on these two IEEE zero
                # cases. Check their exact difference, and follow Inductor.
                mismatches = ((expected == 0) & (eager == 0)
                              & (self.torch.signbit(expected) != self.torch.signbit(eager)))
                expected_indices = [0] if expression == '-x' else [1] if expression == 'x.relu()' else []
                self.assertEqual(mismatches.nonzero().flatten().tolist(), expected_indices)
                self.compare(actual, eager, zero_sign=not expected_indices)

    def test_offsets_constants_code_bindings_guards_and_reset(self):
        fn = program('def f(x):\n return fw.sin(x) * scale', scale=0.37)
        compiled = native.compile(fn)
        x = self.upload(list(range(17)), (17,))[2:13]
        tx = self.upload(list(range(17)), (17,), self.torch)[2:13]
        first = compiled(x)
        original_kernel = kernel(compiled)
        for scale in [0.37, -1.91237, -0.0]:
            fn.__globals__['scale'] = scale
            self.compare(compiled(x), tx.sin() * scale)
        # The first changed float promotes to one reusable runtime parameter.
        self.assertEqual(len(cache(compiled).graphs), 2)
        fn.__globals__['fw'] = object()
        with self.assertRaises(NotImplementedError):
            compiled(x)
        fn.__globals__['fw'] = native
        fn.__code__ = program('def f(x):\n return x.cos()').__code__
        self.compare(compiled(x), tx.cos())
        with patch.object(native.Tensor, 'sin', lambda x: x):
            with self.assertRaises(NotImplementedError):
                compiled(x)
        native.compiler.reset()
        self.assertEqual(cache(compiled).graphs, {})
        self.compare(compiled(x), tx.cos())
        self.assertIsNot(original_kernel, kernel(compiled))
        self.compare(first, tx.sin() * 0.37)
        alias_fn = program('def f(x):\n return op(x)', op=native.sin)
        alias_compiled = native.compile(alias_fn)
        self.compare(alias_compiled(x), tx.sin())
        alias_fn.__globals__['op'] = native.cos
        self.compare(alias_compiled(x), tx.cos())
        alias_fn.__globals__['op'] = lambda x: x
        with self.assertRaises(NotImplementedError):
            alias_compiled(x)

    def test_contraction_rounding_and_shared_intermediates(self):
        values = [3e38, -3e38, 16777216., -16777216., 1.0000001192092896,
                  0., -0., float('inf'), -float('inf'), float('nan')]
        nx = self.upload(values, (len(values),))
        tx = self.upload(values, (len(values),), self.torch)
        for body in ['return x * 2.0 - x',
                     'return x * 1.0000001192092896 - x',
                     'a = x * 1.0000001192092896\n return (a - x) + a',
                     'return (x + 1.0) - x']:
            source = 'def f(x):\n ' + body
            fn, reference_fn = program(source), program(source, self.torch)
            with self.subTest(body=body):
                compiled = native.compile(fn)
                expected = self.torch.compile(reference_fn)(tx)
                self.compare(compiled(nx), expected)
                # A warm call must enter cached code, not analyze or lower nodes.
                with patch.object(frontend, 'lower', side_effect=AssertionError('warm lowering')), \
                     patch.object(frontend, 'analyze', side_effect=AssertionError('warm analysis')), \
                     patch.object(bridge, '_pointwise_compile', side_effect=AssertionError('warm NVRTC')):
                    self.compare(compiled(self.upload(values, (len(values),))), expected)
                if body == 'return x * 2.0 - x':
                    self.assertTrue(self.torch.isinf(reference_fn(tx)[:2]).all())
                    self.assertTrue(self.torch.isfinite(expected[:2]).all())

    def test_invalid_metadata_failure_retry_and_concurrency(self):
        fn = program('def f(x, y):\n return (x * y).sin() - x')
        compiled = native.compile(fn)
        x = self.upload([1., 2., 3., 4.], (2, 2))
        with patch.object(bridge, '_pointwise_compile', side_effect=AssertionError('compiled invalid graph')):
            for args in [(x, x.t()), (x, native.ones(2, 2))]:
                with self.assertRaises(NotImplementedError):
                    compiled(*args)
        self.assertFalse(cache(compiled).graphs)
        with patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/torch-rs-test-nvrtc'):
            with self.assertRaisesRegex(RuntimeError, 'NVRTC'):
                compiled(x, x)
        self.assertFalse(cache(compiled).graphs)
        def work(index):
            y = self.upload([float(index)] * 4, (2, 2))
            return compiled(x, y).cpu().tolist()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(work, range(12)))
        for index, output in enumerate(outputs):
            for value, original in zip(sum(output, []), [1., 2., 3., 4.]):
                self.assertAlmostEqual(value, math.sin(original * index) - original, places=5)
        self.assertEqual(len(cache(compiled).graphs), 1)
        def reset_or_run(index):
            if index % 3 == 0:
                native.compiler.reset()
            return work(index)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(len(list(pool.map(reset_or_run, range(12)))), 12)
        native.compiler.reset()
        del compiled, x
        gc.collect()

    def test_two_product_contraction_overflow_and_cancellation(self):
        left = [2e38, -2e38, 1e38, -1e38, 16777216., -16777216., 0., -0.]
        right = [1e38, -1e38, 2e38, -2e38, 16777215., -16777215., -0., 0.]
        for expression in ('x * 2.0 - y * 2.0', 'x * 2.0 + y * -2.0',
                           'y * 2.0 - x * 2.0', 'y * -2.0 + x * 2.0',
                           'y * -3.713 + x * 1.137', 'x * 1.137 + y * -3.713',
                           'x * 1.0000001192092896 - y * 1.0000001192092896',
                           'x * y - y * y'):
            source = 'def f(x, y):\n return ' + expression
            fn, ref_fn = program(source), program(source, self.torch)
            compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
            for values in ((left, right), (right, left)):
                args = [self.upload(v, (len(v),)) for v in values]
                refs = [self.upload(v, (len(v),), self.torch) for v in values]
                with self.subTest(expression=expression, values=values):
                    self.compare(self.without_replay(fn, compiled, args), reference(*refs), exact=True)
            self.assertIn('fmaf(', kernel(compiled).plan(1))
            self.assertIn('fma.rn.f32', kernel(compiled).ptx)

    def test_negated_product_underflow_and_exact_zero_signs(self):
        left = [1e-38, -1e-38, 1e-38, -1e-38, 0., -0., 0., -0.]
        right = [1e-38, 1e-38, -1e-38, -1e-38, 1., 1., -1., -1.]
        args = [self.upload(v, (8,)) for v in (left, right)]
        refs = [self.upload(v, (8,), self.torch) for v in (left, right)]
        for body in ('return -(x * y)', 'a = x * y\n return -a',
                     'a = x * y\n return -a + a', 'return -(x * 1e-38)'):
            source = 'def f(x, y):\n ' + body
            fn, ref_fn = program(source), program(source, self.torch)
            compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
            with self.subTest(body=body):
                expected = reference(*refs)
                self.compare(self.without_replay(fn, compiled, args), expected, exact=True)
                self.assertIn('fmaf(', kernel(compiled).plan(1))
                if body == 'return -(x * y)':
                    self.assertEqual(expected.signbit().tolist(),
                                     [True, False, False, True, False, False, False, False])
                    self.assertTrue(self.torch.equal(expected.signbit()[:4], ref_fn(*refs).signbit()[:4]))

    def test_public_compile_rejects_custom_globals_before_codegen(self):
        x = self.upload([1., 2.], (2,))
        for expression in ('x.sum() + scale', 'x * scale', '-x'):
            fn, effects = custom_globals_program('def f(x):\n return ' + expression)
            compiled = native.compile(fn)
            for _ in range(2):
                with patch.object(bridge, '_pointwise_compile', side_effect=AssertionError('codegen')):
                    with self.assertRaisesRegex(NotImplementedError, 'globals must be an exact dict'):
                        compiled(x)
                self.assertEqual(effects, [])
                self.assertFalse(cache(compiled).graphs)
        fn, effects = custom_closure_program()
        compiled = native.compile(fn)
        for _ in range(2):
            with self.assertRaisesRegex(NotImplementedError, 'closure must be an exact tuple'):
                compiled(x)
            self.assertEqual(effects, [])
            self.assertFalse(cache(compiled).graphs)

    def test_closure_and_function_tensor_lifetimes(self):
        fw, scale = native, 0.7
        def fn(x):
            return fw.cos(x) + scale
        compiled = native.compile(fn)
        x = self.upload([1., 2.], (2,))
        result = compiled(x)
        fn_ref = weakref.ref(fn)
        del fn, x
        gc.collect()
        self.assertIsNotNone(fn_ref())
        self.compare(result, self.torch.tensor([1., 2.], device='cuda:0').cos() + scale)
        scale = -1.5
        self.compare(compiled(self.upload([3., 4.], (2,))), self.torch.tensor([3., 4.], device='cuda:0').cos() + scale)
        del compiled
        gc.collect()
        self.assertIsNone(fn_ref())

    def test_isolated_default_jit_without_pytorch_imports(self):
        script = '''
import sys, math
class Block:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('installed PyTorch production execution')
sys.meta_path.insert(0, Block())
import torch_rs as m
scale = 0.375
def f(x, y):
    a = m.sin(x)
    return a * a - y.cos() + scale
compiled = m.compile(f)
def forbid(frame, event, arg):
    if event == 'call' and frame.f_code is f.__code__:
        raise AssertionError('body replay')
sys.setprofile(forbid)
for value, scale in ((0.125, 0.375), (-1.875, -1.25), (0.75, 0.375)):
    x = m.tensor([value]).to('cuda:0')
    output = compiled(x, x)
    expected = math.sin(value)**2 - math.cos(value) + scale
    assert abs(output.cpu().tolist()[0] - expected) < 1e-6
    assert output.data_ptr() != x.data_ptr()
sys.setprofile(None)
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', script],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(two_device_reservation(), 'requires explicit two-device reservation')
    def test_two_device_restoration_and_module_ownership(self):
        torch = self.torch
        if torch.cuda.device_count() < 2:
            self.skipTest('requires two CUDA devices')
        fn = program('def f(x):\n return (x * 0.7).sin()')
        compiled = native.compile(fn)
        inputs = []
        for target in (0, 1):
            x = native.tensor([1., 2.]).to(f'cuda:{target}')
            inputs.append(x)
            with torch.cuda.device(1 - target):
                self.compare(compiled(x), torch.tensor([1., 2.], device=f'cuda:{target}').mul(0.7).sin())
                self.assertEqual(torch.cuda.current_device(), 1 - target)
        self.assertEqual({entry.device for entry in kernels(compiled)}, {0, 1})
        with torch.cuda.device(1):
            with self.assertRaisesRegex(RuntimeError, 'device guard'):
                kernel(compiled).run((inputs[1],))
            native.compiler.reset()
            self.assertEqual(torch.cuda.current_device(), 1)
            with patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/torch-rs-test-nvrtc'):
                with self.assertRaisesRegex(RuntimeError, 'NVRTC'):
                    compiled(inputs[0])
            self.assertEqual(torch.cuda.current_device(), 1)


if __name__ == '__main__':
    unittest.main()
