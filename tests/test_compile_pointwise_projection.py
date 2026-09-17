"""Invocation-local binding projection agrees with the independent resolver."""
import unittest
import sys
import weakref
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests.test_compile_pointwise_jit import available, cache, program
from tests.test_compile_pointwise_structured_inputs import _without_cyclic_collection


def _resolver_lifetime_call(projected, rejected, retain_traceback=False):
    # Construct real admitted descriptors directly: admission's helper lifetime
    # and mock host/compile closures cannot affect this resolver observation.
    fn = program('def f(data):\n return data["x"] * gain', gain=object() if rejected else .5)
    ir = frontend.analyze(fn, 1)
    source = ir.dependencies[0]
    leaf = frontend.Value(0)
    tree = frontend.InputTree('dict', (leaf,), ('x',))
    references = [weakref.ref(source), weakref.ref(leaf), weakref.ref(tree)]
    if projected:
        child = source.child('x')
        projection = ({source: ('dict',), child: ('tensor', 0)}, {source: tree, child: leaf})
        references.append(weakref.ref(child))
    else:
        projection = None
    try:
        result = frontend._resolve_bindings(fn, ir, (tree,), projection)
    except NotImplementedError as error:
        if not rejected:
            raise
        if retain_traceback:
            return references, error.__traceback__
    else:
        if rejected:
            raise AssertionError('invalid capture was admitted')
        assert result[1][source] is tree
    return references


@unittest.skipUnless(sys.implementation.name == 'cpython', 'requires CPython reference release')
class ResolverLifetime(unittest.TestCase):
    def test_fallback_success_releases_projected_descriptors_without_collection(self):
        self.check_release(False, False)

    def test_projected_success_releases_descriptors_without_collection(self):
        self.check_release(True, False)

    def test_fallback_invalid_capture_releases_parameter_projection_without_collection(self):
        self.check_release(False, True)

    def test_projected_invalid_capture_releases_descriptors_without_collection(self):
        self.check_release(True, True)

    def check_release(self, projected, rejected):
        with _without_cyclic_collection():
            references = _resolver_lifetime_call(projected, rejected)
            self.assertTrue(all(reference() is None for reference in references))

    def test_retained_traceback_is_an_owner_until_released(self):
        for projected in (False, True):
            with self.subTest(projected=projected), _without_cyclic_collection():
                references, traceback = _resolver_lifetime_call(projected, True, True)
                self.assertTrue(all(reference() is not None for reference in references))
                del traceback
                self.assertTrue(all(reference() is None for reference in references))


class Projection(unittest.TestCase):
    def test_ordered_projection_matches_legacy_and_retains_no_tensor(self):
        x, y = native.ones(1), native.ones(1)
        cases = [
            (x,), (x, y, False, -0.0), (x, False, [y]),
            ({'x': [x], 'other': (y, .5), 'empty': {}},),
            ({'a': x, 'b': [x], 'scalars': [True, -0.0, float('inf')]},),
            (x, [], {}, ()),
        ]
        for args in cases:
            with self.subTest(args=len(args)):
                names = ','.join(f'p{i}' for i in range(len(args)))
                fn = program(f'def f({names}):\n return p0')
                ir = frontend.analyze(fn, len(args))
                roots = ir.dependencies[:len(args)]
                expected_tensors, expected_params = frontend.bind_arguments(args)
                projection = (roots, {}, {})
                tensors, params = frontend.bind_arguments(args, projection)
                self.assertEqual(params, expected_params)
                self.assertEqual([id(t) for t in tensors], [id(t) for t in expected_tensors])
                legacy = frontend.resolve(fn, ir, params)
                combined = frontend._resolve_bindings(fn, ir, params, projection[1:])
                for old, new in zip(legacy, combined):
                    self.assertEqual(list(old.items()), list(new.items()))
                self.assertFalse(any(type(v) is native.Tensor for v in combined[1].values()))

    def test_invalid_trees_have_identical_failures_with_projection(self):
        x = native.ones(1)
        cycle = []
        cycle.append(cycle)
        repeated = []
        deep = x
        for _ in range(65):
            deep = [deep]
        cases = [(x, {'unused': None}), (x, {1: False}), (cycle,),
                 (x, [repeated, repeated]), (x, x, [x]), (deep,),
                 ([x] + [False] * 4095,), (x, object()), (x, 1)]
        for args in cases:
            roots = tuple(frontend.BindingSource('parameter', f'p{i}', i)
                          for i in range(len(args)))
            errors = []
            for projection in (None, (roots, {}, {})):
                with self.assertRaises(NotImplementedError) as caught:
                    frontend.bind_arguments(args, projection)
                errors.append(str(caught.exception))
            self.assertEqual(*errors)

    def test_scalar_canonicalization_and_capture_validation_remain_shared(self):
        x = native.ones(1)
        fn = program('def f(p):\n return p["x"] * gain', gain=.5)
        ir = frontend.analyze(fn, 1)
        for scalar in (float('nan'), float('-inf'), -0.0, True):
            projection = (ir.dependencies[:1], {}, {})
            _, params = frontend.bind_arguments(({'x': x, 'unused': scalar},), projection)
            old, new = frontend.resolve(fn, ir, params), frontend._resolve_bindings(
                fn, ir, params, projection[1:])
            self.assertEqual(list(old[0].items()), list(new[0].items()))
        fn.__globals__['gain'] = object()
        with self.assertRaises(NotImplementedError):
            frontend._resolve_bindings(fn, ir, params, projection[1:])


@unittest.skipUnless(available(), 'requires native CUDA and NVRTC')
class ProjectionCUDA(unittest.TestCase):
    def tearDown(self):
        native.compiler.reset()

    def test_warm_projection_and_code_replacement_during_admission(self):
        fn = program('def f(data):\n return -data["x"]')
        replacement = program('def f(other):\n return other["x"] * .5')
        compiled = native.compile(fn)
        x = native.tensor([2., 4.]).to('cuda:0')
        self.assertEqual(compiled({'x': x}).cpu().tolist(), [-2., -4.])
        resolver = frontend._resolve_bindings
        with patch.object(frontend, '_resolve_bindings', wraps=resolver) as resolve:
            self.assertEqual(compiled({'x': x}).cpu().tolist(), [-2., -4.])
            self.assertIsNotNone(resolve.call_args.args[3])
        admit = bridge._pointwise_admit_inputs

        def replace(inputs):
            result = admit(inputs)
            fn.__code__ = replacement.__code__
            return result

        with patch.object(bridge, '_pointwise_admit_inputs', side_effect=replace):
            with patch.object(frontend, '_resolve_bindings', wraps=resolver) as resolve:
                self.assertEqual(compiled({'x': x}).cpu().tolist(), [1., 2.])
                self.assertIsNone(resolve.call_args.args[3])

    def test_warm_mode_and_method_guards_precede_native_admission(self):
        from torch_rs.overrides import TorchFunctionMode

        class Mode(TorchFunctionMode):
            def __torch_function__(self, *args, **kwargs):
                raise AssertionError("mode callback executed")

        fn = program('def f(data):\n return -data["x"]')
        compiled = native.compile(fn)
        x = native.ones(2).to('cuda:0')
        compiled({'x': x})
        with patch.object(bridge, '_pointwise_admit_inputs',
                          side_effect=AssertionError('early native admission')):
            with Mode(), self.assertRaisesRegex(NotImplementedError, 'active.*mode'):
                compiled({'x': x})
            with patch.object(native.Tensor, 'sin', lambda self: self):
                with self.assertRaisesRegex(NotImplementedError, 'patched Tensor operation'):
                    compiled({'x': x})

    def test_reentrant_replacement_reset_and_unused_leaf_admission(self):
        fn = program('def f(data):\n return -data["x"]')
        replacement = program('def f(other):\n return other["x"] * .5')
        compiled = native.compile(fn)
        x = native.tensor([2., 4.]).to('cuda:0')
        compiled({'x': x})
        admit = bridge._pointwise_admit_inputs
        entered = False

        def reenter(inputs):
            nonlocal entered
            result = admit(inputs)
            if not entered:
                entered = True
                fn.__code__ = replacement.__code__
                native.compiler.reset()
                self.assertEqual(compiled({'x': x}).cpu().tolist(), [1., 2.])
            return result

        with patch.object(bridge, '_pointwise_admit_inputs', side_effect=reenter):
            self.assertEqual(compiled({'x': x}).cpu().tolist(), [1., 2.])
        before = list(cache(compiled).graphs)
        with patch.object(bridge, '_pointwise_admit_inputs',
                          side_effect=AssertionError('tree admission must precede native')):
            with self.assertRaises(NotImplementedError):
                compiled({'x': x, 'unused': None})
        self.assertEqual(list(cache(compiled).graphs), before)
        cpu = native.ones(2)
        for data in ({'x': x, 'unused': cpu}, {'unused': cpu, 'x': x}):
            with self.assertRaisesRegex(NotImplementedError, 'does not compile CPU'):
                compiled(data)


if __name__ == '__main__':
    unittest.main()
