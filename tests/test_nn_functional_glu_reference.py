"""Functional GLU differentials against pinned PyTorch 2.13."""
import copy
import inspect
import pickle
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference
except ImportError:
    reference = None


def values(tensor):
    return np.asarray(tensor) if type(tensor) is torch.Tensor else tensor.detach().cpu().numpy()


def layout_cases(module):
    base = module.tensor(np.linspace(-5, 5, 48, dtype=np.float32).reshape(2, 4, 6).tolist())
    cases = {
        'rank_one': base[0][0],
        'rank_two': base[0],
        'rank_three': base,
        'offset_rank_one': base[1][2],
        'offset_rank_two': base[1],
        'offset_rank_three': module.stack([base, base])[1],
        'transpose_rank_two': base[0].transpose(0, 1),
        'transpose_rank_three': base.transpose(0, 2),
        'offset_transpose': base.transpose(0, 2)[1],
    }
    for shape in [(0,), (2,), (2, 0), (0, 4), (2, 0, 6), (0, 4, 2),
                  (2, 4, 0), (1, 2, 1), (1, 1, 2)]:
        cases[f'shape_{shape}'] = module.zeros(shape)
    cases['empty_transpose'] = module.zeros((2, 0, 6)).transpose(0, 2)
    cases['empty_offset'] = module.zeros((2, 4, 0))[1]
    return cases


@unittest.skipIf(reference is None, 'install the reference dependency group')
class FunctionalGluReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference.__version__.split('+')[0] != '2.13.0':
            raise AssertionError('GLU differentials require pinned PyTorch 2.13.0')

    def assert_matches(self, actual, expected):
        self.assertIs(type(actual), torch.Tensor)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), str(expected.device))
        self.assertEqual(str(actual.layout), str(expected.layout))
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertEqual(actual.numel(), expected.numel())
        self.assertEqual(actual.dim(), expected.dim())
        a, b = values(actual).reshape(-1), values(expected).reshape(-1)
        # The existing sigmoid implementation may round differently by an ULP.
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=0, equal_nan=True)
        exact = (b == 0) | np.isinf(b)
        np.testing.assert_array_equal(a[exact].view(np.uint32), b[exact].view(np.uint32))

    def test_signature_imports_and_function_identity(self):
        actual, expected = torch.nn.functional.glu, reference.nn.functional.glu
        a, b = inspect.signature(actual), inspect.signature(expected)
        self.assertEqual(a.replace(return_annotation=None, parameters=[p.replace(annotation=None) for p in a.parameters.values()]),
                         b.replace(return_annotation=None, parameters=[p.replace(annotation=None) for p in b.parameters.values()]))
        self.assertEqual(actual.__annotations__, {'input': torch.Tensor, 'dim': int, 'return': torch.Tensor})
        self.assertEqual(expected.__annotations__, {'input': reference.Tensor, 'dim': int, 'return': reference.Tensor})
        for attribute in ('__name__', '__qualname__', '__defaults__', '__kwdefaults__'):
            self.assertEqual(getattr(actual, attribute), getattr(expected, attribute))
        self.assertEqual(actual.__module__, 'torch_rs.nn.functional')
        from torch_rs.nn.functional import glu
        self.assertIs(glu, actual)
        namespace = {}
        exec('from torch_rs.nn.functional import *', namespace)
        self.assertIs(namespace['glu'], actual)
        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(pickle.loads(pickle.dumps(actual)), actual)

    def test_layouts_dimensions_fresh_storage_and_input_preservation(self):
        actual_cases, expected_cases = layout_cases(torch), layout_cases(reference)
        for name, source in actual_cases.items():
            for dim in range(-source.dim(), source.dim()):
                if source.shape[dim] % 2:
                    continue
                with self.subTest(case=name, dim=dim):
                    before = values(source).copy().view(np.uint32)
                    metadata = (source.shape, source.stride(), source.storage_offset(), source.data_ptr())
                    first = torch.nn.functional.glu(source, dim)
                    second = torch.nn.functional.glu(input=source, dim=dim)
                    expected = reference.nn.functional.glu(expected_cases[name], dim)
                    self.assert_matches(first, expected)
                    self.assert_matches(second, expected)
                    for other in (source, second):
                        self.assertIsNot(first, other)
                        self.assertFalse(first.is_set_to(other))
                        if first.numel():
                            self.assertNotEqual(first.data_ptr(), other.data_ptr())
                    np.testing.assert_array_equal(values(source).view(np.uint32), before)
                    self.assertEqual(metadata, (source.shape, source.stride(), source.storage_offset(), source.data_ptr()))

    def test_numerical_edges(self):
        bits = np.array([0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000,
                         0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                         0x7fc12345, 0xffc54321, 0x7f812345, 0xff812345], dtype=np.uint32)
        special = np.concatenate([bits.view(np.float32), np.array([-100., -90., -1., 1., 90., 100.], dtype=np.float32)])
        # Cross the edges in both halves, including infinity times a zero gate.
        left, right = np.meshgrid(special, special)
        data = np.stack([left.reshape(-1), right.reshape(-1)])
        source = torch.tensor(memoryview(data.reshape(-1))).reshape(data.shape).transpose(0, 1)
        expected_source = reference.tensor(data).transpose(0, 1)
        before = values(source).copy().view(np.uint32)
        self.assert_matches(torch.nn.functional.glu(source), reference.nn.functional.glu(expected_source))
        np.testing.assert_array_equal(values(source).view(np.uint32), before)

    def test_no_grad_and_detach(self):
        for shape in [(6,), (2, 6), (2, 4, 6), (2, 0)]:
            for transpose in (False, True):
                with self.subTest(shape=shape, transpose=transpose):
                    sources = []
                    for module in (torch, reference):
                        leaf = module.ones(shape, requires_grad=True)
                        source = leaf.transpose(0, -1) if transpose else leaf
                        sources.append((leaf, source))
                    (leaf, source), (ref_leaf, ref_source) = sources
                    with torch.no_grad(), reference.no_grad():
                        self.assert_matches(torch.nn.functional.glu(source), reference.nn.functional.glu(ref_source))
                    self.assert_matches(torch.nn.functional.glu(source.detach()), reference.nn.functional.glu(ref_source.detach()))
                    self.assertTrue(torch.is_grad_enabled())
                    self.assertIsNone(leaf.grad)
                    self.assertIsNone(ref_leaf.grad)

    def assert_gradient_matches(self, actual, expected):
        self.assertIsNotNone(actual)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        np.testing.assert_allclose(values(actual), values(expected), rtol=3e-6, atol=2e-7)

    def test_generated_vector_backward_with_nonuniform_upstream(self):
        rng = np.random.default_rng(20260910)
        for length in [0, 2, 4, 6, 14, 32, 130]:
            data = rng.uniform(-6, 6, length).astype(np.float32).tolist()
            weights = rng.uniform(-2, 2, length // 2).astype(np.float32).tolist()
            for dim in (0, -1):
                with self.subTest(length=length, dim=dim):
                    actual = torch.tensor(data, requires_grad=True)
                    expected = reference.tensor(data, requires_grad=True)
                    before = values(actual).copy().view(np.uint32)
                    output = torch.nn.functional.glu(actual, dim)
                    ref_output = reference.nn.functional.glu(expected, dim)
                    self.assert_matches(output, ref_output)
                    self.assertFalse(output.is_set_to(actual))
                    (output * torch.tensor(weights)).sum().backward()
                    (ref_output * reference.tensor(weights)).sum().backward()
                    self.assert_gradient_matches(actual.grad, expected.grad)
                    np.testing.assert_array_equal(values(actual).view(np.uint32), before)

    def test_repeated_use_accumulation_and_graph_lifetime(self):
        actual = torch.tensor([-3., 2., -1., 0.5, -0.75, 1.25], requires_grad=True)
        expected = reference.tensor(actual.tolist(), requires_grad=True)
        for step in range(2):
            with self.subTest(step=step):
                losses = []
                for module, leaf in ((torch, actual), (reference, expected)):
                    output = module.nn.functional.glu(leaf)
                    # Both repeated uses of one result and separate GLU calls
                    # must accumulate both chunk halves into the same leaf.
                    losses.append((output * module.tensor([0.25, -2., 1.5])
                                   + output * output
                                   + module.nn.functional.glu(leaf, 0)).sum())
                losses[0].backward()
                losses[1].backward()
                self.assert_gradient_matches(actual.grad, expected.grad)
                before = values(actual.grad).copy()
                for loss in losses:
                    with self.assertRaisesRegex(RuntimeError, 'backward through the graph a second time'):
                        loss.backward()
                np.testing.assert_array_equal(values(actual.grad), before)

    def test_backward_through_tracked_views_and_nonleaves(self):
        data = np.linspace(-3, 3, 24, dtype=np.float32).tolist()
        transforms = {
            'identity_view': lambda x: x.view(24),
            'offset': lambda x: x.narrow(0, 4, 12),
            'selected_row': lambda x: x.reshape(4, 6)[2],
            'strided_offset': lambda x: x.reshape(6, 4).transpose(0, 1)[1],
            'chunk_output': lambda x: x.chunk(2)[1],
            'empty_offset': lambda x: x.narrow(0, 3, 0),
            'nonleaf': lambda x: x * 0.5 + x,
        }
        for name, transform in transforms.items():
            for dim in (0, -1):
                with self.subTest(case=name, dim=dim):
                    actual = torch.tensor(data, requires_grad=True)
                    expected = reference.tensor(data, requires_grad=True)
                    source, ref_source = transform(actual), transform(expected)
                    metadata = (source.shape, source.stride(), source.storage_offset(), source.data_ptr())
                    output = torch.nn.functional.glu(source, dim)
                    ref_output = reference.nn.functional.glu(ref_source, dim)
                    self.assert_matches(output, ref_output)
                    weights = np.linspace(-1.5, 2., output.numel(), dtype=np.float32).tolist()
                    (output * torch.tensor(weights)).sum().backward()
                    (ref_output * reference.tensor(weights)).sum().backward()
                    self.assert_gradient_matches(actual.grad, expected.grad)
                    self.assertEqual(metadata, (source.shape, source.stride(), source.storage_offset(), source.data_ptr()))
                    np.testing.assert_array_equal(values(actual), np.asarray(data, dtype=np.float32))

    def test_vector_no_grad_preserves_existing_graph_and_restores_recording(self):
        for length in (0, 6):
            with self.subTest(length=length):
                data = np.linspace(-2, 2, length * 2, dtype=np.float32).reshape(length, 2).tolist()
                actual = torch.tensor(data, requires_grad=True) if length else torch.zeros((0, 2), requires_grad=True)
                expected = reference.tensor(data, requires_grad=True) if length else reference.zeros((0, 2), requires_grad=True)
                source, ref_source = actual.transpose(0, 1)[1], expected.transpose(0, 1)[1]
                with torch.no_grad(), reference.no_grad():
                    self.assert_matches(torch.nn.functional.glu(source), reference.nn.functional.glu(ref_source))
                    self.assertFalse(torch.is_grad_enabled())
                self.assertTrue(torch.is_grad_enabled())
                self.assertIsNone(actual.grad)
                self.assertIsNone(expected.grad)
                output, ref_output = torch.nn.functional.glu(source), reference.nn.functional.glu(ref_source)
                self.assert_matches(output, ref_output)
                output.sum().backward()
                ref_output.sum().backward()
                self.assert_gradient_matches(actual.grad, expected.grad)

    def test_forwarding_mode_preserves_vector_backward(self):
        outputs, gradients, observations = [], [], []
        for module in (torch, reference):
            calls = []
            function = module.nn.functional.glu
            source = module.tensor([-2., 1., 0.25, -0.75], requires_grad=True)

            class Forward(module.overrides.TorchFunctionMode):
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    calls.append((func is function, kwargs))
                    return func(*args, **(kwargs or {}))

            mode = Forward()
            with mode:
                output = function(source, dim=0)
                self.assertIs(module.overrides._get_current_function_mode(), mode)
            self.assertIsNone(module.overrides._get_current_function_mode())
            (output * module.tensor([0.5, -2.])).sum().backward()
            outputs.append(output)
            gradients.append(source.grad)
            observations.append(calls)
        self.assertEqual(observations[0], observations[1])
        self.assert_matches(*outputs)
        self.assert_gradient_matches(*gradients)

    def test_binding_scalar_dimension_and_odd_size_errors(self):
        class Index:
            def __index__(self):
                return 1

        class Duck:
            def dim(self):
                return 1

        def error(module, call):
            try:
                call(module.nn.functional.glu, module)
            except Exception as exc:
                return type(exc), str(exc)
            self.fail('expected argument error')

        calls = [lambda f, m: f(), lambda f, m: f(m.ones(2), 0, 1),
                 lambda f, m: f(m.ones(2), input=m.ones(2)),
                 lambda f, m: f(m.ones(2), 0, dim=0),
                 lambda f, m: f(tensor=m.ones(2)),
                 lambda f, m: f(m.ones(2), out=None),
                 lambda f, m: f(m.ones(2), inplace=False)]
        for invalid in [None, 1, 0.5, [], (), 'input', Duck()]:
            calls.append(lambda f, m, invalid=invalid: f(invalid))
        for dim in [None, True, 1.0, 'last', [], Index(), np.bool_(True),
                    2, -3, 2**80, -(2**80)]:
            calls.append(lambda f, m, dim=dim: f(m.ones((2, 4)), dim))
            calls.append(lambda f, m, dim=dim: f(m.tensor(1.), dim))
        calls.append(lambda f, m: f(m.ones(2), m.tensor(0.)))
        for shape, dim in [((3,), -1), ((2, 3), 1), ((2, 3), -1), ((3, 2, 4), -3), ((0, 3), 1)]:
            calls.append(lambda f, m, shape=shape, dim=dim: f(m.ones(shape), dim))
        for index, call in enumerate(calls):
            with self.subTest(case=index):
                self.assertEqual(error(torch, call), error(reference, call))

        class Integer(int):
            def __int__(self):
                raise AssertionError('must not call __int__')

        for dim in [0, -1, np.int32(0), np.int64(-1), Integer(-1)]:
            self.assert_matches(torch.nn.functional.glu(torch.ones(4), dim),
                                reference.nn.functional.glu(reference.ones(4), dim))

    def dispatch_observations(self, module):
        function = module.nn.functional.glu
        marker, calls, observations = object(), [], []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func is function, tuple(t.__name__ for t in types), len(args), kwargs))
                return marker

        class Derived(Override):
            pass

        for source in [Override(), Derived()]:
            for call in [lambda: function(source), lambda: function(input=source, dim=0),
                         lambda: function(source, -2)]:
                self.assertIs(call(), marker)
        observations.extend(calls)
        calls.clear()

        # Dispatch precedes scalar/dimension validation and the native gradient
        # boundary. A mode may replace the operation without inspecting either.
        source = module.tensor(1., requires_grad=True)

        class Replace(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                observations.append((func is function,
                                     tuple(t.__name__ for t in types),
                                     len(args) == 1 and args[0] is source, kwargs))
                return marker

        with Replace():
            self.assertIs(function(source, dim=None), marker)

        class Raise(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise ValueError('override failure')

        raising_mode = Raise()
        with raising_mode:
            with self.assertRaisesRegex(ValueError, '^override failure$'):
                function(source)
            self.assertIs(module.overrides._get_current_function_mode(), raising_mode)
        self.assertIsNone(module.overrides._get_current_function_mode())

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((self.label, func is function, tuple(t.__name__ for t in types), kwargs))
                return func(*args, **(kwargs or {}))

        source, invalid = module.ones((2, 4)), module.ones(3)
        lower, upper = Mode('lower'), Mode('upper')
        with lower, upper:
            result = function(source)
            with self.assertRaises(RuntimeError):
                function(invalid)
            self.assertIs(module.overrides._get_current_function_mode(), upper)
            function(source, dim=0)
        self.assertIsNone(module.overrides._get_current_function_mode())
        observations.extend(calls)
        observations.append(values(result).tolist())
        return observations

    def test_overrides_nested_modes_and_restoration_after_errors(self):
        self.assertEqual(self.dispatch_observations(torch), self.dispatch_observations(reference))

    def test_cuda_boundary_on_real_hardware(self):
        if not reference.cuda.is_available():
            self.skipTest('requires NVIDIA GPU for native CUDA rejection')
        source = torch.tensor([1., 2., 3., 4.]).to('cuda:0')
        expected_source = reference.tensor([1., 2., 3., 4.], device='cuda:0')
        expected = reference.nn.functional.glu(expected_source)
        reference.cuda.synchronize()
        self.assertTrue(reference.isfinite(expected).all().item())
        with self.assertRaisesRegex(NotImplementedError, 'only exact native CPU float32'):
            torch.nn.functional.glu(source)
        self.assertEqual(source.cpu().tolist(), [1., 2., 3., 4.])


if __name__ == '__main__':
    unittest.main()
