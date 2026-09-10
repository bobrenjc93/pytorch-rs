"""Public tanhshrink differentials against the pinned PyTorch 2.13 API."""
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
    if type(tensor) is torch.Tensor:
        return np.asarray(tensor, dtype=np.float32)
    return tensor.detach().cpu().numpy()


def make_cases(module):
    base = module.tensor(
        np.linspace(-3, 3, 120, dtype=np.float32).reshape(2, 3, 4, 5).tolist(),
        dtype=module.float32,
    )
    # Include signed zeros, subnormals, saturation, infinities, and signed NaNs.
    bits = np.array([
        0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000,
        0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
        0x7fc12345, 0xffc54321, 0x7f812345, 0xff812345,
    ], dtype=np.uint32)
    return {
        "scalar": module.tensor(-0.0),
        "empty_view": module.zeros((2, 0, 3)).transpose(0, 2)[1],
        "contiguous": base,
        "offset": base[1],
        "noncontiguous": base.transpose(0, 3)[1],
        "transpose": base.transpose(0, 3),
        "channels_last": base.contiguous(memory_format=module.channels_last),
        "rank_five": base.reshape(1, 2, 3, 4, 5),
        "special_values": module.tensor(memoryview(bits.view(np.float32))),
    }


@unittest.skipIf(reference is None, "install the reference dependency group")
class FunctionalTanhshrinkReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("tanhshrink differentials require pinned PyTorch 2.13.0")

    def assert_tensor_matches(self, actual, expected, *, exact=False):
        self.assertIs(type(actual), torch.Tensor)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
        self.assertEqual(
            actual.is_contiguous(memory_format=torch.channels_last),
            expected.is_contiguous(memory_format=reference.channels_last),
        )
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), str(expected.device))
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        a, b = values(actual).reshape(-1), values(expected).reshape(-1)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        # Native tanh can differ from PyTorch by a float32 rounding step;
        # subtraction near zero amplifies the relative error. Keep that existing
        # boundary while checking zero signs and infinities exactly.
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-7, equal_nan=True)
        exact_mask = ~np.isnan(b) if exact else (b == 0) | np.isinf(b)
        np.testing.assert_array_equal(
            a[exact_mask].view(np.uint32), b[exact_mask].view(np.uint32)
        )

    def test_owned_nonleaf_composition_inherits_tanh_backward(self):
        for shape in ((), (2,), (1, 2), (1, 1, 2), (2, 0, 1)):
            with self.subTest(shape=shape):
                actual_leaf = torch.full(shape, 0.5, requires_grad=True)
                expected_leaf = reference.full(shape, 0.5, requires_grad=True)
                actual = torch.nn.functional.tanhshrink(actual_leaf + actual_leaf)
                expected = reference.nn.functional.tanhshrink(expected_leaf + expected_leaf)
                self.assert_tensor_matches(actual, expected)
                (actual * -2.5).sum().backward()
                (expected * -2.5).sum().backward()
                self.assert_tensor_matches(actual_leaf.grad, expected_leaf.grad)

    def test_signature_metadata_and_function_identity(self):
        actual = torch.nn.functional.tanhshrink
        expected = reference.nn.functional.tanhshrink
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        for attribute in ("__name__", "__qualname__", "__doc__", "__defaults__",
                          "__kwdefaults__", "__annotations__"):
            self.assertEqual(getattr(actual, attribute), getattr(expected, attribute))
        self.assertEqual(actual.__module__, "torch_rs.nn.functional")
        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(pickle.loads(pickle.dumps(actual)), actual)

    def test_inference_values_strides_fresh_storage_and_nonmutation(self):
        actual_cases, expected_cases = make_cases(torch), make_cases(reference)
        for name, source in actual_cases.items():
            with self.subTest(case=name):
                expected_source = expected_cases[name]
                before = values(source).copy().view(np.uint32)
                metadata = (source.shape, source.stride(), source.storage_offset(), source.data_ptr())
                first = torch.nn.functional.tanhshrink(input=source)
                second = torch.nn.functional.tanhshrink(source)
                expected = reference.nn.functional.tanhshrink(expected_source)
                self.assert_tensor_matches(first, expected, exact=name == "special_values")
                self.assert_tensor_matches(second, expected, exact=name == "special_values")
                composition = source - source.tanh()
                np.testing.assert_array_equal(
                    values(first).view(np.uint32), values(composition).view(np.uint32)
                )
                for result, other in ((first, source), (first, second), (expected, expected_source)):
                    self.assertIsNot(result, other)
                    self.assertFalse(result.is_set_to(other))
                    if result.numel():
                        self.assertNotEqual(result.data_ptr(), other.data_ptr())
                self.assertEqual(metadata, (source.shape, source.stride(), source.storage_offset(), source.data_ptr()))
                np.testing.assert_array_equal(values(source).view(np.uint32), before)

    def test_leaf_gradients_all_supported_ranks_and_accumulation(self):
        shapes = [(), (7,), (2, 7), (2, 3, 7), (2, 3, 4, 7),
                  (1, 1, 1, 1), (0,), (2, 0), (2, 0, 3), (2, 0, 3, 4)]
        for shape in shapes:
            with self.subTest(shape=shape):
                count = int(np.prod(shape))
                data = np.resize(np.array([-3, -1, -0.0, 0.0, 0.125, 0.5, 2], np.float32), count).reshape(shape)
                weights = np.linspace(-1, 2, count, dtype=np.float32).reshape(shape)
                actual_leaf = torch.tensor(data.tolist(), requires_grad=True)
                expected_leaf = reference.tensor(data.tolist(), dtype=reference.float32, requires_grad=True)
                # Nested empty lists lose trailing dimensions; factories retain them.
                if not count:
                    actual_leaf = torch.zeros(shape, requires_grad=True)
                    expected_leaf = reference.zeros(shape, requires_grad=True)
                for _ in range(2):
                    actual = torch.nn.functional.tanhshrink(actual_leaf)
                    expected = reference.nn.functional.tanhshrink(expected_leaf)
                    self.assert_tensor_matches(actual, expected)
                    actual_weight = torch.tensor(weights.tolist()) if count else torch.zeros(shape)
                    expected_weight = reference.tensor(weights.tolist()) if count else reference.zeros(shape)
                    (actual * actual_weight).sum().backward()
                    (expected * expected_weight).sum().backward()
                    self.assert_tensor_matches(actual_leaf.grad, expected_leaf.grad)
                np.testing.assert_array_equal(values(actual_leaf), data)

    def test_no_grad_and_detach_accept_broader_inference_inputs(self):
        def cases(module):
            leaf = module.tensor([[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True)
            return leaf, [leaf, leaf.transpose(0, 1)[1], leaf + leaf,
                          module.ones((1, 1, 1, 1, 2), requires_grad=True),
                          module.tensor([float("inf"), float("nan")], requires_grad=True)]

        actual_leaf, actual_cases = cases(torch)
        expected_leaf, expected_cases = cases(reference)
        for index, (source, expected_source) in enumerate(zip(actual_cases, expected_cases, strict=True)):
            with self.subTest(case=index):
                with torch.no_grad():
                    actual = torch.nn.functional.tanhshrink(source)
                with reference.no_grad():
                    expected = reference.nn.functional.tanhshrink(expected_source)
                self.assert_tensor_matches(actual, expected)
                self.assertFalse(actual.requires_grad)
                self.assert_tensor_matches(
                    torch.nn.functional.tanhshrink(source.detach()),
                    reference.nn.functional.tanhshrink(expected_source.detach()),
                )
        self.assertIsNone(actual_leaf.grad)
        self.assertIsNone(expected_leaf.grad)
        self.assertTrue(torch.is_grad_enabled())
        self.assertTrue(torch.nn.functional.tanhshrink(actual_leaf).requires_grad)

    def dispatch_observations(self, module):
        function = module.nn.functional.tanhshrink
        marker, calls = object(), []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func is function, tuple(t.__name__ for t in types), args, kwargs))
                return marker

        class Derived(Override):
            pass

        class Mode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((func is function, tuple(t.__name__ for t in types), args, kwargs))
                return marker

        observations = []
        for value in (Override(), Derived()):
            for call in (lambda: function(value), lambda: function(input=value)):
                calls.clear()
                self.assertIs(call(), marker)
                self.assertEqual(len(calls), 1)
                func, types, args, kwargs = calls[0]
                observations.append((func, types, len(args) == 1 and args[0] is value, kwargs))
        source = module.tensor([float("inf")], requires_grad=True)
        calls.clear()
        with Mode():
            self.assertIs(function(input=source), marker)
        self.assertEqual(len(calls), 1)
        func, types, args, kwargs = calls[0]
        observations.append((func, types, len(args) == 1 and args[0] is source, kwargs))

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func is function))
                return func(*args, **(kwargs or {}))

        source = module.tensor([0.5], requires_grad=True)
        with ForwardingMode("lower"), ForwardingMode("upper"):
            result = function(source)
        observations.append((order, result.requires_grad))
        return observations

    def test_overrides_modes_and_forwarding(self):
        self.assertEqual(self.dispatch_observations(torch), self.dispatch_observations(reference))

    def test_binding_and_invalid_receiver_errors(self):
        def error(module, call):
            try:
                call(module.nn.functional.tanhshrink, module.tensor([0.5]))
            except Exception as exc:
                return type(exc), str(exc)
            self.fail("expected argument error")

        calls = [lambda f, x: f(), lambda f, x: f(x, x),
                 lambda f, x: f(x, input=x), lambda f, x: f(x, out=None),
                 lambda f, x: f(x, inplace=False), lambda f, x: f(tensor=x),
                 lambda f, x: f(x, dtype=torch.float32),
                 lambda f, x: f(x, device="cpu")]
        for invalid in (None, 1, 0.5, [], (), "input"):
            calls.append(lambda f, x, invalid=invalid: f(invalid))
        for index, call in enumerate(calls):
            with self.subTest(case=index):
                self.assertEqual(error(torch, call), error(reference, call))

    def test_cuda_input_retains_native_rejection(self):
        if not reference.cuda.is_available():
            self.skipTest("requires a real NVIDIA GPU to test native CUDA rejection")
        source = torch.tensor([-1.0, 0.0, 1.0]).to("cuda:0")
        expected_source = reference.tensor([-1.0, 0.0, 1.0], device="cuda:0")
        expected = reference.nn.functional.tanhshrink(expected_source)
        reference.cuda.synchronize()
        self.assertTrue(reference.isfinite(expected).all().item())
        with self.assertRaises((RuntimeError, NotImplementedError)) as native_error:
            source.tanh()
        with self.assertRaises(type(native_error.exception)) as functional_error:
            torch.nn.functional.tanhshrink(source)
        self.assertEqual(str(functional_error.exception), str(native_error.exception))
        self.assertEqual(source.cpu().tolist(), [-1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
