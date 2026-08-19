import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorNumpyReferenceTests(unittest.TestCase):
    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_force_true_values_shapes_and_dtypes_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        matrix_values = np.arange(12, dtype=np.float32).reshape(3, 4)
        actual_matrix = torch.tensor(matrix_values.tolist())
        expected_matrix = reference_torch.tensor(
            matrix_values,
            dtype=reference_torch.float32,
        )
        volume_values = np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2)
        actual_volume = torch.tensor(volume_values.tolist()).contiguous(
            memory_format=torch.channels_last
        )
        expected_volume = reference_torch.tensor(
            volume_values,
            dtype=reference_torch.float32,
        ).contiguous(memory_format=reference_torch.channels_last)

        cases = (
            ("scalar", torch.tensor(-3.25), reference_torch.tensor(-3.25)),
            (
                "empty",
                torch.zeros((2, 0, 3)).transpose(0, 2),
                reference_torch.zeros((2, 0, 3)).transpose(0, 2),
            ),
            ("offset", actual_matrix[1], expected_matrix[1]),
            (
                "offset-strided",
                actual_matrix.transpose(0, 1)[2],
                expected_matrix.transpose(0, 1)[2],
            ),
            (
                "strided",
                actual_matrix.transpose(0, 1),
                expected_matrix.transpose(0, 1),
            ),
            ("channels-last", actual_volume, expected_volume),
        )
        for case, actual_tensor, expected_tensor in cases:
            with self.subTest(case=case):
                actual = actual_tensor.numpy(force=True)
                expected = expected_tensor.numpy(force=True)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.strides, expected.strides)
                np.testing.assert_array_equal(actual, expected)

                if actual.size:
                    actual.flat[0] = np.float32(777.0)
                    expected.flat[0] = np.float32(777.0)
                    np.testing.assert_array_equal(
                        actual_tensor.numpy(force=True),
                        expected_tensor.numpy(force=True),
                    )

    def test_requires_grad_force_true_aliasing_matches_pytorch(self):
        actual_tensor = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            requires_grad=True,
        ).transpose(0, 1)
        expected_tensor = reference_torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        ).transpose(0, 1)

        actual = actual_tensor.numpy(force=True)
        expected = expected_tensor.numpy(force=True)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)

        actual[0, 0] = np.float32(91.0)
        expected[0, 0] = np.float32(91.0)
        np.testing.assert_array_equal(
            actual_tensor.numpy(force=True),
            expected_tensor.numpy(force=True),
        )

    def test_force_true_float32_bits_match_pytorch_2_13(self):
        bits = np.array(
            [0x7FA00001, 0x7FC12345, 0x80000000, 0x00000001],
            dtype=np.uint32,
        )
        actual_tensor = torch.tensor(memoryview(bits.view(np.float32)))
        expected_tensor = reference_torch.from_numpy(bits.view(np.float32).copy())

        actual = actual_tensor.numpy(force=True)
        expected = expected_tensor.numpy(force=True)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

        replacement = np.uint32(0x7FA12345)
        actual.view(np.uint32)[0] = replacement
        expected.view(np.uint32)[0] = replacement
        np.testing.assert_array_equal(
            actual_tensor.numpy(force=True).view(np.uint32),
            expected_tensor.numpy(force=True).view(np.uint32),
        )

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "numpy")
        bound = tensor.numpy
        return {
            "descriptor_type": type(descriptor) is types.MethodDescriptorType,
            "bound_type": type(bound) is types.BuiltinMethodType,
            "descriptor_repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
        }

    def test_descriptor_metadata_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )
        for module in (torch, reference_torch):
            tensor = module.tensor([1.0], dtype=module.float32)
            descriptor = inspect.getattr_static(module.Tensor, "numpy")
            for callable_object in (descriptor, tensor.numpy):
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)

    def test_keyword_only_and_bool_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "numpy")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "numpy")
        calls = (
            (lambda: actual.numpy(True), lambda: expected.numpy(True)),
            (
                lambda: actual_descriptor(actual, True, False),
                lambda: expected_descriptor(expected, True, False),
            ),
            (
                lambda: actual.numpy(force=None),
                lambda: expected.numpy(force=None),
            ),
            (lambda: actual.numpy(force=1), lambda: expected.numpy(force=1)),
            (
                lambda: actual.numpy(force=np.bool_(True)),
                lambda: expected.numpy(force=np.bool_(True)),
            ),
            (
                lambda: actual.numpy(force="yes"),
                lambda: expected.numpy(force="yes"),
            ),
            (
                lambda: actual.numpy(extra=True),
                lambda: expected.numpy(extra=True),
            ),
            (
                lambda: actual.numpy(extra=True, force=1),
                lambda: expected.numpy(extra=True, force=1),
            ),
        )
        for actual_call, expected_call in calls:
            self.assert_error_matches(actual_call, expected_call)

    def mode_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "numpy")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        records = []
        for call, expected_kwargs in (
            (lambda: tensor.numpy(), None),
            (lambda: tensor.numpy(force=False), {"force": False}),
            (lambda: tensor.numpy(force=True), {"force": True}),
        ):
            mode = RecordingMode(marker)
            with mode:
                result = call()
            function, dispatch_types, args, kwargs = mode.calls[0]
            records.append(
                (
                    result is marker,
                    len(mode.calls),
                    function is descriptor,
                    function.__qualname__,
                    dispatch_types,
                    len(args),
                    args[0] is tensor,
                    kwargs,
                    expected_kwargs,
                )
            )

        invalid = RecordingMode(marker)
        try:
            with invalid:
                tensor.numpy(force=1)
        except Exception as error:
            invalid_error = (type(error).__name__, str(error))
        else:
            invalid_error = None

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(
                    (
                        self.label,
                        func is descriptor,
                        types,
                        len(args),
                        args[0] is tensor,
                        kwargs,
                    )
                )
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.numpy(force=True)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        try:
            with lower:
                with declining:
                    tensor.numpy(force=True)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            declining_error = None

        return {
            "records": records,
            "invalid_error": invalid_error,
            "invalid_calls": len(invalid.calls),
            "forwarding_order": order,
            "forwarded_values": forwarded.tolist(),
            "forwarded_dtype": str(forwarded.dtype),
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "lower_calls": len(lower.calls),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def test_default_force_false_and_grad_mode_match_pytorch_2_13(self):
        for case, actual_export, expected_export in (
            ("default", lambda tensor: tensor.numpy(), lambda tensor: tensor.numpy()),
            (
                "false",
                lambda tensor: tensor.numpy(force=False),
                lambda tensor: tensor.numpy(force=False),
            ),
        ):
            actual_tensor = torch.tensor([1.0, 2.0])
            expected_tensor = reference_torch.tensor([1.0, 2.0])
            actual = actual_export(actual_tensor)
            expected = expected_export(expected_tensor)
            with self.subTest(case=case, requires_grad=False):
                self.assertEqual(actual.strides, expected.strides)
                actual[0] = np.float32(7.0)
                expected[0] = np.float32(7.0)
                self.assertEqual(actual_tensor.tolist(), expected_tensor.tolist())

            actual_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
            expected_leaf = reference_torch.tensor([1.0, 2.0], requires_grad=True)
            with self.subTest(case=case, requires_grad=True):
                self.assert_error_matches(
                    lambda: actual_export(actual_leaf),
                    lambda: expected_export(expected_leaf),
                )

            with torch.no_grad(), reference_torch.no_grad():
                actual = actual_export(actual_leaf)
                expected = expected_export(expected_leaf)
            actual[0] = np.float32(8.0)
            expected[0] = np.float32(8.0)
            self.assertEqual(actual_leaf.tolist(), expected_leaf.tolist())


if __name__ == "__main__":
    unittest.main()
