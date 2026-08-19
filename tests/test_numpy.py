import gc
import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = """
numpy(*, force=False) -> numpy.ndarray

Returns the tensor as a NumPy :class:`ndarray`.

If :attr:`force` is ``False`` (the default), the conversion
is performed only if the tensor is on the CPU, does not require grad,
does not have its conjugate bit set, and is a dtype and layout that
NumPy supports. The returned ndarray and the tensor will share their
storage, so changes to the tensor will be reflected in the ndarray
and vice versa.

If :attr:`force` is ``True`` this is equivalent to
calling ``t.detach().cpu().resolve_conj().resolve_neg().numpy()``.
If the tensor isn't on the CPU or the conjugate or negative bit is set,
the tensor won't share its storage with the returned ndarray.
Setting :attr:`force` to ``True`` can be a useful shorthand.

Args:
    force (bool): if ``True``, the ndarray may be a copy of the tensor
               instead of always sharing memory, defaults to ``False``.
"""

class TensorNumpyTests(unittest.TestCase):
    def assert_storage_export(self, tensor, *, case):
        expected = np.asarray(tensor.tolist(), dtype=np.float32).reshape(tensor.shape)
        actual = tensor.numpy(force=True)

        with self.subTest(case=case, observation="metadata"):
            self.assertIs(type(actual), np.ndarray)
            self.assertEqual(actual.dtype, np.dtype(np.float32))
            self.assertEqual(actual.shape, tuple(tensor.shape))
            expected_strides = (
                (0,) * tensor.ndim
                if tensor.numel() == 0
                else tuple(stride * tensor.element_size() for stride in tensor.stride())
            )
            self.assertEqual(actual.strides, expected_strides)
            np.testing.assert_array_equal(actual, expected)
            if actual.size:
                self.assertEqual(actual.__array_interface__["data"][0], tensor.data_ptr())

        second = tensor.numpy(force=True)
        with self.subTest(case=case, observation="aliasing"):
            self.assertIsNot(actual, second)
            if actual.size:
                self.assertTrue(np.shares_memory(actual, second))
                actual.flat[0] = np.float32(12345.0)
                np.testing.assert_array_equal(np.asarray(tensor), actual)
                self.assertEqual(second.flat[0], np.float32(12345.0))

    def test_force_true_shares_scalar_empty_offset_strided_and_channel_last(self):
        matrix_values = np.arange(12, dtype=np.float32).reshape(3, 4)
        matrix = torch.tensor(matrix_values.tolist())
        volume_values = np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2)
        volume = torch.tensor(volume_values.tolist())

        cases = (
            ("scalar", torch.tensor(-3.25)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)),
            ("offset", matrix[1]),
            ("offset-strided", matrix.transpose(0, 1)[2]),
            ("strided", matrix.transpose(0, 1)),
            (
                "channels-last",
                volume.contiguous(memory_format=torch.channels_last),
            ),
        )
        for case, tensor in cases:
            self.assert_storage_export(tensor, case=case)

    def test_force_true_detaches_requires_grad_without_mutating_the_graph(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            requires_grad=True,
        )
        exported = leaf.transpose(0, 1).numpy(force=True)

        np.testing.assert_array_equal(
            exported,
            np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        )
        exported[0, 0] = np.float32(99.0)
        self.assertEqual(leaf.tolist(), [[99.0, 2.0], [3.0, 4.0]])
        self.assertTrue(leaf.requires_grad)
        self.assertTrue(leaf.is_leaf)

        leaf.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])

    def test_default_and_force_false_share_storage_and_check_grad_mode(self):
        for case, export in (
            ("default", lambda tensor: tensor.numpy()),
            ("false", lambda tensor: tensor.numpy(force=False)),
        ):
            tensor = torch.tensor([1.0, 2.0])
            array = export(tensor)
            with self.subTest(case=case, requires_grad=False):
                array[0] = np.float32(7.0)
                self.assertEqual(tensor.tolist(), [7.0, 2.0])

            leaf = torch.tensor([1.0, 2.0], requires_grad=True)
            with self.subTest(case=case, requires_grad=True):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^Can't call numpy\(\) on Tensor that requires grad\. "
                    r"Use tensor\.detach\(\)\.numpy\(\) instead\.$",
                ):
                    export(leaf)

            with torch.no_grad():
                array = export(leaf)
            array[0] = np.float32(8.0)
            self.assertEqual(leaf.tolist(), [8.0, 2.0])

    def test_force_true_preserves_every_float32_bit(self):
        bits = np.array(
            [0x7FA00001, 0x7FC12345, 0x80000000, 0x00000001],
            dtype=np.uint32,
        )
        tensor = torch.tensor(memoryview(bits.view(np.float32)))
        exported = tensor.numpy(force=True)
        np.testing.assert_array_equal(exported.view(np.uint32), bits)

        exported.view(np.uint32)[0] = np.uint32(0x7FA12345)
        self.assertEqual(
            tensor.numpy(force=True).view(np.uint32)[0],
            np.uint32(0x7FA12345),
        )

    def test_export_keeps_detached_storage_alive(self):
        def export_view():
            tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1)
            return tensor.numpy(force=True)

        exported = export_view()
        gc.collect()
        np.testing.assert_array_equal(
            exported,
            np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        )
        exported[0, 0] = np.float32(9.0)
        self.assertEqual(exported[0, 0], np.float32(9.0))

    def test_array_protocol_remains_copying_and_rejects_copy_false(self):
        tensor = torch.tensor([1.0, 2.0])
        exported = np.asarray(tensor)
        exported[0] = np.float32(8.0)
        self.assertEqual(tensor.tolist(), [1.0, 2.0])

        with self.assertRaisesRegex(ValueError, "non-copying NumPy view"):
            np.array(tensor, copy=False)

    def test_descriptor_metadata_matches_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "numpy")
        bound = tensor.numpy

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'numpy' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "numpy")
        self.assertEqual(descriptor.__qualname__, "TensorBase.numpy")
        self.assertEqual(bound.__name__, "numpy")
        self.assertEqual(bound.__qualname__, "Tensor.numpy")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)
        with self.assertRaises(ValueError):
            inspect.signature(bound)

        np.testing.assert_array_equal(
            descriptor(tensor, force=True),
            np.array([1.0], dtype=np.float32),
        )

    def test_keyword_only_and_exact_bool_validation_match_pytorch(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "numpy")
        cases = (
            (
                lambda: tensor.numpy(True),
                "numpy() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: descriptor(tensor, True, False),
                "numpy() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: tensor.numpy(force=None),
                "numpy(): argument 'force' must be bool, not NoneType",
            ),
            (
                lambda: tensor.numpy(force=1),
                "numpy(): argument 'force' must be bool, not int",
            ),
            (
                lambda: tensor.numpy(force=np.bool_(True)),
                "numpy(): argument 'force' must be bool, not numpy.bool",
            ),
            (
                lambda: tensor.numpy(force="yes"),
                "numpy(): argument 'force' must be bool, not str",
            ),
            (
                lambda: tensor.numpy(extra=True),
                "numpy() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.numpy(extra=True, force=1),
                "numpy(): argument 'force' must be bool, not int",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_valid_calls_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "numpy")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        cases = (
            ("default", lambda: tensor.numpy(), None),
            ("false", lambda: tensor.numpy(force=False), {"force": False}),
            ("true", lambda: tensor.numpy(force=True), {"force": True}),
        )
        for case, call, expected_kwargs in cases:
            mode = RecordingMode(marker)
            with self.subTest(case=case), mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(len(args), 1)
            self.assertIs(args[0], tensor)
            self.assertEqual(kwargs, expected_kwargs)

        invalid = RecordingMode(marker)
        with invalid, self.assertRaises(TypeError):
            tensor.numpy(force=1)
        self.assertEqual(invalid.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.numpy(force=True)
        np.testing.assert_array_equal(forwarded, np.array([1.0], dtype=np.float32))
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(len(args), 1)
            self.assertIs(args[0], tensor)
            self.assertEqual(kwargs, {"force": True})

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.Tensor\.numpy'; all "
            r"__torch_function__ handlers returned NotImplemented:",
        ):
            with lower:
                with declining:
                    tensor.numpy(force=True)
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])


if __name__ == "__main__":
    unittest.main()
