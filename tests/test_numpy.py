import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nnumpy(*, force=False) -> numpy.ndarray\n\n"
    "Returns the tensor as a NumPy :class:`ndarray`.\n\n"
    "If :attr:`force` is ``False`` (the default), the conversion\n"
    "is performed only if the tensor is on the CPU, does not require grad,\n"
    "does not have its conjugate bit set, and is a dtype and layout that\n"
    "NumPy supports. The returned ndarray and the tensor will share their\n"
    "storage, so changes to the tensor will be reflected in the ndarray\n"
    "and vice versa.\n\n"
    "If :attr:`force` is ``True`` this is equivalent to\n"
    "calling ``t.detach().cpu().resolve_conj().resolve_neg().numpy()``.\n"
    "If the tensor isn't on the CPU or the conjugate or negative bit is set,\n"
    "the tensor won't share its storage with the returned ndarray.\n"
    "Setting :attr:`force` to ``True`` can be a useful shorthand.\n\n"
    "Args:\n"
    "    force (bool): if ``True``, the ndarray may be a copy of the tensor\n"
    "               instead of always sharing memory, defaults to ``False``.\n"
)

UNSUPPORTED_MESSAGE = (
    "numpy(): force=False is not supported because zero-copy NumPy storage "
    "sharing is not implemented; pass force=True to request an independent copy"
)


class TensorNumpyTests(unittest.TestCase):
    def layout_cases(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = torch.tensor(values.tolist(), dtype=torch.float32)
        strided = source.transpose(0, 2)
        offset = strided[1]

        channel_values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        channels_last = torch.tensor(
            channel_values.tolist(), dtype=torch.float32
        ).contiguous(memory_format=torch.channels_last)

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertTrue(
            channels_last.is_contiguous(memory_format=torch.channels_last)
        )
        self.assertFalse(channels_last.is_contiguous())

        return (
            ("scalar", torch.tensor(-3.5), np.array(-3.5, dtype=np.float32)),
            (
                "empty",
                torch.zeros((2, 0, 3), dtype=torch.float32),
                np.zeros((2, 0, 3), dtype=np.float32),
            ),
            ("offset", offset, values.transpose(2, 1, 0)[1]),
            ("strided", strided, values.transpose(2, 1, 0)),
            ("channels last", channels_last, channel_values),
        )

    def test_force_true_copies_all_supported_layouts_to_float32_numpy(self):
        for case, tensor, expected in self.layout_cases():
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                tensor_values = tensor.tolist()

                first = tensor.numpy(force=True)
                second = tensor.numpy(force=True)

                self.assertIs(type(first), np.ndarray)
                self.assertEqual(first.dtype, np.dtype(np.float32))
                self.assertEqual(first.shape, expected.shape)
                np.testing.assert_array_equal(first, expected)
                np.testing.assert_array_equal(second, expected)
                self.assertIsNot(first, second)

                if first.size:
                    first.reshape(-1)[0] = np.float32(-1234.5)
                    np.testing.assert_array_equal(second, expected)
                    self.assertFalse(np.shares_memory(first, second))

                self.assertEqual(tensor.tolist(), tensor_values)
                np.testing.assert_array_equal(tensor.numpy(force=True), expected)

    def test_force_true_detaches_requires_grad_tensors_without_mutation(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)

        for case, tensor, expected in (
            ("leaf", leaf, [[1.0, 2.0], [3.0, 4.0]]),
            ("non-leaf view", tracked, [[2.0, 6.0], [4.0, 8.0]]),
        ):
            with self.subTest(case=case):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                    tensor.requires_grad,
                    tensor.is_leaf,
                )
                exported = tensor.numpy(force=True)
                np.testing.assert_array_equal(
                    exported, np.asarray(expected, dtype=np.float32)
                )
                exported[0, 0] = -99.0
                self.assertEqual(tensor.tolist(), expected)
                self.assertEqual(
                    (
                        tensor.shape,
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.data_ptr(),
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                    metadata,
                )

        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_default_and_force_false_are_explicitly_unsupported(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        for call in (
            tensor.numpy,
            lambda: tensor.numpy(force=False),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    NotImplementedError, f"^{re.escape(UNSUPPORTED_MESSAGE)}$"
                ):
                    call()

    def test_existing_array_protocol_remains_copy_only(self):
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)

        converted = np.asarray(tensor)
        self.assertEqual(converted.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(
            converted, np.asarray([1.0, 2.0], dtype=np.float32)
        )
        converted[0] = 9.0
        self.assertEqual(tensor.tolist(), [1.0, 2.0])

        with self.assertRaisesRegex(ValueError, "non-copying NumPy view"):
            np.array(tensor, copy=False)

    def test_tensorbase_descriptor_metadata_matches_pytorch_2_13(self):
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
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        np.testing.assert_array_equal(
            descriptor(tensor, force=True), np.asarray([1.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            bound(force=True), np.asarray([1.0], dtype=np.float32)
        )

    def test_keyword_only_binding_and_strict_bool_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "numpy")
        cases = (
            (
                lambda: tensor.numpy(True),
                "numpy() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: tensor.numpy(True, False),
                "numpy() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: descriptor(tensor, True),
                "numpy() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: tensor.numpy(force=1),
                "numpy(): argument 'force' must be bool, not int",
            ),
            (
                lambda: tensor.numpy(force=None),
                "numpy(): argument 'force' must be bool, not NoneType",
            ),
            (
                lambda: tensor.numpy(force=np.bool_(True)),
                "numpy(): argument 'force' must be bool, not numpy.bool",
            ),
            (
                lambda: tensor.numpy(force=1.0),
                "numpy(): argument 'force' must be bool, not float",
            ),
            (
                lambda: tensor.numpy(force="yes"),
                "numpy(): argument 'force' must be bool, not str",
            ),
            (
                lambda: tensor.numpy(force=object()),
                "numpy(): argument 'force' must be bool, not object",
            ),
            (
                lambda: tensor.numpy(unexpected=True),
                "numpy() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: tensor.numpy(force=True, unexpected=True),
                "numpy() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: tensor.numpy(**{"unexpected": True, "force": 1}),
                "numpy(): argument 'force' must be bool, not int",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.numpy() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'numpy' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.numpy() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_match_pytorch_2_13(self):
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

        for case, call, expected_kwargs in (
            ("default", tensor.numpy, None),
            ("false", lambda: tensor.numpy(force=False), {"force": False}),
            ("true", lambda: tensor.numpy(force=True), {"force": True}),
        ):
            with self.subTest(case=case):
                recording = RecordingMode(marker)
                with recording:
                    result = call()
                self.assertIs(result, marker)
                self.assertEqual(len(recording.calls), 1)
                function, dispatch_types, args, kwargs = recording.calls[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(len(args), 1)
                self.assertIs(args[0], tensor)
                self.assertEqual(kwargs, expected_kwargs)

        rejected = RecordingMode(marker)
        with rejected:
            with self.assertRaisesRegex(TypeError, "must be bool, not int"):
                tensor.numpy(force=1)
        self.assertEqual(rejected.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.numpy(force=True)
        self.assertEqual(order, ["upper", "lower"])
        np.testing.assert_array_equal(
            forwarded, np.asarray([1.0], dtype=np.float32)
        )

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaises(TypeError) as raised:
            with lower:
                with declining:
                    tensor.numpy(force=True)
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^Multiple dispatch failed for 'torch\.Tensor\.numpy'; all "
                r"__torch_function__ handlers returned NotImplemented:\n\n"
                r"  - mode object <.*RecordingMode object at 0x[0-9a-f]+>\n\n"
                r"For more information, try re-running with "
                r"TORCH_LOGS=not_implemented$"
            ),
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])


if __name__ == "__main__":
    unittest.main()
