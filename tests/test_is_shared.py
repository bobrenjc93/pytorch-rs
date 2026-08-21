import copy
import importlib
import inspect
import pickle
import re
import sys
import types
import unittest

import torch_rs as torch


METHOD_DOC = (
    "Checks if tensor is in shared memory.\n\n"
    "        This is always ``True`` for CUDA tensors.\n"
    "        "
)

if sys.version_info >= (3, 13):
    # CPython 3.13+ cleans function docstring indentation while preserving
    # the terminating newline; PyTorch's source docstring follows that rule.
    METHOD_DOC = inspect.cleandoc(METHOD_DOC) + "\n"


class TensorIsSharedTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        produced = leaf * 2.0
        tracked = produced.transpose(0, 1)
        tracked.sum().backward()
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=torch.float32,
        )
        strided = source.transpose(0, 1)
        offset = strided[1]
        extreme_empty = (
            torch.zeros((0,), dtype=torch.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        channels_last = torch.zeros(
            (2, 3, 4, 5), dtype=torch.float32
        ).contiguous(memory_format=torch.channels_last)

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertTrue(
            channels_last.is_contiguous(memory_format=torch.channels_last)
        )
        self.assertIsNotNone(leaf.grad)
        return (
            ("scalar", torch.tensor(-3.5, dtype=torch.float32)),
            ("empty", torch.zeros((2, 0, 3), dtype=torch.float32)),
            ("contiguous", source),
            ("channels last", channels_last),
            ("strided view", strided),
            ("offset strided view", offset),
            ("extreme empty view", extreme_empty),
            ("autograd leaf", leaf),
            ("autograd non-leaf", produced),
            ("autograd non-leaf view", tracked),
            ("detached autograd view", tracked.detach()),
            ("accumulated gradient", leaf.grad),
        )

    def metadata(self, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
            tensor.is_leaf,
        )

    def test_every_supported_cpu_tensor_has_process_local_storage(self):
        for case, tensor in self.tensor_cases():
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = self.metadata(tensor)

                first = tensor.is_shared()
                second = tensor.is_shared()

                self.assertIs(type(first), bool)
                self.assertIs(first, False)
                self.assertIs(second, False)
                self.assertEqual(self.metadata(tensor), metadata)

    def test_python_method_ownership_signature_and_documentation(self):
        tensor_module = importlib.import_module("torch_rs._tensor")
        tensor = torch.tensor([1.0])
        function = inspect.getattr_static(torch.Tensor, "is_shared")
        bound = tensor.is_shared

        self.assertIs(type(function), types.FunctionType)
        self.assertIs(type(bound), types.MethodType)
        self.assertRegex(
            repr(function),
            r"^<function Tensor\.is_shared at 0x[0-9a-f]+>$",
        )
        self.assertEqual(function.__name__, "is_shared")
        self.assertEqual(function.__qualname__, "Tensor.is_shared")
        self.assertEqual(function.__module__, "torch_rs._tensor")
        self.assertEqual(bound.__name__, "is_shared")
        self.assertEqual(bound.__qualname__, "Tensor.is_shared")
        self.assertEqual(bound.__module__, "torch_rs._tensor")
        self.assertEqual(function.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertEqual(str(inspect.signature(function)), "(self)")
        self.assertEqual(str(inspect.signature(bound)), "()")
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertFalse(hasattr(bound, "__text_signature__"))
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(bound.__annotations__, {})
        self.assertEqual(function.__dict__, {})
        self.assertEqual(bound.__dict__, {})
        self.assertIs(torch.Tensor.__dict__["is_shared"], function)
        self.assertNotIn("is_shared", torch.Tensor.__base__.__dict__)
        self.assertIs(torch.Tensor.is_shared, function)
        self.assertIs(tensor_module.Tensor.is_shared, function)
        self.assertFalse(hasattr(tensor_module, "is_shared"))
        self.assertIs(function.__get__(None, torch.Tensor), function)
        self.assertIs(bound.__func__, function)
        self.assertIs(bound.__self__, tensor)
        self.assertIs(function(tensor), False)
        self.assertIs(bound(), False)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_python_argument_and_receiver_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        function = inspect.getattr_static(torch.Tensor, "is_shared")
        bound = tensor.is_shared
        cases = (
            (
                lambda: tensor.is_shared(1),
                TypeError,
                "Tensor.is_shared() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: bound(1, 2),
                TypeError,
                "Tensor.is_shared() takes 1 positional argument but 3 were given",
            ),
            (
                lambda: function(tensor, 1),
                TypeError,
                "Tensor.is_shared() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.is_shared(input=tensor),
                TypeError,
                "Tensor.is_shared() got an unexpected keyword argument 'input'",
            ),
            (
                lambda: bound(unexpected=True),
                TypeError,
                "Tensor.is_shared() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: function(tensor, unexpected=True),
                TypeError,
                "Tensor.is_shared() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: function(),
                TypeError,
                "Tensor.is_shared() missing 1 required positional argument: 'self'",
            ),
            (
                lambda: function(1),
                AttributeError,
                "'int' object has no attribute '_typed_storage'",
            ),
            (
                lambda: function(tensor, self=tensor),
                TypeError,
                "Tensor.is_shared() got multiple values for argument 'self'",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        self.assertIs(function(self=tensor), False)

    def test_torch_function_modes_receive_python_function_and_forward(self):
        tensor = torch.tensor([1.0])
        function = inspect.getattr_static(torch.Tensor, "is_shared")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        recording = RecordingMode(marker)
        with recording:
            intercepted = tensor.is_shared()
        self.assertIs(intercepted, marker)
        self.assertEqual(len(recording.calls), 1)
        dispatched_function, dispatch_types, args, kwargs = recording.calls[0]
        self.assertIs(dispatched_function, function)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertEqual(kwargs, {})

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.is_shared()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, False)
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        declining = RecordingMode(NotImplemented)
        with self.assertRaisesRegex(
            TypeError,
            "^"
            + re.escape(
                "no implementation found for 'torch_rs._tensor.is_shared' on "
                "types that implement __torch_function__: [] nor in mode "
            ),
        ):
            with declining:
                tensor.is_shared()
        self.assertEqual(len(declining.calls), 2)
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_operand_override_dispatches_before_storage_fallback(self):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        result = torch.Tensor.is_shared(value)
        self.assertIs(result, marker)
        self.assertEqual(len(Override.calls), 1)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, torch.Tensor.is_shared)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (value,))
        self.assertEqual(kwargs, {})

    def test_duck_typed_receivers_cannot_spoof_the_native_tensor_check(self):
        marker = object()
        events = []

        class Storage:
            def _is_shared(self):
                events.append("storage._is_shared")
                return marker

        class SpoofedTensor:
            @property
            def __class__(self):
                events.append("__class__")
                return torch.Tensor

            def _typed_storage(self):
                events.append("_typed_storage")
                return Storage()

        spoofed = SpoofedTensor()
        self.assertTrue(isinstance(spoofed, torch.Tensor))
        events.clear()
        self.assertIs(torch.Tensor.is_shared(spoofed), marker)
        self.assertEqual(events, ["_typed_storage", "storage._is_shared"])

        class ClassAccessError(Exception):
            pass

        class RaisingClass:
            @property
            def __class__(self):
                events.append("__class__")
                raise ClassAccessError("receiver class must not be read")

            def _typed_storage(self):
                events.append("_typed_storage")
                return Storage()

        raising = RaisingClass()
        with self.assertRaisesRegex(
            ClassAccessError, "^receiver class must not be read$"
        ):
            isinstance(raising, torch.Tensor)
        events.clear()
        self.assertIs(torch.Tensor.is_shared(raising), marker)
        self.assertEqual(events, ["_typed_storage", "storage._is_shared"])

    def test_shared_memory_mutation_and_storage_apis_remain_unsupported(self):
        tensor = torch.tensor([1.0])
        self.assertFalse(hasattr(torch, "is_shared"))
        for name in (
            "share_memory_",
            "storage",
            "_typed_storage",
            "untyped_storage",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.Tensor, name))
                self.assertFalse(hasattr(tensor, name))


if __name__ == "__main__":
    unittest.main()
