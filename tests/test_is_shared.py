import copy
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
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        channels_last = torch.zeros((2, 3, 4, 5)).contiguous(
            memory_format=torch.channels_last
        )
        with torch.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_view = leaf.transpose(0, 1)

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertTrue(
            channels_last.is_contiguous(memory_format=torch.channels_last)
        )
        return leaf, tracked, (
            ("scalar", torch.tensor(-3.5)),
            ("empty", torch.zeros((2, 0, 3))),
            ("contiguous", source),
            ("channels last", channels_last),
            ("strided view", strided),
            ("offset strided view", offset),
            ("extreme empty view", extreme_empty),
            ("autograd leaf", leaf),
            ("autograd non-leaf", produced),
            ("autograd non-leaf view", tracked),
            ("detached autograd view", tracked.detach()),
            ("no-grad output", no_grad_output),
            ("no-grad view", no_grad_view),
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

    def test_supported_cpu_storage_is_never_os_shared_and_is_unchanged(self):
        leaf, tracked, cases = self.tensor_cases()
        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = self.metadata(tensor)
                first = tensor.is_shared()
                second = tensor.is_shared()

                self.assertIs(type(first), bool)
                self.assertIs(first, False)
                self.assertIs(second, False)
                self.assertEqual(self.metadata(tensor), metadata)

        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        gradient = leaf.grad
        gradient_metadata = self.metadata(gradient)
        self.assertIs(gradient.is_shared(), False)
        self.assertIs(leaf.grad, gradient)
        self.assertEqual(self.metadata(gradient), gradient_metadata)

    def test_python_function_ownership_signature_documentation_and_pickle(self):
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
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(bound.__annotations__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertFalse(hasattr(bound, "__text_signature__"))
        self.assertEqual(str(inspect.signature(function)), "(self)")
        self.assertEqual(str(inspect.signature(bound)), "()")
        self.assertIn("is_shared", torch.Tensor.__dict__)
        self.assertTrue(
            all("is_shared" not in owner.__dict__ for owner in torch.Tensor.__mro__[1:])
        )
        self.assertIs(torch._tensor.Tensor, torch.Tensor)
        self.assertIs(torch._tensor.Tensor.is_shared, function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertIs(pickle.loads(pickle.dumps(function)), function)
        self.assertIs(function(self=tensor), False)
        self.assertIs(bound(), False)

    def test_python_binding_and_invalid_receiver_errors_match(self):
        tensor = torch.tensor([1.0])
        function = inspect.getattr_static(torch.Tensor, "is_shared")
        bound = tensor.is_shared
        cases = (
            (
                lambda: function(),
                TypeError,
                "Tensor.is_shared() missing 1 required positional argument: 'self'",
            ),
            (
                lambda: function(tensor, tensor),
                TypeError,
                "Tensor.is_shared() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: bound(tensor),
                TypeError,
                "Tensor.is_shared() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(tensor, unexpected=True),
                TypeError,
                "Tensor.is_shared() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: bound(unexpected=True),
                TypeError,
                "Tensor.is_shared() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: bound(self=tensor),
                TypeError,
                "Tensor.is_shared() got multiple values for argument 'self'",
            ),
            (
                lambda: function(1),
                AttributeError,
                "'int' object has no attribute '_typed_storage'",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_spoofed_tensor_class_uses_the_storage_query_path(self):
        function = inspect.getattr_static(torch.Tensor, "is_shared")
        events = []

        class Storage:
            def _is_shared(self):
                events.append("_is_shared")
                return "storage-result"

        class SpoofedTensor:
            @property
            def __class__(self):
                events.append("__class__")
                return torch.Tensor

            def _typed_storage(self):
                events.append("_typed_storage")
                return Storage()

        value = SpoofedTensor()
        self.assertTrue(isinstance(value, torch.Tensor))
        events.clear()
        self.assertEqual(function(value), "storage-result")
        self.assertEqual(events, ["_typed_storage", "_is_shared"])

    def test_unary_override_and_torch_function_modes_receive_python_function(self):
        tensor = torch.tensor([1.0])
        function = inspect.getattr_static(torch.Tensor, "is_shared")
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(function(value), marker)
        override_function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(override_function, function)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (value,))
        self.assertEqual(kwargs, {})

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        recording = RecordingMode(marker)
        with recording:
            result = tensor.is_shared()
        self.assertIs(result, marker)
        self.assertEqual(len(recording.calls), 1)
        mode_function, dispatch_types, args, kwargs = recording.calls[0]
        self.assertIs(mode_function, function)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(args, (tensor,))
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

        declining = RecordingMode(NotImplemented)
        with self.assertRaises(TypeError) as raised:
            with declining:
                tensor.is_shared()
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^no implementation found for 'torch_rs\._tensor\.is_shared' on "
                r"types that implement __torch_function__: \[\] nor in mode "
                r"<.*RecordingMode object at 0x[0-9a-f]+>$"
            ),
        )
        self.assertEqual(len(declining.calls), 2)
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        rejected = RecordingMode(marker)
        with rejected:
            with self.assertRaises(TypeError):
                tensor.is_shared(unexpected=True)
        self.assertEqual(rejected.calls, [])

    def test_shared_memory_mutation_and_storage_apis_remain_unsupported(self):
        tensor = torch.tensor([1.0])
        for name in (
            "share_memory_",
            "storage",
            "storage_type",
            "untyped_storage",
            "_typed_storage",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.Tensor, name))
                self.assertFalse(hasattr(tensor, name))
        self.assertFalse(hasattr(torch, "Storage"))
        self.assertFalse(hasattr(torch, "UntypedStorage"))


if __name__ == "__main__":
    unittest.main()
