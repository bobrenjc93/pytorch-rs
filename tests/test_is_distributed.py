import importlib
import inspect
import sys
import types
import unittest

import torch_rs as torch

try:
    from .signature_utils import assert_no_argument_signature
except ImportError:
    from signature_utils import assert_no_argument_signature


class TensorIsDistributedTests(unittest.TestCase):
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
            torch.zeros((0,), dtype=torch.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        channels_last = torch.zeros(
            (2, 3, 4, 5), dtype=torch.float32
        ).contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_view = leaf.transpose(0, 1)

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertTrue(
            channels_last.is_contiguous(memory_format=torch.channels_last)
        )
        return leaf, tracked, (
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

    def test_every_supported_tensor_is_local_without_mutation(self):
        leaf, tracked, cases = self.tensor_cases()
        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = self.metadata(tensor)

                first = tensor.is_distributed()
                second = tensor.is_distributed()

                self.assertIs(type(first), bool)
                self.assertIs(first, False)
                self.assertIs(second, False)
                self.assertEqual(self.metadata(tensor), metadata)

        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        gradient_metadata = self.metadata(leaf.grad)
        self.assertIs(leaf.grad.is_distributed(), False)
        self.assertEqual(self.metadata(leaf.grad), gradient_metadata)
        self.assertIs(tracked.is_distributed(), False)

    def test_tensorbase_descriptor_has_pytorch_metadata_and_no_documentation(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_distributed")
        bound = tensor.is_distributed

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'is_distributed' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "is_distributed")
        self.assertEqual(
            descriptor.__qualname__, "TensorBase.is_distributed"
        )
        self.assertEqual(bound.__name__, "is_distributed")
        self.assertEqual(bound.__qualname__, "Tensor.is_distributed")
        self.assertIsNone(descriptor.__doc__)
        self.assertIsNone(bound.__doc__)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(descriptor.__get__(tensor, torch.Tensor)(), False)
        self.assertIs(descriptor(tensor), False)
        self.assertIs(bound(), False)

    def test_invalid_receivers_and_arguments_match_pytorch(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_distributed")
        bound = tensor.is_distributed
        cases = (
            (
                lambda: tensor.is_distributed(1),
                "TensorBase.is_distributed() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.is_distributed() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.is_distributed() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.is_distributed(1, 2),
                "TensorBase.is_distributed() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.is_distributed(input=tensor),
                (
                    "Tensor.is_distributed() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.is_distributed() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.is_distributed() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.is_distributed() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.is_distributed() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'is_distributed' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.is_distributed() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_descriptor_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_distributed")
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
            intercepted = tensor.is_distributed()
        self.assertIs(intercepted, marker)
        self.assertEqual(len(recording.calls), 1)
        function, dispatch_types, args, kwargs = recording.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertIsNone(kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.is_distributed()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, False)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        previous_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(80)
            with self.assertRaisesRegex(
                RecursionError, r"^maximum recursion depth exceeded"
            ):
                with lower:
                    with declining:
                        tensor.is_distributed()
        finally:
            sys.setrecursionlimit(previous_limit)
        self.assertGreater(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])
        self.assertIs(tensor.is_distributed(), False)

    def test_distributed_execution_surface_remains_unsupported(self):
        self.assertFalse(hasattr(torch, "is_distributed"))
        for name in (
            "DTensor",
            "DeviceMesh",
            "ProcessGroup",
            "all_reduce",
            "init_process_group",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.distributed, name))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.distributed.tensor")


if __name__ == "__main__":
    unittest.main()
