import importlib
import inspect
import math
import sys
import types
import unittest

import torch_rs as torch


PROPERTY_DOC = (
    "\nIs ``True`` if the Tensor is stored on the XPU, ``False`` otherwise.\n"
)


class TensorIsXpuTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = leaf * 2.0
        tracked_view = tracked.transpose(0, 1)
        detached = tracked.detach()
        detached_view = tracked_view.detach()
        tracked.sum().backward()

        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        strided_view = source.transpose(0, 1)
        offset_view = strided_view[1]
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )

        self.assertFalse(strided_view.is_contiguous())
        self.assertGreater(offset_view.storage_offset(), 0)
        return (
            *(
                (f"scalar {value!r}", torch.tensor(value))
                for value in (
                    -math.inf,
                    -1.0,
                    -0.0,
                    0.0,
                    1.0,
                    math.inf,
                    math.nan,
                )
            ),
            ("ordinary tensor", source),
            ("empty", torch.zeros((2, 0, 3))),
            ("strided view", strided_view),
            ("offset strided view", offset_view),
            ("extreme empty view", extreme_empty),
            ("detached non-leaf", detached),
            ("detached strided view", detached_view),
            ("autograd leaf", leaf),
            ("autograd non-leaf", tracked),
            ("autograd non-leaf view", tracked_view),
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
            tensor.layout,
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )

    def test_every_supported_cpu_tensor_reports_exact_false(self):
        for case, tensor in self.tensor_cases():
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = self.metadata(tensor)

                result = tensor.is_xpu

                self.assertIs(type(result), bool)
                self.assertIs(result, False)
                self.assertEqual(tensor.device, torch.device("cpu"))
                self.assertEqual(self.metadata(tensor), metadata)

    def test_tensorbase_descriptor_documentation_and_receiver_behavior(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_xpu")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "is_xpu")
        self.assertEqual(descriptor.__qualname__, "TensorBase.is_xpu")
        self.assertEqual(descriptor.__doc__, PROPERTY_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'is_xpu' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.is_xpu, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(descriptor.__get__(tensor, torch.Tensor), False)

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'is_xpu' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

    def test_property_is_read_only_with_pytorch_assignment_errors(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_xpu")
        actions = (
            lambda: setattr(tensor, "is_xpu", True),
            lambda: delattr(tensor, "is_xpu"),
            lambda: descriptor.__set__(tensor, True),
            lambda: descriptor.__delete__(tensor),
        )

        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'is_xpu' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )

    def test_torch_function_modes_receive_descriptor_get_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_xpu")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.is_xpu
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertEqual(function, descriptor.__get__)
        self.assertIs(function.__self__, descriptor)
        self.assertEqual(function.__name__, "__get__")
        self.assertEqual(function.__qualname__, "getset_descriptor.__get__")
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
                forwarded = tensor.is_xpu
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, False)

    def test_not_implemented_reenters_the_declining_top_mode(self):
        tensor = torch.tensor([1.0])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return NotImplemented

        class AcceptingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return object()

        lower = AcceptingMode()
        upper = DecliningMode()
        with self.assertRaisesRegex(
            RecursionError, r"^maximum recursion depth exceeded$"
        ):
            with lower:
                with upper:
                    tensor.is_xpu
        self.assertGreater(upper.calls, 1)
        self.assertEqual(lower.calls, 0)
        self.assertIs(tensor.is_xpu, False)

    def test_xpu_namespace_creation_and_transfer_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "xpu"))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.xpu")

        for specification in ("xpu", "xpu:0"):
            with self.subTest(specification=specification, action="device"):
                with self.assertRaisesRegex(
                    RuntimeError, r"only 'cpu' is implemented"
                ):
                    torch.device(specification)
            for function in (torch.tensor, torch.zeros):
                with self.subTest(
                    specification=specification,
                    action=function.__name__,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, r"only 'cpu' is implemented"
                    ):
                        if function is torch.tensor:
                            function([1.0], device=specification)
                        else:
                            function((1,), device=specification)

        tensor = torch.tensor([1.0])
        with self.assertRaisesRegex(
            NotImplementedError, r"device conversions are not supported"
        ):
            tensor.to("xpu")


if __name__ == "__main__":
    unittest.main()
