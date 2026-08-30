import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


PROPERTY_DOC = (
    "\nReturns a new tensor containing real values of the :attr:`self` tensor "
    "for a complex-valued input tensor.\n"
    "The returned tensor and :attr:`self` share the same underlying storage.\n\n"
    "Returns :attr:`self` if :attr:`self` is a real-valued tensor.\n\n"
    "Example::\n\n"
    "    >>> x=torch.randn(4, dtype=torch.cfloat)\n"
    "    >>> x\n"
    "    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), "
    "(-1.6492-0.0633j), (-0.0638-0.8119j)])\n"
    "    >>> x.real\n"
    "    tensor([ 0.3100, -0.5445, -1.6492, -0.0638])\n\n"
)
FUNCTION_DOC = (
    "\nreal(input) -> Tensor\n\n"
    "Returns a new tensor containing real values of the :attr:`self` tensor.\n"
    "The returned tensor and :attr:`self` share the same underlying storage.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n\n"
    "Example::\n\n"
    "    >>> x=torch.randn(4, dtype=torch.cfloat)\n"
    "    >>> x\n"
    "    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), "
    "(-1.6492-0.0633j), (-0.0638-0.8119j)])\n"
    "    >>> x.real\n"
    "    tensor([ 0.3100, -0.5445, -1.6492, -0.0638])\n\n"
)


class TensorRealTests(unittest.TestCase):
    def tensor_cases(self):
        scalar_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x007F_FFFF,
                0x0080_0000,
                0x3F80_0000,
                0x7F7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        scalar_storage = torch.tensor(memoryview(scalar_bits.view(np.float32)))
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = base.transpose(0, 2)
        offset = strided[1]
        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertEqual(empty.shape, (0, 2))
        self.assertGreater(empty.storage_offset(), 0)
        self.assertTrue(leaf.is_leaf)
        self.assertFalse(non_leaf.is_leaf)
        return (
            *(
                (f"float32 bits 0x{bits:08x}", scalar_storage[index])
                for index, bits in enumerate(scalar_bits)
            ),
            ("empty offset view", empty),
            ("offset strided view", offset),
            ("strided view", strided),
            ("autograd leaf", leaf),
            ("autograd non-leaf", non_leaf),
        )

    def test_supported_tensors_return_the_exact_receiver_without_side_effects(self):
        for case, tensor in self.tensor_cases():
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.dtype,
                    tensor.device,
                    tensor.requires_grad,
                    tensor.is_leaf,
                    tensor.data_ptr(),
                )
                detached = tensor.detach()
                bits = np.asarray(detached).reshape(-1).view(np.uint32).copy()

                result = tensor.real

                self.assertIs(result, tensor)
                self.assertTrue(result.is_set_to(detached))
                self.assertEqual(
                    (
                        result.shape,
                        result.stride(),
                        result.storage_offset(),
                        result.dtype,
                        result.device,
                        result.requires_grad,
                        result.is_leaf,
                        result.data_ptr(),
                    ),
                    metadata,
                )
                np.testing.assert_array_equal(
                    np.asarray(result.detach()).reshape(-1).view(np.uint32), bits
                )

    def test_leaf_and_non_leaf_graphs_are_not_changed(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        self.assertIs(leaf.real, leaf)
        self.assertTrue(leaf.is_leaf)

        non_leaf = (leaf.real * 3.0).transpose(0, 1)[1]
        graph_before = (
            non_leaf.requires_grad,
            non_leaf.is_leaf,
            non_leaf.shape,
            non_leaf.stride(),
            non_leaf.storage_offset(),
            non_leaf.data_ptr(),
        )

        result = non_leaf.real

        self.assertIs(result, non_leaf)
        self.assertEqual(
            (
                result.requires_grad,
                result.is_leaf,
                result.shape,
                result.stride(),
                result.storage_offset(),
                result.data_ptr(),
            ),
            graph_before,
        )
        result.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        gradient = leaf.grad
        self.assertIs(leaf.real, leaf)
        self.assertIs(leaf.grad, gradient)

    def test_tensorbase_descriptor_is_documented_and_read_only(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "real")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "real")
        self.assertEqual(descriptor.__qualname__, "TensorBase.real")
        self.assertEqual(descriptor.__doc__, PROPERTY_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'real' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.real, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(descriptor.__get__(tensor, torch.Tensor), tensor)

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'real' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

        actions = (
            lambda: setattr(tensor, "real", torch.tensor([2.0])),
            lambda: delattr(tensor, "real"),
            lambda: descriptor.__set__(tensor, torch.tensor([2.0])),
            lambda: descriptor.__delete__(tensor),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'real' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )

    def test_torch_function_modes_receive_descriptor_get_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "real")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.real
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
                forwarded = tensor.real
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)

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
                    tensor.real
        self.assertGreater(upper.calls, 1)
        self.assertEqual(lower.calls, 0)
        self.assertIs(tensor.real, tensor)

    def top_level_calls(self, tensor):
        return (
            ("positional", lambda: torch.real(tensor)),
            ("input", lambda: torch.real(input=tensor)),
            ("x", lambda: torch.real(x=tensor)),
            ("a", lambda: torch.real(a=tensor)),
            ("x1", lambda: torch.real(x1=tensor)),
        )

    def test_top_level_supported_tensors_return_the_exact_receiver(self):
        for case, tensor in self.tensor_cases():
            metadata = (
                tensor.shape,
                tensor.stride(),
                tensor.storage_offset(),
                tensor.dtype,
                tensor.device,
                tensor.layout,
                tensor.requires_grad,
                tensor.is_leaf,
                tensor.output_nr,
                tensor.data_ptr(),
            )
            detached = tensor.detach()
            bits = np.asarray(detached).reshape(-1).view(np.uint32).copy()

            for form, call in self.top_level_calls(tensor):
                with self.subTest(case=case, form=form):
                    result = call()

                    self.assertIs(result, tensor)
                    self.assertTrue(result.is_set_to(detached))
                    self.assertEqual(
                        (
                            result.shape,
                            result.stride(),
                            result.storage_offset(),
                            result.dtype,
                            result.device,
                            result.layout,
                            result.requires_grad,
                            result.is_leaf,
                            result.output_nr,
                            result.data_ptr(),
                        ),
                        metadata,
                    )
                    np.testing.assert_array_equal(
                        np.asarray(result.detach()).reshape(-1).view(np.uint32),
                        bits,
                    )

    def test_top_level_real_ignores_tensor_class_shadowing(self):
        tensor = torch.tensor([1.0])
        original = torch.Tensor.real
        marker = object()

        try:
            torch.Tensor.real = property(lambda _self: marker)

            self.assertIs(tensor.real, marker)
            self.assertIs(torch.real(tensor), tensor)
            self.assertIs(torch.real(input=tensor), tensor)
        finally:
            torch.Tensor.real = original

        self.assertIs(tensor.real, tensor)
        self.assertIs(torch.real(tensor), tensor)

    def test_top_level_leaf_and_non_leaf_graphs_are_not_changed(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        self.assertIs(torch.real(leaf), leaf)
        self.assertTrue(leaf.is_leaf)

        non_leaf = (torch.real(leaf) * 3.0).transpose(0, 1)[1]
        graph_before = (
            non_leaf.requires_grad,
            non_leaf.is_leaf,
            non_leaf.shape,
            non_leaf.stride(),
            non_leaf.storage_offset(),
            non_leaf.output_nr,
            non_leaf.data_ptr(),
        )

        result = torch.real(non_leaf)

        self.assertIs(result, non_leaf)
        self.assertEqual(
            (
                result.requires_grad,
                result.is_leaf,
                result.shape,
                result.stride(),
                result.storage_offset(),
                result.output_nr,
                result.data_ptr(),
            ),
            graph_before,
        )
        result.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        gradient = leaf.grad
        self.assertIs(torch.real(leaf), leaf)
        self.assertIs(leaf.grad, gradient)

    def test_top_level_callable_metadata_pickling_and_exports(self):
        function = torch.real
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "real")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.real")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method real of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.real, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("real"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["real"], function)

    def test_top_level_torch_function_modes_and_overrides(self):
        tensor = torch.tensor([1.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        calls = (
            (None, lambda: torch.real(tensor)),
            ("input", lambda: torch.real(input=tensor)),
            ("x", lambda: torch.real(x=tensor)),
            ("a", lambda: torch.real(a=tensor)),
            ("x1", lambda: torch.real(x1=tensor)),
        )
        for keyword, call in calls:
            mode = RecordingMode()
            with mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.real)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor,) if keyword is None else ())
            self.assertEqual(kwargs, None if keyword is None else {keyword: tensor})

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.real(a=tensor)
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(torch.real(x=value), marker)
        function, dispatch_types, args, kwargs = override_calls[0]
        self.assertIs(function, torch.real)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"x": value})

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        message = (
            "Multiple dispatch failed for 'torch.real'; all "
            "__torch_function__ handlers returned NotImplemented:\n\n"
            f"  - tensor subclass {DecliningOverride!r}\n\n"
            "For more information, try re-running with "
            "TORCH_LOGS=not_implemented"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            torch.real(DecliningOverride())
        self.assertIs(torch.real(tensor), tensor)

    def test_top_level_binding_errors_and_unsupported_scope(self):
        tensor = torch.tensor([1.0])
        destination = torch.tensor([17.0])
        cases = (
            (
                lambda: torch.real(),
                'real() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.real(tensor, tensor),
                "real() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.real(tensor, input=tensor),
                "real() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.real(tensor, out=None),
                "real() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.real(tensor, out=destination),
                "real() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.real(tensor, dtype=torch.float32),
                "real() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.real(tensor, device="cpu"),
                "real() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.real(1, extra=True),
                "real(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.real(input=[]),
                "real(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.real(a=1),
                "real(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.real(x=[]),
                "real(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.real(x1=None),
                "real(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.real(a=tensor, x=tensor),
                "real() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.real(x=tensor, a=tensor),
                "real() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.real(input=tensor, x1=tensor),
                "real() got an unexpected keyword argument 'x1'",
            ),
            (
                lambda: torch.real(np.zeros((2, 3), dtype=np.float32)),
                "real(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()
                self.assertEqual(destination.tolist(), [17.0])

        for name in (
            "complex32",
            "complex64",
            "complex128",
            "chalf",
            "cfloat",
            "cdouble",
            "real_",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
        self.assertTrue(hasattr(torch, "imag"))
        self.assertFalse(hasattr(torch.Tensor, "real_"))


if __name__ == "__main__":
    unittest.main()
