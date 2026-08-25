import copy
import ctypes
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


ABS_DOC = """
abs() -> Tensor

See :func:`torch.abs`
"""

TOP_LEVEL_ABS_DOC = """
abs(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Computes the absolute value of each element in :attr:`input`.

.. math::
    \\text{out}_{i} = |\\text{input}_{i}|

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.abs(torch.tensor([-1, -2, 3]))
    tensor([ 1,  2,  3])
"""


class TensorAbsTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

    @staticmethod
    def raw_storage_bits(tensor):
        storage = (ctypes.c_uint32 * tensor.numel()).from_address(tensor.data_ptr())
        return tuple(storage)

    def assert_result(self, output, source, expected_stride, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(output.shape, source.shape)
            self.assertEqual(output.stride(), expected_stride)
            self.assertEqual(output.storage_offset(), 0)
            self.assertFalse(output.requires_grad)
            self.assertTrue(output.is_leaf)
            self.assertIs(output.dtype, torch.float32)
            self.assertEqual(output.device, torch.device("cpu"))
            self.assertFalse(output.is_set_to(source))
            if source.numel():
                self.assertNotEqual(output.data_ptr(), source.data_ptr())
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                self.tensor_bits(output),
                self.tensor_bits(source) & np.uint32(0x7FFF_FFFF),
            )

    @staticmethod
    def make_cases():
        base = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x3EAA_AAAB,
                0xBEAA_AAAB,
                0x3F80_0000,
                0xBF80_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        channels_last = torch.tensor(
            np.linspace(-15.0, 15.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist()
        ).contiguous(memory_format=torch.channels_last)
        channels_last_3d = torch.tensor(
            np.linspace(-90.0, 90.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist()
        ).contiguous(memory_format=torch.channels_last_3d)
        return (
            ("scalar", torch.tensor(-0.0), ()),
            (
                "empty offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (2, 1),
            ),
            ("empty singleton trailing", torch.zeros((0, 1)), (1, 1)),
            ("empty singleton middle", torch.zeros((0, 1, 2)), (2, 2, 1)),
            ("empty singleton surrounding", torch.zeros((1, 0, 1)), (1, 1, 1)),
            ("offset", strided[1], (1, 3)),
            ("noncontiguous", strided, (1, 4, 12)),
            ("channels last", channels_last, (60, 1, 15, 3)),
            ("channels last 3d", channels_last_3d, (360, 1, 90, 18, 3)),
            (
                "IEEE edges",
                torch.tensor(memoryview(special_bits.view(np.float32))),
                (1,),
            ),
        )

    @staticmethod
    def supported_calls(source):
        return (
            ("method", source.abs),
            ("positional", lambda: torch.abs(source)),
            ("input", lambda: torch.abs(input=source)),
            ("x", lambda: torch.abs(x=source)),
            ("a", lambda: torch.abs(a=source)),
            ("x1", lambda: torch.abs(x1=source)),
            ("out none", lambda: torch.abs(source, out=None)),
            ("alias and out none", lambda: torch.abs(x=source, out=None)),
        )

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        for case, source, expected_stride in self.make_cases():
            for form, call in self.supported_calls(source):
                output = call()
                self.assert_result(
                    output, source, expected_stride, case=(case, form)
                )
                if case == "IEEE edges":
                    source_bits = self.raw_storage_bits(source)
                    self.assertEqual(
                        self.raw_storage_bits(output),
                        tuple(bits & 0x7FFF_FFFF for bits in source_bits),
                    )

    def test_grad_recording_is_rejected_before_planning(self):
        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, -4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]
        for form, call in self.supported_calls(source):
            with self.subTest(form=form, mode="recording"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^abs\(\): autograd recording is not supported$",
                ):
                    call()
            with self.subTest(form=form, mode="no_grad"):
                with torch.no_grad():
                    output = call()
                self.assert_result(
                    output, source, (1,), case=(form, "no_grad")
                )

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        for form, call in self.supported_calls(extreme):
            with self.subTest(form=form, case="extreme recording"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^abs\(\): autograd recording is not supported$",
                ):
                    call()
            with self.subTest(form=form, case="extreme no_grad"):
                with torch.no_grad():
                    with self.assertRaisesRegex(
                        RuntimeError, "Stride calculation overflowed"
                    ):
                        call()

        detached = source.detach()
        for form, call in self.supported_calls(detached):
            self.assert_result(call(), detached, (1,), case=(form, "detached"))

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([-4.0])
        descriptor = inspect.getattr_static(torch.Tensor, "abs")
        bound = tensor.abs

        self.assertIs(torch.Tensor.abs, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'abs' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "abs")
        self.assertEqual(descriptor.__qualname__, "TensorBase.abs")
        self.assertEqual(bound.__name__, "abs")
        self.assertEqual(bound.__qualname__, "Tensor.abs")
        self.assertEqual(descriptor.__doc__, ABS_DOC)
        self.assertEqual(bound.__doc__, ABS_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)

        for callable_object, expected_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            if sys.version_info >= (3, 13):
                self.assertEqual(callable_object.__text_signature__, "($self, /)")
                self.assertEqual(
                    str(inspect.signature(callable_object)), expected_signature
                )
            else:
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)

        cases = (
            (lambda: tensor.abs(1), "TensorBase.abs() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.abs() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.abs() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.abs(1, 2),
                "TensorBase.abs() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.abs(input=tensor),
                (
                    "Tensor.abs() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.abs() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.abs() takes no keyword arguments"),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.abs() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.abs() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'abs' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.abs() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_descriptor_and_forward(self):
        tracked = torch.tensor([-4.0], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "abs")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tracked.abs()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tracked)
        self.assertIsNone(kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        plain = torch.tensor([-4.0])
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = plain.abs()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [4.0])

        order.clear()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^abs\(\): autograd recording is not supported$",
        ):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    tracked.abs()
        self.assertEqual(order, ["upper", "lower"])

        invalid_mode = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid_mode:
                plain.abs(1)
        self.assertEqual(invalid_mode.calls, [])

    def test_top_level_modes_and_subclass_overrides_observe_native_limits(self):
        tensor = torch.tensor([-4.0], requires_grad=True)
        destination = torch.tensor([17.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode()
        with mode:
            result = torch.abs(input=tensor, out=destination)
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.abs)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "out": destination})

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.abs(Override()), marker)
        self.assertIs(torch.abs(torch.tensor([-4.0]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.abs)
            self.assertEqual(dispatch_types, (Override,))

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("base")
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("derived")
                return marker

        self.assertIs(torch.abs(BaseOverride(), out=DerivedOverride()), marker)
        self.assertEqual(subclass_order, ["derived"])

        forwarding_order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        plain = tensor.detach()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.abs(input=plain, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [4.0])

        forwarding_order.clear()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^abs\(\): autograd recording is not supported$",
        ):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    torch.abs(input=tensor, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])

        events = []

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append("mode")
                return NotImplemented

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append("override")
                return marker

        with DecliningMode():
            self.assertIs(torch.abs(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

        invalid = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid:
                torch.abs(tensor, tensor)
        self.assertEqual(invalid.calls, [])

    def test_top_level_builtin_metadata_exports_copying_and_pickling(self):
        function = torch.abs
        self.assertIs(function, torch._C.abs)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "abs")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.abs")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, TOP_LEVEL_ABS_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method abs of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.abs, function)
        for action in (
            lambda: setattr(owner, "abs", None),
            lambda: delattr(owner, "abs"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.abs, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("abs"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["abs"], function)

    def test_top_level_binding_and_type_error_precedence(self):
        tensor = torch.tensor([-4.0])
        cases = (
            (
                lambda: torch.abs(),
                'abs() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.abs(tensor, tensor),
                "abs() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.abs(tensor, input=tensor),
                "abs() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.abs(out=tensor),
                'abs() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.abs(extra=tensor),
                'abs() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.abs(1, extra=True),
                "abs(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.abs(input=[]),
                "abs(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.abs(tensor, out=[]),
                "abs(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.abs(tensor, extra=True, out=[]),
                "abs(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.abs(tensor, extra=True),
                "abs() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.abs(input=tensor, a=tensor),
                "abs() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.abs(a=tensor, x=tensor, out=None),
                "abs() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.abs(x=tensor, a=tensor, out=None),
                "abs() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.abs(np.zeros((2, 3), dtype=np.float32)),
                "abs(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_concrete_out_is_rejected_without_mutation(self):
        source = torch.tensor([-4.0, -0.0, 3.0], requires_grad=True)
        destination = torch.tensor([17.0, 19.0, 23.0])
        pointer = destination.data_ptr()
        for form, call in (
            ("positional", lambda: torch.abs(source, out=destination)),
            ("keyword", lambda: torch.abs(input=source, out=destination)),
            ("alias", lambda: torch.abs(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^abs\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.data_ptr(), pointer)
                self.assertEqual(destination.tolist(), [17.0, 19.0, 23.0])
                self.assertIsNone(source.grad)

    def test_absolute_and_inplace_forms_remain_unsupported(self):
        tensor = torch.tensor([-4.0])
        self.assertTrue(hasattr(torch, "abs"))
        self.assertIn("abs", torch.__all__)
        for name in ("absolute", "abs_", "absolute_"):
            with self.subTest(owner="torch", name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
        for name in ("absolute", "abs_", "absolute_"):
            with self.subTest(owner="Tensor", name=name):
                self.assertFalse(hasattr(torch.Tensor, name))
                self.assertFalse(hasattr(tensor, name))
        with self.assertRaises(TypeError):
            tensor.abs(out=None)


if __name__ == "__main__":
    unittest.main()
