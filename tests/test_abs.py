import builtins
import copy
import ctypes
import inspect
import pickle
import sys
import types
import unittest
from multiprocessing.reduction import ForkingPickler

import numpy as np
import torch_rs as torch


ABS_DOC = """
abs() -> Tensor

See :func:`torch.abs`
"""

ABSOLUTE_DOC = """
absolute() -> Tensor

Alias for :func:`abs`
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
            ("abs", source.abs),
            ("absolute", source.absolute),
            ("operator", lambda: builtins.abs(source)),
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

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        for form, call in self.supported_calls(extreme):
            with self.subTest(form=form, mode="extreme recording"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^abs\(\): autograd recording is not supported$",
                ):
                    call()

        for form, call in self.supported_calls(source):
            with torch.no_grad():
                output = call()
            self.assert_result(output, source, (1,), case=(form, "no_grad"))

        for form, call in self.supported_calls(extreme):
            with self.subTest(form=form, mode="extreme no_grad"):
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
        descriptors = {
            "abs": inspect.getattr_static(torch.Tensor, "abs"),
            "absolute": inspect.getattr_static(torch.Tensor, "absolute"),
        }
        operator_descriptor = inspect.getattr_static(torch.Tensor, "__abs__")

        self.assertIs(operator_descriptor, descriptors["abs"])
        self.assertIs(torch.Tensor.__dict__["__abs__"], descriptors["abs"])
        self.assertIsNot(descriptors["absolute"], descriptors["abs"])
        with self.assertRaises(AttributeError):
            inspect.getattr_static(descriptors["abs"].__objclass__, "__abs__")

        for name, doc in (("abs", ABS_DOC), ("absolute", ABSOLUTE_DOC)):
            descriptor = descriptors[name]
            bound = getattr(tensor, name)
            if name == "abs":
                direct_one_argument = lambda: tensor.abs(1)
                direct_two_arguments = lambda: tensor.abs(1, 2)
                direct_keyword = lambda: tensor.abs(input=tensor)
            else:
                direct_one_argument = lambda: tensor.absolute(1)
                direct_two_arguments = lambda: tensor.absolute(1, 2)
                direct_keyword = lambda: tensor.absolute(input=tensor)
            with self.subTest(name=name, contract=True):
                self.assertIs(getattr(torch.Tensor, name), descriptor)
                self.assertIs(type(descriptor), types.MethodDescriptorType)
                self.assertIs(type(bound), types.BuiltinMethodType)
                self.assertEqual(
                    repr(descriptor),
                    f"<method '{name}' of 'torch._C.TensorBase' objects>",
                )
                self.assertEqual(descriptor.__name__, name)
                self.assertEqual(descriptor.__qualname__, f"TensorBase.{name}")
                self.assertEqual(bound.__name__, name)
                self.assertEqual(bound.__qualname__, f"Tensor.{name}")
                self.assertEqual(descriptor.__doc__, doc)
                self.assertEqual(bound.__doc__, doc)
                self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
                self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
                self.assertFalse(hasattr(descriptor, "__module__"))
                self.assertIsNone(bound.__module__)

            for callable_object, expected_signature in (
                (descriptor, "(self, /)"),
                (bound, "()"),
            ):
                with self.subTest(name=name, callable=type(callable_object).__name__):
                    if sys.version_info >= (3, 13):
                        self.assertEqual(
                            callable_object.__text_signature__, "($self, /)"
                        )
                        self.assertEqual(
                            str(inspect.signature(callable_object)),
                            expected_signature,
                        )
                    else:
                        self.assertIsNone(callable_object.__text_signature__)
                        with self.assertRaises(ValueError):
                            inspect.signature(callable_object)

            cases = (
                (
                    direct_one_argument,
                    f"TensorBase.{name}() takes no arguments (1 given)",
                ),
                (
                    lambda: bound(1),
                    f"Tensor.{name}() takes no arguments (1 given)",
                ),
                (
                    lambda: descriptor(tensor, 1),
                    f"TensorBase.{name}() takes no arguments (1 given)",
                ),
                (
                    direct_two_arguments,
                    f"TensorBase.{name}() takes no arguments (2 given)",
                ),
                (
                    direct_keyword,
                    (
                        f"Tensor.{name}() takes no keyword arguments"
                        if sys.version_info < (3, 11)
                        else f"TensorBase.{name}() takes no keyword arguments"
                    ),
                ),
                (
                    lambda: bound(unexpected=True),
                    f"Tensor.{name}() takes no keyword arguments",
                ),
                (
                    lambda: descriptor(tensor, unexpected=True),
                    f"TensorBase.{name}() takes no keyword arguments",
                ),
                (
                    lambda: descriptor(),
                    f"unbound method TensorBase.{name}() needs an argument",
                ),
                (
                    lambda: descriptor(1),
                    f"descriptor '{name}' for 'torch._C.TensorBase' objects "
                    "doesn't apply to a 'int' object",
                ),
                (
                    lambda: descriptor(self=tensor),
                    f"unbound method TensorBase.{name}() needs an argument",
                ),
            )
            for case, (call, message) in enumerate(cases):
                with self.subTest(name=name, case=case):
                    with self.assertRaises(TypeError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), message)

        operator_bound = tensor.__abs__
        self.assertIs(type(operator_bound), types.BuiltinMethodType)
        self.assertEqual(operator_bound.__name__, "abs")
        self.assertEqual(operator_bound.__qualname__, "Tensor.abs")
        self.assertEqual(operator_bound.__doc__, ABS_DOC)

    def test_descriptor_copying_and_pickling_preserve_alias_identities(self):
        tensor = torch.tensor([-4.0])
        descriptors = (
            ("abs", inspect.getattr_static(torch.Tensor, "abs")),
            ("absolute", inspect.getattr_static(torch.Tensor, "absolute")),
            ("__abs__", inspect.getattr_static(torch.Tensor, "__abs__")),
        )
        for name, descriptor in descriptors:
            with self.subTest(name=name, operation="copy"):
                self.assertIs(copy.copy(descriptor), descriptor)
                self.assertIs(copy.deepcopy(descriptor), descriptor)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol, pickler="pickle"):
                    self.assertIs(
                        pickle.loads(pickle.dumps(descriptor, protocol)), descriptor
                    )
                with self.subTest(
                    name=name, protocol=protocol, pickler="ForkingPickler"
                ):
                    self.assertIs(
                        pickle.loads(ForkingPickler.dumps(descriptor, protocol)),
                        descriptor,
                    )

        for name in ("abs", "absolute", "__abs__"):
            bound = getattr(tensor, name)
            with self.subTest(name=name, operation="bound copy"):
                self.assertIs(copy.copy(bound), bound)
                self.assertIs(copy.deepcopy(bound), bound)

    def test_torch_function_modes_receive_descriptor_and_forward(self):
        tracked = torch.tensor([-4.0], requires_grad=True)
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label, order):
                self.label = label
                self.order = order

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.order.append(self.label)
                return func(*args, **(kwargs or {}))

        plain = torch.tensor([-4.0])
        forms = (
            (
                "abs",
                inspect.getattr_static(torch.Tensor, "abs"),
                lambda tensor: tensor.abs(),
                lambda tensor: tensor.abs(1),
            ),
            (
                "absolute",
                inspect.getattr_static(torch.Tensor, "absolute"),
                lambda tensor: tensor.absolute(),
                lambda tensor: tensor.absolute(1),
            ),
            (
                "operator",
                inspect.getattr_static(torch.Tensor, "abs"),
                lambda tensor: builtins.abs(tensor),
                lambda tensor: builtins.abs(tensor, 1),
            ),
        )
        for form, descriptor, call, invalid_call in forms:
            mode = RecordingMode()
            with mode:
                result = call(tracked)
            with self.subTest(form=form, mode="recording"):
                self.assertIs(result, marker)
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs = mode.calls[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, (torch.Tensor,))
                self.assertEqual(len(args), 1)
                self.assertIs(args[0], tracked)
                self.assertIsNone(kwargs)

            order = []
            with ForwardingMode("lower", order):
                with ForwardingMode("upper", order):
                    forwarded = call(plain)
            with self.subTest(form=form, mode="forwarding"):
                self.assertEqual(order, ["upper", "lower"])
                self.assertEqual(forwarded.tolist(), [4.0])

            order.clear()
            with self.subTest(form=form, mode="forwarding tracked"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^abs\(\): autograd recording is not supported$",
                ):
                    with ForwardingMode("lower", order):
                        with ForwardingMode("upper", order):
                            call(tracked)
                self.assertEqual(order, ["upper", "lower"])

            invalid_mode = RecordingMode()
            with self.subTest(form=form, mode="invalid"):
                with self.assertRaises(TypeError):
                    with invalid_mode:
                        invalid_call(plain)
                self.assertEqual(invalid_mode.calls, [])

    def test_alias_top_level_and_inplace_forms_remain_unsupported(self):
        tensor = torch.tensor([-4.0])
        for name in ("abs", "absolute", "abs_", "absolute_"):
            with self.subTest(owner="torch", name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
        for name in ("abs_", "absolute_"):
            with self.subTest(owner="Tensor", name=name):
                self.assertFalse(hasattr(torch.Tensor, name))
                self.assertFalse(hasattr(tensor, name))
        self.assertTrue(hasattr(torch.Tensor, "absolute"))
        self.assertTrue(hasattr(torch.Tensor, "__abs__"))
        for call in (lambda: tensor.abs(out=None), lambda: tensor.absolute(out=None)):
            with self.assertRaises(TypeError):
                call()


if __name__ == "__main__":
    unittest.main()
