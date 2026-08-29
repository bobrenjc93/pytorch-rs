import builtins
import copy
import ctypes
import inspect
import pickle
import re
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

TOP_LEVEL_ABSOLUTE_DOC = """
absolute(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.abs`
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
            self.assertIs(output.layout, torch.strided)
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

    @staticmethod
    def top_level_supported_calls(source):
        return (
            ("torch.abs positional", lambda: torch.abs(source)),
            ("torch.abs input", lambda: torch.abs(input=source)),
            ("torch.abs x", lambda: torch.abs(x=source)),
            ("torch.abs a", lambda: torch.abs(a=source)),
            ("torch.abs x1", lambda: torch.abs(x1=source)),
            ("torch.abs out none", lambda: torch.abs(source, out=None)),
            ("torch.abs alias out none", lambda: torch.abs(x=source, out=None)),
            ("torch.absolute positional", lambda: torch.absolute(source)),
            ("torch.absolute input", lambda: torch.absolute(input=source)),
            ("torch.absolute x", lambda: torch.absolute(x=source)),
            ("torch.absolute a", lambda: torch.absolute(a=source)),
            ("torch.absolute x1", lambda: torch.absolute(x1=source)),
            ("torch.absolute out none", lambda: torch.absolute(source, out=None)),
            (
                "torch.absolute alias out none",
                lambda: torch.absolute(x=source, out=None),
            ),
        )

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        for case, source, expected_stride in self.make_cases():
            for form, call in (
                self.supported_calls(source) + self.top_level_supported_calls(source)
            ):
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
        for form, call in self.top_level_supported_calls(source):
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
        for form, call in self.top_level_supported_calls(extreme):
            with self.subTest(form=form, mode="extreme recording"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^abs\(\): autograd recording is not supported$",
                ):
                    call()

        for form, call in (
            self.supported_calls(source) + self.top_level_supported_calls(source)
        ):
            with torch.no_grad():
                output = call()
            self.assert_result(output, source, (1,), case=(form, "no_grad"))

        for form, call in (
            self.supported_calls(extreme) + self.top_level_supported_calls(extreme)
        ):
            with self.subTest(form=form, mode="extreme no_grad"):
                with torch.no_grad():
                    with self.assertRaisesRegex(
                        RuntimeError, "Stride calculation overflowed"
                    ):
                        call()

        detached = source.detach()
        for form, call in (
            self.supported_calls(detached) + self.top_level_supported_calls(detached)
        ):
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

    def test_top_level_concrete_out_tensor_is_rejected_without_mutation(self):
        source = torch.tensor([-4.0, -0.0, 2.0])
        destination = torch.tensor([17.0, 19.0, 23.0])
        for name, function in (("abs", torch.abs), ("absolute", torch.absolute)):
            for form, call in (
                ("positional", lambda function=function: function(source, out=destination)),
                ("keyword", lambda function=function: function(input=source, out=destination)),
                ("alias", lambda function=function: function(x=source, out=destination)),
            ):
                with self.subTest(name=name, form=form):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"^{name}\(\): the 'out' argument is not supported$",
                    ):
                        call()
                    self.assertEqual(destination.tolist(), [17.0, 19.0, 23.0])

        self.assert_result(
            torch.abs(source, out=None),
            source,
            (1,),
            case=("torch.abs", "explicit out none"),
        )
        self.assert_result(
            torch.absolute(source, out=None),
            source,
            (1,),
            case=("torch.absolute", "explicit out none"),
        )

    def test_top_level_torch_function_modes_and_overrides_observe_original_call(self):
        tracked = torch.tensor([-4.0], requires_grad=True)
        plain = torch.tensor([-4.0])
        destination = torch.tensor([0.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label, order):
                self.label = label
                self.order = order

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.order.append(self.label)
                return func(*args, **(kwargs or {}))

        for name, function in (("abs", torch.abs), ("absolute", torch.absolute)):
            for call, expected_args, expected_kwargs in (
                (lambda function=function: function(tracked), (tracked,), None),
                (
                    lambda function=function: function(input=tracked),
                    (),
                    {"input": tracked},
                ),
                (
                    lambda function=function: function(tracked, out=None),
                    (tracked,),
                    {"out": None},
                ),
                (
                    lambda function=function: function(input=tracked, out=destination),
                    (),
                    {"input": tracked, "out": destination},
                ),
            ):
                mode = RecordingMode()
                with mode:
                    result = call()
                with self.subTest(name=name, kwargs=expected_kwargs):
                    self.assertIs(result, marker)
                    self.assertEqual(len(mode.calls), 1)
                    dispatched, dispatch_types, args, kwargs = mode.calls[0]
                    self.assertIs(dispatched, function)
                    self.assertEqual(dispatch_types, ())
                    self.assertEqual(args, expected_args)
                    self.assertEqual(kwargs, expected_kwargs)

            order = []
            with ForwardingMode("lower", order):
                with ForwardingMode("upper", order):
                    forwarded = function(input=plain, out=None)
            with self.subTest(name=name, mode="forwarding"):
                self.assertEqual(order, ["upper", "lower"])
                self.assertEqual(forwarded.tolist(), [4.0])

            order.clear()
            with self.subTest(name=name, mode="forwarding tracked"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^abs\(\): autograd recording is not supported$",
                ):
                    with ForwardingMode("lower", order):
                        with ForwardingMode("upper", order):
                            function(input=tracked, out=None)
                self.assertEqual(order, ["upper", "lower"])

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        for name, function in (("abs", torch.abs), ("absolute", torch.absolute)):
            for call in (
                lambda function=function: function(Override()),
                lambda function=function: function(torch.tensor([-1.0]), out=Override()),
            ):
                with self.subTest(name=name, override=True):
                    self.assertIs(call(), marker)
                    dispatched, dispatch_types, _, _ = override_calls.pop()
                    self.assertIs(dispatched, function)
                    self.assertEqual(dispatch_types, (Override,))

    def test_top_level_builtin_metadata_documentation_exports_and_pickling(self):
        for name, doc in (
            ("abs", TOP_LEVEL_ABS_DOC),
            ("absolute", TOP_LEVEL_ABSOLUTE_DOC),
        ):
            function = getattr(torch, name)
            with self.subTest(name=name):
                self.assertIs(type(function), types.BuiltinFunctionType)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, f"_VariableFunctionsClass.{name}")
                self.assertEqual(function.__module__, "torch")
                self.assertEqual(function.__doc__, doc)
                self.assertIsNone(function.__text_signature__)
                self.assertRegex(
                    repr(function),
                    rf"^<built-in method {name} of type object at 0x[0-9a-f]+>$",
                )
                with self.assertRaises(ValueError):
                    inspect.signature(function)

                owner = function.__reduce__()[1][0]
                self.assertEqual(owner.__name__, "_VariableFunctionsClass")
                self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
                self.assertEqual(owner.__module__, "torch_rs._C")
                self.assertIs(owner, torch._C._VariableFunctionsClass)
                self.assertIs(getattr(owner, name), function)
                for action in (
                    lambda owner=owner, name=name: setattr(owner, name, None),
                    lambda owner=owner, name=name: delattr(owner, name),
                ):
                    with self.assertRaises(TypeError):
                        action()
                    self.assertIs(getattr(owner, name), function)

                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )
                    self.assertIs(
                        pickle.loads(ForkingPickler.dumps(function, protocol)),
                        function,
                    )

                self.assertEqual(torch.__all__.count(name), 1)
                wildcard_namespace = {}
                exec("from torch_rs import *", wildcard_namespace)
                self.assertIs(wildcard_namespace[name], function)

        self.assertIsNot(torch.abs, torch.absolute)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))

    def test_top_level_binding_and_type_error_precedence(self):
        tensor = torch.tensor([1.0])
        for name, function in (("abs", torch.abs), ("absolute", torch.absolute)):
            cases = (
                (
                    lambda function=function: function(),
                    f'{name}() missing 1 required positional arguments: "input"',
                ),
                (
                    lambda function=function: function(tensor, tensor),
                    f"{name}() takes 1 positional argument but 2 were given",
                ),
                (
                    lambda function=function: function(tensor, input=tensor),
                    f"{name}() got multiple values for argument 'input'",
                ),
                (
                    lambda function=function: function(out=tensor),
                    f'{name}() missing 1 required positional arguments: "input"',
                ),
                (
                    lambda function=function: function(extra=tensor),
                    f'{name}() missing 1 required positional arguments: "input"',
                ),
                (
                    lambda function=function: function(1, extra=True),
                    f"{name}(): argument 'input' (position 1) must be Tensor, not int",
                ),
                (
                    lambda function=function: function(input=[]),
                    f"{name}(): argument 'input' must be Tensor, not list",
                ),
                (
                    lambda function=function: function(tensor, out=[]),
                    f"{name}(): argument 'out' must be Tensor, not list",
                ),
                (
                    lambda function=function: function(tensor, extra=True, out=[]),
                    f"{name}(): argument 'out' must be Tensor, not list",
                ),
                (
                    lambda function=function: function(tensor, extra=True),
                    f"{name}() got an unexpected keyword argument 'extra'",
                ),
                (
                    lambda function=function: function(input=tensor, a=tensor),
                    f"{name}() got an unexpected keyword argument 'a'",
                ),
                (
                    lambda function=function: function(a=tensor, x=tensor, out=None),
                    f"{name}() got an unexpected keyword argument 'a'",
                ),
                (
                    lambda function=function: function(x=tensor, a=tensor, out=None),
                    f"{name}() got an unexpected keyword argument 'x'",
                ),
                (
                    lambda function=function: function(np.zeros((2, 3), dtype=np.float32)),
                    f"{name}(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
                ),
            )
            for call, message in cases:
                with self.subTest(name=name, message=message):
                    with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                        call()

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

    def test_top_level_aliases_are_supported_and_inplace_forms_remain_unsupported(self):
        tensor = torch.tensor([-4.0])
        for name in ("abs", "absolute"):
            with self.subTest(owner="torch", name=name):
                self.assertTrue(hasattr(torch, name))
                self.assertIn(name, torch.__all__)
        for name in ("abs_", "absolute_"):
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
