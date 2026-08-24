import copy
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


TANH_DOC = """
tanh() -> Tensor

See :func:`torch.tanh`
"""

TOP_LEVEL_TANH_DOC = """
tanh(input, *, out=None) -> Tensor

Returns a new tensor with the hyperbolic tangent of the elements
of :attr:`input`.

.. math::
    \\text{out}_{i} = \\tanh(\\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.8986, -0.7279,  1.1745,  0.2611])
    >>> torch.tanh(a)
    tensor([ 0.7156, -0.6218,  0.8257,  0.2553])
"""


class TensorTanhTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

    def assert_result(self, actual, source, expected_stride, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, source.shape)
            self.assertEqual(actual.stride(), expected_stride)
            self.assertEqual(actual.storage_offset(), 0)
            self.assertFalse(actual.requires_grad)
            self.assertTrue(actual.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertFalse(actual.is_set_to(source))
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())

    @staticmethod
    def make_cases():
        base = torch.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
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
            np.linspace(-3.0, 3.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist()
        ).contiguous(memory_format=torch.channels_last)
        channels_last_3d = torch.tensor(
            np.linspace(-3.0, 3.0, 720, dtype=np.float32)
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
            ("channels last", channels_last, channels_last.stride()),
            (
                "channels last 3d",
                channels_last_3d,
                channels_last_3d.stride(),
            ),
            (
                "numerical edges",
                torch.tensor(memoryview(special_bits.view(np.float32))),
                (1,),
            ),
        )

    @staticmethod
    def top_level_calls(source):
        return (
            ("positional", lambda: torch.tanh(source)),
            ("input", lambda: torch.tanh(input=source)),
            ("x", lambda: torch.tanh(x=source)),
            ("a", lambda: torch.tanh(a=source)),
            ("x1", lambda: torch.tanh(x1=source)),
            ("out none", lambda: torch.tanh(source, out=None)),
            ("alias and out none", lambda: torch.tanh(x=source, out=None)),
        )

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        expected_special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3EA4_9D52,
                0xBEA4_9D52,
                0x3F42_F7D6,
                0xBF42_F7D6,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7FC1_2345,
                0xFFC1_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        for case, source, expected_stride in self.make_cases():
            output = source.tanh()
            self.assert_result(output, source, expected_stride, case=case)
            if case == "numerical edges":
                np.testing.assert_array_equal(
                    self.tensor_bits(output), expected_special_bits
                )
            else:
                np.testing.assert_allclose(
                    np.asarray(output, dtype=np.float32),
                    np.tanh(np.asarray(source, dtype=np.float32)),
                    rtol=3.0 * np.finfo(np.float32).eps,
                    atol=np.nextafter(np.float32(0), np.float32(1)),
                )

    def test_top_level_calls_reuse_tensor_tanh_values_layouts_and_storage(self):
        for case, source, expected_stride in self.make_cases():
            expected = source.tanh()
            for form, call in self.top_level_calls(source):
                actual = call()
                self.assert_result(
                    actual, source, expected_stride, case=(case, form)
                )
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )

    def test_active_autograd_is_rejected_and_detached_and_no_grad_inputs_work(self):
        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]

        with self.assertRaisesRegex(
            RuntimeError,
            r"^tanh\(\): autograd recording is not supported$",
        ):
            source.tanh()

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tanh\(\): autograd recording is not supported$",
        ):
            extreme.tanh()

        with torch.no_grad():
            no_grad_output = source.tanh()
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.tanh()
        self.assert_result(no_grad_output, source, (1,), case="no_grad")

        detached = source.detach()
        detached_output = detached.tanh()
        self.assert_result(detached_output, detached, (1,), case="detached")
        np.testing.assert_array_equal(
            self.tensor_bits(no_grad_output), self.tensor_bits(detached_output)
        )

    def test_top_level_preserves_inference_only_autograd_behavior(self):
        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]

        for form, call in self.top_level_calls(source):
            with self.subTest(form=form, mode="recording"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^tanh\(\): autograd recording is not supported$",
                ):
                    call()
            with self.subTest(form=form, mode="no_grad"):
                with torch.no_grad():
                    actual = call()
                    expected = source.tanh()
                self.assert_result(actual, source, (1,), case=(form, "no_grad"))
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        for form, call in self.top_level_calls(extreme):
            with self.subTest(form=form, metadata="autograd precedence"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^tanh\(\): autograd recording is not supported$",
                ):
                    call()
            with self.subTest(form=form, metadata="layout planning"):
                with torch.no_grad():
                    with self.assertRaisesRegex(
                        RuntimeError, "Stride calculation overflowed"
                    ):
                        call()

        detached = source.detach()
        expected = detached.tanh()
        for form, call in self.top_level_calls(detached):
            actual = call()
            self.assert_result(actual, detached, (1,), case=(form, "detached"))
            np.testing.assert_array_equal(
                self.tensor_bits(actual), self.tensor_bits(expected)
            )

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([0.5])
        descriptor = inspect.getattr_static(torch.Tensor, "tanh")
        bound = tensor.tanh

        self.assertIs(torch.Tensor.tanh, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'tanh' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "tanh")
        self.assertEqual(descriptor.__qualname__, "TensorBase.tanh")
        self.assertEqual(bound.__name__, "tanh")
        self.assertEqual(bound.__qualname__, "Tensor.tanh")
        self.assertEqual(descriptor.__doc__, TANH_DOC)
        self.assertEqual(bound.__doc__, TANH_DOC)
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
            (lambda: tensor.tanh(1), "TensorBase.tanh() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.tanh() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.tanh() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.tanh(1, 2),
                "TensorBase.tanh() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.tanh(input=tensor),
                (
                    "Tensor.tanh() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.tanh() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.tanh() takes no keyword arguments"),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.tanh() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.tanh() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'tanh' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.tanh() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_dispatch_before_native_execution(self):
        tensor = torch.tensor([0.5], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "tanh")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.tanh()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertIsNone(kwargs)

        extreme = torch.zeros((0,)).reshape((0, sys.maxsize, 3))
        bypass = RecordingMode()
        with bypass:
            self.assertIs(extreme.tanh(), marker)
        self.assertEqual(len(bypass.calls), 1)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        plain = torch.tensor([0.5])
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = plain.tanh()
        self.assertEqual(order, ["upper", "lower"])
        np.testing.assert_allclose(forwarded.tolist(), [0.46211717], rtol=1.0e-6)

        order.clear()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tanh\(\): autograd recording is not supported$",
        ):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    tensor.tanh()
        self.assertEqual(order, ["upper", "lower"])

        invalid_mode = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid_mode:
                plain.tanh(1)
        self.assertEqual(invalid_mode.calls, [])

    def test_top_level_modes_and_overrides_dispatch_before_native_limits(self):
        tensor = torch.tensor([0.5], requires_grad=True)
        destination = torch.tensor([0.0])
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
            self.assertIs(torch.tanh(input=tensor, out=destination), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.tanh)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "out": destination})

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.tanh(Override()), marker)
        self.assertIs(torch.tanh(torch.tensor([0.5]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for dispatched, types_, _, _ in override_calls:
            self.assertIs(dispatched, torch.tanh)
            self.assertEqual(types_, (Override,))

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

        self.assertIs(torch.tanh(BaseOverride(), out=DerivedOverride()), marker)
        self.assertEqual(subclass_order, ["derived"])

        forwarding_order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        plain = torch.tensor([0.5])
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.tanh(input=plain, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        np.testing.assert_array_equal(
            self.tensor_bits(forwarded), self.tensor_bits(plain.tanh())
        )

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
            self.assertIs(torch.tanh(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

        invalid_mode = RecordingMode()
        with self.assertRaisesRegex(
            TypeError,
            r"^tanh\(\): argument 'out' must be Tensor, not list$",
        ):
            with invalid_mode:
                torch.tanh(tensor, out=[])
        self.assertEqual(invalid_mode.calls, [])

    def test_top_level_concrete_out_is_rejected_without_mutation(self):
        source = torch.tensor([0.0, -0.0, 0.5], requires_grad=True)
        destination = torch.tensor([17.0, 19.0, 23.0])
        for form, call in (
            ("positional", lambda: torch.tanh(source, out=destination)),
            ("keyword", lambda: torch.tanh(input=source, out=destination)),
            ("alias", lambda: torch.tanh(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^tanh\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.tolist(), [17.0, 19.0, 23.0])

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tanh\(\): the 'out' argument is not supported$",
        ):
            torch.tanh(extreme, out=destination)
        self.assertEqual(destination.tolist(), [17.0, 19.0, 23.0])

        with torch.no_grad():
            actual = torch.tanh(source, out=None)
            expected = source.tanh()
        np.testing.assert_array_equal(
            self.tensor_bits(actual), self.tensor_bits(expected)
        )

    def test_top_level_builtin_metadata_exports_copying_and_pickling(self):
        function = torch.tanh
        self.assertIs(function, torch._C.tanh)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "tanh")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.tanh")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, TOP_LEVEL_TANH_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method tanh of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.tanh, function)
        for action in (
            lambda: setattr(owner, "tanh", None),
            lambda: delattr(owner, "tanh"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.tanh, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("tanh"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["tanh"], function)

    def test_top_level_binding_and_type_error_precedence(self):
        tensor = torch.tensor([0.5])
        cases = (
            (
                lambda: torch.tanh(),
                'tanh() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.tanh(tensor, tensor),
                "tanh() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.tanh(tensor, input=tensor),
                "tanh() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.tanh(out=tensor),
                'tanh() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.tanh(extra=tensor),
                'tanh() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.tanh(1, extra=True),
                "tanh(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.tanh(input=[]),
                "tanh(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.tanh(tensor, out=[]),
                "tanh(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.tanh(tensor, extra=True, out=[]),
                "tanh(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.tanh(tensor, extra=True),
                "tanh() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.tanh(input=tensor, a=tensor),
                "tanh() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.tanh(a=tensor, x=tensor, out=None),
                "tanh() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.tanh(x=tensor, a=tensor, out=None),
                "tanh() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.tanh(np.zeros((2, 3), dtype=np.float32)),
                (
                    "tanh(): argument 'input' (position 1) must be Tensor, "
                    "not numpy.ndarray"
                ),
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_functional_and_inplace_forms_remain_unsupported(self):
        tensor = torch.tensor([0.5])
        self.assertFalse(hasattr(torch.nn.functional, "tanh"))
        self.assertFalse(hasattr(torch.Tensor, "tanh_"))
        self.assertFalse(hasattr(tensor, "tanh_"))
        self.assertFalse(hasattr(torch, "tanh_"))
        self.assertNotIn("tanh_", torch.__all__)
        with self.assertRaises(TypeError):
            tensor.tanh(out=None)


if __name__ == "__main__":
    unittest.main()
