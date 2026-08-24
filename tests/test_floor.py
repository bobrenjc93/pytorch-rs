import copy
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


FLOOR_DOC = """
floor() -> Tensor

See :func:`torch.floor`
"""

TOP_LEVEL_FLOOR_DOC = """
floor(input, *, out=None) -> Tensor

Returns a new tensor with the floor of the elements of :attr:`input`,
the largest integer less than or equal to each element.

For integer inputs, follows the array-api convention of returning a
copy of the input tensor.

.. math::
    \\text{out}_{i} = \\left\\lfloor \\text{input}_{i} \\right\\rfloor

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.8166,  1.5308, -0.2530, -0.2091])
    >>> torch.floor(a)
    tensor([-1.,  1., -1., -1.])
"""


class TensorFloorTests(unittest.TestCase):
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
                0x3EFF_FFFF,
                0x3F00_0000,
                0x3F7F_FFFF,
                0x3F80_0000,
                0xBF00_0000,
                0xBF7F_FFFF,
                0xBF80_0000,
                0xBFC0_0000,
                0x3FC0_0000,
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
            ("positional", lambda: torch.floor(source)),
            ("input", lambda: torch.floor(input=source)),
            ("x", lambda: torch.floor(x=source)),
            ("a", lambda: torch.floor(a=source)),
            ("x1", lambda: torch.floor(x1=source)),
            ("out none", lambda: torch.floor(source, out=None)),
            ("alias and out none", lambda: torch.floor(x=source, out=None)),
        )

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        expected_special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0000,
                0xBF80_0000,
                0x0000_0000,
                0xBF80_0000,
                0x0000_0000,
                0xBF80_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x3F80_0000,
                0xBF80_0000,
                0xBF80_0000,
                0xBF80_0000,
                0xC000_0000,
                0x3F80_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC1_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        for case, source, expected_stride in self.make_cases():
            output = source.floor()
            self.assert_result(output, source, expected_stride, case=case)
            if case == "numerical edges":
                np.testing.assert_array_equal(
                    self.tensor_bits(output), expected_special_bits
                )
            else:
                np.testing.assert_array_equal(
                    self.tensor_bits(output),
                    np.floor(np.asarray(source, dtype=np.float32))
                    .reshape(-1)
                    .view(np.uint32),
                )

    def test_top_level_calls_reuse_tensor_floor_bits_layouts_and_storage(self):
        for case, source, expected_stride in self.make_cases():
            expected = source.floor()
            for form, call in self.top_level_calls(source):
                actual = call()
                self.assert_result(
                    actual, source, expected_stride, case=(case, form)
                )
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )

    @staticmethod
    def make_tracked_cases():
        scalar = torch.tensor(-1.25, requires_grad=True)
        empty_leaf = torch.zeros((2, 0, 3), requires_grad=True)
        empty = empty_leaf.transpose(0, 2)[1]
        leaf = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            requires_grad=True,
        )
        strided = leaf.transpose(0, 2)
        channels_last_leaf = torch.tensor(
            np.linspace(-15.0, 15.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist(),
            requires_grad=True,
        )
        channels_last = channels_last_leaf.contiguous(
            memory_format=torch.channels_last
        )
        channels_last_3d_leaf = torch.tensor(
            np.linspace(-90.0, 90.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist(),
            requires_grad=True,
        )
        channels_last_3d = channels_last_3d_leaf.contiguous(
            memory_format=torch.channels_last_3d
        )
        return (
            (scalar, scalar),
            (empty_leaf, empty),
            (leaf, strided[1]),
            (leaf, strided),
            (channels_last_leaf, channels_last),
            (channels_last_3d_leaf, channels_last_3d),
        )

    def test_active_autograd_records_reusable_floor_backward_zero_vjp(self):
        for case, (leaf, source) in enumerate(self.make_tracked_cases()):
            with self.subTest(case=case):
                expected = source.detach().floor()
                output = source.floor()
                self.assertEqual(output.shape, expected.shape)
                self.assertEqual(output.stride(), expected.stride())
                self.assertEqual(output.storage_offset(), 0)
                self.assertTrue(output.requires_grad)
                self.assertFalse(output.is_leaf)
                self.assertFalse(output.is_set_to(source))
                np.testing.assert_array_equal(
                    self.tensor_bits(output), self.tensor_bits(expected)
                )
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
                    ", grad_fn=<FloorBackward0>",
                )

                loss = output if output.numel() == 1 else output.sum()
                loss.backward()
                loss.backward()
                gradient_bits = self.tensor_bits(leaf.grad)
                np.testing.assert_array_equal(
                    gradient_bits, np.zeros(gradient_bits.shape, dtype=np.uint32)
                )

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            extreme.floor()
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.floor()

    def test_zero_vjp_ignores_upstream_values_composes_and_accumulates(self):
        leaf = torch.tensor([-1.25, -0.0, 1.75, 4.5], requires_grad=True)
        weights = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0])
        (leaf.floor() * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), np.zeros((4,), dtype=np.uint32)
        )

        accumulated = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
        (accumulated * 3.0).sum().backward()
        first = self.tensor_bits(accumulated.grad).copy()
        accumulated.floor().sum().backward()
        np.testing.assert_array_equal(self.tensor_bits(accumulated.grad), first)

        composed = torch.tensor([-0.5, 0.5], requires_grad=True)
        loss = composed.sin().floor().sum()
        loss.backward()
        np.testing.assert_array_equal(
            self.tensor_bits(composed.grad), np.zeros((2,), dtype=np.uint32)
        )
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

        higher_order = torch.tensor(0.25, requires_grad=True)
        higher_order_loss = higher_order.floor()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertEqual(higher_order.grad.item(), 0.0)

    def test_detached_and_no_grad_inputs_use_the_inference_path(self):
        for case, (_, source) in enumerate(self.make_tracked_cases()):
            detached = source.detach()
            expected = detached.floor()
            with torch.no_grad():
                actual = source.floor()
            with self.subTest(case=case, mode="no_grad"):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                self.assertFalse(actual.is_set_to(source))
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )
            with self.subTest(case=case, mode="detached"):
                self.assertFalse(expected.is_set_to(detached))
                if detached.numel():
                    self.assertNotEqual(expected.data_ptr(), detached.data_ptr())

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([1.25])
        descriptor = inspect.getattr_static(torch.Tensor, "floor")
        bound = tensor.floor

        self.assertIs(torch.Tensor.floor, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'floor' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "floor")
        self.assertEqual(descriptor.__qualname__, "TensorBase.floor")
        self.assertEqual(bound.__name__, "floor")
        self.assertEqual(bound.__qualname__, "Tensor.floor")
        self.assertEqual(descriptor.__doc__, FLOOR_DOC)
        self.assertEqual(bound.__doc__, FLOOR_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")

        cases = (
            (lambda: tensor.floor(1), "TensorBase.floor() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.floor() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.floor() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.floor(1, 2),
                "TensorBase.floor() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.floor(input=tensor),
                (
                    "Tensor.floor() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.floor() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.floor() takes no keyword arguments"),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.floor() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.floor() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'floor' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.floor() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_dispatch_before_native_execution(self):
        tracked = torch.tensor([1.25], requires_grad=True)
        plain = tracked.detach()
        descriptor = inspect.getattr_static(torch.Tensor, "floor")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            result = tracked.floor()
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

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = plain.floor()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [1.0])

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                tracked_forwarded = tracked.floor()
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(tracked_forwarded.requires_grad)
        self.assertFalse(tracked_forwarded.is_leaf)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                tracked_forwarded
            ),
            ", grad_fn=<FloorBackward0>",
        )

        old_recursion_limit = sys.getrecursionlimit()
        declining = RecordingMode(NotImplemented)
        try:
            sys.setrecursionlimit(80)
            with declining:
                with self.assertRaises(RecursionError):
                    plain.floor()
                self.assertEqual(
                    len(torch.overrides._get_current_function_mode_stack()), 1
                )
        finally:
            sys.setrecursionlimit(old_recursion_limit)
        self.assertGreater(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                plain.floor(1)
        self.assertEqual(invalid.calls, [])

    def test_top_level_autograd_and_no_grad_reuse_method_path(self):
        for case, (_, source) in enumerate(self.make_tracked_cases()):
            detached = source.detach()
            expected = detached.floor()
            for form, call in self.top_level_calls(source):
                with self.subTest(case=case, form=form, mode="recording"):
                    actual = call()
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(actual.storage_offset(), 0)
                    self.assertTrue(actual.requires_grad)
                    self.assertFalse(actual.is_leaf)
                    self.assertFalse(actual.is_set_to(source))
                    np.testing.assert_array_equal(
                        self.tensor_bits(actual), self.tensor_bits(expected)
                    )
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            actual
                        ),
                        ", grad_fn=<FloorBackward0>",
                    )
                with self.subTest(case=case, form=form, mode="no_grad"):
                    with torch.no_grad():
                        actual = call()
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(
                        actual.storage_offset(), expected.storage_offset()
                    )
                    self.assertFalse(actual.requires_grad)
                    self.assertTrue(actual.is_leaf)
                    self.assertFalse(actual.is_set_to(source))
                    np.testing.assert_array_equal(
                        self.tensor_bits(actual), self.tensor_bits(expected)
                    )

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            torch.floor(extreme)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                torch.floor(extreme)

    def test_top_level_concrete_out_is_rejected_without_mutation(self):
        source = torch.tensor([1.25, -1.25], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        destination_pointer = destination.data_ptr()
        expected_bits = self.tensor_bits(destination).copy()
        for form, call in (
            ("positional", lambda: torch.floor(source, out=destination)),
            ("keyword", lambda: torch.floor(input=source, out=destination)),
            ("alias", lambda: torch.floor(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^floor\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.data_ptr(), destination_pointer)
                np.testing.assert_array_equal(
                    self.tensor_bits(destination), expected_bits
                )
                self.assertIsNone(source.grad)

    def test_top_level_modes_and_overrides_observe_original_call(self):
        tensor = torch.tensor([1.25], requires_grad=True)
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
            self.assertIs(torch.floor(input=tensor, out=destination), marker)
        self.assertEqual(
            mode.calls,
            [(torch.floor, (), (), {"input": tensor, "out": destination})],
        )

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.floor(Override()), marker)
        self.assertIs(torch.floor(torch.tensor([1.25]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.floor)
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

        self.assertIs(torch.floor(BaseOverride(), out=DerivedOverride()), marker)
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
                forwarded = torch.floor(input=plain, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [1.0])

        forwarding_order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                tracked_forwarded = torch.floor(input=tensor, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                tracked_forwarded
            ),
            ", grad_fn=<FloorBackward0>",
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
            self.assertIs(torch.floor(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

        invalid = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid:
                torch.floor(tensor, tensor)
        self.assertEqual(invalid.calls, [])

    def test_top_level_builtin_metadata_exports_copying_and_pickling(self):
        function = torch.floor
        self.assertIs(function, torch._C.floor)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "floor")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.floor")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, TOP_LEVEL_FLOOR_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method floor of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.floor, function)
        for action in (
            lambda: setattr(owner, "floor", None),
            lambda: delattr(owner, "floor"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.floor, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("floor"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["floor"], function)

    def test_top_level_binding_and_type_error_precedence(self):
        tensor = torch.tensor([1.25])
        cases = (
            (
                lambda: torch.floor(),
                'floor() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.floor(tensor, tensor),
                "floor() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.floor(tensor, input=tensor),
                "floor() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.floor(out=tensor),
                'floor() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.floor(extra=tensor),
                'floor() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.floor(1, extra=True),
                "floor(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.floor(input=[]),
                "floor(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.floor(tensor, out=[]),
                "floor(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.floor(tensor, extra=True, out=[]),
                "floor(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.floor(tensor, extra=True),
                "floor() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.floor(input=tensor, a=tensor),
                "floor() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.floor(a=tensor, x=tensor, out=None),
                "floor() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.floor(x=tensor, a=tensor, out=None),
                "floor() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.floor(np.zeros((2, 3), dtype=np.float32)),
                (
                    "floor(): argument 'input' (position 1) must be Tensor, "
                    "not numpy.ndarray"
                ),
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_inplace_forms_remain_unsupported(self):
        tensor = torch.tensor([1.25])
        self.assertTrue(hasattr(torch, "floor"))
        self.assertFalse(hasattr(torch.Tensor, "floor_"))
        self.assertFalse(hasattr(tensor, "floor_"))
        self.assertFalse(hasattr(torch, "floor_"))
        self.assertNotIn("floor_", torch.__all__)
        with self.assertRaises(TypeError):
            tensor.floor(out=None)


if __name__ == "__main__":
    unittest.main()
