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


CEIL_DOC = """
ceil() -> Tensor

See :func:`torch.ceil`
"""

TOP_LEVEL_CEIL_DOC = """
ceil(input, *, out=None) -> Tensor

Returns a new tensor with the ceil of the elements of :attr:`input`,
the smallest integer greater than or equal to each element.

For integer inputs, follows the array-api convention of returning a
copy of the input tensor.

.. math::
    \\text{out}_{i} = \\left\\lceil \\text{input}_{i} \\right\\rceil

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.6341, -1.4208, -1.0900,  0.5826])
    >>> torch.ceil(a)
    tensor([-0., -1., -1.,  1.])
"""


SPECIAL_INPUT_BITS = np.asarray(
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

SPECIAL_OUTPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x3F80_0000,
        0x8000_0000,
        0x3F80_0000,
        0x8000_0000,
        0x3F80_0000,
        0x8000_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x8000_0000,
        0x8000_0000,
        0xBF80_0000,
        0xBF80_0000,
        0x4000_0000,
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


def make_cases(module):
    base = module.tensor(
        np.linspace(-3.75, 3.75, 24, dtype=np.float32)
        .reshape(2, 3, 4)
        .tolist(),
        dtype=module.float32,
    )
    strided = base.transpose(0, 2)
    channels_last = module.tensor(
        np.linspace(-15.0, 15.0, 120, dtype=np.float32)
        .reshape(2, 3, 4, 5)
        .tolist(),
        dtype=module.float32,
    ).contiguous(memory_format=module.channels_last)
    channels_last_3d = module.tensor(
        np.linspace(-90.0, 90.0, 720, dtype=np.float32)
        .reshape(2, 3, 4, 5, 6)
        .tolist(),
        dtype=module.float32,
    ).contiguous(memory_format=module.channels_last_3d)
    return (
        ("scalar", module.tensor(-0.0, dtype=module.float32), ()),
        (
            "empty offset",
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            (2, 1),
        ),
        ("empty singleton trailing", module.zeros((0, 1)), (1, 1)),
        ("empty singleton middle", module.zeros((0, 1, 2)), (2, 2, 1)),
        ("empty singleton surrounding", module.zeros((1, 0, 1)), (1, 1, 1)),
        ("offset", strided[1], (1, 3)),
        ("noncontiguous", strided, (1, 4, 12)),
        ("channels last", channels_last, channels_last.stride()),
        ("channels last 3d", channels_last_3d, channels_last_3d.stride()),
        (
            "numerical edges",
            module.tensor(memoryview(SPECIAL_INPUT_BITS.view(np.float32))),
            (1,),
        ),
    )


class TensorCeilTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

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

    @staticmethod
    def top_level_calls(source):
        return (
            ("positional", lambda: torch.ceil(source)),
            ("input", lambda: torch.ceil(input=source)),
            ("x", lambda: torch.ceil(x=source)),
            ("a", lambda: torch.ceil(a=source)),
            ("x1", lambda: torch.ceil(x1=source)),
            ("out none", lambda: torch.ceil(source, out=None)),
            ("alias and out none", lambda: torch.ceil(x=source, out=None)),
        )

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        for case, source, expected_stride in make_cases(torch):
            output = source.ceil()
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

            actual_bits = self.tensor_bits(output)
            if case == "numerical edges":
                expected_bits = SPECIAL_OUTPUT_BITS
            else:
                values = np.asarray(source, dtype=np.float32).reshape(-1)
                expected_bits = np.ceil(values).astype(np.float32).view(np.uint32)
            with self.subTest(case=case, values=True):
                np.testing.assert_array_equal(actual_bits, expected_bits)

    def test_top_level_calls_reuse_tensor_ceil_bits_layouts_and_storage(self):
        for case, source, expected_stride in make_cases(torch):
            expected = source.ceil()
            for form, call in self.top_level_calls(source):
                actual = call()
                with self.subTest(case=case, form=form, metadata=True):
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
                with self.subTest(case=case, form=form, values=True):
                    np.testing.assert_array_equal(
                        self.tensor_bits(actual), self.tensor_bits(expected)
                    )

    def test_active_autograd_records_reusable_ceil_backward_zero_vjp(self):
        for case, (leaf, source) in enumerate(self.make_tracked_cases()):
            with self.subTest(case=case):
                expected = source.detach().ceil()
                output = source.ceil()
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
                    ", grad_fn=<CeilBackward0>",
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
            extreme.ceil()
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.ceil()

    def test_zero_vjp_ignores_upstream_values_composes_and_accumulates(self):
        leaf = torch.tensor([-1.25, -0.0, 1.75, 4.5], requires_grad=True)
        weights = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0])
        (leaf.ceil() * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), np.zeros((4,), dtype=np.uint32)
        )

        accumulated = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
        (accumulated * 3.0).sum().backward()
        first = self.tensor_bits(accumulated.grad).copy()
        accumulated.ceil().sum().backward()
        np.testing.assert_array_equal(self.tensor_bits(accumulated.grad), first)

        composed = torch.tensor([-0.5, 0.5], requires_grad=True)
        loss = composed.sin().ceil().sum()
        loss.backward()
        np.testing.assert_array_equal(
            self.tensor_bits(composed.grad), np.zeros((2,), dtype=np.uint32)
        )
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

        higher_order = torch.tensor(0.25, requires_grad=True)
        higher_order_loss = higher_order.ceil()
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
            expected = detached.ceil()
            with torch.no_grad():
                actual = source.ceil()
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
        descriptor = inspect.getattr_static(torch.Tensor, "ceil")
        bound = tensor.ceil

        self.assertIs(torch.Tensor.ceil, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'ceil' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "ceil")
        self.assertEqual(descriptor.__qualname__, "TensorBase.ceil")
        self.assertEqual(bound.__name__, "ceil")
        self.assertEqual(bound.__qualname__, "Tensor.ceil")
        self.assertEqual(descriptor.__doc__, CEIL_DOC)
        self.assertEqual(bound.__doc__, CEIL_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")

        cases = (
            (lambda: tensor.ceil(1), "TensorBase.ceil() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.ceil() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.ceil() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.ceil(1, 2),
                "TensorBase.ceil() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.ceil(input=tensor),
                (
                    "Tensor.ceil() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.ceil() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.ceil() takes no keyword arguments"),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.ceil() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.ceil() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'ceil' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.ceil() needs an argument",
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
        descriptor = inspect.getattr_static(torch.Tensor, "ceil")
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
            result = tracked.ceil()
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
                forwarded = plain.ceil()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [2.0])

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                tracked_forwarded = tracked.ceil()
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(tracked_forwarded.requires_grad)
        self.assertFalse(tracked_forwarded.is_leaf)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                tracked_forwarded
            ),
            ", grad_fn=<CeilBackward0>",
        )

        old_recursion_limit = sys.getrecursionlimit()
        declining = RecordingMode(NotImplemented)
        try:
            sys.setrecursionlimit(80)
            with declining:
                with self.assertRaises(RecursionError):
                    plain.ceil()
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
                plain.ceil(1)
        self.assertEqual(invalid.calls, [])

    def test_top_level_autograd_and_no_grad_reuse_method_path(self):
        for case, (_, source) in enumerate(self.make_tracked_cases()):
            detached = source.detach()
            expected = detached.ceil()
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
                        ", grad_fn=<CeilBackward0>",
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
            torch.ceil(extreme)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                torch.ceil(extreme)

    def test_top_level_concrete_out_is_rejected_without_mutation(self):
        source = torch.tensor([1.25, -1.25], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        destination_pointer = destination.data_ptr()
        expected_bits = self.tensor_bits(destination).copy()
        for form, call in (
            ("positional", lambda: torch.ceil(source, out=destination)),
            ("keyword", lambda: torch.ceil(input=source, out=destination)),
            ("alias", lambda: torch.ceil(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^ceil\(\): the 'out' argument is not supported$",
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
            self.assertIs(torch.ceil(input=tensor, out=destination), marker)
        self.assertEqual(
            mode.calls,
            [(torch.ceil, (), (), {"input": tensor, "out": destination})],
        )

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.ceil(Override()), marker)
        self.assertIs(torch.ceil(torch.tensor([1.25]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.ceil)
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

        self.assertIs(torch.ceil(BaseOverride(), out=DerivedOverride()), marker)
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
                forwarded = torch.ceil(input=plain, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [2.0])

        forwarding_order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                tracked_forwarded = torch.ceil(input=tensor, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                tracked_forwarded
            ),
            ", grad_fn=<CeilBackward0>",
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
            self.assertIs(torch.ceil(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

        invalid = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid:
                torch.ceil(tensor, tensor)
        self.assertEqual(invalid.calls, [])

    def test_top_level_builtin_metadata_exports_copying_and_pickling(self):
        function = torch.ceil
        self.assertIs(function, torch._C.ceil)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "ceil")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.ceil")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, TOP_LEVEL_CEIL_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method ceil of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.ceil, function)
        for action in (
            lambda: setattr(owner, "ceil", None),
            lambda: delattr(owner, "ceil"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.ceil, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("ceil"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["ceil"], function)

    def test_top_level_binding_and_type_error_precedence(self):
        tensor = torch.tensor([1.25])
        cases = (
            (
                lambda: torch.ceil(),
                'ceil() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.ceil(tensor, tensor),
                "ceil() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.ceil(tensor, input=tensor),
                "ceil() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.ceil(out=tensor),
                'ceil() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.ceil(extra=tensor),
                'ceil() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.ceil(1, extra=True),
                "ceil(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.ceil(input=[]),
                "ceil(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.ceil(tensor, out=[]),
                "ceil(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.ceil(tensor, extra=True, out=[]),
                "ceil(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.ceil(tensor, extra=True),
                "ceil() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.ceil(input=tensor, a=tensor),
                "ceil() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.ceil(a=tensor, x=tensor, out=None),
                "ceil() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.ceil(x=tensor, a=tensor, out=None),
                "ceil() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.ceil(np.zeros((2, 3), dtype=np.float32)),
                (
                    "ceil(): argument 'input' (position 1) must be Tensor, "
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
        self.assertTrue(hasattr(torch, "ceil"))
        self.assertFalse(hasattr(torch.Tensor, "ceil_"))
        self.assertFalse(hasattr(tensor, "ceil_"))
        self.assertFalse(hasattr(torch, "ceil_"))
        self.assertNotIn("ceil_", torch.__all__)
        with self.assertRaises(TypeError):
            tensor.ceil(out=None)


if __name__ == "__main__":
    unittest.main()
