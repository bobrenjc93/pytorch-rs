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

    def assert_result(
        self,
        output,
        source,
        expected_stride,
        *,
        case,
        requires_grad=False,
        is_leaf=True,
    ):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(output.shape, source.shape)
            self.assertEqual(output.stride(), expected_stride)
            self.assertEqual(output.storage_offset(), 0)
            self.assertEqual(output.requires_grad, requires_grad)
            self.assertEqual(output.is_leaf, is_leaf)
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
    def top_level_calls(source):
        return (
            ("torch.abs positional", lambda: torch.abs(source)),
            ("torch.abs input", lambda: torch.abs(input=source)),
            ("torch.abs x", lambda: torch.abs(x=source)),
            ("torch.abs a", lambda: torch.abs(a=source)),
            ("torch.abs x1", lambda: torch.abs(x1=source)),
            ("torch.abs out none", lambda: torch.abs(source, out=None)),
            ("torch.abs alias and out none", lambda: torch.abs(x=source, out=None)),
            ("torch.absolute positional", lambda: torch.absolute(source)),
            ("torch.absolute input", lambda: torch.absolute(input=source)),
            ("torch.absolute x", lambda: torch.absolute(x=source)),
            ("torch.absolute a", lambda: torch.absolute(a=source)),
            ("torch.absolute x1", lambda: torch.absolute(x1=source)),
            ("torch.absolute out none", lambda: torch.absolute(source, out=None)),
            (
                "torch.absolute alias and out none",
                lambda: torch.absolute(x=source, out=None),
            ),
        )

    @staticmethod
    def autograd_case(case):
        if case == "scalar":
            leaf = torch.tensor(-3.0, requires_grad=True)
            return leaf, leaf, None, ()
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return leaf, leaf.transpose(0, 2)[1], None, (2, 1)
        if case == "offset":
            leaf = torch.tensor(
                [[-2.0, -0.0, 1.0], [2.0, -4.0, 8.0]],
                requires_grad=True,
            )
            return leaf, leaf.transpose(0, 1)[1], None, (1,)
        if case == "noncontiguous":
            values = np.asarray(
                [[-3.0, -0.0, 2.0], [4.0, -5.0, 0.0]], dtype=np.float32
            )
            leaf = torch.tensor(values.tolist(), requires_grad=True)
            return leaf, leaf.transpose(0, 1), None, (1, 3)
        if case == "weighted edge":
            input_bits = np.asarray(
                (
                    0x0000_0000,
                    0x8000_0000,
                    0x0000_0001,
                    0x8000_0001,
                    0x0080_0000,
                    0x8080_0000,
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
            weight_bits = np.asarray(
                (
                    0x3F80_0000,
                    0xBF80_0000,
                    0x3F00_0000,
                    0x3F00_0000,
                    0x0000_0001,
                    0x0000_0001,
                    0x3F80_0000,
                    0xBF80_0000,
                    0x3E80_0000,
                    0x3E80_0000,
                    0x3F80_0000,
                    0xBF80_0000,
                    0x3F80_0000,
                    0xBF80_0000,
                    0x7FC0_1234,
                    0xFFC0_5678,
                ),
                dtype=np.uint32,
            )
            leaf = torch.tensor(
                memoryview(input_bits.view(np.float32)), requires_grad=True
            )
            weights = torch.tensor(memoryview(weight_bits.view(np.float32)))
            return leaf, leaf, weights, (1,)
        raise AssertionError(f"unknown abs autograd case: {case}")

    @staticmethod
    def expected_sign_zero_gradient(values):
        values = np.asarray(values, dtype=np.float32)
        return np.where(
            np.isnan(values) | (values == np.float32(0.0)),
            np.float32(0.0),
            np.where(values < np.float32(0.0), np.float32(-1.0), np.float32(1.0)),
        ).astype(np.float32)

    def expected_leaf_gradient(self, case, leaf):
        expected = np.zeros(tuple(leaf.shape), dtype=np.float32)
        if case == "scalar":
            return np.asarray(-1.0, dtype=np.float32)
        if case == "offset":
            expected[:, 1] = np.asarray([0.0, -1.0], dtype=np.float32)
        elif case == "noncontiguous":
            expected = self.expected_sign_zero_gradient(
                np.asarray(leaf, dtype=np.float32)
            )
        return expected

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

            expected = source.abs()
            for form, call in self.top_level_calls(source):
                output = call()
                self.assert_result(
                    output, source, expected_stride, case=(case, form)
                )
                np.testing.assert_array_equal(
                    self.tensor_bits(output), self.tensor_bits(expected)
                )

    def test_active_autograd_records_abs_backward_through_full_sum(self):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            leaf, source, _, expected_stride = self.autograd_case(case)
            with self.subTest(case=case):
                output = source.abs()
                self.assert_result(
                    output,
                    source,
                    expected_stride,
                    case=(case, "forward"),
                    requires_grad=True,
                    is_leaf=False,
                )
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
                    ", grad_fn=<AbsBackward0>",
                )
                loss = output if output.numel() == 1 else output.sum()
                loss.backward()

                expected = self.expected_leaf_gradient(case, leaf)
                self.assertEqual(leaf.grad.shape, leaf.shape)
                np.testing.assert_array_equal(
                    np.asarray(leaf.grad, dtype=np.float32), expected
                )

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        for form, call in (*self.supported_calls(extreme), *self.top_level_calls(extreme)):
            with self.subTest(form=form, mode="extreme recording"):
                with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                    call()

    def test_abs_backward_sign_zero_edges_accumulation_and_higher_order_boundary(self):
        leaf, source, weights, _ = self.autograd_case("weighted edge")
        output = source.abs()
        self.assert_result(
            output,
            source,
            (1,),
            case="weighted edge forward",
            requires_grad=True,
            is_leaf=False,
        )
        (output * weights).sum().backward()
        expected_gradient_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x3F00_0000,
                0xBF00_0000,
                0x0000_0001,
                0x8000_0001,
                0x3F80_0000,
                0x3F80_0000,
                0x3E80_0000,
                0xBE80_0000,
                0x3F80_0000,
                0x3F80_0000,
                0x0000_0000,
                0x8000_0000,
                0x7FC0_1234,
                0xFFC0_5678,
            ),
            dtype=np.uint32,
        )
        np.testing.assert_array_equal(
            np.asarray(leaf.grad, dtype=np.float32).view(np.uint32),
            expected_gradient_bits,
        )

        accumulated = torch.tensor([-2.0, -0.0, 3.0], requires_grad=True)
        accumulated.abs().sum().backward()
        accumulated.abs().sum().backward()
        np.testing.assert_array_equal(
            np.asarray(accumulated.grad, dtype=np.float32),
            np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
        )

        freed = torch.tensor([-2.0, 3.0], requires_grad=True)
        loss = freed.abs().sum()
        loss.backward()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

        higher_order = torch.tensor(-2.0, requires_grad=True)
        higher_order_loss = higher_order.abs()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertEqual(higher_order.grad.item(), -1.0)

    def test_top_level_and_operator_autograd_reuse_method_path(self):
        forms = tuple(
            (form, call)
            for form, call in (
                *self.supported_calls(torch.tensor(1.0)),
                *self.top_level_calls(torch.tensor(1.0)),
            )
        )
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form, _ in forms:
                leaf, source, _, expected_stride = self.autograd_case(case)
                call = dict(
                    (
                        *self.supported_calls(source),
                        *self.top_level_calls(source),
                    )
                )[form]
                output = call()
                self.assert_result(
                    output,
                    source,
                    expected_stride,
                    case=(case, form, "forward"),
                    requires_grad=True,
                    is_leaf=False,
                )
                loss = output if output.numel() == 1 else output.sum()
                loss.backward()
                self.assertEqual(leaf.grad.shape, leaf.shape)
                np.testing.assert_array_equal(
                    np.asarray(leaf.grad, dtype=np.float32),
                    self.expected_leaf_gradient(case, leaf),
                )

        leaf = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
        for name in ("abs", "absolute"):
            output = getattr(torch, name)(leaf, out=None)
            output.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad, dtype=np.float32),
            np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
        )

    def test_detached_and_no_grad_inputs_use_the_inference_path(self):
        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, -4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]
        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        for form, call in (*self.supported_calls(source), *self.top_level_calls(source)):
            with torch.no_grad():
                output = call()
            self.assert_result(output, source, (1,), case=(form, "no_grad"))

        for form, call in (*self.supported_calls(extreme), *self.top_level_calls(extreme)):
            with self.subTest(form=form, mode="extreme no_grad"):
                with torch.no_grad():
                    with self.assertRaisesRegex(
                        RuntimeError, "Stride calculation overflowed"
                    ):
                        call()

        detached = source.detach()
        for form, call in (*self.supported_calls(detached), *self.top_level_calls(detached)):
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
                tracked_forward = torch.tensor([-4.0], requires_grad=True)
                with ForwardingMode("lower", order):
                    with ForwardingMode("upper", order):
                        forwarded_tracked = call(tracked_forward)
                self.assertEqual(order, ["upper", "lower"])
                self.assertTrue(forwarded_tracked.requires_grad)
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(
                        forwarded_tracked
                    ),
                    ", grad_fn=<AbsBackward0>",
                )
                forwarded_tracked.sum().backward()
                self.assertEqual(tracked_forward.grad.tolist(), [-1.0])

            invalid_mode = RecordingMode()
            with self.subTest(form=form, mode="invalid"):
                with self.assertRaises(TypeError):
                    with invalid_mode:
                        invalid_call(plain)
                self.assertEqual(invalid_mode.calls, [])

    def test_top_level_concrete_out_tensor_is_rejected_without_mutation(self):
        source = torch.tensor([-4.0, 0.0], requires_grad=True)
        for name in ("abs", "absolute"):
            for form in ("positional", "keyword", "alias"):
                destination = torch.tensor([17.0, 19.0])
                with self.subTest(name=name, form=form):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"^{name}\(\): the 'out' argument is not supported$",
                    ):
                        if form == "positional":
                            getattr(torch, name)(source, out=destination)
                        elif form == "keyword":
                            getattr(torch, name)(input=source, out=destination)
                        else:
                            getattr(torch, name)(x=source, out=destination)
                    self.assertEqual(destination.tolist(), [17.0, 19.0])

            detached = source.detach()
            self.assert_result(
                getattr(torch, name)(detached, out=None),
                detached,
                (1,),
                case=(name, "explicit out none"),
            )

    def test_top_level_torch_function_modes_and_overrides(self):
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

        for name in ("abs", "absolute"):
            function = getattr(torch, name)

            mode = RecordingMode()
            with mode:
                self.assertIs(function(input=tracked, out=destination), marker)
            with self.subTest(name=name, mode="recording"):
                self.assertEqual(len(mode.calls), 1)
                dispatched, dispatch_types, args, kwargs = mode.calls[0]
                self.assertIs(dispatched, function)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, ())
                self.assertEqual(kwargs, {"input": tracked, "out": destination})

            override_calls = []

            class Override:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    override_calls.append((func, types, args, kwargs))
                    return marker

            with self.subTest(name=name, mode="override"):
                self.assertIs(function(Override()), marker)
                self.assertIs(function(plain, out=Override()), marker)
                self.assertEqual(len(override_calls), 2)
                for dispatched, dispatch_types, _, _ in override_calls:
                    self.assertIs(dispatched, function)
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

            with self.subTest(name=name, mode="subclass precedence"):
                self.assertIs(function(BaseOverride(), out=DerivedOverride()), marker)
                self.assertEqual(subclass_order, ["derived"])

            forwarding_order = []
            with ForwardingMode("lower", forwarding_order):
                with ForwardingMode("upper", forwarding_order):
                    forwarded = function(input=plain, out=None)
            with self.subTest(name=name, mode="forwarding"):
                self.assertEqual(forwarding_order, ["upper", "lower"])
                self.assertEqual(forwarded.tolist(), [4.0])

            forwarding_order.clear()
            with self.subTest(name=name, mode="forwarding tracked"):
                tracked_forward = torch.tensor([-4.0], requires_grad=True)
                with ForwardingMode("lower", forwarding_order):
                    with ForwardingMode("upper", forwarding_order):
                        forwarded_tracked = function(input=tracked_forward, out=None)
                self.assertEqual(forwarding_order, ["upper", "lower"])
                self.assertTrue(forwarded_tracked.requires_grad)
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(
                        forwarded_tracked
                    ),
                    ", grad_fn=<AbsBackward0>",
                )
                forwarded_tracked.sum().backward()
                self.assertEqual(tracked_forward.grad.tolist(), [-1.0])

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

            with self.subTest(name=name, mode="fallback override"):
                with DecliningMode():
                    self.assertIs(function(FallbackOverride()), marker)
                self.assertEqual(events, ["mode", "override"])

            invalid_mode = RecordingMode()
            with self.subTest(name=name, mode="invalid"):
                with self.assertRaises(TypeError):
                    with invalid_mode:
                        function()
                self.assertEqual(invalid_mode.calls, [])

    def test_top_level_callable_metadata_documentation_pickling_and_exports(self):
        self.assertIsNot(torch.abs, torch.absolute)
        for name, doc in (
            ("abs", TOP_LEVEL_ABS_DOC),
            ("absolute", TOP_LEVEL_ABSOLUTE_DOC),
        ):
            function = getattr(torch, name)
            with self.subTest(name=name, contract=True):
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
                    lambda: setattr(owner, name, None),
                    lambda: delattr(owner, name),
                ):
                    with self.assertRaises(TypeError):
                        action()
                    self.assertIs(getattr(owner, name), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )

                self.assertEqual(torch.__all__.count(name), 1)
                self.assertNotIn("_VariableFunctionsClass", torch.__all__)
                self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
                wildcard_namespace = {}
                exec("from torch_rs import *", wildcard_namespace)
                self.assertIs(wildcard_namespace[name], function)

    def test_top_level_binding_type_and_unsupported_extension_errors(self):
        tensor = torch.tensor([-4.0])
        for name in ("abs", "absolute"):
            function = getattr(torch, name)
            cases = (
                (
                    lambda: function(),
                    f'{name}() missing 1 required positional arguments: "input"',
                ),
                (
                    lambda: function(tensor, tensor),
                    f"{name}() takes 1 positional argument but 2 were given",
                ),
                (
                    lambda: function(tensor, input=tensor),
                    f"{name}() got multiple values for argument 'input'",
                ),
                (
                    lambda: function(out=tensor),
                    f'{name}() missing 1 required positional arguments: "input"',
                ),
                (
                    lambda: function(extra=tensor),
                    f'{name}() missing 1 required positional arguments: "input"',
                ),
                (
                    lambda: function(1, extra=True),
                    f"{name}(): argument 'input' (position 1) must be Tensor, not int",
                ),
                (
                    lambda: function(input=[]),
                    f"{name}(): argument 'input' must be Tensor, not list",
                ),
                (
                    lambda: function(tensor, out=[]),
                    f"{name}(): argument 'out' must be Tensor, not list",
                ),
                (
                    lambda: function(tensor, extra=True, out=[]),
                    f"{name}(): argument 'out' must be Tensor, not list",
                ),
                (
                    lambda: function(tensor, extra=True),
                    f"{name}() got an unexpected keyword argument 'extra'",
                ),
                (
                    lambda: function(input=tensor, a=tensor),
                    f"{name}() got an unexpected keyword argument 'a'",
                ),
                (
                    lambda: function(a=tensor, x=tensor, out=None),
                    f"{name}() got an unexpected keyword argument 'a'",
                ),
                (
                    lambda: function(x=tensor, a=tensor, out=None),
                    f"{name}() got an unexpected keyword argument 'x'",
                ),
                (
                    lambda: function(tensor, dtype=torch.float32),
                    f"{name}() got an unexpected keyword argument 'dtype'",
                ),
                (
                    lambda: function(tensor, device=torch.device("cpu")),
                    f"{name}() got an unexpected keyword argument 'device'",
                ),
                (
                    lambda: function(np.zeros((2, 3), dtype=np.float32)),
                    f"{name}(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
                ),
            )
            for call, message in cases:
                with self.subTest(name=name, message=message):
                    with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                        call()

    def test_top_level_and_inplace_forms_boundaries(self):
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
        with self.assertRaisesRegex(
            TypeError, r"^type 'torch_rs\.Tensor' is not an acceptable base type$"
        ):
            type("TensorSubclass", (torch.Tensor,), {})
        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([-4.0], device="cuda")


if __name__ == "__main__":
    unittest.main()
