import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


class IntSubclass(int):
    pass


class IndexDimension:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


class StatefulIndexDimension:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def __index__(self):
        value = next(self.values)
        self.calls.append(value)
        return value


class NewZerosTests(unittest.TestCase):
    def source(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        return leaf, (leaf * 2.0).transpose(0, 1)

    def assert_zero_tensor(self, tensor, shape, stride, requires_grad=False):
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.stride(), stride)
        self.assertEqual(
            np.asarray(tensor.detach()).reshape(-1).tolist(),
            [0.0] * tensor.numel(),
        )
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertEqual(tensor.requires_grad, requires_grad)
        self.assertTrue(tensor.is_leaf)
        self.assertTrue(tensor.is_contiguous())
        self.assertEqual(tensor.storage_offset(), 0)

    def test_integer_and_sequence_sizes_create_fresh_contiguous_storage(self):
        _, source = self.source()
        cases = (
            (2, (2,), (1,)),
            ((2, 3), (2, 3), (3, 1)),
            ([2, 0, 3], (2, 0, 3), (3, 3, 1)),
            (torch.Size([2, 1, 3]), (2, 1, 3), (3, 3, 1)),
            ((), (), ()),
        )
        for size, shape, stride in cases:
            with self.subTest(size=size):
                result = source.new_zeros(size)
                self.assert_zero_tensor(result, shape, stride)
                self.assertFalse(result.is_set_to(source))

        first = source.new_zeros((2, 3))
        second = source.new_zeros((2, 3))
        self.assertNotEqual(first.data_ptr(), source.data_ptr())
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(source.tolist(), [[2.0, 8.0], [4.0, 10.0], [6.0, 12.0]])

    def test_supported_options_and_autograd_leaf_state(self):
        leaf, source = self.source()
        cases = (
            {},
            {
                "dtype": None,
                "device": None,
                "requires_grad": None,
                "layout": None,
                "pin_memory": None,
            },
            {
                "dtype": torch.float32,
                "device": "cpu",
                "requires_grad": False,
                "layout": torch.strided,
                "pin_memory": False,
            },
            {"device": torch.device("cpu:0")},
        )
        for options in cases:
            with self.subTest(options=options):
                self.assert_zero_tensor(
                    source.new_zeros((2, 3), **options), (2, 3), (3, 1)
                )

        with torch.no_grad():
            result = source.new_zeros((2, 3), requires_grad=True)
        self.assert_zero_tensor(result, (2, 3), (3, 1), requires_grad=True)
        result.sum().backward()
        self.assertEqual(result.grad.tolist(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        self.assertIsNone(leaf.grad)

    def test_size_uses_pytorch_index_conversion(self):
        source = torch.tensor([1.0])
        for size in (IntSubclass(2), np.int64(2), np.uint32(2)):
            with self.subTest(size=size):
                self.assertEqual(source.new_zeros(size).shape, (2,))

        scalar = StatefulIndexDimension((2, 3, 4))
        self.assertEqual(source.new_zeros(scalar).shape, (4,))
        self.assertEqual(scalar.calls, [2, 3, 4])

        sequence = StatefulIndexDimension((2, 3))
        self.assertEqual(source.new_zeros([sequence]).shape, (3,))
        self.assertEqual(sequence.calls, [2, 3])

        later = StatefulIndexDimension((2, 3))
        self.assertEqual(source.new_zeros([1, later]).shape, (1, 2))
        self.assertEqual(later.calls, [2])
        self.assertEqual(source.new_zeros([1, True]).shape, (1, 1))

    def test_size_binding_and_conversion_errors(self):
        source = torch.tensor([1.0])
        exact_cases = (
            (
                lambda: source.new_zeros(),
                TypeError,
                'new_zeros() missing 1 required positional arguments: "size"',
            ),
            (
                lambda: source.new_zeros(None),
                TypeError,
                "new_zeros(): argument 'size' (position 1) must be tuple of ints, not NoneType",
            ),
            (
                lambda: source.new_zeros(size=2),
                TypeError,
                "new_zeros(): argument 'size' must be tuple of ints, not int",
            ),
            (
                lambda: source.new_zeros(range(2)),
                TypeError,
                "new_zeros(): argument 'size' (position 1) must be tuple of ints, not range",
            ),
            (
                lambda: source.new_zeros(True),
                TypeError,
                "new_zeros(): argument 'size' (position 1) must be tuple of ints, not bool",
            ),
            (
                lambda: source.new_zeros([True]),
                TypeError,
                "new_zeros(): argument 'size' (position 1) must be tuple of ints, but found element of type bool at pos 0",
            ),
            (
                lambda: source.new_zeros(size=[2.0]),
                TypeError,
                "new_zeros(): argument 'size' must be tuple of ints, not list",
            ),
            (
                lambda: source.new_zeros((2, -1, 3)),
                RuntimeError,
                "Trying to create tensor with negative dimension -1: [2, -1, 3]",
            ),
            (
                lambda: source.new_zeros(sys.maxsize),
                RuntimeError,
                f"Storage size calculation overflowed with sizes=[{sys.maxsize}]",
            ),
            (
                lambda: source.new_zeros((0, sys.maxsize, 2)),
                RuntimeError,
                "Stride calculation overflowed",
            ),
        )
        for call, error_type, message in exact_cases:
            with self.subTest(message=message), self.assertRaises(error_type) as raised:
                call()
            self.assertEqual(str(raised.exception), message)

        for size, position in ((2**63, 1), ((2, 2**63), 2)):
            with self.subTest(size=size), self.assertRaises(TypeError) as raised:
                source.new_zeros(size)
            self.assertRegex(
                str(raised.exception),
                re.compile(
                    rf"^new_zeros\(\): argument 'size' failed to unpack the object "
                    rf'at pos {position} with error "Overflow when unpacking long long"$'
                ),
            )

    def test_variadic_sizes_and_unsupported_options_are_rejected(self):
        source = torch.tensor([1.0])
        with self.assertRaisesRegex(
            TypeError, r"^new_zeros\(\) takes 1 positional argument but 2 were given$"
        ):
            source.new_zeros(2, 3)

        cases = (
            (
                lambda: source.new_zeros((2,), dtype=object()),
                TypeError,
                "new_zeros(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: source.new_zeros((2,), device=object()),
                TypeError,
                "new_zeros(): argument 'device' must be torch.device, not object",
            ),
            (
                lambda: source.new_zeros((2,), requires_grad=1),
                TypeError,
                "new_zeros(): argument 'requires_grad' must be bool, not int",
            ),
            (
                lambda: source.new_zeros((2,), layout=object()),
                TypeError,
                "new_zeros(): argument 'layout' must be torch.layout, not object",
            ),
            (
                lambda: source.new_zeros((2,), pin_memory=1),
                TypeError,
                "new_zeros(): argument 'pin_memory' must be bool, not int",
            ),
            (
                lambda: source.new_zeros((2,), device="cuda"),
                RuntimeError,
                "new_zeros(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: source.new_zeros((2,), pin_memory=True),
                RuntimeError,
                "new_zeros(): pin_memory=True is not supported; only unpinned CPU storage is implemented",
            ),
            (
                lambda: source.new_zeros((2,), unexpected=True),
                TypeError,
                "new_zeros() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: source.new_zeros((2,), size=(3,)),
                TypeError,
                "new_zeros() got multiple values for argument 'size'",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message), self.assertRaises(error_type) as raised:
                call()
            self.assertEqual(str(raised.exception), message)

        for name in ("new_empty", "new_ones", "new_full", "new_tensor"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.Tensor, name))
                self.assertFalse(hasattr(source, name))

        with self.assertRaisesRegex(
            RuntimeError,
            rf"^Storage size calculation overflowed with sizes=\[{sys.maxsize}\]$",
        ):
            source.new_zeros(sys.maxsize, pin_memory=True)

    def test_tensorbase_descriptor_matches_the_public_method_shape(self):
        source = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "new_zeros")
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertEqual(descriptor.__name__, "new_zeros")
        self.assertEqual(descriptor.__qualname__, "TensorBase.new_zeros")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(
            repr(descriptor), "<method 'new_zeros' of 'torch._C.TensorBase' objects>"
        )
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)
        self.assertEqual(descriptor(source, (2,)).tolist(), [0.0, 0.0])

    def test_argument_torch_function_overrides_dispatch_in_precedence_order(self):
        source = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "new_zeros")
        marker = object()
        calls = []
        index_calls = []

        class IndexOverride:
            def __index__(self):
                index_calls.append("index")
                return 2

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append(("index", func, types, args, kwargs))
                return marker

        dimension = IndexOverride()
        size = [1, dimension]
        self.assertIs(source.new_zeros(size), marker)
        self.assertEqual(index_calls, [])
        label, function, dispatch_types, args, kwargs = calls.pop()
        self.assertEqual(label, "index")
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (IndexOverride,))
        self.assertIs(args[0], source)
        self.assertIs(args[1], size)
        self.assertIsNone(kwargs)

        class OptionOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append(("option", func, types, args, kwargs))
                return marker

        for option in ("dtype", "layout", "device", "pin_memory", "requires_grad"):
            value = OptionOverride()
            with self.subTest(option=option):
                self.assertIs(source.new_zeros((2,), **{option: value}), marker)
                label, function, dispatch_types, args, kwargs = calls.pop()
                self.assertEqual(label, "option")
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, (OptionOverride,))
                self.assertEqual(args, (source, (2,)))
                self.assertIs(kwargs[option], value)

        order = []

        class ParentOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("parent", types))
                return marker

        class ChildOverride(ParentOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("child", types))
                return NotImplemented

        self.assertIs(
            source.new_zeros([ParentOverride(), ChildOverride()]), marker
        )
        self.assertEqual(
            order,
            [
                ("child", (ChildOverride, ParentOverride)),
                ("parent", (ChildOverride, ParentOverride)),
            ],
        )

    def test_argument_overrides_follow_active_torch_function_modes(self):
        source = torch.tensor([1.0])
        mode_marker = object()
        override_marker = object()
        order = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("override", types))
                return override_marker

        class Mode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(("mode", types))
                return self.result

        value = Override()
        with Mode(mode_marker):
            self.assertIs(source.new_zeros([1, value]), mode_marker)
        self.assertEqual(order, [("mode", (Override,))])

        order.clear()
        with Mode(NotImplemented):
            self.assertIs(source.new_zeros([1, value]), override_marker)
        self.assertEqual(
            order,
            [("mode", (Override,)), ("override", (Override,))],
        )

        class DecliningOverride:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.Tensor\.new_zeros'; all ",
        ):
            source.new_zeros([1, DecliningOverride()])
        self.assertEqual(DecliningOverride.calls, 1)

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        source = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "new_zeros")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        positional = RecordingMode(marker)
        with positional:
            result = source.new_zeros((2, 3))
        self.assertIs(result, marker)
        function, dispatch_types, args, kwargs = positional.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (source, (2, 3)))
        self.assertIsNone(kwargs)

        keyword = RecordingMode(marker)
        with keyword:
            result = source.new_zeros(
                size=[2, 3], requires_grad=True, layout=torch.strided
            )
        self.assertIs(result, marker)
        function, dispatch_types, args, kwargs = keyword.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (source,))
        self.assertEqual(
            kwargs,
            {"size": [2, 3], "requires_grad": True, "layout": torch.strided},
        )

        deferred_index = StatefulIndexDimension((2, 3, 4))
        deferred = RecordingMode(marker)
        with deferred:
            result = source.new_zeros(deferred_index, device="not-a-device")
        self.assertIs(result, marker)
        self.assertEqual(deferred_index.calls, [2, 3])
        self.assertIs(deferred.calls[0][2][1], deferred_index)

        invalid = RecordingMode(marker)
        with invalid, self.assertRaises(TypeError):
            source.new_zeros(object())
        self.assertEqual(invalid.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = source.new_zeros((2, 0, 3), requires_grad=True)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_zero_tensor(
            forwarded, (2, 0, 3), (3, 3, 1), requires_grad=True
        )

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.Tensor\.new_zeros'; all ",
        ):
            with lower:
                with declining:
                    source.new_zeros((2,))
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])


if __name__ == "__main__":
    unittest.main()
