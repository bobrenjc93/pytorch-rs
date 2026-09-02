import copy
import importlib
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch


DIV_DOC = "\ndiv(value, *, rounding_mode=None) -> Tensor\n\nSee :func:`torch.div`\n"
DIVIDE_DOC = (
    "\ndivide(value, *, rounding_mode=None) -> Tensor\n\nSee :func:`torch.divide`\n"
)


class TensorDivisionMethodTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def test_methods_reuse_true_division_values_layouts_and_edge_cases(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [-0.0], [float("inf")]])
        expected = left / right
        for name in ("div", "divide"):
            calls = (
                ("positional tensors", lambda name=name: getattr(left, name)(right)),
                ("other keyword", lambda name=name: getattr(left, name)(other=right)),
                ("x2 keyword", lambda name=name: getattr(left, name)(x2=right)),
                (
                    "explicit true division",
                    lambda name=name: getattr(left, name)(right, rounding_mode=None),
                ),
            )
            for case, call in calls:
                self.assert_tensor_matches(call(), expected, case=(name, case))

        offset_view = left[1]
        for scalar in (True, -2, 2.5, np.bool_(True), np.int64(3), np.float32(-0.0)):
            for name in ("div", "divide"):
                self.assert_tensor_matches(
                    getattr(offset_view, name)(scalar),
                    offset_view / scalar,
                    case=(name, "offset scalar", type(scalar).__name__, scalar),
                )
                self.assert_tensor_matches(
                    getattr(offset_view, name)(other=scalar),
                    offset_view / scalar,
                    case=(name, "keyword scalar", type(scalar).__name__, scalar),
                )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        for name in ("div", "divide"):
            self.assert_tensor_matches(
                getattr(empty, name)(broadcast),
                empty / broadcast,
                case=(name, "strided broadcast empty"),
            )

        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
            ),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        divisors = torch.tensor([1.0, -1.0, 0.0, -0.0, float("inf"), float("-inf"), 2.0])
        for name in ("div", "divide"):
            result = getattr(special, name)(divisors)
            self.assert_tensor_matches(
                result,
                special / divisors,
                case=(name, "signed zero nan infinity"),
            )
            self.assertFalse(result.is_set_to(special))
            self.assertFalse(result.is_set_to(divisors))
            if result.numel():
                self.assertNotEqual(result.data_ptr(), special.data_ptr())
                self.assertNotEqual(result.data_ptr(), divisors.data_ptr())

    def test_rounding_modes_reuse_division_and_unary_rounding_edges(self):
        rounding_modes = ("trunc", "floor")

        finite_left = torch.tensor(
            [[-3.75], [-2.5], [-1.25], [-0.0], [0.0], [1.25], [2.5], [3.75]]
        )
        finite_right = torch.tensor([[2.0, -2.0]])
        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        empty_other = torch.tensor([[[2.0, -2.0]]])
        offset_noncontiguous = torch.tensor(
            [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]
        ).transpose(0, 1)[1]
        self.assertGreater(offset_noncontiguous.storage_offset(), 0)
        self.assertFalse(offset_noncontiguous.is_contiguous())

        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0xBF80_0000,
                0x3F80_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
            ),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        divisors = torch.tensor(
            [1.0, -1.0, float("inf"), float("-inf"), 2.0, 2.0, 2.0]
        )
        expected_special_bits = {
            "trunc": np.asarray(
                (
                    0x0000_0000,
                    0x0000_0000,
                    0x8000_0000,
                    0x8000_0000,
                    0x7F80_0000,
                    0xFF80_0000,
                    0x7FC1_2345,
                ),
                dtype=np.uint32,
            ),
            "floor": np.asarray(
                (
                    0x0000_0000,
                    0x0000_0000,
                    0xBF80_0000,
                    0xBF80_0000,
                    0xFFC0_0000,
                    0xFFC0_0000,
                    0x7FC1_2345,
                ),
                dtype=np.uint32,
            ),
        }

        for name in ("div", "divide"):
            for mode in rounding_modes:
                round_expected = getattr(finite_left / finite_right, mode)()
                self.assert_tensor_matches(
                    getattr(finite_left, name)(finite_right, rounding_mode=mode),
                    round_expected,
                    case=(name, mode, "negative finite broadcast"),
                )

                self.assert_tensor_matches(
                    getattr(empty, name)(empty_other, rounding_mode=mode),
                    empty / empty_other,
                    case=(name, mode, "empty broadcast"),
                )

                self.assert_tensor_matches(
                    getattr(offset_noncontiguous, name)(
                        torch.tensor([2.0, -2.0]), rounding_mode=mode
                    ),
                    getattr(offset_noncontiguous / torch.tensor([2.0, -2.0]), mode)(),
                    case=(name, mode, "offset noncontiguous"),
                )

                for scalar in (-2.0, np.float32(-0.0)):
                    self.assert_tensor_matches(
                        getattr(offset_noncontiguous, name)(scalar, rounding_mode=mode),
                        getattr(offset_noncontiguous / scalar, mode)(),
                        case=(name, mode, "scalar", type(scalar).__name__, scalar),
                    )

                result = getattr(special, name)(divisors, rounding_mode=mode)
                self.assert_tensor_matches(
                    result,
                    torch.tensor(memoryview(expected_special_bits[mode].view(np.float32))),
                    case=(name, mode, "signed zero nan infinity"),
                )
                self.assertFalse(result.is_set_to(special))
                self.assertFalse(result.is_set_to(divisors))
                if result.numel():
                    self.assertNotEqual(result.data_ptr(), special.data_ptr())
                    self.assertNotEqual(result.data_ptr(), divisors.data_ptr())

    def test_active_autograd_is_rejected_but_no_grad_uses_native_division(self):
        for name in ("div", "divide"):
            left = torch.tensor([[2.0, 3.0]], requires_grad=True)
            right = torch.tensor([[5.0], [7.0]], requires_grad=True)
            with self.subTest(name=name, case="tensor operands"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): autograd recording is not supported$",
                ):
                    getattr(left.transpose(0, 1), name)(right.transpose(0, 1))
                self.assertIsNone(left.grad)
                self.assertIsNone(right.grad)

            with self.subTest(name=name, case="scalar operand"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): autograd recording is not supported$",
                ):
                    getattr(left, name)(2.0)

            for mode in ("trunc", "floor"):
                with self.subTest(name=name, case="tensor operands", rounding_mode=mode):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"^{name}\(\): autograd recording is not supported$",
                    ):
                        getattr(left.transpose(0, 1), name)(
                            right.transpose(0, 1), rounding_mode=mode
                        )
                with self.subTest(name=name, case="scalar operand", rounding_mode=mode):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"^{name}\(\): autograd recording is not supported$",
                    ):
                        getattr(left, name)(2.0, rounding_mode=mode)

            with self.subTest(name=name, case="no_grad"):
                with torch.no_grad():
                    tensor_output = getattr(left.transpose(0, 1), name)(
                        right.transpose(0, 1)
                    )
                    scalar_output = getattr(left, name)(2.0)
                    rounded_tensor_output = getattr(left.transpose(0, 1), name)(
                        right.transpose(0, 1), rounding_mode="floor"
                    )
                    rounded_scalar_output = getattr(left, name)(
                        2.0, rounding_mode="trunc"
                    )
                    expected_tensor_output = left.transpose(0, 1) / right.transpose(
                        0, 1
                    )
                    expected_rounded_tensor = expected_tensor_output.floor()
                    expected_rounded_scalar = (left / 2.0).trunc()
                self.assertFalse(tensor_output.requires_grad)
                self.assertTrue(tensor_output.is_leaf)
                self.assert_tensor_matches(
                    tensor_output,
                    expected_tensor_output,
                    case=(name, "no_grad tensor"),
                )
                self.assertFalse(scalar_output.requires_grad)
                self.assertTrue(scalar_output.is_leaf)
                self.assertFalse(rounded_tensor_output.requires_grad)
                self.assertTrue(rounded_tensor_output.is_leaf)
                self.assert_tensor_matches(
                    rounded_tensor_output,
                    expected_rounded_tensor,
                    case=(name, "no_grad rounded tensor"),
                )
                self.assertFalse(rounded_scalar_output.requires_grad)
                self.assertTrue(rounded_scalar_output.is_leaf)
                self.assert_tensor_matches(
                    rounded_scalar_output,
                    expected_rounded_scalar,
                    case=(name, "no_grad rounded scalar"),
                )

        active = torch.tensor([2.0], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, r"^div\(\): autograd recording is not supported$"
        ):
            active / torch.tensor([1.0])
        with self.assertRaisesRegex(
            RuntimeError, r"^div\(\): autograd recording is not supported$"
        ):
            2.0 / active

    def test_modes_and_overrides_observe_valid_calls_before_native_limits(self):
        left = torch.tensor([2.0])
        right = torch.tensor([4.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for name in ("div", "divide"):
            descriptor = inspect.getattr_static(torch.Tensor, name)
            calls = (
                (lambda name=name: getattr(left, name)(right), None),
                (lambda name=name: getattr(left, name)(2.0), None),
                (lambda name=name: getattr(left, name)(other=right), ("other",)),
                (lambda name=name: getattr(left, name)(x2=right), ("x2",)),
                (
                    lambda name=name: getattr(left, name)(right, rounding_mode="floor"),
                    ("rounding_mode",),
                ),
            )
            for call, expected_keywords in calls:
                mode = RecordingMode()
                with mode:
                    self.assertIs(call(), marker)
                function, dispatch_types, args, kwargs = mode.calls[0]
                with self.subTest(name=name, keywords=expected_keywords):
                    self.assertIs(function, descriptor)
                    self.assertEqual(dispatch_types, ())
                    self.assertIs(args[0], left)
                    if expected_keywords is None:
                        self.assertIsNone(kwargs)
                    else:
                        self.assertEqual(tuple(kwargs), expected_keywords)

            invalid_mode = RecordingMode()
            with invalid_mode:
                with self.assertRaises(TypeError):
                    getattr(left, name)([])
            self.assertEqual(invalid_mode.calls, [])

            order = []

            class ForwardingMode(torch.overrides.TorchFunctionMode):
                def __init__(self, label):
                    self.label = label

                def __torch_function__(self, func, types, args=(), kwargs=None):
                    order.append(self.label)
                    return func(*args, **(kwargs or {}))

            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    forwarded = getattr(left, name)(other=right)
            self.assertEqual(order, ["upper", "lower"])
            self.assert_tensor_matches(forwarded, left / right, case=(name, "forwarded"))

            events = []

            class Override:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append((func, types, args, kwargs))
                    return marker

            self.assertIs(getattr(left, name)(Override()), marker)
            function, dispatch_types, args, kwargs = events[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, (Override,))
            self.assertIs(args[0], left)
            self.assertIsInstance(args[1], Override)
            self.assertIsNone(kwargs)

            events.clear()
            self.assertIs(
                getattr(left, name)(right, rounding_mode=Override()), marker
            )
            _, dispatch_types, args, kwargs = events[0]
            self.assertEqual(dispatch_types, (Override,))
            self.assertIs(args[0], left)
            self.assertIs(args[1], right)
            self.assertEqual(tuple(kwargs), ("rounding_mode",))

            class DecliningOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    return NotImplemented

            with self.assertRaises(TypeError) as raised:
                getattr(left, name)(DecliningOverride())
            self.assertIn(
                f"Multiple dispatch failed for 'torch.Tensor.{name}'",
                str(raised.exception),
            )

    def test_callable_metadata_copy_pickle_reload_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        descriptors = {
            "div": inspect.getattr_static(torch.Tensor, "div"),
            "divide": inspect.getattr_static(torch.Tensor, "divide"),
        }
        self.assertIsNot(descriptors["div"], descriptors["divide"])

        for name, doc in (("div", DIV_DOC), ("divide", DIVIDE_DOC)):
            descriptor = descriptors[name]
            bound = getattr(tensor, name)
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
                self.assertNotIn(name, torch.Tensor.__dict__)

            for callable_object in (descriptor, bound):
                with self.subTest(name=name, callable=type(callable_object).__name__):
                    self.assertIsNone(callable_object.__text_signature__)
                    with self.assertRaises(ValueError):
                        inspect.signature(callable_object)
                    self.assertIs(copy.copy(callable_object), callable_object)
                    self.assertIs(copy.deepcopy(callable_object), callable_object)

            self.assert_tensor_matches(
                descriptor(tensor, other=tensor),
                tensor / tensor,
                case=(name, "unbound call"),
            )
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(descriptor, protocol)), descriptor
                    )

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        for name, descriptor in descriptors.items():
            self.assertIs(inspect.getattr_static(torch.Tensor, name), descriptor)

        for name in ("div_", "divide_"):
            self.assertFalse(hasattr(torch.Tensor, name))
            self.assertFalse(hasattr(torch, name))

    def test_unsupported_arguments_and_boundaries_do_not_mutate(self):
        tensor = torch.tensor([4.0])
        destination = torch.tensor([17.0])

        for name in ("div", "divide"):
            method = getattr(tensor, name)
            with self.subTest(name=name, boundary="rounding bad string"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^div expected rounding_mode to be one of None, 'trunc', "
                    r"or 'floor' but found 'bad'$",
                ):
                    method(2, rounding_mode="bad")
            with self.subTest(name=name, boundary="out"):
                with self.assertRaises(TypeError):
                    method(tensor, out=destination)
                self.assertEqual(destination.tolist(), [17.0])
            with self.subTest(name=name, boundary="dtype"):
                with self.assertRaises(TypeError):
                    method(tensor, dtype=torch.float32)
            with self.subTest(name=name, boundary="device"):
                with self.assertRaises(TypeError):
                    method(tensor, device=torch.device("cpu"))
            with self.subTest(name=name, boundary="unsupported operand"):
                with self.assertRaises(TypeError):
                    method([])
            with self.subTest(name=name, boundary="unsupported rounding type"):
                with self.assertRaises(TypeError):
                    method(tensor, rounding_mode=1)

        with self.assertRaisesRegex(
            TypeError,
            r"^div\(\) received an invalid combination of arguments - got "
            r"\(\), but expected one of:",
        ):
            tensor.div()
        with self.assertRaisesRegex(
            TypeError,
            r"^divide\(\) received an invalid combination of arguments - got "
            r"\(\), but expected one of:",
        ):
            tensor.divide()


if __name__ == "__main__":
    unittest.main()
