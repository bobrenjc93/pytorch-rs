import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


DIV_DOC = "\ndiv(value, *, rounding_mode=None) -> Tensor\n\nSee :func:`torch.div`\n"
DIVIDE_DOC = (
    "\ndivide(value, *, rounding_mode=None) -> Tensor\n\nSee :func:`torch.divide`\n"
)


class TensorDivMethodTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case, source=None):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            if source is not None:
                self.assertFalse(actual.is_set_to(source))
                if actual.numel() and source.numel():
                    self.assertNotEqual(actual.data_ptr(), source.data_ptr())
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def test_methods_reuse_true_division_values_layouts_and_edge_cases(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [4.0], [0.5]])
        for name in ("div", "divide"):
            method = getattr(left, name)
            calls = (
                ("positional tensors", lambda: method(right)),
                ("other keyword", lambda: method(other=right)),
                ("x2 keyword", lambda: method(x2=right)),
                ("rounding mode none", lambda: method(right, rounding_mode=None)),
            )
            for case, call in calls:
                self.assert_tensor_matches(
                    call(), left / right, case=(name, case), source=left
                )

        offset = left[1]
        for scalar in (
            True,
            False,
            -2,
            2.5,
            np.bool_(False),
            np.int64(3),
            np.float32(-0.0),
        ):
            for name in ("div", "divide"):
                self.assert_tensor_matches(
                    getattr(offset, name)(scalar),
                    offset / scalar,
                    case=(name, "offset scalar", type(scalar).__name__, scalar),
                    source=offset,
                )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        for name in ("div", "divide"):
            self.assert_tensor_matches(
                getattr(empty, name)(broadcast),
                empty / broadcast,
                case=(name, "strided broadcast empty"),
                source=empty,
            )

        special_bits = np.asarray(
            (
                0x7FC1_2345,
                0x7F80_0000,
                0xFF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x0000_0000,
                0x8000_0000,
                0x0000_0000,
                0x8000_0000,
            ),
            dtype=np.uint32,
        )
        denominator_bits = np.asarray(
            (
                0x3F80_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x0000_0000,
                0x0000_0000,
                0x4000_0000,
                0x4000_0000,
                0xC000_0000,
                0xC000_0000,
            ),
            dtype=np.uint32,
        )
        numerator = torch.tensor(memoryview(special_bits.view(np.float32)))
        denominator = torch.tensor(memoryview(denominator_bits.view(np.float32)))
        for name in ("div", "divide"):
            self.assert_tensor_matches(
                getattr(numerator, name)(denominator),
                numerator / denominator,
                case=(name, "signed zero nan infinity"),
                source=numerator,
            )

    def test_no_grad_allows_gradient_requiring_operands_but_active_autograd_rejects(self):
        for name in ("div", "divide"):
            left = torch.tensor([[2.0, 4.0]], requires_grad=True)
            right = torch.tensor([[1.0], [2.0]], requires_grad=True)
            with torch.no_grad():
                tensor_output = getattr(left.transpose(0, 1), name)(
                    right.transpose(0, 1)
                )
                scalar_output = getattr(left, name)(2.0)
                expected_tensor = left.transpose(0, 1) / right.transpose(0, 1)
                expected_scalar = left / 2.0
            self.assertFalse(tensor_output.requires_grad)
            self.assertTrue(tensor_output.is_leaf)
            self.assertFalse(scalar_output.requires_grad)
            self.assert_tensor_matches(
                tensor_output,
                expected_tensor,
                case=(name, "no_grad tensor"),
            )
            self.assert_tensor_matches(
                scalar_output, expected_scalar, case=(name, "no_grad scalar")
            )

            with self.assertRaisesRegex(
                RuntimeError,
                rf"^{name}\(\): autograd recording is not supported$",
            ):
                getattr(left, name)(torch.ones((1, 2)))
            with self.assertRaisesRegex(
                RuntimeError,
                rf"^{name}\(\): autograd recording is not supported$",
            ):
                getattr(torch.ones((1, 2)), name)(right)
            with self.assertRaisesRegex(
                RuntimeError,
                rf"^{name}\(\): autograd recording is not supported$",
            ):
                getattr(left, name)(2.0)

    def test_torch_function_modes_and_overrides_observe_methods_before_native_limits(self):
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
                (lambda name=name: getattr(left, name)(4.0), None),
                (lambda name=name: getattr(left, name)(other=right), ("other",)),
                (lambda name=name: getattr(left, name)(x2=right), ("x2",)),
                (
                    lambda name=name: getattr(left, name)(right, rounding_mode="trunc"),
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
                    forwarded = getattr(left, name)(other=right, rounding_mode=None)
            self.assertEqual(order, ["upper", "lower"])
            self.assert_tensor_matches(
                forwarded, left / right, case=(name, "forwarded")
            )

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
            rounding_mode = Override()
            self.assertIs(
                getattr(left, name)(right, rounding_mode=rounding_mode), marker
            )
            _, dispatch_types, _, kwargs = events[0]
            self.assertEqual(dispatch_types, (Override,))
            self.assertEqual(tuple(kwargs), ("rounding_mode",))
            self.assertIs(kwargs["rounding_mode"], rounding_mode)

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
        tensor = torch.tensor([1.0])
        destination = torch.tensor([17.0])
        for name in ("div", "divide"):
            method = getattr(tensor, name)
            for rounding_mode in ("trunc", "floor", "bad", True):
                with self.subTest(name=name, rounding_mode=rounding_mode):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        rf"^{name}\(\): rounding_mode values other than None are not supported$",
                    ):
                        method(2, rounding_mode=rounding_mode)

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
            with self.subTest(name=name, boundary="missing other"):
                with self.assertRaises(TypeError):
                    method()
            with self.subTest(name=name, boundary="too many positional"):
                with self.assertRaises(TypeError):
                    method(tensor, tensor)
            with self.subTest(name=name, boundary="unsupported operand"):
                with self.assertRaises(TypeError):
                    method([])
            with self.subTest(name=name, boundary="none operand"):
                with self.assertRaises(TypeError):
                    method(None)
            with self.subTest(name=name, boundary="wide unsigned"):
                with self.assertRaisesRegex(TypeError, "^an integer is required$"):
                    method(np.uint64(2**63))
            with self.subTest(name=name, boundary="positive integer overflow"):
                with self.assertRaisesRegex(OverflowError, "^int too big to convert$"):
                    method(2**64)
            with self.subTest(name=name, boundary="negative integer overflow"):
                with self.assertRaisesRegex(
                    OverflowError, "^can't convert negative int to unsigned$"
                ):
                    method(-(2**63) - 1)
            with self.subTest(name=name, boundary="shape mismatch"):
                with self.assertRaises(RuntimeError):
                    getattr(torch.ones((2, 2)), name)(torch.ones((3,)))


if __name__ == "__main__":
    unittest.main()
