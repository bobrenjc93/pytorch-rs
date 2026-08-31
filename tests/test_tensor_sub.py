import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


SUB_DOC = "\nsub(other, *, alpha=1) -> Tensor\n\nSee :func:`torch.sub`.\n"
SUBTRACT_DOC = (
    "\nsubtract(other, *, alpha=1) -> Tensor\n\nSee :func:`torch.subtract`.\n"
)


class TensorSubMethodTests(unittest.TestCase):
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

    def test_methods_reuse_subtraction_values_layouts_and_edge_cases(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [3.0], [4.0]])
        expected = left - right
        for name in ("sub", "subtract"):
            calls = (
                ("positional tensors", lambda name=name: getattr(left, name)(right)),
                ("other keyword", lambda name=name: getattr(left, name)(other=right)),
                ("x2 keyword", lambda name=name: getattr(left, name)(x2=right)),
                (
                    "default alpha",
                    lambda name=name: getattr(left, name)(right, alpha=1),
                ),
                (
                    "default numpy alpha",
                    lambda name=name: getattr(left, name)(right, alpha=np.int64(1)),
                ),
            )
            for case, call in calls:
                self.assert_tensor_matches(call(), expected, case=(name, case))

        offset = left[1]
        for scalar in (-2, 2.5, np.bool_(True), np.int64(3), np.float32(-0.0)):
            for name in ("sub", "subtract"):
                self.assert_tensor_matches(
                    getattr(offset, name)(scalar),
                    torch.sub(offset, scalar),
                    case=(name, "offset scalar", type(scalar).__name__, scalar),
                )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        for name in ("sub", "subtract"):
            self.assert_tensor_matches(
                getattr(empty, name)(broadcast),
                empty - broadcast,
                case=(name, "strided broadcast empty"),
            )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        zeros = torch.zeros((5,))
        for name in ("sub", "subtract"):
            self.assert_tensor_matches(
                getattr(special, name)(zeros),
                special - zeros,
                case=(name, "signed zero nan infinity"),
            )

    def test_autograd_no_grad_and_shared_operands_reuse_native_path(self):
        for name in ("sub", "subtract"):
            method_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
            method_right = torch.tensor(
                [[5.0], [7.0], [11.0]], requires_grad=True
            )
            operator_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
            operator_right = torch.tensor(
                [[5.0], [7.0], [11.0]], requires_grad=True
            )

            method_output = getattr(method_left.transpose(0, 1), name)(
                method_right.transpose(0, 1)
            )
            operator_output = operator_left.transpose(0, 1) - operator_right.transpose(
                0, 1
            )
            self.assert_tensor_matches(
                method_output, operator_output, case=(name, "tracked views")
            )
            method_output.sum().backward()
            operator_output.sum().backward()
            self.assert_tensor_matches(
                method_left.grad, operator_left.grad, case=(name, "left gradient")
            )
            self.assert_tensor_matches(
                method_right.grad, operator_right.grad, case=(name, "right gradient")
            )

            shared_method = torch.tensor([2.0, -3.0], requires_grad=True)
            shared_operator = torch.tensor([2.0, -3.0], requires_grad=True)
            getattr(shared_method, name)(shared_method).sum().backward()
            (shared_operator - shared_operator).sum().backward()
            self.assert_tensor_matches(
                shared_method.grad,
                shared_operator.grad,
                case=(name, "shared operand gradient"),
            )

            scalar_method = torch.tensor([2.0, -3.0], requires_grad=True)
            scalar_operator = torch.tensor([2.0, -3.0], requires_grad=True)
            getattr(scalar_method, name)(4.0).sum().backward()
            (scalar_operator - 4.0).sum().backward()
            self.assert_tensor_matches(
                scalar_method.grad,
                scalar_operator.grad,
                case=(name, "scalar gradient"),
            )

            empty_method = torch.zeros((2, 0, 3), requires_grad=True)
            empty_operator = torch.zeros((2, 0, 3), requires_grad=True)
            getattr(empty_method, name)(torch.ones((1, 1, 3))).sum().backward()
            (empty_operator - torch.ones((1, 1, 3))).sum().backward()
            self.assert_tensor_matches(
                empty_method.grad, empty_operator.grad, case=(name, "empty gradient")
            )

            no_grad_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
            no_grad_right = torch.tensor([[3.0], [4.0]], requires_grad=True)
            with torch.no_grad():
                tensor_output = getattr(no_grad_left.transpose(0, 1), name)(
                    no_grad_right.transpose(0, 1)
                )
                scalar_output = getattr(no_grad_left, name)(2.0)
            self.assertFalse(tensor_output.requires_grad)
            self.assertFalse(scalar_output.requires_grad)
            self.assertTrue(getattr(no_grad_left, name)(no_grad_right).requires_grad)

    def test_modes_and_overrides_observe_methods_before_native_limits(self):
        left = torch.tensor([2.0])
        right = torch.tensor([3.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for name in ("sub", "subtract"):
            descriptor = inspect.getattr_static(torch.Tensor, name)
            calls = (
                (lambda name=name: getattr(left, name)(right), None),
                (lambda name=name: getattr(left, name)(4.0), None),
                (lambda name=name: getattr(left, name)(other=right), ("other",)),
                (lambda name=name: getattr(left, name)(x2=right), ("x2",)),
                (lambda name=name: getattr(left, name)(right, alpha=2), ("alpha",)),
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
            self.assert_tensor_matches(forwarded, left - right, case=(name, "forwarded"))

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
            self.assertIs(getattr(left, name)(right, alpha=Override()), marker)
            _, dispatch_types, _, kwargs = events[0]
            self.assertEqual(dispatch_types, (Override,))
            self.assertEqual(tuple(kwargs), ("alpha",))

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
            self.assertIn(
                f"  - tensor subclass <class '{DecliningOverride.__module__}."
                f"{DecliningOverride.__qualname__}'>",
                str(raised.exception),
            )

    def test_callable_metadata_copy_pickle_reload_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        descriptors = {
            "sub": inspect.getattr_static(torch.Tensor, "sub"),
            "subtract": inspect.getattr_static(torch.Tensor, "subtract"),
        }
        self.assertIsNot(descriptors["sub"], descriptors["subtract"])

        for name, doc in (("sub", SUB_DOC), ("subtract", SUBTRACT_DOC)):
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

        for name in ("sub_", "subtract_"):
            self.assertFalse(hasattr(torch.Tensor, name))
            self.assertFalse(hasattr(torch, name))

    def test_unsupported_arguments_and_boundaries_do_not_mutate(self):
        tensor = torch.tensor([1.0])
        destination = torch.tensor([17.0])
        for name in ("sub", "subtract"):
            method = getattr(tensor, name)
            with self.subTest(name=name, boundary="alpha"):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf"^{name}\(\): alpha values other than 1 are not supported$",
                ):
                    method(tensor, alpha=2)
            with self.subTest(name=name, boundary="bool alpha"):
                with self.assertRaisesRegex(
                    RuntimeError, "^Boolean alpha only supported for Boolean results\\.$"
                ):
                    method(tensor, alpha=True)
            with self.subTest(name=name, boundary="bool other"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^Subtraction, the `-` operator, with a bool tensor is not "
                    r"supported\. If you are trying to invert a mask, use the `~` "
                    r"or `logical_not\(\)` operator instead\.$",
                ):
                    method(True)
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
            with self.subTest(name=name, boundary="positional alpha"):
                with self.assertRaises(TypeError):
                    method(2.0, 1)
            with self.subTest(name=name, boundary="unsupported operand"):
                with self.assertRaises(TypeError):
                    method([])

        with self.assertRaisesRegex(
            TypeError,
            r"^sub\(\) received an invalid combination of arguments - got "
            r"\(\), but expected \(Tensor other, \*, Number alpha = 1\)$",
        ):
            tensor.sub()
        with self.assertRaisesRegex(
            TypeError,
            r"^sub\(\) got an unexpected keyword argument 'out'$",
        ):
            tensor.sub(tensor, out=destination)
        with self.assertRaisesRegex(
            TypeError,
            r"^subtract\(\) received an invalid combination of arguments - got "
            r"\(Tensor, out=Tensor\), but expected one of:\n"
            r" \* \(Tensor other, \*, Number alpha = 1\)\n"
            r" \* \(Number other, Number alpha = 1\)"
            r"\n      didn't match because some of the keywords were incorrect: out\n$",
        ):
            tensor.subtract(tensor, out=destination)


if __name__ == "__main__":
    unittest.main()
