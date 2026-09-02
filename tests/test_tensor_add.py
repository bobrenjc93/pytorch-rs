import copy
import importlib
import inspect
import pickle
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


ADD_DOC = (
    "\nadd(other, *, alpha=1) -> Tensor\n\n"
    "Add a scalar or tensor to :attr:`self` tensor. If both :attr:`alpha`\n"
    "and :attr:`other` are specified, each element of :attr:`other` is scaled by\n"
    ":attr:`alpha` before being used.\n\n"
    "When :attr:`other` is a tensor, the shape of :attr:`other` must be\n"
    ":ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying\n"
    "tensor\n\n"
    "See :func:`torch.add`\n"
)


class TensorAddMethodTests(unittest.TestCase):
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

    def test_method_reuses_addition_values_layouts_and_edge_cases(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [-0.0], [float("inf")]])
        expected = left + right

        calls = (
            ("positional tensors", lambda: left.add(right)),
            ("other keyword", lambda: left.add(other=right)),
            ("x2 keyword", lambda: left.add(x2=right)),
            ("default alpha", lambda: left.add(right, alpha=1)),
            ("default numpy alpha", lambda: left.add(right, alpha=np.int64(1))),
            (
                "default numpy bool alpha",
                lambda: left.add(right, alpha=np.bool_(True)),
            ),
        )
        for case, call in calls:
            self.assert_tensor_matches(call(), expected, case=case)

        self.assert_tensor_matches(
            left.add(1, right),
            left + right,
            case="legacy positional default alpha tensor other",
        )
        legacy_calls = (
            ("two positional scalars", lambda: left.add(1, 2), left + 2),
            ("keyword scalar other", lambda: left.add(1, other=2), left + 2),
            ("keyword tensor other", lambda: left.add(1, other=right), left + right),
            ("x2 tensor other", lambda: left.add(1, x2=right), left + right),
        )
        for case, call, expected_legacy in legacy_calls:
            self.assert_tensor_matches(
                call(),
                expected_legacy,
                case=("legacy positional default alpha", case),
            )

        offset = left[1]
        for scalar in (True, -2, 2.5, np.bool_(True), np.int64(3), np.float32(-0.0)):
            self.assert_tensor_matches(
                offset.add(scalar),
                offset + scalar,
                case=("offset scalar", type(scalar).__name__, scalar),
            )
            self.assert_tensor_matches(
                offset.add(other=scalar),
                offset + scalar,
                case=("keyword scalar", type(scalar).__name__, scalar),
            )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            empty.add(broadcast),
            empty + broadcast,
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        zeros = torch.zeros((5,))
        result = special.add(zeros)
        self.assert_tensor_matches(result, special + zeros, case="signed zero nan infinity")
        self.assertFalse(result.is_set_to(special))
        self.assertFalse(result.is_set_to(zeros))
        if result.numel():
            self.assertNotEqual(result.data_ptr(), special.data_ptr())
            self.assertNotEqual(result.data_ptr(), zeros.data_ptr())

    def test_autograd_no_grad_and_shared_operands_reuse_native_path(self):
        method_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        method_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        operator_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        operator_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)

        method_output = method_left.transpose(0, 1).add(method_right.transpose(0, 1))
        operator_output = operator_left.transpose(0, 1) + operator_right.transpose(
            0, 1
        )
        self.assert_tensor_matches(method_output, operator_output, case="tracked views")
        method_output.sum().backward()
        operator_output.sum().backward()
        self.assert_tensor_matches(method_left.grad, operator_left.grad, case="left gradient")
        self.assert_tensor_matches(
            method_right.grad, operator_right.grad, case="right gradient"
        )

        shared_method = torch.tensor([2.0, -3.0], requires_grad=True)
        shared_operator = torch.tensor([2.0, -3.0], requires_grad=True)
        shared_method.add(shared_method).sum().backward()
        (shared_operator + shared_operator).sum().backward()
        self.assert_tensor_matches(
            shared_method.grad, shared_operator.grad, case="shared operand gradient"
        )

        scalar_method = torch.tensor([2.0, -3.0], requires_grad=True)
        scalar_operator = torch.tensor([2.0, -3.0], requires_grad=True)
        scalar_method.add(4.0).sum().backward()
        (scalar_operator + 4.0).sum().backward()
        self.assert_tensor_matches(
            scalar_method.grad, scalar_operator.grad, case="scalar gradient"
        )

        empty_method = torch.zeros((2, 0, 3), requires_grad=True)
        empty_operator = torch.zeros((2, 0, 3), requires_grad=True)
        empty_method.add(torch.ones((1, 1, 3))).sum().backward()
        (empty_operator + torch.ones((1, 1, 3))).sum().backward()
        self.assert_tensor_matches(empty_method.grad, empty_operator.grad, case="empty gradient")

        no_grad_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        no_grad_right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            tensor_output = no_grad_left.transpose(0, 1).add(
                no_grad_right.transpose(0, 1)
            )
            scalar_output = no_grad_left.add(2.0)
        self.assertFalse(tensor_output.requires_grad)
        self.assertFalse(scalar_output.requires_grad)
        self.assertTrue(no_grad_left.add(no_grad_right).requires_grad)

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

        descriptor = inspect.getattr_static(torch.Tensor, "add")
        calls = (
            (lambda: left.add(right), None),
            (lambda: left.add(4.0), None),
            (lambda: left.add(1, other=right), ("other",)),
            (lambda: left.add(other=right), ("other",)),
            (lambda: left.add(x2=right), ("x2",)),
            (lambda: left.add(right, alpha=2), ("alpha",)),
        )
        for call, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(keywords=expected_keywords):
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
                left.add([])
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
                forwarded = left.add(other=right)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(forwarded, left + right, case="forwarded")

        events = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append((func, types, args, kwargs))
                return marker

        self.assertIs(left.add(Override()), marker)
        function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (Override,))
        self.assertIs(args[0], left)
        self.assertIsInstance(args[1], Override)
        self.assertIsNone(kwargs)

        events.clear()
        self.assertIs(left.add(right, alpha=Override()), marker)
        _, dispatch_types, _, kwargs = events[0]
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(tuple(kwargs), ("alpha",))

        events.clear()
        legacy_alpha = Override()
        self.assertIs(left.add(legacy_alpha, right), marker)
        function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (Override,))
        self.assertIs(args[0], left)
        self.assertIs(args[1], legacy_alpha)
        self.assertIs(args[2], right)
        self.assertIsNone(kwargs)

        def make_legacy_override(label):
            class LegacyOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    legacy_events.append((label, func, types, args, kwargs))
                    return label

            LegacyOverride.__name__ = label
            return LegacyOverride

        First = make_legacy_override("first")
        Second = make_legacy_override("second")
        legacy_cases = (
            ("positional other", lambda: left.add(First(), Second()), None),
            ("other keyword", lambda: left.add(First(), other=Second()), ("other",)),
            ("x2 keyword", lambda: left.add(First(), x2=Second()), ("x2",)),
        )
        for case, call, expected_keywords in legacy_cases:
            legacy_events = []
            with self.subTest(legacy_dispatch=case):
                self.assertEqual(call(), "first")
                self.assertEqual(len(legacy_events), 1)
                label, function, dispatch_types, args, kwargs = legacy_events[0]
                self.assertEqual(label, "first")
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, (First, Second))
                self.assertIs(args[0], left)
                self.assertIsInstance(args[1], First)
                if expected_keywords is None:
                    self.assertIsInstance(args[2], Second)
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(tuple(kwargs), expected_keywords)
                    self.assertIsInstance(kwargs[expected_keywords[0]], Second)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            left.add(DecliningOverride())
        self.assertIn("Multiple dispatch failed for 'torch.Tensor.add'", str(raised.exception))
        self.assertIn(
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>",
            str(raised.exception),
        )

    def test_callable_metadata_copy_pickle_reload_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "add")
        bound = tensor.add

        self.assertIs(getattr(torch.Tensor, "add"), descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(repr(descriptor), "<method 'add' of 'torch._C.TensorBase' objects>")
        self.assertEqual(descriptor.__name__, "add")
        self.assertEqual(descriptor.__qualname__, "TensorBase.add")
        self.assertEqual(bound.__name__, "add")
        self.assertEqual(bound.__qualname__, "Tensor.add")
        self.assertEqual(descriptor.__doc__, ADD_DOC)
        self.assertEqual(bound.__doc__, ADD_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertNotIn("add", torch.Tensor.__dict__)

        for callable_object in (descriptor, bound):
            with self.subTest(callable=type(callable_object).__name__):
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)
                self.assertIs(copy.copy(callable_object), callable_object)
                self.assertIs(copy.deepcopy(callable_object), callable_object)

        self.assert_tensor_matches(
            descriptor(tensor, other=tensor),
            tensor + tensor,
            case="unbound call",
        )
        with self.assertRaises(TypeError):
            descriptor(1, 2)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(descriptor, protocol)), descriptor)

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(inspect.getattr_static(torch.Tensor, "add"), descriptor)
        self.assertFalse(hasattr(torch.Tensor, "add_"))
        self.assertIsNot(torch.add, descriptor)

    def test_descriptor_pickle_survives_package_reinitialization(self):
        descriptor = inspect.getattr_static(torch.Tensor, "add")
        saved_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == "torch_rs" or name.startswith("torch_rs.")
        }
        try:
            for name in saved_modules:
                sys.modules.pop(name, None)
            importlib.import_module("torch_rs")
        finally:
            for name in tuple(sys.modules):
                if name == "torch_rs" or name.startswith("torch_rs."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(descriptor, protocol)), descriptor)

    def test_unsupported_arguments_and_boundaries_do_not_mutate(self):
        tensor = torch.tensor([1.0])
        destination = torch.tensor([17.0])
        alias_other = torch.tensor([3.0])
        alias_x2 = torch.tensor([1.0])

        class UnexpectedOverride:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return object()

        with self.assertRaisesRegex(
            NotImplementedError,
            r"^add\(\): alpha values other than 1 are not supported$",
        ):
            tensor.add(tensor, alpha=2)
        with self.assertRaisesRegex(
            RuntimeError, "^Boolean alpha only supported for Boolean results\\.$"
        ):
            tensor.add(tensor, alpha=True)
        with self.assertRaises(TypeError):
            tensor.add(tensor, out=destination)
        self.assertEqual(destination.tolist(), [17.0])
        with self.assertRaises(TypeError):
            tensor.add(tensor, dtype=torch.float32)
        with self.assertRaises(TypeError):
            tensor.add(tensor, device=torch.device("cpu"))
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^add\(\): alpha values other than 1 are not supported$",
        ):
            tensor.add(2, torch.tensor([3.0]))
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^add\(\): alpha values other than 1 are not supported$",
        ):
            tensor.add(2, 1)
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^add\(\): alpha values other than 1 are not supported$",
        ):
            tensor.add(2, other=torch.tensor([3.0]))
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^add\(\): alpha values other than 1 are not supported$",
        ):
            tensor.add(2, x2=torch.tensor([3.0]))
        with self.assertRaises(TypeError):
            tensor.add([])
        with self.assertRaisesRegex(
            TypeError,
            r"^add\(\) received an invalid combination of arguments - got "
            r"\(\), but expected \(Tensor other, \*, Number alpha = 1\)$",
        ):
            tensor.add()
        with self.assertRaisesRegex(
            TypeError,
            r"^add\(\) got an unexpected keyword argument 'out'$",
        ):
            tensor.add(tensor, out=destination)

        ambiguous_alias_cases = (
            ("x2 then other", {"x2": alias_x2, "other": alias_other}),
            ("other then x2", {"other": alias_other, "x2": alias_x2}),
            (
                "override x2 then other",
                {"x2": UnexpectedOverride(), "other": alias_other},
            ),
            (
                "other then override x2",
                {"other": alias_other, "x2": UnexpectedOverride()},
            ),
        )
        for case, kwargs in ambiguous_alias_cases:
            with self.subTest(boundary=case):
                UnexpectedOverride.calls = 0
                with self.assertRaisesRegex(
                    TypeError, r"^add\(\) got an unexpected keyword argument 'x2'$"
                ):
                    tensor.add(**kwargs)
                self.assertEqual(UnexpectedOverride.calls, 0)


if __name__ == "__main__":
    unittest.main()
