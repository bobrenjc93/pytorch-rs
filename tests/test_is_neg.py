import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nis_neg() -> bool\n\n"
    "Returns True if the negative bit of :attr:`self` is set to true.\n"
)


class TensorIsNegTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        strided_view = source.transpose(0, 1)
        offset_view = strided_view[1]
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )

        self.assertFalse(strided_view.is_contiguous())
        self.assertGreater(offset_view.storage_offset(), 0)
        return leaf, tracked, (
            ("scalar", torch.tensor(-3.5)),
            ("empty", torch.zeros((2, 0, 3))),
            ("eager negative", source.neg()),
            ("strided view", strided_view),
            ("offset strided view", offset_view),
            ("extreme empty view", extreme_empty),
            ("autograd leaf", leaf),
            ("autograd non-leaf view", tracked),
            ("detached autograd view", tracked.detach()),
        )

    def top_level_calls(self, tensor):
        return (
            ("positional", torch.is_neg(tensor)),
            ("input", torch.is_neg(input=tensor)),
            ("x", torch.is_neg(x=tensor)),
            ("a", torch.is_neg(a=tensor)),
            ("x1", torch.is_neg(x1=tensor)),
        )

    def test_supported_tensors_have_a_clear_negative_bit_without_mutation(self):
        leaf, tracked, cases = self.tensor_cases()
        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                    tensor.dtype,
                    tensor.device,
                    tensor.requires_grad,
                    tensor.is_leaf,
                )

                result = tensor.is_neg()

                self.assertIs(type(result), bool)
                self.assertIs(result, False)
                self.assertEqual(
                    (
                        tensor.shape,
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.data_ptr(),
                        tensor.dtype,
                        tensor.device,
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                    metadata,
                )

        tracked.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 2), 2.0, dtype=np.float32)
        )
        self.assertIs(leaf.is_neg(), False)
        self.assertIs(tracked.is_neg(), False)

    def test_top_level_supported_tensors_have_a_clear_bit_without_mutation(self):
        leaf, tracked, cases = self.tensor_cases()
        for case, tensor in cases:
            metadata = (
                tensor.shape,
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
                tensor.dtype,
                tensor.device,
                tensor.requires_grad,
                tensor.is_leaf,
            )
            for form, result in self.top_level_calls(tensor):
                with self.subTest(
                    case=case,
                    form=form,
                    shape=tensor.shape,
                    stride=tensor.stride(),
                ):
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
                    self.assertEqual(
                        (
                            tensor.shape,
                            tensor.stride(),
                            tensor.storage_offset(),
                            tensor.data_ptr(),
                            tensor.dtype,
                            tensor.device,
                            tensor.requires_grad,
                            tensor.is_leaf,
                        ),
                        metadata,
                    )

        tracked.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 2), 2.0, dtype=np.float32)
        )
        self.assertIs(torch.is_neg(leaf), False)
        self.assertIs(torch.is_neg(tracked), False)

    def test_tensorbase_descriptor_metadata_matches_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_neg")
        bound = tensor.is_neg

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'is_neg' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "is_neg")
        self.assertEqual(descriptor.__qualname__, "TensorBase.is_neg")
        self.assertEqual(bound.__name__, "is_neg")
        self.assertEqual(bound.__qualname__, "Tensor.is_neg")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        for callable_object, expected_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            if sys.version_info >= (3, 13):
                self.assertEqual(
                    callable_object.__text_signature__, "($self, /)"
                )
                self.assertEqual(
                    str(inspect.signature(callable_object)), expected_signature
                )
            else:
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(descriptor(tensor), False)

    def test_invalid_calls_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_neg")
        bound = tensor.is_neg
        cases = (
            (
                lambda: tensor.is_neg(1),
                "TensorBase.is_neg() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.is_neg() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.is_neg() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.is_neg(1, 2),
                "TensorBase.is_neg() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.is_neg(input=tensor),
                "TensorBase.is_neg() takes no keyword arguments",
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.is_neg() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.is_neg() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.is_neg() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'is_neg' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.is_neg() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_method_descriptor_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_neg")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.is_neg()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
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
                forwarded = tensor.is_neg()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, False)

    def test_top_level_callable_metadata_null_documentation_and_exports(self):
        function = torch.is_neg
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "is_neg")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.is_neg")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__doc__)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method is_neg of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.is_neg, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("is_neg"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["is_neg"], function)

    def test_top_level_binding_and_type_error_precedence(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.is_neg(),
                'is_neg() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.is_neg(tensor, tensor),
                "is_neg() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.is_neg(tensor, input=tensor),
                "is_neg() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.is_neg(tensor, extra=True, input=tensor),
                "is_neg() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.is_neg(tensor, input=tensor, extra=True),
                "is_neg() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.is_neg(extra=tensor),
                'is_neg() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.is_neg(1, extra=True),
                "is_neg(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.is_neg(input=[]),
                "is_neg(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.is_neg(a=1),
                "is_neg(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.is_neg(x=[]),
                "is_neg(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.is_neg(x1=None),
                "is_neg(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.is_neg(a=tensor, x=tensor),
                "is_neg() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.is_neg(x=tensor, a=tensor),
                "is_neg() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.is_neg(input=tensor, x1=tensor),
                "is_neg() got an unexpected keyword argument 'x1'",
            ),
            (
                lambda: torch.is_neg(x=tensor, x1=tensor),
                "is_neg() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.is_neg(x1=tensor, x=tensor),
                "is_neg() got an unexpected keyword argument 'x1'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_top_level_torch_function_modes_and_overrides(self):
        tensor = torch.tensor([1.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        calls = (
            (None, lambda: torch.is_neg(tensor)),
            ("input", lambda: torch.is_neg(input=tensor)),
            ("x", lambda: torch.is_neg(x=tensor)),
            ("a", lambda: torch.is_neg(a=tensor)),
            ("x1", lambda: torch.is_neg(x1=tensor)),
        )
        for keyword, call in calls:
            mode = RecordingMode()
            with mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.is_neg)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor,) if keyword is None else ())
            self.assertEqual(kwargs, None if keyword is None else {keyword: tensor})

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.is_neg(a=tensor)
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, False)

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(torch.is_neg(x=value), marker)
        function, dispatch_types, args, kwargs = override_calls[0]
        self.assertIs(function, torch.is_neg)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"x": value})

        fallback_order = []

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                fallback_order.append(("mode", func, types, args, kwargs))
                return NotImplemented

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_order.append(("override", func, types, args, kwargs))
                return marker

        fallback = FallbackOverride()
        with DecliningMode():
            self.assertIs(torch.is_neg(input=fallback), marker)
        self.assertEqual([entry[0] for entry in fallback_order], ["mode", "override"])
        for _, function, dispatch_types, args, kwargs in fallback_order:
            self.assertIs(function, torch.is_neg)
            self.assertEqual(dispatch_types, (FallbackOverride,))
            self.assertEqual(args, ())
            self.assertEqual(kwargs, {"input": fallback})

    def test_top_level_not_implemented_errors_match_variable_dispatch(self):
        tensor = torch.tensor([1.0])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        mode = DecliningMode()
        message = (
            "Multiple dispatch failed for 'torch.is_neg'; all "
            "__torch_function__ handlers returned NotImplemented:\n\n"
            f"  - mode object {mode!r}\n\n"
            "For more information, try re-running with "
            "TORCH_LOGS=not_implemented"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            with mode:
                torch.is_neg(tensor)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        message = (
            "Multiple dispatch failed for 'torch.is_neg'; all "
            "__torch_function__ handlers returned NotImplemented:\n\n"
            f"  - tensor subclass {DecliningOverride!r}\n\n"
            "For more information, try re-running with "
            "TORCH_LOGS=not_implemented"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            torch.is_neg(DecliningOverride())

        self.assertIs(torch.is_neg(tensor), False)

    def test_top_level_scope_does_not_add_lazy_negative_views(self):
        self.assertTrue(hasattr(torch, "is_neg"))
        self.assertFalse(hasattr(torch, "_neg_view"))


if __name__ == "__main__":
    unittest.main()
