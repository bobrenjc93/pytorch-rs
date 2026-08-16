import inspect
import pickle
import re
import sys
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

from signature_utils import assert_no_argument_signature


METHOD_DOC = "\nadjoint() -> Tensor\n\nAlias for :func:`adjoint`\n"
FUNCTION_DOC = (
    "\nadjoint(input: Tensor) -> Tensor\n"
    "Returns a view of the tensor conjugated and with the last two dimensions "
    "transposed.\n\n"
    "``x.adjoint()`` is equivalent to ``x.transpose(-2, -1).conj()`` for complex "
    "tensors and\n"
    "to ``x.transpose(-2, -1)`` for real tensors.\n\n"
    "Args:\n"
    "    {input}\n\n"
    "Example::\n\n"
    "    >>> x = torch.arange(4, dtype=torch.float)\n"
    "    >>> A = torch.complex(x, x).reshape(2, 2)\n"
    "    >>> A\n"
    "    tensor([[0.+0.j, 1.+1.j],\n"
    "            [2.+2.j, 3.+3.j]])\n"
    "    >>> A.adjoint()\n"
    "    tensor([[0.-0.j, 2.-2.j],\n"
    "            [1.-1.j, 3.-3.j]])\n"
    "    >>> (A.adjoint() == A.mH).all()\n"
    "    tensor(True)\n"
)
SCALAR_WARNING = "adjoint() is deprecated on 0-D tensors. Consider using x.conj()."


def stable_warning_message(message):
    return message.split(" (Triggered internally at ", 1)[0]


class TensorAdjointTests(unittest.TestCase):
    def top_level_adjoint_calls(self, source):
        return (
            ("positional", torch.adjoint(source)),
            ("input", torch.adjoint(input=source)),
            ("x", torch.adjoint(x=source)),
            ("a", torch.adjoint(a=source)),
            ("x1", torch.adjoint(x1=source)),
        )

    def assert_native_adjoint_view(self, source):
        expected = source.mH
        result = source.adjoint()

        self.assertIsNot(result, source)
        self.assertTrue(result.is_set_to(expected))
        self.assertTrue(result.is_set_to(source.transpose(-2, -1)))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.stride(), expected.stride())
        self.assertEqual(result.storage_offset(), source.storage_offset())
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertFalse(result.is_conj())
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))

        restored = result.adjoint()
        self.assertIsNot(restored, source)
        self.assertTrue(restored.is_set_to(source))

    def test_00_scalar_warning_is_once_only_and_points_to_the_caller(self):
        scalar = torch.tensor(2.5, requires_grad=True)
        marker = object()

        class InterceptingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return marker

        with warnings.catch_warnings(record=True) as intercepted_warnings:
            warnings.simplefilter("always")
            with InterceptingMode():
                intercepted = torch.adjoint(scalar)
        self.assertIs(intercepted, marker)
        self.assertEqual(intercepted_warnings, [])

        metadata = (
            scalar.shape,
            scalar.stride(),
            scalar.storage_offset(),
            scalar.dtype,
            scalar.device,
            scalar.requires_grad,
            scalar.is_leaf,
            scalar.data_ptr(),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warning_line = inspect.currentframe().f_lineno + 1
            first = torch.adjoint(scalar)
            second = scalar.adjoint()
            keyword_results = (
                torch.adjoint(input=scalar),
                torch.adjoint(x=scalar),
                torch.adjoint(a=scalar),
                torch.adjoint(x1=scalar),
            )

        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, UserWarning)
        self.assertEqual(stable_warning_message(str(caught[0].message)), SCALAR_WARNING)
        self.assertEqual(caught[0].filename, __file__)
        self.assertEqual(caught[0].lineno, warning_line)
        self.assertIs(first, scalar)
        self.assertIs(second, scalar)
        for result in keyword_results:
            self.assertIs(result, scalar)
        self.assertEqual(
            (
                first.shape,
                first.stride(),
                first.storage_offset(),
                first.dtype,
                first.device,
                first.requires_grad,
                first.is_leaf,
                first.data_ptr(),
            ),
            metadata,
        )

    def test_matrix_batched_empty_offset_and_strided_views_reuse_mh(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        dense = torch.tensor(values.tolist())
        cases = (
            ("matrix", torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
            ("batched", dense),
            ("empty", torch.zeros((2, 0, 3))),
            ("strided", dense.transpose(0, 3)),
            ("offset strided", dense.transpose(0, 3)[1]),
            ("empty offset", torch.zeros((3, 0, 2)).transpose(0, 2)[1]),
        )

        for case, source in cases:
            with self.subTest(case=case, shape=source.shape, stride=source.stride()):
                self.assert_native_adjoint_view(source)

    def test_vector_error_and_extreme_empty_boundaries_match_mh_engine(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\.adjoint\(\) is only supported on matrices or batches of matrices\. Got 1-D tensor\.$",
        ):
            torch.tensor([1.0, 2.0, 3.0]).adjoint()

        maximum = sys.maxsize
        with self.assertRaisesRegex(
            RuntimeError, r"^numel: integer multiplication overflow$"
        ):
            torch.zeros((maximum, 0, maximum)).adjoint()

        offset = torch.zeros((maximum, 0, 1))[maximum - 1]
        result = offset.adjoint()
        self.assertEqual(result.shape, (1, 0))
        self.assertEqual(result.stride(), (1, 1))
        self.assertEqual(result.storage_offset(), maximum - 1)
        self.assertEqual(result.data_ptr(), offset.data_ptr())
        self.assertEqual(result.tolist(), [[]])

    def test_autograd_and_no_grad_follow_the_mh_view_path(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        view = leaf.adjoint()
        self.assertTrue(view.requires_grad)
        self.assertFalse(view.is_leaf)
        self.assertTrue(view.is_set_to(leaf.mH))
        self.assertEqual(view.shape, (3, 2))
        self.assertEqual(view.stride(), (1, 3))
        self.assertEqual(view.data_ptr(), leaf.data_ptr())

        (view * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.asarray([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32),
        )

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty.adjoint().sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.numel(), 0)

        no_grad_source = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        with torch.no_grad():
            no_grad_view = no_grad_source.adjoint()
        self.assertTrue(no_grad_view.requires_grad)
        self.assertTrue(no_grad_view.is_leaf)
        self.assertEqual(no_grad_view.shape, (3, 2))
        self.assertEqual(no_grad_view.stride(), (1, 3))
        self.assertEqual(no_grad_view.data_ptr(), no_grad_source.data_ptr())
        (no_grad_view * no_grad_view).sum().backward()
        self.assertIsNone(no_grad_source.grad)

        scalar = torch.tensor(3.0, requires_grad=True)
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            self.assertIs(scalar.adjoint(), scalar)

    def test_tensorbase_method_descriptor_metadata_and_invalid_calls(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        descriptor = inspect.getattr_static(torch.Tensor, "adjoint")
        bound = tensor.adjoint

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertTrue(callable(descriptor))
        self.assertEqual(descriptor.__name__, "adjoint")
        self.assertEqual(descriptor.__qualname__, "TensorBase.adjoint")
        self.assertEqual(bound.__name__, "adjoint")
        self.assertEqual(bound.__qualname__, "Tensor.adjoint")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertEqual(
            repr(descriptor),
            "<method 'adjoint' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.adjoint, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertTrue(descriptor(tensor).is_set_to(tensor.mH))
        self.assertTrue(bound(**{}).is_set_to(tensor.mH))
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")

        cases = (
            (
                lambda: tensor.adjoint(1),
                "TensorBase.adjoint() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.adjoint(1, 2),
                "TensorBase.adjoint() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.adjoint(dim=0),
                "TensorBase.adjoint() takes no keyword arguments",
            ),
            (
                lambda: bound(1),
                "Tensor.adjoint() takes no arguments (1 given)",
            ),
            (
                lambda: bound(dim=0),
                "Tensor.adjoint() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.adjoint() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.adjoint() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'adjoint' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.adjoint() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_method_descriptor_and_forward(self):
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        descriptor = inspect.getattr_static(torch.Tensor, "adjoint")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.adjoint()
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
                forwarded = tensor.adjoint()
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(forwarded.is_set_to(tensor.mH))

    def test_not_implemented_reenters_the_declining_top_mode(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return NotImplemented

        class AcceptingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return object()

        lower = AcceptingMode()
        upper = DecliningMode()
        with self.assertRaisesRegex(
            RecursionError, r"^maximum recursion depth exceeded$"
        ):
            with lower:
                with upper:
                    tensor.adjoint()
        self.assertGreater(upper.calls, 1)
        self.assertEqual(lower.calls, 0)
        self.assertTrue(tensor.adjoint().is_set_to(tensor.mH))

    def test_top_level_call_forms_reuse_matrix_adjoint_views(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        dense = torch.tensor(values.tolist())
        cases = (
            ("matrix", torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
            ("batched", dense),
            ("empty", torch.zeros((2, 0, 3))),
            ("strided", dense.transpose(0, 3)),
            ("offset strided", dense.transpose(0, 3)[1]),
            ("empty offset", torch.zeros((3, 0, 2)).transpose(0, 2)[1]),
        )

        for case, source in cases:
            expected = source.mH
            for form, result in self.top_level_adjoint_calls(source):
                with self.subTest(
                    case=case, form=form, shape=source.shape, stride=source.stride()
                ):
                    self.assertIsNot(result, source)
                    self.assertTrue(result.is_set_to(expected))
                    self.assertTrue(result.is_set_to(source.adjoint()))
                    self.assertEqual(result.shape, expected.shape)
                    self.assertEqual(result.stride(), expected.stride())
                    self.assertEqual(result.storage_offset(), source.storage_offset())
                    self.assertIs(result.dtype, source.dtype)
                    self.assertEqual(result.device, source.device)
                    self.assertEqual(result.data_ptr(), source.data_ptr())
                    self.assertFalse(result.is_conj())
                    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))

    def test_top_level_vector_error_and_extreme_empty_boundaries(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\.adjoint\(\) is only supported on matrices or batches of matrices\. Got 1-D tensor\.$",
        ):
            torch.adjoint(torch.tensor([1.0, 2.0, 3.0]))

        maximum = sys.maxsize
        with self.assertRaisesRegex(
            RuntimeError, r"^numel: integer multiplication overflow$"
        ):
            torch.adjoint(input=torch.zeros((maximum, 0, maximum)))

        offset = torch.zeros((maximum, 0, 1))[maximum - 1]
        result = torch.adjoint(a=offset)
        self.assertEqual(result.shape, (1, 0))
        self.assertEqual(result.stride(), (1, 1))
        self.assertEqual(result.storage_offset(), maximum - 1)
        self.assertEqual(result.data_ptr(), offset.data_ptr())
        self.assertEqual(result.tolist(), [[]])

    def test_top_level_autograd_and_no_grad_reuse_the_method_path(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        for form, view in self.top_level_adjoint_calls(leaf):
            with self.subTest(form=form):
                self.assertTrue(view.requires_grad)
                self.assertFalse(view.is_leaf)
                self.assertTrue(view.is_set_to(leaf.adjoint()))
                self.assertEqual(view.shape, (3, 2))
                self.assertEqual(view.stride(), (1, 3))
                self.assertEqual(view.data_ptr(), leaf.data_ptr())

        (torch.adjoint(a=leaf) * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.asarray([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32),
        )

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.adjoint(x=empty).sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.numel(), 0)

        no_grad_source = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        with torch.no_grad():
            no_grad_view = torch.adjoint(input=no_grad_source)
        self.assertTrue(no_grad_view.requires_grad)
        self.assertTrue(no_grad_view.is_leaf)
        self.assertEqual(no_grad_view.shape, (3, 2))
        self.assertEqual(no_grad_view.stride(), (1, 3))
        self.assertEqual(no_grad_view.data_ptr(), no_grad_source.data_ptr())
        (no_grad_view * no_grad_view).sum().backward()
        self.assertIsNone(no_grad_source.grad)

    def test_top_level_callable_metadata_documentation_and_exports(self):
        function = torch.adjoint
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "adjoint")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.adjoint")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method adjoint of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.adjoint, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("adjoint"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["adjoint"], function)

    def test_top_level_binding_and_type_error_precedence(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        cases = (
            (
                lambda: torch.adjoint(),
                'adjoint() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.adjoint(tensor, tensor),
                "adjoint() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.adjoint(tensor, input=tensor),
                "adjoint() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.adjoint(tensor, extra=True, input=tensor),
                "adjoint() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.adjoint(tensor, input=tensor, extra=True),
                "adjoint() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.adjoint(extra=tensor),
                'adjoint() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.adjoint(1, extra=True),
                "adjoint(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.adjoint(input=[]),
                "adjoint(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.adjoint(a=1),
                "adjoint(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.adjoint(x=[]),
                "adjoint(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.adjoint(x1=None),
                "adjoint(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.adjoint(a=tensor, x=tensor),
                "adjoint() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.adjoint(x=tensor, a=tensor),
                "adjoint() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.adjoint(input=tensor, a=tensor),
                "adjoint() got an unexpected keyword argument 'a'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_top_level_torch_function_modes_and_overrides(self):
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        for keyword in (None, "input", "x", "a", "x1"):
            mode = RecordingMode()
            with mode:
                result = (
                    torch.adjoint(tensor)
                    if keyword is None
                    else torch.adjoint(**{keyword: tensor})
                )
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.adjoint)
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
                forwarded = torch.adjoint(a=tensor)
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(forwarded.is_set_to(tensor.adjoint()))

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(torch.adjoint(x=value), marker)
        function, dispatch_types, args, kwargs = override_calls[0]
        self.assertIs(function, torch.adjoint)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"x": value})


if __name__ == "__main__":
    unittest.main()
