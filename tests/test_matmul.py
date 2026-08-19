import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = "\nmatmul(tensor2) -> Tensor\n\nSee :func:`torch.matmul`\n"
FUNCTION_DOC = """
matmul(input, other, *, out=None) -> Tensor

Matrix product of two tensors.

The behavior depends on the dimensionality of the tensors as follows:

- If both tensors are 1-dimensional, the dot product (scalar) is returned.
- If both arguments are 2-dimensional, the matrix-matrix product is returned.
- If the first argument is 1-dimensional and the second argument is 2-dimensional,
  a 1 is prepended to its dimension for the purpose of the matrix multiply.
  After the matrix multiply, the prepended dimension is removed.
- If the first argument is 2-dimensional and the second argument is 1-dimensional,
  the matrix-vector product is returned.
- If both arguments are at least 1-dimensional and at least one argument is
  N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
  argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
  batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
  1 is appended to its dimension for the purpose of the batched matrix multiply and removed after.

  The first N-2 dimensions of each argument, the batch dimensions, are
  :ref:`broadcast <broadcasting-semantics>` (and thus must be broadcastable).
  The last 2, the matrix dimensions, are handled as in the matrix-matrix product.

  For example, if :attr:`input` is a
  :math:`(j \\times 1 \\times n \\times m)` tensor and :attr:`other` is a :math:`(k \\times m \\times p)`
  tensor, the batch dimensions are :math:`(j \\times 1)` and :math:`(k)`,
  and the matrix dimensions are :math:`(n \\times m)` and :math:`(m \\times p)`.
  :attr:`out` will be a :math:`(j \\times k \\times n \\times p)` tensor.

This operation has support for arguments with :ref:`sparse layouts<sparse-docs>`. In particular the
matrix-matrix (both arguments 2-dimensional) supports sparse arguments with the same restrictions
as :func:`torch.mm`


.. warning::
    Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,
    or may not have autograd support. If you notice missing functionality please
    open a feature request.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

.. note::

    The 1-dimensional dot product version of this function does not support an :attr:`out` parameter.

Arguments:
    input (Tensor): the first tensor to be multiplied
    other (Tensor): the second tensor to be multiplied

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> # vector x vector
    >>> tensor1 = torch.randn(3)
    >>> tensor2 = torch.randn(3)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([])
    >>> # matrix x vector
    >>> tensor1 = torch.randn(3, 4)
    >>> tensor2 = torch.randn(4)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([3])
    >>> # batched matrix x broadcasted vector
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(4)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3])
    >>> # batched matrix x batched matrix
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(10, 4, 5)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3, 5])
    >>> # batched matrix x broadcasted matrix
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(4, 5)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3, 5])

"""


def matmul_layout_cases(module):
    offset_left = module.tensor(
        np.arange(18, dtype=np.float32).reshape(3, 2, 3).tolist()
    )[1]
    offset_right = module.tensor(
        np.arange(12, dtype=np.float32).reshape(2, 3, 2).tolist()
    )[1]
    offset_transposed_right = module.tensor(
        np.arange(12, dtype=np.float32).reshape(2, 2, 3).tolist()
    )[1].transpose(0, 1)

    strided_left = module.tensor(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    ).transpose(0, 1)
    strided_right = module.tensor(
        [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]
    ).transpose(0, 1)

    special_left = module.tensor(
        [[float("inf"), 1.0], [float("-inf"), -1.0], [float("nan"), 2.0]]
    )
    special_right = module.tensor([[1.0, -1.0], [0.5, 1.0]])

    return (
        (
            "contiguous",
            module.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            module.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        ),
        ("offset contiguous", offset_left, offset_right),
        ("offset transpose-contiguous rhs", offset_left, offset_transposed_right),
        (
            "awkward transpose-contiguous rhs",
            module.tensor(np.arange(35, dtype=np.float32).reshape(5, 7).tolist()),
            module.tensor(np.arange(21, dtype=np.float32).reshape(3, 7).tolist())
            .transpose(0, 1),
        ),
        ("strided", strided_left, strided_right),
        (
            "offset empty rows",
            module.zeros((2, 0, 2)).transpose(0, 2)[1],
            module.ones((2, 4)),
        ),
        ("empty inner", module.ones((2, 0)), module.zeros((0, 3))),
        ("non-finite", special_left, special_right),
        ("non-finite transpose-contiguous rhs", special_left, special_right.T),
    )


class TensorMatmulTests(unittest.TestCase):
    def assert_delegates_to_operator(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def layout_cases(self):
        return matmul_layout_cases(torch)

    def test_positional_and_keyword_calls_delegate_to_matrix_operator(self):
        for case, left, right in self.layout_cases():
            expected = left @ right
            self.assert_delegates_to_operator(
                left.matmul(right), expected, case=(case, "positional")
            )
            self.assert_delegates_to_operator(
                left.matmul(other=right), expected, case=(case, "keyword")
            )
            self.assert_delegates_to_operator(
                left.matmul(x2=right), expected, case=(case, "x2 alias")
            )

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
        left = torch.tensor([[1.0]])
        right = torch.tensor([[2.0]])
        descriptor = inspect.getattr_static(torch.Tensor, "matmul")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        calls = (
            ("positional", lambda: left.matmul(right), None),
            ("other keyword", lambda: left.matmul(other=right), "other"),
            ("x2 keyword", lambda: left.matmul(x2=right), "x2"),
        )
        for case, call, keyword in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, ())
                self.assertIs(args[0], left)
                if keyword is None:
                    self.assertEqual(len(args), 2)
                    self.assertIs(args[1], right)
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(len(args), 1)
                    self.assertEqual(tuple(kwargs), (keyword,))
                    self.assertIs(kwargs[keyword], right)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        expected = left @ right
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                actual = left.matmul(other=right)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_delegates_to_operator(actual, expected, case="forwarded modes")

        invalid_calls = (
            (
                lambda: left.matmul([]),
                "matmul(): argument 'other' (position 1) must be Tensor, not list",
            ),
            (
                lambda: left.matmul(x2=[]),
                "matmul(): argument 'other' must be Tensor, not list",
            ),
            (
                lambda: left.matmul(x2=right, wat=right),
                "matmul() got an unexpected keyword argument 'x2'",
            ),
        )
        for call, message in invalid_calls:
            mode = RecordingMode()
            with mode:
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()
            self.assertEqual(mode.calls, [])

    def test_other_torch_function_override_dispatches_after_declining_mode(self):
        left = torch.tensor([[1.0]])
        descriptor = inspect.getattr_static(torch.Tensor, "matmul")
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        calls = (
            ("positional", lambda value: left.matmul(value), None),
            ("other keyword", lambda value: left.matmul(other=value), "other"),
            ("x2 keyword", lambda value: left.matmul(x2=value), "x2"),
        )
        for case, call, keyword in calls:
            value = Override()
            Override.calls.clear()
            self.assertIs(call(value), marker)
            self.assertEqual(len(Override.calls), 1)
            function, dispatch_types, args, kwargs = Override.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, (Override,))
                self.assertIs(args[0], left)
                if keyword is None:
                    self.assertEqual(len(args), 2)
                    self.assertIs(args[1], value)
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(len(args), 1)
                    self.assertEqual(tuple(kwargs), (keyword,))
                    self.assertIs(kwargs[keyword], value)

        mode_calls = []

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                mode_calls.append((func, types, args, kwargs))
                return NotImplemented

        value = Override()
        Override.calls.clear()
        with DecliningMode():
            self.assertIs(left.matmul(x2=value), marker)
        self.assertEqual(len(mode_calls), 1)
        self.assertEqual(len(Override.calls), 1)
        self.assertIs(mode_calls[0][0], descriptor)
        self.assertEqual(mode_calls[0][1], (Override,))
        self.assertEqual(len(mode_calls[0][2]), 1)
        self.assertIs(mode_calls[0][3]["x2"], value)
        self.assertIs(Override.calls[0][0], descriptor)
        self.assertEqual(len(Override.calls[0][2]), 1)
        self.assertIs(Override.calls[0][3]["x2"], value)

    def test_existing_operator_autograd_behavior_is_preserved(self):
        method_left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        method_right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        operator_left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        operator_right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        method_output = method_left.matmul(other=method_right)
        operator_output = operator_left @ operator_right
        self.assert_delegates_to_operator(
            method_output, operator_output, case="requires-grad operands"
        )
        self.assertFalse(method_output.requires_grad)
        self.assertTrue(method_output.is_leaf)

        for output in (method_output, operator_output):
            with self.assertRaises(RuntimeError) as raised:
                output.sum().backward()
            self.assertEqual(
                str(raised.exception),
                "element 0 of tensors does not require grad and does not have a grad_fn",
            )
        for operand in (method_left, method_right, operator_left, operator_right):
            self.assertIsNone(operand.grad)

    def test_rank_two_shape_errors_reuse_operator_and_other_ranks_stay_unsupported(self):
        for left_shape, right_shape in (
            ((2, 3), (4, 2)),
            ((0, 3), (4, 0)),
        ):
            left = torch.zeros(left_shape)
            right = torch.zeros(right_shape)
            message = (
                "mat1 and mat2 shapes cannot be multiplied "
                f"({left_shape[0]}x{left_shape[1]} and "
                f"{right_shape[0]}x{right_shape[1]})"
            )
            for call in (
                lambda left=left, right=right: left.matmul(right),
                lambda left=left, right=right: left.matmul(other=right),
                lambda left=left, right=right: left.matmul(x2=right),
                lambda left=left, right=right: left @ right,
            ):
                with self.subTest(left=left_shape, right=right_shape):
                    with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                        call()

        rank_cases = (
            (torch.tensor(1.0), torch.ones((1, 1))),
            (torch.ones((2,)), torch.ones((2, 2))),
            (torch.ones((1, 2, 2)), torch.ones((2, 2))),
        )
        for left, right in rank_cases:
            with self.subTest(left=left.shape, right=right.shape):
                with self.assertRaises(RuntimeError) as method_error:
                    left.matmul(right)
                with self.assertRaises(RuntimeError) as operator_error:
                    left @ right
                self.assertEqual(str(method_error.exception), str(operator_error.exception))
                self.assertIn("requires two rank-2 tensors", str(method_error.exception))

    def test_descriptor_metadata_unbound_call_and_binding_errors(self):
        tensor = torch.tensor([[1.0]])
        descriptor = inspect.getattr_static(torch.Tensor, "matmul")
        bound = tensor.matmul

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertEqual(descriptor.__qualname__, "TensorBase.matmul")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(bound.__qualname__, "Tensor.matmul")
        self.assertIsNone(bound.__module__)
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "matmul")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assert_delegates_to_operator(
            descriptor(tensor, other=tensor), tensor @ tensor, case="unbound keyword"
        )
        self.assert_delegates_to_operator(
            descriptor(tensor, x2=tensor), tensor @ tensor, case="unbound x2 alias"
        )

        cases = (
            (lambda: tensor.matmul(), 'matmul() missing 1 required positional arguments: "other"'),
            (
                lambda: tensor.matmul(tensor, tensor),
                "matmul() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.matmul(tensor, other=tensor),
                "matmul() got multiple values for argument 'other'",
            ),
            (
                lambda: tensor.matmul(tensor, out=tensor),
                "matmul() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: tensor.matmul(wat=tensor),
                'matmul() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.matmul([]),
                "matmul(): argument 'other' (position 1) must be Tensor, not list",
            ),
            (
                lambda: tensor.matmul(other=None),
                "matmul(): argument 'other' must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.matmul([], out=tensor),
                "matmul(): argument 'other' (position 1) must be Tensor, not list",
            ),
            (
                lambda: tensor.matmul(x2=[]),
                "matmul(): argument 'other' must be Tensor, not list",
            ),
            (
                lambda: tensor.matmul(tensor, x2=tensor),
                "matmul() got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: tensor.matmul(x2=tensor, wat=tensor),
                "matmul() got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: tensor.matmul(**{"wat": tensor, "x2": tensor}),
                "matmul() got an unexpected keyword argument 'wat'",
            ),
            (
                lambda: tensor.matmul(x2=tensor, other=[]),
                "matmul(): argument 'other' must be Tensor, not list",
            ),
            (
                lambda: tensor.matmul(x2=[], other=tensor),
                "matmul() got an unexpected keyword argument 'x2'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        with self.assertRaisesRegex(
            TypeError, r"^unbound method TensorBase\.matmul\(\) needs an argument$"
        ):
            descriptor()
        with self.assertRaisesRegex(
            TypeError,
            r"^descriptor 'matmul' for 'torch\._C\.TensorBase' objects "
            r"doesn't apply to a 'int' object$",
        ):
            descriptor(1, tensor)


class TopLevelMatmulTests(unittest.TestCase):
    def assert_matches_method(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def test_positional_canonical_and_alias_calls_reuse_tensor_matmul(self):
        for case, left, right in matmul_layout_cases(torch):
            expected = left.matmul(right)
            calls = (
                ("positional", lambda: torch.matmul(left, right)),
                (
                    "canonical keywords",
                    lambda: torch.matmul(input=left, other=right),
                ),
                ("x1/x2 aliases", lambda: torch.matmul(x1=left, x2=right)),
            )
            for style, call in calls:
                self.assert_matches_method(call(), expected, case=(case, style))

    def test_shape_rank_autograd_and_out_limits_remain_narrow(self):
        for left_shape, right_shape in (
            ((2, 3), (4, 2)),
            ((0, 3), (4, 0)),
        ):
            left = torch.zeros(left_shape)
            right = torch.zeros(right_shape)
            message = (
                "mat1 and mat2 shapes cannot be multiplied "
                f"({left_shape[0]}x{left_shape[1]} and "
                f"{right_shape[0]}x{right_shape[1]})"
            )
            for call in (
                lambda: torch.matmul(left, right),
                lambda: torch.matmul(input=left, other=right),
                lambda: torch.matmul(x1=left, x2=right),
            ):
                with self.subTest(left=left_shape, right=right_shape):
                    with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                        call()

        rank_cases = (
            (torch.tensor(1.0), torch.ones((1, 1))),
            (torch.ones((2,)), torch.ones((2, 2))),
            (torch.ones((1, 2, 2)), torch.ones((2, 2))),
        )
        for left, right in rank_cases:
            with self.subTest(left=left.shape, right=right.shape):
                with self.assertRaises(RuntimeError) as function_error:
                    torch.matmul(left, right)
                with self.assertRaises(RuntimeError) as method_error:
                    left.matmul(right)
                self.assertEqual(str(function_error.exception), str(method_error.exception))
                self.assertIn("requires two rank-2 tensors", str(function_error.exception))

        left = torch.ones((1, 1), requires_grad=True)
        right = torch.ones((1, 1), requires_grad=True)
        result = torch.matmul(input=left, other=right)
        expected = left.matmul(right)
        self.assert_matches_method(result, expected, case="requires-grad operands")
        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)

        with self.assertRaisesRegex(
            TypeError, r"^matmul\(\) got an unexpected keyword argument 'out'$"
        ):
            torch.matmul(left, right, out=torch.zeros((1, 1)))

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
        left = torch.tensor([[1.0]])
        right = torch.tensor([[2.0]])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        calls = (
            ("positional", lambda: torch.matmul(left, right), None),
            (
                "canonical",
                lambda: torch.matmul(input=left, other=right),
                ("input", "other"),
            ),
            (
                "aliases",
                lambda: torch.matmul(x1=left, x2=right),
                ("x1", "x2"),
            ),
        )
        for case, call, keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, torch.matmul)
                self.assertEqual(dispatch_types, ())
                if keywords is None:
                    self.assertEqual(len(args), 2)
                    self.assertIs(args[0], left)
                    self.assertIs(args[1], right)
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(args, ())
                    self.assertEqual(tuple(kwargs), keywords)
                    self.assertIs(kwargs[keywords[0]], left)
                    self.assertIs(kwargs[keywords[1]], right)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                actual = torch.matmul(x1=left, x2=right)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_matches_method(actual, left.matmul(right), case="forwarded modes")

        for call in (
            lambda: torch.matmul([], right),
            lambda: torch.matmul(left, []),
            lambda: torch.matmul(left, right, unexpected=True),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

    def test_operand_override_order_types_and_declining_errors(self):
        native = torch.tensor([[1.0]])
        events = []
        marker = object()

        class LeftOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("left", func, types, args, kwargs))
                return NotImplemented

        class RightOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("right", func, types, args, kwargs))
                return marker

        self.assertIs(torch.matmul(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.matmul)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertIsInstance(args[0], LeftOverride)
            self.assertIsInstance(args[1], RightOverride)
            self.assertIsNone(kwargs)

        events.clear()
        self.assertIs(torch.matmul(input=native, other=RightOverride()), marker)
        self.assertEqual(len(events), 1)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.matmul)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other"))
        self.assertIs(kwargs["input"], native)
        self.assertIsInstance(kwargs["other"], RightOverride)

        subclass_events = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_events.append("base")
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_events.append(("derived", types))
                return marker

        self.assertIs(torch.matmul(BaseOverride(), DerivedOverride()), marker)
        self.assertEqual(
            subclass_events,
            [("derived", (DerivedOverride, BaseOverride))],
        )

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.matmul(DecliningOverride(), native)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.matmul'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with "
            "TORCH_LOGS=not_implemented",
        )

    def test_binding_type_metadata_documentation_and_exports(self):
        tensor = torch.tensor([[1.0]])
        cases = (
            (
                lambda: torch.matmul(),
                'matmul() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.matmul(tensor),
                'matmul() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.matmul(tensor, tensor, tensor),
                "matmul() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.matmul([], tensor),
                "matmul(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.matmul(tensor, []),
                "matmul(): argument 'other' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.matmul(input=None, other=tensor),
                "matmul(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.matmul(x1=tensor, x2=[]),
                "matmul(): argument 'other' must be Tensor, not list",
            ),
            (
                lambda: torch.matmul(tensor, tensor, input=tensor),
                "matmul() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.matmul(tensor, tensor, x2=tensor),
                "matmul() got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: torch.matmul(foo=tensor),
                'matmul() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.matmul(tensor, tensor, extra=True),
                "matmul() got an unexpected keyword argument 'extra'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        function = torch.matmul
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "matmul")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.matmul")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertRegex(
            repr(function),
            r"^<built-in method matmul of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.matmul, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("matmul"), 2)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["matmul"], function)


if __name__ == "__main__":
    unittest.main()
