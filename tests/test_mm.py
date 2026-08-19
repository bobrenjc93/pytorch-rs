import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = "\nmm(mat2) -> Tensor\n\nSee :func:`torch.mm`\n"


def mm_layout_cases(module):
    offset_left = module.tensor(
        np.arange(18, dtype=np.float32).reshape(3, 2, 3).tolist()
    )[1]
    offset_right = module.tensor(
        np.arange(12, dtype=np.float32).reshape(2, 3, 2).tolist()
    )[1]
    strided_left = module.tensor(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    ).transpose(0, 1)
    strided_right = module.tensor(
        [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]
    ).transpose(0, 1)
    return (
        (
            "contiguous",
            module.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            module.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        ),
        ("offset contiguous", offset_left, offset_right),
        ("strided", strided_left, strided_right),
        (
            "offset empty rows",
            module.zeros((2, 0, 2)).transpose(0, 2)[1],
            module.ones((2, 4)),
        ),
        ("empty inner", module.ones((2, 0)), module.zeros((0, 3))),
        (
            "non-finite",
            module.tensor(
                [
                    [float("inf"), 1.0],
                    [float("-inf"), -1.0],
                    [float("nan"), 2.0],
                ]
            ),
            module.tensor([[1.0, -1.0], [0.5, 1.0]]),
        ),
    )


class TensorMmTests(unittest.TestCase):
    def assert_tensor_bits_equal(self, actual, expected, *, case):
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

    def test_rank_two_layouts_delegate_to_existing_matmul_kernel(self):
        for case, left, right in mm_layout_cases(torch):
            expected = left.matmul(right)
            self.assert_tensor_bits_equal(
                left.mm(right), expected, case=(case, "positional")
            )
            self.assert_tensor_bits_equal(
                left.mm(mat2=right), expected, case=(case, "mat2 keyword")
            )

    def test_autograd_no_grad_and_existing_matmul_behavior(self):
        left = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        right = torch.tensor(
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], requires_grad=True
        )
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        output = left.mm(mat2=right)
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        (output * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(left.grad),
            [[23.0, 29.0, 35.0], [53.0, 67.0, 81.0]],
        )
        np.testing.assert_array_equal(
            np.asarray(right.grad),
            [[13.0, 18.0], [17.0, 24.0], [21.0, 30.0]],
        )

        left_only = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        fixed_right = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        left_only.mm(fixed_right).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(left_only.grad), [[11.0, 15.0], [11.0, 15.0]]
        )

        fixed_left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        right_only = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]], requires_grad=True
        )
        fixed_left.mm(right_only).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(right_only.grad), [[4.0, 4.0], [6.0, 6.0]]
        )

        no_grad_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        no_grad_right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            untracked = no_grad_left.mm(no_grad_right)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        np.testing.assert_array_equal(np.asarray(untracked), [[11.0]])

        # Tensor.matmul, the @ operator, and top-level torch.matmul retain their
        # pre-existing inference-only autograd behavior.
        for result in (
            no_grad_left.matmul(no_grad_right),
            no_grad_left @ no_grad_right,
            torch.matmul(no_grad_left, no_grad_right),
        ):
            self.assertFalse(result.requires_grad)
            self.assertTrue(result.is_leaf)

    def test_autograd_handles_strided_views_and_empty_dimensions(self):
        left_base = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        right_base = torch.tensor(
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], requires_grad=True
        )
        left = left_base.transpose(0, 1)
        right = right_base.transpose(0, 1)
        (left.mm(right) * torch.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(left_base.grad),
            [[27.0, 61.0], [30.0, 68.0], [33.0, 75.0]],
        )
        np.testing.assert_array_equal(
            np.asarray(right_base.grad),
            [[7.0, 15.0, 23.0], [10.0, 22.0, 34.0]],
        )

        empty_left = torch.zeros((2, 0), requires_grad=True)
        empty_right = torch.zeros((0, 3), requires_grad=True)
        empty_left.mm(empty_right).sum().backward()
        self.assertEqual(empty_left.grad.shape, (2, 0))
        self.assertEqual(empty_right.grad.shape, (0, 3))

        no_rows = torch.zeros((0, 2), requires_grad=True)
        populated_right = torch.ones((2, 3), requires_grad=True)
        no_rows.mm(populated_right).sum().backward()
        self.assertEqual(no_rows.grad.shape, (0, 2))
        np.testing.assert_array_equal(
            np.asarray(populated_right.grad), np.zeros((2, 3), dtype=np.float32)
        )

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
        left = torch.tensor([[1.0]])
        right = torch.tensor([[2.0]])
        descriptor = inspect.getattr_static(torch.Tensor, "mm")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        for case, call, keyword in (
            ("positional", lambda: left.mm(right), None),
            ("keyword", lambda: left.mm(mat2=right), "mat2"),
        ):
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

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                actual = left.mm(mat2=right)
        self.assertEqual(order, ["upper", "lower"])
        np.testing.assert_array_equal(np.asarray(actual), [[2.0]])

        for call, message in (
            (
                lambda: left.mm([]),
                "mm(): argument 'mat2' (position 1) must be Tensor, not list",
            ),
            (
                lambda: left.mm(mat2=[]),
                "mm(): argument 'mat2' must be Tensor, not list",
            ),
            (
                lambda: left.mm(mat2=right, wat=right),
                "mm() got an unexpected keyword argument 'wat'",
            ),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()
            self.assertEqual(mode.calls, [])

    def test_mat2_torch_function_override_runs_after_declining_mode(self):
        left = torch.tensor([[1.0]])
        descriptor = inspect.getattr_static(torch.Tensor, "mm")
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for case, call, keyword in (
            ("positional", lambda value: left.mm(value), None),
            ("keyword", lambda value: left.mm(mat2=value), "mat2"),
        ):
            value = Override()
            Override.calls.clear()
            self.assertIs(call(value), marker)
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
                    self.assertEqual(tuple(kwargs), ("mat2",))
                    self.assertIs(kwargs["mat2"], value)

        mode_calls = []

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                mode_calls.append((func, types, args, kwargs))
                return NotImplemented

        value = Override()
        Override.calls.clear()
        with DecliningMode():
            self.assertIs(left.mm(mat2=value), marker)
        self.assertEqual(len(mode_calls), 1)
        self.assertEqual(len(Override.calls), 1)
        self.assertIs(mode_calls[0][0], descriptor)
        self.assertEqual(mode_calls[0][1], (Override,))
        self.assertEqual(len(mode_calls[0][2]), 1)
        self.assertIs(mode_calls[0][3]["mat2"], value)
        self.assertIs(Override.calls[0][0], descriptor)

    def test_descriptor_metadata_tensorbase_ownership_and_binding_errors(self):
        tensor = torch.tensor([[1.0]])
        descriptor = inspect.getattr_static(torch.Tensor, "mm")
        bound = tensor.mm

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertEqual(descriptor.__qualname__, "TensorBase.mm")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(bound.__qualname__, "Tensor.mm")
        self.assertIsNone(bound.__module__)
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "mm")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assert_tensor_bits_equal(
            descriptor(tensor, mat2=tensor), tensor.matmul(tensor), case="unbound"
        )

        cases = (
            (lambda: tensor.mm(), 'mm() missing 1 required positional arguments: "mat2"'),
            (
                lambda: tensor.mm(tensor, tensor),
                "mm() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.mm(tensor, mat2=tensor),
                "mm() got multiple values for argument 'mat2'",
            ),
            (
                lambda: tensor.mm(tensor, out=tensor),
                "mm() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: tensor.mm(other=tensor),
                'mm() missing 1 required positional arguments: "mat2"',
            ),
            (
                lambda: tensor.mm(x2=tensor),
                'mm() missing 1 required positional arguments: "mat2"',
            ),
            (
                lambda: tensor.mm([]),
                "mm(): argument 'mat2' (position 1) must be Tensor, not list",
            ),
            (
                lambda: tensor.mm(mat2=None),
                "mm(): argument 'mat2' must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.mm([], out=tensor),
                "mm(): argument 'mat2' (position 1) must be Tensor, not list",
            ),
            (
                lambda: tensor.mm(mat2=[], wat=tensor),
                "mm(): argument 'mat2' must be Tensor, not list",
            ),
            (
                lambda: tensor.mm(mat2=tensor, wat=tensor),
                "mm() got an unexpected keyword argument 'wat'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        with self.assertRaisesRegex(
            TypeError, r"^unbound method TensorBase\.mm\(\) needs an argument$"
        ):
            descriptor()
        with self.assertRaisesRegex(
            TypeError,
            r"^descriptor 'mm' for 'torch\._C\.TensorBase' objects "
            r"doesn't apply to a 'int' object$",
        ):
            descriptor(1, tensor)

    def test_non_matrices_out_and_top_level_mm_stay_unsupported(self):
        matrix = torch.ones((2, 2))
        for left, right, message in (
            (torch.tensor(1.0), matrix, "self must be a matrix"),
            (torch.ones((2,)), matrix, "self must be a matrix"),
            (torch.ones((1, 2, 2)), matrix, "self must be a matrix"),
            (matrix, torch.tensor(1.0), "mat2 must be a matrix"),
            (matrix, torch.ones((2,)), "mat2 must be a matrix"),
            (matrix, torch.ones((1, 2, 2)), "mat2 must be a matrix"),
        ):
            with self.subTest(left=left.shape, right=right.shape):
                with self.assertRaisesRegex(RuntimeError, f"^{message}$"):
                    left.mm(right)

        for left_shape, right_shape in (((2, 3), (4, 2)), ((0, 3), (4, 0))):
            left = torch.zeros(left_shape)
            right = torch.zeros(right_shape)
            message = (
                "mat1 and mat2 shapes cannot be multiplied "
                f"({left_shape[0]}x{left_shape[1]} and "
                f"{right_shape[0]}x{right_shape[1]})"
            )
            with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                left.mm(right)

        with self.assertRaisesRegex(
            TypeError, r"^mm\(\) got an unexpected keyword argument 'out'$"
        ):
            matrix.mm(matrix, out=matrix)
        self.assertFalse(hasattr(torch, "mm"))
        self.assertNotIn("mm", torch.__all__)
        self.assertTrue(hasattr(torch, "matmul"))


if __name__ == "__main__":
    unittest.main()
