import inspect
import re
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch


FUNCTION_DOC = (
    "\nt(input) -> Tensor\n\n"
    "Expects :attr:`input` to be <= 2-D tensor and transposes dimensions 0\n"
    "and 1.\n\n"
    "0-D and 1-D tensors are returned as is. When input is a 2-D tensor this\n"
    "is equivalent to ``transpose(input, 0, 1)``.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n\n"
    "Example::\n\n"
    "    >>> x = torch.randn(())\n"
    "    >>> x\n"
    "    tensor(0.1995)\n"
    "    >>> torch.t(x)\n"
    "    tensor(0.1995)\n"
    "    >>> x = torch.randn(3)\n"
    "    >>> x\n"
    "    tensor([ 2.4320, -0.4608,  0.7702])\n"
    "    >>> torch.t(x)\n"
    "    tensor([ 2.4320, -0.4608,  0.7702])\n"
    "    >>> x = torch.randn(2, 3)\n"
    "    >>> x\n"
    "    tensor([[ 0.4875,  0.9158, -0.5872],\n"
    "            [ 0.3938, -0.6929,  0.6932]])\n"
    "    >>> torch.t(x)\n"
    "    tensor([[ 0.4875,  0.3938],\n"
    "            [ 0.9158, -0.6929],\n"
    "            [-0.5872,  0.6932]])\n\n"
    "See also :func:`torch.transpose`.\n"
)


class TorchTTests(unittest.TestCase):
    def assert_tensor(self, actual, expected, *, shape, stride, offset=0):
        self.assertEqual(actual.shape, shape)
        self.assertEqual(actual.stride(), stride)
        self.assertEqual(actual.storage_offset(), offset)
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected, dtype=np.float32)
        )

    def test_positional_input_and_legacy_aliases_return_shared_storage_views(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        offset_matrix = leaf.transpose(0, 2)[1]
        cases = (
            (torch.tensor(3.5, requires_grad=True), 3.5, (), (), 0),
            (
                torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
                [1.0, 2.0, 3.0],
                (3,),
                (1,),
                0,
            ),
            (
                torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    requires_grad=True,
                ),
                [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
                (3, 2),
                (1, 3),
                0,
            ),
            (offset_matrix, values[:, :, 1], (2, 3), (12, 4), 1),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for source, expected, shape, stride, offset in cases:
                views = (
                    torch.t(source),
                    torch.t(input=source),
                    torch.t(a=source),
                    torch.t(x=source),
                )
                for view in views:
                    with self.subTest(source_shape=source.shape):
                        self.assertIsNot(view, source)
                        self.assertIs(view.dtype, source.dtype)
                        self.assertEqual(view.device, source.device)
                        self.assertEqual(view.requires_grad, source.requires_grad)
                        self.assertEqual(view.data_ptr(), source.data_ptr())
                        self.assert_tensor(
                            view,
                            expected,
                            shape=shape,
                            stride=stride,
                            offset=offset,
                        )
                        self.assertEqual(view.tolist(), source.t().tolist())
                        self.assertEqual(view.stride(), source.t().stride())
        self.assertEqual(caught, [])

    def test_rank_guard_preserves_the_tensor_method_error(self):
        for rank in (3, 4, 65):
            source = torch.zeros((0,) + (1,) * (rank - 1))
            message = f"t() expects a tensor with <= 2 dimensions, but self is {rank}D"
            for call in (lambda: torch.t(source), lambda: torch.t(input=source)):
                with self.subTest(rank=rank):
                    with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                        call()

    def test_autograd_and_no_grad_follow_the_tensor_method(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        transformed = torch.t(input=leaf.transpose(0, 2)[1])
        weights = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertTrue(transformed.requires_grad)
        self.assertFalse(transformed.is_leaf)
        (transformed * weights).sum().backward()

        expected = np.zeros_like(values)
        expected[:, :, 1] = np.asarray(weights)
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

        scalar = torch.tensor(2.0, requires_grad=True)
        (torch.t(scalar) * 7.0).backward()
        self.assertEqual(scalar.grad.item(), 7.0)

        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with torch.no_grad():
            view = torch.t(input=source)
        self.assertTrue(view.requires_grad)
        self.assertTrue(view.is_leaf)
        self.assertEqual(view.data_ptr(), source.data_ptr())

    def test_callable_metadata_and_documentation(self):
        function = torch.t
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertTrue(callable(function))
        self.assertEqual(function.__name__, "t")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        with self.assertRaises(ValueError):
            inspect.signature(function)
        self.assertIn("t", torch.__all__)

    def test_binding_and_tensor_type_errors(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.t(),
                't() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.t(tensor, tensor),
                "t() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.t(tensor, input=tensor),
                "t() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.t(tensor, extra=True, input=tensor),
                "t() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.t(tensor, input=tensor, extra=True),
                "t() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.t(extra=tensor),
                't() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.t(1, extra=True),
                "t(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.t(input=[]),
                "t(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.t(a=1),
                "t(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.t(x=[]),
                "t(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.t(np.zeros((2, 3), dtype=np.float32)),
                "t(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
