import inspect
import pickle
import re
import types
import unittest

import numpy as np

import torch_rs as torch


METHOD_DOC = "\nunsqueeze(dim) -> Tensor\n\nSee :func:`torch.unsqueeze`\n"
FUNCTION_DOC = (
    "\nunsqueeze(input, dim) -> Tensor\n\n"
    "Returns a new tensor with a dimension of size one inserted at the\n"
    "specified position.\n\n"
    "The returned tensor shares the same underlying data with this tensor.\n\n"
    "A :attr:`dim` value within the range ``[-input.dim() - 1, input.dim() + 1)``\n"
    "can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`\n"
    "applied at :attr:`dim` = ``dim + input.dim() + 1``.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n"
    "    dim (int): the index at which to insert the singleton dimension\n\n"
    "Example::\n\n"
    "    >>> x = torch.tensor([1, 2, 3, 4])\n"
    "    >>> torch.unsqueeze(x, 0)\n"
    "    tensor([[ 1,  2,  3,  4]])\n"
    "    >>> torch.unsqueeze(x, 1)\n"
    "    tensor([[ 1],\n"
    "            [ 2],\n"
    "            [ 3],\n"
    "            [ 4]])\n"
)


def layout_cases():
    values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
    base = torch.tensor(values.tolist())
    return (
        ("scalar", torch.tensor(-0.0), (1,), (1,), 0),
        ("empty", torch.zeros((2, 0, 3)), (1, 2, 0, 3), (6, 3, 3, 1), 0),
        ("offset", base[1], (1, 2, 3, 4), (24, 12, 4, 1), 24),
        (
            "noncontiguous",
            base.transpose(0, 3)[1],
            (1, 2, 3, 2),
            (24, 12, 4, 24),
            1,
        ),
    )


def trailing_layout_cases():
    values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
    base = torch.tensor(values.tolist())
    return (
        ("scalar", torch.tensor(-0.0), (1,), (1,), 0),
        ("empty", torch.zeros((2, 0, 3)), (2, 0, 3, 1), (3, 3, 1, 1), 0),
        ("offset", base[1], (2, 3, 4, 1), (12, 4, 1, 1), 24),
        (
            "noncontiguous",
            base.transpose(0, 3)[1],
            (2, 3, 2, 1),
            (12, 4, 24, 1),
            1,
        ),
    )


class TensorUnsqueezeTests(unittest.TestCase):
    def assert_leading_unsqueeze(self, source, result, shape, stride, offset):
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.stride(), stride)
        self.assertEqual(result.storage_offset(), offset)
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertFalse(result.is_set_to(source))
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        self.assertEqual(result.tolist(), [source.tolist()])

    def assert_trailing_unsqueeze(self, source, result, shape, stride, offset):
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.stride(), stride)
        self.assertEqual(result.storage_offset(), offset)
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertFalse(result.is_set_to(source))
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        self.assertEqual(
            result.tolist(), np.expand_dims(np.asarray(source), axis=-1).tolist()
        )

    def test_method_edge_calls_reuse_native_unsqueeze_views(self):
        for case, source, shape, stride, offset in layout_cases():
            rank = source.dim()
            calls = (
                ("front positional", lambda: source.unsqueeze(0)),
                ("front negative", lambda: source.unsqueeze(-rank - 1)),
                ("front keyword", lambda: source.unsqueeze(dim=0)),
                ("front axis alias", lambda: source.unsqueeze(axis=0)),
            )
            for form, call in calls:
                with self.subTest(case=case, form=form):
                    self.assert_leading_unsqueeze(
                        source, call(), shape, stride, offset
                    )

        for case, source, shape, stride, offset in trailing_layout_cases():
            rank = source.dim()
            calls = (
                ("back positional", lambda: source.unsqueeze(rank)),
                ("back negative", lambda: source.unsqueeze(-1)),
                ("back keyword", lambda: source.unsqueeze(dim=rank)),
                ("back axis alias", lambda: source.unsqueeze(axis=rank)),
            )
            for form, call in calls:
                with self.subTest(case=case, form=form):
                    self.assert_trailing_unsqueeze(
                        source, call(), shape, stride, offset
                    )

    def test_top_level_edge_calls_reuse_native_unsqueeze_views(self):
        for case, source, shape, stride, offset in layout_cases():
            rank = source.dim()
            calls = (
                ("front positional", lambda: torch.unsqueeze(source, 0)),
                ("front negative", lambda: torch.unsqueeze(source, -rank - 1)),
                ("front keywords", lambda: torch.unsqueeze(input=source, dim=0)),
                ("front reordered", lambda: torch.unsqueeze(dim=0, input=source)),
                ("front x alias", lambda: torch.unsqueeze(x=source, axis=0)),
                ("front a alias", lambda: torch.unsqueeze(a=source, dim=0)),
                ("front x1 alias", lambda: torch.unsqueeze(x1=source, dim=0)),
            )
            for form, call in calls:
                with self.subTest(case=case, form=form):
                    self.assert_leading_unsqueeze(
                        source, call(), shape, stride, offset
                    )

        for case, source, shape, stride, offset in trailing_layout_cases():
            rank = source.dim()
            calls = (
                ("back positional", lambda: torch.unsqueeze(source, rank)),
                ("back negative", lambda: torch.unsqueeze(source, -1)),
                ("back keywords", lambda: torch.unsqueeze(input=source, dim=rank)),
                ("back axis alias", lambda: torch.unsqueeze(input=source, axis=rank)),
            )
            for form, call in calls:
                with self.subTest(case=case, form=form):
                    self.assert_trailing_unsqueeze(
                        source, call(), shape, stride, offset
                    )

    def make_autograd_case(self, case):
        if case == "scalar":
            leaf = torch.tensor(-2.0, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return leaf, leaf

        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        if case == "offset":
            return leaf, leaf[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 3)[1]
        raise AssertionError(f"unknown case: {case}")

    def expected_gradient(self, case):
        if case == "scalar":
            return np.asarray(1.0, dtype=np.float32)
        if case == "empty":
            return np.zeros((2, 0, 3), dtype=np.float32)
        if case == "offset":
            expected = np.zeros((2, 2, 3, 4), dtype=np.float32)
            expected[1] = 1.0
            return expected
        if case == "noncontiguous":
            expected = np.zeros((2, 2, 3, 4), dtype=np.float32)
            expected[:, :, :, 1] = 1.0
            return expected
        raise AssertionError(f"unknown case: {case}")

    def test_autograd_and_no_grad_cover_every_supported_layout(self):
        for direction, cases, call in (
            ("leading", layout_cases(), lambda source: source.unsqueeze(0)),
            (
                "trailing",
                trailing_layout_cases(),
                lambda source: torch.unsqueeze(source, -1),
            ),
        ):
            for case, _, shape, stride, offset in cases:
                with self.subTest(direction=direction, case=case, mode="autograd"):
                    leaf, source = self.make_autograd_case(case)
                    result = call(source)
                    if direction == "leading":
                        self.assert_leading_unsqueeze(
                            source, result, shape, stride, offset
                        )
                    else:
                        self.assert_trailing_unsqueeze(
                            source, result, shape, stride, offset
                        )
                    self.assertTrue(result.requires_grad)
                    self.assertFalse(result.is_leaf)
                    self.assertEqual(result.output_nr, 0)
                    result.sum().backward()
                    np.testing.assert_array_equal(
                        np.asarray(leaf.grad), self.expected_gradient(case)
                    )

                with self.subTest(direction=direction, case=case, mode="no_grad"):
                    leaf, source = self.make_autograd_case(case)
                    with torch.no_grad():
                        result = call(source)
                    if direction == "leading":
                        self.assert_leading_unsqueeze(
                            source, result, shape, stride, offset
                        )
                    else:
                        self.assert_trailing_unsqueeze(
                            source, result, shape, stride, offset
                        )
                    self.assertTrue(result.requires_grad)
                    self.assertTrue(result.is_leaf)
                    self.assertEqual(result.output_nr, 0)
                    self.assertIsNone(leaf.grad)

    def test_newaxis_indexing_still_uses_the_same_edge_views(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(tensor.unsqueeze(0).is_set_to(tensor[None]))
        self.assertTrue(torch.unsqueeze(tensor, -1).is_set_to(tensor[..., torch.newaxis]))

    def test_binding_errors_and_deliberate_unsupported_boundaries(self):
        tensor = torch.zeros((2, 3))
        scalar = torch.tensor(1.0)
        cases = (
            (
                lambda: tensor.unsqueeze(),
                TypeError,
                'unsqueeze() missing 1 required positional arguments: "dim"',
            ),
            (
                lambda: tensor.unsqueeze(0, 0),
                TypeError,
                "unsqueeze() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.unsqueeze(0, dim=0),
                TypeError,
                "unsqueeze() got multiple values for argument 'dim'",
            ),
            (
                lambda: tensor.unsqueeze(extra=0),
                TypeError,
                'unsqueeze() missing 1 required positional arguments: "dim"',
            ),
            (
                lambda: tensor.unsqueeze(None),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not NoneType",
            ),
            (
                lambda: tensor.unsqueeze([0]),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not list",
            ),
            (
                lambda: tensor.unsqueeze((0,)),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not tuple",
            ),
            (
                lambda: tensor.unsqueeze(True),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not bool",
            ),
            (
                lambda: tensor.unsqueeze(0, out=None),
                TypeError,
                "unsqueeze() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: tensor.unsqueeze(2**100),
                ValueError,
                "Overflow when unpacking long long",
            ),
            (
                lambda: tensor.unsqueeze(3),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got 3)",
            ),
            (
                lambda: tensor.unsqueeze(-4),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got -4)",
            ),
            (
                lambda: scalar.unsqueeze(1),
                IndexError,
                "Dimension out of range (expected to be in range of [-1, 0], but got 1)",
            ),
            (
                lambda: torch.unsqueeze(),
                TypeError,
                'unsqueeze() missing 2 required positional argument: "input", "dim"',
            ),
            (
                lambda: torch.unsqueeze(tensor),
                TypeError,
                'unsqueeze() missing 1 required positional arguments: "dim"',
            ),
            (
                lambda: torch.unsqueeze([], 0),
                TypeError,
                "unsqueeze(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.unsqueeze(tensor, [0]),
                TypeError,
                "unsqueeze(): argument 'dim' (position 2) must be int, not list",
            ),
            (
                lambda: torch.unsqueeze(tensor, (0,)),
                TypeError,
                "unsqueeze(): argument 'dim' (position 2) must be int, not tuple",
            ),
            (
                lambda: torch.unsqueeze(tensor, True),
                TypeError,
                "unsqueeze(): argument 'dim' (position 2) must be int, not bool",
            ),
            (
                lambda: torch.unsqueeze(tensor, 0, out=None),
                TypeError,
                "unsqueeze() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.unsqueeze(tensor, 2**100),
                ValueError,
                "Overflow when unpacking long long",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message), self.assertRaises(error_type) as raised:
                call()
            self.assertEqual(str(raised.exception), message)

        for call, message in (
            (lambda: tensor.unsqueeze(1), "Tensor.unsqueeze"),
            (lambda: tensor.unsqueeze(-2), "Tensor.unsqueeze"),
            (lambda: torch.unsqueeze(tensor, 1), "torch.unsqueeze"),
            (lambda: torch.unsqueeze(tensor, -2), "torch.unsqueeze"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                RuntimeError,
                rf"^{re.escape(message)} only supports leading and trailing dimensions$",
            ):
                call()

        self.assertEqual(torch.unsqueeze(tensor, np.int64(0)).shape, (1, 2, 3))
        self.assertEqual(tensor.unsqueeze(np.uint64(2)).shape, (2, 3, 1))

    def test_torch_function_overrides_and_modes_are_not_supported(self):
        tensor = torch.zeros((2, 3))
        calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return object()

        with self.assertRaisesRegex(
            TypeError,
            r"^unsqueeze\(\): argument 'input' \(position 1\) must be Tensor, not Override$",
        ):
            torch.unsqueeze(Override(), 0)
        self.assertEqual(calls, [])

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        for call, message in (
            (lambda: tensor.unsqueeze(0), "Tensor.unsqueeze"),
            (lambda: torch.unsqueeze(tensor, 0), "torch.unsqueeze"),
        ):
            mode = RecordingMode()
            with self.subTest(message=message):
                with mode, self.assertRaisesRegex(
                    NotImplementedError,
                    rf"^{re.escape(message)} does not support TorchFunctionMode$",
                ):
                    call()
                self.assertEqual(mode.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_callable_metadata_and_exports(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "unsqueeze")
        bound = tensor.unsqueeze
        function = torch.unsqueeze
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertEqual(descriptor.__name__, "unsqueeze")
        self.assertEqual(descriptor.__qualname__, "TensorBase.unsqueeze")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertEqual(bound.__qualname__, "Tensor.unsqueeze")
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertIsNone(bound.__text_signature__)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "unsqueeze")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.unsqueeze")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(torch.__all__.count("unsqueeze"), 1)
        self.assertTrue(wildcard_namespace["unsqueeze"] is function)
        self.assertTrue(owner.unsqueeze is function)
        self.assertTrue(owner is torch._C._VariableFunctionsClass)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        self.assertTrue(
            all(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            )
        )


if __name__ == "__main__":
    unittest.main()
