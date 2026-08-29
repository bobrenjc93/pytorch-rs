import gc
import inspect
import pickle
import sys
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


class TensorUnsqueezeTests(unittest.TestCase):
    def layout_cases(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        return (
            ("scalar", torch.tensor(-0.0), np.asarray(-0.0, dtype=np.float32)),
            ("empty", torch.zeros((2, 0, 3)), np.zeros((2, 0, 3), dtype=np.float32)),
            ("offset", base[1], values[1]),
            ("noncontiguous", base.transpose(0, 3)[1], values.transpose(3, 1, 2, 0)[1]),
        )

    def assert_unsqueeze_view(self, source, result, expected_values, dim):
        rank = len(source.shape)
        axis = dim if dim >= 0 else dim + rank + 1
        expected_shape = list(source.shape)
        expected_shape.insert(axis, 1)
        expected_stride = list(source.stride())
        expected_stride.insert(
            axis, 1 if axis == rank else source.stride()[axis] * source.shape[axis]
        )

        self.assertIsNot(result, source)
        self.assertEqual(result.shape, tuple(expected_shape))
        self.assertEqual(result.stride(), tuple(expected_stride))
        self.assertEqual(result.storage_offset(), source.storage_offset())
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertFalse(result.is_set_to(source))
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        np.testing.assert_array_equal(
            np.asarray(result), np.expand_dims(expected_values, axis=axis)
        )

    def test_method_and_top_level_insert_every_dimension_as_native_views(self):
        for case, source, expected_values in self.layout_cases():
            rank = len(source.shape)
            for axis in range(rank + 1):
                dims = (axis, axis - rank - 1)
                calls = (
                    ("method positional", lambda dim=dims[0]: source.unsqueeze(dim)),
                    ("method dim keyword", lambda dim=dims[1]: source.unsqueeze(dim=dim)),
                    ("method axis keyword", lambda dim=dims[0]: source.unsqueeze(axis=dim)),
                    ("top positional", lambda dim=dims[0]: torch.unsqueeze(source, dim)),
                    (
                        "top dim keyword",
                        lambda dim=dims[1]: torch.unsqueeze(input=source, dim=dim),
                    ),
                    (
                        "top legacy aliases",
                        lambda dim=dims[0]: torch.unsqueeze(x=source, axis=dim),
                    ),
                )
                for spelling, call in calls:
                    with self.subTest(case=case, axis=axis, spelling=spelling):
                        result = call()
                        self.assert_unsqueeze_view(
                            source, result, expected_values, dims[0]
                        )

        scalar = torch.tensor(-0.0).unsqueeze(0)
        self.assertEqual(np.asarray(scalar).view(np.uint32).item(), 0x8000_0000)

    def test_autograd_and_no_grad_use_the_view_path(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=6, dtype=np.float32).reshape(3, 1, 2)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        source = (leaf * 2.0).transpose(0, 2)[1]
        view = torch.unsqueeze(source, dim=1)

        self.assertTrue(view.requires_grad)
        self.assertFalse(view.is_leaf)
        self.assertEqual(view.output_nr, 0)
        self.assertEqual(view.shape, (3, 1, 2))
        self.assertEqual(view.stride(), (4, 24, 12))
        self.assertEqual(view.data_ptr(), source.data_ptr())

        (view * torch.tensor(weights.tolist())).sum().backward()
        expected_gradient = np.zeros_like(values)
        expected_gradient[:, :, 1] = 2.0 * weights[:, 0, :].T
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected_gradient)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.unsqueeze(empty, -2).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(empty.grad), np.zeros((2, 0, 3), dtype=np.float32)
        )

        no_grad_source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with torch.no_grad():
            untracked = no_grad_source.unsqueeze(-1)
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.output_nr, 0)
        self.assertEqual(untracked.shape, (2, 2, 1))
        self.assertEqual(untracked.stride(), (2, 1, 1))
        self.assertEqual(untracked.data_ptr(), no_grad_source.data_ptr())
        self.assertIsNone(no_grad_source.grad)

    def test_storage_and_autograd_survive_source_lifetime(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retained_view():
            source = torch.tensor(values.tolist()).transpose(0, 2)[1]
            return source.unsqueeze(1)

        surviving = retained_view()
        gc.collect()
        self.assertEqual(surviving.shape, (3, 1, 2))
        self.assertEqual(surviving.stride(), (4, 24, 12))
        self.assertEqual(surviving.storage_offset(), 1)
        np.testing.assert_array_equal(
            np.asarray(surviving), np.expand_dims(values[:, :, 1].T, axis=1)
        )

        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_autograd_view():
            source = (leaf * 2.0).transpose(0, 2)[1]
            return torch.unsqueeze(source, -2)

        tracked = retained_autograd_view()
        gc.collect()
        weights = torch.tensor(
            [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]
        )
        (tracked * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[:, :, 1] = 2.0 * np.asarray(weights)[:, 0, :].T
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    @unittest.skipUnless(
        sys.maxsize == (1 << 63) - 1,
        "signed 64-bit stride wrapping requires a 64-bit Python build",
    )
    def test_extreme_empty_stride_boundaries_match_native_newaxis_path(self):
        non_concrete = torch.zeros((0,)).reshape((1 << 62, 0, 2))
        with self.assertRaisesRegex(
            RuntimeError,
            "SymIntArrayRef expected to contain only concrete integers",
        ):
            non_concrete.unsqueeze(0)

        negative_boundary = torch.zeros((0,)).reshape((1 << 62, 0, 3))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^as_strided: Negative strides are not supported at the moment, "
            r"got strides: \[-4611686018427387904, 3, 3, 1\]$",
        ):
            torch.unsqueeze(negative_boundary, dim=0)

        middle_zero = torch.zeros((0,)).reshape((2, sys.maxsize, 0, 3))
        middle = middle_zero.unsqueeze(2)
        self.assertEqual(middle.shape, (2, sys.maxsize, 1, 0, 3))
        self.assertEqual(middle.stride(), (sys.maxsize - 2, 3, 0, 3, 1))
        self.assertEqual(middle.data_ptr(), middle_zero.data_ptr())

    def test_dimension_types_bindings_and_boundaries_match_pytorch_2_13(self):
        class IntegerSubclass(int):
            pass

        tensor = torch.zeros((2, 3))
        self.assertEqual(tensor.unsqueeze(IntegerSubclass(1)).shape, (2, 1, 3))
        self.assertEqual(torch.unsqueeze(tensor, np.int64(-1)).shape, (2, 3, 1))

        cases = (
            (
                lambda: tensor.unsqueeze(),
                TypeError,
                'unsqueeze() missing 1 required positional arguments: "dim"',
            ),
            (
                lambda: tensor.unsqueeze(0, 1),
                TypeError,
                "unsqueeze() takes 1 positional argument but 2 were given",
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
                lambda: torch.unsqueeze(tensor, 0, 1),
                TypeError,
                "unsqueeze() takes 2 positional arguments but 3 were given",
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
                lambda: torch.tensor(1.0).unsqueeze(1),
                IndexError,
                "Dimension out of range (expected to be in range of [-1, 0], but got 1)",
            ),
            (
                lambda: tensor.unsqueeze(True),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not bool",
            ),
            (
                lambda: torch.unsqueeze(tensor, np.bool_(True)),
                TypeError,
                "unsqueeze(): argument 'dim' (position 2) must be int, not numpy.bool",
            ),
            (
                lambda: tensor.unsqueeze(1.5),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not float",
            ),
            (
                lambda: torch.unsqueeze(input=tensor, dim="1"),
                TypeError,
                "unsqueeze(): argument 'dim' must be int, not str",
            ),
            (
                lambda: tensor.unsqueeze((0,)),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not tuple",
            ),
            (
                lambda: tensor.unsqueeze(2**100),
                ValueError,
                "Overflow when unpacking long long",
            ),
            (
                lambda: torch.unsqueeze(1, 0),
                TypeError,
                "unsqueeze(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.unsqueeze(input=[], dim=0),
                TypeError,
                "unsqueeze(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: tensor.unsqueeze(0, dim=1),
                TypeError,
                "unsqueeze() got multiple values for argument 'dim'",
            ),
            (
                lambda: torch.unsqueeze(tensor, 0, dim=1),
                TypeError,
                "unsqueeze() got multiple values for argument 'dim'",
            ),
            (
                lambda: tensor.unsqueeze(0, out=tensor),
                TypeError,
                "unsqueeze() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.unsqueeze(tensor, 0, out=tensor),
                TypeError,
                "unsqueeze() got an unexpected keyword argument 'out'",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_callable_metadata_exports_and_in_place_boundary(self):
        function = torch.unsqueeze
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "unsqueeze")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.unsqueeze")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        descriptor = inspect.getattr_static(torch.Tensor, "unsqueeze")
        tensor = torch.zeros((2, 3))
        bound = tensor.unsqueeze
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "unsqueeze")
        self.assertEqual(descriptor.__qualname__, "TensorBase.unsqueeze")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertIsNone(descriptor.__text_signature__)
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)
        with self.assertRaises(ValueError):
            inspect.signature(bound)
        self.assertEqual(descriptor(tensor, 1).shape, (2, 1, 3))
        with self.assertRaisesRegex(
            TypeError, r"^unbound method TensorBase.unsqueeze\(\) needs an argument$"
        ):
            descriptor()

        owner = function.__reduce__()[1][0]
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.unsqueeze, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)

        self.assertEqual(torch.__all__.count("unsqueeze"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["unsqueeze"], function)
        self.assertFalse(hasattr(torch, "unsqueeze_"))
        self.assertFalse(hasattr(torch.Tensor, "unsqueeze_"))
        with self.assertRaises(AttributeError):
            tensor.unsqueeze_(0)

    def test_unsupported_boundaries_remain_explicit(self):
        tensor = torch.ones((2, 3))
        leading_mixed = (
            (None,),
            (None, None),
            (None, 0),
            (None, Ellipsis),
            (slice(None), None),
            (0, None),
        )
        for index in leading_mixed:
            with self.subTest(index=repr(index)):
                with self.assertRaisesRegex(IndexError, "only integers"):
                    tensor[index]

        repeated_or_extended = (
            (Ellipsis, None, None),
            (Ellipsis, None, 0),
            (Ellipsis, 0, None),
        )
        for index in repeated_or_extended:
            with self.subTest(index=repr(index)):
                with self.assertRaises(IndexError):
                    tensor[index]

        class OverrideOnly:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return object()

        with self.assertRaisesRegex(
            TypeError,
            "unsqueeze\\(\\): argument 'input' \\(position 1\\) must be Tensor, not",
        ):
            torch.unsqueeze(OverrideOnly(), 0)

        with self.assertRaisesRegex(
            TypeError,
            "unsqueeze\\(\\): argument 'input' must be Tensor, not",
        ):
            torch.unsqueeze(input=OverrideOnly(), dim=0)

    def test_torch_function_mode_observes_valid_exact_native_calls(self):
        marker = object()
        tensor = torch.zeros((2, 3))

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.unsqueeze(1)
        self.assertIs(result, marker)
        self.assertEqual(
            mode.calls,
            [
                (
                    inspect.getattr_static(torch.Tensor, "unsqueeze"),
                    (),
                    (tensor, 1),
                    None,
                )
            ],
        )

        mode.calls.clear()
        with mode:
            result = torch.unsqueeze(input=tensor, dim=-1)
        self.assertIs(result, marker)
        self.assertEqual(mode.calls, [(torch.unsqueeze, (), (), {"input": tensor, "dim": -1})])

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            self.assertEqual(tensor.unsqueeze(axis=1).shape, (2, 1, 3))
        with ForwardingMode():
            self.assertEqual(torch.unsqueeze(x=tensor, axis=-1).shape, (2, 3, 1))


if __name__ == "__main__":
    unittest.main()
