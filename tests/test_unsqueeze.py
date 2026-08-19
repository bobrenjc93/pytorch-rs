import inspect
import sys
import types
import unittest

import numpy as np

import torch_rs as torch


METHOD_DOC = "\nunsqueeze(dim) -> Tensor\n\nSee :func:`torch.unsqueeze`\n"


def offset_noncontiguous_source(*, requires_grad=False):
    values = [float(value) for value in range(48)]
    return torch.tensor(values, requires_grad=requires_grad).reshape(2, 2, 3, 4)[
        1
    ].transpose(0, 1)


class TensorUnsqueezeTests(unittest.TestCase):
    def assert_same_unsqueeze_view(self, source, result):
        repeated = source.unsqueeze(0)
        self.assertEqual(result.tolist(), repeated.tolist())
        self.assertEqual(result.shape, repeated.shape)
        self.assertEqual(result.stride(), repeated.stride())
        self.assertEqual(result.storage_offset(), source.storage_offset())
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertTrue(result.is_set_to(repeated))
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        self.assertEqual(result.layout, source.layout)

    def test_positional_keyword_and_negative_leading_axis_views(self):
        source = offset_noncontiguous_source()
        self.assertEqual(source.shape, (3, 2, 4))
        self.assertEqual(source.stride(), (4, 12, 1))
        self.assertEqual(source.storage_offset(), 24)

        calls = (
            ("positional", lambda: source.unsqueeze(0)),
            ("keyword", lambda: source.unsqueeze(dim=0)),
            ("negative", lambda: source.unsqueeze(-4)),
            ("negative keyword", lambda: source.unsqueeze(dim=-4)),
        )
        for case, call in calls:
            with self.subTest(case=case):
                result = call()
                self.assert_same_unsqueeze_view(source, result)
                self.assertEqual(result.shape, (1, 3, 2, 4))
                self.assertEqual(result.stride(), (12, 4, 12, 1))
                self.assertEqual(
                    result.tolist(),
                    [
                        [
                            [
                                [24.0, 25.0, 26.0, 27.0],
                                [36.0, 37.0, 38.0, 39.0],
                            ],
                            [
                                [28.0, 29.0, 30.0, 31.0],
                                [40.0, 41.0, 42.0, 43.0],
                            ],
                            [
                                [32.0, 33.0, 34.0, 35.0],
                                [44.0, 45.0, 46.0, 47.0],
                            ],
                        ],
                    ],
                )

        scalar = source[2, 1, 3]
        for dimension in (0, -1):
            with self.subTest(scalar_dimension=dimension):
                result = scalar.unsqueeze(dimension)
                self.assertEqual(result.shape, (1,))
                self.assertEqual(result.stride(), (1,))
                self.assertEqual(result.storage_offset(), 47)
                self.assertEqual(result.tolist(), [47.0])
                self.assert_same_unsqueeze_view(scalar, result)

    def test_empty_and_strided_views_preserve_layout_and_offset(self):
        contiguous_empty = torch.zeros((0, 3))
        contiguous_result = contiguous_empty.unsqueeze(0)
        self.assertEqual(contiguous_result.shape, (1, 0, 3))
        self.assertEqual(contiguous_result.stride(), (0, 3, 1))
        self.assertEqual(contiguous_result.storage_offset(), 0)
        self.assertEqual(contiguous_result.data_ptr(), 0)
        self.assert_same_unsqueeze_view(contiguous_empty, contiguous_result)

        strided_empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        self.assertEqual(strided_empty.shape, (0, 2))
        self.assertEqual(strided_empty.stride(), (3, 3))
        self.assertEqual(strided_empty.storage_offset(), 1)
        strided_result = strided_empty.unsqueeze(-3)
        self.assertEqual(strided_result.shape, (1, 0, 2))
        self.assertEqual(strided_result.stride(), (0, 3, 3))
        self.assertEqual(strided_result.storage_offset(), 1)
        self.assertEqual(strided_result.data_ptr(), 0)
        self.assert_same_unsqueeze_view(strided_empty, strided_result)

    def test_extreme_empty_strides_use_signed_wrapping_arithmetic(self):
        even = torch.zeros((0,)).reshape((sys.maxsize, 0, 2))
        self.assertEqual(even.stride(), (2, 2, 1))
        for call in (lambda: even.unsqueeze(0), lambda: even.unsqueeze(-4)):
            with self.subTest(shape=even.shape, call=call):
                with self.assertRaises(RuntimeError) as raised:
                    call()
                self.assertEqual(
                    str(raised.exception),
                    "as_strided: Negative strides are not supported at the moment, "
                    "got strides: [-2, 2, 2, 1]",
                )

        odd = torch.zeros((0,)).reshape((sys.maxsize, 0, 3))
        self.assertEqual(odd.stride(), (3, 3, 1))
        for result in (odd.unsqueeze(0), odd.unsqueeze(dim=-4)):
            with self.subTest(shape=odd.shape, stride=result.stride()):
                self.assertEqual(result.shape, (1, sys.maxsize, 0, 3))
                self.assertEqual(result.stride(), (sys.maxsize - 2, 3, 3, 1))
                self.assertEqual(result.storage_offset(), 0)
                self.assertEqual(result.data_ptr(), 0)
                self.assertTrue(result.is_set_to(odd.unsqueeze(0)))

    def test_autograd_repeated_backward_empty_and_no_grad(self):
        leaf = torch.tensor([float(value) for value in range(48)], requires_grad=True)
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        result = source.unsqueeze(-4)
        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_leaf)
        self.assertEqual(result.output_nr, 0)
        self.assert_same_unsqueeze_view(source, result)

        loss = (result * 3.0).sum()
        loss.backward()
        self.assertEqual(leaf.grad.tolist(), [0.0] * 24 + [6.0] * 24)

        empty_leaf = torch.zeros((2, 0, 3), requires_grad=True)
        empty_source = empty_leaf.transpose(0, 2)[1]
        empty_result = empty_source.unsqueeze(0)
        self.assertEqual(empty_result.shape, (1, 0, 2))
        self.assertEqual(empty_result.stride(), (0, 3, 3))
        self.assertEqual(empty_result.storage_offset(), 1)
        empty_result.sum().backward()
        self.assertEqual(empty_leaf.grad.shape, (2, 0, 3))
        self.assertEqual(empty_leaf.grad.tolist(), [[], []])

        no_grad_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with torch.no_grad():
            no_grad_result = no_grad_source.unsqueeze(dim=0)
        self.assertTrue(no_grad_result.requires_grad)
        self.assertTrue(no_grad_result.is_leaf)
        self.assertEqual(no_grad_result.output_nr, 0)
        self.assert_same_unsqueeze_view(no_grad_source, no_grad_result)
        (no_grad_result * no_grad_result).sum().backward()
        self.assertIsNone(no_grad_source.grad)
        self.assertIsNone(no_grad_result.grad)

    def test_binding_integer_conversion_and_dimension_errors(self):
        tensor = torch.zeros((2, 3))

        class IntegerSubclass(int):
            pass

        self.assertEqual(tensor.unsqueeze(IntegerSubclass(0)).shape, (1, 2, 3))
        self.assertEqual(tensor.unsqueeze(np.int64(-3)).shape, (1, 2, 3))

        calls = []

        class IndexOnly:
            def __index__(self):
                calls.append("index")
                return 0

        with self.assertRaises(TypeError) as raised:
            tensor.unsqueeze(IndexOnly())
        self.assertEqual(
            str(raised.exception),
            "unsqueeze(): argument 'dim' (position 1) must be int, not IndexOnly",
        )
        self.assertEqual(calls, [])

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
                lambda: tensor.unsqueeze(dim=0, extra=True),
                TypeError,
                "unsqueeze() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.unsqueeze("0", extra=True),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not str",
            ),
            (
                lambda: tensor.unsqueeze(dim="0", extra=True),
                TypeError,
                "unsqueeze(): argument 'dim' must be int, not str",
            ),
            (
                lambda: tensor.unsqueeze(True),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not bool",
            ),
            (
                lambda: tensor.unsqueeze(0.0),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not float",
            ),
            (
                lambda: tensor.unsqueeze(None),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not NoneType",
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
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        with self.assertRaisesRegex(ValueError, "^Overflow when unpacking long long$"):
            tensor.unsqueeze(2**100)

        for dimension in (1, 2, -2, -1):
            with self.subTest(unsupported_dimension=dimension):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^Tensor\\.unsqueeze only supports dimension 0$",
                ):
                    tensor.unsqueeze(dimension)

    def test_legacy_keyword_lookup_spoofing_and_nul_diagnostics(self):
        tensor = torch.zeros((2, 3))

        class PlainKeyword(str):
            pass

        class TrueKeyword(str):
            def __eq__(self, other):
                return True

            __hash__ = str.__hash__

        class FalseKeyword(str):
            def __eq__(self, other):
                return False

            __hash__ = str.__hash__

        class RaisingKeyword(str):
            def __eq__(self, other):
                raise RuntimeError("keyword equality failure")

            __hash__ = str.__hash__

        class MismatchedHashKeyword(str):
            def __eq__(self, other):
                return True

            def __hash__(self):
                return 0

        for keyword in (PlainKeyword("dim"), TrueKeyword("dim")):
            with self.subTest(keyword_type=type(keyword).__name__):
                self.assertEqual(tensor.unsqueeze(**{keyword: 0}).shape, (1, 2, 3))

        missing = 'unsqueeze() missing 1 required positional arguments: "dim"'
        for keyword in (
            FalseKeyword("dim"),
            RaisingKeyword("dim"),
            MismatchedHashKeyword("dim"),
            TrueKeyword("unexpected"),
        ):
            with self.subTest(keyword_type=type(keyword).__name__):
                with self.assertRaises(TypeError) as raised:
                    tensor.unsqueeze(**{keyword: 0})
                self.assertEqual(str(raised.exception), missing)

        cases = (
            (
                lambda: tensor.unsqueeze(0, **{FalseKeyword("dim"): 1}),
                TypeError,
                "unsqueeze() got an unexpected keyword argument 'dim'",
            ),
            (
                lambda: tensor.unsqueeze(0, **{TrueKeyword("unexpected"): 1}),
                TypeError,
                "unsqueeze() got multiple values for argument 'unexpected'",
            ),
            (
                lambda: tensor.unsqueeze(0, **{RaisingKeyword("dim"): 1}),
                RuntimeError,
                "keyword equality failure",
            ),
            (
                lambda: tensor.unsqueeze(
                    **{"dim": 0, TrueKeyword("unexpected"): 1}
                ),
                TypeError,
                "invalid keyword arguments",
            ),
            (
                lambda: tensor.unsqueeze(**{"dim": 0, FalseKeyword("dim"): 1}),
                TypeError,
                "unsqueeze() got an unexpected keyword argument 'dim'",
            ),
            (
                lambda: tensor.unsqueeze(0, **{"bad\0tail": 1}),
                TypeError,
                "unsqueeze() got an unexpected keyword argument 'bad",
            ),
            (
                lambda: tensor.unsqueeze(dim=0, **{"bad\0tail": 1}),
                TypeError,
                "unsqueeze() got an unexpected keyword argument 'bad",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_tensorbase_descriptor_metadata_and_unbound_calls(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "unsqueeze")
        bound = tensor.unsqueeze

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "unsqueeze")
        self.assertEqual(descriptor.__qualname__, "TensorBase.unsqueeze")
        self.assertEqual(bound.__name__, "unsqueeze")
        self.assertEqual(bound.__qualname__, "Tensor.unsqueeze")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertEqual(
            repr(descriptor),
            "<method 'unsqueeze' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.unsqueeze, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertNotIn("unsqueeze", torch.Tensor.__dict__)
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(descriptor(tensor, 0).shape, (1, 2, 3))
        self.assertEqual(descriptor(tensor, dim=-3).shape, (1, 2, 3))

        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.unsqueeze() needs an argument",
            ),
            (
                lambda: descriptor(self=tensor, dim=0),
                "unbound method TensorBase.unsqueeze() needs an argument",
            ),
            (
                lambda: descriptor(1, 0),
                "descriptor 'unsqueeze' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        self.assertFalse(hasattr(torch, "unsqueeze"))
        self.assertNotIn("unsqueeze", torch.__all__)

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "unsqueeze")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        cases = (
            ("positional", lambda: tensor.unsqueeze(0), (tensor, 0), None),
            ("keyword", lambda: tensor.unsqueeze(dim=0), (tensor,), {"dim": 0}),
            ("negative", lambda: tensor.unsqueeze(-3), (tensor, -3), None),
            ("unsupported", lambda: tensor.unsqueeze(1), (tensor, 1), None),
            ("out of range", lambda: tensor.unsqueeze(9), (tensor, 9), None),
            ("overflow", lambda: tensor.unsqueeze(2**100), (tensor, 2**100), None),
        )
        for case, call, expected_args, expected_kwargs in cases:
            mode = RecordingMode(marker)
            with self.subTest(case=case), mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

        invalid = RecordingMode(marker)
        with invalid, self.assertRaises(TypeError):
            tensor.unsqueeze("0")
        self.assertEqual(invalid.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append((self.label, func, dispatch_types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.unsqueeze(dim=-3)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor,))
            self.assertEqual(kwargs, {"dim": -3})
        self.assert_same_unsqueeze_view(tensor, forwarded)

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            with DecliningMode():
                tensor.unsqueeze(0)
        self.assertTrue(
            str(raised.exception).startswith(
                "Multiple dispatch failed for 'torch.Tensor.unsqueeze'; "
                "all __torch_function__ handlers returned NotImplemented:"
            )
        )
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)


if __name__ == "__main__":
    unittest.main()
