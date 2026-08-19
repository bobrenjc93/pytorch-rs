import inspect
import re
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
    def layout_cases(self):
        return (
            ("scalar", torch.tensor(3.0)),
            ("vector", torch.tensor([1.0, 2.0, 3.0])),
            ("matrix", torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
            ("offset-noncontiguous", offset_noncontiguous_source()),
            ("empty", torch.zeros((2, 0, 3))),
            ("empty-offset", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
        )

    def assert_leading_alias(self, result, source):
        leading_stride = (
            1 if source.dim() == 0 else source.shape[0] * source.stride()[0]
        )
        self.assertIsNot(result, source)
        self.assertEqual(result.tolist(), [source.tolist()])
        self.assertEqual(result.shape, (1, *source.shape))
        self.assertEqual(result.stride(), (leading_stride, *source.stride()))
        self.assertEqual(result.storage_offset(), source.storage_offset())
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        self.assertTrue(result.is_set_to(source.unsqueeze(0)))

    def test_positional_keyword_and_equivalent_negative_forms_are_views(self):
        for case, source in self.layout_cases():
            equivalent_negative = -(source.dim() + 1)
            calls = (
                ("positional", lambda source=source: source.unsqueeze(0)),
                ("keyword", lambda source=source: source.unsqueeze(dim=0)),
                (
                    "negative",
                    lambda source=source, dim=equivalent_negative: source.unsqueeze(dim),
                ),
                (
                    "negative-keyword",
                    lambda source=source, dim=equivalent_negative: source.unsqueeze(
                        dim=dim
                    ),
                ),
            )
            for form, call in calls:
                with self.subTest(case=case, form=form):
                    self.assert_leading_alias(call(), source)

        source = offset_noncontiguous_source()
        result = source.unsqueeze(-4)
        self.assertEqual(result.shape, (1, 3, 2, 4))
        self.assertEqual(result.stride(), (12, 4, 12, 1))
        self.assertEqual(result.storage_offset(), 24)

        empty_offset = torch.zeros((2, 0, 3)).transpose(0, 2)[1].unsqueeze(0)
        self.assertEqual(empty_offset.shape, (1, 0, 2))
        self.assertEqual(empty_offset.stride(), (0, 3, 3))
        self.assertEqual(empty_offset.storage_offset(), 1)
        self.assertEqual(empty_offset.numel(), 0)

    def test_extreme_empty_leading_stride_uses_signed_wrapping(self):
        maximum = sys.maxsize
        source = torch.zeros((maximum, 0, maximum))
        result = source.unsqueeze(0)

        self.assertEqual(source.stride(), (maximum, maximum, 1))
        self.assertEqual(result.shape, (1, maximum, 0, maximum))
        self.assertEqual(result.stride(), (1, maximum, maximum, 1))
        self.assertEqual(result.storage_offset(), 0)
        self.assertEqual(result.numel(), 0)
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertTrue(result.is_set_to(source.unsqueeze(-4)))

    def test_autograd_no_grad_and_empty_gradients_use_the_view_engine(self):
        leaf = torch.tensor([float(value) for value in range(48)], requires_grad=True)
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        result = source.unsqueeze(-4)

        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_leaf)
        self.assertEqual(result.output_nr, 0)
        self.assertEqual(result.shape, (1, 3, 2, 4))
        self.assertEqual(result.stride(), (12, 4, 12, 1))
        self.assertEqual(result.storage_offset(), 24)
        self.assertTrue(result.is_set_to(source.view((1, 3, 2, 4))))

        (result * 3.0).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [0.0] * 24 + [6.0] * 24)

        no_grad_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with torch.no_grad():
            untracked = no_grad_source.unsqueeze(dim=0)
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.output_nr, 0)
        self.assertTrue(untracked.is_set_to(no_grad_source.view((1, 2, 2))))

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty_result = empty.unsqueeze(0)
        self.assertEqual(empty_result.shape, (1, 2, 0, 3))
        self.assertEqual(empty_result.stride(), (6, 3, 3, 1))
        empty_result.sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])

    def test_integer_conversion_errors_and_deliberate_surface_limits(self):
        tensor = torch.zeros((2, 3))

        class IntegerSubclass(int):
            pass

        class IndexOnly:
            def __index__(self):
                return 0

        self.assertEqual(tensor.unsqueeze(IntegerSubclass(0)).shape, (1, 2, 3))
        self.assertEqual(tensor.unsqueeze(np.int64(-3)).shape, (1, 2, 3))
        self.assertEqual(tensor.unsqueeze(np.uint32(0)).shape, (1, 2, 3))

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
                lambda: tensor.unsqueeze(0, extra=True),
                TypeError,
                "unsqueeze() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.unsqueeze(None),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not NoneType",
            ),
            (
                lambda: tensor.unsqueeze(0.0),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not float",
            ),
            (
                lambda: tensor.unsqueeze(True),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not bool",
            ),
            (
                lambda: tensor.unsqueeze(dim="0"),
                TypeError,
                "unsqueeze(): argument 'dim' must be int, not str",
            ),
            (
                lambda: tensor.unsqueeze(IndexOnly()),
                TypeError,
                "unsqueeze(): argument 'dim' (position 1) must be int, not IndexOnly",
            ),
            (
                lambda: tensor.unsqueeze(None, extra=True),
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
            (
                lambda: torch.tensor(1.0).unsqueeze(-2),
                IndexError,
                "Dimension out of range (expected to be in range of [-1, 0], but got -2)",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        with self.assertRaisesRegex(ValueError, "^Overflow when unpacking long long$"):
            tensor.unsqueeze(2**100)

        for dimension in (1, 2, -1, -2):
            with self.subTest(unsupported_dimension=dimension):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^Tensor\\.unsqueeze only supports dimension 0$",
                ):
                    tensor.unsqueeze(dimension)

        self.assertFalse(hasattr(torch, "unsqueeze"))
        self.assertNotIn("unsqueeze", torch.__all__)

    def test_tensorbase_descriptor_metadata_and_unbound_behavior(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "unsqueeze")
        bound = tensor.unsqueeze

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "unsqueeze")
        self.assertEqual(descriptor.__qualname__, "TensorBase.unsqueeze")
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
        self.assertTrue(descriptor(tensor, 0).is_set_to(tensor.unsqueeze(0)))
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.unsqueeze() needs an argument",
            ),
            (
                lambda: descriptor(1, 0),
                "descriptor 'unsqueeze' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor, dim=0),
                "unbound method TensorBase.unsqueeze() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_observe_original_calls_and_forward(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "unsqueeze")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        positional = RecordingMode(marker)
        with positional:
            positional_result = tensor.unsqueeze(0)
        self.assertIs(positional_result, marker)
        function, dispatch_types, args, kwargs = positional.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor, 0))
        self.assertIsNone(kwargs)

        keyword = RecordingMode(marker)
        with keyword:
            keyword_result = tensor.unsqueeze(dim=-3)
        self.assertIs(keyword_result, marker)
        function, dispatch_types, args, kwargs = keyword.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor,))
        self.assertEqual(kwargs, {"dim": -3})

        deferred = RecordingMode(marker)
        with deferred:
            deferred_result = tensor.unsqueeze(2**100)
        self.assertIs(deferred_result, marker)
        self.assertEqual(len(deferred.calls), 1)

        nonleading = RecordingMode(marker)
        with nonleading:
            nonleading_result = tensor.unsqueeze(1)
        self.assertIs(nonleading_result, marker)
        self.assertEqual(len(nonleading.calls), 1)

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                tensor.unsqueeze(None)
        self.assertEqual(invalid.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.unsqueeze(dim=-3)
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(forwarded.is_set_to(tensor.unsqueeze(0)))

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaises(TypeError) as raised:
            with lower:
                with declining:
                    tensor.unsqueeze(0)
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^Multiple dispatch failed for 'torch\.Tensor\.unsqueeze'; all "
                r"__torch_function__ handlers returned NotImplemented:\n\n"
                r"  - mode object <.*RecordingMode object at 0x[0-9a-f]+>\n\n"
                r"For more information, try re-running with "
                r"TORCH_LOGS=not_implemented$"
            ),
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])


if __name__ == "__main__":
    unittest.main()
