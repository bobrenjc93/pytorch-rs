import inspect
import types
import unittest

import numpy as np

import torch_rs as torch


METHOD_DOC = "\nunbind(dim=0) -> seq\n\nSee :func:`torch.unbind`\n"


def offset_noncontiguous_source(*, requires_grad=False):
    values = [float(value) for value in range(48)]
    return torch.tensor(values, requires_grad=requires_grad).reshape(2, 2, 3, 4)[
        1
    ].transpose(0, 1)


class TensorUnbindTests(unittest.TestCase):
    def assert_rows_match_indexing(self, source, rows):
        self.assertIs(type(rows), tuple)
        self.assertEqual(len(rows), source.shape[0])
        for index, row in enumerate(rows):
            direct = source[index]
            with self.subTest(index=index):
                self.assertEqual(row.tolist(), direct.tolist())
                self.assertEqual(row.shape, direct.shape)
                self.assertEqual(row.stride(), direct.stride())
                self.assertEqual(row.storage_offset(), direct.storage_offset())
                self.assertEqual(row.data_ptr(), direct.data_ptr())
                self.assertTrue(row.is_set_to(direct))
                self.assertIs(row.dtype, source.dtype)
                self.assertEqual(row.device, source.device)

    def test_default_positional_and_keyword_calls_return_first_axis_views(self):
        source = offset_noncontiguous_source()
        self.assertEqual(source.shape, (3, 2, 4))
        self.assertEqual(source.stride(), (4, 12, 1))
        self.assertEqual(source.storage_offset(), 24)

        expected_values = (
            [[24.0, 25.0, 26.0, 27.0], [36.0, 37.0, 38.0, 39.0]],
            [[28.0, 29.0, 30.0, 31.0], [40.0, 41.0, 42.0, 43.0]],
            [[32.0, 33.0, 34.0, 35.0], [44.0, 45.0, 46.0, 47.0]],
        )
        calls = (
            ("default", lambda: source.unbind()),
            ("positional", lambda: source.unbind(0)),
            ("keyword", lambda: source.unbind(dim=0)),
            ("normalized negative", lambda: source.unbind(-3)),
        )
        for case, call in calls:
            with self.subTest(case=case):
                rows = call()
                self.assert_rows_match_indexing(source, rows)
                self.assertEqual(tuple(row.tolist() for row in rows), expected_values)
                self.assertEqual(
                    tuple(row.storage_offset() for row in rows), (24, 28, 32)
                )
                self.assertEqual(tuple(row.output_nr for row in rows), (0, 0, 0))

        vector = torch.tensor([1.0, 2.0, 3.0])
        scalars = vector.unbind(-1)
        self.assertEqual(tuple(value.tolist() for value in scalars), (1.0, 2.0, 3.0))
        self.assertTrue(
            all(value.is_set_to(vector[index]) for index, value in enumerate(scalars))
        )

    def test_autograd_output_numbers_no_grad_and_empty_shapes(self):
        leaf = torch.tensor([float(value) for value in range(48)], requires_grad=True)
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        rows = source.unbind()

        self.assertEqual(tuple(row.output_nr for row in rows), (0, 1, 2))
        self.assertTrue(all(row.requires_grad for row in rows))
        self.assertTrue(all(not row.is_leaf for row in rows))
        self.assert_rows_match_indexing(source, rows)

        (rows[0] * rows[2]).sum().backward()
        expected_gradient = [0.0] * 48
        for first, last in zip(
            (*range(24, 28), *range(36, 40)),
            (*range(32, 36), *range(44, 48)),
            strict=True,
        ):
            expected_gradient[first] = 4.0 * last
            expected_gradient[last] = 4.0 * first
        self.assertEqual(leaf.grad.tolist(), expected_gradient)

        no_grad_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        with torch.no_grad():
            no_grad_rows = no_grad_source.unbind()
        self.assertEqual(tuple(row.output_nr for row in no_grad_rows), (0, 0, 0))
        self.assertTrue(all(row.requires_grad for row in no_grad_rows))
        self.assertTrue(all(row.is_leaf for row in no_grad_rows))

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty_rows = empty.unbind()
        self.assertEqual(tuple(row.output_nr for row in empty_rows), (0, 1))
        self.assertEqual(tuple(row.shape for row in empty_rows), ((0, 3), (0, 3)))
        self.assertEqual(tuple(row.numel() for row in empty_rows), (0, 0))
        self.assert_rows_match_indexing(empty, empty_rows)
        empty_rows[1].sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])

        self.assertEqual(torch.zeros((0, 2), requires_grad=True).unbind(), ())

    def test_supported_call_errors_and_deliberate_surface_limits(self):
        tensor = torch.zeros((2, 3))
        scalar = torch.tensor(1.0)
        cases = (
            (
                lambda: tensor.unbind(0, 0),
                TypeError,
                "unbind() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: tensor.unbind(0, dim=0),
                TypeError,
                "unbind() got multiple values for argument 'dim'",
            ),
            (
                lambda: tensor.unbind(extra=0),
                TypeError,
                "unbind() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.unbind(None),
                TypeError,
                "unbind(): argument 'dim' (position 1) must be int, not NoneType",
            ),
            (
                lambda: tensor.unbind(0.0),
                TypeError,
                "unbind(): argument 'dim' (position 1) must be int, not float",
            ),
            (
                lambda: tensor.unbind(True),
                TypeError,
                "unbind(): argument 'dim' (position 1) must be int, not bool",
            ),
            (
                lambda: tensor.unbind(dim="0"),
                TypeError,
                "unbind(): argument 'dim' must be int, not str",
            ),
            (
                lambda: tensor.unbind("0", extra=True),
                TypeError,
                "unbind(): argument 'dim' (position 1) must be int, not str",
            ),
            (
                lambda: scalar.unbind(),
                IndexError,
                "Dimension specified as 0 but tensor has no dimensions",
            ),
            (
                lambda: scalar.unbind(0),
                IndexError,
                "Dimension specified as 0 but tensor has no dimensions",
            ),
            (
                lambda: scalar.unbind(-1),
                IndexError,
                "Dimension specified as 0 but tensor has no dimensions",
            ),
            (
                lambda: scalar.unbind(1),
                IndexError,
                "Dimension out of range (expected to be in range of [-1, 0], but got 1)",
            ),
            (
                lambda: scalar.unbind(-2),
                IndexError,
                "Dimension out of range (expected to be in range of [-1, 0], but got -2)",
            ),
            (
                lambda: tensor.unbind(2),
                IndexError,
                "Dimension out of range (expected to be in range of [-2, 1], but got 2)",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        with self.assertRaisesRegex(ValueError, "^Overflow when unpacking long long$"):
            tensor.unbind(2**100)
        self.assertEqual(len(tensor.unbind(np.int64(0))), 2)

        with self.assertRaisesRegex(
            RuntimeError, "^Tensor\\.unbind only supports dimension 0$"
        ):
            tensor.unbind(1)

        self.assertFalse(hasattr(torch, "unbind"))
        self.assertNotIn("unbind", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "chunk"))
        self.assertFalse(hasattr(torch, "chunk"))
        self.assertNotIn("chunk", torch.__all__)

    def test_tensorbase_descriptor_metadata_and_unbound_calls(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "unbind")
        bound = tensor.unbind

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "unbind")
        self.assertEqual(descriptor.__qualname__, "TensorBase.unbind")
        self.assertEqual(bound.__name__, "unbind")
        self.assertEqual(bound.__qualname__, "Tensor.unbind")
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
            "<method 'unbind' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.unbind, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        for rows in (
            descriptor(tensor),
            descriptor(tensor, 0),
            descriptor(tensor, dim=0),
        ):
            self.assert_rows_match_indexing(tensor, rows)

        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.unbind() needs an argument",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.unbind() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'unbind' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "unbind")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        cases = (
            ("default", lambda: tensor.unbind(), (tensor,), None),
            ("positional", lambda: tensor.unbind(0), (tensor, 0), None),
            ("keyword", lambda: tensor.unbind(dim=0), (tensor,), {"dim": 0}),
            ("unsupported replacement", lambda: tensor.unbind(1), (tensor, 1), None),
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
            tensor.unbind("0")
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
                forwarded = tensor.unbind(dim=0)
        self.assert_rows_match_indexing(tensor, forwarded)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor,))
            self.assertEqual(kwargs, {"dim": 0})

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            with DecliningMode():
                tensor.unbind()
        self.assertTrue(
            str(raised.exception).startswith(
                "Multiple dispatch failed for 'torch.Tensor.unbind'; "
                "all __torch_function__ handlers returned NotImplemented:"
            )
        )
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)


if __name__ == "__main__":
    unittest.main()
