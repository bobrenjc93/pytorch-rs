import inspect
import pickle
import types
import unittest

import numpy as np

import torch_rs as torch


METHOD_DOC = "\nunbind(dim=0) -> seq\n\nSee :func:`torch.unbind`\n"
FUNCTION_DOC = (
    "\nunbind(input, dim=0) -> seq\n\n"
    "Removes a tensor dimension.\n\n"
    "Returns a tuple of all slices along a given dimension, already without it.\n\n"
    "Arguments:\n"
    "    input (Tensor): the tensor to unbind\n"
    "    dim (int): dimension to remove\n\n"
    "Example::\n\n"
    "    >>> torch.unbind(torch.tensor([[1, 2, 3],\n"
    "    >>>                            [4, 5, 6],\n"
    "    >>>                            [7, 8, 9]]))\n"
    "    (tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))\n"
)


def offset_noncontiguous_source(*, requires_grad=False):
    values = [float(value) for value in range(48)]
    return torch.tensor(values, requires_grad=requires_grad).reshape(2, 2, 3, 4)[
        1
    ].transpose(0, 1)


class TensorUnbindTests(unittest.TestCase):
    def assert_unbind_matches_select(self, source, outputs, dimension=0):
        axis = dimension if dimension >= 0 else dimension + len(source.shape)
        self.assertIs(type(outputs), tuple)
        self.assertEqual(len(outputs), source.shape[axis])
        for index, output in enumerate(outputs):
            direct = source.select(axis, index)
            with self.subTest(index=index):
                self.assertEqual(output.tolist(), direct.tolist())
                self.assertEqual(output.shape, direct.shape)
                self.assertEqual(output.stride(), direct.stride())
                self.assertEqual(output.storage_offset(), direct.storage_offset())
                self.assertEqual(output.data_ptr(), direct.data_ptr())
                self.assertTrue(output.is_set_to(direct))
                self.assertIs(output.dtype, source.dtype)
                self.assertEqual(output.device, source.device)

    def assert_rows_match_indexing(self, source, rows):
        self.assert_unbind_matches_select(source, rows)

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

    def test_arbitrary_dimension_views_match_select(self):
        contiguous = torch.tensor([float(value) for value in range(24)]).reshape(
            2, 3, 4
        )
        offset = torch.tensor([float(value) for value in range(120)]).reshape(
            2, 3, 4, 5
        )[1]
        noncontiguous = offset_noncontiguous_source()
        empty_middle = torch.zeros((2, 3, 0, 4))
        empty_unbound = torch.zeros((2, 0, 3))

        cases = (
            ("contiguous middle", contiguous, 1),
            ("contiguous trailing", contiguous, 2),
            ("offset middle", offset, 1),
            ("noncontiguous middle", noncontiguous, 1),
            ("negative trailing", noncontiguous, -1),
            ("empty retained dimension", empty_middle, 1),
            ("empty unbound dimension", empty_unbound, 1),
        )
        for case, source, dimension in cases:
            with self.subTest(case=case, surface="method"):
                self.assert_unbind_matches_select(
                    source, source.unbind(dimension), dimension
                )
            with self.subTest(case=case, surface="top-level"):
                self.assert_unbind_matches_select(
                    source, torch.unbind(source, dimension), dimension
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

        full_sum_leaf = torch.tensor(
            [float(value) for value in range(48)], requires_grad=True
        )
        full_sum_source = (full_sum_leaf * 2.0).reshape(2, 2, 3, 4)[
            1
        ].transpose(0, 1)
        columns = full_sum_source.unbind(1)
        self.assertEqual(tuple(column.output_nr for column in columns), (0, 1))
        self.assert_unbind_matches_select(full_sum_source, columns, 1)
        loss = columns[0].sum()
        for column in columns[1:]:
            loss = loss + column.sum()
        loss.backward()
        self.assertEqual(full_sum_leaf.grad.tolist(), [0.0] * 24 + [2.0] * 24)

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

        self.assertEqual(len(tensor.unbind(1)), 3)

        self.assertTrue(hasattr(torch, "unbind"))
        self.assertIn("unbind", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "chunk"))
        self.assertFalse(hasattr(torch, "chunk"))
        self.assertNotIn("chunk", torch.__all__)

    def test_top_level_forms_reuse_first_axis_view_engine(self):
        source = offset_noncontiguous_source()
        calls = (
            ("default", lambda: torch.unbind(source)),
            ("positional", lambda: torch.unbind(source, 0)),
            ("dimension keyword", lambda: torch.unbind(source, dim=0)),
            (
                "all keywords",
                lambda: torch.unbind(input=source, dim=0),
            ),
            (
                "reordered keywords",
                lambda: torch.unbind(dim=0, input=source),
            ),
            ("input alias x", lambda: torch.unbind(x=source, dim=0)),
            ("input alias a", lambda: torch.unbind(a=source)),
            ("input alias x1", lambda: torch.unbind(x1=source, dim=0)),
            ("normalized negative", lambda: torch.unbind(source, -3)),
        )
        expected = source.unbind()
        for case, call in calls:
            with self.subTest(case=case):
                rows = call()
                self.assert_rows_match_indexing(source, rows)
                self.assertEqual(
                    tuple(row.tolist() for row in rows),
                    tuple(row.tolist() for row in expected),
                )
                self.assertTrue(
                    all(row.is_set_to(expected[index]) for index, row in enumerate(rows))
                )

    def test_top_level_autograd_output_numbers_no_grad_and_empty_shapes(self):
        leaf = torch.tensor([float(value) for value in range(48)], requires_grad=True)
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        rows = torch.unbind(source)

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
            no_grad_rows = torch.unbind(input=no_grad_source, dim=0)
        self.assertEqual(tuple(row.output_nr for row in no_grad_rows), (0, 0, 0))
        self.assertTrue(all(row.requires_grad for row in no_grad_rows))
        self.assertTrue(all(row.is_leaf for row in no_grad_rows))

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty_rows = torch.unbind(empty, 0)
        self.assertEqual(tuple(row.output_nr for row in empty_rows), (0, 1))
        self.assertEqual(tuple(row.shape for row in empty_rows), ((0, 3), (0, 3)))
        empty_rows[1].sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])
        self.assertEqual(torch.unbind(torch.zeros((0, 2), requires_grad=True)), ())

        full_sum_leaf = torch.tensor(
            [float(value) for value in range(48)], requires_grad=True
        )
        full_sum_source = (full_sum_leaf * 2.0).reshape(2, 2, 3, 4)[
            1
        ].transpose(0, 1)
        columns = torch.unbind(full_sum_source, 1)
        self.assertEqual(tuple(column.output_nr for column in columns), (0, 1))
        self.assert_unbind_matches_select(full_sum_source, columns, 1)
        loss = columns[0].sum()
        for column in columns[1:]:
            loss = loss + column.sum()
        loss.backward()
        self.assertEqual(full_sum_leaf.grad.tolist(), [0.0] * 24 + [2.0] * 24)

    def test_top_level_binding_errors_and_deliberate_surface_limits(self):
        tensor = torch.zeros((2, 3))
        scalar = torch.tensor(1.0)
        cases = (
            (
                lambda: torch.unbind(),
                TypeError,
                'unbind() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.unbind(dim=0),
                TypeError,
                'unbind() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.unbind(tensor, 0, 0),
                TypeError,
                "unbind() takes from 1 to 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.unbind(tensor, input=tensor),
                TypeError,
                "unbind() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.unbind(tensor, 0, dim=0),
                TypeError,
                "unbind() got multiple values for argument 'dim'",
            ),
            (
                lambda: torch.unbind(tensor, extra=0),
                TypeError,
                "unbind() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.unbind(x=tensor, extra=0),
                TypeError,
                "unbind() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.unbind([], 0),
                TypeError,
                "unbind(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.unbind(input=[], dim=0),
                TypeError,
                "unbind(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.unbind(tensor, None),
                TypeError,
                "unbind(): argument 'dim' (position 2) must be int, not NoneType",
            ),
            (
                lambda: torch.unbind(tensor, 0.0),
                TypeError,
                "unbind(): argument 'dim' (position 2) must be int, not float",
            ),
            (
                lambda: torch.unbind(tensor, True),
                TypeError,
                "unbind(): argument 'dim' (position 2) must be int, not bool",
            ),
            (
                lambda: torch.unbind(tensor, dim="0"),
                TypeError,
                "unbind(): argument 'dim' must be int, not str",
            ),
            (
                lambda: torch.unbind(tensor, "0", extra=True),
                TypeError,
                "unbind(): argument 'dim' (position 2) must be int, not str",
            ),
            (
                lambda: torch.unbind(scalar),
                IndexError,
                "Dimension specified as 0 but tensor has no dimensions",
            ),
            (
                lambda: torch.unbind(scalar, -1),
                IndexError,
                "Dimension specified as 0 but tensor has no dimensions",
            ),
            (
                lambda: torch.unbind(tensor, 2),
                IndexError,
                "Dimension out of range (expected to be in range of [-2, 1], but got 2)",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message), self.assertRaises(error_type) as raised:
                call()
            self.assertEqual(str(raised.exception), message)

        with self.assertRaisesRegex(ValueError, "^Overflow when unpacking long long$"):
            torch.unbind(tensor, 2**100)
        self.assertEqual(len(torch.unbind(tensor, np.int64(0))), 2)
        self.assertEqual(len(torch.unbind(tensor, 1)), 3)
        self.assertEqual(len(torch.unbind(tensor, -1)), 3)

        self.assertFalse(hasattr(torch.Tensor, "chunk"))
        self.assertFalse(hasattr(torch, "chunk"))
        self.assertNotIn("chunk", torch.__all__)

    def test_top_level_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.zeros((2, 3))
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        cases = (
            ("default", lambda: torch.unbind(tensor), (tensor,), None),
            ("positional", lambda: torch.unbind(tensor, 0), (tensor, 0), None),
            (
                "dimension keyword",
                lambda: torch.unbind(tensor, dim=0),
                (tensor,),
                {"dim": 0},
            ),
            (
                "all keywords",
                lambda: torch.unbind(input=tensor, dim=0),
                (),
                {"input": tensor, "dim": 0},
            ),
            (
                "nonzero dimension replacement",
                lambda: torch.unbind(tensor, 1),
                (tensor, 1),
                None,
            ),
            (
                "overflow replacement",
                lambda: torch.unbind(tensor, 2**100),
                (tensor, 2**100),
                None,
            ),
        )
        for case, call, expected_args, expected_kwargs in cases:
            mode = RecordingMode(marker)
            with self.subTest(case=case), mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.unbind)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

        invalid = RecordingMode(marker)
        with invalid, self.assertRaises(TypeError):
            torch.unbind(tensor, "0")
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
                forwarded = torch.unbind(input=tensor, dim=0)
        self.assert_rows_match_indexing(tensor, forwarded)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, torch.unbind)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, ())
            self.assertEqual(kwargs, {"input": tensor, "dim": 0})

        declining = RecordingMode(NotImplemented)
        with self.assertRaises(TypeError) as raised:
            with declining:
                torch.unbind(tensor)
        self.assertTrue(
            str(raised.exception).startswith(
                "Multiple dispatch failed for 'torch.unbind'; "
                "all __torch_function__ handlers returned NotImplemented:"
            )
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

    def test_top_level_tensor_like_overrides_use_public_function(self):
        marker = object()
        calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                calls.append((func, dispatch_types, args, kwargs))
                return marker

        value = Override()
        cases = (
            (lambda: torch.unbind(value), (value,), None),
            (lambda: torch.unbind(value, 0), (value, 0), None),
            (
                lambda: torch.unbind(input=value, dim=0),
                (),
                {"input": value, "dim": 0},
            ),
            (lambda: torch.unbind(value, 1), (value, 1), None),
            (lambda: torch.unbind(value, 2**100), (value, 2**100), None),
        )
        for call, expected_args, expected_kwargs in cases:
            with self.subTest(args=expected_args, kwargs=expected_kwargs):
                self.assertIs(call(), marker)
            function, dispatch_types, args, kwargs = calls[-1]
            self.assertIs(function, torch.unbind)
            self.assertEqual(dispatch_types, (Override,))
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

        call_count = len(calls)
        with self.assertRaises(TypeError):
            torch.unbind(value, "0")
        self.assertEqual(len(calls), call_count)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.unbind(DecliningOverride())
        self.assertTrue(
            str(raised.exception).startswith(
                "Multiple dispatch failed for 'torch.unbind'; "
                "all __torch_function__ handlers returned NotImplemented:"
            )
        )

    def test_top_level_callable_metadata_documentation_and_exports(self):
        function = torch.unbind
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "unbind")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.unbind")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method unbind of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.unbind, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("unbind"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["unbind"], function)

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
            ("nonzero dimension replacement", lambda: tensor.unbind(1), (tensor, 1), None),
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
