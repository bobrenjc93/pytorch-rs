import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelCatReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.cat differentials require pinned PyTorch 2.13.0"
            )

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    @staticmethod
    def fresh_storage_observation(result, inputs):
        return tuple(
            (not result.is_set_to(input), result.data_ptr() != input.data_ptr())
            for input in inputs
            if input.numel()
        )

    @staticmethod
    def error(call):
        try:
            call()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("call unexpectedly succeeded")

    def test_list_tuple_empty_operands_order_metadata_and_storage_match_pytorch_2_13(
        self,
    ):
        actual_base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected_base = reference_torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=reference_torch.float32,
        )
        actual_offset = actual_base.transpose(0, 1)[1]
        expected_offset = expected_base.transpose(0, 1)[1]

        cases = (
            (
                "list dim 0",
                [actual_offset, torch.tensor([]), torch.tensor([-0.0, 7.5])],
                [
                    expected_offset,
                    reference_torch.tensor([], dtype=reference_torch.float32),
                    reference_torch.tensor([-0.0, 7.5], dtype=reference_torch.float32),
                ],
                0,
            ),
            (
                "tuple dim -1",
                (torch.tensor([]), torch.tensor([3.25]), torch.tensor([])),
                (
                    reference_torch.tensor([], dtype=reference_torch.float32),
                    reference_torch.tensor([3.25], dtype=reference_torch.float32),
                    reference_torch.tensor([], dtype=reference_torch.float32),
                ),
                -1,
            ),
            (
                "single input",
                [torch.tensor([11.0, 13.0])],
                [reference_torch.tensor([11.0, 13.0], dtype=reference_torch.float32)],
                0,
            ),
            (
                "keywords",
                [torch.tensor([8.0]), torch.tensor([]), torch.tensor([9.0])],
                [
                    reference_torch.tensor([8.0], dtype=reference_torch.float32),
                    reference_torch.tensor([], dtype=reference_torch.float32),
                    reference_torch.tensor([9.0], dtype=reference_torch.float32),
                ],
                0,
            ),
            (
                "axis alias",
                [torch.tensor([14.0]), torch.tensor([15.0])],
                [
                    reference_torch.tensor([14.0], dtype=reference_torch.float32),
                    reference_torch.tensor([15.0], dtype=reference_torch.float32),
                ],
                0,
            ),
        )
        for case, actual_inputs, expected_inputs, dimension in cases:
            with self.subTest(case=case):
                if case == "keywords":
                    actual = torch.cat(tensors=actual_inputs, dim=dimension, out=None)
                    expected = reference_torch.cat(
                        tensors=expected_inputs, dim=dimension, out=None
                    )
                elif case == "axis alias":
                    actual = torch.cat(actual_inputs, axis=dimension)
                    expected = reference_torch.cat(expected_inputs, axis=dimension)
                else:
                    actual = torch.cat(actual_inputs, dim=dimension)
                    expected = reference_torch.cat(expected_inputs, dim=dimension)
                self.assert_matches(actual, expected, case=case)
                self.assertEqual(
                    self.fresh_storage_observation(actual, actual_inputs),
                    self.fresh_storage_observation(expected, expected_inputs),
                )

    def test_rank_two_row_column_empty_and_noncontiguous_inputs_match_pytorch_2_13(
        self,
    ):
        actual_base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        expected_base = reference_torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=reference_torch.float32,
        )
        actual_offset = actual_base[1].transpose(0, 1)
        expected_offset = expected_base[1].transpose(0, 1)
        cases = (
            (
                "rows dim 0",
                [actual_offset, torch.zeros((0, 3)), torch.tensor([[-0.0, 100.0, 101.0]])],
                [
                    expected_offset,
                    reference_torch.zeros((0, 3), dtype=reference_torch.float32),
                    reference_torch.tensor(
                        [[-0.0, 100.0, 101.0]],
                        dtype=reference_torch.float32,
                    ),
                ],
                0,
            ),
            (
                "rows dim -2",
                (torch.zeros((0, 3)), actual_offset),
                (
                    reference_torch.zeros((0, 3), dtype=reference_torch.float32),
                    expected_offset,
                ),
                -2,
            ),
            (
                "columns dim 1",
                [
                    actual_offset,
                    torch.zeros((4, 0)),
                    torch.tensor([[-0.0], [200.0], [201.0], [202.0]]),
                ],
                [
                    expected_offset,
                    reference_torch.zeros((4, 0), dtype=reference_torch.float32),
                    reference_torch.tensor(
                        [[-0.0], [200.0], [201.0], [202.0]],
                        dtype=reference_torch.float32,
                    ),
                ],
                1,
            ),
            (
                "columns dim -1",
                (torch.zeros((4, 0)), actual_offset),
                (
                    reference_torch.zeros((4, 0), dtype=reference_torch.float32),
                    expected_offset,
                ),
                -1,
            ),
            (
                "empty columns",
                [torch.zeros((2, 0)), torch.zeros((2, 0))],
                [
                    reference_torch.zeros((2, 0), dtype=reference_torch.float32),
                    reference_torch.zeros((2, 0), dtype=reference_torch.float32),
                ],
                1,
            ),
            (
                "neutral empty first dim 0",
                [torch.tensor([]), actual_offset],
                [
                    reference_torch.tensor([], dtype=reference_torch.float32),
                    expected_offset,
                ],
                0,
            ),
            (
                "neutral empty middle dim 0",
                [
                    torch.tensor([[300.0, 301.0, 302.0]]),
                    torch.tensor([]),
                    torch.tensor([[-0.0, 100.0, 101.0]]),
                ],
                [
                    reference_torch.tensor(
                        [[300.0, 301.0, 302.0]],
                        dtype=reference_torch.float32,
                    ),
                    reference_torch.tensor([], dtype=reference_torch.float32),
                    reference_torch.tensor(
                        [[-0.0, 100.0, 101.0]],
                        dtype=reference_torch.float32,
                    ),
                ],
                0,
            ),
            (
                "neutral empty last dim 1",
                [actual_offset, torch.tensor([])],
                [
                    expected_offset,
                    reference_torch.tensor([], dtype=reference_torch.float32),
                ],
                1,
            ),
            (
                "single rank 2",
                [torch.tensor([[1.0, 2.0], [3.0, 4.0]])],
                [
                    reference_torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0]],
                        dtype=reference_torch.float32,
                    )
                ],
                -1,
            ),
            (
                "axis alias",
                [torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])],
                [
                    reference_torch.tensor(
                        [[1.0], [2.0]], dtype=reference_torch.float32
                    ),
                    reference_torch.tensor(
                        [[3.0], [4.0]], dtype=reference_torch.float32
                    ),
                ],
                1,
            ),
        )
        for case, actual_inputs, expected_inputs, dimension in cases:
            with self.subTest(case=case):
                if case == "axis alias":
                    actual = torch.cat(actual_inputs, axis=dimension)
                    expected = reference_torch.cat(expected_inputs, axis=dimension)
                else:
                    actual = torch.cat(actual_inputs, dim=dimension)
                    expected = reference_torch.cat(expected_inputs, dim=dimension)
                self.assert_matches(actual, expected, case=case)
                self.assertEqual(
                    self.fresh_storage_observation(actual, actual_inputs),
                    self.fresh_storage_observation(expected, expected_inputs),
                )

    def test_concat_alias_values_metadata_storage_and_exports_match_pytorch_2_13(
        self,
    ):
        for name in ("concat", "concatenate"):
            actual_function = getattr(torch, name)
            expected_function = getattr(reference_torch, name)
            actual_base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            expected_base = reference_torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=reference_torch.float32,
            )
            actual_offset = actual_base.transpose(0, 1)[1]
            expected_offset = expected_base.transpose(0, 1)[1]

            cases = (
                (
                    "list dim 0",
                    [actual_offset, torch.tensor([]), torch.tensor([-0.0, 7.5])],
                    [
                        expected_offset,
                        reference_torch.tensor([], dtype=reference_torch.float32),
                        reference_torch.tensor([-0.0, 7.5], dtype=reference_torch.float32),
                    ],
                    0,
                ),
                (
                    "tuple dim -1",
                    (torch.tensor([]), torch.tensor([3.25]), torch.tensor([])),
                    (
                        reference_torch.tensor([], dtype=reference_torch.float32),
                        reference_torch.tensor([3.25], dtype=reference_torch.float32),
                        reference_torch.tensor([], dtype=reference_torch.float32),
                    ),
                    -1,
                ),
                (
                    "keywords",
                    [torch.tensor([8.0]), torch.tensor([]), torch.tensor([9.0])],
                    [
                        reference_torch.tensor([8.0], dtype=reference_torch.float32),
                        reference_torch.tensor([], dtype=reference_torch.float32),
                        reference_torch.tensor([9.0], dtype=reference_torch.float32),
                    ],
                    0,
                ),
                (
                    "axis alias",
                    [torch.tensor([14.0]), torch.tensor([15.0])],
                    [
                        reference_torch.tensor([14.0], dtype=reference_torch.float32),
                        reference_torch.tensor([15.0], dtype=reference_torch.float32),
                    ],
                    0,
                ),
            )
            for case, actual_inputs, expected_inputs, dimension in cases:
                with self.subTest(alias=name, case=case):
                    if case == "keywords":
                        actual = actual_function(
                            tensors=actual_inputs,
                            dim=dimension,
                            out=None,
                        )
                        expected = expected_function(
                            tensors=expected_inputs,
                            dim=dimension,
                            out=None,
                        )
                    elif case == "axis alias":
                        actual = actual_function(actual_inputs, axis=dimension)
                        expected = expected_function(expected_inputs, axis=dimension)
                    else:
                        actual = actual_function(actual_inputs, dim=dimension)
                        expected = expected_function(expected_inputs, dim=dimension)
                    self.assert_matches(actual, expected, case=f"{name} {case}")
                    self.assertEqual(
                        self.fresh_storage_observation(actual, actual_inputs),
                        self.fresh_storage_observation(expected, expected_inputs),
                    )

            with self.subTest(alias=name, case="rank-2 axis 1"):
                actual_inputs = [
                    torch.tensor([[1.0], [2.0]]),
                    torch.tensor([]),
                    torch.tensor([[3.0], [4.0]]),
                ]
                expected_inputs = [
                    reference_torch.tensor(
                        [[1.0], [2.0]], dtype=reference_torch.float32
                    ),
                    reference_torch.tensor([], dtype=reference_torch.float32),
                    reference_torch.tensor(
                        [[3.0], [4.0]], dtype=reference_torch.float32
                    ),
                ]
                actual = actual_function(actual_inputs, axis=1)
                expected = expected_function(expected_inputs, axis=1)
                self.assert_matches(actual, expected, case=f"{name} rank-2")
                self.assertEqual(
                    self.fresh_storage_observation(actual, actual_inputs),
                    self.fresh_storage_observation(expected, expected_inputs),
                )

        def public_surface(module):
            return (
                tuple(
                    (
                        name,
                        hasattr(module, name),
                        getattr(module, name).__name__,
                        getattr(module, name).__qualname__,
                        getattr(module, name).__module__,
                        getattr(module._C._VariableFunctionsClass, name)
                        is getattr(module, name),
                        module.__all__.count(name),
                    )
                    for name in ("cat", "concat", "concatenate")
                ),
                module.cat is module.concat,
                module.cat is module.concatenate,
                module.concat is module.concatenate,
            )

        self.assertEqual(public_surface(torch), public_surface(reference_torch))

    def test_autograd_repeated_inputs_and_no_grad_match_pytorch_2_13(self):
        actual_left = torch.tensor([1.0, 2.0], requires_grad=True)
        actual_right = torch.tensor([3.0], requires_grad=True)
        expected_left = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32, requires_grad=True
        )
        expected_right = reference_torch.tensor(
            [3.0], dtype=reference_torch.float32, requires_grad=True
        )

        actual_tracked = torch.cat([actual_left, actual_right, actual_left], dim=0)
        expected_tracked = reference_torch.cat(
            [expected_left, expected_right, expected_left], dim=0
        )
        self.assert_matches(actual_tracked, expected_tracked, case="rank-1 autograd")
        weights = [1.0, 2.0, 3.0, 4.0, 5.0]
        (actual_tracked * torch.tensor(weights)).sum().backward()
        (
            expected_tracked
            * reference_torch.tensor(weights, dtype=reference_torch.float32)
        ).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_left.grad), expected_left.grad.detach().numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(actual_right.grad), expected_right.grad.detach().numpy()
        )

        actual_matrix_left = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        actual_matrix_right = torch.tensor([[5.0], [6.0]], requires_grad=True)
        expected_matrix_left = reference_torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        expected_matrix_right = reference_torch.tensor(
            [[5.0], [6.0]], dtype=reference_torch.float32, requires_grad=True
        )
        actual_matrix = torch.cat(
            [actual_matrix_left, actual_matrix_right, actual_matrix_left],
            dim=1,
        )
        expected_matrix = reference_torch.cat(
            [expected_matrix_left, expected_matrix_right, expected_matrix_left],
            dim=1,
        )
        self.assert_matches(actual_matrix, expected_matrix, case="rank-2 autograd")
        matrix_weights = [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
        (actual_matrix * torch.tensor(matrix_weights)).sum().backward()
        (
            expected_matrix
            * reference_torch.tensor(matrix_weights, dtype=reference_torch.float32)
        ).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_matrix_left.grad),
            expected_matrix_left.grad.detach().numpy(),
        )
        np.testing.assert_array_equal(
            np.asarray(actual_matrix_right.grad),
            expected_matrix_right.grad.detach().numpy(),
        )

        actual_row_top = torch.tensor([[7.0, 8.0]], requires_grad=True)
        actual_row_bottom = torch.tensor(
            [[9.0, 10.0], [11.0, 12.0]], requires_grad=True
        )
        expected_row_top = reference_torch.tensor(
            [[7.0, 8.0]], dtype=reference_torch.float32, requires_grad=True
        )
        expected_row_bottom = reference_torch.tensor(
            [[9.0, 10.0], [11.0, 12.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_rows = torch.cat(
            [actual_row_top, actual_row_bottom, actual_row_top],
            dim=0,
        )
        expected_rows = reference_torch.cat(
            [expected_row_top, expected_row_bottom, expected_row_top],
            dim=0,
        )
        self.assert_matches(actual_rows, expected_rows, case="rank-2 row autograd")
        row_weights = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        (actual_rows * torch.tensor(row_weights)).sum().backward()
        (
            expected_rows
            * reference_torch.tensor(row_weights, dtype=reference_torch.float32)
        ).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_row_top.grad),
            expected_row_top.grad.detach().numpy(),
        )
        np.testing.assert_array_equal(
            np.asarray(actual_row_bottom.grad),
            expected_row_bottom.grad.detach().numpy(),
        )

        actual_neutral_empty = torch.tensor([], requires_grad=True)
        actual_neutral_matrix = torch.tensor(
            [[13.0, 14.0], [15.0, 16.0]], requires_grad=True
        )
        expected_neutral_empty = reference_torch.tensor(
            [], dtype=reference_torch.float32, requires_grad=True
        )
        expected_neutral_matrix = reference_torch.tensor(
            [[13.0, 14.0], [15.0, 16.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_neutral = torch.cat(
            [actual_neutral_empty, actual_neutral_matrix, actual_neutral_empty],
            dim=1,
        )
        expected_neutral = reference_torch.cat(
            [
                expected_neutral_empty,
                expected_neutral_matrix,
                expected_neutral_empty,
            ],
            dim=1,
        )
        self.assert_matches(
            actual_neutral,
            expected_neutral,
            case="rank-2 neutral empty autograd",
        )
        actual_neutral.sum().backward()
        expected_neutral.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_neutral_empty.grad),
            expected_neutral_empty.grad.detach().numpy(),
        )
        np.testing.assert_array_equal(
            np.asarray(actual_neutral_matrix.grad),
            expected_neutral_matrix.grad.detach().numpy(),
        )

        actual_left = torch.tensor([1.0, 2.0], requires_grad=True)
        actual_right = torch.tensor([3.0], requires_grad=True)
        expected_left = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32, requires_grad=True
        )
        expected_right = reference_torch.tensor(
            [3.0], dtype=reference_torch.float32, requires_grad=True
        )
        with torch.no_grad():
            actual = torch.cat([actual_left, actual_right], dim=-1)
        with reference_torch.no_grad():
            expected = reference_torch.cat([expected_left, expected_right], dim=-1)

        self.assert_matches(actual, expected, case="no_grad")
        self.assertIsNone(actual_left.grad)
        self.assertIsNone(actual_right.grad)
        self.assertIsNone(expected_left.grad)
        self.assertIsNone(expected_right.grad)

    def test_representative_errors_match_pytorch_2_13(self):
        actual_vector = torch.tensor([1.0])
        expected_vector = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            ("empty", lambda: torch.cat([]), lambda: reference_torch.cat([])),
            (
                "scalar",
                lambda: torch.cat([torch.tensor(1.0)]),
                lambda: reference_torch.cat(
                    [reference_torch.tensor(1.0, dtype=reference_torch.float32)]
                ),
            ),
            (
                "non sequence",
                lambda: torch.cat(actual_vector),
                lambda: reference_torch.cat(expected_vector),
            ),
            (
                "bad element",
                lambda: torch.cat([actual_vector, 1]),
                lambda: reference_torch.cat([expected_vector, 1]),
            ),
            (
                "rank mismatch",
                lambda: torch.cat([torch.tensor([[1.0]]), actual_vector], dim=0),
                lambda: reference_torch.cat(
                    [
                        reference_torch.tensor(
                            [[1.0]], dtype=reference_torch.float32
                        ),
                        expected_vector,
                    ],
                    dim=0,
                ),
            ),
            (
                "shape mismatch dim 0",
                lambda: torch.cat([torch.ones((2, 2)), torch.ones((3, 3))], dim=0),
                lambda: reference_torch.cat(
                    [
                        reference_torch.ones((2, 2), dtype=reference_torch.float32),
                        reference_torch.ones((3, 3), dtype=reference_torch.float32),
                    ],
                    dim=0,
                ),
            ),
            (
                "shape mismatch dim 1",
                lambda: torch.cat([torch.ones((2, 2)), torch.ones((3, 3))], dim=1),
                lambda: reference_torch.cat(
                    [
                        reference_torch.ones((2, 2), dtype=reference_torch.float32),
                        reference_torch.ones((3, 3), dtype=reference_torch.float32),
                    ],
                    dim=1,
                ),
            ),
            (
                "high dim",
                lambda: torch.cat([torch.tensor([[1.0]])], dim=2),
                lambda: reference_torch.cat(
                    [reference_torch.tensor([[1.0]], dtype=reference_torch.float32)],
                    dim=2,
                ),
            ),
            (
                "low dim",
                lambda: torch.cat([torch.tensor([[1.0]])], dim=-3),
                lambda: reference_torch.cat(
                    [reference_torch.tensor([[1.0]], dtype=reference_torch.float32)],
                    dim=-3,
                ),
            ),
        )
        for case, actual_call, expected_call in cases:
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def test_axis_dim_conflicts_match_pytorch_2_13(self):
        actual = [torch.tensor([1.0])]
        expected = [reference_torch.tensor([1.0], dtype=reference_torch.float32)]
        cases = (
            (
                "dim keyword",
                lambda: torch.cat(actual, dim=0, axis=0),
                lambda: reference_torch.cat(expected, dim=0, axis=0),
            ),
            (
                "dim positional",
                lambda: torch.cat(actual, 0, axis=0),
                lambda: reference_torch.cat(expected, 0, axis=0),
            ),
        )
        for case, actual_call, expected_call in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^cat\(\) got an unexpected keyword argument 'axis'$",
                ):
                    actual_call()
                with self.assertRaisesRegex(
                    TypeError,
                    r"^cat\(\) got an unexpected keyword argument 'axis'$",
                ):
                    expected_call()

    def dispatch_observation(self, module, function_name):
        function = getattr(module, function_name)
        left = module.tensor([1.0])
        right = module.tensor([2.0])
        inputs = [left, right]
        marker = object()

        def stack_labels():
            return tuple(
                getattr(mode, "label", type(mode).__name__)
                for mode in module.overrides._get_current_function_mode_stack()
            )

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs, stack_labels()))
                return self.result

        accepting = RecordingMode(marker)
        with accepting:
            accepted = function(inputs, dim=0)
        func, dispatch_types, args, kwargs, stack = accepting.calls[0]
        mode_observation = (
            accepted is marker,
            func is function,
            tuple(item.__name__ for item in dispatch_types),
            args[0] is inputs,
            tuple(kwargs),
            kwargs["dim"],
            stack,
        )

        calls = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append(
                    (
                        self.label,
                        func is function,
                        tuple(item.__name__ for item in types),
                        args[0] is inputs,
                        tuple(kwargs),
                        kwargs["dim"],
                        stack_labels(),
                    )
                )
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(inputs, dim=0)

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

        events = []
        override_inputs = [LeftOverride(), module.tensor([]), RightOverride()]
        override_result = function(override_inputs, dim=0)
        override_observation = (
            override_result is marker,
            tuple(event[0] for event in events),
            tuple(
                (
                    event[1] is function,
                    tuple(item.__name__ for item in event[2]),
                    event[3][0] is override_inputs,
                    tuple(event[4]),
                    event[4]["dim"],
                )
                for event in events
            ),
        )

        class DimensionOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                argument_events.append(("dim", func, types, args, kwargs))
                return marker

        class OutOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                argument_events.append(("out", func, types, args, kwargs))
                return marker

        class TensorsOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                argument_events.append(("tensors", func, types, args, kwargs))
                return marker

        argument_events = []
        dimension = DimensionOverride()
        out = OutOverride()
        tensors = TensorsOverride()
        dim_result = function([left], dim=dimension)
        out_result = function([left], out=out)
        tensors_result = function(tensors)
        argument_observation = (
            dim_result is marker,
            out_result is marker,
            tensors_result is marker,
            tuple(
                (
                    event[0],
                    event[1] is function,
                    tuple(item.__name__ for item in event[2]),
                    len(event[3]),
                    None if event[4] is None else tuple(event[4]),
                )
                for event in argument_events
            ),
        )

        declining_mode = RecordingMode(NotImplemented)
        try:
            with declining_mode:
                function(inputs, dim=0)
        except Exception as error:
            mode_decline = (
                type(error).__name__,
                str(error).split("\n\n", maxsplit=1)[0],
                len(declining_mode.calls),
            )
        else:
            mode_decline = ("accepted", None, len(declining_mode.calls))

        class DecliningOverride:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return NotImplemented

        try:
            function([DecliningOverride()], dim=0)
        except Exception as error:
            override_decline = (
                type(error).__name__,
                str(error).split("\n\n", maxsplit=1)[0],
                DecliningOverride.calls,
            )
        else:
            override_decline = ("accepted", None, DecliningOverride.calls)

        return (
            mode_observation,
            tuple(calls),
            tuple(forwarded.tolist()),
            override_observation,
            argument_observation,
            mode_decline,
            override_decline,
            stack_labels(),
        )

    def test_torch_function_dispatch_matches_pytorch_2_13(self):
        for name in ("cat", "concat", "concatenate"):
            with self.subTest(function=name):
                self.assertEqual(
                    self.dispatch_observation(torch, name),
                    self.dispatch_observation(reference_torch, name),
                )


if __name__ == "__main__":
    unittest.main()
