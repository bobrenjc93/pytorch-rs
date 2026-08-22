import inspect
import pickle
import re
import types
import unittest

import torch_rs as torch


class AllcloseTests(unittest.TestCase):
    def assert_allclose_result(self, input, other, expected, **kwargs):
        result = torch.allclose(input, other, **kwargs)
        self.assertIs(type(result), bool)
        self.assertIs(result, expected)

    def test_contiguous_offset_noncontiguous_and_empty_tensors(self):
        strided = torch.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        ).transpose(0, 1)
        offset = torch.tensor(
            [[10.0, 20.0], [1.0, 3.0], [2.0, 4.0]]
        ).transpose(0, 1)[1]
        strided_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        offset_empty = strided_empty[1]

        self.assertEqual(strided.stride(), (1, 2))
        self.assertEqual(offset.stride(), (2,))
        self.assertEqual(offset.storage_offset(), 1)
        self.assertEqual(strided_empty.stride(), (1, 3, 3))
        self.assertEqual(offset_empty.storage_offset(), 1)

        cases = (
            (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[1.0, 2.00001], [3.0, 4.0]]),
                {},
                True,
            ),
            (
                strided,
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0002]]),
                {},
                False,
            ),
            (offset, torch.tensor([20.0, 3.0, 4.00002]), {}, True),
            (torch.zeros((2, 0, 3)), torch.ones((2, 0, 3)), {}, True),
            (strided_empty, torch.ones((3, 0, 2)), {}, True),
            (offset_empty, torch.ones((0, 2)), {}, True),
        )
        for input, other, kwargs, expected in cases:
            with self.subTest(
                input_shape=input.shape,
                input_stride=input.stride(),
                other_stride=other.stride(),
            ):
                self.assert_allclose_result(input, other, expected, **kwargs)

    def test_numerical_edge_cases(self):
        smallest_subnormal = float.fromhex("0x1p-149")
        maximum = 3.4028234663852886e38
        cases = (
            ([10000.0, 1.0e-7], [10000.1, 1.0e-8], {}, False),
            ([10000.0, 1.0e-8], [10000.1, 1.0e-9], {}, True),
            ([0.0, -0.0], [-0.0, 0.0], {"rtol": 0.0, "atol": 0.0}, True),
            (
                [float("inf"), -float("inf")],
                [float("inf"), -float("inf")],
                {},
                True,
            ),
            ([float("inf")], [-float("inf")], {"atol": float("inf")}, False),
            ([float("inf")], [1.0], {"atol": float("inf")}, False),
            ([maximum], [-maximum], {"atol": float("inf")}, False),
            ([maximum], [0.0], {"atol": float("inf")}, True),
            ([float("nan")], [float("nan")], {}, False),
            (
                [float("nan")],
                [float("nan")],
                {"equal_nan": True},
                True,
            ),
            (
                [float("nan")],
                [1.0],
                {"atol": float("inf"), "equal_nan": True},
                False,
            ),
            ([1.0], [2.0], {"rtol": 0.5, "atol": 0.0}, True),
            ([2.0], [1.0], {"rtol": 0.5, "atol": 0.0}, False),
            ([1.5], [1.0], {"rtol": 0.0, "atol": 0.5}, True),
            ([1.5], [1.0], {"rtol": 0.0, "atol": 0.499}, False),
            (
                [smallest_subnormal],
                [0.0],
                {"rtol": 0.0, "atol": smallest_subnormal * 0.5},
                False,
            ),
            (
                [smallest_subnormal],
                [0.0],
                {"rtol": 0.0, "atol": smallest_subnormal * 0.50000001},
                True,
            ),
        )
        for case, (input_values, other_values, kwargs, expected) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_allclose_result(
                    torch.tensor(input_values),
                    torch.tensor(other_values),
                    expected,
                    **kwargs,
                )

    def test_predicate_does_not_change_autograd_state(self):
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        input = leaf * 2.0
        other = torch.tensor([2.0, 4.0])
        state_before = (
            leaf.requires_grad,
            leaf.is_leaf,
            leaf.grad,
            input.requires_grad,
            input.is_leaf,
            input.data_ptr(),
            input.storage_offset(),
            input.stride(),
        )

        self.assert_allclose_result(input, other, True)
        self.assertEqual(
            (
                leaf.requires_grad,
                leaf.is_leaf,
                leaf.grad,
                input.requires_grad,
                input.is_leaf,
                input.data_ptr(),
                input.storage_offset(),
                input.stride(),
            ),
            state_before,
        )

        input.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [2.0, 2.0])

    def test_unequal_shapes_are_rejected_even_when_broadcastable(self):
        cases = (
            (torch.tensor(1.0), torch.tensor([1.0])),
            (torch.zeros((1, 2)), torch.zeros((2,))),
            (torch.zeros((2, 1)), torch.zeros((2, 3))),
            (torch.zeros((0,)), torch.zeros((1, 0))),
        )
        for input, other in cases:
            message = (
                "allclose(): same-shaped tensors are required, but input has shape "
                f"{list(input.shape)!r} and other has shape {list(other.shape)!r}"
            )
            with self.subTest(input_shape=input.shape, other_shape=other.shape):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    torch.allclose(input, other)

    def test_tolerance_validation(self):
        tensor = torch.tensor([1.0])
        cases = (
            (lambda: torch.allclose(tensor, tensor, rtol=-1.0e-5),
             "rtol must be greater than or equal to zero, but got -1e-05"),
            (lambda: torch.allclose(tensor, tensor, atol=-1.0e-8),
             "atol must be greater than or equal to zero, but got -1e-08"),
            (lambda: torch.allclose(tensor, tensor, rtol=float("nan")),
             "rtol must be greater than or equal to zero, but got nan"),
            (lambda: torch.allclose(tensor, tensor, atol=-float("inf")),
             "atol must be greater than or equal to zero, but got -inf"),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    call()

        self.assertIs(torch.allclose(tensor, tensor, rtol=-0.0, atol=-0.0), True)
        self.assertIs(torch.allclose(torch.zeros((0,)), torch.ones((0,))), True)
        self.assertIs(torch.allclose(tensor, tensor, rtol=torch.tensor(0.0)), True)

    def test_builtin_metadata_pickle_and_exports(self):
        self.assertIs(type(torch.allclose), types.BuiltinFunctionType)
        self.assertEqual(torch.allclose.__name__, "allclose")
        self.assertEqual(torch.allclose.__qualname__, "_VariableFunctionsClass.allclose")
        self.assertEqual(torch.allclose.__module__, "torch")
        self.assertIsNone(torch.allclose.__text_signature__)
        self.assertIsNone(torch.allclose.__self__)
        self.assertTrue(callable(torch.allclose))
        self.assertEqual(
            next(line for line in torch.allclose.__doc__.splitlines() if line),
            "allclose(input: Tensor, other: Tensor, rtol: float = 1e-05, "
            "atol: float = 1e-08, equal_nan: bool = False) -> bool",
        )
        with self.assertRaises(ValueError):
            inspect.signature(torch.allclose)
        self.assertIs(pickle.loads(pickle.dumps(torch.allclose)), torch.allclose)
        self.assertEqual(torch.__all__.count("allclose"), 1)

    def test_binding_and_operand_type_errors(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.allclose(),
                'allclose() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.allclose(tensor),
                'allclose() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.allclose(None),
                "allclose(): argument 'input' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: torch.allclose(tensor, tensor, 1.0, 1.0, False, None),
                "allclose() takes from 2 to 5 positional arguments but 6 were given",
            ),
            (
                lambda: torch.allclose([], tensor),
                "allclose(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.allclose(tensor, []),
                "allclose(): argument 'other' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.allclose(tensor, tensor, "bad"),
                "allclose(): argument 'rtol' (position 3) must be float, not str",
            ),
            (
                lambda: torch.allclose(tensor, tensor, atol=None),
                "allclose(): argument 'atol' must be float, not NoneType",
            ),
            (
                lambda: torch.allclose(tensor, tensor, equal_nan=1),
                "allclose(): argument 'equal_nan' must be bool, not int",
            ),
            (
                lambda: torch.allclose(tensor, tensor, extra=True),
                "allclose() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, other=tensor),
                "allclose() got multiple values for argument 'other'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, 0.0, rtol=0.0),
                "allclose() got multiple values for argument 'rtol'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        self.assertIs(torch.allclose(input=tensor, other=tensor), True)
        self.assertIs(torch.allclose(x=tensor, x2=tensor), True)
        self.assertIs(
            torch.allclose(tensor, tensor, 0.0, 0.0, True),
            True,
        )

    def test_torch_function_modes_and_operand_overrides(self):
        tensor = torch.tensor([1.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            result = torch.allclose(
                input=tensor,
                other=tensor,
                rtol=-1.0,
                equal_nan=True,
            )
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.allclose)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other", "rtol", "equal_nan"))
        self.assertIs(kwargs["input"], tensor)
        self.assertIs(kwargs["other"], tensor)

        events = []

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

        left = LeftOverride()
        right = RightOverride()
        self.assertIs(torch.allclose(left, right, 0.0, 0.0, True), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.allclose)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 5)
            self.assertIs(args[0], left)
            self.assertIs(args[1], right)
            self.assertIsNone(kwargs)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            "^Multiple dispatch failed for 'torch\\.allclose'; all "
            "__torch_function__ handlers returned NotImplemented:",
        ):
            torch.allclose(DecliningOverride(), tensor)

        invalid_mode = RecordingMode(marker)
        with invalid_mode:
            with self.assertRaises(TypeError):
                torch.allclose(tensor, [], rtol="bad")
        self.assertEqual(invalid_mode.calls, [])


if __name__ == "__main__":
    unittest.main()
