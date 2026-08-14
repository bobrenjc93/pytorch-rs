import inspect
import sys
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorSwapdimsReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_allclose(
                np.asarray(actual),
                expected.detach().cpu().numpy(),
                rtol=2.0e-6,
                atol=1.0e-6,
                equal_nan=True,
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_seeded_views_metadata_and_consumers_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0x5A9D_213)
        shapes = [(), (0,), (2, 0, 3), (1, 3, 2), (2, 3, 4)]
        for _ in range(28):
            rank = int(rng.integers(0, 6))
            shapes.append(tuple(int(value) for value in rng.integers(0, 5, size=rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.normal(size=elements).astype(np.float32).reshape(shape)
            if elements == 0:
                actual = torch.zeros(shape)
                expected = reference_torch.zeros(shape, dtype=reference_torch.float32)
            else:
                actual = torch.tensor(values.item() if shape == () else values.tolist())
                expected = reference_torch.tensor(values, dtype=reference_torch.float32)

            if len(shape) >= 2 and shape[0] > 0 and case % 4 == 1:
                actual = actual.transpose(0, -1)[-1]
                expected = expected.transpose(0, -1)[-1]

            rank = len(actual.shape)
            dimensions = [0, -1] if rank == 0 else list(range(-rank, rank))
            for chain in range(4):
                dim0 = dimensions[int(rng.integers(0, len(dimensions)))]
                dim1 = dimensions[int(rng.integers(0, len(dimensions)))]
                actual_pointer = actual.data_ptr()
                expected_pointer = expected.data_ptr()
                call_style = (case + chain) % 4
                if call_style == 0:
                    actual = actual.swapdims(dim0, dim1)
                    expected = expected.swapdims(dim0, dim1)
                elif call_style == 1:
                    actual = actual.swapdims(dim0=dim0, dim1=dim1)
                    expected = expected.swapdims(dim0=dim0, dim1=dim1)
                elif call_style == 2:
                    actual = torch.swapdims(actual, dim0, dim1)
                    expected = reference_torch.swapdims(expected, dim0, dim1)
                else:
                    actual = torch.swapdims(input=actual, dim0=dim0, dim1=dim1)
                    expected = reference_torch.swapdims(
                        input=expected, dim0=dim0, dim1=dim1
                    )
                self.assertEqual(actual.data_ptr(), actual_pointer)
                self.assertEqual(expected.data_ptr(), expected_pointer)

            self.assert_matches(actual, expected, case=f"view-{case}")
            for operation, actual_output, expected_output in (
                ("clone", actual.clone(), expected.clone()),
                ("sin", actual.sin(), expected.sin()),
                ("scalar", actual + 1.25, expected + 1.25),
                ("reshape", actual.reshape(-1), expected.reshape(-1)),
                ("sum", actual.sum(), expected.sum()),
            ):
                self.assert_matches(
                    actual_output, expected_output, case=f"{operation}-{case}"
                )

    def test_autograd_and_storage_view_metadata_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(4, 3, 2)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual = torch.swapdims(actual_leaf, 0, -1)
        expected = reference_torch.swapdims(expected_leaf, 0, -1)

        self.assert_matches(actual, expected, case="tracked-view")
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertEqual(actual.data_ptr(), actual_leaf.data_ptr())
        self.assertEqual(expected.data_ptr(), expected_leaf.data_ptr())
        (actual * torch.tensor(weights.tolist())).sum().backward()
        (expected * reference_torch.tensor(weights)).sum().backward()
        self.assert_matches(actual_leaf.grad, expected_leaf.grad, case="gradient")

        actual_scalar = torch.tensor(2.5, requires_grad=True)
        expected_scalar = reference_torch.tensor(2.5, requires_grad=True)
        actual_scalar_view = torch.swapdims(
            input=actual_scalar, dim0=0, dim1=-1
        )
        expected_scalar_view = reference_torch.swapdims(
            input=expected_scalar, dim0=0, dim1=-1
        )
        (actual_scalar_view * 7.0).backward()
        (expected_scalar_view * 7.0).backward()
        self.assert_matches(
            actual_scalar.grad, expected_scalar.grad, case="scalar-gradient"
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        torch.swapdims(actual_empty, 0, -1).sum().backward()
        reference_torch.swapdims(expected_empty, 0, -1).sum().backward()
        self.assert_matches(actual_empty.grad, expected_empty.grad, case="empty-gradient")

        with torch.no_grad():
            actual_untracked = torch.swapdims(actual_leaf, 0, 1)
        with reference_torch.no_grad():
            expected_untracked = reference_torch.swapdims(expected_leaf, 0, 1)
        self.assertEqual(actual_untracked.requires_grad, expected_untracked.requires_grad)
        self.assertEqual(actual_untracked.is_leaf, expected_untracked.is_leaf)

        actual_offset_source = torch.tensor(values.tolist()).transpose(0, 2)[1]
        expected_offset_source = reference_torch.tensor(values).transpose(0, 2)[1]
        actual_offset = torch.swapdims(actual_offset_source, 0, 1)
        expected_offset = reference_torch.swapdims(
            expected_offset_source, 0, 1
        )
        self.assert_matches(actual_offset, expected_offset, case="offset-view")
        self.assertEqual(actual_offset.data_ptr(), actual_offset_source.data_ptr())
        self.assertEqual(expected_offset.data_ptr(), expected_offset_source.data_ptr())

    def test_binding_dimension_and_overflow_errors_match_pytorch_2_13(self):
        class UserOverflow(np.int64):
            def __index__(self):
                raise OverflowError("user overflow")

        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 3))
        expected = reference_torch.zeros((2, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            timedelta = np.timedelta64(1)
        cases = (
            (lambda: actual.swapdims(), lambda: expected.swapdims()),
            (lambda: actual.swapdims(0), lambda: expected.swapdims(0)),
            (lambda: actual.swapdims(0, 1, 2), lambda: expected.swapdims(0, 1, 2)),
            (
                lambda: actual.swapdims(dim1=0, unexpected=None),
                lambda: expected.swapdims(dim1=0, unexpected=None),
            ),
            (
                lambda: actual.swapdims(0, 1, unexpected=None),
                lambda: expected.swapdims(0, 1, unexpected=None),
            ),
            (
                lambda: actual.swapdims(0, dim0=1, dim1=0),
                lambda: expected.swapdims(0, dim0=1, dim1=0),
            ),
            (lambda: actual.swapdims(1.5), lambda: expected.swapdims(1.5)),
            (
                lambda: actual.swapdims(dim0=np.bool_(True)),
                lambda: expected.swapdims(dim0=np.bool_(True)),
            ),
            (
                lambda: actual.swapdims(np.float64(1.5), 0),
                lambda: expected.swapdims(np.float64(1.5), 0),
            ),
            (
                lambda: actual.swapdims(1.5, 0, unexpected=None),
                lambda: expected.swapdims(1.5, 0, unexpected=None),
            ),
            (
                lambda: actual.swapdims(2**100, unexpected=None),
                lambda: expected.swapdims(2**100, unexpected=None),
            ),
            (
                lambda: actual.swapdims(2**100, 0, unexpected=None),
                lambda: expected.swapdims(2**100, 0, unexpected=None),
            ),
            (
                lambda: actual.swapdims(2**100, 1.5),
                lambda: expected.swapdims(2**100, 1.5),
            ),
            (
                lambda: actual.swapdims(timedelta, 0),
                lambda: expected.swapdims(timedelta, 0),
            ),
            (
                lambda: actual.swapdims(np.uint64(2**63), 0),
                lambda: expected.swapdims(np.uint64(2**63), 0),
            ),
            (
                lambda: actual.swapdims(2**100, timedelta),
                lambda: expected.swapdims(2**100, timedelta),
            ),
            (
                lambda: actual.swapdims(timedelta, 2**100),
                lambda: expected.swapdims(timedelta, 2**100),
            ),
            (
                lambda: actual.swapdims(2**100, UserOverflow(0)),
                lambda: expected.swapdims(2**100, UserOverflow(0)),
            ),
            (lambda: actual.swapdims(2**100, 0), lambda: expected.swapdims(2**100, 0)),
            (lambda: actual.swapdims(-3, 0), lambda: expected.swapdims(-3, 0)),
            (lambda: actual.swapdims(0, 2), lambda: expected.swapdims(0, 2)),
            (lambda: torch.swapdims(), lambda: reference_torch.swapdims()),
            (lambda: torch.swapdims(actual), lambda: reference_torch.swapdims(expected)),
            (
                lambda: torch.swapdims(actual, 0),
                lambda: reference_torch.swapdims(expected, 0),
            ),
            (
                lambda: torch.swapdims(actual, 0, 1, 2),
                lambda: reference_torch.swapdims(expected, 0, 1, 2),
            ),
            (
                lambda: torch.swapdims(dim0=0, dim1=1),
                lambda: reference_torch.swapdims(dim0=0, dim1=1),
            ),
            (
                lambda: torch.swapdims(input=actual, dim1=0, unexpected=None),
                lambda: reference_torch.swapdims(
                    input=expected, dim1=0, unexpected=None
                ),
            ),
            (
                lambda: torch.swapdims(actual, 0, 1, unexpected=None),
                lambda: reference_torch.swapdims(
                    expected, 0, 1, unexpected=None
                ),
            ),
            (
                lambda: torch.swapdims(actual, 0, dim0=1, dim1=0),
                lambda: reference_torch.swapdims(expected, 0, dim0=1, dim1=0),
            ),
            (lambda: torch.swapdims(1), lambda: reference_torch.swapdims(1)),
            (
                lambda: torch.swapdims(input=1),
                lambda: reference_torch.swapdims(input=1),
            ),
            (
                lambda: torch.swapdims(actual, 1.5),
                lambda: reference_torch.swapdims(expected, 1.5),
            ),
            (
                lambda: torch.swapdims(input=actual, dim0=np.bool_(True)),
                lambda: reference_torch.swapdims(
                    input=expected, dim0=np.bool_(True)
                ),
            ),
            (
                lambda: torch.swapdims(actual, np.float64(1.5), 0),
                lambda: reference_torch.swapdims(expected, np.float64(1.5), 0),
            ),
            (
                lambda: torch.swapdims(actual, 1.5, 0, unexpected=None),
                lambda: reference_torch.swapdims(
                    expected, 1.5, 0, unexpected=None
                ),
            ),
            (
                lambda: torch.swapdims(actual, 2**100, 0, unexpected=None),
                lambda: reference_torch.swapdims(
                    expected, 2**100, 0, unexpected=None
                ),
            ),
            (
                lambda: torch.swapdims(actual, 2**100, 1.5),
                lambda: reference_torch.swapdims(expected, 2**100, 1.5),
            ),
            (
                lambda: torch.swapdims(actual, timedelta, 0),
                lambda: reference_torch.swapdims(expected, timedelta, 0),
            ),
            (
                lambda: torch.swapdims(actual, np.uint64(2**63), 0),
                lambda: reference_torch.swapdims(expected, np.uint64(2**63), 0),
            ),
            (
                lambda: torch.swapdims(actual, 2**100, timedelta),
                lambda: reference_torch.swapdims(expected, 2**100, timedelta),
            ),
            (
                lambda: torch.swapdims(actual, timedelta, 2**100),
                lambda: reference_torch.swapdims(expected, timedelta, 2**100),
            ),
            (
                lambda: torch.swapdims(actual, 2**100, UserOverflow(0)),
                lambda: reference_torch.swapdims(
                    expected, 2**100, UserOverflow(0)
                ),
            ),
            (
                lambda: torch.swapdims(actual, 2**100, 0),
                lambda: reference_torch.swapdims(expected, 2**100, 0),
            ),
            (
                lambda: torch.swapdims(actual, -3, 0),
                lambda: reference_torch.swapdims(expected, -3, 0),
            ),
            (
                lambda: torch.swapdims(actual, 0, 2),
                lambda: reference_torch.swapdims(expected, 0, 2),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(error_case=case):
                self.assert_error_matches(actual_call, expected_call)

        actual_scalar = torch.tensor(1.0)
        expected_scalar = reference_torch.tensor(1.0)
        self.assert_error_matches(
            lambda: actual_scalar.swapdims(-2, 0),
            lambda: expected_scalar.swapdims(-2, 0),
        )
        self.assert_error_matches(
            lambda: torch.swapdims(actual_scalar, -2, 0),
            lambda: reference_torch.swapdims(expected_scalar, -2, 0),
        )

        actual_extreme = torch.zeros((sys.maxsize, 0, 2, 2))
        expected_extreme = reference_torch.zeros((sys.maxsize, 0, 2, 2))
        self.assert_error_matches(
            lambda: actual_extreme.swapdims(1, 3),
            lambda: expected_extreme.swapdims(1, 3),
        )
        self.assert_error_matches(
            lambda: torch.swapdims(actual_extreme, 1, 3),
            lambda: reference_torch.swapdims(expected_extreme, 1, 3),
        )

    def test_stateful_dimension_conversion_order_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        def stateful_dimensions():
            state = {"dim1_converted": False, "calls": []}

            class StatefulInteger(np.int64):
                def __new__(cls, role):
                    value = np.int64.__new__(cls, 0)
                    value.role = role
                    return value

                def __index__(self):
                    state["calls"].append(self.role)
                    if self.role == "dim1":
                        state["dim1_converted"] = True
                        return 1
                    return 0 if state["dim1_converted"] else 2

            return state, StatefulInteger("dim0"), StatefulInteger("dim1")

        actual_state, actual_dim0, actual_dim1 = stateful_dimensions()
        expected_state, expected_dim0, expected_dim1 = stateful_dimensions()
        actual = torch.swapdims(
            torch.zeros((2, 3, 4)), actual_dim0, actual_dim1
        )
        expected = reference_torch.swapdims(
            reference_torch.zeros((2, 3, 4)), expected_dim0, expected_dim1
        )

        self.assert_matches(actual, expected, case="stateful-conversion-order")
        self.assertEqual(actual_state["calls"], expected_state["calls"])
        self.assertEqual(actual_state["calls"], ["dim1", "dim0"])

    def test_descriptor_matches_pytorch_2_13(self):
        actual_descriptor = torch.Tensor.swapdims
        expected_descriptor = reference_torch.Tensor.swapdims
        self.assertEqual(
            actual_descriptor.__text_signature__, expected_descriptor.__text_signature__
        )
        self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)
        for descriptor in (actual_descriptor, expected_descriptor):
            with self.subTest(descriptor=descriptor):
                with self.assertRaises(ValueError):
                    inspect.signature(descriptor)

        actual_function = torch.swapdims
        expected_function = reference_torch.swapdims
        self.assertIs(type(actual_function), types.BuiltinFunctionType)
        self.assertIs(type(expected_function), types.BuiltinFunctionType)
        self.assertEqual(actual_function.__name__, expected_function.__name__)
        self.assertEqual(
            actual_function.__text_signature__, expected_function.__text_signature__
        )
        self.assertEqual(actual_function.__doc__, expected_function.__doc__)
        self.assertEqual("swapdims" in torch.__all__, "swapdims" in reference_torch.__all__)
        self.assertEqual(actual_function.__module__, torch.tensor.__module__)
        self.assertEqual(expected_function.__module__, "torch")
        for function in (actual_function, expected_function):
            with self.subTest(function=function):
                with self.assertRaises(ValueError):
                    inspect.signature(function)


if __name__ == "__main__":
    unittest.main()
