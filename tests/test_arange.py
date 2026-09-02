import inspect
import math
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


NUMPY_FLOAT_TYPES = (np.float16, np.float32, np.float64, np.longdouble)
NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)


class FloatSubclass(float):
    pass


class IntSubclass(int):
    pass


class NumpyFloatSubclass(np.float32):
    pass


class NumpyIntegerSubclass(np.int64):
    pass


class SpoofedNumpyFloat:
    def __init__(self):
        self.float_calls = 0

    @property
    def __class__(self):
        return np.float32

    def __float__(self):
        self.float_calls += 1
        return 3.0


class ArangeTests(unittest.TestCase):
    def assert_default_tensor(self, tensor, values, *, requires_grad=False):
        self.assertEqual(tuple(tensor.shape), (len(values),))
        self.assertEqual(tensor.stride(), (1,))
        self.assertEqual(tensor.storage_offset(), 0)
        self.assertEqual(tensor.numel(), len(values))
        self.assertEqual(tensor.tolist(), values)
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertIs(tensor.layout, torch.strided)
        self.assertIs(tensor.requires_grad, requires_grad)
        self.assertTrue(tensor.is_leaf)
        self.assertIsNone(tensor.grad)

    def assert_error(self, call, error_type, message):
        with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
            call()

    def test_exact_float_endpoint_supports_positional_and_keyword_forms(self):
        cases = (
            (0.0, []),
            (-0.0, []),
            (math.nextafter(0.0, 1.0), [0.0]),
            (0.25, [0.0]),
            (1.0, [0.0]),
            (math.nextafter(1.0, 2.0), [0.0, 1.0]),
            (2.5, [0.0, 1.0, 2.0]),
            (4.0, [0.0, 1.0, 2.0, 3.0]),
        )
        for end, expected in cases:
            for form, call in (
                ("positional", lambda end=end: torch.arange(end)),
                ("keyword", lambda end=end: torch.arange(end=end)),
            ):
                with self.subTest(end=end, form=form):
                    self.assert_default_tensor(call(), expected)

    def test_numpy_floating_endpoints_support_all_one_bound_float32_forms(self):
        scalar_types = (*NUMPY_FLOAT_TYPES, NumpyFloatSubclass)
        cases = (
            (0.0, []),
            (-0.0, []),
            (0.25, [0.0]),
            (1.0, [0.0]),
            (2.5, [0.0, 1.0, 2.0]),
            (4.0, [0.0, 1.0, 2.0, 3.0]),
        )
        dtype_cases = (
            ("omitted", {}),
            ("float32", {"dtype": torch.float32}),
            ("float", {"dtype": torch.float}),
        )
        for scalar_type in scalar_types:
            for raw_end, expected in cases:
                end = scalar_type(raw_end)
                for dtype_name, options in dtype_cases:
                    for form, call in (
                        (
                            "positional",
                            lambda end=end, options=options: torch.arange(
                                end, **options
                            ),
                        ),
                        (
                            "keyword",
                            lambda end=end, options=options: torch.arange(
                                end=end, **options
                            ),
                        ),
                    ):
                        with self.subTest(
                            scalar_type=scalar_type.__name__,
                            end=end,
                            dtype=dtype_name,
                            form=form,
                        ):
                            self.assert_default_tensor(call(), expected)

    def test_exact_integer_endpoint_requires_explicit_float32_dtype(self):
        cases = (
            (0, []),
            (1, [0.0]),
            (3, [0.0, 1.0, 2.0]),
            (8, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        for end, expected in cases:
            for dtype in (torch.float32, torch.float):
                for form, call in (
                    (
                        "positional",
                        lambda end=end, dtype=dtype: torch.arange(
                            end, dtype=dtype
                        ),
                    ),
                    (
                        "keyword",
                        lambda end=end, dtype=dtype: torch.arange(
                            end=end, dtype=dtype
                        ),
                    ),
                ):
                    with self.subTest(end=end, dtype=dtype, form=form):
                        self.assert_default_tensor(call(), expected)

    def test_two_bound_exact_integer_endpoints_support_implicit_step(self):
        cases = (
            (0, 0, []),
            (0, 4, [0.0, 1.0, 2.0, 3.0]),
            (2, 6, [2.0, 3.0, 4.0, 5.0]),
            (-3, 2, [-3.0, -2.0, -1.0, 0.0, 1.0]),
            (
                16_777_217,
                16_777_221,
                [16_777_216.0, 16_777_218.0, 16_777_220.0, 16_777_220.0],
            ),
        )
        for start, end, expected in cases:
            for dtype in (torch.float32, torch.float):
                for form, call in (
                    (
                        "positional",
                        lambda start=start, end=end, dtype=dtype: torch.arange(
                            start, end, dtype=dtype
                        ),
                    ),
                    (
                        "positional_start_keyword_end",
                        lambda start=start, end=end, dtype=dtype: torch.arange(
                            start, end=end, dtype=dtype
                        ),
                    ),
                    (
                        "keywords",
                        lambda start=start, end=end, dtype=dtype: torch.arange(
                            start=start, end=end, dtype=dtype
                        ),
                    ),
                ):
                    with self.subTest(
                        start=start, end=end, dtype=dtype, form=form
                    ):
                        self.assert_default_tensor(call(), expected)

    def test_two_bound_numpy_integer_endpoints_support_implicit_step(self):
        cases = tuple(
            (scalar_type(1), scalar_type(4), [1.0, 2.0, 3.0])
            for scalar_type in NUMPY_INTEGER_TYPES
        ) + (
            (np.int64(-2), np.int32(2), [-2.0, -1.0, 0.0, 1.0]),
            (
                NumpyIntegerSubclass(1),
                NumpyIntegerSubclass(3),
                [1.0, 2.0],
            ),
        )
        for start, end, expected in cases:
            for dtype in (torch.float32, torch.float):
                for form, call in (
                    (
                        "positional",
                        lambda start=start, end=end, dtype=dtype: torch.arange(
                            start, end, dtype=dtype
                        ),
                    ),
                    (
                        "positional_start_keyword_end",
                        lambda start=start, end=end, dtype=dtype: torch.arange(
                            start, end=end, dtype=dtype
                        ),
                    ),
                    (
                        "keywords",
                        lambda start=start, end=end, dtype=dtype: torch.arange(
                            start=start, end=end, dtype=dtype
                        ),
                    ),
                ):
                    with self.subTest(
                        start_type=type(start).__name__,
                        end_type=type(end).__name__,
                        dtype=dtype,
                        form=form,
                    ):
                        self.assert_default_tensor(call(), expected)

    def test_default_equivalent_metadata_is_accepted(self):
        option_cases = (
            {},
            {"out": None},
            {"dtype": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"layout": None},
            {"layout": torch.strided},
            {"device": None},
            {"device": "cpu"},
            {"device": "cpu:0"},
            {"device": torch.device("cpu")},
            {"pin_memory": None},
            {"pin_memory": False},
            {"requires_grad": None},
            {"requires_grad": False},
            {
                "out": None,
                "dtype": torch.float32,
                "layout": torch.strided,
                "device": torch.device("cpu"),
                "pin_memory": False,
                "requires_grad": False,
            },
        )
        for options in option_cases:
            with self.subTest(options=options):
                self.assert_default_tensor(
                    torch.arange(end=2.5, **options),
                    [0.0, 1.0, 2.0],
                )

    def test_two_bound_integer_default_equivalent_metadata_is_accepted(self):
        option_cases = (
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"dtype": torch.float32, "out": None},
            {"dtype": torch.float32, "layout": None},
            {"dtype": torch.float32, "layout": torch.strided},
            {"dtype": torch.float32, "device": None},
            {"dtype": torch.float32, "device": "cpu"},
            {"dtype": torch.float32, "device": "cpu:0"},
            {"dtype": torch.float32, "device": torch.device("cpu")},
            {"dtype": torch.float32, "pin_memory": None},
            {"dtype": torch.float32, "pin_memory": False},
            {"dtype": torch.float32, "requires_grad": None},
            {"dtype": torch.float32, "requires_grad": False},
            {
                "out": None,
                "dtype": torch.float32,
                "layout": torch.strided,
                "device": torch.device("cpu"),
                "pin_memory": False,
                "requires_grad": False,
            },
        )
        for options in option_cases:
            with self.subTest(options=options):
                self.assert_default_tensor(
                    torch.arange(1, 4, **options),
                    [1.0, 2.0, 3.0],
                )

    def test_requires_grad_creates_leaves_inside_and_outside_no_grad(self):
        ordinary = torch.arange(4.0, requires_grad=True)
        with torch.no_grad():
            no_grad = torch.arange(end=4.0, requires_grad=True)

        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        for context, leaf in (("ordinary", ordinary), ("no_grad", no_grad)):
            with self.subTest(context=context):
                self.assert_default_tensor(
                    leaf,
                    [0.0, 1.0, 2.0, 3.0],
                    requires_grad=True,
                )
                for expected in (
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 4.0, 6.0, 8.0],
                ):
                    (leaf * weights).sum().backward()
                    self.assertEqual(leaf.grad.tolist(), expected)

        for end in (0.0, -0.0):
            with self.subTest(end=end):
                with torch.no_grad():
                    empty = torch.arange(end, requires_grad=True)
                self.assert_default_tensor(empty, [], requires_grad=True)

    def test_numpy_floating_requires_grad_creates_leaves_under_no_grad(self):
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        for scalar_type in NUMPY_FLOAT_TYPES:
            ordinary = torch.arange(scalar_type(4.0), requires_grad=True)
            with torch.no_grad():
                no_grad = torch.arange(
                    end=scalar_type(4.0),
                    dtype=torch.float32,
                    requires_grad=True,
                )
                empty = torch.arange(
                    scalar_type(-0.0),
                    dtype=torch.float,
                    requires_grad=True,
                )

            for context, leaf in (("ordinary", ordinary), ("no_grad", no_grad)):
                with self.subTest(
                    scalar_type=scalar_type.__name__, context=context
                ):
                    self.assert_default_tensor(
                        leaf,
                        [0.0, 1.0, 2.0, 3.0],
                        requires_grad=True,
                    )
                    for expected in (
                        [1.0, 2.0, 3.0, 4.0],
                        [2.0, 4.0, 6.0, 8.0],
                    ):
                        (leaf * weights).sum().backward()
                        self.assertEqual(leaf.grad.tolist(), expected)

            self.assert_default_tensor(empty, [], requires_grad=True)

    def test_explicit_float32_integer_requires_grad_creates_leaves_under_no_grad(self):
        ordinary = torch.arange(4, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            no_grad = torch.arange(
                end=4, dtype=torch.float, requires_grad=True
            )
            empty = torch.arange(0, dtype=torch.float32, requires_grad=True)

        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        for context, leaf in (("ordinary", ordinary), ("no_grad", no_grad)):
            with self.subTest(context=context):
                self.assert_default_tensor(
                    leaf,
                    [0.0, 1.0, 2.0, 3.0],
                    requires_grad=True,
                )
                for expected in (
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 4.0, 6.0, 8.0],
                ):
                    (leaf * weights).sum().backward()
                    self.assertEqual(leaf.grad.tolist(), expected)

        self.assert_default_tensor(empty, [], requires_grad=True)

    def test_two_bound_integer_requires_grad_creates_leaves_under_no_grad(self):
        ordinary = torch.arange(-1, 3, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            no_grad = torch.arange(
                start=-1, end=3, dtype=torch.float, requires_grad=True
            )
            empty = torch.arange(
                np.int64(2),
                np.int64(2),
                dtype=torch.float32,
                requires_grad=True,
            )

        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        for context, leaf in (("ordinary", ordinary), ("no_grad", no_grad)):
            with self.subTest(context=context):
                self.assert_default_tensor(
                    leaf,
                    [-1.0, 0.0, 1.0, 2.0],
                    requires_grad=True,
                )
                for expected in (
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 4.0, 6.0, 8.0],
                ):
                    (leaf * weights).sum().backward()
                    self.assertEqual(leaf.grad.tolist(), expected)

        self.assert_default_tensor(empty, [], requires_grad=True)

    def test_each_result_owns_fresh_storage(self):
        for requires_grad in (False, True):
            with self.subTest(requires_grad=requires_grad):
                first = torch.arange(8.5, requires_grad=requires_grad)
                second = torch.arange(8.5, requires_grad=requires_grad)
                self.assertNotEqual(first.data_ptr(), 0)
                self.assertNotEqual(second.data_ptr(), 0)
                self.assertNotEqual(first.data_ptr(), second.data_ptr())
                self.assertFalse(first.is_set_to(second))

                empty_first = torch.arange(0.0, requires_grad=requires_grad)
                empty_second = torch.arange(-0.0, requires_grad=requires_grad)
                self.assertEqual(empty_first.data_ptr(), 0)
                self.assertEqual(empty_second.data_ptr(), 0)
                self.assertFalse(empty_first.is_set_to(empty_second))

    def test_numpy_floating_results_own_fresh_storage(self):
        for scalar_type in NUMPY_FLOAT_TYPES:
            for requires_grad in (False, True):
                with self.subTest(
                    scalar_type=scalar_type.__name__,
                    requires_grad=requires_grad,
                ):
                    first = torch.arange(
                        scalar_type(8.5), requires_grad=requires_grad
                    )
                    second = torch.arange(
                        end=scalar_type(8.5),
                        dtype=torch.float32,
                        requires_grad=requires_grad,
                    )
                    self.assertNotEqual(first.data_ptr(), 0)
                    self.assertNotEqual(second.data_ptr(), 0)
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())
                    self.assertFalse(first.is_set_to(second))

                    empty_first = torch.arange(
                        scalar_type(0.0), requires_grad=requires_grad
                    )
                    empty_second = torch.arange(
                        end=scalar_type(-0.0),
                        dtype=torch.float,
                        requires_grad=requires_grad,
                    )
                    self.assertEqual(empty_first.data_ptr(), 0)
                    self.assertEqual(empty_second.data_ptr(), 0)
                    self.assertFalse(empty_first.is_set_to(empty_second))

    def test_explicit_float32_integer_results_own_fresh_storage(self):
        for requires_grad in (False, True):
            with self.subTest(requires_grad=requires_grad):
                first = torch.arange(
                    8, dtype=torch.float32, requires_grad=requires_grad
                )
                second = torch.arange(
                    end=8, dtype=torch.float, requires_grad=requires_grad
                )
                self.assertNotEqual(first.data_ptr(), 0)
                self.assertNotEqual(second.data_ptr(), 0)
                self.assertNotEqual(first.data_ptr(), second.data_ptr())
                self.assertFalse(first.is_set_to(second))

                empty_first = torch.arange(
                    0, dtype=torch.float32, requires_grad=requires_grad
                )
                empty_second = torch.arange(
                    end=0, dtype=torch.float, requires_grad=requires_grad
                )
                self.assertEqual(empty_first.data_ptr(), 0)
                self.assertEqual(empty_second.data_ptr(), 0)
                self.assertFalse(empty_first.is_set_to(empty_second))

    def test_two_bound_integer_results_own_fresh_storage(self):
        for requires_grad in (False, True):
            with self.subTest(requires_grad=requires_grad):
                first = torch.arange(
                    1, 9, dtype=torch.float32, requires_grad=requires_grad
                )
                second = torch.arange(
                    start=1,
                    end=9,
                    dtype=torch.float,
                    requires_grad=requires_grad,
                )
                self.assertNotEqual(first.data_ptr(), 0)
                self.assertNotEqual(second.data_ptr(), 0)
                self.assertNotEqual(first.data_ptr(), second.data_ptr())
                self.assertFalse(first.is_set_to(second))

                empty_first = torch.arange(
                    4, 4, dtype=torch.float32, requires_grad=requires_grad
                )
                empty_second = torch.arange(
                    start=np.int64(4),
                    end=np.int64(4),
                    dtype=torch.float,
                    requires_grad=requires_grad,
                )
                self.assertEqual(empty_first.data_ptr(), 0)
                self.assertEqual(empty_second.data_ptr(), 0)
                self.assertFalse(empty_first.is_set_to(empty_second))

    def test_negative_and_nonfinite_endpoints_match_pytorch_errors(self):
        for end in (-math.nextafter(0.0, 1.0), -0.25, -1.0):
            for form, call in (
                ("positional", lambda end=end: torch.arange(end)),
                ("keyword", lambda end=end: torch.arange(end=end)),
            ):
                with self.subTest(end=end, form=form):
                    self.assert_error(
                        call,
                        RuntimeError,
                        "upper bound and lower bound inconsistent with step sign",
                    )

        for end, rendered in (
            (float("nan"), "nan"),
            (float("-nan"), "-nan"),
            (float("inf"), "inf"),
            (float("-inf"), "-inf"),
        ):
            with self.subTest(end=rendered):
                self.assert_error(
                    lambda end=end: torch.arange(end),
                    RuntimeError,
                    f"unsupported range: 0 -> {rendered}",
                )

        for form, call in (
            (
                "positional",
                lambda: torch.arange(-1, dtype=torch.float32),
            ),
            (
                "keyword",
                lambda: torch.arange(end=-1, dtype=torch.float),
            ),
        ):
            with self.subTest(end=-1, form=form):
                self.assert_error(
                    call,
                    RuntimeError,
                    "upper bound and lower bound inconsistent with step sign",
                )

    def test_numpy_floating_negative_and_nonfinite_errors(self):
        for scalar_type in NUMPY_FLOAT_TYPES:
            end = scalar_type(-0.25)
            for form, call in (
                ("positional", lambda end=end: torch.arange(end)),
                (
                    "keyword",
                    lambda end=end: torch.arange(
                        end=end, dtype=torch.float32
                    ),
                ),
            ):
                with self.subTest(
                    scalar_type=scalar_type.__name__, end=end, form=form
                ):
                    self.assert_error(
                        call,
                        RuntimeError,
                        "upper bound and lower bound inconsistent with step sign",
                    )

            for end in (
                scalar_type("nan"),
                scalar_type("-nan"),
                scalar_type("inf"),
                scalar_type("-inf"),
            ):
                converted = float(end)
                if math.isnan(converted):
                    rendered = (
                        "-nan"
                        if math.copysign(1.0, converted) < 0.0
                        else "nan"
                    )
                else:
                    rendered = "-inf" if converted < 0.0 else "inf"
                for form, call in (
                    ("positional", lambda end=end: torch.arange(end)),
                    (
                        "keyword",
                        lambda end=end: torch.arange(
                            end=end, dtype=torch.float
                        ),
                    ),
                ):
                    with self.subTest(
                        scalar_type=scalar_type.__name__,
                        end=rendered,
                        form=form,
                    ):
                        self.assert_error(
                            call,
                            RuntimeError,
                            f"unsupported range: 0 -> {rendered}",
                        )

    def test_oversized_endpoints_fail_before_allocation(self):
        cases = (
            (
                math.nextafter(float(2**63), 0.0),
                "Storage size calculation overflowed with sizes=[9223372036854774784]",
            ),
            (
                float(2**63),
                "IntArrayRef contains an int that cannot be represented as a SymInt: -9223372036854775808",
            ),
            (math.nextafter(float(2**63), math.inf), "invalid size, possible overflow?"),
            (1.0e100, "invalid size, possible overflow?"),
        )
        for end, message in cases:
            with self.subTest(end=end):
                self.assert_error(
                    lambda end=end: torch.arange(end), RuntimeError, message
                )

    def test_numpy_floating_oversized_endpoints_fail_before_allocation(self):
        cases = (
            (
                np.nextafter(np.float32(2**63), np.float32(0.0)),
                "Storage size calculation overflowed with sizes=[9223371487098961920]",
            ),
            (
                np.float32(2**63),
                "IntArrayRef contains an int that cannot be represented as a SymInt: -9223372036854775808",
            ),
            (
                np.nextafter(np.float32(2**63), np.float32(np.inf)),
                "invalid size, possible overflow?",
            ),
            (
                np.nextafter(np.float64(2**63), np.float64(0.0)),
                "Storage size calculation overflowed with sizes=[9223372036854774784]",
            ),
            (
                np.float64(2**63),
                "IntArrayRef contains an int that cannot be represented as a SymInt: -9223372036854775808",
            ),
            (
                np.nextafter(np.float64(2**63), np.float64(np.inf)),
                "invalid size, possible overflow?",
            ),
            (np.longdouble("1e100"), "invalid size, possible overflow?"),
        )
        for end, message in cases:
            for form, call in (
                ("positional", lambda end=end: torch.arange(end)),
                (
                    "keyword",
                    lambda end=end: torch.arange(
                        end=end, dtype=torch.float32
                    ),
                ),
            ):
                with self.subTest(end=end, form=form):
                    self.assert_error(call, RuntimeError, message)

    def test_explicit_float32_integer_boundaries_fail_before_allocation(self):
        cases = (
            (
                -(2**63),
                RuntimeError,
                "upper bound and lower bound inconsistent with step sign",
            ),
            (
                -(2**63) - 1,
                OverflowError,
                "can't convert negative int to unsigned",
            ),
            (
                2**63 - 1,
                RuntimeError,
                "IntArrayRef contains an int that cannot be represented as a SymInt: -9223372036854775808",
            ),
            (
                2**63,
                RuntimeError,
                "IntArrayRef contains an int that cannot be represented as a SymInt: -9223372036854775808",
            ),
            (2**64 - 1, RuntimeError, "invalid size, possible overflow?"),
            (2**64, OverflowError, "int too big to convert"),
        )
        for end, error_type, message in cases:
            for form, call in (
                (
                    "positional",
                    lambda end=end: torch.arange(end, dtype=torch.float32),
                ),
                (
                    "keyword",
                    lambda end=end: torch.arange(end=end, dtype=torch.float),
                ),
            ):
                with self.subTest(end=end, form=form):
                    self.assert_error(call, error_type, message)

    def test_two_bound_integer_empty_and_boundary_errors(self):
        for start, end in ((4, 1), (-1, -4), (2**63 + 2048, 2**63)):
            with self.subTest(start=start, end=end):
                self.assert_error(
                    lambda start=start, end=end: torch.arange(
                        start, end, dtype=torch.float32
                    ),
                    RuntimeError,
                    "upper bound and lower bound inconsistent with step sign",
                )

        zero_length_cases = (
            (3, 3),
            (2**63, 2**63 + 1),
            (2**64 - 1, 2**64 - 1),
            (-(2**63), -(2**63) + 2),
        )
        for start, end in zero_length_cases:
            with self.subTest(start=start, end=end):
                self.assert_default_tensor(
                    torch.arange(start, end, dtype=torch.float32), []
                )

        error_cases = (
            (
                -(2**63) - 1,
                0,
                OverflowError,
                "can't convert negative int to unsigned",
            ),
            (
                0,
                2**63 - 1,
                RuntimeError,
                "IntArrayRef contains an int that cannot be represented as a SymInt: -9223372036854775808",
            ),
            (
                0,
                2**63,
                RuntimeError,
                "IntArrayRef contains an int that cannot be represented as a SymInt: -9223372036854775808",
            ),
            (
                -(2**63),
                2**63 - 1,
                RuntimeError,
                "invalid size, possible overflow?",
            ),
            (
                2**64 - 1,
                2**64,
                OverflowError,
                "int too big to convert",
            ),
            (
                np.uint64(2**63),
                np.uint64(2**63 + 1),
                TypeError,
                "an integer is required",
            ),
        )
        for start, end, error_type, message in error_cases:
            with self.subTest(start=start, end=end):
                self.assert_error(
                    lambda start=start, end=end: torch.arange(
                        start, end, dtype=torch.float32
                    ),
                    error_type,
                    message,
                )

    def test_other_endpoint_types_remain_unsupported(self):
        calls = (
            lambda: torch.arange(3),
            lambda: torch.arange(end=3),
            lambda: torch.arange(3, dtype=None),
            lambda: torch.arange(end=3, dtype=None),
            lambda: torch.arange(True, dtype=torch.float32),
            lambda: torch.arange(True, 4, dtype=torch.float32),
            lambda: torch.arange(1, False, dtype=torch.float32),
            lambda: torch.arange(IntSubclass(3), dtype=torch.float32),
            lambda: torch.arange(IntSubclass(1), 4, dtype=torch.float32),
            lambda: torch.arange(1, IntSubclass(4), dtype=torch.float32),
            lambda: torch.arange(FloatSubclass(3.0), dtype=torch.float32),
            lambda: torch.arange(FloatSubclass(1.0), 4, dtype=torch.float32),
            lambda: torch.arange(1, FloatSubclass(4.0), dtype=torch.float32),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        for end in (
            np.bool_(True),
            np.int8(3),
            np.int64(3),
            np.uint64(3),
            np.complex64(3.0),
            np.complex128(3.0),
        ):
            for dtype in (None, torch.float32):
                options = {} if dtype is None else {"dtype": dtype}
                for form, call in (
                    (
                        "positional",
                        lambda end=end, options=options: torch.arange(
                            end, **options
                        ),
                    ),
                    (
                        "keyword",
                        lambda end=end, options=options: torch.arange(
                            end=end, **options
                        ),
                    ),
                ):
                    with self.subTest(
                        end_type=type(end).__name__,
                        dtype=dtype,
                        form=form,
                    ):
                        with self.assertRaises(TypeError):
                            call()

        for start, end in (
            (np.bool_(True), np.int64(4)),
            (np.int64(1), np.bool_(False)),
            (np.complex64(1.0), np.int64(4)),
            (np.int64(1), np.complex128(4.0)),
        ):
            with self.subTest(start_type=type(start).__name__, end_type=type(end).__name__):
                with self.assertRaises(TypeError):
                    torch.arange(start, end, dtype=torch.float32)

    def test_spoofed_numpy_floating_type_is_rejected_without_conversion(self):
        for dtype in (None, torch.float32, torch.float):
            options = {} if dtype is None else {"dtype": dtype}
            for form in ("positional", "keyword"):
                endpoint = SpoofedNumpyFloat()
                self.assertTrue(isinstance(endpoint, np.generic))
                self.assertTrue(isinstance(endpoint, np.floating))
                position = " (position 1)" if form == "positional" else ""
                if form == "positional":
                    call = lambda: torch.arange(endpoint, **options)
                else:
                    call = lambda: torch.arange(end=endpoint, **options)
                with self.subTest(dtype=dtype, form=form):
                    self.assert_error(
                        call,
                        TypeError,
                        f"arange(): argument 'end'{position} must be an exact Python float, not SpoofedNumpyFloat",
                    )
                    self.assertEqual(endpoint.float_calls, 0)

    def test_overloads_outputs_and_nondefault_options_remain_unsupported(self):
        overloads = (
            lambda: torch.arange(0.0, 3.0),
            lambda: torch.arange(0.0, 3.0, 1.0),
            lambda: torch.arange(2.5, end=3.0),
            lambda: torch.arange(start=0.0, end=3.0),
            lambda: torch.arange(3.0, step=1.0),
            lambda: torch.arange(0, 3),
            lambda: torch.arange(0, 3, dtype=None),
            lambda: torch.arange(3, step=1, dtype=torch.float32),
            lambda: torch.arange(0, 3, 1, dtype=torch.float32),
            lambda: torch.arange(start=0, end=3, step=1, dtype=torch.float32),
            lambda: torch.arange(np.float32(0.0), np.float32(3.0)),
            lambda: torch.arange(
                np.float32(0.0), np.float32(3.0), np.float32(1.0)
            ),
            lambda: torch.arange(
                np.float32(2.5), end=np.float32(3.0)
            ),
            lambda: torch.arange(
                start=np.float32(0.0), end=np.float32(3.0)
            ),
            lambda: torch.arange(np.float32(3.0), step=np.float32(1.0)),
        )
        for call in overloads:
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^arange\(\): (only one-bound float endpoints and "
                    r"two-bound integer endpoints with explicit "
                    r"dtype=torch\.float32 are supported|explicit step is "
                    r"not supported|two-bound integer ranges require "
                    r"explicit dtype=torch\.float32|argument 'start'.*must "
                    r"be an exact Python or NumPy integer)",
                ):
                    call()

        destination = torch.full((3,), 9.0)
        for call in (
            lambda: torch.arange(2.5, out=destination),
            lambda: torch.arange(3, dtype=torch.float32, out=destination),
            lambda: torch.arange(0, 3, dtype=torch.float32, out=destination),
            lambda: torch.arange(np.float32(2.5), out=destination),
            lambda: torch.arange(
                end=np.longdouble(3.0),
                dtype=torch.float32,
                out=destination,
            ),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^arange\(\): the 'out' argument is not supported$",
                ):
                    call()
        self.assertEqual(destination.tolist(), [9.0, 9.0, 9.0])

        unsupported_options = (
            lambda: torch.arange(2.5, dtype=object()),
            lambda: torch.arange(2.5, layout=object()),
            lambda: torch.arange(2.5, device="cuda"),
            lambda: torch.arange(2.5, pin_memory=True),
            lambda: torch.arange(3, dtype=torch.float32, layout=object()),
            lambda: torch.arange(3, dtype=torch.float32, device="cuda"),
            lambda: torch.arange(3, dtype=torch.float32, pin_memory=True),
            lambda: torch.arange(0, 3, dtype=object()),
            lambda: torch.arange(0, 3, dtype=torch.float32, layout=object()),
            lambda: torch.arange(0, 3, dtype=torch.float32, device="cuda"),
            lambda: torch.arange(0, 3, dtype=torch.float32, pin_memory=True),
            lambda: torch.arange(np.float16(2.5), dtype=object()),
            lambda: torch.arange(np.float32(2.5), layout=object()),
            lambda: torch.arange(np.float64(2.5), device="cuda"),
            lambda: torch.arange(
                np.longdouble(2.5),
                dtype=torch.float32,
                device=torch.device("cuda"),
            ),
            lambda: torch.arange(np.float32(2.5), pin_memory=True),
        )
        for call in unsupported_options:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, RuntimeError)):
                    call()

    def test_requires_grad_true_preserves_existing_error_precedence(self):
        destination = torch.full((3,), 9.0)
        cases = (
            (
                lambda: torch.arange(3, requires_grad=True),
                TypeError,
                "arange(): argument 'end' (position 1) must be an exact Python float, not int",
            ),
            (
                lambda: torch.arange(3, dtype=object(), requires_grad=object()),
                TypeError,
                "arange(): argument 'end' (position 1) must be an exact Python float, not int",
            ),
            (
                lambda: torch.arange(0.0, 3.0, requires_grad=True),
                TypeError,
                "arange(): argument 'start' (position 1) must be an exact Python or NumPy integer, not float",
            ),
            (
                lambda: torch.arange(0, 3, requires_grad=True),
                TypeError,
                "arange(): two-bound integer ranges require explicit dtype=torch.float32",
            ),
            (
                lambda: torch.arange(
                    0, 3, step=1, dtype=torch.float32, requires_grad=True
                ),
                TypeError,
                "arange(): explicit step is not supported; only implicit step=1 is implemented",
            ),
            (
                lambda: torch.arange(2.5, out=destination, requires_grad=True),
                RuntimeError,
                "arange(): the 'out' argument is not supported",
            ),
            (
                lambda: torch.arange(
                    0,
                    3,
                    dtype=torch.float32,
                    out=destination,
                    requires_grad=True,
                ),
                RuntimeError,
                "arange(): the 'out' argument is not supported",
            ),
            (
                lambda: torch.arange(2.5, device="cuda", requires_grad=True),
                RuntimeError,
                "arange(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.arange(
                    0,
                    3,
                    dtype=torch.float32,
                    device="cuda",
                    requires_grad=True,
                ),
                RuntimeError,
                "arange(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.arange(2.5, pin_memory=True, requires_grad=True),
                RuntimeError,
                "arange(): pin_memory=True is not supported; only unpinned CPU storage is implemented",
            ),
            (
                lambda: torch.arange(
                    0,
                    3,
                    dtype=torch.float32,
                    pin_memory=True,
                    requires_grad=True,
                ),
                RuntimeError,
                "arange(): pin_memory=True is not supported; only unpinned CPU storage is implemented",
            ),
            (
                lambda: torch.arange(-1.0, requires_grad=True),
                RuntimeError,
                "upper bound and lower bound inconsistent with step sign",
            ),
            (
                lambda: torch.arange(
                    3, 0, dtype=torch.float32, requires_grad=True
                ),
                RuntimeError,
                "upper bound and lower bound inconsistent with step sign",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                self.assert_error(call, error_type, message)
        self.assertEqual(destination.tolist(), [9.0, 9.0, 9.0])

    def test_torch_function_mode_intercepts_raw_calls_before_native_validation(self):
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func,
                        types,
                        args,
                        kwargs,
                        tuple(torch.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        cases = (
            (lambda: torch.arange(2.5), (2.5,), None),
            (lambda: torch.arange(end=2.5), (), {"end": 2.5}),
            (lambda: torch.arange(3), (3,), None),
            (
                lambda: torch.arange(3, dtype=torch.float32),
                (3,),
                {"dtype": torch.float32},
            ),
            (lambda: torch.arange(0.0, 3.0), (0.0, 3.0), None),
        )
        for call, expected_args, expected_kwargs in cases:
            mode = RecordingMode()
            with self.subTest(args=expected_args, kwargs=expected_kwargs):
                with mode:
                    self.assertIs(call(), marker)
                    self.assertEqual(
                        torch.overrides._get_current_function_mode_stack(), [mode]
                    )
                self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs, handler_stack = mode.calls[0]
                self.assertIs(function, torch.arange)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)
                self.assertEqual(handler_stack, ())

    def test_torch_function_mode_forwards_nested_calls_and_restores_the_stack(self):
        events = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(
                    (
                        self.label,
                        tuple(
                            mode.label
                            for mode in torch.overrides._get_current_function_mode_stack()
                        ),
                        func,
                        types,
                        args,
                        kwargs,
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                result = torch.arange(end=2.5)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), [lower, upper]
                )
            self.assertEqual(torch.overrides._get_current_function_mode_stack(), [lower])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])
        self.assert_default_tensor(result, [0.0, 1.0, 2.0])
        self.assertEqual(
            [
                (label, stack, function is torch.arange, types, args, kwargs)
                for label, stack, function, types, args, kwargs in events
            ],
            [
                ("upper", ("lower",), True, (), (), {"end": 2.5}),
                ("lower", (), True, (), (), {"end": 2.5}),
            ],
        )

        expected = ValueError("handler failed")

        class RaisingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise expected

        raising = RaisingMode()
        with lower:
            with raising:
                with self.assertRaises(ValueError) as raised:
                    torch.arange(2.5)
                self.assertIs(raised.exception, expected)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), [lower, raising]
                )
            self.assertEqual(torch.overrides._get_current_function_mode_stack(), [lower])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        forwarding = ForwardingMode("native-error")
        with forwarding:
            with self.assertRaisesRegex(
                RuntimeError,
                "^upper bound and lower bound inconsistent with step sign$",
            ):
                torch.arange(-1.0)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [forwarding]
            )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        declining = DecliningMode()
        with declining:
            with self.assertRaisesRegex(
                TypeError,
                r"^Multiple dispatch failed for 'torch\.arange'; all "
                r"__torch_function__ handlers returned NotImplemented:",
            ):
                torch.arange(2.5)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [declining]
            )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_callable_metadata_exports_and_pickling_match_generated_builtins(self):
        function = torch.arange
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "arange")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.arange")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method arange of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.arange, function)
        self.assertEqual(torch.__all__.count("arange"), 1)
        self.assertIs(wildcard_namespace["arange"], function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)


if __name__ == "__main__":
    unittest.main()
