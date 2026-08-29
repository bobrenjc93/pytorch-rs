import inspect
import math
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


NUMPY_FLOAT_TYPES = (np.float16, np.float32, np.float64, np.longdouble)


class NumpyFloatSubclass(np.float32):
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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ArangeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("arange differentials require pinned PyTorch 2.13.0")

    def tensor_contract(self, module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "values": tensor.tolist(),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is module.float32,
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
        }

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_values_shapes_and_default_metadata_match_pytorch_2_13(self):
        endpoints = (
            0.0,
            -0.0,
            math.nextafter(0.0, 1.0),
            0.25,
            math.nextafter(1.0, 0.0),
            1.0,
            math.nextafter(1.0, 2.0),
            2.5,
            8.0,
        )
        for end in endpoints:
            for form in ("positional", "keyword"):
                with self.subTest(end=end, form=form):
                    if form == "positional":
                        actual = torch.arange(end)
                        expected = reference_torch.arange(end)
                    else:
                        actual = torch.arange(end=end)
                        expected = reference_torch.arange(end=end)
                    self.assertEqual(
                        self.tensor_contract(torch, actual),
                        self.tensor_contract(reference_torch, expected),
                    )

    def test_default_equivalent_exact_float_start_step_forms_match_pytorch_2_13(self):
        endpoints = (
            0.0,
            -0.0,
            0.25,
            1.0,
            2.5,
            8.0,
        )
        for end in endpoints:
            for form in (
                "two positional",
                "three positional float step",
                "three positional int step",
                "keyword start end",
                "keyword start end step",
                "positional start keyword end",
                "keyword step before end",
                "positional start end keyword step",
            ):
                with self.subTest(end=end, form=form):
                    if form == "two positional":
                        actual = torch.arange(0.0, end)
                        expected = reference_torch.arange(0.0, end)
                    elif form == "three positional float step":
                        actual = torch.arange(0.0, end, 1.0)
                        expected = reference_torch.arange(0.0, end, 1.0)
                    elif form == "three positional int step":
                        actual = torch.arange(0.0, end, 1)
                        expected = reference_torch.arange(0.0, end, 1)
                    elif form == "keyword start end":
                        actual = torch.arange(start=0.0, end=end)
                        expected = reference_torch.arange(start=0.0, end=end)
                    elif form == "keyword start end step":
                        actual = torch.arange(start=-0.0, end=end, step=1.0)
                        expected = reference_torch.arange(
                            start=-0.0, end=end, step=1.0
                        )
                    elif form == "positional start keyword end":
                        actual = torch.arange(0.0, end=end)
                        expected = reference_torch.arange(0.0, end=end)
                    elif form == "keyword step before end":
                        actual = torch.arange(0.0, step=1.0, end=end)
                        expected = reference_torch.arange(
                            0.0, step=1.0, end=end
                        )
                    else:
                        actual = torch.arange(0.0, end, step=1)
                        expected = reference_torch.arange(0.0, end, step=1)
                    self.assertEqual(
                        self.tensor_contract(torch, actual),
                        self.tensor_contract(reference_torch, expected),
                    )

    def test_numpy_floating_one_bound_forms_match_pytorch_2_13(self):
        scalar_types = (*NUMPY_FLOAT_TYPES, NumpyFloatSubclass)
        for scalar_type in scalar_types:
            endpoints = tuple(
                scalar_type(value)
                for value in (0.0, -0.0, 0.25, 1.0, 2.5, 8.0)
            )
            for end in endpoints:
                for dtype_name in (None, "float32", "float"):
                    actual_options = (
                        {}
                        if dtype_name is None
                        else {"dtype": getattr(torch, dtype_name)}
                    )
                    expected_options = (
                        {}
                        if dtype_name is None
                        else {"dtype": getattr(reference_torch, dtype_name)}
                    )
                    for form in ("positional", "keyword"):
                        with self.subTest(
                            scalar_type=scalar_type.__name__,
                            end=end,
                            dtype=dtype_name,
                            form=form,
                        ):
                            if form == "positional":
                                actual = torch.arange(end, **actual_options)
                                expected = reference_torch.arange(
                                    end, **expected_options
                                )
                            else:
                                actual = torch.arange(
                                    end=end, **actual_options
                                )
                                expected = reference_torch.arange(
                                    end=end, **expected_options
                                )
                            self.assertEqual(
                                self.tensor_contract(torch, actual),
                                self.tensor_contract(reference_torch, expected),
                            )

    def test_numpy_floating_default_equivalent_start_step_forms_match_pytorch_2_13(self):
        scalar_types = (*NUMPY_FLOAT_TYPES, NumpyFloatSubclass)
        for scalar_type in scalar_types:
            endpoints = tuple(
                scalar_type(value)
                for value in (0.0, -0.0, 0.25, 1.0, 2.5, 8.0)
            )
            for end in endpoints:
                start = scalar_type(0.0)
                step = scalar_type(1.0)
                for dtype_name in (None, "float32", "float"):
                    actual_options = (
                        {}
                        if dtype_name is None
                        else {"dtype": getattr(torch, dtype_name)}
                    )
                    expected_options = (
                        {}
                        if dtype_name is None
                        else {"dtype": getattr(reference_torch, dtype_name)}
                    )
                    for form in ("two positional", "three positional", "keywords"):
                        with self.subTest(
                            scalar_type=scalar_type.__name__,
                            end=end,
                            dtype=dtype_name,
                            form=form,
                        ):
                            if form == "two positional":
                                actual = torch.arange(
                                    start, end, **actual_options
                                )
                                expected = reference_torch.arange(
                                    start, end, **expected_options
                                )
                            elif form == "three positional":
                                actual = torch.arange(
                                    start, end, step, **actual_options
                                )
                                expected = reference_torch.arange(
                                    start, end, step, **expected_options
                                )
                            else:
                                actual = torch.arange(
                                    start=start,
                                    end=end,
                                    step=step,
                                    **actual_options,
                                )
                                expected = reference_torch.arange(
                                    start=start,
                                    end=end,
                                    step=step,
                                    **expected_options,
                                )
                            self.assertEqual(
                                self.tensor_contract(torch, actual),
                                self.tensor_contract(reference_torch, expected),
                            )

    def test_spoofed_numpy_floating_type_rejection_matches_pytorch_2_13(self):
        def outcome(module, endpoint, dtype_name, form):
            options = (
                {}
                if dtype_name is None
                else {"dtype": getattr(module, dtype_name)}
            )
            try:
                if form == "positional":
                    module.arange(endpoint, **options)
                else:
                    module.arange(end=endpoint, **options)
            except Exception as error:
                return type(error).__name__, endpoint.float_calls
            return "accepted", endpoint.float_calls

        for dtype_name in (None, "float32", "float"):
            for form in ("positional", "keyword"):
                actual_endpoint = SpoofedNumpyFloat()
                expected_endpoint = SpoofedNumpyFloat()
                self.assertTrue(isinstance(actual_endpoint, np.generic))
                self.assertTrue(isinstance(actual_endpoint, np.floating))
                with self.subTest(dtype=dtype_name, form=form):
                    actual = outcome(
                        torch, actual_endpoint, dtype_name, form
                    )
                    expected = outcome(
                        reference_torch, expected_endpoint, dtype_name, form
                    )
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual, ("TypeError", 0))

    def test_explicit_float32_integer_endpoints_match_pytorch_2_13(self):
        for end in (0, 1, 3, 8):
            for dtype_name in ("float32", "float"):
                actual_dtype = getattr(torch, dtype_name)
                expected_dtype = getattr(reference_torch, dtype_name)
                for form in ("positional", "keyword"):
                    with self.subTest(
                        end=end, dtype=dtype_name, form=form
                    ):
                        if form == "positional":
                            actual = torch.arange(end, dtype=actual_dtype)
                            expected = reference_torch.arange(
                                end, dtype=expected_dtype
                            )
                        else:
                            actual = torch.arange(
                                end=end, dtype=actual_dtype
                            )
                            expected = reference_torch.arange(
                                end=end, dtype=expected_dtype
                            )
                        self.assertEqual(
                            self.tensor_contract(torch, actual),
                            self.tensor_contract(reference_torch, expected),
                        )

    def test_explicit_float32_integer_start_step_forms_match_pytorch_2_13(self):
        for end in (0, 1, 3, 8):
            for dtype_name in ("float32", "float"):
                actual_dtype = getattr(torch, dtype_name)
                expected_dtype = getattr(reference_torch, dtype_name)
                for form in (
                    "two positional",
                    "three positional",
                    "keywords",
                    "positional start keyword end",
                ):
                    with self.subTest(end=end, dtype=dtype_name, form=form):
                        if form == "two positional":
                            actual = torch.arange(
                                0, end, dtype=actual_dtype
                            )
                            expected = reference_torch.arange(
                                0, end, dtype=expected_dtype
                            )
                        elif form == "three positional":
                            actual = torch.arange(
                                0, end, 1, dtype=actual_dtype
                            )
                            expected = reference_torch.arange(
                                0, end, 1, dtype=expected_dtype
                            )
                        elif form == "keywords":
                            actual = torch.arange(
                                start=0,
                                end=end,
                                step=1,
                                dtype=actual_dtype,
                            )
                            expected = reference_torch.arange(
                                start=0,
                                end=end,
                                step=1,
                                dtype=expected_dtype,
                            )
                        else:
                            actual = torch.arange(
                                0, end=end, dtype=actual_dtype
                            )
                            expected = reference_torch.arange(
                                0, end=end, dtype=expected_dtype
                            )
                        self.assertEqual(
                            self.tensor_contract(torch, actual),
                            self.tensor_contract(reference_torch, expected),
                        )

    def test_default_equivalent_options_match_pytorch_2_13(self):
        option_factories = (
            lambda module: {},
            lambda module: {"out": None},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": "cpu:0"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"pin_memory": None},
            lambda module: {"pin_memory": False},
            lambda module: {"requires_grad": None},
            lambda module: {"requires_grad": False},
        )
        for option_factory in option_factories:
            actual_options = option_factory(torch)
            expected_options = option_factory(reference_torch)
            for form in ("one-bound", "two-bound", "three-bound"):
                with self.subTest(options=actual_options, form=form):
                    if form == "one-bound":
                        actual = torch.arange(2.5, **actual_options)
                        expected = reference_torch.arange(
                            2.5, **expected_options
                        )
                    elif form == "two-bound":
                        actual = torch.arange(0.0, 2.5, **actual_options)
                        expected = reference_torch.arange(
                            0.0, 2.5, **expected_options
                        )
                    else:
                        actual = torch.arange(0.0, 2.5, 1.0, **actual_options)
                        expected = reference_torch.arange(
                            0.0, 2.5, 1.0, **expected_options
                        )
                    self.assertEqual(
                        self.tensor_contract(torch, actual),
                        self.tensor_contract(reference_torch, expected),
                    )

    def test_requires_grad_leaves_and_accumulation_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            ordinary = module.arange(4.0, requires_grad=True)
            with module.no_grad():
                no_grad = module.arange(end=4.0, requires_grad=True)
                empty = module.arange(-0.0, requires_grad=True)

            module_outcomes = []
            weights = module.tensor(
                [1.0, 2.0, 3.0, 4.0], dtype=module.float32
            )
            for leaf in (ordinary, no_grad):
                gradients = []
                for _ in range(2):
                    (leaf * weights).sum().backward()
                    gradients.append(leaf.grad.tolist())
                module_outcomes.append(
                    (self.tensor_contract(module, leaf), gradients)
                )
            module_outcomes.append(self.tensor_contract(module, empty))
            outcomes.append(module_outcomes)

        self.assertEqual(outcomes[0], outcomes[1])

    def test_default_equivalent_start_step_leaf_semantics_match_pytorch_2_13(self):
        cases = (
            lambda module: module.arange(0.0, 4.0, requires_grad=True),
            lambda module: module.arange(0.0, 4.0, 1.0, requires_grad=True),
            lambda module: module.arange(
                start=0.0, end=4.0, step=1.0, requires_grad=True
            ),
            lambda module: module.arange(
                0, 4, 1, dtype=module.float32, requires_grad=True
            ),
        )
        empty_cases = (
            lambda module: module.arange(0.0, 0.0, requires_grad=True),
            lambda module: module.arange(
                start=-0.0, end=-0.0, step=1, requires_grad=True
            ),
            lambda module: module.arange(
                0, 0, dtype=module.float32, requires_grad=True
            ),
        )
        for index, make_leaf in enumerate(cases):
            outcomes = []
            for module in (torch, reference_torch):
                with module.no_grad():
                    leaf = make_leaf(module)
                weights = module.tensor(
                    [1.0, 2.0, 3.0, 4.0], dtype=module.float32
                )
                gradients = []
                for _ in range(2):
                    (leaf * weights).sum().backward()
                    gradients.append(leaf.grad.tolist())
                outcomes.append((self.tensor_contract(module, leaf), gradients))

            with self.subTest(case=index):
                self.assertEqual(outcomes[0], outcomes[1])

        for index, make_empty in enumerate(empty_cases):
            outcomes = []
            for module in (torch, reference_torch):
                with module.no_grad():
                    outcomes.append(self.tensor_contract(module, make_empty(module)))
            with self.subTest(empty_case=index):
                self.assertEqual(outcomes[0], outcomes[1])

    def test_numpy_floating_leaf_semantics_match_pytorch_2_13(self):
        for scalar_type in NUMPY_FLOAT_TYPES:
            outcomes = []
            for module in (torch, reference_torch):
                ordinary = module.arange(
                    scalar_type(4.0), requires_grad=True
                )
                with module.no_grad():
                    no_grad = module.arange(
                        end=scalar_type(4.0),
                        dtype=module.float32,
                        requires_grad=True,
                    )
                    empty = module.arange(
                        scalar_type(-0.0),
                        dtype=module.float,
                        requires_grad=True,
                    )

                module_outcomes = []
                weights = module.tensor(
                    [1.0, 2.0, 3.0, 4.0], dtype=module.float32
                )
                for leaf in (ordinary, no_grad):
                    gradients = []
                    for _ in range(2):
                        (leaf * weights).sum().backward()
                        gradients.append(leaf.grad.tolist())
                    module_outcomes.append(
                        (self.tensor_contract(module, leaf), gradients)
                    )
                module_outcomes.append(self.tensor_contract(module, empty))
                outcomes.append(module_outcomes)

            with self.subTest(scalar_type=scalar_type.__name__):
                self.assertEqual(outcomes[0], outcomes[1])

    def test_explicit_float32_integer_leaf_semantics_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            ordinary = module.arange(
                4, dtype=module.float32, requires_grad=True
            )
            with module.no_grad():
                no_grad = module.arange(
                    end=4, dtype=module.float, requires_grad=True
                )
                empty = module.arange(
                    0, dtype=module.float32, requires_grad=True
                )

            weights = module.tensor(
                [1.0, 2.0, 3.0, 4.0], dtype=module.float32
            )
            module_outcomes = []
            for leaf in (ordinary, no_grad):
                gradients = []
                for _ in range(2):
                    (leaf * weights).sum().backward()
                    gradients.append(leaf.grad.tolist())
                module_outcomes.append(
                    (self.tensor_contract(module, leaf), gradients)
                )
            module_outcomes.append(self.tensor_contract(module, empty))
            outcomes.append(module_outcomes)

        self.assertEqual(outcomes[0], outcomes[1])

    def test_fresh_storage_matches_pytorch_2_13(self):
        for requires_grad in (False, True):
            with self.subTest(requires_grad=requires_grad):
                actual_first = torch.arange(8.5, requires_grad=requires_grad)
                actual_second = torch.arange(8.5, requires_grad=requires_grad)
                expected_first = reference_torch.arange(
                    8.5, requires_grad=requires_grad
                )
                expected_second = reference_torch.arange(
                    8.5, requires_grad=requires_grad
                )
                self.assertEqual(
                    actual_first.data_ptr() != actual_second.data_ptr(),
                    expected_first.data_ptr() != expected_second.data_ptr(),
                )
                self.assertEqual(
                    actual_first.is_set_to(actual_second),
                    expected_first.is_set_to(expected_second),
                )

                actual_empty_first = torch.arange(
                    0.0, requires_grad=requires_grad
                )
                actual_empty_second = torch.arange(
                    -0.0, requires_grad=requires_grad
                )
                expected_empty_first = reference_torch.arange(
                    0.0, requires_grad=requires_grad
                )
                expected_empty_second = reference_torch.arange(
                    -0.0, requires_grad=requires_grad
                )
                self.assertEqual(
                    actual_empty_first.is_set_to(actual_empty_second),
                    expected_empty_first.is_set_to(expected_empty_second),
                )

    def test_default_equivalent_start_step_storage_matches_pytorch_2_13(self):
        cases = (
            (
                lambda module, requires_grad: module.arange(
                    0.0, 8.5, requires_grad=requires_grad
                ),
                lambda module, requires_grad: module.arange(
                    start=-0.0, end=8.5, requires_grad=requires_grad
                ),
                lambda module, requires_grad: module.arange(
                    0.0, 0.0, requires_grad=requires_grad
                ),
                lambda module, requires_grad: module.arange(
                    start=-0.0, end=-0.0, requires_grad=requires_grad
                ),
            ),
            (
                lambda module, requires_grad: module.arange(
                    0.0, 8.5, 1.0, requires_grad=requires_grad
                ),
                lambda module, requires_grad: module.arange(
                    start=0.0, end=8.5, step=1, requires_grad=requires_grad
                ),
                lambda module, requires_grad: module.arange(
                    0.0, 0.0, 1.0, requires_grad=requires_grad
                ),
                lambda module, requires_grad: module.arange(
                    -0.0, -0.0, 1, requires_grad=requires_grad
                ),
            ),
            (
                lambda module, requires_grad: module.arange(
                    0, 8, dtype=module.float32, requires_grad=requires_grad
                ),
                lambda module, requires_grad: module.arange(
                    0, 8, 1, dtype=module.float, requires_grad=requires_grad
                ),
                lambda module, requires_grad: module.arange(
                    0, 0, dtype=module.float32, requires_grad=requires_grad
                ),
                lambda module, requires_grad: module.arange(
                    0, 0, 1, dtype=module.float, requires_grad=requires_grad
                ),
            ),
        )
        for case_index, (
            make_first,
            make_second,
            make_empty_first,
            make_empty_second,
        ) in enumerate(cases):
            outcomes = []
            for module in (torch, reference_torch):
                module_outcomes = []
                for requires_grad in (False, True):
                    first = make_first(module, requires_grad)
                    second = make_second(module, requires_grad)
                    empty_first = make_empty_first(module, requires_grad)
                    empty_second = make_empty_second(module, requires_grad)
                    module_outcomes.append(
                        (
                            first.data_ptr() != second.data_ptr(),
                            first.is_set_to(second),
                            empty_first.data_ptr() == 0,
                            empty_second.data_ptr() == 0,
                            empty_first.is_set_to(empty_second),
                        )
                    )
                outcomes.append(module_outcomes)
            with self.subTest(case=case_index):
                self.assertEqual(outcomes[0], outcomes[1])

    def test_numpy_floating_fresh_storage_matches_pytorch_2_13(self):
        for scalar_type in NUMPY_FLOAT_TYPES:
            outcomes = []
            for module in (torch, reference_torch):
                module_outcomes = []
                for requires_grad in (False, True):
                    first = module.arange(
                        scalar_type(8.5), requires_grad=requires_grad
                    )
                    second = module.arange(
                        end=scalar_type(8.5),
                        dtype=module.float32,
                        requires_grad=requires_grad,
                    )
                    empty_first = module.arange(
                        scalar_type(0.0), requires_grad=requires_grad
                    )
                    empty_second = module.arange(
                        end=scalar_type(-0.0),
                        dtype=module.float,
                        requires_grad=requires_grad,
                    )
                    module_outcomes.append(
                        (
                            first.data_ptr() != second.data_ptr(),
                            first.is_set_to(second),
                            empty_first.data_ptr() == 0,
                            empty_second.data_ptr() == 0,
                            empty_first.is_set_to(empty_second),
                        )
                    )
                outcomes.append(module_outcomes)

            with self.subTest(scalar_type=scalar_type.__name__):
                self.assertEqual(outcomes[0], outcomes[1])

    def test_explicit_float32_integer_storage_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            module_outcomes = []
            for requires_grad in (False, True):
                first = module.arange(
                    8,
                    dtype=module.float32,
                    requires_grad=requires_grad,
                )
                second = module.arange(
                    end=8,
                    dtype=module.float,
                    requires_grad=requires_grad,
                )
                empty_first = module.arange(
                    0,
                    dtype=module.float32,
                    requires_grad=requires_grad,
                )
                empty_second = module.arange(
                    end=0,
                    dtype=module.float,
                    requires_grad=requires_grad,
                )
                module_outcomes.append(
                    (
                        first.data_ptr() != second.data_ptr(),
                        first.is_set_to(second),
                        empty_first.data_ptr() == 0,
                        empty_second.data_ptr() == 0,
                        empty_first.is_set_to(empty_second),
                    )
                )
            outcomes.append(module_outcomes)

        self.assertEqual(outcomes[0], outcomes[1])

    def test_negative_and_nonfinite_errors_match_pytorch_2_13(self):
        endpoints = (
            -math.nextafter(0.0, 1.0),
            -0.25,
            -1.0,
            float("nan"),
            float("-nan"),
            float("inf"),
            float("-inf"),
        )
        for end in endpoints:
            for form in ("positional", "keyword", "two-bound", "three-bound"):
                with self.subTest(end=end, form=form):
                    if form == "positional":
                        self.assert_error_matches(
                            lambda end=end: torch.arange(end),
                            lambda end=end: reference_torch.arange(end),
                        )
                    elif form == "keyword":
                        self.assert_error_matches(
                            lambda end=end: torch.arange(end=end),
                            lambda end=end: reference_torch.arange(end=end),
                        )
                    elif form == "two-bound":
                        self.assert_error_matches(
                            lambda end=end: torch.arange(0.0, end),
                            lambda end=end: reference_torch.arange(0.0, end),
                        )
                    else:
                        self.assert_error_matches(
                            lambda end=end: torch.arange(0.0, end, 1.0),
                            lambda end=end: reference_torch.arange(0.0, end, 1.0),
                        )

        for dtype_name in ("float32", "float"):
            actual_dtype = getattr(torch, dtype_name)
            expected_dtype = getattr(reference_torch, dtype_name)
            for form in ("positional", "keyword", "two-bound", "three-bound"):
                with self.subTest(end=-1, dtype=dtype_name, form=form):
                    if form == "positional":
                        self.assert_error_matches(
                            lambda dtype=actual_dtype: torch.arange(
                                -1, dtype=dtype
                            ),
                            lambda dtype=expected_dtype: reference_torch.arange(
                                -1, dtype=dtype
                            ),
                        )
                    elif form == "keyword":
                        self.assert_error_matches(
                            lambda dtype=actual_dtype: torch.arange(
                                end=-1, dtype=dtype
                            ),
                            lambda dtype=expected_dtype: reference_torch.arange(
                                end=-1, dtype=dtype
                            ),
                        )
                    elif form == "two-bound":
                        self.assert_error_matches(
                            lambda dtype=actual_dtype: torch.arange(
                                0, -1, dtype=dtype
                            ),
                            lambda dtype=expected_dtype: reference_torch.arange(
                                0, -1, dtype=dtype
                            ),
                        )
                    else:
                        self.assert_error_matches(
                            lambda dtype=actual_dtype: torch.arange(
                                0, -1, 1, dtype=dtype
                            ),
                            lambda dtype=expected_dtype: reference_torch.arange(
                                0, -1, 1, dtype=dtype
                            ),
                        )

    def test_numpy_floating_negative_and_nonfinite_errors_match_pytorch_2_13(self):
        for scalar_type in NUMPY_FLOAT_TYPES:
            endpoints = (
                scalar_type(-0.25),
                scalar_type("nan"),
                scalar_type("-nan"),
                scalar_type("inf"),
                scalar_type("-inf"),
            )
            for end in endpoints:
                for dtype_name in (None, "float32", "float"):
                    actual_options = (
                        {}
                        if dtype_name is None
                        else {"dtype": getattr(torch, dtype_name)}
                    )
                    expected_options = (
                        {}
                        if dtype_name is None
                        else {"dtype": getattr(reference_torch, dtype_name)}
                    )
                    for form in ("positional", "keyword", "two-bound", "three-bound"):
                        with self.subTest(
                            scalar_type=scalar_type.__name__,
                            end=end,
                            dtype=dtype_name,
                            form=form,
                        ):
                            if form == "positional":
                                self.assert_error_matches(
                                    lambda end=end, options=actual_options: torch.arange(
                                        end, **options
                                    ),
                                    lambda end=end, options=expected_options: reference_torch.arange(
                                        end, **options
                                    ),
                                )
                            elif form == "keyword":
                                self.assert_error_matches(
                                    lambda end=end, options=actual_options: torch.arange(
                                        end=end, **options
                                    ),
                                    lambda end=end, options=expected_options: reference_torch.arange(
                                        end=end, **options
                                    ),
                                )
                            elif form == "two-bound":
                                self.assert_error_matches(
                                    lambda scalar_type=scalar_type, end=end, options=actual_options: torch.arange(
                                        scalar_type(0.0), end, **options
                                    ),
                                    lambda scalar_type=scalar_type, end=end, options=expected_options: reference_torch.arange(
                                        scalar_type(0.0), end, **options
                                    ),
                                )
                            else:
                                self.assert_error_matches(
                                    lambda scalar_type=scalar_type, end=end, options=actual_options: torch.arange(
                                        scalar_type(0.0),
                                        end,
                                        scalar_type(1.0),
                                        **options,
                                    ),
                                    lambda scalar_type=scalar_type, end=end, options=expected_options: reference_torch.arange(
                                        scalar_type(0.0),
                                        end,
                                        scalar_type(1.0),
                                        **options,
                                    ),
                                )

    def test_oversized_endpoint_errors_match_pytorch_2_13(self):
        endpoints = (
            math.nextafter(float(2**63), 0.0),
            float(2**63),
            math.nextafter(float(2**63), math.inf),
            1.0e100,
        )
        for end in endpoints:
            for form in ("one-bound", "two-bound", "three-bound"):
                with self.subTest(end=end, form=form):
                    if form == "one-bound":
                        self.assert_error_matches(
                            lambda end=end: torch.arange(end),
                            lambda end=end: reference_torch.arange(end),
                        )
                    elif form == "two-bound":
                        self.assert_error_matches(
                            lambda end=end: torch.arange(0.0, end),
                            lambda end=end: reference_torch.arange(0.0, end),
                        )
                    else:
                        self.assert_error_matches(
                            lambda end=end: torch.arange(0.0, end, 1.0),
                            lambda end=end: reference_torch.arange(0.0, end, 1.0),
                        )

    def test_numpy_floating_oversized_errors_match_pytorch_2_13(self):
        endpoints = (
            np.nextafter(np.float32(2**63), np.float32(0.0)),
            np.float32(2**63),
            np.nextafter(np.float32(2**63), np.float32(np.inf)),
            np.nextafter(np.float64(2**63), np.float64(0.0)),
            np.float64(2**63),
            np.nextafter(np.float64(2**63), np.float64(np.inf)),
            np.longdouble("1e100"),
        )
        for end in endpoints:
            for dtype_name in (None, "float32"):
                actual_options = (
                    {}
                    if dtype_name is None
                    else {"dtype": torch.float32}
                )
                expected_options = (
                    {}
                    if dtype_name is None
                    else {"dtype": reference_torch.float32}
                )
                for form in ("positional", "keyword", "two-bound", "three-bound"):
                    with self.subTest(
                        end=end, dtype=dtype_name, form=form
                    ):
                        if form == "positional":
                            self.assert_error_matches(
                                lambda end=end, options=actual_options: torch.arange(
                                    end, **options
                                ),
                                lambda end=end, options=expected_options: reference_torch.arange(
                                    end, **options
                                ),
                            )
                        elif form == "keyword":
                            self.assert_error_matches(
                                lambda end=end, options=actual_options: torch.arange(
                                    end=end, **options
                                ),
                                lambda end=end, options=expected_options: reference_torch.arange(
                                    end=end, **options
                                ),
                            )
                        elif form == "two-bound":
                            self.assert_error_matches(
                                lambda end=end, options=actual_options: torch.arange(
                                    np.float32(0.0), end, **options
                                ),
                                lambda end=end, options=expected_options: reference_torch.arange(
                                    np.float32(0.0), end, **options
                                ),
                            )
                        else:
                            self.assert_error_matches(
                                lambda end=end, options=actual_options: torch.arange(
                                    np.float32(0.0), end, np.float32(1.0), **options
                                ),
                                lambda end=end, options=expected_options: reference_torch.arange(
                                    np.float32(0.0), end, np.float32(1.0), **options
                                ),
                            )

    def test_explicit_float32_integer_boundary_errors_match_pytorch_2_13(self):
        endpoints = (
            -(2**63),
            -(2**63) - 1,
            2**63 - 1,
            2**63,
            2**64 - 1,
            2**64,
        )
        for end in endpoints:
            for dtype_name in ("float32", "float"):
                actual_dtype = getattr(torch, dtype_name)
                expected_dtype = getattr(reference_torch, dtype_name)
                for form in ("positional", "keyword", "two-bound", "three-bound"):
                    with self.subTest(
                        end=end, dtype=dtype_name, form=form
                    ):
                        if form == "positional":
                            self.assert_error_matches(
                                lambda end=end, dtype=actual_dtype: torch.arange(
                                    end, dtype=dtype
                                ),
                                lambda end=end, dtype=expected_dtype: reference_torch.arange(
                                    end, dtype=dtype
                                ),
                            )
                        elif form == "keyword":
                            self.assert_error_matches(
                                lambda end=end, dtype=actual_dtype: torch.arange(
                                    end=end, dtype=dtype
                                ),
                                lambda end=end, dtype=expected_dtype: reference_torch.arange(
                                    end=end, dtype=dtype
                                ),
                            )
                        elif form == "two-bound":
                            self.assert_error_matches(
                                lambda end=end, dtype=actual_dtype: torch.arange(
                                    0, end, dtype=dtype
                                ),
                                lambda end=end, dtype=expected_dtype: reference_torch.arange(
                                    0, end, dtype=dtype
                                ),
                            )
                        else:
                            self.assert_error_matches(
                                lambda end=end, dtype=actual_dtype: torch.arange(
                                    0, end, 1, dtype=dtype
                                ),
                                lambda end=end, dtype=expected_dtype: reference_torch.arange(
                                    0, end, 1, dtype=dtype
                                ),
                            )

    def mode_dispatch_observation(self, module):
        function = module.arange
        marker = object()
        intercepted = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func is function,
                        types,
                        args,
                        kwargs,
                        len(module.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        for call in (
            lambda: function(2.5),
            lambda: function(end=2.5),
            lambda: function(3),
            lambda: function(0.0, 3.0),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
                restored_inside = (
                    module.overrides._get_current_function_mode_stack() == [mode]
                )
            intercepted.append(
                (
                    result is marker,
                    mode.calls,
                    restored_inside,
                    module.overrides._get_current_function_mode_stack() == [],
                )
            )

        forwarding_events = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_events.append(
                    (
                        self.label,
                        tuple(
                            mode.label
                            for mode in module.overrides._get_current_function_mode_stack()
                        ),
                        func is function,
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
                forwarded = function(end=2.5)
                nested_restored = (
                    module.overrides._get_current_function_mode_stack()
                    == [lower, upper]
                )
            lower_restored = (
                module.overrides._get_current_function_mode_stack() == [lower]
            )
        stack_empty = module.overrides._get_current_function_mode_stack() == []

        expected_error = ValueError("handler failed")

        class RaisingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise expected_error

        raising = RaisingMode()
        with lower:
            with raising:
                try:
                    function(2.5)
                except Exception as error:
                    handler_error = (
                        type(error).__name__,
                        str(error),
                        error.args,
                        error is expected_error,
                    )
                else:
                    handler_error = None
                handler_error_restored = (
                    module.overrides._get_current_function_mode_stack()
                    == [lower, raising]
                )
            handler_lower_restored = (
                module.overrides._get_current_function_mode_stack() == [lower]
            )

        native_error_mode = ForwardingMode("native-error")
        with native_error_mode:
            try:
                function(-1.0)
            except Exception as error:
                native_error = (type(error).__name__, str(error), error.args)
            else:
                native_error = None
            native_error_restored = (
                module.overrides._get_current_function_mode_stack()
                == [native_error_mode]
            )

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        declining = DecliningMode()
        with declining:
            try:
                function(2.5)
            except Exception as error:
                declining_error = (
                    type(error).__name__,
                    re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
                    error.args[1:] if len(error.args) > 1 else (),
                )
            else:
                declining_error = None
            declining_restored = (
                module.overrides._get_current_function_mode_stack() == [declining]
            )

        return (
            intercepted,
            forwarding_events,
            self.tensor_contract(module, forwarded),
            nested_restored,
            lower_restored,
            stack_empty,
            handler_error,
            handler_error_restored,
            handler_lower_restored,
            native_error,
            native_error_restored,
            declining_error,
            declining_restored,
            module.overrides._get_current_function_mode_stack() == [],
        )

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation(torch),
            self.mode_dispatch_observation(reference_torch),
        )

    def callable_contract(self, module):
        function = module.arange
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.arange is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("arange"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "wildcard_identity": wildcard_namespace["arange"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
