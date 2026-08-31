import inspect
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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelSumReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError("torch.sum differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("torch.sum unexpectedly accepted an invalid call")

    def assert_scalar_matches(self, actual, expected, actual_source, expected_source, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.numel(), expected.numel())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertIs(expected.dtype, reference_torch.float32)
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertFalse(actual.is_set_to(actual_source))
            self.assertFalse(expected.is_set_to(expected_source))
        with self.subTest(case=case, value=True):
            self.assertEqual(
                np.asarray(actual).view(np.uint32).item(),
                expected.detach().cpu().numpy().view(np.uint32).item(),
            )

    @staticmethod
    def make_cases(module):
        dense = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        noncontiguous = dense.transpose(0, 2)
        return (
            ("scalar", module.tensor(-3.5, dtype=module.float32)),
            ("negative zero", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("contiguous offset", dense[1]),
            ("offset", noncontiguous[1]),
            ("noncontiguous", noncontiguous),
        )

    @staticmethod
    def call_sum(module, source, form):
        if form == "positional":
            return module.sum(source)
        if form == "positional none dim":
            return module.sum(source, None)
        if form == "keyword none dim":
            return module.sum(source, dim=None)
        if form == "none dim keepdim false":
            return module.sum(source, None, False)
        if form == "none dim out none":
            return module.sum(source, dim=None, out=None)
        if form == "dtype none":
            return module.sum(source, dtype=None)
        if form == "dtype float32":
            return module.sum(source, dtype=module.float32)
        if form == "dtype float alias":
            return module.sum(source, dtype=module.float)
        if form == "alias and dtype":
            return module.sum(x=source, dtype=module.float32)
        if form == "none dim dtype out none":
            return module.sum(
                input=source, dim=None, keepdim=False, dtype=module.float32, out=None
            )
        return module.sum(**{form: source})

    @staticmethod
    def rank_one_strided_vector(module, values, *, requires_grad=False):
        rows = len(values)
        columns = 5
        selected_column = 2
        matrix = np.full((rows, columns), np.float32(0.5), dtype=np.float32)
        matrix[:, selected_column] = np.asarray(values, dtype=np.float32)
        source = module.tensor(
            matrix.tolist(), dtype=module.float32, requires_grad=requires_grad
        )
        return source, source.transpose(0, 1)[selected_column]

    @staticmethod
    def autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(-3.0, dtype=module.float32, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf.transpose(0, 2)[1]

        leaf = module.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        if case == "offset":
            return leaf, leaf[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 2)
        raise AssertionError(f"unknown torch.sum autograd case: {case}")

    def test_supported_values_metadata_and_storage_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        forms = (
            "positional",
            "positional none dim",
            "keyword none dim",
            "none dim keepdim false",
            "none dim out none",
            "input",
            "x",
            "a",
            "x1",
            "dtype none",
            "dtype float32",
            "dtype float alias",
            "alias and dtype",
            "none dim dtype out none",
        )
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(case, expected_name)
            for form in forms:
                self.assert_scalar_matches(
                    self.call_sum(torch, actual_input, form),
                    self.call_sum(reference_torch, expected_input, form),
                    actual_input,
                    expected_input,
                    case=(case, form),
                )

    def test_autograd_accumulation_empty_and_no_grad_match_pytorch_2_13(self):
        forms = (
            "positional",
            "positional none dim",
            "keyword none dim",
            "none dim keepdim false",
            "none dim out none",
            "dtype none",
            "dtype float32",
            "dtype float alias",
            "alias and dtype",
            "none dim dtype out none",
        )
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form in forms:
                actual_leaf, actual_input = self.autograd_case(torch, case)
                expected_leaf, expected_input = self.autograd_case(
                    reference_torch, case
                )
                actual_loss = self.call_sum(torch, actual_input, form)
                expected_loss = self.call_sum(reference_torch, expected_input, form)
                self.assert_scalar_matches(
                    actual_loss,
                    expected_loss,
                    actual_input,
                    expected_input,
                    case=(case, form),
                )

                actual_loss.backward()
                actual_loss.backward()
                expected_loss.backward()
                expected_loss.backward()
                np.testing.assert_array_equal(
                    np.asarray(actual_leaf.grad),
                    expected_leaf.grad.detach().cpu().numpy(),
                )

        actual_leaf = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        expected_leaf = reference_torch.tensor(
            [1.0, -2.0, 3.0],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        with torch.no_grad():
            actual = torch.sum(input=actual_leaf, dim=None, dtype=torch.float)
        with reference_torch.no_grad():
            expected = reference_torch.sum(
                input=expected_leaf, dim=None, dtype=reference_torch.float
            )
        self.assert_scalar_matches(
            actual, expected, actual_leaf, expected_leaf, case="no_grad"
        )
        self.assertIsNone(actual_leaf.grad)

    def test_rank_one_transpose_selected_offset_sum_edges_match_pytorch_2_13(self):
        cases = (
            ("signed zero", [-0.0, 0.0, -0.0, 0.0]),
            ("nan", [1.0, np.nan, 2.0, -3.0]),
            ("positive infinity", [1.0, np.inf, 2.0, 3.0]),
            ("negative infinity", [1.0, -np.inf, 2.0, 3.0]),
            ("sequential cancellation", [1.0e20, -1.0e20, 3.0, -0.0]),
        )

        for case, values in cases:
            _, actual = self.rank_one_strided_vector(torch, values)
            _, expected = self.rank_one_strided_vector(reference_torch, values)
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertFalse(actual.is_contiguous())
            self.assertFalse(expected.is_contiguous())
            self.assert_scalar_matches(
                torch.sum(actual),
                reference_torch.sum(expected),
                actual,
                expected,
                case=("rank-one offset", case),
            )

    def test_rank_one_transpose_selected_offset_sum_autograd_match_pytorch_2_13(
        self,
    ):
        actual_empty = torch.zeros((0, 5), dtype=torch.float32, requires_grad=True)
        expected_empty = reference_torch.zeros(
            (0, 5), dtype=reference_torch.float32, requires_grad=True
        )
        actual_empty_view = actual_empty.transpose(0, 1)[2]
        expected_empty_view = expected_empty.transpose(0, 1)[2]
        self.assertEqual(actual_empty_view.shape, tuple(expected_empty_view.shape))
        self.assertEqual(actual_empty_view.stride(), expected_empty_view.stride())
        self.assertEqual(
            actual_empty_view.storage_offset(), expected_empty_view.storage_offset()
        )
        self.assert_scalar_matches(
            torch.sum(actual_empty_view),
            reference_torch.sum(expected_empty_view),
            actual_empty_view,
            expected_empty_view,
            case="rank-one empty",
        )
        torch.sum(actual_empty_view).backward()
        reference_torch.sum(expected_empty_view).backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), expected_empty.grad.detach().cpu().numpy()
        )

        values = np.arange(1, 21, dtype=np.float32).reshape(4, 5)[:, 2]
        actual_leaf, actual_view = self.rank_one_strided_vector(
            torch, values, requires_grad=True
        )
        expected_leaf, expected_view = self.rank_one_strided_vector(
            reference_torch, values, requires_grad=True
        )
        actual_loss = torch.sum(actual_view)
        expected_loss = reference_torch.sum(expected_view)
        self.assert_scalar_matches(
            actual_loss, expected_loss, actual_view, expected_view, case="rank-one tracked"
        )
        for _ in range(2):
            actual_loss.backward()
            expected_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.detach().cpu().numpy()
        )

        with torch.no_grad():
            actual_untracked = torch.sum(actual_view)
        with reference_torch.no_grad():
            expected_untracked = reference_torch.sum(expected_view)
        self.assert_scalar_matches(
            actual_untracked,
            expected_untracked,
            actual_view,
            expected_view,
            case="rank-one no_grad",
        )

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        function = module.sum
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
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
            "owner_callable_identity": owner.sum is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature": self.signature_outcome(function),
            "all_count": module.__all__.count("sum"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["sum"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    @staticmethod
    def dispatch_observation(module):
        tensor = module.tensor(
            [[1.0, -2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        destination = module.tensor([0.0, 0.0], dtype=module.float32)
        function = module.sum
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_observations = []
        mode_calls = (
            ("positional", lambda: function(tensor), None),
            ("input", lambda: function(input=tensor), ("input",)),
            ("x", lambda: function(x=tensor), ("x",)),
            ("dtype", lambda: function(tensor, dtype=module.float32), ("dtype",)),
            (
                "dim none out none",
                lambda: function(tensor, None, keepdim=False, out=None),
                ("keepdim", "out"),
            ),
            ("dim positional", lambda: function(tensor, 0), None),
            (
                "dim keyword",
                lambda: function(input=tensor, dim=0, keepdim=True),
                ("input", "dim", "keepdim"),
            ),
        )
        for label, call, expected_keywords in mode_calls:
            mode = RecordingMode()
            with mode:
                result = call()
            dispatched_function, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    label,
                    result is marker,
                    dispatched_function is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    expected_keywords,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for label, call, expected_keyword in (
            ("input", lambda value: function(value), None),
            ("dtype", lambda value: function(tensor, dtype=value), "dtype"),
            ("out", lambda value: function(tensor, 0, out=value), "out"),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            dispatched_function, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    label,
                    result is marker,
                    dispatched_function is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    expected_keyword is not None
                    and kwargs is not None
                    and kwargs[expected_keyword] is value,
                )
            )

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("base", tuple(item.__name__ for item in types))
                )
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("derived", tuple(item.__name__ for item in types))
                )
                return marker

        subclass_result = function(BaseOverride(), dtype=DerivedOverride())

        forward_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forward_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=tensor, dtype=module.float32)
        forwarded.backward()

        fallback_events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_events.append("override")
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback_result = function(FallbackOverride())

        invalid_observations = []
        for call in (
            lambda: function(),
            lambda: function(tensor, out=None),
            lambda: function(tensor, extra=True),
            lambda: function(1, dtype=module.float32),
        ):
            mode = RecordingMode()
            try:
                with mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(mode.calls))
                )

        return (
            mode_observations,
            override_observations,
            subclass_result is marker,
            subclass_order,
            forward_order,
            forwarded.item(),
            tensor.grad.tolist(),
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_modes_and_subclass_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_binding_errors_match_pytorch_2_13(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones((2, 3), dtype=reference_torch.float32)
        actual_destination = torch.tensor([17.0, 19.0, 23.0])
        expected_destination = reference_torch.tensor(
            [17.0, 19.0, 23.0],
            dtype=reference_torch.float32,
        )
        cases = (
            (lambda: torch.sum(), lambda: reference_torch.sum()),
            (lambda: torch.sum(extra=True), lambda: reference_torch.sum(extra=True)),
            (lambda: torch.sum(out=None), lambda: reference_torch.sum(out=None)),
            (lambda: torch.sum(1), lambda: reference_torch.sum(1)),
            (
                lambda: torch.sum(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.sum(np.zeros((2, 3), dtype=np.float32)),
            ),
            (
                lambda: torch.sum(1, dtype=torch.float32),
                lambda: reference_torch.sum(1, dtype=reference_torch.float32),
            ),
            (
                lambda: torch.sum(actual, dtype=1),
                lambda: reference_torch.sum(expected, dtype=1),
            ),
            (
                lambda: torch.sum(actual, dtype=object()),
                lambda: reference_torch.sum(expected, dtype=object()),
            ),
            (
                lambda: torch.sum(actual, None, dtype=1),
                lambda: reference_torch.sum(expected, None, dtype=1),
            ),
            (
                lambda: torch.sum(actual, reference_torch.float32),
                lambda: reference_torch.sum(expected, reference_torch.float32),
            ),
            (
                lambda: torch.sum(actual, 0, False, torch.float32),
                lambda: reference_torch.sum(
                    expected, 0, False, reference_torch.float32
                ),
            ),
            (
                lambda: torch.sum(actual, input=actual),
                lambda: reference_torch.sum(expected, input=expected),
            ),
            (
                lambda: torch.sum(actual, keepdim=True),
                lambda: reference_torch.sum(expected, keepdim=True),
            ),
            (
                lambda: torch.sum(actual, out=None),
                lambda: reference_torch.sum(expected, out=None),
            ),
            (
                lambda: torch.sum(actual, out=actual_destination),
                lambda: reference_torch.sum(expected, out=expected_destination),
            ),
            (
                lambda: torch.sum(actual, expected),
                lambda: reference_torch.sum(expected, expected),
            ),
            (
                lambda: torch.sum(actual, 0, dtype=1),
                lambda: reference_torch.sum(expected, 0, dtype=1),
            ),
            (
                lambda: torch.sum(actual, 0, out=[]),
                lambda: reference_torch.sum(expected, 0, out=[]),
            ),
            (
                lambda: torch.sum(actual, None, out=[]),
                lambda: reference_torch.sum(expected, None, out=[]),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
        self.assertEqual(actual_destination.tolist(), [17.0, 19.0, 23.0])

    def test_dimension_out_and_cross_dtype_boundaries_remain_unsupported(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones((2, 3), dtype=reference_torch.float32)

        expected_dim_results = (
            reference_torch.sum(expected, 0),
            reference_torch.sum(expected, dim=0),
            reference_torch.sum(expected, (0, 1)),
            reference_torch.sum(expected, [0, 1]),
            reference_torch.sum(expected, 0, keepdim=True),
            reference_torch.sum(expected, None, keepdim=True),
            reference_torch.sum(expected, 0, dtype=reference_torch.float32),
        )
        self.assertEqual(
            [tuple(result.shape) for result in expected_dim_results],
            [(3,), (3,), (), (), (1, 3), (1, 1), (3,)],
        )

        actual_dim_calls = (
            lambda: torch.sum(actual, 0),
            lambda: torch.sum(input=actual, dim=0),
            lambda: torch.sum(actual, (0, 1)),
            lambda: torch.sum(actual, [0, 1]),
            lambda: torch.sum(actual, 0, keepdim=True),
            lambda: torch.sum(actual, None, keepdim=True),
            lambda: torch.sum(actual, 0, dtype=torch.float32),
        )
        for case, call in enumerate(actual_dim_calls):
            with self.subTest(case=case):
                with self.assertRaises(NotImplementedError):
                    call()

        actual_destination = torch.tensor([17.0, 19.0, 23.0])
        expected_destination = reference_torch.tensor(
            [17.0, 19.0, 23.0],
            dtype=reference_torch.float32,
        )
        with self.assertRaises(NotImplementedError):
            torch.sum(actual, 0, out=actual_destination)
        expected_out = reference_torch.sum(expected, 0, out=expected_destination)
        self.assertIs(expected_out, expected_destination)
        self.assertEqual(actual_destination.tolist(), [17.0, 19.0, 23.0])
        self.assertEqual(expected_destination.tolist(), [2.0, 2.0, 2.0])

        actual_scalar_destination = torch.tensor(17.0)
        expected_scalar_destination = reference_torch.tensor(
            17.0, dtype=reference_torch.float32
        )
        with self.assertRaises(NotImplementedError):
            torch.sum(actual, None, out=actual_scalar_destination)
        expected_scalar_out = reference_torch.sum(
            expected, None, out=expected_scalar_destination
        )
        self.assertIs(expected_scalar_out, expected_scalar_destination)
        self.assertEqual(actual_scalar_destination.item(), 17.0)
        self.assertEqual(expected_scalar_destination.item(), 6.0)

        with self.assertRaises(TypeError):
            torch.sum(actual, dtype=reference_torch.float64)
        expected_float64 = reference_torch.sum(expected, dtype=reference_torch.float64)
        self.assertIs(expected_float64.dtype, reference_torch.float64)

        with self.assertRaises(TypeError):
            torch.sum(actual, None, dtype=reference_torch.float64)
        expected_none_dim_float64 = reference_torch.sum(
            expected, None, dtype=reference_torch.float64
        )
        self.assertIs(expected_none_dim_float64.dtype, reference_torch.float64)


if __name__ == "__main__":
    unittest.main()
