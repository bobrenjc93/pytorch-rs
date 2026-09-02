import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


EXPECTED_OVERLOADS = (
    "but expected one of:\n"
    " * (Tensor input, *, torch.dtype dtype = None)\n"
    " * (Tensor input, tuple of ints dim, bool keepdim = False, *, "
    "torch.dtype dtype = None, Tensor out = None)\n"
)


class TopLevelSumTests(unittest.TestCase):
    def assert_scalar_matches(self, actual, expected, source, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), 0)
            self.assertEqual(actual.numel(), 1)
            self.assertTrue(actual.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertFalse(actual.is_set_to(source))
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())
        with self.subTest(case=case, value=True):
            self.assertEqual(
                np.float32(actual.item()).view(np.uint32).item(),
                np.float32(expected.item()).view(np.uint32).item(),
            )

    def assert_keepdim_matches(self, actual, expected, source, *, case):
        expected_shape = (1,) * len(source.shape)
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected_shape)
            self.assertEqual(actual.stride(), (1,) * len(expected_shape))
            self.assertEqual(actual.storage_offset(), 0)
            self.assertEqual(actual.numel(), 1)
            self.assertTrue(actual.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertFalse(actual.is_set_to(source))
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())
        with self.subTest(case=case, value=True):
            self.assertEqual(
                np.float32(actual.item()).view(np.uint32).item(),
                np.float32(expected.item()).view(np.uint32).item(),
            )

    @staticmethod
    def value_cases():
        dense = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        noncontiguous = dense.transpose(0, 2)
        return (
            ("scalar", torch.tensor(-3.5)),
            ("negative zero", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("singleton", torch.tensor([[[7.0]]])[0]),
            ("contiguous offset", dense[1]),
            ("offset", noncontiguous[1]),
            ("noncontiguous", noncontiguous),
        )

    @staticmethod
    def supported_calls(source):
        return (
            ("positional", lambda: torch.sum(source)),
            ("positional none dim", lambda: torch.sum(source, None)),
            ("keyword none dim", lambda: torch.sum(source, dim=None)),
            ("none dim keepdim false", lambda: torch.sum(source, None, False)),
            ("none dim out none", lambda: torch.sum(source, dim=None, out=None)),
            ("input", lambda: torch.sum(input=source)),
            ("x", lambda: torch.sum(x=source)),
            ("a", lambda: torch.sum(a=source)),
            ("x1", lambda: torch.sum(x1=source)),
            ("dtype none", lambda: torch.sum(source, dtype=None)),
            ("dtype float32", lambda: torch.sum(source, dtype=torch.float32)),
            ("dtype float alias", lambda: torch.sum(source, dtype=torch.float)),
            ("alias and dtype", lambda: torch.sum(x=source, dtype=torch.float32)),
            (
                "none dim dtype out none",
                lambda: torch.sum(
                    input=source, dim=None, keepdim=False, dtype=torch.float32, out=None
                ),
            ),
        )

    @staticmethod
    def supported_keepdim_calls(source):
        return (
            ("positional none dim keepdim true", lambda: torch.sum(source, None, True)),
            (
                "mixed none dim keepdim true",
                lambda: torch.sum(source, None, keepdim=True),
            ),
            (
                "keyword none dim keepdim true",
                lambda: torch.sum(source, dim=None, keepdim=True),
            ),
            (
                "input keyword keepdim true",
                lambda: torch.sum(input=source, dim=None, keepdim=True),
            ),
            (
                "keepdim true dtype none",
                lambda: torch.sum(source, dim=None, keepdim=True, dtype=None),
            ),
            (
                "keepdim true dtype float32",
                lambda: torch.sum(
                    input=source,
                    dim=None,
                    keepdim=True,
                    dtype=torch.float32,
                    out=None,
                ),
            ),
        )

    @staticmethod
    def rank_one_strided_vector(values, *, requires_grad=False):
        rows = len(values)
        columns = 5
        selected_column = 2
        matrix = np.full((rows, columns), np.float32(0.5), dtype=np.float32)
        matrix[:, selected_column] = np.asarray(values, dtype=np.float32)
        source = torch.tensor(
            matrix.tolist(), dtype=torch.float32, requires_grad=requires_grad
        )
        return source, source.transpose(0, 1)[selected_column], matrix[:, selected_column]

    @staticmethod
    def sequential_float32_sum(values):
        total = np.float32(0.0)
        for value in np.asarray(values, dtype=np.float32):
            total = np.float32(total + value)
        return total

    @staticmethod
    def autograd_case(case):
        if case == "scalar":
            leaf = torch.tensor(-3.0, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return leaf, leaf.transpose(0, 2)[1]

        leaf = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        if case == "offset":
            return leaf, leaf[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 2)
        raise AssertionError(f"unknown sum autograd case: {case}")

    def test_supported_forms_delegate_to_tensor_sum_values_metadata_and_storage(self):
        for case, source in self.value_cases():
            expected = source.sum()
            for form, call in self.supported_calls(source):
                self.assert_scalar_matches(call(), expected, source, case=(case, form))

    def test_keepdim_full_reduction_preserves_rank_values_and_metadata(self):
        for case, source in self.value_cases():
            expected = source.sum()
            for form, call in self.supported_keepdim_calls(source):
                self.assert_keepdim_matches(call(), expected, source, case=(case, form))

    def test_supported_forms_preserve_autograd_accumulation_and_no_grad(self):
        forms = tuple(form for form, _ in self.supported_calls(torch.tensor(1.0)))
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form in forms:
                function_leaf, function_input = self.autograd_case(case)
                method_leaf, method_input = self.autograd_case(case)
                output = dict(self.supported_calls(function_input))[form]()
                expected = method_input.sum()
                self.assert_scalar_matches(
                    output, expected, function_input, case=(case, form, "forward")
                )

                output.backward()
                output.backward()
                expected.backward()
                expected.backward()
                self.assertEqual(function_leaf.grad.shape, method_leaf.grad.shape)
                np.testing.assert_array_equal(
                    np.asarray(function_leaf.grad), np.asarray(method_leaf.grad)
                )

        leaf = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        with torch.no_grad():
            untracked = torch.sum(leaf, dim=None, dtype=torch.float)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertIsNone(leaf.grad)
        self.assertTrue(torch.sum(leaf, None, dtype=torch.float32).requires_grad)

    def test_keepdim_full_reduction_preserves_no_grad_and_final_scalar_backward(self):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            function_leaf, function_input = self.autograd_case(case)
            baseline_leaf, baseline_input = self.autograd_case(case)
            kept = torch.sum(function_input, dim=None, keepdim=True)
            expected = baseline_input.sum()
            self.assert_keepdim_matches(
                kept, expected, function_input, case=(case, "forward")
            )

            kept.sum().backward()
            expected.backward()
            self.assertEqual(function_leaf.grad.shape, baseline_leaf.grad.shape)
            np.testing.assert_array_equal(
                np.asarray(function_leaf.grad), np.asarray(baseline_leaf.grad)
            )

        leaf = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        with torch.no_grad():
            untracked = torch.sum(leaf, dim=None, keepdim=True, dtype=torch.float)
        self.assert_keepdim_matches(
            untracked, leaf.detach().sum(), leaf, case="no_grad"
        )
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertIsNone(leaf.grad)

    def test_rank_one_transpose_selected_offset_sum_edges(self):
        cases = (
            ("signed zero", [-0.0, 0.0, -0.0, 0.0]),
            ("nan", [1.0, np.nan, 2.0, -3.0]),
            ("positive infinity", [1.0, np.inf, 2.0, 3.0]),
            ("negative infinity", [1.0, -np.inf, 2.0, 3.0]),
            ("sequential cancellation", [1.0e20, -1.0e20, 3.0, -0.0]),
        )

        for case, values in cases:
            _, view, selected = self.rank_one_strided_vector(values)
            expected = view.sum()
            self.assertEqual(view.shape, (len(values),))
            self.assertEqual(view.stride(), (5,))
            self.assertEqual(view.storage_offset(), 2)
            self.assertFalse(view.is_contiguous())
            self.assert_scalar_matches(
                torch.sum(view),
                expected,
                view,
                case=("rank-one transpose-selected offset", case),
            )
            self.assertEqual(
                np.float32(expected.item()).view(np.uint32).item(),
                self.sequential_float32_sum(selected).view(np.uint32).item(),
            )

    def test_rank_one_transpose_selected_offset_sum_empty_no_grad_and_repeated_backward(
        self,
    ):
        empty = torch.zeros((0, 5), requires_grad=True)
        empty_view = empty.transpose(0, 1)[2]
        self.assertEqual(empty_view.shape, (0,))
        self.assertEqual(empty_view.stride(), (5,))
        self.assertEqual(empty_view.storage_offset(), 2)
        self.assert_scalar_matches(
            torch.sum(empty_view), empty_view.sum(), empty_view, case="rank-one empty"
        )
        torch.sum(empty_view).backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.tolist(), [])

        leaf, view, selected = self.rank_one_strided_vector(
            np.arange(1, 21, dtype=np.float32).reshape(4, 5)[:, 2],
            requires_grad=True,
        )
        loss = torch.sum(view)
        self.assertTrue(loss.requires_grad)
        self.assertFalse(loss.is_leaf)
        loss.backward()
        loss.backward()
        expected_gradient = np.zeros((4, 5), dtype=np.float32)
        expected_gradient[:, 2] = 2.0
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected_gradient)

        with torch.no_grad():
            untracked = torch.sum(view)
        self.assert_scalar_matches(
            untracked,
            torch.tensor(
                float(self.sequential_float32_sum(selected)), dtype=torch.float32
            ),
            view,
            case="rank-one no_grad",
        )
        self.assertEqual(
            np.float32(untracked.item()).view(np.uint32).item(),
            self.sequential_float32_sum(selected).view(np.uint32).item(),
        )
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)

    def test_modes_and_overrides_observe_calls_before_native_limits(self):
        tensor = torch.tensor([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.sum(input=tensor, dtype=torch.float32), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.sum)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "dtype": torch.float32})

        dim_mode = RecordingMode()
        with dim_mode:
            self.assertIs(torch.sum(tensor, 0, keepdim=True), marker)
        self.assertEqual(len(dim_mode.calls), 1)
        function, dispatch_types, args, kwargs = dim_mode.calls[0]
        self.assertIs(function, torch.sum)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], tensor)
        self.assertEqual(args[1], 0)
        self.assertEqual(kwargs, {"keepdim": True})

        none_dim_mode = RecordingMode()
        with none_dim_mode:
            self.assertIs(torch.sum(tensor, None, keepdim=False, out=None), marker)
        self.assertEqual(len(none_dim_mode.calls), 1)
        function, dispatch_types, args, kwargs = none_dim_mode.calls[0]
        self.assertIs(function, torch.sum)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], tensor)
        self.assertIsNone(args[1])
        self.assertEqual(kwargs, {"keepdim": False, "out": None})

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.sum(Override()), marker)
        self.assertIs(torch.sum(tensor, dtype=Override()), marker)
        self.assertIs(torch.sum(tensor, 0, out=Override()), marker)
        self.assertEqual(len(override_calls), 3)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.sum)
            self.assertEqual(dispatch_types, (Override,))

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("base")
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("derived")
                return marker

        self.assertIs(torch.sum(BaseOverride(), dtype=DerivedOverride()), marker)
        self.assertEqual(subclass_order, ["derived"])

        forwarding_order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.sum(input=tensor, dtype=torch.float32)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.item(), 6.0)
        forwarded.backward()
        self.assertEqual(tensor.grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])

        events = []

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append("mode")
                return NotImplemented

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append("override")
                return marker

        with DecliningMode():
            self.assertIs(torch.sum(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

        invalid_mode = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid_mode:
                torch.sum(tensor, out=None)
        self.assertEqual(invalid_mode.calls, [])

    def test_callable_metadata_documentation_pickling_and_exports(self):
        function = torch.sum
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "sum")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.sum")
        self.assertEqual(function.__module__, "torch")
        self.assertIn("sum(input, *, dtype=None) -> Tensor", function.__doc__)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method sum of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.sum, function)
        for action in (
            lambda: setattr(owner, "sum", None),
            lambda: delattr(owner, "sum"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.sum, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("sum"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["sum"], function)

    def test_binding_errors_match_the_pytorch_overload(self):
        tensor = torch.ones((2, 3))
        destination = torch.tensor([17.0, 19.0, 23.0])
        invalid = "sum() received an invalid combination of arguments - got "
        cases = (
            (lambda: torch.sum(), f"{invalid}(), {EXPECTED_OVERLOADS}"),
            (
                lambda: torch.sum(extra=True),
                'sum() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sum(out=None),
                'sum() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sum(1),
                "sum(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.sum(np.zeros((2, 3), dtype=np.float32)),
                "sum(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
            (
                lambda: torch.sum(1, dtype=torch.float32),
                f"{invalid}(int, dtype=torch.dtype), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, dtype=1),
                f"{invalid}(Tensor, dtype=int), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, dtype=object()),
                f"{invalid}(Tensor, dtype=object), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, None, dtype=1),
                "sum(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: torch.sum(tensor, torch.float32),
                f"{invalid}(Tensor, torch.dtype), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, 0, False, torch.float32),
                "sum() takes from 2 to 3 positional arguments but 4 were given",
            ),
            (
                lambda: torch.sum(tensor, input=tensor),
                f"{invalid}(Tensor, input=Tensor), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, keepdim=True),
                f"{invalid}(Tensor, keepdim=bool), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, out=None),
                f"{invalid}(Tensor, out=NoneType), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, out=destination),
                f"{invalid}(Tensor, out=Tensor), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, tensor),
                f"{invalid}(Tensor, Tensor), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, 0, dtype=1),
                "sum(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: torch.sum(tensor, 0, out=[]),
                "sum(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.sum(tensor, None, out=[]),
                "sum(): argument 'out' must be Tensor, not list",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()
        self.assertEqual(destination.tolist(), [17.0, 19.0, 23.0])

    def test_dim_keepdim_out_and_cross_dtype_forms_remain_unsupported(self):
        tensor = torch.ones((2, 3))
        destination = torch.tensor([17.0, 19.0, 23.0])
        cases = (
            ("positional dim", lambda: torch.sum(tensor, 0)),
            ("keyword dim", lambda: torch.sum(input=tensor, dim=0)),
            ("tuple dim", lambda: torch.sum(tensor, (0, 1))),
            ("list dim", lambda: torch.sum(tensor, [0, 1])),
            ("keepdim", lambda: torch.sum(tensor, 0, keepdim=True)),
            (
                "none dim keepdim true concrete out",
                lambda: torch.sum(tensor, None, keepdim=True, out=destination),
            ),
            ("out", lambda: torch.sum(tensor, 0, out=destination)),
            ("none dim concrete out", lambda: torch.sum(tensor, None, out=destination)),
            ("dtype plus dim", lambda: torch.sum(tensor, 0, dtype=torch.float32)),
        )
        for case, call in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^sum\(\): only full reductions with dim=None support keepdim; dim and out reductions are not supported$",
                ):
                    call()
        self.assertEqual(destination.tolist(), [17.0, 19.0, 23.0])


if __name__ == "__main__":
    unittest.main()
