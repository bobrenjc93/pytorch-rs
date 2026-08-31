import inspect
import math
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = """
mean(dim=None, keepdim=False, dtype=None) -> Tensor

See :func:`torch.mean`
"""

EXPECTED_METHOD_OVERLOADS = (
    "but expected one of:\n"
    " * (*, torch.dtype dtype = None)\n"
    " * (tuple of ints dim, bool keepdim = False, *, "
    "torch.dtype dtype = None)\n"
)

EXPECTED_TOP_LEVEL_OVERLOADS = (
    "but expected one of:\n"
    " * (Tensor input, *, torch.dtype dtype = None, Tensor out = None)\n"
    " * (Tensor input, tuple of ints dim, bool keepdim = False, *, "
    "torch.dtype dtype = None, Tensor out = None)\n"
)


class TensorMeanTests(unittest.TestCase):
    def assert_scalar(self, value, expected, source, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(value.shape, ())
            self.assertEqual(value.stride(), ())
            self.assertEqual(value.storage_offset(), 0)
            self.assertEqual(value.numel(), 1)
            self.assertTrue(value.is_contiguous())
            self.assertIs(value.dtype, torch.float32)
            self.assertEqual(value.device, torch.device("cpu"))
            self.assertFalse(value.is_set_to(source))
            if source.numel():
                self.assertNotEqual(value.data_ptr(), source.data_ptr())
        with self.subTest(case=case, value=True):
            if np.isnan(expected):
                self.assertTrue(math.isnan(value.item()))
            else:
                self.assertEqual(
                    np.float32(value.item()).view(np.uint32).item(),
                    np.float32(expected).view(np.uint32).item(),
                )

    @staticmethod
    def sequential_float32_mean(values):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return np.float32(np.nan)
        total = np.float32(0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            for value in values:
                total = np.float32(total + value)
            return np.float32(total / np.float32(values.size))

    @staticmethod
    def value_cases():
        dense_values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        dense = torch.tensor(dense_values.tolist())
        noncontiguous = dense.transpose(0, 2)
        return (
            ("division rounding", torch.tensor([1.0, 2.0, 4.0]), [1.0, 2.0, 4.0]),
            ("scalar", torch.tensor(-3.5), [-3.5]),
            ("negative zero", torch.tensor(-0.0), [-0.0]),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1], []),
            ("singleton", torch.tensor([[[7.0]]])[0], [7.0]),
            ("contiguous offset", dense[1], dense_values[1]),
            ("offset", noncontiguous[1], dense_values.transpose(2, 1, 0)[1]),
            ("noncontiguous", noncontiguous, dense_values.transpose(2, 1, 0)),
            ("nan", torch.tensor([1.0, float("nan"), 2.0]), [1.0, np.nan, 2.0]),
            ("positive infinity", torch.tensor([1.0, float("inf"), 2.0]), [1.0, np.inf, 2.0]),
            ("negative infinity", torch.tensor([1.0, float("-inf"), 2.0]), [1.0, -np.inf, 2.0]),
            ("mixed infinities", torch.tensor([float("inf"), float("-inf")]), [np.inf, -np.inf]),
        )

    @staticmethod
    def supported_method_calls(source):
        return (
            ("default", lambda: source.mean()),
            ("positional none dim", lambda: source.mean(None)),
            ("keyword none dim", lambda: source.mean(dim=None)),
            ("none dim keepdim false", lambda: source.mean(None, False)),
            ("dtype none", lambda: source.mean(dtype=None)),
            ("dtype float32", lambda: source.mean(dtype=torch.float32)),
            ("dtype float alias", lambda: source.mean(dtype=torch.float)),
            (
                "none dim dtype float32",
                lambda: source.mean(dim=None, keepdim=False, dtype=torch.float32),
            ),
        )

    @staticmethod
    def supported_top_level_calls(source):
        return (
            ("positional", lambda: torch.mean(source)),
            ("positional none dim", lambda: torch.mean(source, None)),
            ("keyword none dim", lambda: torch.mean(source, dim=None)),
            ("none dim keepdim false", lambda: torch.mean(source, None, False)),
            ("none dim out none", lambda: torch.mean(source, dim=None, out=None)),
            ("out none", lambda: torch.mean(source, out=None)),
            ("input", lambda: torch.mean(input=source)),
            ("x", lambda: torch.mean(x=source)),
            ("a", lambda: torch.mean(a=source)),
            ("x1", lambda: torch.mean(x1=source)),
            ("dtype none", lambda: torch.mean(source, dtype=None)),
            ("dtype float32", lambda: torch.mean(source, dtype=torch.float32)),
            ("dtype float alias", lambda: torch.mean(source, dtype=torch.float)),
            (
                "all keyword defaults",
                lambda: torch.mean(
                    input=source, dim=None, keepdim=False, dtype=torch.float32, out=None
                ),
            ),
        )

    def test_supported_forms_match_sum_divided_by_count_metadata_and_storage(self):
        for name, source, expected_values in self.value_cases():
            expected = self.sequential_float32_mean(expected_values)
            for form, call in self.supported_method_calls(source):
                self.assert_scalar(call(), expected, source, case=(name, "method", form))
            for form, call in self.supported_top_level_calls(source):
                self.assert_scalar(call(), expected, source, case=(name, "top-level", form))

    def test_mean_preserves_first_order_autograd_and_no_grad(self):
        for form, make_loss in (
            (
                "method",
                lambda view: view.mean(dim=None, keepdim=False, dtype=torch.float32),
            ),
            (
                "top-level",
                lambda view: torch.mean(view, dim=None, keepdim=False, dtype=torch.float32),
            ),
        ):
            with self.subTest(form=form, case="noncontiguous autograd"):
                leaf = torch.tensor(
                    [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], requires_grad=True
                )
                loss = make_loss(leaf.transpose(0, 1))
                self.assertTrue(loss.requires_grad)
                self.assertFalse(loss.is_leaf)
                loss.backward()
                loss.backward()
                np.testing.assert_array_equal(
                    np.asarray(leaf.grad),
                    np.full((2, 3), np.float32(2.0 / 6.0), dtype=np.float32),
                )

            with self.subTest(form=form, case="empty autograd"):
                empty = torch.zeros((2, 0, 3), requires_grad=True)
                empty_loss = make_loss(empty.transpose(0, 2)[1])
                self.assertTrue(math.isnan(empty_loss.item()))
                empty_loss.backward()
                empty_loss.backward()
                self.assertEqual(empty.grad.shape, empty.shape)
                self.assertEqual(empty.grad.tolist(), [[], []])

        leaf = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        with torch.no_grad():
            untracked = torch.mean(leaf, dim=None, dtype=torch.float)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertIsNone(leaf.grad)
        self.assertTrue(leaf.mean(None, dtype=torch.float32).requires_grad)

        expected_bits = np.asarray(
            [np.float32(7.0) / np.float32(3.0)] * 3,
            dtype=np.float32,
        ).view(np.uint32)
        for form, make_mean in (
            ("method", lambda tensor: tensor.mean()),
            ("top-level", lambda tensor: torch.mean(tensor)),
        ):
            with self.subTest(form=form, case="division rounding autograd"):
                rounding_leaf = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
                (make_mean(rounding_leaf) * 7.0).backward()
                np.testing.assert_array_equal(
                    np.asarray(rounding_leaf.grad).view(np.uint32),
                    expected_bits,
                )

    def test_top_level_modes_and_overrides_observe_calls_before_native_limits(self):
        tensor = torch.tensor([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)
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
            self.assertIs(torch.mean(input=tensor, dtype=torch.float32), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.mean)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "dtype": torch.float32})

        dim_mode = RecordingMode()
        with dim_mode:
            self.assertIs(torch.mean(tensor, 0, keepdim=True), marker)
        self.assertEqual(len(dim_mode.calls), 1)
        function, dispatch_types, args, kwargs = dim_mode.calls[0]
        self.assertIs(function, torch.mean)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], tensor)
        self.assertEqual(args[1], 0)
        self.assertEqual(kwargs, {"keepdim": True})

        none_dim_mode = RecordingMode()
        with none_dim_mode:
            self.assertIs(torch.mean(tensor, None, keepdim=False, out=None), marker)
        self.assertEqual(len(none_dim_mode.calls), 1)
        function, dispatch_types, args, kwargs = none_dim_mode.calls[0]
        self.assertIs(function, torch.mean)
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

        self.assertIs(torch.mean(Override()), marker)
        self.assertIs(torch.mean(tensor, dtype=Override()), marker)
        self.assertIs(torch.mean(tensor, 0, out=Override()), marker)
        self.assertEqual(len(override_calls), 3)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.mean)
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

        self.assertIs(torch.mean(BaseOverride(), dtype=DerivedOverride()), marker)
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
                forwarded = torch.mean(input=tensor, dtype=torch.float32)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.item(), 1.5)
        forwarded.backward()
        self.assertEqual(tensor.grad.tolist(), [[0.25, 0.25], [0.25, 0.25]])

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
            self.assertIs(torch.mean(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

        invalid_mode = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid_mode:
                torch.mean(tensor, input=tensor)
        self.assertEqual(invalid_mode.calls, [])

    def test_callable_metadata_documentation_pickling_and_exports(self):
        tensor = torch.tensor([1.0, 2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "mean")
        bound = tensor.mean

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "mean")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        function = torch.mean
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "mean")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.mean")
        self.assertEqual(function.__module__, "torch")
        self.assertIn("mean(input, *, dtype=None) -> Tensor", function.__doc__)
        self.assertIsNone(function.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("mean"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["mean"], function)

    def test_method_rejects_invalid_arguments_and_unsupported_reductions(self):
        tensor = torch.ones((2, 3))
        invalid = "mean() received an invalid combination of arguments - got "
        type_error_cases = (
            (
                lambda: tensor.mean(dtype=1),
                f"{invalid}(dtype=int, ), {EXPECTED_METHOD_OVERLOADS}",
            ),
            (
                lambda: tensor.mean(dtype=object()),
                f"{invalid}(dtype=object, ), {EXPECTED_METHOD_OVERLOADS}",
            ),
            (
                lambda: tensor.mean(dim=None, dtype=1),
                "mean(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: tensor.mean(None, False, dtype=object()),
                "mean(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: tensor.mean(torch.float32),
                f"{invalid}(torch.dtype), {EXPECTED_METHOD_OVERLOADS}",
            ),
            (
                lambda: tensor.mean(extra=True),
                f"{invalid}(extra=bool, ), {EXPECTED_METHOD_OVERLOADS}",
            ),
            (
                lambda: tensor.mean(keepdim=False),
                f"{invalid}(keepdim=bool, ), {EXPECTED_METHOD_OVERLOADS}",
            ),
            (
                lambda: tensor.mean(keepdim=True),
                f"{invalid}(keepdim=bool, ), {EXPECTED_METHOD_OVERLOADS}",
            ),
            (
                lambda: tensor.mean(0, False, torch.float32),
                "mean() takes from 1 to 2 positional arguments but 3 were given",
            ),
            (
                lambda: tensor.mean(out=None),
                f"{invalid}(out=NoneType, ), {EXPECTED_METHOD_OVERLOADS}",
            ),
        )
        for call, message in type_error_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        unsupported_cases = (
            ("positional dim", lambda: tensor.mean(0)),
            ("keyword dim", lambda: tensor.mean(dim=0)),
            ("tuple dim", lambda: tensor.mean((0, 1))),
            ("list dim", lambda: tensor.mean(dim=[0, 1])),
            ("positional keepdim true", lambda: tensor.mean(None, True)),
        )
        for case, call in unsupported_cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^mean\(\): dim, keepdim, out, and dtype conversion reductions are not supported$",
                ):
                    call()

    def test_top_level_rejects_invalid_arguments_and_unsupported_reductions(self):
        tensor = torch.ones((2, 3))
        destination = torch.tensor([17.0, 19.0, 23.0])
        invalid = "mean() received an invalid combination of arguments - got "
        type_error_cases = (
            (lambda: torch.mean(), f"{invalid}(), {EXPECTED_TOP_LEVEL_OVERLOADS}"),
            (
                lambda: torch.mean(extra=True),
                'mean() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.mean(1),
                "mean(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.mean(1, dtype=torch.float32),
                f"{invalid}(int, dtype=torch.dtype), {EXPECTED_TOP_LEVEL_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, dtype=1),
                f"{invalid}(Tensor, dtype=int), {EXPECTED_TOP_LEVEL_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, keepdim=False),
                f"{invalid}(Tensor, keepdim=bool), {EXPECTED_TOP_LEVEL_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, keepdim=True),
                f"{invalid}(Tensor, keepdim=bool), {EXPECTED_TOP_LEVEL_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, None, dtype=1),
                "mean(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: torch.mean(tensor, torch.float32),
                f"{invalid}(Tensor, torch.dtype), {EXPECTED_TOP_LEVEL_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, 0, False, torch.float32),
                "mean() takes from 2 to 3 positional arguments but 4 were given",
            ),
            (
                lambda: torch.mean(tensor, input=tensor),
                f"{invalid}(Tensor, input=Tensor), {EXPECTED_TOP_LEVEL_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, out=[]),
                "mean(): argument 'out' must be Tensor, not list",
            ),
        )
        for call, message in type_error_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        unsupported_cases = (
            ("positional dim", lambda: torch.mean(tensor, 0)),
            ("keyword dim", lambda: torch.mean(input=tensor, dim=0)),
            ("tuple dim", lambda: torch.mean(tensor, (0, 1))),
            ("list dim", lambda: torch.mean(tensor, [0, 1])),
            ("keepdim", lambda: torch.mean(tensor, 0, keepdim=True)),
            ("none dim keepdim true", lambda: torch.mean(tensor, None, keepdim=True)),
            ("out", lambda: torch.mean(tensor, out=destination)),
            ("none dim concrete out", lambda: torch.mean(tensor, None, out=destination)),
            ("dtype plus dim", lambda: torch.mean(tensor, 0, dtype=torch.float32)),
        )
        for case, call in unsupported_cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^mean\(\): dim, keepdim, out, and dtype conversion reductions are not supported$",
                ):
                    call()
        self.assertEqual(destination.tolist(), [17.0, 19.0, 23.0])


if __name__ == "__main__":
    unittest.main()
