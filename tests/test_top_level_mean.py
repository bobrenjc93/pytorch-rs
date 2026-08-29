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


def scalar_bits(tensor):
    return np.asarray(tensor, dtype=np.float32).view(np.uint32).item()


class TopLevelMeanTests(unittest.TestCase):
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
            self.assertEqual(scalar_bits(actual), scalar_bits(expected))

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
            ("contiguous offset", dense[1]),
            ("offset", noncontiguous[1]),
            ("noncontiguous", noncontiguous),
            (
                "NaN",
                torch.tensor(
                    memoryview(
                        np.asarray([0x7FC1_2345, 0x3F80_0000], dtype=np.uint32).view(
                            np.float32
                        )
                    ),
                    dtype=torch.float32,
                ),
            ),
            ("infinity", torch.tensor([float("inf"), 1.0])),
        )

    @staticmethod
    def supported_calls(source):
        return (
            ("positional", lambda: torch.mean(source)),
            ("input", lambda: torch.mean(input=source)),
            ("x", lambda: torch.mean(x=source)),
            ("a", lambda: torch.mean(a=source)),
            ("x1", lambda: torch.mean(x1=source)),
            ("dtype none", lambda: torch.mean(source, dtype=None)),
            ("dtype float32", lambda: torch.mean(source, dtype=torch.float32)),
            ("dtype float alias", lambda: torch.mean(source, dtype=torch.float)),
            ("alias and dtype", lambda: torch.mean(x=source, dtype=torch.float32)),
        )

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
        raise AssertionError(f"unknown mean autograd case: {case}")

    def test_supported_forms_delegate_to_tensor_mean_values_metadata_and_storage(self):
        for case, source in self.value_cases():
            expected = source.mean()
            for form, call in self.supported_calls(source):
                self.assert_scalar_matches(call(), expected, source, case=(case, form))

    def test_supported_forms_preserve_autograd_accumulation_and_no_grad(self):
        forms = tuple(form for form, _ in self.supported_calls(torch.tensor(1.0)))
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form in forms:
                function_leaf, function_input = self.autograd_case(case)
                method_leaf, method_input = self.autograd_case(case)
                output = dict(self.supported_calls(function_input))[form]()
                expected = method_input.mean()
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
            untracked = torch.mean(leaf, dtype=torch.float)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertIsNone(leaf.grad)
        self.assertTrue(torch.mean(leaf, dtype=torch.float32).requires_grad)

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

        invalid_mode = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid_mode:
                torch.mean(tensor, out=None)
        self.assertEqual(invalid_mode.calls, [])

    def test_callable_metadata_documentation_pickling_and_exports(self):
        function = torch.mean
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "mean")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.mean")
        self.assertEqual(function.__module__, "torch")
        self.assertIn("mean(input, *, dtype=None) -> Tensor", function.__doc__)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method mean of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.mean, function)
        for action in (
            lambda: setattr(owner, "mean", None),
            lambda: delattr(owner, "mean"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.mean, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("mean"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["mean"], function)

    def test_binding_errors_document_the_supported_overload(self):
        tensor = torch.ones((2, 3))
        destination = torch.tensor([17.0, 19.0, 23.0])
        invalid = "mean() received an invalid combination of arguments - got "
        cases = (
            (lambda: torch.mean(), f"{invalid}(), {EXPECTED_OVERLOADS}"),
            (
                lambda: torch.mean(extra=True),
                'mean() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.mean(out=None),
                'mean() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.mean(1),
                "mean(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.mean(np.zeros((2, 3), dtype=np.float32)),
                "mean(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
            (
                lambda: torch.mean(1, dtype=torch.float32),
                f"{invalid}(int, dtype=torch.dtype), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, dtype=1),
                f"{invalid}(Tensor, dtype=int), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, dtype=object()),
                f"{invalid}(Tensor, dtype=object), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, torch.float32),
                f"{invalid}(Tensor, torch.dtype), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, 0, False, torch.float32),
                "mean() takes from 2 to 3 positional arguments but 4 were given",
            ),
            (
                lambda: torch.mean(tensor, input=tensor),
                f"{invalid}(Tensor, input=Tensor), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, keepdim=True),
                f"{invalid}(Tensor, keepdim=bool), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, out=None),
                f"{invalid}(Tensor, out=NoneType), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, out=destination),
                f"{invalid}(Tensor, out=Tensor), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, tensor),
                f"{invalid}(Tensor, Tensor), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, 0, dtype=1),
                f"{invalid}(Tensor, int, dtype=int), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.mean(tensor, 0, out=[]),
                f"{invalid}(Tensor, int, out=list), {EXPECTED_OVERLOADS}",
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
        self.assertFalse(hasattr(torch, "bool"))
        self.assertFalse(hasattr(torch, "int64"))
        self.assertFalse(hasattr(torch, "float64"))
        cases = (
            ("positional dim", lambda: torch.mean(tensor, 0)),
            ("positional none dim", lambda: torch.mean(tensor, None)),
            ("keyword dim", lambda: torch.mean(input=tensor, dim=0)),
            ("none dim", lambda: torch.mean(tensor, dim=None)),
            ("tuple dim", lambda: torch.mean(tensor, (0, 1))),
            ("list dim", lambda: torch.mean(tensor, [0, 1])),
            ("keepdim", lambda: torch.mean(tensor, 0, keepdim=True)),
            ("out", lambda: torch.mean(tensor, 0, out=destination)),
            ("dtype plus dim", lambda: torch.mean(tensor, 0, dtype=torch.float32)),
        )
        for case, call in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^mean\(\): dim, keepdim, and out reductions are not supported$",
                ):
                    call()
        self.assertEqual(destination.tolist(), [17.0, 19.0, 23.0])


if __name__ == "__main__":
    unittest.main()
