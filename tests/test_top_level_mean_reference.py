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
class TopLevelMeanReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError("torch.mean differentials require pinned PyTorch 2.13.0")

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
        cancellation = (
            np.where(
                np.arange(120) % 2 == 0, np.float32(1.0e8), np.float32(-1.0e8)
            )
            + (np.arange(120, dtype=np.float32) % 7)
        ).reshape(3, 40)
        multiple_nan_values = np.asarray(
            [
                0x7FC1_2345,
                0x0000_0000,
                0xFFC5_4321,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
            ],
            dtype=np.uint32,
        ).view(np.float32)
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
            (
                "finite cancellation",
                module.tensor(
                    [0.0, 0.0, 1.0, 3.0, 123456789.0], dtype=module.float32
                ),
            ),
            (
                "noncontiguous cancellation",
                module.tensor(cancellation.tolist(), dtype=module.float32).transpose(0, 1),
            ),
            (
                "multiple NaNs",
                module.tensor(multiple_nan_values, dtype=module.float32),
            ),
            (
                "dense transposed multiple NaNs",
                module.tensor(
                    multiple_nan_values.reshape(2, 4).tolist(), dtype=module.float32
                ).transpose(0, 1),
            ),
            (
                "positive NaN",
                module.tensor(
                    np.asarray([0x7FC1_2345, 0x3F80_0000], dtype=np.uint32).view(
                        np.float32
                    ),
                    dtype=module.float32,
                ),
            ),
            ("infinity", module.tensor([float("inf"), 1.0], dtype=module.float32)),
        )

    @staticmethod
    def call_mean(module, source, form):
        if form == "positional":
            return module.mean(source)
        if form == "dtype none":
            return module.mean(source, dtype=None)
        if form == "dtype float32":
            return module.mean(source, dtype=module.float32)
        if form == "dtype float alias":
            return module.mean(source, dtype=module.float)
        if form == "alias and dtype":
            return module.mean(x=source, dtype=module.float32)
        if form == "out none":
            return module.mean(source, out=None)
        return module.mean(**{form: source})

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
        raise AssertionError(f"unknown torch.mean autograd case: {case}")

    def test_supported_values_metadata_and_storage_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "dtype none",
            "dtype float32",
            "dtype float alias",
            "alias and dtype",
            "out none",
        )
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(case, expected_name)
            for form in forms:
                self.assert_scalar_matches(
                    self.call_mean(torch, actual_input, form),
                    self.call_mean(reference_torch, expected_input, form),
                    actual_input,
                    expected_input,
                    case=(case, form),
                )

    def test_autograd_accumulation_empty_and_no_grad_match_pytorch_2_13(self):
        forms = (
            "positional",
            "dtype none",
            "dtype float32",
            "dtype float alias",
            "alias and dtype",
            "out none",
        )
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form in forms:
                actual_leaf, actual_input = self.autograd_case(torch, case)
                expected_leaf, expected_input = self.autograd_case(
                    reference_torch, case
                )
                actual_loss = self.call_mean(torch, actual_input, form)
                expected_loss = self.call_mean(reference_torch, expected_input, form)
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
            actual = torch.mean(input=actual_leaf, dtype=torch.float)
        with reference_torch.no_grad():
            expected = reference_torch.mean(
                input=expected_leaf, dtype=reference_torch.float
            )
        self.assert_scalar_matches(
            actual, expected, actual_leaf, expected_leaf, case="no_grad"
        )
        self.assertIsNone(actual_leaf.grad)

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        function = module.mean
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
            "owner_callable_identity": owner.mean is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature": self.signature_outcome(function),
            "all_count": module.__all__.count("mean"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["mean"] is function,
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
        destination = module.tensor(17.0, dtype=module.float32)
        function = module.mean
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
            ("out none", lambda: function(tensor, out=None), ("out",)),
            ("out", lambda: function(tensor, out=destination), ("out",)),
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
            ("default out", lambda value: function(tensor, out=value), "out"),
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

        return (
            mode_observations,
            override_observations,
            forward_order,
            forwarded.item(),
            tensor.grad.tolist(),
        )

    def test_modes_and_subclass_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_dimension_out_and_cross_dtype_boundaries_remain_unsupported(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones((2, 3), dtype=reference_torch.float32)

        expected_dim_results = (
            reference_torch.mean(expected, 0),
            reference_torch.mean(expected, None),
            reference_torch.mean(input=expected, dim=0),
            reference_torch.mean(expected, dim=None),
            reference_torch.mean(expected, (0, 1)),
            reference_torch.mean(expected, [0, 1]),
            reference_torch.mean(expected, 0, keepdim=True),
            reference_torch.mean(expected, 0, dtype=reference_torch.float32),
        )
        self.assertEqual(
            [tuple(result.shape) for result in expected_dim_results],
            [(3,), (), (3,), (), (), (), (1, 3), (3,)],
        )

        actual_dim_calls = (
            lambda: torch.mean(actual, 0),
            lambda: torch.mean(actual, None),
            lambda: torch.mean(input=actual, dim=0),
            lambda: torch.mean(actual, dim=None),
            lambda: torch.mean(actual, (0, 1)),
            lambda: torch.mean(actual, [0, 1]),
            lambda: torch.mean(actual, 0, keepdim=True),
            lambda: torch.mean(actual, 0, dtype=torch.float32),
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
            torch.mean(actual, out=actual_destination)
        expected_scalar_destination = reference_torch.tensor(
            17.0, dtype=reference_torch.float32
        )
        expected_scalar_out = reference_torch.mean(
            expected, out=expected_scalar_destination
        )
        self.assertIs(expected_scalar_out, expected_scalar_destination)
        self.assertEqual(expected_scalar_destination.shape, ())
        self.assertEqual(expected_scalar_destination.item(), 1.0)

        with self.assertRaises(NotImplementedError):
            torch.mean(actual, 0, out=actual_destination)
        expected_out = reference_torch.mean(expected, 0, out=expected_destination)
        self.assertIs(expected_out, expected_destination)
        self.assertEqual(actual_destination.tolist(), [17.0, 19.0, 23.0])
        self.assertEqual(expected_destination.tolist(), [1.0, 1.0, 1.0])

        with self.assertRaises(TypeError):
            torch.mean(actual, dtype=reference_torch.float64)
        expected_float64 = reference_torch.mean(
            expected, dtype=reference_torch.float64
        )
        self.assertIs(expected_float64.dtype, reference_torch.float64)

        for dtype in (reference_torch.int64, reference_torch.bool):
            with self.subTest(dtype=dtype):
                with self.assertRaises(RuntimeError):
                    reference_torch.ones((2, 3), dtype=dtype).mean()


if __name__ == "__main__":
    unittest.main()
