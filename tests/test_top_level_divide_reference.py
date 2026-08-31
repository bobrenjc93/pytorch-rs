import copy
import importlib
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
class TopLevelDivideReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.div differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

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

    def test_supported_values_layouts_scalars_and_no_grad_match_pytorch_2_13(self):
        actual_left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]],
            dtype=reference_torch.float32,
        ).transpose(0, 2)
        actual_right = torch.tensor([[1.0], [2.0], [4.0]])
        expected_right = reference_torch.tensor(
            [[1.0], [2.0], [4.0]], dtype=reference_torch.float32
        )

        for actual_function, expected_function in (
            (torch.div, reference_torch.div),
            (torch.divide, reference_torch.divide),
        ):
            calls = (
                (
                    "positional tensors",
                    lambda actual_function=actual_function: actual_function(
                        actual_left, actual_right
                    ),
                    lambda expected_function=expected_function: expected_function(
                        expected_left, expected_right
                    ),
                ),
                (
                    "canonical keywords",
                    lambda actual_function=actual_function: actual_function(
                        input=actual_left, other=actual_right
                    ),
                    lambda expected_function=expected_function: expected_function(
                        input=expected_left, other=expected_right
                    ),
                ),
                (
                    "legacy aliases",
                    lambda actual_function=actual_function: actual_function(
                        x1=actual_left, x2=actual_right
                    ),
                    lambda expected_function=expected_function: expected_function(
                        x1=expected_left, x2=expected_right
                    ),
                ),
                (
                    "rounding none",
                    lambda actual_function=actual_function: actual_function(
                        actual_left, actual_right, rounding_mode=None
                    ),
                    lambda expected_function=expected_function: expected_function(
                        expected_left, expected_right, rounding_mode=None
                    ),
                ),
                (
                    "out none",
                    lambda actual_function=actual_function: actual_function(
                        actual_left, actual_right, out=None
                    ),
                    lambda expected_function=expected_function: expected_function(
                        expected_left, expected_right, out=None
                    ),
                ),
                (
                    "tensor scalar",
                    lambda actual_function=actual_function: actual_function(
                        actual_left[1], np.float32(-0.0)
                    ),
                    lambda expected_function=expected_function: expected_function(
                        expected_left[1], np.float32(-0.0)
                    ),
                ),
                (
                    "scalar tensor",
                    lambda actual_function=actual_function: actual_function(
                        np.int64(4), actual_left[1]
                    ),
                    lambda expected_function=expected_function: expected_function(
                        np.int64(4), expected_left[1]
                    ),
                ),
                (
                    "keyword scalar tensor",
                    lambda actual_function=actual_function: actual_function(
                        input=-2.5, other=actual_left[1]
                    ),
                    lambda expected_function=expected_function: expected_function(
                        input=-2.5, other=expected_left[1]
                    ),
                ),
            )
            for case, actual_call, expected_call in calls:
                self.assert_matches(
                    actual_call(), expected_call(), case=(actual_function.__name__, case)
                )

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        self.assert_matches(
            torch.div(actual_empty, torch.ones((1, 1, 2))),
            reference_torch.div(
                expected_empty, reference_torch.ones((1, 1, 2))
            ),
            case="strided broadcast empty",
        )

        actual_grad = torch.tensor([[1.0, -0.0]], requires_grad=True)
        expected_grad = reference_torch.tensor(
            [[1.0, -0.0]], dtype=reference_torch.float32, requires_grad=True
        )
        with torch.no_grad():
            actual_no_grad = torch.divide(2.0, actual_grad.transpose(0, 1))
        with reference_torch.no_grad():
            expected_no_grad = reference_torch.divide(
                2.0, expected_grad.transpose(0, 1)
            )
        self.assert_matches(actual_no_grad, expected_no_grad, case="no_grad")

        for actual_function in (torch.div, torch.divide):
            with self.assertRaisesRegex(
                RuntimeError,
                rf"^{actual_function.__name__}\(\): autograd recording is not supported$",
            ):
                actual_function(actual_grad, 2.0)

    def test_signed_zero_nan_and_infinity_match_pytorch_2_13(self):
        left_bits = np.asarray(
            (
                0x3F80_0000,
                0xBF80_0000,
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
            ),
            dtype=np.uint32,
        )
        right_bits = np.asarray(
            (
                0x8000_0000,
                0x0000_0000,
                0x3F80_0000,
                0xBF80_0000,
                0xFF80_0000,
                0x7F80_0000,
                0x3F80_0000,
            ),
            dtype=np.uint32,
        )
        actual_left = torch.tensor(memoryview(left_bits.view(np.float32)))
        actual_right = torch.tensor(memoryview(right_bits.view(np.float32)))
        expected_left = reference_torch.tensor(memoryview(left_bits.view(np.float32)))
        expected_right = reference_torch.tensor(memoryview(right_bits.view(np.float32)))

        for actual_function, expected_function in (
            (torch.div, reference_torch.div),
            (torch.divide, reference_torch.divide),
        ):
            self.assert_matches(
                actual_function(actual_left, actual_right),
                expected_function(expected_left, expected_right),
                case=actual_function.__name__,
            )
            self.assert_matches(
                actual_function(-0.0, actual_right),
                expected_function(-0.0, expected_right),
                case=(actual_function.__name__, "left scalar"),
            )
            self.assert_matches(
                actual_function(actual_left, -0.0),
                expected_function(expected_left, -0.0),
                case=(actual_function.__name__, "right scalar"),
            )

    def dispatch_observation(self, module, function_name):
        left = module.tensor([4.0])
        right = module.tensor([2.0])
        destination = module.tensor([0.0])
        function = getattr(module, function_name)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        observations = []
        for call in (
            lambda: function(left, right),
            lambda: function(left, 4.0),
            lambda: function(4.0, left),
            lambda: function(input=4.0, other=left, rounding_mode=None),
            lambda: function(left, right, rounding_mode="floor", out=destination),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                )
            )

        override_events = []

        class LeftOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_events.append(("left", func, types, len(args), kwargs))
                return NotImplemented

        class RightOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_events.append(("right", func, types, len(args), kwargs))
                return marker

        override_result = function(LeftOverride(), RightOverride())
        normalized_override_events = tuple(
            (
                label,
                func is function,
                tuple(item.__name__ for item in types),
                arg_count,
                kwargs is None,
            )
            for label, func, types, arg_count, kwargs in override_events
        )

        return observations, override_result is marker, normalized_override_events

    def test_modes_and_overrides_match_pytorch_2_13(self):
        for name in ("div", "divide"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.dispatch_observation(torch, name),
                    self.dispatch_observation(reference_torch, name),
                )

    def callable_contract(self, module, name):
        function = getattr(module, name)
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
            "distinct_from_other_alias": function is not getattr(
                module, "divide" if name == "div" else "div"
            ),
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": getattr(owner, name) is function,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count(name),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace[name] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_imports_and_reload_match_pytorch_2_13(self):
        from torch_rs import div as imported_div
        from torch_rs import divide as imported_divide

        self.assertIs(imported_div, torch.div)
        self.assertIs(imported_divide, torch.divide)
        for name in ("div", "divide"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.callable_contract(torch, name),
                    self.callable_contract(reference_torch, name),
                )

        old_div = torch.div
        old_divide = torch.divide
        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.div, old_div)
        self.assertIs(torch.divide, old_divide)

    def test_unsupported_boundaries_are_explicit(self):
        tensor = torch.tensor([1.0])
        destination = torch.tensor([0.0])
        for function in (torch.div, torch.divide):
            with self.subTest(function=function.__name__, rounding_mode="trunc"):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf"^{function.__name__}\(\): non-None rounding_mode is not supported$",
                ):
                    function(tensor, tensor, rounding_mode="trunc")
            with self.subTest(function=function.__name__, out=True):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{function.__name__}\(\): the 'out' argument is not supported$",
                ):
                    function(tensor, tensor, out=destination)
                self.assertEqual(destination.tolist(), [0.0])


if __name__ == "__main__":
    unittest.main()
