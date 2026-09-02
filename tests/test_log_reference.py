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


SPECIAL_INPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x807F_FFFF,
        0x0080_0000,
        0x8080_0000,
        0x3EAA_AAAB,
        0xBEAA_AAAB,
        0x3F80_0000,
        0xBF80_0000,
        0x4000_0000,
        0xC000_0000,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x7F80_0000,
        0xFF80_0000,
        0x7F81_2345,
        0xFF81_2345,
        0x7FC1_2345,
        0xFFC5_4321,
    ),
    dtype=np.uint32,
)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorLogReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.log differentials require pinned PyTorch 2.13.0")

    def assert_tensor_matches(self, actual, expected, *, case, exact_bits=False):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        actual_values = np.asarray(actual, dtype=np.float32)
        expected_values = expected.detach().cpu().numpy()
        with self.subTest(case=case, values=True):
            if exact_bits:
                np.testing.assert_array_equal(
                    actual_values.reshape(-1).view(np.uint32),
                    expected_values.reshape(-1).view(np.uint32),
                )
            else:
                np.testing.assert_allclose(
                    actual_values,
                    expected_values,
                    rtol=2.0e-6,
                    atol=np.nextafter(np.float32(0), np.float32(1)),
                    equal_nan=True,
                )

    @staticmethod
    def tensor_cases(module):
        base = module.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x3EAA_AAAB,
                0xBEAA_AAAB,
                0x3F80_0000,
                0xBF80_0000,
                0x4000_0000,
                0xC000_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        return (
            ("scalar", module.tensor(1.0, dtype=module.float32), False),
            (
                "empty offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
                False,
            ),
            (
                "empty singleton trailing",
                module.zeros((0, 1), dtype=module.float32),
                False,
            ),
            ("contiguous", base, False),
            ("offset", strided[1], False),
            ("noncontiguous", strided, False),
            (
                "numerical edges",
                module.tensor(memoryview(special_bits.view(np.float32))),
                True,
            ),
        )

    @staticmethod
    def call_top_level(module, tensor, form):
        if form == "positional":
            return module.log(tensor)
        if form == "out none":
            return module.log(tensor, out=None)
        if form == "alias and out none":
            return module.log(x=tensor, out=None)
        return module.log(**{form: tensor})

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.log unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    @staticmethod
    def make_autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(2.0, dtype=module.float32, requires_grad=True)
            return leaf, leaf, None
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf.transpose(0, 2)[1], None

        values = np.linspace(0.25, 6.0, 24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )
        if case == "offset":
            source = leaf[1]
            weights = module.tensor(
                np.linspace(-2.0, 2.0, 12, dtype=np.float32)
                .reshape(3, 4)
                .tolist(),
                dtype=module.float32,
            )
            return leaf, source, weights
        if case == "noncontiguous":
            source = leaf.transpose(0, 2)[1]
            weights = module.tensor(
                np.linspace(-2.0, 2.0, 6, dtype=np.float32)
                .reshape(3, 2)
                .tolist(),
                dtype=module.float32,
            )
            return leaf, source, weights
        raise AssertionError(f"unknown Tensor.log autograd case: {case}")

    def test_values_layouts_fresh_storage_and_no_grad_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual, exact_bits = actual_case
            expected_name, expected, expected_exact_bits = expected_case
            self.assertEqual(case, expected_name)
            self.assertEqual(exact_bits, expected_exact_bits)
            actual_before = (
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32).copy()
            )
            expected_before = (
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32).copy()
            )
            actual_output = actual.log()
            expected_output = expected.log()
            self.assert_tensor_matches(
                actual_output,
                expected_output,
                case=case,
                exact_bits=exact_bits,
            )
            self.assertFalse(actual_output.is_set_to(actual))
            self.assertFalse(expected_output.is_set_to(expected))
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                actual_before,
            )
            np.testing.assert_array_equal(
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
                expected_before,
            )

        actual_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_leaf = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32, requires_grad=True
        )
        with torch.no_grad():
            actual_no_grad = actual_leaf.log()
        with reference_torch.no_grad():
            expected_no_grad = expected_leaf.log()
        self.assert_tensor_matches(actual_no_grad, expected_no_grad, case="no_grad")

    def test_top_level_forms_and_unary_layouts_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input, exact_bits = actual_case
            expected_name, expected_input, expected_exact_bits = expected_case
            self.assertEqual(case, expected_name)
            self.assertEqual(exact_bits, expected_exact_bits)
            for form in forms:
                actual = self.call_top_level(torch, actual_input, form)
                expected = self.call_top_level(reference_torch, expected_input, form)
                self.assert_tensor_matches(
                    actual,
                    expected,
                    case=(case, form),
                    exact_bits=exact_bits,
                )
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())

    def test_scalar_empty_offset_and_noncontiguous_autograd_match_pytorch_2_13(self):
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form in forms:
                actual_leaf, actual_input, actual_weights = self.make_autograd_case(
                    torch, case
                )
                expected_leaf, expected_input, expected_weights = (
                    self.make_autograd_case(reference_torch, case)
                )
                actual_output = self.call_top_level(torch, actual_input, form)
                expected_output = self.call_top_level(
                    reference_torch, expected_input, form
                )
                self.assert_tensor_matches(
                    actual_output,
                    expected_output,
                    case=(case, form, "forward"),
                )
                if case == "scalar":
                    self.assertTrue(actual_output.requires_grad)
                    self.assertFalse(actual_output.is_leaf)
                    self.assertEqual(
                        type(expected_output.grad_fn).__name__, "LogBackward0"
                    )

                if actual_weights is None:
                    actual_loss = (
                        actual_output if case == "scalar" else actual_output.sum()
                    )
                    expected_loss = (
                        expected_output if case == "scalar" else expected_output.sum()
                    )
                else:
                    actual_loss = (actual_output * actual_weights).sum()
                    expected_loss = (expected_output * expected_weights).sum()
                actual_loss.backward()
                expected_loss.backward()
                self.assert_tensor_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(case, form, "gradient"),
                )

    def test_special_value_autograd_matches_pytorch_2_13_bitwise(self):
        weight_bits = np.asarray(
            (
                0x3F80_0000,
                0xBF80_0000,
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x3F00_0000,
                0xBF00_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7FC0_1234,
                0xFFC0_5678,
            ),
            dtype=np.uint32,
        )
        snapshots = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                memoryview(SPECIAL_INPUT_BITS.view(np.float32)),
                dtype=module.float32,
                requires_grad=True,
            )
            weights = module.tensor(
                memoryview(weight_bits.view(np.float32)), dtype=module.float32
            )
            output = module.log(leaf, out=None)
            (output * weights).sum().backward()
            snapshots.append((output, leaf.grad))

        self.assert_tensor_matches(
            snapshots[0][0],
            snapshots[1][0],
            case="special forward",
            exact_bits=True,
        )
        self.assert_tensor_matches(
            snapshots[0][1],
            snapshots[1][1],
            case="special gradient",
            exact_bits=True,
        )

    def test_accumulation_freed_graph_no_grad_and_detach_match_pytorch_2_13(self):
        snapshots = []
        for module in (torch, reference_torch):
            accumulated = module.tensor(
                [0.25, 1.0, 4.0, 16.0],
                dtype=module.float32,
                requires_grad=True,
            )
            module.log(accumulated, out=None).sum().backward()
            first = np.asarray(accumulated.grad, dtype=np.float32).copy()
            module.log(input=accumulated).sum().backward()
            second = np.asarray(accumulated.grad, dtype=np.float32).copy()

            freed = module.tensor(
                [0.5, 1.0, 2.0], dtype=module.float32, requires_grad=True
            )
            loss = module.log(freed, out=None).sum()
            loss.backward()
            second_backward_error = self.error(loss.backward)

            no_grad_leaf = module.tensor(
                [[0.25, 1.0, 2.0], [4.0, 8.0, 16.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            no_grad_input = no_grad_leaf.transpose(0, 1)[1]
            with module.no_grad():
                no_grad_output = module.log(no_grad_input, out=None)
            detached_output = module.log(no_grad_input.detach())

            snapshots.append(
                (
                    first,
                    second,
                    second_backward_error,
                    no_grad_output,
                    detached_output,
                    no_grad_leaf.grad,
                )
            )

        np.testing.assert_array_equal(snapshots[0][0], snapshots[1][0])
        np.testing.assert_array_equal(snapshots[0][1], snapshots[1][1])
        self.assertEqual(snapshots[0][2], snapshots[1][2])
        self.assert_tensor_matches(
            snapshots[0][3], snapshots[1][3], case="no_grad output"
        )
        self.assert_tensor_matches(
            snapshots[0][4], snapshots[1][4], case="detached output"
        )
        self.assertIsNone(snapshots[0][5])
        self.assertIsNone(snapshots[1][5])

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "log")
        bound = tensor.log
        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "descriptor_repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "signatures": (
                self.signature_outcome(descriptor),
                self.signature_outcome(bound),
            ),
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
            "errors": tuple(
                self.error(call)
                for call in (
                    lambda: tensor.log(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.log(1, 2),
                    lambda: tensor.log(input=tensor),
                    lambda: bound(unexpected=True),
                    lambda: descriptor(tensor, unexpected=True),
                    lambda: descriptor(),
                    lambda: descriptor(1),
                    lambda: descriptor(self=tensor),
                )
            ),
        }

    def test_tensor_method_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def top_level_callable_contract(self, module):
        function = module.log
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
            "owner_callable_identity": owner.log is function,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("log"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["log"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_top_level_callable_contract_matches_pytorch_2_13_except_docs(self):
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

        actual_function = torch.log
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.log, actual_function)

    @staticmethod
    def top_level_dispatch_observation(module):
        tensor = module.tensor([1.0], dtype=module.float32)
        destination = module.tensor([0.0], dtype=module.float32)
        function = module.log
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_observations = []
        for call in (
            lambda: function(tensor),
            lambda: function(input=tensor),
            lambda: function(x=tensor),
            lambda: function(input=tensor, out=destination),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call in (
            lambda value: function(value),
            lambda value: function(input=value),
            lambda value: function(tensor, out=value),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                )
            )

        return mode_observations, override_observations

    def test_top_level_mode_and_override_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_dispatch_observation(torch),
            self.top_level_dispatch_observation(reference_torch),
        )

    def test_top_level_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.log(), lambda: reference_torch.log()),
            (
                lambda: torch.log(actual, actual),
                lambda: reference_torch.log(expected, expected),
            ),
            (
                lambda: torch.log(actual, input=actual),
                lambda: reference_torch.log(expected, input=expected),
            ),
            (
                lambda: torch.log(out=actual),
                lambda: reference_torch.log(out=expected),
            ),
            (
                lambda: torch.log(extra=actual),
                lambda: reference_torch.log(extra=expected),
            ),
            (
                lambda: torch.log(1, extra=True),
                lambda: reference_torch.log(1, extra=True),
            ),
            (lambda: torch.log(input=[]), lambda: reference_torch.log(input=[])),
            (
                lambda: torch.log(actual, out=[]),
                lambda: reference_torch.log(expected, out=[]),
            ),
            (
                lambda: torch.log(actual, extra=True, out=[]),
                lambda: reference_torch.log(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.log(actual, extra=True),
                lambda: reference_torch.log(expected, extra=True),
            ),
            (
                lambda: torch.log(input=actual, a=actual),
                lambda: reference_torch.log(input=expected, a=expected),
            ),
            (
                lambda: torch.log(a=actual, x=actual, out=None),
                lambda: reference_torch.log(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.log(x=actual, a=actual, out=None),
                lambda: reference_torch.log(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.log(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.log(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def test_unsupported_out_dtype_device_subclass_and_higher_order_boundaries(self):
        actual = torch.tensor([1.0, 2.0], requires_grad=True)
        self.assertTrue(actual.log().requires_grad)
        self.assertTrue(torch.log(actual).requires_grad)

        expected = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32, requires_grad=True
        )
        self.assertTrue(expected.log().requires_grad)
        self.assertTrue(reference_torch.log(expected).requires_grad)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            torch.log(actual).sum().backward(create_graph=True)

        with torch.no_grad():
            actual_no_grad = torch.log(actual, out=None)
        with reference_torch.no_grad():
            expected_no_grad = reference_torch.log(expected, out=None)
        self.assert_tensor_matches(actual_no_grad, expected_no_grad, case="no_grad")

        destination = torch.tensor([17.0, 19.0])
        with self.assertRaisesRegex(
            RuntimeError,
            r"^log\(\): the 'out' argument is not supported$",
        ):
            torch.log(torch.tensor([1.0, 2.0]), out=destination)
        self.assertEqual(destination.tolist(), [17.0, 19.0])

        expected_out = reference_torch.tensor(
            [17.0, 19.0], dtype=reference_torch.float32
        )
        self.assertIs(
            reference_torch.log(
                reference_torch.tensor([1.0, 2.0], dtype=reference_torch.float32),
                out=expected_out,
            ),
            expected_out,
        )
        self.assertEqual(
            expected_out.tolist(),
            [0.0, float(np.log(np.float32(2.0)))],
        )

        self.assertFalse(hasattr(torch.Tensor, "log_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "log_"))
        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            TypeError,
            r"^tensor\(\): argument 'dtype' must be torch.dtype, not object$",
        ):
            torch.tensor([1.0], dtype=object())
        self.assertEqual(
            reference_torch.tensor([1.0], dtype=reference_torch.float64)
            .log()
            .dtype,
            reference_torch.float64,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda")
        with self.assertRaisesRegex(
            TypeError, r"^type 'torch_rs\.Tensor' is not an acceptable base type$"
        ):
            type("TensorSubclass", (torch.Tensor,), {})


if __name__ == "__main__":
    unittest.main()
