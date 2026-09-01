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
class CosReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError("torch.cos differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
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
            actual_values = np.asarray(actual, dtype=np.float32)
            expected_values = expected.detach().cpu().numpy()
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=2.0e-6,
                atol=np.nextafter(np.float32(0), np.float32(1)),
                equal_nan=True,
            )
            shared_zeros = (actual_values == 0) & (expected_values == 0)
            np.testing.assert_array_equal(
                np.signbit(actual_values[shared_zeros]),
                np.signbit(expected_values[shared_zeros]),
            )

    @staticmethod
    def make_cases(module):
        base = module.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3F00_0000,
                0xBF00_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            (
                "numerical edges",
                module.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

    @staticmethod
    def call_cos(module, tensor, form):
        if form == "method":
            return tensor.cos()
        if form == "positional":
            return module.cos(tensor)
        if form == "out none":
            return module.cos(tensor, out=None)
        if form == "alias and out none":
            return module.cos(x=tensor, out=None)
        return module.cos(**{form: tensor})

    @staticmethod
    def make_autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(1.5, dtype=module.float32, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf.transpose(0, 2)[1]

        leaf = module.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        if case == "offset":
            return leaf, leaf.transpose(0, 2)[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 2)
        raise AssertionError(f"unknown autograd case: {case}")

    def test_scalar_empty_offset_noncontiguous_and_edge_results_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        forms = (
            "method",
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
            case, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(case, expected_name)
            for form in forms:
                self.assert_matches(
                    self.call_cos(torch, actual_input, form),
                    self.call_cos(reference_torch, expected_input, form),
                    case=(case, form),
                )

    def test_full_sum_autograd_matches_negative_sine_vjp_pytorch_2_13(self):
        forms = (
            "method",
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
                actual_leaf, actual_input = self.make_autograd_case(torch, case)
                expected_leaf, expected_input = self.make_autograd_case(
                    reference_torch, case
                )
                actual = self.call_cos(torch, actual_input, form)
                expected = self.call_cos(reference_torch, expected_input, form)

                self.assert_matches(actual, expected, case=(case, form, "output"))
                actual.sum().backward()
                expected.sum().backward()
                self.assert_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(case, form, "gradient"),
                )

        input_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x3F00_0000,
                0xBF00_0000,
                0x3F80_0000,
                0xC000_0000,
                0x4049_0FDB,
                0x5015_02F9,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        input_values = input_bits.view(np.float32)
        actual_leaf = torch.tensor(memoryview(input_values), requires_grad=True)
        expected_leaf = reference_torch.tensor(input_values, requires_grad=True)

        torch.cos(actual_leaf).sum().backward()
        reference_torch.cos(expected_leaf).sum().backward()
        expected_formula = -expected_leaf.detach().sin()
        np.testing.assert_array_equal(
            expected_leaf.grad.detach().numpy().view(np.uint32),
            expected_formula.detach().numpy().view(np.uint32),
        )
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad).view(np.uint32),
            expected_formula.detach().numpy().view(np.uint32),
        )

    def tensor_callable_contract(self, module):
        tensor = module.tensor([0.5], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "cos")
        bound = tensor.cos
        try:
            descriptor_signature = str(inspect.signature(descriptor))
        except Exception as error:
            descriptor_signature = type(error).__name__
        try:
            bound_signature = str(inspect.signature(bound))
        except Exception as error:
            bound_signature = type(error).__name__
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
            "signatures": (descriptor_signature, bound_signature),
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
        }

    def top_level_callable_contract(self, module):
        function = module.cos
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
            "owner_callable_identity": owner.cos is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("cos"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["cos"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_imports_copy_pickle_and_reload_match_pytorch_2_13(self):
        self.assertEqual(
            self.tensor_callable_contract(torch),
            self.tensor_callable_contract(reference_torch),
        )
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

        old = torch.cos
        native = torch._C
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.cos, old)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.cos, old)

    def dispatch_observation(self, module):
        tensor = module.tensor([0.5], dtype=module.float32, requires_grad=True)
        destination = module.tensor([0.0], dtype=module.float32)
        function = module.cos
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_calls = (
            (lambda: tensor.cos(), inspect.getattr_static(module.Tensor, "cos")),
            (lambda: function(tensor), function),
            (lambda: function(input=tensor), function),
            (lambda: function(tensor, out=None), function),
            (lambda: function(input=tensor, out=destination), function),
        )
        for call, expected_function in mode_calls:
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is expected_function,
                    dispatch_types == (
                        (module.Tensor,) if expected_function is not function else ()
                    ),
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

        for call, keyword in (
            (lambda value: function(value), None),
            (lambda value: function(input=value), "input"),
            (lambda value: function(tensor, out=value), "out"),
            (lambda value: function(x=value, out=None), "x"),
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
                    keyword is not None
                    and kwargs is not None
                    and kwargs[keyword] is value,
                )
            )

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(("base", tuple(item.__name__ for item in types)))
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("derived", tuple(item.__name__ for item in types))
                )
                return marker

        subclass_result = function(BaseOverride(), out=DerivedOverride())

        forward_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forward_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=tensor, out=None)
        forwarded.sum().backward()
        forwarded_observation = (
            forwarded.requires_grad,
            forwarded.is_leaf,
            tuple(forwarded.shape),
            forwarded.stride(),
            forwarded.storage_offset(),
            tuple(np.asarray(forwarded.detach()).reshape(-1)),
            tuple(np.asarray(tensor.grad).reshape(-1)),
        )

        return (
            mode_observations,
            override_observations,
            subclass_result is marker,
            subclass_order,
            forward_order,
            forwarded_observation,
        )

    def test_torch_function_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.cos(), lambda: reference_torch.cos()),
            (
                lambda: torch.cos(actual, actual),
                lambda: reference_torch.cos(expected, expected),
            ),
            (
                lambda: torch.cos(actual, input=actual),
                lambda: reference_torch.cos(expected, input=expected),
            ),
            (
                lambda: torch.cos(out=actual),
                lambda: reference_torch.cos(out=expected),
            ),
            (
                lambda: torch.cos(1, extra=True),
                lambda: reference_torch.cos(1, extra=True),
            ),
            (lambda: torch.cos(input=[]), lambda: reference_torch.cos(input=[])),
            (
                lambda: torch.cos(actual, out=[]),
                lambda: reference_torch.cos(expected, out=[]),
            ),
            (
                lambda: torch.cos(actual, extra=True, out=[]),
                lambda: reference_torch.cos(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.cos(actual, extra=True),
                lambda: reference_torch.cos(expected, extra=True),
            ),
            (
                lambda: torch.cos(input=actual, a=actual),
                lambda: reference_torch.cos(input=expected, a=expected),
            ),
            (
                lambda: torch.cos(a=actual, x=actual, out=None),
                lambda: reference_torch.cos(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.cos(x=actual, a=actual, out=None),
                lambda: reference_torch.cos(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.cos(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.cos(np.zeros((2, 3), dtype=np.float32)),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
