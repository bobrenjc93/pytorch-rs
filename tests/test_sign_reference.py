import copy
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .test_sign import SPECIAL_OUTPUT_BITS, make_cases
else:
    from test_sign import SPECIAL_OUTPUT_BITS, make_cases

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorSignReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.sign differentials require pinned PyTorch 2.13.0")

    @staticmethod
    def tensor_values(tensor):
        if type(tensor) is torch.Tensor:
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy()

    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                self.tensor_values(actual).reshape(-1).view(np.uint32),
                self.tensor_values(expected).reshape(-1).view(np.uint32),
            )

    @staticmethod
    def call_sign(module, tensor, form):
        if form == "positional":
            return module.sign(tensor)
        if form == "out none":
            return module.sign(tensor, out=None)
        if form == "alias and out none":
            return module.sign(x=tensor, out=None)
        return module.sign(**{form: tensor})

    @staticmethod
    def make_autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(
                -1.25, dtype=module.float32, requires_grad=True
            )
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf.transpose(0, 2)[1]

        leaf = module.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        strided = leaf.transpose(0, 2)
        if case == "offset":
            return leaf, strided[1]
        if case == "noncontiguous":
            return leaf, strided
        raise AssertionError(f"unknown sign autograd case: {case}")

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("sign unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def test_method_values_layouts_offsets_empty_tensors_and_storage_match_pytorch(self):
        actual_cases = make_cases(torch)
        expected_cases = make_cases(reference_torch)
        for (case, actual_input, actual_stride), (
            expected_case,
            expected_input,
            expected_stride,
        ) in zip(actual_cases, expected_cases, strict=True):
            self.assertEqual(case, expected_case)
            self.assertEqual(actual_stride, expected_stride)
            actual = actual_input.sign()
            expected = expected_input.sign()
            self.assert_tensor_matches(actual, expected, case=case)
            self.assertFalse(actual.is_set_to(actual_input))
            self.assertFalse(expected.is_set_to(expected_input))
            if actual_input.numel():
                self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())
            if case == "numerical edges":
                np.testing.assert_array_equal(
                    self.tensor_values(actual).reshape(-1).view(np.uint32),
                    SPECIAL_OUTPUT_BITS,
                )
                np.testing.assert_array_equal(
                    self.tensor_values(expected).reshape(-1).view(np.uint32),
                    SPECIAL_OUTPUT_BITS,
                )

    def test_top_level_values_layouts_bits_and_storage_match_pytorch_2_13(self):
        actual_cases = make_cases(torch)
        expected_cases = make_cases(reference_torch)
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for (case, actual_input, actual_stride), (
            expected_case,
            expected_input,
            expected_stride,
        ) in zip(actual_cases, expected_cases, strict=True):
            self.assertEqual(case, expected_case)
            self.assertEqual(actual_stride, expected_stride)
            for form in forms:
                actual = self.call_sign(torch, actual_input, form)
                expected = self.call_sign(reference_torch, expected_input, form)
                self.assert_tensor_matches(actual, expected, case=(case, form))
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(
                        expected.data_ptr(), expected_input.data_ptr()
                    )

    def test_seeded_float32_values_match_pytorch_2_13_exactly(self):
        rng = np.random.default_rng(0x516E_213)
        shapes = [(), (0,), (2, 0, 5), (3, 1, 7)]
        for _ in range(24):
            rank = int(rng.integers(0, 6))
            shapes.append(
                tuple(int(value) for value in rng.integers(0, 9, size=rank))
            )

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.uniform(-1.0e6, 1.0e6, size=elements).astype(np.float32)
            if elements:
                values[::7] = np.float32(0.0)
                values[1::11] = np.float32(-0.0)
                values[2::13] = np.float32(np.inf)
                values[3::17] = np.float32(-np.inf)
                values[4::19] = np.float32(np.nan)
            values = values.reshape(shape)

            actual_input = (
                torch.zeros(shape)
                if elements == 0
                else torch.tensor(values.item() if shape == () else values.tolist())
            )
            expected_input = reference_torch.tensor(
                values, dtype=reference_torch.float32
            )
            self.assert_tensor_matches(
                actual_input.sign(), expected_input.sign(), case=(case, shape)
            )

    def test_autograd_zero_vjp_matches_pytorch_2_13(self):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            actual_leaf, actual_input = self.make_autograd_case(torch, case)
            expected_leaf, expected_input = self.make_autograd_case(
                reference_torch, case
            )
            actual_output = actual_input.sign()
            expected_output = expected_input.sign()
            self.assert_tensor_matches(
                actual_output, expected_output, case=(case, "forward")
            )

            actual_loss = actual_output if case == "scalar" else actual_output.sum()
            expected_loss = expected_output if case == "scalar" else expected_output.sum()
            actual_loss.backward()
            expected_loss.backward()
            self.assert_tensor_matches(
                actual_leaf.grad,
                expected_leaf.grad,
                case=(case, "gradient"),
            )

    def test_top_level_autograd_and_no_grad_match_pytorch_2_13(self):
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
                actual_leaf, actual_input = self.make_autograd_case(torch, case)
                expected_leaf, expected_input = self.make_autograd_case(
                    reference_torch, case
                )
                actual_output = self.call_sign(torch, actual_input, form)
                expected_output = self.call_sign(
                    reference_torch, expected_input, form
                )
                self.assert_tensor_matches(
                    actual_output,
                    expected_output,
                    case=(case, form, "forward"),
                )

                actual_loss = (
                    actual_output if case == "scalar" else actual_output.sum()
                )
                expected_loss = (
                    expected_output if case == "scalar" else expected_output.sum()
                )
                actual_loss.backward()
                expected_loss.backward()
                self.assert_tensor_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(case, form, "gradient"),
                )

        actual = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        expected = reference_torch.tensor(
            [-1.0, 0.0, 1.0], dtype=reference_torch.float32, requires_grad=True
        )
        with torch.no_grad():
            actual_result = torch.sign(actual, out=None)
        with reference_torch.no_grad():
            expected_result = reference_torch.sign(expected, out=None)
        self.assert_tensor_matches(actual_result, expected_result, case="no_grad")

    def callable_contract(self, module):
        tensor = module.tensor([1.25], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "sign")
        bound = tensor.sign
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
                    lambda: tensor.sign(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.sign(1, 2),
                    lambda: tensor.sign(input=tensor),
                    lambda: bound(unexpected=True),
                    lambda: descriptor(tensor, unexpected=True),
                    lambda: descriptor(),
                    lambda: descriptor(1),
                    lambda: descriptor(self=tensor),
                )
            ),
        }

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    def top_level_callable_contract(self, module):
        function = module.sign
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
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.sign is function,
            "all_count": module.__all__.count("sign"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["sign"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_top_level_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

    @staticmethod
    def dispatch_observation(module):
        tensor = module.tensor([1.25], dtype=module.float32)
        destination = module.tensor([0.0], dtype=module.float32)
        function = module.sign
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for call, keyword_names in (
            (lambda: function(tensor), None),
            (lambda: function(input=tensor), ("input",)),
            (lambda: function(x=tensor), ("x",)),
            (lambda: function(tensor, out=None), ("out",)),
            (lambda: function(input=tensor, out=None), ("input", "out")),
            (lambda: function(tensor, out=destination), ("out",)),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
            dispatched, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    dispatched is function,
                    dispatch_types == (),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword_names,
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
            dispatched, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    dispatched is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword is not None
                    and kwargs is not None
                    and kwargs[keyword] is value,
                )
            )

        spoofed_tensor_observations = []

        class Tensor:
            __module__ = "torch"
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, keyword in (
            (lambda value: function(value), None),
            (lambda value: function(tensor, out=value), "out"),
        ):
            value = Tensor()
            Tensor.calls.clear()
            result = call(value)
            dispatched, dispatch_types, args, kwargs = Tensor.calls[0]
            spoofed_tensor_observations.append(
                (
                    result is marker,
                    dispatched is function,
                    tuple(item.__name__ for item in dispatch_types),
                    tuple(item.__module__ for item in dispatch_types),
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
            lambda: function([], out=destination),
            lambda: function(tensor, out=[]),
            lambda: function(tensor, extra=True),
            lambda: function(tensor, tensor),
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
            spoofed_tensor_observations,
            subclass_result is marker,
            subclass_order,
            forward_order,
            forwarded.tolist(),
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_top_level_modes_and_subclass_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_top_level_declining_override_diagnostics_match_pytorch_2_13(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        self.assertEqual(
            self.error(lambda: torch.sign(Override())),
            self.error(lambda: reference_torch.sign(Override())),
        )
        self.assertEqual(
            self.error(lambda: torch.sign(torch.tensor([1.25]), out=Override())),
            self.error(
                lambda: reference_torch.sign(
                    reference_torch.tensor([1.25]), out=Override()
                )
            ),
        )

    def test_binding_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.25])
        expected = reference_torch.tensor([1.25])
        cases = (
            (lambda: torch.sign(), lambda: reference_torch.sign()),
            (
                lambda: torch.sign(actual, actual),
                lambda: reference_torch.sign(expected, expected),
            ),
            (
                lambda: torch.sign(actual, input=actual),
                lambda: reference_torch.sign(expected, input=expected),
            ),
            (
                lambda: torch.sign(out=actual),
                lambda: reference_torch.sign(out=expected),
            ),
            (
                lambda: torch.sign(extra=actual),
                lambda: reference_torch.sign(extra=expected),
            ),
            (
                lambda: torch.sign(1, extra=True),
                lambda: reference_torch.sign(1, extra=True),
            ),
            (lambda: torch.sign(input=[]), lambda: reference_torch.sign(input=[])),
            (
                lambda: torch.sign(actual, out=[]),
                lambda: reference_torch.sign(expected, out=[]),
            ),
            (
                lambda: torch.sign(actual, extra=True, out=[]),
                lambda: reference_torch.sign(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.sign(actual, extra=True),
                lambda: reference_torch.sign(expected, extra=True),
            ),
            (
                lambda: torch.sign(input=actual, a=actual),
                lambda: reference_torch.sign(input=expected, a=expected),
            ),
            (
                lambda: torch.sign(a=actual, x=actual, out=None),
                lambda: reference_torch.sign(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.sign(x=actual, a=actual, out=None),
                lambda: reference_torch.sign(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.sign(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.sign(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                actual_error = self.error(actual_call)
                expected_error = self.error(expected_call)
                self.assertEqual(actual_error, expected_error)

    def test_unsupported_foreign_dtype_and_device_tensors_are_rejected(self):
        unsupported = [
            reference_torch.tensor([1.0], dtype=reference_torch.float64),
            reference_torch.tensor([1], dtype=reference_torch.int64),
        ]
        if reference_torch.cuda.is_available():
            unsupported.append(
                reference_torch.tensor(
                    [1.0], dtype=reference_torch.float32, device="cuda"
                )
            )
        for tensor in unsupported:
            with self.subTest(dtype=str(tensor.dtype), device=str(tensor.device)):
                with self.assertRaises(TypeError):
                    torch.sign(tensor)


if __name__ == "__main__":
    unittest.main()
