import copy
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .test_trunc import SPECIAL_OUTPUT_BITS, make_cases
else:
    from test_trunc import SPECIAL_OUTPUT_BITS, make_cases

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelFixReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.fix differentials require pinned PyTorch 2.13.0")

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
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
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
    def call_fix(module, tensor, form):
        if form == "positional":
            return module.fix(tensor)
        if form == "out none":
            return module.fix(tensor, out=None)
        if form == "alias and out none":
            return module.fix(x=tensor, out=None)
        return module.fix(**{form: tensor})

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("torch.fix unexpectedly accepted an invalid call")

    def test_values_ieee_bits_layouts_aliases_and_storage_match_pytorch_2_13(self):
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
                actual = self.call_fix(torch, actual_input, form)
                expected = self.call_fix(reference_torch, expected_input, form)
                self.assert_tensor_matches(actual, expected, case=(case, form))
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(
                        expected.data_ptr(), expected_input.data_ptr()
                    )
                if case == "numerical edges":
                    np.testing.assert_array_equal(
                        self.tensor_values(actual).reshape(-1).view(np.uint32),
                        SPECIAL_OUTPUT_BITS,
                    )
                    np.testing.assert_array_equal(
                        self.tensor_values(expected).reshape(-1).view(np.uint32),
                        SPECIAL_OUTPUT_BITS,
                    )

    def test_seeded_float32_values_match_pytorch_2_13_exactly(self):
        rng = np.random.default_rng(0xF1_213)
        shapes = [(), (0,), (2, 0, 5), (3, 1, 7)]
        for _ in range(20):
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
                torch.fix(actual_input),
                reference_torch.fix(expected_input),
                case=(case, shape),
            )

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        function = module.fix
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature": self.signature_outcome(function),
            "distinct_from_trunc": function is not module.trunc,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.fix is function,
            "owner_distinct_from_trunc": owner.fix is not owner.trunc,
            "all_count": module.__all__.count("fix"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["fix"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_distinct_builtin_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    @staticmethod
    def dispatch_observation(module):
        tensor = module.tensor([1.25], dtype=module.float32)
        destination = module.tensor([0.0], dtype=module.float32)
        function = module.fix
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
            (lambda: function(tensor), None),
            (lambda: function(input=tensor), ("input",)),
            (lambda: function(x=tensor), ("x",)),
            (lambda: function(tensor, out=None), ("out",)),
            (lambda: function(input=tensor, out=None), ("input", "out")),
            (lambda: function(tensor, out=destination), ("out",)),
        )
        for call, keyword_names in mode_calls:
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

        forwarding_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
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
            subclass_result is marker,
            subclass_order,
            forwarding_order,
            forwarded.tolist(),
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
        actual = torch.tensor([1.25])
        expected = reference_torch.tensor([1.25])
        cases = (
            (lambda: torch.fix(), lambda: reference_torch.fix()),
            (
                lambda: torch.fix(actual, actual),
                lambda: reference_torch.fix(expected, expected),
            ),
            (
                lambda: torch.fix(actual, input=actual),
                lambda: reference_torch.fix(expected, input=expected),
            ),
            (
                lambda: torch.fix(out=actual),
                lambda: reference_torch.fix(out=expected),
            ),
            (
                lambda: torch.fix(1, extra=True),
                lambda: reference_torch.fix(1, extra=True),
            ),
            (lambda: torch.fix(input=[]), lambda: reference_torch.fix(input=[])),
            (
                lambda: torch.fix(actual, out=[]),
                lambda: reference_torch.fix(expected, out=[]),
            ),
            (
                lambda: torch.fix(actual, extra=True),
                lambda: reference_torch.fix(expected, extra=True),
            ),
            (
                lambda: torch.fix(input=actual, a=actual),
                lambda: reference_torch.fix(input=expected, a=expected),
            ),
            (
                lambda: torch.fix(a=actual, x=actual, out=None),
                lambda: reference_torch.fix(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.fix(x=actual, a=actual, out=None),
                lambda: reference_torch.fix(x=expected, a=expected, out=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def test_inference_only_and_unsupported_boundaries_are_explicit(self):
        values = np.linspace(-3.75, 3.75, 24, dtype=np.float32).reshape(2, 3, 4)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_input = actual_leaf.transpose(0, 2)[1]
        expected_input = expected_leaf.transpose(0, 2)[1]

        with self.assertRaisesRegex(
            RuntimeError,
            r"^fix\(\): autograd recording is not supported$",
        ):
            torch.fix(actual_input)
        expected_tracked = reference_torch.fix(expected_input)
        self.assertTrue(expected_tracked.requires_grad)
        self.assertEqual(type(expected_tracked.grad_fn).__name__, "TruncBackward0")

        with torch.no_grad():
            actual_no_grad = torch.fix(actual_input, out=None)
        with reference_torch.no_grad():
            expected_no_grad = reference_torch.fix(expected_input, out=None)
        self.assert_tensor_matches(actual_no_grad, expected_no_grad, case="no_grad")

        actual_detached = torch.fix(actual_input.detach())
        expected_detached = reference_torch.fix(expected_input.detach())
        self.assert_tensor_matches(actual_detached, expected_detached, case="detached")

        actual_out = torch.tensor([17.0, 19.0])
        actual_out_pointer = actual_out.data_ptr()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^fix\(\): the 'out' argument is not supported$",
        ):
            torch.fix(torch.tensor([1.25, -1.25]), out=actual_out)
        self.assertEqual(actual_out.data_ptr(), actual_out_pointer)
        self.assertEqual(actual_out.tolist(), [17.0, 19.0])

        expected_out = reference_torch.tensor(
            [17.0, 19.0], dtype=reference_torch.float32
        )
        self.assertIs(
            reference_torch.fix(
                reference_torch.tensor([1.25, -1.25]), out=expected_out
            ),
            expected_out,
        )
        self.assertEqual(expected_out.tolist(), [1.0, -1.0])

        self.assertFalse(hasattr(torch.Tensor, "fix"))
        self.assertTrue(hasattr(reference_torch.Tensor, "fix"))
        self.assertFalse(hasattr(torch.Tensor, "fix_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "fix_"))
        self.assertFalse(hasattr(torch, "fix_"))
        self.assertTrue(hasattr(reference_torch, "fix_"))


if __name__ == "__main__":
    unittest.main()
