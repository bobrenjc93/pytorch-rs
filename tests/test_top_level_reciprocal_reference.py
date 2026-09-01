import copy
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
class TopLevelReciprocalReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError(
                "torch.reciprocal differentials require pinned PyTorch 2.13.0"
            )

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
        raise AssertionError("torch.reciprocal unexpectedly accepted an invalid call")

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
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    @staticmethod
    def make_cases(module):
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
                0x0080_0000,
                0x8080_0000,
                0x3EAA_AAAB,
                0xBEAA_AAAB,
                0x3F80_0000,
                0xBF80_0000,
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
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            (
                "empty singleton trailing",
                module.zeros((0, 1), dtype=module.float32),
            ),
            (
                "empty singleton middle",
                module.zeros((0, 1, 2), dtype=module.float32),
            ),
            (
                "empty singleton surrounding",
                module.zeros((1, 0, 1), dtype=module.float32),
            ),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            (
                "numerical edges",
                module.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

    @staticmethod
    def call_reciprocal(module, tensor, form):
        if form == "positional":
            return module.reciprocal(tensor)
        if form == "out none":
            return module.reciprocal(tensor, out=None)
        if form == "alias and out none":
            return module.reciprocal(x=tensor, out=None)
        return module.reciprocal(**{form: tensor})

    @staticmethod
    def autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(4.0, dtype=module.float32, requires_grad=True)
            return leaf, leaf, None
        if case == "empty":
            leaf = module.zeros((2, 0, 3), dtype=module.float32, requires_grad=True)
            return leaf, leaf.transpose(0, 2)[1], None
        if case == "offset":
            leaf = module.tensor(
                np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
                dtype=module.float32,
                requires_grad=True,
            )
            return leaf, leaf[1], None
        if case == "noncontiguous":
            leaf = module.tensor(
                np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
                dtype=module.float32,
                requires_grad=True,
            )
            input = leaf.transpose(0, 2)[1]
            weights = module.tensor(
                np.arange(1, 7, dtype=np.float32).reshape(3, 2).tolist(),
                dtype=module.float32,
            )
            return leaf, input, weights
        raise AssertionError(f"unknown reciprocal autograd case: {case}")

    def test_scalar_empty_offset_noncontiguous_and_ieee_bits_match_pytorch_2_13(
        self,
    ):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
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
            case, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(case, expected_name)
            for form in forms:
                actual = self.call_reciprocal(torch, actual_input, form)
                expected = self.call_reciprocal(
                    reference_torch, expected_input, form
                )
                self.assert_matches(actual, expected, case=(case, form))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(
                        expected.data_ptr(), expected_input.data_ptr()
                    )

    def test_autograd_scalar_empty_offset_noncontiguous_and_forms_match_pytorch_2_13(
        self,
    ):
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
                actual_leaf, actual_input, actual_weights = self.autograd_case(
                    torch, case
                )
                expected_leaf, expected_input, expected_weights = self.autograd_case(
                    reference_torch, case
                )
                actual_output = self.call_reciprocal(torch, actual_input, form)
                expected_output = self.call_reciprocal(
                    reference_torch, expected_input, form
                )
                self.assert_matches(
                    actual_output, expected_output, case=(case, form, "forward")
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
                self.assert_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(case, form, "gradient"),
                )

    def test_autograd_special_values_match_pytorch_2_13_bitwise(self):
        input_bits = np.asarray(
            (
                0x00000000,
                0x80000000,
                0x00000001,
                0x80000001,
                0x00800000,
                0x80800000,
                0x3E800000,
                0x3F800000,
                0x40000000,
                0x40800000,
                0x7F7FFFFF,
                0xFF7FFFFF,
                0x7F800000,
                0xFF800000,
                0x7F812345,
                0xFF812345,
                0x7FC12345,
                0xFFC54321,
            ),
            dtype=np.uint32,
        )
        weight_bits = np.asarray(
            (
                0x3F800000,
                0xBF800000,
                0x00000000,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x3F000000,
                0xBF000000,
                0x3F800000,
                0xBF800000,
                0x3F800000,
                0xBF800000,
                0x3F800000,
                0xBF800000,
                0x3F800000,
                0xBF800000,
                0x7FC01234,
                0xFFC05678,
            ),
            dtype=np.uint32,
        )
        tensors = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                memoryview(input_bits.view(np.float32)), requires_grad=True
            )
            weights = module.tensor(memoryview(weight_bits.view(np.float32)))
            output = module.reciprocal(leaf)
            (output * weights).sum().backward()
            tensors.append((output, leaf.grad))

        self.assert_matches(tensors[0][0], tensors[1][0], case="special forward")
        self.assert_matches(tensors[0][1], tensors[1][1], case="special gradient")

        tail_input_bits = np.asarray(
            (0x3F800000, 0x3F800000, 0x3F800000, 0xFFC54321),
            dtype=np.uint32,
        )
        tail_weight_bits = np.asarray(
            (0x3F800000, 0x3F800000, 0x3F800000, 0xFFC0BBBB),
            dtype=np.uint32,
        )
        tail_tensors = []
        for module in (torch, reference_torch):
            tail_leaf = module.tensor(
                memoryview(tail_input_bits.view(np.float32)), requires_grad=True
            )
            tail_weights = module.tensor(
                memoryview(tail_weight_bits.view(np.float32))
            )
            tail_output = module.reciprocal(tail_leaf)
            (tail_output * tail_weights).sum().backward()
            tail_tensors.append((tail_output, tail_leaf.grad))

        self.assert_matches(
            tail_tensors[0][0], tail_tensors[1][0], case="tail special forward"
        )
        self.assert_matches(
            tail_tensors[0][1], tail_tensors[1][1], case="tail special gradient"
        )

    def test_autograd_accumulation_and_graph_freeing_match_pytorch_2_13(self):
        snapshots = []
        for module in (torch, reference_torch):
            accumulated = module.tensor(
                [1.0, 2.0, -4.0], dtype=module.float32, requires_grad=True
            )
            module.reciprocal(accumulated).sum().backward()
            first = np.asarray(accumulated.grad).copy()
            module.reciprocal(input=accumulated).sum().backward()
            second = np.asarray(accumulated.grad).copy()

            freed = module.tensor(
                [1.0, 2.0, -4.0], dtype=module.float32, requires_grad=True
            )
            loss = module.reciprocal(freed, out=None).sum()
            loss.backward()
            second_backward_error = self.error(loss.backward)
            snapshots.append((first, second, second_backward_error))

        np.testing.assert_array_equal(snapshots[0][0], snapshots[1][0])
        np.testing.assert_array_equal(snapshots[0][1], snapshots[1][1])
        self.assertEqual(snapshots[0][2], snapshots[1][2])

    def test_requires_grad_inputs_match_inside_no_grad(self):
        actual_leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        actual_input = actual_leaf.transpose(0, 1)[1]
        expected_input = expected_leaf.transpose(0, 1)[1]

        for form in ("positional", "input", "x", "a", "x1", "out none"):
            with torch.no_grad():
                actual = self.call_reciprocal(torch, actual_input, form)
            with reference_torch.no_grad():
                expected = self.call_reciprocal(
                    reference_torch, expected_input, form
                )
            self.assert_matches(actual, expected, case=form)

    def callable_contract(self, module):
        function = module.reciprocal
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
            "owner_callable_identity": owner.reciprocal is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("reciprocal"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["reciprocal"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def dispatch_observation(self, module):
        tensor = module.tensor([4.0])
        destination = module.tensor([0.0])
        function = module.reciprocal
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
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
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
        forwarded_observation = (
            forwarded.requires_grad,
            forwarded.is_leaf,
            tuple(forwarded.shape),
            forwarded.stride(),
            forwarded.storage_offset(),
            tuple(np.asarray(forwarded).reshape(-1)),
        )

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
            forward_order,
            forwarded_observation,
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_modes_and_subclass_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_declining_override_diagnostic_matches_pytorch_2_13(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        self.assert_error_matches(
            lambda: torch.reciprocal(Override()),
            lambda: reference_torch.reciprocal(Override()),
        )
        self.assert_error_matches(
            lambda: torch.reciprocal(torch.tensor([1.0]), out=Override()),
            lambda: reference_torch.reciprocal(
                reference_torch.tensor([1.0]), out=Override()
            ),
        )

    def test_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.reciprocal(), lambda: reference_torch.reciprocal()),
            (
                lambda: torch.reciprocal(actual, actual),
                lambda: reference_torch.reciprocal(expected, expected),
            ),
            (
                lambda: torch.reciprocal(actual, input=actual),
                lambda: reference_torch.reciprocal(expected, input=expected),
            ),
            (
                lambda: torch.reciprocal(out=actual),
                lambda: reference_torch.reciprocal(out=expected),
            ),
            (
                lambda: torch.reciprocal(1, extra=True),
                lambda: reference_torch.reciprocal(1, extra=True),
            ),
            (
                lambda: torch.reciprocal(input=[]),
                lambda: reference_torch.reciprocal(input=[]),
            ),
            (
                lambda: torch.reciprocal(actual, out=[]),
                lambda: reference_torch.reciprocal(expected, out=[]),
            ),
            (
                lambda: torch.reciprocal(actual, extra=True, out=[]),
                lambda: reference_torch.reciprocal(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.reciprocal(actual, extra=True),
                lambda: reference_torch.reciprocal(expected, extra=True),
            ),
            (
                lambda: torch.reciprocal(input=actual, a=actual),
                lambda: reference_torch.reciprocal(input=expected, a=expected),
            ),
            (
                lambda: torch.reciprocal(a=actual, x=actual, out=None),
                lambda: reference_torch.reciprocal(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.reciprocal(x=actual, a=actual, out=None),
                lambda: reference_torch.reciprocal(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.reciprocal(
                    np.zeros((2, 3), dtype=np.float32)
                ),
                lambda: reference_torch.reciprocal(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_deliberately_unsupported_surface_remains_narrow(self):
        actual = torch.tensor([2.0], requires_grad=True)
        expected = reference_torch.tensor([2.0], requires_grad=True)
        self.assertTrue(torch.reciprocal(actual).requires_grad)
        self.assertTrue(reference_torch.reciprocal(expected).requires_grad)
        self.assert_matches(
            torch.reciprocal(actual.detach()),
            reference_torch.reciprocal(expected.detach()),
            case="detach",
        )

        destination = torch.tensor([17.0])
        with self.assertRaisesRegex(
            RuntimeError,
            r"^reciprocal\(\): the 'out' argument is not supported$",
        ):
            torch.reciprocal(torch.tensor([2.0]), out=destination)
        self.assertEqual(destination.tolist(), [17.0])

        self.assertTrue(hasattr(torch.Tensor, "reciprocal"))
        self.assertFalse(hasattr(torch.Tensor, "reciprocal_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "reciprocal"))
        self.assertTrue(hasattr(reference_torch.Tensor, "reciprocal_"))
        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(reference_torch, "float64"))
        with self.assertRaises(RuntimeError):
            torch.device("cuda")
        self.assertEqual(reference_torch.device("cuda").type, "cuda")


if __name__ == "__main__":
    unittest.main()
