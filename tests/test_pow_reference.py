import inspect
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorPowReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("pow differentials require pinned PyTorch 2.13.0")

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
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
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
                0x0080_0000,
                0x8080_0000,
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
            ("offset", strided[1]),
            ("noncontiguous", strided),
            (
                "signed zero and non-finites",
                module.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

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
        raise AssertionError(f"unknown pow autograd case: {case}")

    @staticmethod
    def supported_calls(module, source):
        return (
            ("method int", lambda: source.pow(2)),
            ("method float", lambda: source.pow(2.0)),
            ("method keyword", lambda: source.pow(exponent=2)),
            ("dunder", lambda: source.__pow__(2)),
            ("dunder keyword", lambda: source.__pow__(exponent=2)),
            ("operator", lambda: source**2),
            ("top positional int", lambda: module.pow(source, 2)),
            ("top positional float", lambda: module.pow(source, 2.0)),
            ("top keyword exponent", lambda: module.pow(source, exponent=2)),
            ("top all keywords", lambda: module.pow(input=source, exponent=2)),
            ("top out none", lambda: module.pow(source, 2, out=None)),
        )

    def test_scalar_empty_offset_noncontiguous_and_ieee_values_match_pytorch_2_13(
        self,
    ):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for (case, actual), (_, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            for (form, actual_call), (_, expected_call) in zip(
                self.supported_calls(torch, actual),
                self.supported_calls(reference_torch, expected),
                strict=True,
            ):
                actual_output = actual_call()
                expected_output = expected_call()
                self.assert_tensor_matches(
                    actual_output, expected_output, case=(case, form)
                )
                self.assertFalse(actual_output.is_set_to(actual))
                self.assertFalse(expected_output.is_set_to(expected))

    def test_backward_through_sum_matches_pytorch_2_13(self):
        forms = tuple(form for form, _ in self.supported_calls(torch, torch.tensor(1.0)))
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form in forms:
                actual_leaf, actual_input = self.autograd_case(torch, case)
                expected_leaf, expected_input = self.autograd_case(reference_torch, case)
                actual_output = dict(self.supported_calls(torch, actual_input))[form]()
                expected_output = dict(
                    self.supported_calls(reference_torch, expected_input)
                )[form]()

                self.assert_tensor_matches(
                    actual_output, expected_output, case=(case, form, "forward")
                )
                actual_output.sum().backward()
                expected_output.sum().backward()
                self.assert_tensor_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(case, form, "gradient"),
                )

    def test_operator_reflected_fallback_matches_pytorch_2_13(self):
        def forwarded_operator(module, tensor, rhs):
            class ForwardingMode(module.overrides.TorchFunctionMode):
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    return func(*args, **(kwargs or {}))

            with ForwardingMode():
                return tensor**rhs

        class ReflectedPower:
            def __init__(self, marker):
                self.marker = marker
                self.other = None

            def __rpow__(self, other):
                self.other = other
                return self.marker

        actual_tensor = torch.tensor([2.0])
        expected_tensor = reference_torch.tensor([2.0], dtype=reference_torch.float32)
        actual_marker = object()
        expected_marker = object()
        actual_rhs = ReflectedPower(actual_marker)
        expected_rhs = ReflectedPower(expected_marker)

        self.assertIs(actual_tensor.__pow__(actual_rhs), NotImplemented)
        self.assertIs(expected_tensor.__pow__(expected_rhs), NotImplemented)
        self.assertIs(actual_tensor ** actual_rhs, actual_marker)
        self.assertIs(expected_tensor ** expected_rhs, expected_marker)
        self.assertIs(actual_rhs.other, actual_tensor)
        self.assertIs(expected_rhs.other, expected_tensor)

        actual_rhs = ReflectedPower(actual_marker)
        expected_rhs = ReflectedPower(expected_marker)
        self.assertIs(
            forwarded_operator(torch, actual_tensor, actual_rhs), actual_marker
        )
        self.assertIs(
            forwarded_operator(reference_torch, expected_tensor, expected_rhs),
            expected_marker,
        )
        self.assertIs(actual_rhs.other, actual_tensor)
        self.assertIs(expected_rhs.other, expected_tensor)

    def test_direct_dunder_call_shapes_match_pytorch_2_13(self):
        actual_tensor = torch.tensor([2.0, -3.0], dtype=torch.float32)
        expected_tensor = reference_torch.tensor(
            [2.0, -3.0], dtype=reference_torch.float32
        )

        self.assert_tensor_matches(
            actual_tensor.__pow__(exponent=2),
            expected_tensor.__pow__(exponent=2),
            case="direct keyword exponent",
        )
        self.assert_tensor_matches(
            pow(actual_tensor, 2, None),
            pow(expected_tensor, 2, None),
            case="builtin pow modulo none",
        )
        self.assertIs(actual_tensor.__pow__(2, None), NotImplemented)
        self.assertIs(expected_tensor.__pow__(2, None), NotImplemented)
        self.assertIs(actual_tensor.__pow__(2, modulo=None), NotImplemented)
        self.assertIs(expected_tensor.__pow__(2, modulo=None), NotImplemented)
        self.assertIs(actual_tensor.__pow__(exponent=2, modulo=None), NotImplemented)
        self.assertIs(expected_tensor.__pow__(exponent=2, modulo=None), NotImplemented)

    def test_operator_torch_function_dispatch_matches_pytorch_2_13(self):
        def mode_call(module, tensor, result, call):
            class RecordingMode(module.overrides.TorchFunctionMode):
                def __init__(self):
                    self.calls = []

                def __torch_function__(self, func, types, args=(), kwargs=None):
                    self.calls.append((func, types, args, kwargs))
                    return result

            mode = RecordingMode()
            with mode:
                output = call(tensor)
            return output, mode.calls[0]

        actual_tensor = torch.tensor([2.0])
        expected_tensor = reference_torch.tensor([2.0], dtype=reference_torch.float32)
        actual_marker = object()
        expected_marker = object()
        self.assertIs(
            mode_call(torch, actual_tensor, actual_marker, lambda value: value**2)[0],
            actual_marker,
        )
        self.assertIs(
            mode_call(
                reference_torch,
                expected_tensor,
                expected_marker,
                lambda value: value**2,
            )[0],
            expected_marker,
        )

        cases = (
            ("operator", lambda value: value**2, (2,), {}),
            (
                "dunder keyword",
                lambda value: value.__pow__(exponent=2),
                (),
                {"exponent": 2},
            ),
        )
        for form, pow_call, expected_tail, expected_kwargs in cases:
            with self.subTest(form=form):
                for module, tensor, (_, call) in (
                    (
                        torch,
                        actual_tensor,
                        mode_call(torch, actual_tensor, actual_marker, pow_call),
                    ),
                    (
                        reference_torch,
                        expected_tensor,
                        mode_call(
                            reference_torch, expected_tensor, expected_marker, pow_call
                        ),
                    ),
                ):
                    function, dispatch_types, args, kwargs = call
                    self.assertIs(
                        function, inspect.getattr_static(module.Tensor, "__pow__")
                    )
                    self.assertEqual(dispatch_types, (module.Tensor,))
                    self.assertEqual(len(args), 1 + len(expected_tail))
                    self.assertIs(args[0], tensor)
                    self.assertEqual(args[1:], expected_tail)
                    self.assertEqual(kwargs, expected_kwargs)

        for module, tensor, (_, call) in (
            (
                torch,
                actual_tensor,
                mode_call(torch, actual_tensor, actual_marker, lambda value: value.__pow__(2)),
            ),
            (
                reference_torch,
                expected_tensor,
                mode_call(
                    reference_torch,
                    expected_tensor,
                    expected_marker,
                    lambda value: value.__pow__(2),
                ),
            ),
        ):
            function, dispatch_types, args, kwargs = call
            self.assertIs(function, inspect.getattr_static(module.Tensor, "__pow__"))
            self.assertEqual(dispatch_types, (module.Tensor,))
            self.assertEqual(len(args), 2)
            self.assertIs(args[0], tensor)
            self.assertEqual(args[1], 2)
            self.assertEqual(kwargs, {})

        def override_call(module, tensor, result):
            events = []

            class Override:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append((func, types, args, kwargs))
                    return result

            rhs = Override()
            output = tensor**rhs
            return output, rhs, Override, events[0]

        actual = override_call(torch, actual_tensor, actual_marker)
        expected = override_call(reference_torch, expected_tensor, expected_marker)
        self.assertIs(actual[0], actual_marker)
        self.assertIs(expected[0], expected_marker)

        for module, tensor, (_, rhs, override_type, call) in (
            (torch, actual_tensor, actual),
            (reference_torch, expected_tensor, expected),
        ):
            function, dispatch_types, args, kwargs = call
            self.assertIs(function, inspect.getattr_static(module.Tensor, "__pow__"))
            self.assertEqual(dispatch_types, (module.Tensor, override_type))
            self.assertEqual(len(args), 2)
            self.assertIs(args[0], tensor)
            self.assertIs(args[1], rhs)
            self.assertEqual(kwargs, {})

    def test_positional_modulo_torch_function_dispatch_matches_pytorch_2_13(self):
        def override_call(module, tensor, result, call):
            events = []

            class ModuloOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append((func, types, args, kwargs))
                    return result

            modulo = ModuloOverride()
            output = call(tensor, modulo)
            return output, modulo, ModuloOverride, events[0]

        actual_tensor = torch.tensor([2.0])
        expected_tensor = reference_torch.tensor([2.0], dtype=reference_torch.float32)
        actual_marker = object()
        expected_marker = object()

        cases = (
            ("direct dunder", lambda tensor, modulo: tensor.__pow__(2, modulo)),
            ("builtin ternary", lambda tensor, modulo: pow(tensor, 2, modulo)),
        )
        for form, pow_call in cases:
            with self.subTest(form=form):
                actual = override_call(torch, actual_tensor, actual_marker, pow_call)
                expected = override_call(
                    reference_torch, expected_tensor, expected_marker, pow_call
                )
                self.assertIs(actual[0], actual_marker)
                self.assertIs(expected[0], expected_marker)

                for module, tensor, (_, modulo, override_type, call) in (
                    (torch, actual_tensor, actual),
                    (reference_torch, expected_tensor, expected),
                ):
                    function, dispatch_types, args, kwargs = call
                    self.assertIs(
                        function, inspect.getattr_static(module.Tensor, "__pow__")
                    )
                    self.assertEqual(dispatch_types, (module.Tensor, override_type))
                    self.assertEqual(len(args), 3)
                    self.assertIs(args[0], tensor)
                    self.assertEqual(args[1], 2)
                    self.assertIs(args[2], modulo)
                    self.assertEqual(kwargs, {})


if __name__ == "__main__":
    unittest.main()
