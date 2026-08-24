import importlib
import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional


class FunctionalMseLossTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor).reshape(-1).view(np.uint32)

    @classmethod
    def tensor_state(cls, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            cls.tensor_bits(tensor).copy(),
        )

    def assert_matches_composition(
        self,
        actual,
        expected,
        *,
        case,
        expected_stride=None,
    ):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(
                actual.stride(),
                expected.stride() if expected_stride is None else expected_stride,
            )
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                self.tensor_bits(actual),
                self.tensor_bits(expected),
            )

    def layout_cases(self):
        offset_input_base = torch.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4).tolist()
        )
        offset_target_base = torch.tensor(
            np.linspace(-3.0, 4.0, 48, dtype=np.float32)
            .reshape(2, 2, 3, 4)
            .tolist()
        )
        noncontiguous_input = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist()
        ).transpose(1, 2)
        noncontiguous_target = torch.tensor(
            np.linspace(-2.0, 2.0, 24, dtype=np.float32)
            .reshape(2, 4, 3)
            .tolist()
        ).transpose(1, 2)
        mixed_layout_target = torch.tensor(
            np.linspace(3.0, -3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        offset_strided_input = torch.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 4, 3).tolist()
        )[1].transpose(1, 2)
        offset_strided_target = torch.tensor(
            np.linspace(5.0, -5.0, 48, dtype=np.float32)
            .reshape(2, 2, 4, 3)
            .tolist()
        )[1].transpose(1, 2)
        channels_last_input = offset_input_base.contiguous(
            memory_format=torch.channels_last
        )
        channels_last_target = offset_target_base.contiguous(
            memory_format=torch.channels_last
        )
        empty_input = torch.zeros((2, 0, 3)).transpose(0, 2)
        empty_target = torch.ones((2, 0, 3)).transpose(0, 2)
        mixed_singleton_input = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 1, 3).tolist()
        )
        mixed_singleton_target = torch.tensor(
            np.linspace(-1.0, 1.0, 6, dtype=np.float32)
            .reshape(3, 1, 2)
            .tolist()
        ).permute(2, 1, 0)
        same = torch.tensor([[1.0, -2.0], [3.0, -4.0]])

        return (
            ("scalar", torch.tensor(-0.0), torch.tensor(2.5)),
            ("empty", empty_input, empty_target),
            ("offset", offset_input_base[1], offset_target_base[0]),
            (
                "matching noncontiguous",
                noncontiguous_input,
                noncontiguous_target,
            ),
            ("mixed noncontiguous", noncontiguous_input, mixed_layout_target),
            (
                "offset noncontiguous",
                offset_strided_input,
                offset_strided_target,
            ),
            ("channels last", channels_last_input, channels_last_target),
            (
                "mixed singleton strides",
                mixed_singleton_input,
                mixed_singleton_target,
            ),
            ("same operand", same, same),
        )

    def sum_cases(self):
        selected_layouts = {
            name: (input, target)
            for name, input, target in self.layout_cases()
            if name in {"scalar", "empty", "offset", "matching noncontiguous"}
        }
        return (
            ("scalar", *selected_layouts["scalar"]),
            ("empty", *selected_layouts["empty"]),
            ("offset", *selected_layouts["offset"]),
            ("noncontiguous", *selected_layouts["matching noncontiguous"]),
            (
                "nan",
                torch.tensor([1.0, float("nan"), 3.0]),
                torch.zeros((3,)),
            ),
            (
                "square overflow",
                torch.tensor([np.finfo(np.float32).max]),
                torch.zeros((1,)),
            ),
            (
                "reduction overflow",
                torch.tensor([1.0e19, 1.0e19, 1.0e19, 1.0e19]),
                torch.zeros((4,)),
            ),
        )

    @staticmethod
    def call(input, target, form, reduction="none"):
        if form == "reduction keyword":
            return functional.mse_loss(input, target, reduction=reduction)
        if form == "legacy none keywords":
            return functional.mse_loss(
                input=input,
                target=target,
                size_average=None,
                reduce=None,
                reduction=reduction,
                weight=None,
            )
        if form == "five positional":
            return functional.mse_loss(input, target, None, None, reduction)
        return functional.mse_loss(input, target, None, None, reduction, None)

    def test_import_signature_documentation_and_exports(self):
        imported = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn.functional import mse_loss

        self.assertIs(imported, functional)
        self.assertIs(mse_loss, functional.mse_loss)
        self.assertIs(type(mse_loss), types.FunctionType)
        self.assertEqual(mse_loss.__name__, "mse_loss")
        self.assertEqual(mse_loss.__qualname__, "mse_loss")
        self.assertEqual(mse_loss.__module__, "torch_rs.nn.functional")
        self.assertEqual(mse_loss.__defaults__, (None, None, "mean", None))
        self.assertIsNone(mse_loss.__kwdefaults__)
        self.assertFalse(hasattr(mse_loss, "__text_signature__"))
        self.assertTrue(
            mse_loss.__doc__.startswith(
                "\nmse_loss(input, target, size_average=None, reduce=None, "
                "reduction='mean', weight=None)"
            )
        )
        normalized_doc = " ".join(mse_loss.__doc__.split())
        for documented_limit in (
            "exact, same-shaped",
            "CPU ``float32`` storage",
            "``reduction='none'`` or ``reduction='sum'``",
            "``size_average=None``",
            "``reduce=None``",
            "``weight=None``",
            "native subtraction and stride-preserving square kernels",
            "native full reduction",
            "fresh scalar tensor",
            "fresh, independent tensor",
            "Broadcasting",
            "Tensor subclasses",
            "active ``TorchFunctionMode`` contexts",
            "active autograd recording",
            "inside ``torch.no_grad()``",
        ):
            self.assertIn(documented_limit, normalized_doc)

        signature = inspect.signature(mse_loss)
        self.assertEqual(
            tuple(signature.parameters),
            (
                "input",
                "target",
                "size_average",
                "reduce",
                "reduction",
                "weight",
            ),
        )
        self.assertIs(signature.parameters["input"].annotation, torch.Tensor)
        self.assertIs(signature.parameters["target"].annotation, torch.Tensor)
        self.assertEqual(signature.parameters["size_average"].default, None)
        self.assertEqual(signature.parameters["reduce"].default, None)
        self.assertEqual(signature.parameters["reduction"].default, "mean")
        self.assertEqual(signature.parameters["weight"].default, None)
        self.assertIs(signature.return_annotation, torch.Tensor)
        self.assertFalse(hasattr(torch, "_nn_functional_mse_loss"))

        wildcard = {}
        exec("from torch_rs.nn.functional import *", wildcard)
        self.assertIs(wildcard["mse_loss"], mse_loss)

    def test_supported_forms_compose_subtraction_and_square(self):
        for case, input, target in self.layout_cases():
            difference = input - target
            expected = difference.square()
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)
            for form in (
                "reduction keyword",
                "legacy none keywords",
                "five positional",
                "six positional",
            ):
                actual = self.call(input, target, form)
                self.assert_matches_composition(
                    actual,
                    expected,
                    case=(case, form),
                    expected_stride=difference.stride(),
                )
                with self.subTest(case=(case, form), nonmutation=True):
                    self.assertEqual(
                        self.tensor_state(input)[:-1], input_state[:-1]
                    )
                    self.assertEqual(
                        self.tensor_state(target)[:-1], target_state[:-1]
                    )
                    np.testing.assert_array_equal(
                        self.tensor_state(input)[-1], input_state[-1]
                    )
                    np.testing.assert_array_equal(
                        self.tensor_state(target)[-1], target_state[-1]
                    )

    def test_mixed_layout_singleton_keeps_binary_tensoriterator_stride(self):
        input = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 1, 3).tolist()
        )
        target = torch.tensor(
            np.linspace(-1.0, 1.0, 6, dtype=np.float32)
            .reshape(3, 1, 2)
            .tolist()
        ).permute(2, 1, 0)

        self.assertEqual(input.stride(), (3, 3, 1))
        self.assertEqual(target.stride(), (1, 2, 2))
        difference = input - target
        self.assertEqual(difference.stride(), (3, 6, 1))
        self.assertEqual(difference.square().stride(), (3, 3, 1))

        actual = functional.mse_loss(input, target, reduction="none")
        self.assertEqual(actual.stride(), (3, 6, 1))
        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.tensor_bits(difference.square()),
        )

    def test_sum_reduction_composes_subtraction_square_and_full_reduction(self):
        for case, input, target in self.sum_cases():
            with torch.no_grad():
                expected = (input - target).square().sum()
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)
            for form in (
                "reduction keyword",
                "legacy none keywords",
                "five positional",
                "six positional",
            ):
                actual = self.call(input, target, form, reduction="sum")
                self.assert_matches_composition(
                    actual,
                    expected,
                    case=(case, form),
                )
                with self.subTest(case=(case, form), scalar=True):
                    self.assertEqual(actual.shape, torch.Size([]))
                    self.assertEqual(actual.stride(), ())
                    self.assertEqual(actual.storage_offset(), 0)
                with self.subTest(case=(case, form), nonmutation=True):
                    self.assertEqual(
                        self.tensor_state(input)[:-1], input_state[:-1]
                    )
                    self.assertEqual(
                        self.tensor_state(target)[:-1], target_state[:-1]
                    )
                    np.testing.assert_array_equal(
                        self.tensor_state(input)[-1], input_state[-1]
                    )
                    np.testing.assert_array_equal(
                        self.tensor_state(target)[-1], target_state[-1]
                    )

    def test_sum_preserves_small_terms_after_a_large_squared_error(self):
        input = torch.tensor([4096.0] + [1.0] * 1000)
        target = torch.zeros((1001,))

        actual = functional.mse_loss(input, target, reduction="sum")

        self.assertEqual(actual.item(), 16_778_196.0)
        self.assertNotEqual(actual.item(), 16_777_216.0)

    def test_every_call_returns_fresh_independent_storage(self):
        for case, input, target in self.layout_cases():
            for reduction in ("none", "sum"):
                first = functional.mse_loss(input, target, reduction=reduction)
                second = functional.mse_loss(input, target, reduction=reduction)
                with self.subTest(case=case, reduction=reduction):
                    self.assertIsNot(first, second)
                    self.assertFalse(first.is_set_to(second))
                    self.assertFalse(first.is_set_to(input))
                    self.assertFalse(first.is_set_to(target))
                    if first.numel() != 0:
                        self.assertNotEqual(first.data_ptr(), second.data_ptr())
                        self.assertNotEqual(first.data_ptr(), input.data_ptr())
                        self.assertNotEqual(first.data_ptr(), target.data_ptr())

    def test_float32_edge_values_match_kernel_composition_bits(self):
        input = torch.tensor(
            [
                -0.0,
                0.0,
                1.0e-20,
                -1.0e-20,
                1.0,
                -1.0,
                1.0e10,
                -1.0e10,
                np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
            ]
        )
        target = torch.tensor(
            [0.0, -0.0, 0.0, 0.0, -1.0, 1.0, -1.0e10, 1.0e10, 0.0, 0.0]
        )
        actual = functional.mse_loss(input, target, reduction="none")
        expected = (input - target).square()
        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.tensor_bits(expected),
        )

    def test_requires_grad_operands_need_no_grad(self):
        for input_requires_grad, target_requires_grad in (
            (True, False),
            (False, True),
            (True, True),
        ):
            input = torch.tensor(
                [[1.0, -2.0], [3.0, -4.0]],
                requires_grad=input_requires_grad,
            )
            target = torch.tensor(
                [[0.5, 2.0], [-3.0, 4.5]],
                requires_grad=target_requires_grad,
            )
            for reduction in ("none", "sum"):
                with self.subTest(
                    input_requires_grad=input_requires_grad,
                    target_requires_grad=target_requires_grad,
                    reduction=reduction,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"^mse_loss\(\): autograd recording is not supported$",
                    ):
                        functional.mse_loss(input, target, reduction=reduction)

                    with torch.no_grad():
                        actual = functional.mse_loss(
                            input,
                            target,
                            reduction=reduction,
                        )
                        difference = input - target
                        expected = difference.square()
                        if reduction == "sum":
                            expected = expected.sum()
                    self.assert_matches_composition(
                        actual,
                        expected,
                        case="no_grad",
                        expected_stride=(
                            difference.stride() if reduction == "none" else None
                        ),
                    )
                    self.assertFalse(actual.requires_grad)
                    self.assertTrue(actual.is_leaf)
                    self.assertIsNone(input.grad)
                    self.assertIsNone(target.grad)

    def test_unsupported_options_shapes_and_operands_are_rejected(self):
        input = torch.ones((2, 3))
        target = torch.zeros((2, 3))

        reduction_error = (
            "torch_rs.nn.functional.mse_loss only supports "
            "reduction='none' or reduction='sum'"
        )
        for reduction in ("mean", "batchmean", None, 1, object()):
            with self.subTest(reduction=reduction):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    f"^{re.escape(reduction_error)}$",
                ):
                    functional.mse_loss(input, target, reduction=reduction)
        with self.assertRaisesRegex(
            NotImplementedError,
            f"^{re.escape(reduction_error)}$",
        ):
            functional.mse_loss(object(), object(), reduction="mean")

        legacy_error = (
            "torch_rs.nn.functional.mse_loss only supports "
            "size_average=None and reduce=None"
        )
        for reduction in ("none", "sum"):
            for legacy_arguments in (
                {"size_average": False},
                {"size_average": True},
                {"reduce": False},
                {"reduce": True},
                {"size_average": False, "reduce": False},
            ):
                with self.subTest(
                    reduction=reduction,
                    legacy_arguments=legacy_arguments,
                ):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        f"^{re.escape(legacy_error)}$",
                    ):
                        functional.mse_loss(
                            input,
                            target,
                            reduction=reduction,
                            **legacy_arguments,
                        )

        weight_error = "torch_rs.nn.functional.mse_loss only supports weight=None"
        for reduction in ("none", "sum"):
            for weight in (torch.ones((2, 3)), 1.0, [1.0, 1.0]):
                with self.subTest(reduction=reduction, weight=type(weight)):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        f"^{re.escape(weight_error)}$",
                    ):
                        functional.mse_loss(
                            input,
                            target,
                            reduction=reduction,
                            weight=weight,
                        )

        broadcast_error = (
            "torch_rs.nn.functional.mse_loss does not support broadcasting"
        )
        for reduction in ("none", "sum"):
            for other in (
                torch.zeros((3,)),
                torch.zeros((2, 1)),
                torch.zeros(()),
                torch.zeros((2, 2)),
            ):
                with self.subTest(
                    reduction=reduction,
                    target_shape=other.shape,
                ):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        f"^{re.escape(broadcast_error)}$",
                    ):
                        functional.mse_loss(input, other, reduction=reduction)

        class Override:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return object()

        exact_tensor_error = (
            "mse_loss() only supports exact native Tensor input and target operands"
        )
        for reduction in ("none", "sum"):
            for actual_input, actual_target in (
                (Override(), target),
                (input, Override()),
                (1.0, target),
                (input, [0.0]),
            ):
                with self.subTest(
                    reduction=reduction,
                    input_type=type(actual_input),
                    target_type=type(actual_target),
                ):
                    with self.assertRaisesRegex(
                        TypeError,
                        f"^{re.escape(exact_tensor_error)}$",
                    ):
                        functional.mse_loss(
                            actual_input,
                            actual_target,
                            reduction=reduction,
                        )
        self.assertEqual(Override.calls, 0)

    def test_active_torch_function_mode_is_rejected_without_dispatch(self):
        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return object()

        mode = RecordingMode()
        for target, reduction, weight in (
            (torch.zeros((2, 3)), "sum", None),
            (torch.zeros((3,)), "mean", object()),
        ):
            with self.subTest(reduction=reduction):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^mse_loss\(\) does not support an active TorchFunctionMode$",
                ):
                    with mode:
                        functional.mse_loss(
                            torch.ones((2, 3), requires_grad=True),
                            target,
                            reduction=reduction,
                            weight=weight,
                        )
        self.assertEqual(mode.calls, 0)

    def test_python_argument_binding_matches_the_canonical_signature(self):
        input = torch.ones((1,))
        target = torch.zeros((1,))
        cases = (
            (
                lambda: functional.mse_loss(),
                "mse_loss() missing 2 required positional arguments: 'input' and 'target'",
            ),
            (
                lambda: functional.mse_loss(input),
                "mse_loss() missing 1 required positional argument: 'target'",
            ),
            (
                lambda: functional.mse_loss(input, target, input=input),
                "mse_loss() got multiple values for argument 'input'",
            ),
            (
                lambda: functional.mse_loss(
                    input, target, None, None, "none", None, None
                ),
                "mse_loss() takes from 2 to 6 positional arguments but 7 were given",
            ),
            (
                lambda: functional.mse_loss(
                    input, target, reduction="none", unexpected=True
                ),
                "mse_loss() got an unexpected keyword argument 'unexpected'",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
