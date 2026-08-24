import importlib
import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional


class FunctionalL1LossTests(unittest.TestCase):
    @staticmethod
    def float32_from_bits(bits):
        return np.array([bits], dtype=np.uint32).view(np.float32)[0].item()

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

    def assert_tensor_unchanged(self, tensor, before):
        self.assertEqual(self.tensor_state(tensor)[:-1], before[:-1])
        np.testing.assert_array_equal(self.tensor_state(tensor)[-1], before[-1])

    def assert_matches_absolute_difference(
        self,
        actual,
        difference,
        *,
        case,
        expected_stride,
    ):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, difference.shape)
            self.assertEqual(actual.stride(), expected_stride)
            self.assertEqual(actual.storage_offset(), 0)
            self.assertEqual(actual.requires_grad, difference.requires_grad)
            self.assertEqual(actual.is_leaf, difference.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            expected_bits = self.tensor_bits(difference) & np.uint32(0x7FFF_FFFF)
            np.testing.assert_array_equal(self.tensor_bits(actual), expected_bits)

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
            ("scalar", torch.tensor(-0.0), torch.tensor(2.5), ()),
            ("empty", empty_input, empty_target, (2, 2, 1)),
            ("offset", offset_input_base[1], offset_target_base[0], (12, 4, 1)),
            (
                "matching noncontiguous",
                noncontiguous_input,
                noncontiguous_target,
                (12, 1, 3),
            ),
            (
                "mixed noncontiguous",
                noncontiguous_input,
                mixed_layout_target,
                (12, 1, 3),
            ),
            (
                "offset noncontiguous",
                offset_strided_input,
                offset_strided_target,
                (12, 1, 3),
            ),
            (
                "channels last",
                channels_last_input,
                channels_last_target,
                (24, 1, 8, 2),
            ),
            (
                "mixed singleton strides",
                mixed_singleton_input,
                mixed_singleton_target,
                (3, 3, 1),
            ),
            ("same operand", same, same, (2, 1)),
        )

    @staticmethod
    def call(input, target, form):
        if form == "reduction keyword":
            return functional.l1_loss(input, target, reduction="none")
        if form == "legacy none keywords":
            return functional.l1_loss(
                input=input,
                target=target,
                size_average=None,
                reduce=None,
                reduction="none",
                weight=None,
            )
        if form == "five positional":
            return functional.l1_loss(input, target, None, None, "none")
        return functional.l1_loss(input, target, None, None, "none", None)

    def test_import_signature_documentation_and_exports(self):
        imported = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn.functional import l1_loss

        self.assertIs(imported, functional)
        self.assertIs(l1_loss, functional.l1_loss)
        self.assertIs(type(l1_loss), types.FunctionType)
        self.assertEqual(l1_loss.__name__, "l1_loss")
        self.assertEqual(l1_loss.__qualname__, "l1_loss")
        self.assertEqual(l1_loss.__module__, "torch_rs.nn.functional")
        self.assertEqual(l1_loss.__defaults__, (None, None, "mean", None))
        self.assertIsNone(l1_loss.__kwdefaults__)
        self.assertFalse(hasattr(l1_loss, "__text_signature__"))
        self.assertTrue(
            l1_loss.__doc__.startswith(
                "\nl1_loss(input, target, size_average=None, reduce=None, "
                "reduction='mean', weight=None)"
            )
        )
        normalized_doc = " ".join(l1_loss.__doc__.split())
        for documented_limit in (
            "exact, same-shaped",
            "CPU ``float32`` storage",
            "``reduction='none'``",
            "``size_average=None``",
            "``reduce=None``",
            "``weight=None``",
            "native subtraction and absolute-value kernels",
            "fresh, independent tensor",
            "Broadcasting",
            "Tensor subclasses",
            "active ``TorchFunctionMode`` contexts",
            "active autograd recording",
            "inside ``torch.no_grad()``",
        ):
            self.assertIn(documented_limit, normalized_doc)

        signature = inspect.signature(l1_loss)
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
        self.assertFalse(hasattr(torch, "_nn_functional_l1_loss"))

        wildcard = {}
        exec("from torch_rs.nn.functional import *", wildcard)
        self.assertIs(wildcard["l1_loss"], l1_loss)

    def test_supported_forms_reuse_subtraction_and_absolute_value(self):
        for case, input, target, expected_stride in self.layout_cases():
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)
            for form in (
                "reduction keyword",
                "legacy none keywords",
                "five positional",
                "six positional",
            ):
                with torch.no_grad():
                    difference = input - target
                actual = self.call(input, target, form)
                self.assert_matches_absolute_difference(
                    actual,
                    difference,
                    case=(case, form),
                    expected_stride=expected_stride,
                )
                with self.subTest(case=(case, form), nonmutation=True):
                    self.assert_tensor_unchanged(input, input_state)
                    self.assert_tensor_unchanged(target, target_state)

    def test_every_call_returns_fresh_independent_storage(self):
        for case, input, target, _ in self.layout_cases():
            first = functional.l1_loss(input, target, reduction="none")
            second = functional.l1_loss(input, target, reduction="none")
            with self.subTest(case=case):
                self.assertIsNot(first, second)
                self.assertFalse(first.is_set_to(second))
                self.assertFalse(first.is_set_to(input))
                self.assertFalse(first.is_set_to(target))
                if first.numel() != 0:
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())
                    self.assertNotEqual(first.data_ptr(), input.data_ptr())
                    self.assertNotEqual(first.data_ptr(), target.data_ptr())

    def test_float32_zero_infinity_and_nan_bits_match_the_composition(self):
        input = torch.tensor(
            [
                0.0,
                -0.0,
                float("inf"),
                -float("inf"),
                float("inf"),
                -float("inf"),
                float("nan"),
                -float("nan"),
                np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
            ]
        )
        target = torch.tensor(
            [
                -0.0,
                0.0,
                float("inf"),
                -float("inf"),
                -float("inf"),
                float("inf"),
                0.0,
                0.0,
                -np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ]
        )
        difference = input - target
        actual = functional.l1_loss(input, target, reduction="none")
        self.assert_matches_absolute_difference(
            actual,
            difference,
            case="float32 edges",
            expected_stride=(1,),
        )

    def test_paired_nan_uses_the_target_payload_before_absolute_value(self):
        input_bits = [0x7FC1_1111, 0xFFC2_2222]
        target_bits = [0xFFC2_2222, 0x7FC1_1111]
        input = torch.tensor(
            [self.float32_from_bits(bits) for bits in input_bits]
        )
        target = torch.tensor(
            [self.float32_from_bits(bits) for bits in target_bits]
        )

        np.testing.assert_array_equal(self.tensor_bits(input), input_bits)
        np.testing.assert_array_equal(self.tensor_bits(target), target_bits)
        difference = input - target
        np.testing.assert_array_equal(self.tensor_bits(difference), target_bits)

        actual = functional.l1_loss(input, target, reduction="none")
        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            [0x7FC2_2222, 0x7FC1_1111],
        )

    def test_requires_grad_operands_are_rejected_before_work_and_work_under_no_grad(self):
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
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)
            with self.subTest(
                input_requires_grad=input_requires_grad,
                target_requires_grad=target_requires_grad,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^l1_loss\(\): autograd recording is not supported$",
                ):
                    functional.l1_loss(input, target, reduction="none")
                self.assert_tensor_unchanged(input, input_state)
                self.assert_tensor_unchanged(target, target_state)
                self.assertIsNone(input.grad)
                self.assertIsNone(target.grad)

                with torch.no_grad():
                    difference = input - target
                    actual = functional.l1_loss(input, target, reduction="none")
                self.assert_matches_absolute_difference(
                    actual,
                    difference,
                    case="no_grad",
                    expected_stride=difference.stride(),
                )
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)

    def test_unsupported_options_shapes_and_operands_are_rejected_before_work(self):
        input = torch.ones((2, 3), requires_grad=True)
        target = torch.zeros((2, 3))
        input_state = self.tensor_state(input)
        target_state = self.tensor_state(target)

        reduction_error = (
            "torch_rs.nn.functional.l1_loss only supports reduction='none'"
        )
        for reduction in ("mean", "sum", "batchmean", None, 1, object()):
            with self.subTest(reduction=reduction):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    f"^{re.escape(reduction_error)}$",
                ):
                    functional.l1_loss(input, target, reduction=reduction)

        legacy_error = (
            "torch_rs.nn.functional.l1_loss only supports "
            "size_average=None and reduce=None"
        )
        for legacy_arguments in (
            {"size_average": False},
            {"size_average": True},
            {"reduce": False},
            {"reduce": True},
            {"size_average": False, "reduce": False},
        ):
            with self.subTest(legacy_arguments=legacy_arguments):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    f"^{re.escape(legacy_error)}$",
                ):
                    functional.l1_loss(
                        input,
                        target,
                        reduction="none",
                        **legacy_arguments,
                    )

        weight_error = "torch_rs.nn.functional.l1_loss only supports weight=None"
        for weight in (torch.ones((2, 3)), 1.0, [1.0, 1.0]):
            with self.subTest(weight=type(weight)):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    f"^{re.escape(weight_error)}$",
                ):
                    functional.l1_loss(
                        input,
                        target,
                        reduction="none",
                        weight=weight,
                    )

        broadcast_error = "torch_rs.nn.functional.l1_loss does not support broadcasting"
        for other in (
            torch.zeros((3,)),
            torch.zeros((2, 1)),
            torch.zeros(()),
            torch.zeros((2, 2)),
        ):
            other_state = self.tensor_state(other)
            with self.subTest(target_shape=other.shape):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    f"^{re.escape(broadcast_error)}$",
                ):
                    functional.l1_loss(input, other, reduction="none")
                self.assert_tensor_unchanged(other, other_state)

        class TensorSubclassOverride:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return object()

        exact_tensor_error = (
            "l1_loss() only supports exact native Tensor input and target operands"
        )
        for actual_input, actual_target in (
            (TensorSubclassOverride(), target),
            (input, TensorSubclassOverride()),
            (1.0, target),
            (input, [0.0]),
        ):
            with self.subTest(
                input_type=type(actual_input),
                target_type=type(actual_target),
            ):
                with self.assertRaisesRegex(
                    TypeError,
                    f"^{re.escape(exact_tensor_error)}$",
                ):
                    functional.l1_loss(
                        actual_input,
                        actual_target,
                        reduction="none",
                    )
        self.assertEqual(TensorSubclassOverride.calls, 0)
        self.assert_tensor_unchanged(input, input_state)
        self.assert_tensor_unchanged(target, target_state)

    def test_active_torch_function_mode_is_rejected_without_dispatch_or_mutation(self):
        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return object()

        input = torch.ones((2, 3), requires_grad=True)
        target = torch.zeros((3,))
        input_state = self.tensor_state(input)
        target_state = self.tensor_state(target)
        mode = RecordingMode()
        with self.assertRaisesRegex(
            TypeError,
            r"^l1_loss\(\) does not support an active TorchFunctionMode$",
        ):
            with mode:
                functional.l1_loss(
                    input,
                    target,
                    reduction="mean",
                    weight=object(),
                )
        self.assertEqual(mode.calls, 0)
        self.assert_tensor_unchanged(input, input_state)
        self.assert_tensor_unchanged(target, target_state)

    def test_python_argument_binding_matches_the_canonical_signature(self):
        input = torch.ones((1,))
        target = torch.zeros((1,))
        cases = (
            (
                lambda: functional.l1_loss(),
                "l1_loss() missing 2 required positional arguments: 'input' and 'target'",
            ),
            (
                lambda: functional.l1_loss(input),
                "l1_loss() missing 1 required positional argument: 'target'",
            ),
            (
                lambda: functional.l1_loss(input, target, input=input),
                "l1_loss() got multiple values for argument 'input'",
            ),
            (
                lambda: functional.l1_loss(
                    input, target, None, None, "none", None, None
                ),
                "l1_loss() takes from 2 to 6 positional arguments but 7 were given",
            ),
            (
                lambda: functional.l1_loss(
                    input, target, reduction="none", unexpected=True
                ),
                "l1_loss() got an unexpected keyword argument 'unexpected'",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
