import importlib
import inspect
import re
import types
import unittest
import warnings

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

    def scalar_broadcast_cases(self):
        scalar = torch.tensor([99.0, 1.25])[1]
        contiguous = torch.tensor(
            np.linspace(-3.0, 4.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        offset_strided = torch.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 4, 3).tolist()
        )[1].transpose(1, 2)
        channels_last = torch.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4).tolist()
        ).contiguous(memory_format=torch.channels_last)
        empty_strided = torch.zeros((2, 0, 3)).transpose(0, 2)

        return (
            ("scalar input contiguous", scalar, contiguous),
            ("scalar target contiguous", contiguous, scalar),
            ("scalar input offset strided", scalar, offset_strided),
            ("scalar target offset strided", offset_strided, scalar),
            ("scalar input channels last", scalar, channels_last),
            ("scalar target channels last", channels_last, scalar),
            ("scalar input empty strided", scalar, empty_strided),
            ("scalar target empty strided", empty_strided, scalar),
        )

    @staticmethod
    def broadcast_warning(input, target):
        return (
            f"Using a target size ({target.size()}) that is different to the "
            f"input size ({input.size()}). This will likely lead to incorrect "
            "results due to broadcasting. Please ensure they have the same size."
        )

    @staticmethod
    def call(input, target, form):
        if form == "reduction keyword":
            return functional.mse_loss(input, target, reduction="none")
        if form == "legacy none keywords":
            return functional.mse_loss(
                input=input,
                target=target,
                size_average=None,
                reduce=None,
                reduction="none",
                weight=None,
            )
        if form == "five positional":
            return functional.mse_loss(input, target, None, None, "none")
        return functional.mse_loss(input, target, None, None, "none", None)

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
            "exact ``torch_rs.Tensor`` operands",
            "CPU ``float32`` storage",
            "exactly one may be a rank-0 scalar",
            "scalar broadcasting emits PyTorch's size mismatch warning",
            "``reduction='none'``",
            "``size_average=None``",
            "``reduce=None``",
            "``weight=None``",
            "fuses subtraction and square into one native pass",
            "fresh, independent tensor",
            "Other broadcasting",
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

    def test_supported_forms_match_subtraction_and_square_composition(self):
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

    def test_scalar_broadcast_forms_layouts_warnings_and_storage(self):
        for case, input, target in self.scalar_broadcast_cases():
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
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    actual = self.call(input, target, form)
                    repeat = self.call(input, target, form)
                with self.subTest(case=(case, form), warnings=True):
                    self.assertEqual(len(caught), 2)
                    self.assertTrue(
                        all(item.category is UserWarning for item in caught)
                    )
                    self.assertEqual(
                        [str(item.message) for item in caught],
                        [self.broadcast_warning(input, target)] * 2,
                    )
                self.assert_matches_composition(
                    actual,
                    expected,
                    case=(case, form),
                    expected_stride=(
                        expected.stride()
                        if expected.numel() == 0
                        else difference.stride()
                    ),
                )
                with self.subTest(case=(case, form), storage=True):
                    self.assertIsNot(actual, repeat)
                    self.assertFalse(actual.is_set_to(repeat))
                    self.assertFalse(actual.is_set_to(input))
                    self.assertFalse(actual.is_set_to(target))
                    if actual.numel() != 0:
                        self.assertNotEqual(actual.data_ptr(), repeat.data_ptr())
                        self.assertNotEqual(actual.data_ptr(), input.data_ptr())
                        self.assertNotEqual(actual.data_ptr(), target.data_ptr())
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

    def test_scalar_broadcast_warning_points_to_caller_and_can_raise(self):
        input = torch.tensor(1.0)
        target = torch.zeros((2, 3))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warning_line = inspect.currentframe().f_lineno + 1
            functional.mse_loss(input, target, reduction="none")

        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, UserWarning)
        self.assertEqual(str(caught[0].message), self.broadcast_warning(input, target))
        self.assertEqual(caught[0].filename, __file__)
        self.assertEqual(caught[0].lineno, warning_line)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaisesRegex(
                UserWarning,
                f"^{re.escape(self.broadcast_warning(input, target))}$",
            ):
                functional.mse_loss(input, target, reduction="none")

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

    def test_every_call_returns_fresh_independent_storage(self):
        for case, input, target in self.layout_cases():
            first = functional.mse_loss(input, target, reduction="none")
            second = functional.mse_loss(input, target, reduction="none")
            with self.subTest(case=case):
                self.assertIsNot(first, second)
                self.assertFalse(first.is_set_to(second))
                self.assertFalse(first.is_set_to(input))
                self.assertFalse(first.is_set_to(target))
                if first.numel() != 0:
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())
                    self.assertNotEqual(first.data_ptr(), input.data_ptr())
                    self.assertNotEqual(first.data_ptr(), target.data_ptr())

    def test_float32_edge_values_match_kernel_composition_bits(self):
        input_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7F81_2345,
                0xFF85_4321,
            ],
            dtype=np.uint32,
        )
        target_bits = np.asarray(
            [
                0x8000_0000,
                0x0000_0000,
                0x8000_0001,
                0x0000_0001,
                0x807F_FFFF,
                0x007F_FFFF,
                0x8080_0000,
                0x0080_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xFF7F_FFFF,
                0x7F7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0xFFC6_789A,
                0x7FC2_ABCD,
                0xFF86_789A,
                0x7F82_ABCD,
            ],
            dtype=np.uint32,
        )
        input = torch.tensor(memoryview(input_bits.view(np.float32))).view(3, 6)
        target = torch.tensor(memoryview(target_bits.view(np.float32))).view(3, 6)

        for case, actual_input, actual_target in (
            ("contiguous", input, target),
            ("transposed", input.transpose(0, 1), target.transpose(0, 1)),
        ):
            difference = actual_input - actual_target
            actual = functional.mse_loss(
                actual_input,
                actual_target,
                reduction="none",
            )
            expected = difference.square()
            with self.subTest(case=case):
                self.assertEqual(actual.stride(), difference.stride())
                np.testing.assert_array_equal(
                    self.tensor_bits(actual),
                    self.tensor_bits(expected),
                )

    def test_scalar_broadcast_nan_payloads_match_pytorch_2_13_bits(self):
        values = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7F81_2345,
                0xFF85_4321,
            ],
            dtype=np.uint32,
        )
        tensor = torch.tensor(memoryview(values.view(np.float32))).view(3, 6)
        tensor = tensor.transpose(0, 1)
        scalar_bits = 0x7F86_789A
        scalar_values = np.asarray([0, scalar_bits], dtype=np.uint32)
        scalar = torch.tensor(memoryview(scalar_values.view(np.float32)))[1]
        quiet_scalar = scalar_bits | 0x0040_0000
        logical_bits = values.reshape(3, 6).transpose().reshape(-1)
        scalar_left_bits = np.full(logical_bits.size, quiet_scalar, dtype=np.uint32)
        nan_bits = (logical_bits & 0x7F80_0000) == 0x7F80_0000
        nan_bits &= (logical_bits & 0x007F_FFFF) != 0
        scalar_right_bits = np.where(
            nan_bits,
            logical_bits | 0x0040_0000,
            quiet_scalar,
        ).astype(np.uint32)

        for case, input, target, expected_bits in (
            ("scalar input", scalar, tensor, scalar_left_bits),
            ("scalar target", tensor, scalar, scalar_right_bits),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                actual = functional.mse_loss(input, target, reduction="none")
            with self.subTest(case=case):
                self.assertEqual(actual.shape, tensor.shape)
                self.assertEqual(actual.stride(), tensor.stride())
                np.testing.assert_array_equal(
                    self.tensor_bits(actual),
                    expected_bits,
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
            with self.subTest(
                input_requires_grad=input_requires_grad,
                target_requires_grad=target_requires_grad,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^mse_loss\(\): autograd recording is not supported$",
                ):
                    functional.mse_loss(input, target, reduction="none")

                with torch.no_grad():
                    actual = functional.mse_loss(input, target, reduction="none")
                    difference = input - target
                    expected = difference.square()
                self.assert_matches_composition(
                    actual,
                    expected,
                    case="no_grad",
                    expected_stride=difference.stride(),
                )
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                self.assertIsNone(input.grad)
                self.assertIsNone(target.grad)

        scalar = torch.tensor(1.0, requires_grad=True)
        tensor = torch.zeros((2, 3))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(
                RuntimeError,
                r"^mse_loss\(\): autograd recording is not supported$",
            ):
                functional.mse_loss(scalar, tensor, reduction="none")
        self.assertEqual(len(caught), 1)
        self.assertEqual(
            str(caught[0].message),
            self.broadcast_warning(scalar, tensor),
        )

        with warnings.catch_warnings(record=True) as caught, torch.no_grad():
            warnings.simplefilter("always")
            actual = functional.mse_loss(scalar, tensor, reduction="none")
        self.assertEqual(len(caught), 1)
        self.assertEqual(actual.shape, tensor.shape)
        self.assertEqual(actual.stride(), tensor.stride())
        self.assertFalse(actual.requires_grad)
        self.assertTrue(actual.is_leaf)
        self.assertIsNone(scalar.grad)

    def test_unsupported_options_shapes_and_operands_are_rejected(self):
        input = torch.ones((2, 3))
        target = torch.zeros((2, 3))

        reduction_error = (
            "torch_rs.nn.functional.mse_loss only supports reduction='none'"
        )
        for reduction in ("mean", "sum", "batchmean", None, 1, object()):
            with self.subTest(reduction=reduction):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    f"^{re.escape(reduction_error)}$",
                ):
                    functional.mse_loss(input, target, reduction=reduction)

        legacy_error = (
            "torch_rs.nn.functional.mse_loss only supports "
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
                    functional.mse_loss(
                        input,
                        target,
                        reduction="none",
                        **legacy_arguments,
                    )

        weight_error = "torch_rs.nn.functional.mse_loss only supports weight=None"
        for weight in (torch.ones((2, 3)), 1.0, [1.0, 1.0]):
            with self.subTest(weight=type(weight)):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    f"^{re.escape(weight_error)}$",
                ):
                    functional.mse_loss(
                        input,
                        target,
                        reduction="none",
                        weight=weight,
                    )

        broadcast_error = (
            "torch_rs.nn.functional.mse_loss does not support broadcasting"
        )
        for actual_input, actual_target in (
            (input, torch.zeros((1,))),
            (input, torch.zeros((3,))),
            (input, torch.zeros((2, 1))),
            (input, torch.zeros((2, 2))),
            (torch.zeros((1,)), input),
        ):
            with self.subTest(
                input_shape=actual_input.shape,
                target_shape=actual_target.shape,
            ):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    f"^{re.escape(broadcast_error)}$",
                ):
                    functional.mse_loss(
                        actual_input,
                        actual_target,
                        reduction="none",
                    )

        class Override:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return object()

        exact_tensor_error = (
            "mse_loss() only supports exact native Tensor input and target operands"
        )
        for actual_input, actual_target in (
            (Override(), target),
            (input, Override()),
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
                    functional.mse_loss(
                        actual_input,
                        actual_target,
                        reduction="none",
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
        with self.assertRaisesRegex(
            TypeError,
            r"^mse_loss\(\) does not support an active TorchFunctionMode$",
        ):
            with mode:
                functional.mse_loss(
                    torch.ones((2, 3), requires_grad=True),
                    torch.zeros((3,)),
                    reduction="mean",
                    weight=object(),
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
