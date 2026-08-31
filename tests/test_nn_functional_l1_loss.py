import importlib
import inspect
import os
import re
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional


class FunctionalL1LossTests(unittest.TestCase):
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

    def broadcast_cases(self):
        scalar = torch.tensor(-0.0)
        offset_scalar = torch.tensor([17.0, 0.5])[1]
        matrix = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist()
        )
        offset_matrix = torch.tensor(
            np.arange(12, dtype=np.float32).reshape(2, 2, 3).tolist()
        )[1]
        noncontiguous_matrix = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(3, 2).tolist()
        ).transpose(0, 1)
        empty_contiguous = torch.zeros((0, 4))
        empty_strided = torch.zeros((2, 0, 3)).transpose(0, 2)

        return (
            ("scalar target", matrix, scalar),
            ("vector target", matrix, torch.tensor([1.0, 2.0, 3.0])),
            ("column target", matrix, torch.tensor([[1.0], [2.0]])),
            ("scalar input", scalar, offset_matrix),
            ("empty scalar input", scalar, empty_contiguous),
            ("empty scalar target", empty_contiguous, scalar),
            ("noncontiguous scalar input", offset_scalar, noncontiguous_matrix),
            ("noncontiguous scalar target", noncontiguous_matrix, offset_scalar),
            (
                "empty singleton broadcast",
                empty_strided,
                torch.ones((1, 0, 1)),
            ),
        )

    @staticmethod
    def broadcast_warning(input, target):
        return (
            f"Using a target size (torch.Size({list(target.shape)})) that is "
            f"different to the input size (torch.Size({list(input.shape)})). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size."
        )

    @staticmethod
    def full_l1_sum_expected(input, target):
        difference = functional.l1_loss(input, target, reduction="none")
        values = np.asarray(difference)
        if difference.numel() == 0:
            return torch.tensor(memoryview(np.asarray([0.0], dtype=np.float32)))[0]

        offsets_and_values = []
        for coordinate in np.ndindex(tuple(difference.shape)):
            offset = sum(
                index * stride
                for index, stride in zip(coordinate, difference.stride(), strict=True)
            )
            offsets_and_values.append((offset, values[coordinate]))

        offsets = sorted(offset for offset, _ in offsets_and_values)
        if offsets == list(range(offsets[0], offsets[0] + difference.numel())):
            sequence = (
                value
                for _, value in sorted(
                    offsets_and_values,
                    key=lambda item: item[0],
                )
            )
        else:
            sequence = values.reshape(-1).tolist()

        total = FunctionalL1LossTests.pytorch_float32_full_sum(sequence)
        return torch.tensor(memoryview(np.asarray([total], dtype=np.float32)))[0]

    @staticmethod
    def pytorch_float32_full_sum(values):
        values = np.asarray(list(values), dtype=np.float32)
        selected_nan = FunctionalL1LossTests.pytorch_sum_preferred_nan(values)
        if selected_nan is not None:
            return selected_nan
        return FunctionalL1LossTests.pytorch_float32_finite_sum(values)

    @staticmethod
    def pytorch_sum_preferred_nan(values):
        for index in FunctionalL1LossTests.pytorch_sum_nan_priority(len(values)):
            bits = values[index].view(np.uint32).item()
            if (bits & 0x7FFF_FFFF) > 0x7F80_0000:
                return np.asarray(
                    [bits | 0x0040_0000],
                    dtype=np.uint32,
                ).view(np.float32)[0]
        return None

    @staticmethod
    def pytorch_sum_nan_priority(length):
        parallel = FunctionalL1LossTests.pytorch_sum_parallel_chunks(length)
        if parallel is not None:
            thread_count, chunk_size = parallel
            return [
                start + index
                for thread_index in range(thread_count - 1, -1, -1)
                if (start := thread_index * chunk_size) < length
                for index in FunctionalL1LossTests.pytorch_sum_serial_nan_priority(
                    min(length, start + chunk_size) - start
                )
            ]

        return FunctionalL1LossTests.pytorch_sum_serial_nan_priority(length)

    @staticmethod
    def pytorch_sum_serial_nan_priority(length):
        if length < 5:
            return range(length)
        if length < 8:
            return [0, *range(4, length), 1, 2, 3]
        chunks = length // 8
        full_length = chunks * 8
        chunk_priority = FunctionalL1LossTests.pytorch_sum_chunk_priority(chunks)
        return [
            *(chunk * 8 + lane for lane in range(7, -1, -1) for chunk in chunk_priority),
            *range(full_length, length),
        ]

    @staticmethod
    def pytorch_sum_chunk_priority(count):
        if count <= 4:
            return range(count)
        groups, remainder = divmod(count, 4)
        group_priority = FunctionalL1LossTests.pytorch_sum_group_priority(groups)
        priority = [group * 4 for group in group_priority]
        priority.extend(range(groups * 4, groups * 4 + remainder))
        for lane in range(1, 4):
            priority.extend(group * 4 + lane for group in group_priority)
        return priority

    @staticmethod
    def pytorch_sum_group_priority(count):
        cascade_block_groups = 16
        level2_period = 16 * 16
        level3_period = 16 * 16 * 16

        level1_start = count - count % cascade_block_groups
        level2_start = level1_start - level1_start % level2_period
        level3_start = level2_start - level2_start % level3_period
        return [
            *range(level1_start, count),
            *range(level2_start, level1_start),
            *range(level3_start, level2_start),
            *range(0, level3_start),
        ]

    @staticmethod
    def pytorch_sum_parallel_chunks(length):
        grain_size = 32_768
        if length <= grain_size:
            return None
        if hasattr(os, "sched_getaffinity"):
            available_cpus = len(os.sched_getaffinity(0))
        else:
            available_cpus = os.cpu_count() or 1
        available_threads = max(available_cpus // 2, 1)
        thread_count = min(
            (length + grain_size - 1) // grain_size,
            available_threads,
        )
        if thread_count <= 1:
            return None
        return thread_count, (length + thread_count - 1) // thread_count

    @staticmethod
    def pytorch_float32_add(left, right):
        return np.float32(np.float32(left) + np.float32(right))

    @staticmethod
    def pytorch_float32_finite_sum(values):
        values = np.asarray(values, dtype=np.float32)
        parallel = FunctionalL1LossTests.pytorch_sum_parallel_chunks(len(values))
        if parallel is not None:
            thread_count, chunk_size = parallel
            partials = []
            for thread_index in range(thread_count):
                start = thread_index * chunk_size
                if start >= len(values):
                    break
                partials.append(
                    FunctionalL1LossTests.pytorch_float32_finite_sum_serial(
                        values[start : min(len(values), start + chunk_size)]
                    ),
                )
            return FunctionalL1LossTests.pytorch_float32_finite_sum_serial(
                np.asarray(partials, dtype=np.float32)
            )

        return FunctionalL1LossTests.pytorch_float32_finite_sum_serial(values)

    @staticmethod
    def pytorch_float32_finite_sum_serial(values):
        add = FunctionalL1LossTests.pytorch_float32_add

        if len(values) < 5:
            total = np.float32(0.0)
            for value in values:
                total = add(total, value)
            return total
        if len(values) < 8:
            total = np.float32(values[0])
            for value in values[4:]:
                total = add(total, value)
            for value in values[1:4]:
                total = add(total, value)
            return total

        width = 8
        vectors_per_group = 4
        cascade_block_groups = 16
        level2_period = 16 * 16
        level3_period = 16 * 16 * 16

        def zero_accumulator():
            return np.zeros((vectors_per_group, width), dtype=np.float32)

        def add_accumulator(accumulator, addend):
            for vector in range(vectors_per_group):
                for lane in range(width):
                    accumulator[vector, lane] = add(
                        accumulator[vector, lane],
                        addend[vector, lane],
                    )

        def group_accumulator(start_group, end_group):
            accumulator = zero_accumulator()
            for group in range(start_group, end_group):
                group_base = group * vectors_per_group * width
                for vector in range(vectors_per_group):
                    base = group_base + vector * width
                    for lane in range(width):
                        accumulator[vector, lane] = add(
                            accumulator[vector, lane],
                            values[base + lane],
                        )
            return accumulator

        full_vectors = len(values) // width
        full_groups = full_vectors // vectors_per_group
        level1 = zero_accumulator()
        level2 = zero_accumulator()
        level3 = zero_accumulator()
        processed_groups = 0
        while processed_groups + cascade_block_groups <= full_groups:
            block = group_accumulator(
                processed_groups,
                processed_groups + cascade_block_groups,
            )
            add_accumulator(level1, block)
            processed_groups += cascade_block_groups
            if processed_groups % level2_period == 0:
                add_accumulator(level2, level1)
                level1 = zero_accumulator()
                if processed_groups % level3_period == 0:
                    add_accumulator(level3, level2)
                    level2 = zero_accumulator()

        current = group_accumulator(processed_groups, full_groups)
        add_accumulator(current, level1)
        add_accumulator(current, level2)
        add_accumulator(current, level3)

        for vector in range(full_groups * vectors_per_group, full_vectors):
            base = vector * width
            for lane in range(width):
                current[0, lane] = add(current[0, lane], values[base + lane])

        lanes = np.zeros(width, dtype=np.float32)
        for lane in range(width):
            lane_total = add(current[0, lane], current[1, lane])
            lane_total = add(lane_total, current[2, lane])
            lane_total = add(lane_total, current[3, lane])
            lanes[lane] = lane_total

        tail = np.float32(0.0)
        for value in values[full_vectors * width :]:
            tail = add(tail, value)

        total = add(lanes[0], tail)
        for lane_total in lanes[1:]:
            total = add(total, lane_total)
        return total

    @staticmethod
    def call(input, target, form, reduction="none"):
        if form == "reduction keyword":
            return functional.l1_loss(input, target, reduction=reduction)
        if form == "legacy none keywords":
            return functional.l1_loss(
                input=input,
                target=target,
                size_average=None,
                reduce=None,
                reduction=reduction,
                weight=None,
            )
        if form == "five positional":
            return functional.l1_loss(input, target, None, None, reduction)
        return functional.l1_loss(input, target, None, None, reduction, None)

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
            "exact ``torch_rs.Tensor`` operands",
            "CPU ``float32`` storage",
            "broadcastable shapes",
            "``reduction='none'``",
            "``reduction='sum'``",
            "``size_average=None``",
            "``reduce=None``",
            "``weight=None``",
            "fuses same-shape row-major contiguous operands",
            "rank-0 scalar broadcasts over row-major contiguous tensors",
            "one native absolute-difference pass",
            "full-tensor sum reduction",
            "subtraction and absolute-value behavior",
            "fresh, independent tensor",
            "size-mismatch warning",
            "Unbroadcastable shapes",
            "``reduction='mean'``",
            "weights",
            "Tensor subclasses",
            "active ``TorchFunctionMode`` contexts",
            "active autograd recording",
            "inside ``torch.no_grad()``",
        ):
            self.assertIn(documented_limit, normalized_doc)

        signature = inspect.signature(l1_loss)
        self.assertEqual(
            tuple(signature.parameters),
            ("input", "target", "size_average", "reduce", "reduction", "weight"),
        )
        self.assertIs(signature.parameters["input"].annotation, torch.Tensor)
        self.assertIs(signature.parameters["target"].annotation, torch.Tensor)
        self.assertEqual(signature.parameters["size_average"].default, None)
        self.assertEqual(signature.parameters["reduce"].default, None)
        self.assertEqual(signature.parameters["reduction"].default, "mean")
        self.assertEqual(signature.parameters["weight"].default, None)
        self.assertIs(signature.return_annotation, torch.Tensor)
        self.assertFalse(hasattr(torch, "_nn_functional_l1_loss"))
        self.assertFalse(hasattr(torch.nn, "L1Loss"))

        wildcard = {}
        exec("from torch_rs.nn.functional import *", wildcard)
        self.assertIs(wildcard["l1_loss"], l1_loss)

    def test_supported_forms_match_subtraction_and_abs_composition(self):
        for case, input, target in self.layout_cases():
            difference = input - target
            expected = difference.abs()
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

    def test_sum_reduction_supported_forms_match_l1_none_full_sum(self):
        for case, input, target in self.layout_cases():
            expected = self.full_l1_sum_expected(input, target)
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)
            for form in (
                "reduction keyword",
                "legacy none keywords",
                "five positional",
                "six positional",
            ):
                actual = self.call(input, target, form, reduction="sum")
                repeated = self.call(input, target, form, reduction="sum")
                self.assert_matches_composition(
                    actual,
                    expected,
                    case=(case, form),
                )
                with self.subTest(case=(case, form), scalar=True):
                    self.assertEqual(actual.shape, ())
                    self.assertEqual(actual.stride(), ())
                    self.assertEqual(actual.storage_offset(), 0)
                    self.assertEqual(actual.numel(), 1)
                with self.subTest(case=(case, form), storage=True):
                    self.assertIsNot(actual, repeated)
                    self.assertFalse(actual.is_set_to(repeated))
                    self.assertFalse(actual.is_set_to(input))
                    self.assertFalse(actual.is_set_to(target))
                    self.assertNotEqual(actual.data_ptr(), repeated.data_ptr())
                    self.assertNotEqual(actual.data_ptr(), input.data_ptr())
                    self.assertNotEqual(actual.data_ptr(), target.data_ptr())
                with self.subTest(case=(case, form), nonmutation=True):
                    self.assertEqual(self.tensor_state(input)[:-1], input_state[:-1])
                    self.assertEqual(
                        self.tensor_state(target)[:-1],
                        target_state[:-1],
                    )
                    np.testing.assert_array_equal(
                        self.tensor_state(input)[-1],
                        input_state[-1],
                    )
                    np.testing.assert_array_equal(
                        self.tensor_state(target)[-1],
                        target_state[-1],
                    )

    def test_bandwidth_sized_same_shape_contiguous_matches_composition(self):
        input_values = np.linspace(
            -1024.0,
            1024.0,
            1024 * 1024,
            dtype=np.float32,
        ).reshape(1024, 1024)
        target_values = np.linspace(
            2048.0,
            -2048.0,
            1024 * 1024,
            dtype=np.float32,
        ).reshape(1024, 1024)
        input = torch.tensor(memoryview(input_values.reshape(-1))).view(1024, 1024)
        target = torch.tensor(memoryview(target_values.reshape(-1))).view(1024, 1024)

        self.assertTrue(input.is_contiguous())
        self.assertTrue(target.is_contiguous())
        actual = functional.l1_loss(input, target, reduction="none")
        expected = (input - target).abs()

        self.assert_matches_composition(
            actual,
            expected,
            case="bandwidth-sized contiguous",
        )
        self.assertFalse(actual.is_set_to(input))
        self.assertFalse(actual.is_set_to(target))
        self.assertNotEqual(actual.data_ptr(), input.data_ptr())
        self.assertNotEqual(actual.data_ptr(), target.data_ptr())

    def test_broadcasted_inputs_match_composition_warning_and_storage(self):
        for case, input, target in self.broadcast_cases():
            difference = input - target
            expected = difference.abs()
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warning_line = inspect.currentframe().f_lineno + 1
                actual = functional.l1_loss(input, target, reduction="none")

            with self.subTest(case=case, warning=True):
                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, UserWarning)
                self.assertEqual(
                    str(caught[0].message),
                    self.broadcast_warning(input, target),
                )
                self.assertEqual(caught[0].filename, __file__)
                self.assertEqual(caught[0].lineno, warning_line)

            self.assert_matches_composition(actual, expected, case=case)
            with self.subTest(case=case, storage=True):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    repeated = functional.l1_loss(input, target, reduction="none")
                self.assertIsNot(actual, repeated)
                self.assertFalse(actual.is_set_to(repeated))
                self.assertFalse(actual.is_set_to(input))
                self.assertFalse(actual.is_set_to(target))
                if actual.numel() != 0:
                    self.assertNotEqual(actual.data_ptr(), repeated.data_ptr())
                    self.assertNotEqual(actual.data_ptr(), input.data_ptr())
                    self.assertNotEqual(actual.data_ptr(), target.data_ptr())

            with self.subTest(case=case, nonmutation=True):
                self.assertEqual(self.tensor_state(input)[:-1], input_state[:-1])
                self.assertEqual(self.tensor_state(target)[:-1], target_state[:-1])
                np.testing.assert_array_equal(
                    self.tensor_state(input)[-1], input_state[-1]
                )
                np.testing.assert_array_equal(
                    self.tensor_state(target)[-1], target_state[-1]
                )

    def test_sum_reduction_broadcasted_inputs_warn_return_scalar_and_do_not_alias(
        self,
    ):
        for case, input, target in self.broadcast_cases():
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warning_line = inspect.currentframe().f_lineno + 1
                actual = functional.l1_loss(input, target, reduction="sum")

            with self.subTest(case=case, warning=True):
                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, UserWarning)
                self.assertEqual(
                    str(caught[0].message),
                    self.broadcast_warning(input, target),
                )
                self.assertEqual(caught[0].filename, __file__)
                self.assertEqual(caught[0].lineno, warning_line)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expected = self.full_l1_sum_expected(input, target)
                repeated = functional.l1_loss(input, target, reduction="sum")

            self.assert_matches_composition(actual, expected, case=case)
            with self.subTest(case=case, scalar=True):
                self.assertEqual(actual.shape, ())
                self.assertEqual(actual.stride(), ())
                self.assertEqual(actual.storage_offset(), 0)
                self.assertEqual(actual.numel(), 1)

            with self.subTest(case=case, storage=True):
                self.assertIsNot(actual, repeated)
                self.assertFalse(actual.is_set_to(repeated))
                self.assertFalse(actual.is_set_to(input))
                self.assertFalse(actual.is_set_to(target))
                self.assertNotEqual(actual.data_ptr(), repeated.data_ptr())
                self.assertNotEqual(actual.data_ptr(), input.data_ptr())
                self.assertNotEqual(actual.data_ptr(), target.data_ptr())

            with self.subTest(case=case, nonmutation=True):
                self.assertEqual(self.tensor_state(input)[:-1], input_state[:-1])
                self.assertEqual(self.tensor_state(target)[:-1], target_state[:-1])
                np.testing.assert_array_equal(
                    self.tensor_state(input)[-1],
                    input_state[-1],
                )
                np.testing.assert_array_equal(
                    self.tensor_state(target)[-1],
                    target_state[-1],
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
        expected = difference.abs()
        self.assertEqual(difference.stride(), (3, 6, 1))
        self.assertEqual(expected.stride(), (3, 3, 1))

        actual = functional.l1_loss(input, target, reduction="none")
        self.assertEqual(actual.stride(), (3, 3, 1))
        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.tensor_bits(expected),
        )

    def test_every_call_returns_fresh_independent_storage(self):
        for case, input, target in self.layout_cases():
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
            expected = difference.abs()
            expected_bits = self.tensor_bits(expected).copy()
            target_bits_for_case = self.tensor_bits(actual_target)
            target_nan = (target_bits_for_case & 0x7FFF_FFFF) > 0x7F80_0000
            expected_bits[target_nan] = (
                target_bits_for_case[target_nan] | 0x0040_0000
            ) & 0x7FFF_FFFF
            actual = functional.l1_loss(
                actual_input,
                actual_target,
                reduction="none",
            )
            with self.subTest(case=case):
                self.assertEqual(actual.stride(), expected.stride())
                np.testing.assert_array_equal(
                    self.tensor_bits(actual),
                    expected_bits,
                )

    def test_scalar_broadcast_float32_edges_match_kernel_composition_bits(self):
        tensor_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7F81_2345,
                0xFF85_4321,
            ],
            dtype=np.uint32,
        )
        contiguous_tensor = torch.tensor(
            memoryview(tensor_bits.view(np.float32))
        ).view(2, 7)
        empty_tensor = torch.zeros((0, 7))

        for scalar_bits in (
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC6_789A,
            0x7F86_789A,
        ):
            scalar_values = np.asarray([scalar_bits], dtype=np.uint32).view(np.float32)
            scalar = torch.tensor(memoryview(scalar_values))[0]
            for layout, tensor in (
                ("contiguous", contiguous_tensor),
                ("empty", empty_tensor),
                ("noncontiguous fallback", contiguous_tensor.transpose(0, 1)),
            ):
                for scalar_on_left in (True, False):
                    input, target = (
                        (scalar, tensor) if scalar_on_left else (tensor, scalar)
                    )
                    difference = input - target
                    expected = difference.abs()
                    expected_bits = self.tensor_bits(expected).copy()
                    if target.shape == ():
                        target_bits_for_case = np.full(
                            expected_bits.shape,
                            self.tensor_bits(target)[0],
                            dtype=np.uint32,
                        )
                    else:
                        target_bits_for_case = self.tensor_bits(target)
                    target_nan = (target_bits_for_case & 0x7FFF_FFFF) > 0x7F80_0000
                    expected_bits[target_nan] = (
                        target_bits_for_case[target_nan] | 0x0040_0000
                    ) & 0x7FFF_FFFF
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        actual = functional.l1_loss(input, target, reduction="none")

                    with self.subTest(
                        layout=layout,
                        scalar_bits=hex(scalar_bits),
                        scalar_on_left=scalar_on_left,
                        warning=True,
                    ):
                        self.assertEqual(len(caught), 1)
                        self.assertIs(caught[0].category, UserWarning)
                        self.assertEqual(
                            str(caught[0].message),
                            self.broadcast_warning(input, target),
                        )

                    with self.subTest(
                        layout=layout,
                        scalar_bits=hex(scalar_bits),
                        scalar_on_left=scalar_on_left,
                    ):
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertEqual(actual.stride(), expected.stride())
                        self.assertEqual(
                            actual.storage_offset(),
                            expected.storage_offset(),
                        )
                        self.assertEqual(
                            actual.is_contiguous(),
                            expected.is_contiguous(),
                        )
                        self.assertEqual(actual.requires_grad, expected.requires_grad)
                        self.assertEqual(actual.is_leaf, expected.is_leaf)
                        self.assertIs(actual.dtype, torch.float32)
                        self.assertEqual(actual.device, torch.device("cpu"))
                        np.testing.assert_array_equal(
                            self.tensor_bits(actual),
                            expected_bits,
                        )
                        if layout == "noncontiguous fallback":
                            self.assertFalse(tensor.is_contiguous())
                            self.assertEqual(actual.stride(), expected.stride())

    def test_sum_reduction_float32_edge_values_match_l1_none_full_sum_bits(self):
        input_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7F81_2345,
                0xFF85_4321,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
            ],
            dtype=np.uint32,
        )
        target_bits = np.asarray(
            [
                0x8000_0000,
                0x0000_0000,
                0x8000_0001,
                0x0000_0001,
                0xFF80_0000,
                0x7F80_0000,
                0xFFC6_789A,
                0x7FC2_ABCD,
                0xFF86_789A,
                0x7F82_ABCD,
                0x0000_0000,
                0x8000_0000,
            ],
            dtype=np.uint32,
        )
        input = torch.tensor(memoryview(input_bits.view(np.float32))).view(3, 4)
        target = torch.tensor(memoryview(target_bits.view(np.float32))).view(3, 4)

        for case, actual_input, actual_target in (
            ("contiguous", input, target),
            ("transposed", input.transpose(0, 1), target.transpose(0, 1)),
        ):
            actual = functional.l1_loss(
                actual_input,
                actual_target,
                reduction="sum",
            )
            expected = self.full_l1_sum_expected(actual_input, actual_target)
            self.assert_matches_composition(
                actual,
                expected,
                case=case,
            )

    def test_sum_reduction_uses_pytorch_finite_accumulation_order(self):
        target_bits = np.asarray(
            [
                0x3F20_1FC7,
                0x3695_46B8,
                0x3A72_B89E,
                0x4082_7EFF,
                0x46DC_0C5D,
            ],
            dtype=np.uint32,
        )
        input = torch.zeros((5,), dtype=torch.float32)
        target = torch.tensor(memoryview(target_bits.view(np.float32)))
        expected = torch.tensor(
            memoryview(np.asarray([0x46DC_15C5], dtype=np.uint32).view(np.float32))
        )[0]

        actual = functional.l1_loss(input, target, reduction="sum")

        self.assert_matches_composition(
            actual,
            expected,
            case="finite accumulation order",
        )

    def test_sum_reduction_uses_pytorch_parallel_accumulation_order(self):
        for length, expected_bits in (
            (32_773, 0x454C_D4D0),
            (1_048_576, 0x47CC_CCCF),
        ):
            input = torch.zeros((length,), dtype=torch.float32)
            target = torch.tensor(
                memoryview(np.full(length, 0.1, dtype=np.float32))
            )
            expected = torch.tensor(
                memoryview(
                    np.asarray([expected_bits], dtype=np.uint32).view(np.float32)
                )
            )[0]

            actual = functional.l1_loss(input, target, reduction="sum")

            self.assert_matches_composition(
                actual,
                expected,
                case=("parallel finite accumulation order", length),
            )

    def test_sum_reduction_uses_pytorch_cascade_nan_payload_precedence(self):
        for length, left_index, right_index in (
            (544, 31, 543),
            (32_773, 31, 32_772),
        ):
            target_bits = np.zeros(length, dtype=np.uint32)
            target_bits[left_index] = 0x7F80_0001
            target_bits[right_index] = 0x7F80_1001
            input = torch.zeros((length,), dtype=torch.float32)
            target = torch.tensor(memoryview(target_bits.view(np.float32)))
            expected = torch.tensor(
                memoryview(
                    np.asarray([0x7FC0_1001], dtype=np.uint32).view(np.float32)
                )
            )[0]

            actual = functional.l1_loss(input, target, reduction="sum")

            self.assert_matches_composition(
                actual,
                expected,
                case=("NaN payload precedence", length),
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
                for reduction in ("none", "sum"):
                    with self.subTest(reduction=reduction):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            r"^l1_loss\(\): autograd recording is not supported$",
                        ):
                            functional.l1_loss(input, target, reduction=reduction)

                        with torch.no_grad():
                            actual = functional.l1_loss(
                                input,
                                target,
                                reduction=reduction,
                            )
                            difference = input - target
                            expected = difference.abs()
                            if reduction == "sum":
                                expected = expected.sum()
                        self.assert_matches_composition(
                            actual,
                            expected,
                            case=("no_grad", reduction),
                        )
                        self.assertFalse(actual.requires_grad)
                        self.assertTrue(actual.is_leaf)
                        self.assertIsNone(input.grad)
                        self.assertIsNone(target.grad)

    def test_scalar_broadcast_requires_grad_operands_need_no_grad(self):
        def scalar_input(input_requires_grad, target_requires_grad):
            return (
                torch.tensor(-0.5, requires_grad=input_requires_grad),
                torch.tensor(
                    [[1.0, -2.0], [3.0, -4.0]],
                    requires_grad=target_requires_grad,
                ),
            )

        def scalar_target(input_requires_grad, target_requires_grad):
            return (
                torch.tensor(
                    [[1.0, -2.0], [3.0, -4.0]],
                    requires_grad=input_requires_grad,
                ),
                torch.tensor(-0.5, requires_grad=target_requires_grad),
            )

        for case, factory in (
            ("scalar input", scalar_input),
            ("scalar target", scalar_target),
        ):
            for input_requires_grad, target_requires_grad in (
                (True, False),
                (False, True),
                (True, True),
            ):
                input, target = factory(input_requires_grad, target_requires_grad)
                with self.subTest(
                    case=case,
                    input_requires_grad=input_requires_grad,
                    target_requires_grad=target_requires_grad,
                ):
                    for reduction in ("none", "sum"):
                        with self.subTest(reduction=reduction):
                            with self.assertWarnsRegex(
                                UserWarning,
                                "Using a target size",
                            ):
                                with self.assertRaisesRegex(
                                    RuntimeError,
                                    r"^l1_loss\(\): autograd recording is not supported$",
                                ):
                                    functional.l1_loss(
                                        input,
                                        target,
                                        reduction=reduction,
                                    )

                            with warnings.catch_warnings(), torch.no_grad():
                                warnings.simplefilter("ignore")
                                actual = functional.l1_loss(
                                    input,
                                    target,
                                    reduction=reduction,
                                )
                                expected = (input - target).abs()
                                if reduction == "sum":
                                    expected = expected.sum()
                            self.assert_matches_composition(
                                actual,
                                expected,
                                case=("no_grad", reduction),
                            )
                            self.assertFalse(actual.requires_grad)
                            self.assertTrue(actual.is_leaf)
                            self.assertIsNone(input.grad)
                            self.assertIsNone(target.grad)

    def test_unsupported_options_shapes_and_operands_are_rejected(self):
        input = torch.ones((2, 3))
        target = torch.zeros((2, 3))

        reduction_error = (
            "torch_rs.nn.functional.l1_loss only supports "
            "reduction='none' or 'sum'"
        )
        for reduction in ("mean", "batchmean", None, 1, object()):
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

        unbroadcastable_target = torch.zeros((2, 2))
        for reduction in ("none", "sum"):
            with self.subTest(reduction=reduction, unbroadcastable=True):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"^The size of tensor a \(3\) must match the size of tensor b "
                        r"\(2\) at non-singleton dimension 1$",
                    ):
                        functional.l1_loss(
                            input,
                            unbroadcastable_target,
                            reduction=reduction,
                        )
                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, UserWarning)
                self.assertEqual(
                    str(caught[0].message),
                    self.broadcast_warning(input, unbroadcastable_target),
                )

        class Override:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return object()

        exact_tensor_error = (
            "l1_loss() only supports exact native Tensor input and target operands"
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
                    functional.l1_loss(
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
            r"^l1_loss\(\) does not support an active TorchFunctionMode$",
        ):
            with mode:
                functional.l1_loss(
                    torch.ones((2, 3), requires_grad=True),
                    torch.zeros((3,)),
                    reduction="mean",
                )
        self.assertEqual(mode.calls, 0)

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
