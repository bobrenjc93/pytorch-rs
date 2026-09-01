import gc
import inspect
import json
import pickle
import re
import subprocess
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelReshapeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("reshape differentials require pinned PyTorch 2.13.0")

    def assert_matches(self, actual, expected, actual_source, expected_source, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertEqual(
                actual.data_ptr() == actual_source.data_ptr(),
                expected.data_ptr() == expected_source.data_ptr(),
            )
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_array_equal(
                np.asarray(actual), expected.detach().cpu().numpy()
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_overflow_shape_error_matches(self, actual_call, expected_call, position):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        marker = (
            f"reshape(): argument 'shape' failed to unpack the object at pos {position} "
            'with error "Overflow when unpacking long long'
        )
        self.assertIn(marker, str(actual_raised.exception))
        self.assertIn(marker, str(expected_raised.exception))

    def test_layouts_aliasing_call_forms_and_lifetimes_match_pytorch_2_13(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_base = torch.tensor(values.tolist(), requires_grad=True)
        expected_base = reference_torch.tensor(values, requires_grad=True)
        cases = (
            ("scalar", torch.tensor(-0.0), reference_torch.tensor(-0.0), ()),
            (
                "empty-offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                reference_torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (2, 0),
            ),
            ("contiguous", actual_base, expected_base, (6, 4)),
            ("offset", actual_base[1], expected_base[1], (2, 6)),
            (
                "transpose-copy",
                actual_base.transpose(0, 2),
                expected_base.transpose(0, 2),
                (6, 4),
            ),
        )

        retained = []
        for index, (case, actual_source, expected_source, shape) in enumerate(cases):
            if index == 0:
                actual = torch.reshape(actual_source, tuple(shape))
                expected = reference_torch.reshape(expected_source, tuple(shape))
            elif index == 1:
                actual = torch.reshape(actual_source, list(shape))
                expected = reference_torch.reshape(expected_source, list(shape))
            elif index == 2:
                actual = torch.reshape(input=actual_source, shape=torch.Size(shape))
                expected = reference_torch.reshape(
                    input=expected_source, shape=reference_torch.Size(shape)
                )
            elif index == 3:
                actual = torch.reshape(x=actual_source, shape=tuple(shape))
                expected = reference_torch.reshape(x=expected_source, shape=tuple(shape))
            else:
                actual = torch.reshape(x1=actual_source, shape=tuple(shape))
                expected = reference_torch.reshape(x1=expected_source, shape=tuple(shape))
            self.assertIsNot(actual, actual_source)
            self.assertIsNot(expected, expected_source)
            self.assert_matches(actual, expected, actual_source, expected_source, case)
            retained.append((actual, expected))

        for case, actual_source, expected_source, shape in cases[1:]:
            with self.subTest(case=case, form="variadic"):
                actual = torch.reshape(actual_source, *shape)
                expected = reference_torch.reshape(expected_source, tuple(shape))
                self.assertIsNot(actual, actual_source)
                self.assertIsNot(expected, expected_source)
                self.assert_matches(actual, expected, actual_source, expected_source, case)

        del actual_base, expected_base, cases
        gc.collect()
        np.testing.assert_array_equal(
            np.asarray(retained[3][0]), retained[3][1].detach().cpu().numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(retained[4][0]), retained[4][1].detach().cpu().numpy()
        )

    def test_inferred_empty_and_errors_match_pytorch_2_13(self):
        actual_source = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        expected_source = reference_torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        )
        self.assert_matches(
            torch.reshape(actual_source, (2, -1, 2)),
            reference_torch.reshape(expected_source, (2, -1, 2)),
            actual_source,
            expected_source,
            "inferred",
        )
        self.assert_matches(
            torch.reshape(actual_source, 2, -1, 2),
            reference_torch.reshape(expected_source, (2, -1, 2)),
            actual_source,
            expected_source,
            "variadic inferred",
        )

        maximum = sys.maxsize
        actual_empty = torch.zeros((0,))
        expected_empty = reference_torch.zeros((0,))
        actual = torch.reshape(actual_empty, (0, maximum, maximum))
        expected = reference_torch.reshape(expected_empty, (0, maximum, maximum))
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.data_ptr() == actual_empty.data_ptr(), True)
        self.assertEqual(expected.data_ptr() == expected_empty.data_ptr(), True)
        self.assertEqual(actual.tolist(), expected.tolist())

        actual = torch.reshape(actual_empty, 0, maximum, maximum)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.data_ptr() == actual_empty.data_ptr(), True)
        self.assertEqual(actual.tolist(), expected.tolist())

        for shape in ((2, 2), (-1, -1), (2, -2), (0, -1)):
            with self.subTest(shape=shape):
                self.assert_error_matches(
                    lambda shape=shape: torch.reshape(torch.zeros((6,)), shape),
                    lambda shape=shape: reference_torch.reshape(
                        reference_torch.zeros((6,)), shape
                    ),
                )

    def test_autograd_repeated_backward_and_no_grad_match_pytorch_2_13(self):
        gradients = []
        states = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            view = (
                module.reshape(leaf, 3, 2)
                if module is torch
                else module.reshape(leaf, (3, 2))
            )
            states.append((view.requires_grad, view.is_leaf))
            weights = module.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
            (view * weights).sum().backward()
            gradients.append(np.asarray(leaf.grad).copy())

            copy_leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            copied = (
                module.reshape(copy_leaf.transpose(0, 1), 6)
                if module is torch
                else module.reshape(copy_leaf.transpose(0, 1), [6])
            )
            states.append((copied.requires_grad, copied.is_leaf))
            weights = module.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            (copied * weights).sum().backward()
            gradients.append(np.asarray(copy_leaf.grad).copy())

            repeated_leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            loss = (
                module.reshape(repeated_leaf.transpose(0, 1), 3, 2)
                if module is torch
                else module.reshape(repeated_leaf.transpose(0, 1), (3, 2))
            ).sum()
            loss.backward()
            loss.backward()
            gradients.append(np.asarray(repeated_leaf.grad).copy())

            source = module.tensor(
                [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
            )
            with module.no_grad():
                alias = (
                    module.reshape(source, 4)
                    if module is torch
                    else module.reshape(source, (4,))
                )
                copied = (
                    module.reshape(source.transpose(0, 1), 4)
                    if module is torch
                    else module.reshape(source.transpose(0, 1), (4,))
                )
            states.append(
                (
                    alias.requires_grad,
                    alias.is_leaf,
                    copied.requires_grad,
                    copied.is_leaf,
                )
            )

        np.testing.assert_array_equal(gradients[0], gradients[3])
        np.testing.assert_array_equal(gradients[1], gradients[4])
        np.testing.assert_array_equal(gradients[2], gradients[5])
        self.assertEqual(states[:3], states[3:])

    def test_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = reference_torch.tensor([1.0, 2.0, 3.0, 4.0])
        cases = (
            (lambda: torch.reshape(), lambda: reference_torch.reshape()),
            (lambda: torch.reshape(actual), lambda: reference_torch.reshape(expected)),
            (
                lambda: torch.reshape(actual, (2, 2), input=actual),
                lambda: reference_torch.reshape(expected, (2, 2), input=expected),
            ),
            (
                lambda: torch.reshape(actual, (2, 2), shape=(4,)),
                lambda: reference_torch.reshape(expected, (2, 2), shape=(4,)),
            ),
            (
                lambda: torch.reshape(input=actual, size=(2, 2)),
                lambda: reference_torch.reshape(input=expected, size=(2, 2)),
            ),
            (
                lambda: torch.reshape(input=actual, shape=4),
                lambda: reference_torch.reshape(input=expected, shape=4),
            ),
            (
                lambda: torch.reshape(shape=(2, 2)),
                lambda: reference_torch.reshape(shape=(2, 2)),
            ),
            (
                lambda: torch.reshape(actual, torch.float32),
                lambda: reference_torch.reshape(expected, reference_torch.float32),
            ),
            (
                lambda: torch.reshape(actual, [True]),
                lambda: reference_torch.reshape(expected, [True]),
            ),
            (
                lambda: torch.reshape(actual, [2.0, 2]),
                lambda: reference_torch.reshape(expected, [2.0, 2]),
            ),
            (
                lambda: torch.reshape(input=actual, shape=[True]),
                lambda: reference_torch.reshape(input=expected, shape=[True]),
            ),
            (
                lambda: torch.reshape([1.0], (1,)),
                lambda: reference_torch.reshape([1.0], (1,)),
            ),
            (
                lambda: torch.reshape(input=[1.0], shape=(1,)),
                lambda: reference_torch.reshape(input=[1.0], shape=(1,)),
            ),
            (
                lambda: torch.reshape(actual, (2, 2), extra=True),
                lambda: reference_torch.reshape(expected, (2, 2), extra=True),
            ),
            (
                lambda: torch.reshape(x=actual, a=actual, shape=(2, 2)),
                lambda: reference_torch.reshape(x=expected, a=expected, shape=(2, 2)),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_oversized_shape_dimension_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        too_large = 2**63
        too_small = -(2**63) - 1
        cases = (
            (
                "positional-tuple",
                lambda: torch.reshape(actual, (too_large,)),
                lambda: reference_torch.reshape(expected, (too_large,)),
                1,
            ),
            (
                "keyword-list",
                lambda: torch.reshape(input=actual, shape=[too_large]),
                lambda: reference_torch.reshape(input=expected, shape=[too_large]),
                1,
            ),
            (
                "second-dimension",
                lambda: torch.reshape(actual, (1, too_large)),
                lambda: reference_torch.reshape(expected, (1, too_large)),
                2,
            ),
            (
                "negative-overflow",
                lambda: torch.reshape(actual, (too_small,)),
                lambda: reference_torch.reshape(expected, (too_small,)),
                1,
            ),
            (
                "numpy-uint64",
                lambda: torch.reshape(actual, (np.uint64(too_large),)),
                lambda: reference_torch.reshape(expected, (np.uint64(too_large),)),
                1,
            ),
        )
        for name, actual_call, expected_call, position in cases:
            with self.subTest(name=name):
                self.assert_overflow_shape_error_matches(
                    actual_call, expected_call, position
                )

    def stateful_index_contract(self, module):
        class StatefulIndexDimension:
            def __init__(self, values):
                self.values = list(values)
                self.calls = 0

            def __index__(self):
                value = self.values[self.calls]
                self.calls += 1
                return value

        tensor = module.tensor([1.0, 2.0, 3.0, 4.0])

        dimension = StatefulIndexDimension((1, 2))
        result = module.reshape(tensor, (dimension, 2))
        positional = (tuple(result.shape), dimension.calls, np.asarray(result).tolist())

        dimension = StatefulIndexDimension((1, 2))
        result = module.reshape(input=tensor, shape=[dimension, 2])
        keyword_list = (tuple(result.shape), dimension.calls, np.asarray(result).tolist())

        dimension = StatefulIndexDimension((2**63, 2))
        result = module.reshape(tensor, (dimension, 2))
        first_overflow = (tuple(result.shape), dimension.calls, np.asarray(result).tolist())

        dimension = StatefulIndexDimension((2, 1))
        try:
            module.reshape(tensor, (dimension, 2))
        except Exception as error:
            invalid = (type(error).__name__, str(error), dimension.calls)
        else:
            invalid = None

        dimension = StatefulIndexDimension((2, 2**63))
        try:
            module.reshape(tensor, (dimension, 2))
        except Exception as error:
            message = str(error)
            overflow = (
                type(error).__name__,
                "failed to unpack the object at pos 1" in message,
                "Overflow when unpacking long long" in message,
                dimension.calls,
            )
        else:
            overflow = None

        dimension = StatefulIndexDimension((2, 2.0))
        try:
            module.reshape(tensor, (dimension, 2))
        except Exception as error:
            second_type = (type(error).__name__, str(error), dimension.calls)
        else:
            second_type = None

        return {
            "positional": positional,
            "keyword_list": keyword_list,
            "first_overflow": first_overflow,
            "invalid": invalid,
            "overflow": overflow,
            "second_type": second_type,
        }

    def test_stateful_index_dimension_conversion_matches_pytorch_2_13(self):
        self.assertEqual(
            self.stateful_index_contract(torch),
            self.stateful_index_contract(reference_torch),
        )

    def dispatch_contract(self, module):
        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                override_calls.append((func, dispatch_types, args, kwargs))
                return "override"

        value = Override()
        override_result = module.reshape(x=value, shape=(2, 2))
        override_function, override_types, override_args, override_kwargs = (
            override_calls[0]
        )

        mode_calls = []

        class Mode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                mode_calls.append((func, dispatch_types, args, kwargs))
                return "mode"

        tensor = module.tensor([1.0, 2.0, 3.0, 4.0])
        with Mode():
            mode_result = module.reshape(input=tensor, shape=(2, 2))
        mode_function, mode_types, mode_args, mode_kwargs = mode_calls[0]

        mode_calls.clear()
        with Mode():
            late_invalid_mode_result = module.reshape(tensor, (1, 2.0))
        late_invalid_mode_call = mode_calls[0]

        mode_calls.clear()
        try:
            with Mode():
                module.reshape(tensor, (2.0, 1))
        except Exception as error:
            first_invalid_mode = (type(error).__name__, str(error), len(mode_calls))
        else:
            first_invalid_mode = None

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append((self.label, func, dispatch_types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = module.reshape(tensor, (2, 2))

        shape_calls = []

        class ShapeOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                shape_calls.append((func, dispatch_types, args, kwargs))
                return "shape"

        shape_object = ShapeOverride()
        shape_object_result = module.reshape(tensor, shape_object)
        shape_object_call = shape_calls[-1]
        shape_tuple = (ShapeOverride(), 2)
        shape_tuple_result = module.reshape(tensor, shape_tuple)
        shape_tuple_call = shape_calls[-1]
        shape_list = [ShapeOverride(), 2]
        shape_list_result = module.reshape(input=tensor, shape=shape_list)
        shape_list_call = shape_calls[-1]
        late_invalid_shape = (1, 2.0, ShapeOverride())
        late_invalid_shape_result = module.reshape(tensor, late_invalid_shape)
        late_invalid_shape_call = shape_calls[-1]

        input_shape_calls = []

        class InputOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                input_shape_calls.append((func, dispatch_types, args, kwargs))
                return "input"

        input_shape_result = module.reshape(InputOverride(), ShapeOverride())
        input_shape_call = input_shape_calls[-1]
        late_invalid_input = InputOverride()
        late_invalid_input_result = module.reshape(late_invalid_input, (1, 2.0))
        late_invalid_input_call = input_shape_calls[-1]
        input_shape_calls.clear()
        try:
            module.reshape(InputOverride(), (2.0, 2))
        except Exception as error:
            first_invalid_input = (type(error).__name__, str(error), len(input_shape_calls))
        else:
            first_invalid_input = None

        subclass_calls = []

        class BaseShape:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                subclass_calls.append(("base", dispatch_types))
                return "base"

        class DerivedShape(BaseShape):
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                subclass_calls.append(("derived", dispatch_types))
                return "derived"

        subclass_result = module.reshape(tensor, (BaseShape(), DerivedShape()))

        try:
            module.reshape(tensor, (2.0, ShapeOverride()))
        except Exception as error:
            invalid_before_shape_override = (type(error).__name__, str(error))
        else:
            invalid_before_shape_override = None

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        try:
            module.reshape(DecliningOverride(), (2, 2))
        except Exception as error:
            decline_error = (type(error).__name__, str(error))
        else:
            decline_error = None

        try:
            module.reshape(tensor, (DecliningOverride(), 2))
        except Exception as error:
            decline_shape_error = (type(error).__name__, str(error))
        else:
            decline_shape_error = None

        return {
            "override_result": override_result,
            "override_function": override_function is module.reshape,
            "override_types": tuple(value.__name__ for value in override_types),
            "override_args": len(override_args),
            "override_kwargs": tuple(override_kwargs),
            "override_value_identity": override_kwargs["x"] is value,
            "override_shape": override_kwargs["shape"],
            "mode_result": mode_result,
            "mode_function": mode_function is module.reshape,
            "mode_types": tuple(value.__name__ for value in mode_types),
            "mode_args": len(mode_args),
            "mode_kwargs": tuple(mode_kwargs),
            "mode_value_identity": mode_kwargs["input"] is tensor,
            "mode_shape": mode_kwargs["shape"],
            "late_invalid_mode_result": late_invalid_mode_result,
            "late_invalid_mode_function": late_invalid_mode_call[0] is module.reshape,
            "late_invalid_mode_types": tuple(
                value.__name__ for value in late_invalid_mode_call[1]
            ),
            "late_invalid_mode_args": (
                late_invalid_mode_call[2][0] is tensor,
                late_invalid_mode_call[2][1],
            ),
            "late_invalid_mode_kwargs": late_invalid_mode_call[3],
            "first_invalid_mode": first_invalid_mode,
            "forward_order": tuple(item[0] for item in order),
            "forward_functions": tuple(item[1] is module.reshape for item in order),
            "forward_types": tuple(item[2] for item in order),
            "forward_arg_lengths": tuple(len(item[3]) for item in order),
            "forward_kwargs": tuple(None if item[4] is None else tuple(item[4]) for item in order),
            "forwarded_shape": tuple(forwarded.shape),
            "forwarded_stride": forwarded.stride(),
            "forwarded_values": np.asarray(forwarded).tolist(),
            "shape_object_result": shape_object_result,
            "shape_object_function": shape_object_call[0] is module.reshape,
            "shape_object_types": tuple(value.__name__ for value in shape_object_call[1]),
            "shape_object_args": (
                shape_object_call[2][0] is tensor,
                shape_object_call[2][1] is shape_object,
            ),
            "shape_object_kwargs": shape_object_call[3],
            "shape_tuple_result": shape_tuple_result,
            "shape_tuple_function": shape_tuple_call[0] is module.reshape,
            "shape_tuple_types": tuple(value.__name__ for value in shape_tuple_call[1]),
            "shape_tuple_args": (
                shape_tuple_call[2][0] is tensor,
                shape_tuple_call[2][1] is shape_tuple,
            ),
            "shape_tuple_kwargs": shape_tuple_call[3],
            "shape_list_result": shape_list_result,
            "shape_list_function": shape_list_call[0] is module.reshape,
            "shape_list_types": tuple(value.__name__ for value in shape_list_call[1]),
            "shape_list_args": len(shape_list_call[2]),
            "shape_list_kwargs": tuple(shape_list_call[3]),
            "shape_list_input_identity": shape_list_call[3]["input"] is tensor,
            "shape_list_shape_identity": shape_list_call[3]["shape"] is shape_list,
            "late_invalid_shape_result": late_invalid_shape_result,
            "late_invalid_shape_function": late_invalid_shape_call[0] is module.reshape,
            "late_invalid_shape_types": tuple(
                value.__name__ for value in late_invalid_shape_call[1]
            ),
            "late_invalid_shape_args": (
                late_invalid_shape_call[2][0] is tensor,
                late_invalid_shape_call[2][1] is late_invalid_shape,
            ),
            "late_invalid_shape_kwargs": late_invalid_shape_call[3],
            "input_shape_result": input_shape_result,
            "input_shape_function": input_shape_call[0] is module.reshape,
            "input_shape_types": tuple(value.__name__ for value in input_shape_call[1]),
            "input_shape_arg_length": len(input_shape_call[2]),
            "input_shape_kwargs": input_shape_call[3],
            "late_invalid_input_result": late_invalid_input_result,
            "late_invalid_input_function": late_invalid_input_call[0] is module.reshape,
            "late_invalid_input_types": tuple(
                value.__name__ for value in late_invalid_input_call[1]
            ),
            "late_invalid_input_args": (
                late_invalid_input_call[2][0] is late_invalid_input,
                late_invalid_input_call[2][1],
            ),
            "late_invalid_input_kwargs": late_invalid_input_call[3],
            "first_invalid_input": first_invalid_input,
            "subclass_result": subclass_result,
            "subclass_calls": tuple(
                (label, tuple(value.__name__ for value in dispatch_types))
                for label, dispatch_types in subclass_calls
            ),
            "invalid_before_shape_override": invalid_before_shape_override,
            "decline_error": decline_error,
            "decline_shape_error": decline_shape_error,
        }

    def test_overrides_and_modes_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_contract(torch),
            self.dispatch_contract(reference_torch),
        )

    def test_disabled_torch_function_shape_handlers_match_pytorch_2_13(self):
        source = r"""
import json
import torch
import torch_rs

class Disabled:
    __torch_function__ = torch._C._disabled_torch_function_impl

def capture(module):
    tensor = module.tensor([1.0, 2.0, 3.0, 4.0])
    cases = (
        lambda: module.reshape(tensor, (Disabled(), 2)),
        lambda: module.reshape(tensor, (1, Disabled())),
        lambda: module.reshape(tensor, Disabled()),
    )
    outcomes = []
    for call in cases:
        try:
            call()
        except Exception as error:
            outcomes.append((type(error).__name__, str(error).splitlines()[0]))
        else:
            outcomes.append(("ok", ""))
    return outcomes

print(json.dumps({"actual": capture(torch_rs), "expected": capture(torch)}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", source],
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        outcomes = json.loads(completed.stdout)
        self.assertEqual(outcomes["actual"], outcomes["expected"])

    def callable_contract(self, module):
        function = module.reshape
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
            "owner_callable_identity": owner.reshape is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("reshape"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["reshape"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_and_exports_match_pytorch_2_13(self):
        actual = self.callable_contract(torch)
        expected = self.callable_contract(reference_torch)
        self.assertTrue(actual.pop("doc").startswith("\nreshape(input, *shape) -> Tensor"))
        expected.pop("doc")
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
