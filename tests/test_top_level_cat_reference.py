import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelCatReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.cat differentials require pinned PyTorch 2.13.0"
            )

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
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    @staticmethod
    def fresh_storage_observation(result, inputs):
        return tuple(
            (not result.is_set_to(input), result.data_ptr() != input.data_ptr())
            for input in inputs
            if input.numel()
        )

    def test_list_tuple_empty_operands_order_metadata_and_storage_match_pytorch_2_13(
        self,
    ):
        actual_base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected_base = reference_torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=reference_torch.float32,
        )
        actual_offset = actual_base.transpose(0, 1)[1]
        expected_offset = expected_base.transpose(0, 1)[1]

        cases = (
            (
                "list dim 0",
                [actual_offset, torch.tensor([]), torch.tensor([-0.0, 7.5])],
                [
                    expected_offset,
                    reference_torch.tensor([], dtype=reference_torch.float32),
                    reference_torch.tensor([-0.0, 7.5], dtype=reference_torch.float32),
                ],
                0,
            ),
            (
                "tuple dim -1",
                (torch.tensor([]), torch.tensor([3.25]), torch.tensor([])),
                (
                    reference_torch.tensor([], dtype=reference_torch.float32),
                    reference_torch.tensor([3.25], dtype=reference_torch.float32),
                    reference_torch.tensor([], dtype=reference_torch.float32),
                ),
                -1,
            ),
            (
                "single input",
                [torch.tensor([11.0, 13.0])],
                [reference_torch.tensor([11.0, 13.0], dtype=reference_torch.float32)],
                0,
            ),
            (
                "keywords",
                [torch.tensor([8.0]), torch.tensor([]), torch.tensor([9.0])],
                [
                    reference_torch.tensor([8.0], dtype=reference_torch.float32),
                    reference_torch.tensor([], dtype=reference_torch.float32),
                    reference_torch.tensor([9.0], dtype=reference_torch.float32),
                ],
                0,
            ),
            (
                "axis alias",
                [torch.tensor([14.0]), torch.tensor([15.0])],
                [
                    reference_torch.tensor([14.0], dtype=reference_torch.float32),
                    reference_torch.tensor([15.0], dtype=reference_torch.float32),
                ],
                0,
            ),
        )
        for case, actual_inputs, expected_inputs, dimension in cases:
            with self.subTest(case=case):
                if case == "keywords":
                    actual = torch.cat(tensors=actual_inputs, dim=dimension, out=None)
                    expected = reference_torch.cat(
                        tensors=expected_inputs, dim=dimension, out=None
                    )
                elif case == "axis alias":
                    actual = torch.cat(actual_inputs, axis=dimension)
                    expected = reference_torch.cat(expected_inputs, axis=dimension)
                else:
                    actual = torch.cat(actual_inputs, dim=dimension)
                    expected = reference_torch.cat(expected_inputs, dim=dimension)
                self.assert_matches(actual, expected, case=case)
                self.assertEqual(
                    self.fresh_storage_observation(actual, actual_inputs),
                    self.fresh_storage_observation(expected, expected_inputs),
                )

    def test_concat_alias_values_metadata_storage_and_exports_match_pytorch_2_13(
        self,
    ):
        for name in ("concat", "concatenate"):
            actual_function = getattr(torch, name)
            expected_function = getattr(reference_torch, name)
            actual_base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            expected_base = reference_torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=reference_torch.float32,
            )
            actual_offset = actual_base.transpose(0, 1)[1]
            expected_offset = expected_base.transpose(0, 1)[1]

            cases = (
                (
                    "list dim 0",
                    [actual_offset, torch.tensor([]), torch.tensor([-0.0, 7.5])],
                    [
                        expected_offset,
                        reference_torch.tensor([], dtype=reference_torch.float32),
                        reference_torch.tensor([-0.0, 7.5], dtype=reference_torch.float32),
                    ],
                    0,
                ),
                (
                    "tuple dim -1",
                    (torch.tensor([]), torch.tensor([3.25]), torch.tensor([])),
                    (
                        reference_torch.tensor([], dtype=reference_torch.float32),
                        reference_torch.tensor([3.25], dtype=reference_torch.float32),
                        reference_torch.tensor([], dtype=reference_torch.float32),
                    ),
                    -1,
                ),
                (
                    "keywords",
                    [torch.tensor([8.0]), torch.tensor([]), torch.tensor([9.0])],
                    [
                        reference_torch.tensor([8.0], dtype=reference_torch.float32),
                        reference_torch.tensor([], dtype=reference_torch.float32),
                        reference_torch.tensor([9.0], dtype=reference_torch.float32),
                    ],
                    0,
                ),
                (
                    "axis alias",
                    [torch.tensor([14.0]), torch.tensor([15.0])],
                    [
                        reference_torch.tensor([14.0], dtype=reference_torch.float32),
                        reference_torch.tensor([15.0], dtype=reference_torch.float32),
                    ],
                    0,
                ),
            )
            for case, actual_inputs, expected_inputs, dimension in cases:
                with self.subTest(alias=name, case=case):
                    if case == "keywords":
                        actual = actual_function(
                            tensors=actual_inputs,
                            dim=dimension,
                            out=None,
                        )
                        expected = expected_function(
                            tensors=expected_inputs,
                            dim=dimension,
                            out=None,
                        )
                    elif case == "axis alias":
                        actual = actual_function(actual_inputs, axis=dimension)
                        expected = expected_function(expected_inputs, axis=dimension)
                    else:
                        actual = actual_function(actual_inputs, dim=dimension)
                        expected = expected_function(expected_inputs, dim=dimension)
                    self.assert_matches(actual, expected, case=f"{name} {case}")
                    self.assertEqual(
                        self.fresh_storage_observation(actual, actual_inputs),
                        self.fresh_storage_observation(expected, expected_inputs),
                    )

        def public_surface(module):
            return (
                tuple(
                    (
                        name,
                        hasattr(module, name),
                        getattr(module, name).__name__,
                        getattr(module, name).__qualname__,
                        getattr(module, name).__module__,
                        getattr(module._C._VariableFunctionsClass, name)
                        is getattr(module, name),
                        module.__all__.count(name),
                    )
                    for name in ("cat", "concat", "concatenate")
                ),
                module.cat is module.concat,
                module.cat is module.concatenate,
                module.concat is module.concatenate,
            )

        self.assertEqual(public_surface(torch), public_surface(reference_torch))

    def test_no_grad_grad_requiring_operands_match_pytorch_2_13(self):
        actual_left = torch.tensor([1.0, 2.0], requires_grad=True)
        actual_right = torch.tensor([3.0], requires_grad=True)
        expected_left = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32, requires_grad=True
        )
        expected_right = reference_torch.tensor(
            [3.0], dtype=reference_torch.float32, requires_grad=True
        )

        with torch.no_grad():
            actual = torch.cat([actual_left, actual_right], dim=-1)
        with reference_torch.no_grad():
            expected = reference_torch.cat([expected_left, expected_right], dim=-1)

        self.assert_matches(actual, expected, case="no_grad")
        self.assertIsNone(actual_left.grad)
        self.assertIsNone(actual_right.grad)
        self.assertIsNone(expected_left.grad)
        self.assertIsNone(expected_right.grad)

    def test_axis_dim_conflicts_match_pytorch_2_13(self):
        actual = [torch.tensor([1.0])]
        expected = [reference_torch.tensor([1.0], dtype=reference_torch.float32)]
        cases = (
            (
                "dim keyword",
                lambda: torch.cat(actual, dim=0, axis=0),
                lambda: reference_torch.cat(expected, dim=0, axis=0),
            ),
            (
                "dim positional",
                lambda: torch.cat(actual, 0, axis=0),
                lambda: reference_torch.cat(expected, 0, axis=0),
            ),
        )
        for case, actual_call, expected_call in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^cat\(\) got an unexpected keyword argument 'axis'$",
                ):
                    actual_call()
                with self.assertRaisesRegex(
                    TypeError,
                    r"^cat\(\) got an unexpected keyword argument 'axis'$",
                ):
                    expected_call()

    def dispatch_observation(self, module, function_name):
        function = getattr(module, function_name)
        left = module.tensor([1.0])
        right = module.tensor([2.0])
        inputs = [left, right]
        marker = object()

        def stack_labels():
            return tuple(
                getattr(mode, "label", type(mode).__name__)
                for mode in module.overrides._get_current_function_mode_stack()
            )

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs, stack_labels()))
                return self.result

        accepting = RecordingMode(marker)
        with accepting:
            accepted = function(inputs, dim=0)
        func, dispatch_types, args, kwargs, stack = accepting.calls[0]
        mode_observation = (
            accepted is marker,
            func is function,
            tuple(item.__name__ for item in dispatch_types),
            args[0] is inputs,
            tuple(kwargs),
            kwargs["dim"],
            stack,
        )

        calls = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append(
                    (
                        self.label,
                        func is function,
                        tuple(item.__name__ for item in types),
                        args[0] is inputs,
                        tuple(kwargs),
                        kwargs["dim"],
                        stack_labels(),
                    )
                )
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(inputs, dim=0)

        class LeftOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("left", func, types, args, kwargs))
                return NotImplemented

        class RightOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("right", func, types, args, kwargs))
                return marker

        events = []
        override_inputs = [LeftOverride(), module.tensor([]), RightOverride()]
        override_result = function(override_inputs, dim=0)
        override_observation = (
            override_result is marker,
            tuple(event[0] for event in events),
            tuple(
                (
                    event[1] is function,
                    tuple(item.__name__ for item in event[2]),
                    event[3][0] is override_inputs,
                    tuple(event[4]),
                    event[4]["dim"],
                )
                for event in events
            ),
        )

        class DimensionOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                argument_events.append(("dim", func, types, args, kwargs))
                return marker

        class OutOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                argument_events.append(("out", func, types, args, kwargs))
                return marker

        class TensorsOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                argument_events.append(("tensors", func, types, args, kwargs))
                return marker

        argument_events = []
        dimension = DimensionOverride()
        out = OutOverride()
        tensors = TensorsOverride()
        dim_result = function([left], dim=dimension)
        out_result = function([left], out=out)
        tensors_result = function(tensors)
        argument_observation = (
            dim_result is marker,
            out_result is marker,
            tensors_result is marker,
            tuple(
                (
                    event[0],
                    event[1] is function,
                    tuple(item.__name__ for item in event[2]),
                    len(event[3]),
                    None if event[4] is None else tuple(event[4]),
                )
                for event in argument_events
            ),
        )

        declining_mode = RecordingMode(NotImplemented)
        try:
            with declining_mode:
                function(inputs, dim=0)
        except Exception as error:
            mode_decline = (
                type(error).__name__,
                str(error).split("\n\n", maxsplit=1)[0],
                len(declining_mode.calls),
            )
        else:
            mode_decline = ("accepted", None, len(declining_mode.calls))

        class DecliningOverride:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return NotImplemented

        try:
            function([DecliningOverride()], dim=0)
        except Exception as error:
            override_decline = (
                type(error).__name__,
                str(error).split("\n\n", maxsplit=1)[0],
                DecliningOverride.calls,
            )
        else:
            override_decline = ("accepted", None, DecliningOverride.calls)

        return (
            mode_observation,
            tuple(calls),
            tuple(forwarded.tolist()),
            override_observation,
            argument_observation,
            mode_decline,
            override_decline,
            stack_labels(),
        )

    def test_torch_function_dispatch_matches_pytorch_2_13(self):
        for name in ("cat", "concat", "concatenate"):
            with self.subTest(function=name):
                self.assertEqual(
                    self.dispatch_observation(torch, name),
                    self.dispatch_observation(reference_torch, name),
                )


if __name__ == "__main__":
    unittest.main()
