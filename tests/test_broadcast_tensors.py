import copy
import importlib
import inspect
import pickle
import re
import sys
import types
import unittest
from contextlib import ExitStack
from unittest import mock

import numpy as np
import torch_rs as torch


EXACT_TENSORS_ERROR = (
    "broadcast_tensors() only supports exact native Tensor inputs"
)
EXPANSION_ERROR = "torch_rs.broadcast_tensors does not support shape expansion"
FUNCTION_DOC = (
    r"""broadcast_tensors(*tensors) -> List of Tensors

    Broadcasts the given tensors according to :ref:`broadcasting-semantics`.

    Args:
        *tensors: any number of tensors of the same type

    .. warning::

        More than one element of a broadcasted tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensors, please clone them first.

    Example::

        >>> x = torch.arange(3).view(1, 3)
        >>> y = torch.arange(2).view(2, 1)
        >>> a, b = torch.broadcast_tensors(x, y)
        >>> a.size()
        torch.Size([2, 3])
        >>> a
        tensor([[0, 1, 2],
                [0, 1, 2]])
    """
)

if sys.version_info >= (3, 13):
    FUNCTION_DOC = inspect.cleandoc(FUNCTION_DOC) + "\n"


class BroadcastTensorsTests(unittest.TestCase):
    def make_same_shape_sources(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        contiguous = torch.tensor(
            np.arange(12, dtype=np.float32).reshape(3, 4).tolist()
        )
        noncontiguous = torch.tensor(
            np.arange(12, dtype=np.float32).reshape(4, 3).tolist()
        ).transpose(0, 1)
        return base[1], contiguous, noncontiguous

    def test_zero_one_and_many_inputs_return_exact_objects(self):
        self.assertEqual(torch.broadcast_tensors(), ())

        scalar = torch.tensor(-0.0)
        self.assertEqual(torch.broadcast_tensors(scalar), (scalar,))
        self.assertIs(torch.broadcast_tensors(scalar)[0], scalar)

        sources = self.make_same_shape_sources()
        metadata = tuple(
            (
                source.shape,
                source.stride(),
                source.storage_offset(),
                source.data_ptr(),
                source.dtype,
                source.device,
                source.layout,
                source.requires_grad,
                source.is_leaf,
                source.output_nr,
            )
            for source in sources
        )
        result = torch.broadcast_tensors(*sources)

        self.assertIs(type(result), tuple)
        self.assertEqual(len(result), len(sources))
        for item, source, expected_metadata in zip(
            result, sources, metadata, strict=True
        ):
            self.assertIs(item, source)
            self.assertEqual(
                (
                    item.shape,
                    item.stride(),
                    item.storage_offset(),
                    item.data_ptr(),
                    item.dtype,
                    item.device,
                    item.layout,
                    item.requires_grad,
                    item.is_leaf,
                    item.output_nr,
                ),
                expected_metadata,
            )

        repeated = torch.broadcast_tensors(sources[0], sources[0])
        self.assertIs(repeated[0], sources[0])
        self.assertIs(repeated[1], sources[0])

    def test_empty_and_strided_inputs_preserve_all_metadata(self):
        contiguous = torch.zeros((2, 0, 3))
        noncontiguous = torch.zeros((3, 0, 2)).transpose(0, 2)
        offset = torch.zeros((2, 2, 0, 3))[1]
        sources = (contiguous, noncontiguous, offset)

        result = torch.functional.broadcast_tensors(*sources)
        self.assertEqual(result, sources)
        for item, source in zip(result, sources, strict=True):
            self.assertIs(item, source)
            self.assertEqual(item.shape, source.shape)
            self.assertEqual(item.stride(), source.stride())
            self.assertEqual(item.storage_offset(), source.storage_offset())
            self.assertEqual(item.data_ptr(), source.data_ptr())
            self.assertTrue(item.is_set_to(source))

    def test_autograd_history_and_no_grad_state_are_unchanged(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        nonleaf = leaf * 2.0
        view = nonleaf.transpose(0, 1)

        result = torch.broadcast_tensors(leaf, nonleaf, view)
        self.assertEqual(result, (leaf, nonleaf, view))
        self.assertIs(result[0], leaf)
        self.assertIs(result[1], nonleaf)
        self.assertIs(result[2], view)
        self.assertTrue(result[0].is_leaf)
        self.assertFalse(result[1].is_leaf)
        self.assertFalse(result[2].is_leaf)

        result[2].sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])

        with torch.no_grad():
            no_grad_result = torch.broadcast_tensors(nonleaf, view)
        self.assertIs(no_grad_result[0], nonleaf)
        self.assertIs(no_grad_result[1], view)
        self.assertFalse(no_grad_result[0].is_leaf)
        self.assertFalse(no_grad_result[1].is_leaf)

    def test_exact_tensors_dispatch_through_nested_modes(self):
        sources = self.make_same_shape_sources()[:2]
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        accepting = RecordingMode(marker)
        with accepting:
            result = torch.broadcast_tensors(*sources)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [accepting],
            )
        self.assertIs(result, marker)
        self.assertEqual(
            accepting.calls,
            [(torch.broadcast_tensors, (torch.Tensor,), sources, {})],
        )

        calls = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append(
                    (
                        self.label,
                        func,
                        types,
                        args,
                        kwargs,
                        tuple(
                            torch.overrides._get_current_function_mode_stack()
                        ),
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                result = torch.broadcast_tensors(*sources)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(),
                    [lower, upper],
                )

        self.assertEqual([call[0] for call in calls], ["upper", "lower"])
        self.assertTrue(all(call[1] is torch.broadcast_tensors for call in calls))
        self.assertTrue(all(call[2] == (torch.Tensor,) for call in calls))
        self.assertTrue(all(call[3] == sources for call in calls))
        self.assertTrue(all(call[4] == {} for call in calls))
        self.assertEqual(calls[0][5], (lower,))
        self.assertEqual(calls[1][5], ())
        self.assertIs(result[0], sources[0])
        self.assertIs(result[1], sources[1])

    def test_zero_inputs_dispatch_through_the_native_variable_function(self):
        native_function = torch._C.broadcast_tensors
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        accepting = RecordingMode(marker)
        with accepting:
            result = torch.broadcast_tensors()
        self.assertIs(result, marker)
        self.assertEqual(
            accepting.calls,
            [(native_function, (), ((),), None)],
        )

        calls = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                result = torch.broadcast_tensors()
        self.assertEqual(result, ())
        self.assertEqual([call[0] for call in calls], ["upper", "lower"])
        self.assertTrue(all(call[1] is native_function for call in calls))
        self.assertTrue(all(call[2] == () for call in calls))
        self.assertTrue(all(call[3] == ((),) for call in calls))
        self.assertIsNone(calls[0][4])
        self.assertEqual(calls[1][4], {})

        three_argument_calls = []

        class ThreeArgumentMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=()):
                three_argument_calls.append((func, types, args))
                return marker

        with ThreeArgumentMode():
            result = torch.broadcast_tensors()
        self.assertIs(result, marker)
        self.assertEqual(
            three_argument_calls,
            [(native_function, (), ((),))],
        )

    def test_declining_and_raising_modes_restore_the_stack(self):
        sources = self.make_same_shape_sources()[:2]

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return NotImplemented

        declining = DecliningMode()
        with declining:
            with self.assertRaisesRegex(
                TypeError,
                "^no implementation found for "
                "'torch_rs\\.functional\\.broadcast_tensors' on types that "
                "implement __torch_function__: \\[\\] nor in mode ",
            ):
                torch.broadcast_tensors(*sources)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [declining],
            )
        self.assertEqual(
            declining.calls,
            [
                (torch.broadcast_tensors, (torch.Tensor,), sources, {}),
                (torch.broadcast_tensors, (), sources, {}),
            ],
        )

        expected_error = ValueError("mode failed")

        class RaisingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise expected_error

        raising = RaisingMode()
        with raising:
            with self.assertRaises(ValueError) as raised:
                torch.broadcast_tensors(*sources)
            self.assertIs(raised.exception, expected_error)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [raising],
            )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_unsupported_inputs_fail_before_dispatch_or_tensor_allocation(self):
        left = torch.tensor([[1.0], [2.0]], requires_grad=True)
        right = torch.tensor([[3.0, 4.0]], requires_grad=True)
        left_before = (left.tolist(), left.stride(), left.storage_offset(), left.grad)
        right_before = (
            right.tolist(),
            right.stride(),
            right.storage_offset(),
            right.grad,
        )

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        invalid_calls = (
            lambda: torch.broadcast_tensors(None),
            lambda: torch.broadcast_tensors(left, 1),
            lambda: torch.broadcast_tensors(Override(), left),
            lambda: torch.broadcast_tensors(left, right, Override()),
            lambda: torch.broadcast_tensors((left, right)),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaisesRegex(
                TypeError, f"^{re.escape(EXACT_TENSORS_ERROR)}$"
            ):
                call()
        self.assertEqual(Override.calls, [])

        expansion_calls = (
            lambda: torch.broadcast_tensors(left, right),
            lambda: torch.broadcast_tensors(torch.tensor(1.0), left),
            lambda: torch.broadcast_tensors(
                torch.zeros((2, 3)), torch.zeros((3, 2))
            ),
        )
        with ExitStack() as stack:
            factories = [
                stack.enter_context(
                    mock.patch.object(
                        torch,
                        name,
                        side_effect=AssertionError(f"{name} must not be called"),
                        create=True,
                    )
                )
                for name in (
                    "broadcast_shapes",
                    "empty",
                    "empty_like",
                    "full_like",
                    "ones",
                    "ones_like",
                    "zeros",
                    "zeros_like",
                )
            ]
            for call in expansion_calls[:2]:
                with self.subTest(call=call), self.assertRaisesRegex(
                    NotImplementedError, f"^{re.escape(EXPANSION_ERROR)}$"
                ):
                    call()
        for factory in factories:
            factory.assert_not_called()
        with self.assertRaisesRegex(
            NotImplementedError, f"^{re.escape(EXPANSION_ERROR)}$"
        ):
            expansion_calls[2]()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        mode = RecordingMode()
        for call, error_type, message in (
            (lambda: torch.broadcast_tensors(left, right), NotImplementedError, EXPANSION_ERROR),
            (
                lambda: torch.broadcast_tensors(left, right, Override()),
                TypeError,
                EXACT_TENSORS_ERROR,
            ),
        ):
            with self.subTest(error_type=error_type), mode, self.assertRaisesRegex(
                error_type, f"^{re.escape(message)}$"
            ):
                call()
        self.assertEqual(mode.calls, [])
        self.assertEqual(
            (left.tolist(), left.stride(), left.storage_offset(), left.grad),
            left_before,
        )
        self.assertEqual(
            (right.tolist(), right.stride(), right.storage_offset(), right.grad),
            right_before,
        )

        with self.assertRaisesRegex(
            TypeError,
            "^broadcast_tensors\\(\\) got an unexpected keyword argument "
            "'tensors'$",
        ):
            torch.broadcast_tensors(tensors=(left,))

    def test_function_metadata_exports_and_pickle(self):
        functional = importlib.import_module("torch_rs.functional")
        function = torch.broadcast_tensors

        self.assertIs(torch.functional, functional)
        self.assertIs(function, functional.broadcast_tensors)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "broadcast_tensors")
        self.assertEqual(function.__qualname__, "broadcast_tensors")
        self.assertEqual(function.__module__, "torch_rs.functional")
        self.assertEqual(str(inspect.signature(function)), "(*tensors)")
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(functional.__all__.count("broadcast_tensors"), 1)
        self.assertEqual(torch.__all__.count("broadcast_tensors"), 1)

        functional_namespace = {}
        exec("from torch_rs.functional import *", functional_namespace)
        self.assertIs(functional_namespace["broadcast_tensors"], function)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertIs(top_level_namespace["broadcast_tensors"], function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)


if __name__ == "__main__":
    unittest.main()
