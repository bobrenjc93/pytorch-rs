import enum
import inspect
import pickle
import re
import subprocess
import sys
import textwrap
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nmovedim(source, destination) -> Tensor\n\n"
    "See :func:`torch.movedim`\n"
)
FUNCTION_DOC = (
    "\nmovedim(input, source, destination) -> Tensor\n\n"
    "Moves the dimension(s) of :attr:`input` at the position(s) in :attr:`source`\n"
    "to the position(s) in :attr:`destination`.\n\n"
    "Other dimensions of :attr:`input` that are not explicitly moved remain in\n"
    "their original order and appear at the positions not specified in :attr:`destination`.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n"
    "    source (int or tuple of ints): Original positions of the dims to move. These must be unique.\n"
    "    destination (int or tuple of ints): Destination positions for each of the original dims. These must also be unique.\n\n"
    "Examples::\n\n"
    "    >>> t = torch.randn(3,2,1)\n"
    "    >>> t\n"
    "    tensor([[[-0.3362],\n"
    "            [-0.8437]],\n\n"
    "            [[-0.9627],\n"
    "            [ 0.1727]],\n\n"
    "            [[ 0.5173],\n"
    "            [-0.1398]]])\n"
    "    >>> torch.movedim(t, 1, 0).shape\n"
    "    torch.Size([2, 3, 1])\n"
    "    >>> torch.movedim(t, 1, 0)\n"
    "    tensor([[[-0.3362],\n"
    "            [-0.9627],\n"
    "            [ 0.5173]],\n\n"
    "            [[-0.8437],\n"
    "            [ 0.1727],\n"
    "            [-0.1398]]])\n"
    "    >>> torch.movedim(t, (1, 2), (0, 1)).shape\n"
    "    torch.Size([2, 1, 3])\n"
    "    >>> torch.movedim(t, (1, 2), (0, 1))\n"
    "    tensor([[[-0.3362, -0.9627,  0.5173]],\n\n"
    "            [[-0.8437,  0.1727, -0.1398]]])\n"
)
OVERLOADS = (
    "but expected one of:\n"
    " * (int source, int destination)\n"
    " * (tuple of ints source, tuple of ints destination)\n"
)
TOP_LEVEL_OVERLOADS = (
    "but expected one of:\n"
    " * (Tensor input, int source, int destination)\n"
    " * (Tensor input, tuple of ints source, tuple of ints destination)\n"
)


class Dimension(enum.IntEnum):
    FIRST = 0
    LAST = -1


class IntegerSubclass(int):
    pass


class CustomIndex:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return 0


class TensorMovedimTests(unittest.TestCase):
    def assert_values(self, tensor, expected):
        np.testing.assert_array_equal(np.asarray(tensor), np.asarray(expected))

    def test_integer_axes_move_through_shared_storage_permutations(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        tensor = torch.tensor(values.tolist())
        original_metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
        )

        case = 0
        for source_axis in range(values.ndim):
            for destination_axis in range(values.ndim):
                source = source_axis if case % 2 == 0 else source_axis - values.ndim
                destination = (
                    destination_axis
                    if case % 3 == 0
                    else destination_axis - values.ndim
                )
                if case % 3 == 0:
                    moved = tensor.movedim(source, destination)
                elif case % 3 == 1:
                    moved = tensor.movedim(
                        source=source, destination=destination
                    )
                else:
                    moved = tensor.movedim(
                        destination=destination, source=source
                    )

                order = [axis for axis in range(values.ndim) if axis != source_axis]
                order.insert(destination_axis, source_axis)
                with self.subTest(source=source, destination=destination):
                    self.assertEqual(
                        moved.shape,
                        np.moveaxis(values, source_axis, destination_axis).shape,
                    )
                    self.assertEqual(
                        moved.stride(),
                        tuple(tensor.stride()[axis] for axis in order),
                    )
                    self.assertEqual(
                        moved.storage_offset(), tensor.storage_offset()
                    )
                    self.assertEqual(moved.data_ptr(), tensor.data_ptr())
                    self.assertIs(moved.dtype, torch.float32)
                    self.assertEqual(moved.device, torch.device("cpu"))
                    self.assert_values(
                        moved,
                        np.moveaxis(values, source_axis, destination_axis),
                    )
                case += 1

        self.assertEqual(
            (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
            ),
            original_metadata,
        )

    def test_scalar_empty_offset_and_strided_views_preserve_metadata(self):
        scalar = torch.tensor(2.5)
        for source in (0, -1):
            for destination in (0, -1):
                with self.subTest(kind="scalar", source=source, destination=destination):
                    moved = scalar.movedim(source, destination)
                    self.assertIsNot(moved, scalar)
                    self.assertEqual(moved.shape, ())
                    self.assertEqual(moved.stride(), ())
                    self.assertEqual(moved.storage_offset(), 0)
                    self.assertEqual(moved.data_ptr(), scalar.data_ptr())
                    self.assertEqual(moved.item(), 2.5)

        empty = torch.zeros((2, 0, 3))
        moved_empty = empty.movedim(source=-1, destination=0)
        self.assertEqual(moved_empty.shape, (3, 2, 0))
        self.assertEqual(
            moved_empty.stride(),
            (empty.stride()[2], empty.stride()[0], empty.stride()[1]),
        )
        self.assertEqual(moved_empty.storage_offset(), empty.storage_offset())
        self.assertEqual(moved_empty.data_ptr(), empty.data_ptr())
        self.assertEqual(moved_empty.numel(), 0)

        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        source = base.transpose(0, 2)[1]
        self.assertGreater(source.storage_offset(), 0)
        self.assertFalse(source.is_contiguous())
        moved = source.movedim(-1, 0)
        self.assertEqual(moved.shape, (2, 3))
        self.assertEqual(
            moved.stride(), (source.stride()[1], source.stride()[0])
        )
        self.assertEqual(moved.storage_offset(), source.storage_offset())
        self.assertEqual(moved.data_ptr(), source.data_ptr())
        self.assert_values(moved, np.moveaxis(values.transpose(2, 1, 0)[1], -1, 0))

    def test_extreme_empty_metadata_uses_permutation_overflow_checks(self):
        tensor = torch.zeros((sys.maxsize, 0, 2, 2))
        moved = tensor.movedim(0, 2)
        order = (1, 2, 0, 3)
        self.assertEqual(moved.shape, (0, 2, sys.maxsize, 2))
        self.assertEqual(
            moved.stride(), tuple(tensor.stride()[axis] for axis in order)
        )
        self.assertEqual(moved.data_ptr(), tensor.data_ptr())

        with self.assertRaises(RuntimeError) as raised:
            tensor.movedim(source=1, destination=3)
        self.assertEqual(
            str(raised.exception), "numel: integer multiplication overflow"
        )

    def test_autograd_empty_backward_and_no_grad_match_view_policy(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(4, 2, 3)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        moved = leaf.movedim(-1, 0)

        self.assertTrue(moved.requires_grad)
        self.assertFalse(moved.is_leaf)
        self.assertEqual(moved.data_ptr(), leaf.data_ptr())
        (moved * torch.tensor(weights.tolist())).sum().backward()
        self.assert_values(leaf.grad, np.moveaxis(weights, 0, -1))

        identity_leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        identity = identity_leaf.movedim(1, 1)
        identity_weights = np.array([[2.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        (identity * torch.tensor(identity_weights.tolist())).sum().backward()
        self.assert_values(identity_leaf.grad, identity_weights)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty.movedim(0, -1).sum().backward()
        self.assert_values(empty.grad, np.zeros((2, 0, 3), dtype=np.float32))

        with torch.no_grad():
            untracked = leaf.movedim(source=0, destination=1)
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.data_ptr(), leaf.data_ptr())
        self.assertEqual(untracked.shape, (3, 2, 4))
        self.assertEqual(untracked.stride(), (4, 12, 1))

    def test_python_and_numpy_integer_scalars_and_conversion_order(self):
        tensor = torch.zeros((2, 3, 4))
        cases = (
            (0, 2, (3, 4, 2)),
            (IntegerSubclass(1), IntegerSubclass(0), (3, 2, 4)),
            (Dimension.FIRST, Dimension.LAST, (3, 4, 2)),
            (np.int8(-1), np.int32(0), (4, 2, 3)),
            (np.uint64(2), np.int64(1), (2, 4, 3)),
        )
        for source, destination, shape in cases:
            with self.subTest(source=repr(source), destination=repr(destination)):
                self.assertEqual(tensor.movedim(source, destination).shape, shape)

        state = {"destination_converted": False, "calls": []}

        class StatefulInteger(np.int64):
            def __new__(cls, role):
                value = np.int64.__new__(cls, 0)
                value.role = role
                return value

            def __index__(self):
                state["calls"].append(self.role)
                if self.role == "destination":
                    state["destination_converted"] = True
                    return 1
                return 0 if state["destination_converted"] else 2

        moved = tensor.movedim(
            StatefulInteger("source"), StatefulInteger("destination")
        )
        self.assertEqual(state["calls"], ["destination", "source"])
        self.assertEqual(moved.shape, (3, 2, 4))
        self.assertEqual(moved.stride(), (4, 12, 1))

    def test_integer_binding_overflow_and_dimension_errors(self):
        tensor = torch.zeros((2, 3, 4))
        custom = CustomIndex()
        invalid_types = (
            (True, 0, "bool", "int"),
            (np.bool_(False), 0, "numpy.bool", "int"),
            (custom, 0, "CustomIndex", "int"),
            (1.5, 0, "float", "int"),
            (0, "1", "int", "str"),
        )
        for source, destination, source_type, destination_type in invalid_types:
            with self.subTest(source=repr(source), destination=repr(destination)):
                with self.assertRaises(TypeError) as raised:
                    tensor.movedim(source, destination)
                self.assertEqual(
                    str(raised.exception),
                    "movedim() received an invalid combination of arguments - got "
                    f"({source_type}, {destination_type}), but expected one of:\n"
                    " * (int source, int destination)\n"
                    "      didn't match because some of the arguments have invalid "
                    f"types: ({'!' if source_type != 'int' else ''}{source_type}"
                    f"{'!' if source_type != 'int' else ''}, "
                    f"{'!' if destination_type != 'int' else ''}{destination_type}"
                    f"{'!' if destination_type != 'int' else ''})\n"
                    " * (tuple of ints source, tuple of ints destination)\n"
                    "      didn't match because some of the arguments have invalid "
                    f"types: (!{source_type}!, !{destination_type}!)\n",
                )
        self.assertEqual(custom.calls, 0)

        for value in (2**63, -(2**63) - 1, np.uint64(2**63)):
            for source_first in (False, True):
                with self.subTest(value=repr(value), source_first=source_first):
                    with self.assertRaises(ValueError) as raised:
                        if source_first:
                            tensor.movedim(value, 0)
                        else:
                            tensor.movedim(0, value)
                    self.assertEqual(
                        str(raised.exception), "Overflow when unpacking long long"
                    )

        for source, destination, invalid in (
            (3, 0, 3),
            (-4, 0, -4),
            (0, 3, 3),
            (0, -4, -4),
        ):
            with self.subTest(source=source, destination=destination):
                with self.assertRaises(IndexError) as raised:
                    tensor.movedim(source, destination)
                self.assertEqual(
                    str(raised.exception),
                    "Dimension out of range (expected to be in range of "
                    f"[-3, 2], but got {invalid})",
                )

        scalar = torch.tensor(1.0)
        with self.assertRaisesRegex(
            IndexError,
            r"^Dimension out of range \(expected to be in range of "
            r"\[-1, 0\], but got -2\)$",
        ):
            scalar.movedim(-2, 0)

    def test_binding_errors_match_the_integer_overload_shape(self):
        tensor = torch.zeros((2, 3, 4))
        cases = (
            (
                lambda: tensor.movedim(),
                f"movedim() received an invalid combination of arguments - got (), {OVERLOADS}",
            ),
            (
                lambda: tensor.movedim(0),
                "movedim() received an invalid combination of arguments - got "
                f"(int), {OVERLOADS}",
            ),
            (
                lambda: tensor.movedim(0, 1, 2),
                "movedim() received an invalid combination of arguments - got "
                f"(int, int, int), {OVERLOADS}",
            ),
            (
                lambda: tensor.movedim(source=0),
                "movedim() received an invalid combination of arguments - got "
                f"(source=int, ), {OVERLOADS}",
            ),
            (
                lambda: tensor.movedim(0, source=1),
                "movedim() received an invalid combination of arguments - got "
                "(int, source=int), but expected one of:\n"
                " * (int source, int destination)\n"
                "      didn't match because some of the keywords were incorrect: source\n"
                " * (tuple of ints source, tuple of ints destination)\n"
                "      didn't match because some of the keywords were incorrect: source\n",
            ),
            (
                lambda: tensor.movedim(0, 1, extra=True),
                "movedim() received an invalid combination of arguments - got "
                f"(int, int, extra=bool), {OVERLOADS}",
            ),
        )
        for call, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), expected)

        with self.assertRaises(TypeError) as raised:
            tensor.movedim(0, **{"bad\0tail": 1})
        self.assertEqual(
            str(raised.exception),
            "movedim() received an invalid combination of arguments - got (int, bad",
        )

    def test_keyword_subclasses_cannot_panic_or_misbind(self):
        tensor = torch.zeros((2, 3, 4))

        class PlainKeyword(str):
            pass

        class TrueKeyword(str):
            def __eq__(self, other):
                return True

            __hash__ = str.__hash__

        class FalseKeyword(str):
            def __eq__(self, other):
                return False

            __hash__ = str.__hash__

        class RaisingKeyword(str):
            def __eq__(self, other):
                raise RuntimeError("keyword equality failure")

            __hash__ = str.__hash__

        class MismatchedHashKeyword(str):
            def __eq__(self, other):
                return True

            def __hash__(self):
                return 0

        for keyword_type in (PlainKeyword, TrueKeyword):
            with self.subTest(keyword_type=keyword_type.__name__, outcome="accepted"):
                moved = tensor.movedim(
                    **{keyword_type("source"): 0, "destination": 1}
                )
                self.assertEqual(moved.shape, (3, 2, 4))
                self.assertEqual(moved.stride(), (4, 12, 1))

        for keyword_type in (
            FalseKeyword,
            RaisingKeyword,
            MismatchedHashKeyword,
        ):
            with self.subTest(keyword_type=keyword_type.__name__, outcome="rejected"):
                with self.assertRaises(TypeError) as raised:
                    tensor.movedim(
                        **{keyword_type("source"): 0, "destination": 1}
                    )
                self.assertIn(
                    "movedim() received an invalid combination of arguments",
                    str(raised.exception),
                )

        with self.assertRaises(TypeError):
            tensor.movedim(**{FalseKeyword("unknown"): 0, "other": 1})

    def test_spoofed_numpy_integer_class_is_rejected_without_indexing(self):
        tensor = torch.zeros((2, 3, 4))
        calls = []

        class SpoofedInteger:
            @property
            def __class__(self):
                return np.int64

            def __index__(self):
                calls.append("index")
                return 0

        for source, destination in (
            (SpoofedInteger(), 2),
            (0, SpoofedInteger()),
        ):
            with self.subTest(source=type(source).__name__, destination=type(destination).__name__):
                with self.assertRaises(TypeError):
                    tensor.movedim(source, destination)
        self.assertEqual(calls, [])

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux RLIMIT_AS")
    def test_large_keyword_error_returns_bad_alloc_instead_of_aborting(self):
        script = textwrap.dedent(
            """\
            import os
            import resource

            import torch_rs as torch

            tensor = torch.zeros((2, 3, 4))
            keywords = {
                "a" * (1024 * 1024): 0,
                "b" * (1024 * 1024): 1,
            }
            with open("/proc/self/statm", encoding="ascii") as statm:
                virtual_pages = int(statm.read().split()[0])
            current_virtual_size = virtual_pages * os.sysconf("SC_PAGE_SIZE")
            limit = current_virtual_size + 4 * 1024 * 1024
            _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if hard_limit != resource.RLIM_INFINITY and limit > hard_limit:
                raise SystemExit(77)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard_limit))

            try:
                tensor.movedim(**keywords)
            except RuntimeError as error:
                assert str(error) == "std::bad_alloc", repr(error)
            else:
                raise AssertionError("the constrained call unexpectedly succeeded")
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        if completed.returncode == 77:
            self.skipTest("process hard address-space limit is too low")
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_singleton_sequence_dimensions_reuse_the_integer_view_engine(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        tensors = (
            ("scalar", torch.tensor(2.5), -1, 0),
            ("empty", torch.zeros((2, 0, 3)), -1, 0),
            ("offset", base.transpose(0, 2)[1], -1, 0),
            ("noncontiguous", base.transpose(0, 2), 0, -1),
        )
        operations = (
            (
                "Tensor.movedim",
                lambda tensor, source, destination: tensor.movedim(
                    source, destination
                ),
            ),
            (
                "Tensor.moveaxis",
                lambda tensor, source, destination: tensor.moveaxis(
                    source, destination
                ),
            ),
            (
                "torch.movedim",
                lambda tensor, source, destination: torch.movedim(
                    tensor, source, destination
                ),
            ),
            (
                "torch.moveaxis",
                lambda tensor, source, destination: torch.moveaxis(
                    tensor, source, destination
                ),
            ),
        )
        sequence_forms = (
            ("tuple", lambda dimension: (dimension,)),
            ("list", lambda dimension: [dimension]),
            ("Size", lambda dimension: torch.Size([dimension])),
        )

        for case, tensor, source, destination in tensors:
            expected = tensor.movedim(source, destination)
            for operation_name, operation in operations:
                for source_name, source_form in sequence_forms:
                    for destination_name, destination_form in sequence_forms:
                        with self.subTest(
                            case=case,
                            operation=operation_name,
                            source=source_name,
                            destination=destination_name,
                        ):
                            moved = operation(
                                tensor,
                                source_form(source),
                                destination_form(destination),
                            )
                            self.assertEqual(moved.shape, expected.shape)
                            self.assertEqual(moved.stride(), expected.stride())
                            self.assertEqual(
                                moved.storage_offset(), expected.storage_offset()
                            )
                            self.assertEqual(moved.data_ptr(), tensor.data_ptr())
                            self.assert_values(moved, expected)

        source = base.transpose(0, 2)[1]
        expected = source.movedim(-1, 0)
        keyword_results = (
            source.movedim(source=(-1,), destination=[0]),
            source.moveaxis(source=[-1], destination=torch.Size([0])),
            torch.movedim(
                input=source,
                source=torch.Size([-1]),
                destination=(0,),
            ),
            torch.moveaxis(
                input=source,
                source=(-1,),
                destination=torch.Size([0]),
            ),
        )
        for moved in keyword_results:
            self.assertEqual(moved.shape, expected.shape)
            self.assertEqual(moved.stride(), expected.stride())
            self.assertEqual(moved.storage_offset(), expected.storage_offset())
            self.assertEqual(moved.data_ptr(), source.data_ptr())
            self.assert_values(moved, expected)

    def test_singleton_sequence_autograd_empty_and_no_grad_reuse_view_policy(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(
            4, 2, 3
        )
        operations = (
            (
                "Tensor.movedim",
                lambda tensor, source, destination: tensor.movedim(
                    source, destination
                ),
            ),
            (
                "Tensor.moveaxis",
                lambda tensor, source, destination: tensor.moveaxis(
                    source, destination
                ),
            ),
            (
                "torch.movedim",
                lambda tensor, source, destination: torch.movedim(
                    tensor, source, destination
                ),
            ),
            (
                "torch.moveaxis",
                lambda tensor, source, destination: torch.moveaxis(
                    tensor, source, destination
                ),
            ),
        )

        for operation_name, operation in operations:
            with self.subTest(operation=operation_name):
                leaf = torch.tensor(values.tolist(), requires_grad=True)
                moved = operation(leaf, (-1,), [0])
                self.assertTrue(moved.requires_grad)
                self.assertFalse(moved.is_leaf)
                self.assertEqual(moved.data_ptr(), leaf.data_ptr())
                (moved * torch.tensor(weights.tolist())).sum().backward()
                self.assert_values(leaf.grad, np.moveaxis(weights, 0, -1))

                empty = torch.zeros((2, 0, 3), requires_grad=True)
                operation(empty, torch.Size([0]), (-1,)).sum().backward()
                self.assert_values(
                    empty.grad, np.zeros((2, 0, 3), dtype=np.float32)
                )

                with torch.no_grad():
                    untracked = operation(leaf, [0], torch.Size([1]))
                self.assertTrue(untracked.requires_grad)
                self.assertTrue(untracked.is_leaf)
                self.assertEqual(untracked.data_ptr(), leaf.data_ptr())
                self.assertEqual(untracked.shape, (3, 2, 4))
                self.assertEqual(untracked.stride(), (4, 12, 1))

    def test_singleton_sequence_index_protocol_conversion_and_mode_dispatch(self):
        tensor = torch.zeros((2, 3, 4))
        operations = (
            (
                "Tensor.movedim",
                lambda source, destination: tensor.movedim(
                    source, destination
                ),
            ),
            (
                "Tensor.moveaxis",
                lambda source, destination: tensor.moveaxis(
                    source, destination
                ),
            ),
            (
                "torch.movedim",
                lambda source, destination: torch.movedim(
                    tensor, source, destination
                ),
            ),
            (
                "torch.moveaxis",
                lambda source, destination: torch.moveaxis(
                    tensor, source, destination
                ),
            ),
        )

        for operation_name, operation in operations:
            state = {"destination_converted": False, "calls": []}

            class StatefulIndex:
                def __init__(self, role):
                    self.role = role

                def __index__(self):
                    state["calls"].append(self.role)
                    if self.role == "destination":
                        state["destination_converted"] = True
                        return 1
                    return 0 if state["destination_converted"] else 2

            moved = operation(
                (StatefulIndex("source"),),
                [StatefulIndex("destination")],
            )
            with self.subTest(operation=operation_name, behavior="conversion"):
                self.assertEqual(
                    state["calls"],
                    ["source", "destination", "destination", "source"],
                )
                self.assertEqual(moved.shape, (3, 2, 4))
                self.assertEqual(moved.stride(), (4, 12, 1))

            source = CustomIndex()
            destination = CustomIndex()
            marker = object()

            class RecordingMode(torch.overrides.TorchFunctionMode):
                def __init__(self):
                    self.calls = []

                def __torch_function__(self, func, types, args=(), kwargs=None):
                    self.calls.append((func, types, args, kwargs))
                    return marker

            mode = RecordingMode()
            with mode:
                result = operation((source,), [destination])
            with self.subTest(operation=operation_name, behavior="mode"):
                self.assertIs(result, marker)
                self.assertEqual(source.calls, 1)
                self.assertEqual(destination.calls, 1)
                self.assertEqual(len(mode.calls), 1)
                _, _, forwarded_args, forwarded_kwargs = mode.calls[0]
                self.assertIs(forwarded_args[0], tensor)
                self.assertIs(forwarded_args[1][0], source)
                self.assertIs(forwarded_args[2][0], destination)
                self.assertIsNone(forwarded_kwargs)

    def test_unsupported_sequence_dimensions_and_mixed_forms_are_rejected(self):
        tensor = torch.zeros((2, 3, 4))
        self.assertTrue(hasattr(torch, "movedim"))
        self.assertTrue(hasattr(torch, "moveaxis"))
        self.assertIn("moveaxis", torch.__all__)
        self.assertTrue(hasattr(torch.Tensor, "moveaxis"))
        for source, destination in (
            (0, (2,)),
            ((0,), 2),
            ((0, 2), (2, 0)),
            ([0, 2], [2, 0]),
            ((), ()),
            ([], []),
            (torch.Size(), torch.Size()),
            (torch.Size([0, 2]), torch.Size([2, 0])),
        ):
            with self.subTest(source=source, destination=destination):
                for operation in (
                    lambda: tensor.movedim(source, destination),
                    lambda: torch.movedim(tensor, source, destination),
                    lambda: torch.moveaxis(tensor, source, destination),
                    lambda: tensor.moveaxis(source, destination),
                ):
                    with self.assertRaisesRegex(
                        TypeError,
                        r"received an invalid combination of arguments",
                    ):
                        operation()

    def test_tensorbase_descriptor_metadata_and_unbound_behavior(self):
        tensor = torch.zeros((2, 3, 4))
        descriptor = inspect.getattr_static(torch.Tensor, "movedim")
        bound = tensor.movedim

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "movedim")
        self.assertEqual(descriptor.__qualname__, "TensorBase.movedim")
        self.assertEqual(bound.__name__, "movedim")
        self.assertEqual(bound.__qualname__, "Tensor.movedim")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertEqual(
            repr(descriptor),
            "<method 'movedim' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.movedim, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        result = descriptor(tensor, 0, -1)
        self.assertEqual(result.shape, (3, 4, 2))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())
        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.movedim() needs an argument",
            ),
            (
                lambda: descriptor(1, 0, 1),
                "descriptor 'movedim' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor, source=0, destination=1),
                "unbound method TensorBase.movedim() needs an argument",
            ),
        )
        for call, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), expected)

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.zeros((2, 3, 4))
        descriptor = inspect.getattr_static(torch.Tensor, "movedim")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        positional = RecordingMode(marker)
        with positional:
            result = tensor.movedim(0, -1)
        self.assertIs(result, marker)
        function, dispatch_types, args, kwargs = positional.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 3)
        self.assertIs(args[0], tensor)
        self.assertEqual(args[1:], (0, -1))
        self.assertIsNone(kwargs)

        keyword = RecordingMode(marker)
        with keyword:
            result = tensor.movedim(destination=-1, source=0)
        self.assertIs(result, marker)
        function, dispatch_types, args, kwargs = keyword.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor,))
        self.assertEqual(kwargs, {"destination": -1, "source": 0})

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.movedim(source=0, destination=-1)
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.shape, (3, 4, 2))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

    def test_mode_dispatch_follows_binding_and_precedes_conversion(self):
        tensor = torch.zeros((2, 3, 4))
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for source, destination in ((2**100, 0), (3, -4)):
            mode = RecordingMode(marker)
            with mode:
                result = tensor.movedim(source, destination)
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)

        for source, destination in ((True, 0), (1.5, 0), (0, CustomIndex())):
            mode = RecordingMode(marker)
            with self.assertRaises(TypeError):
                with mode:
                    tensor.movedim(source, destination)
            self.assertEqual(mode.calls, [])

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaises(TypeError) as raised:
            with lower:
                with declining:
                    tensor.movedim(0, 1)
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^Multiple dispatch failed for 'torch\.Tensor\.movedim'; all "
                r"__torch_function__ handlers returned NotImplemented:\n\n"
                r"  - mode object <.*RecordingMode object at 0x[0-9a-f]+>\n\n"
                r"For more information, try re-running with "
                r"TORCH_LOGS=not_implemented$"
            ),
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])


class TopLevelMovedimTests(unittest.TestCase):
    def assert_values(self, tensor, expected):
        np.testing.assert_array_equal(np.asarray(tensor), np.asarray(expected))

    def movedim_calls(self, tensor, source, destination):
        return (
            ("positional", torch.movedim(tensor, source, destination)),
            (
                "dimension keywords",
                torch.movedim(tensor, source=source, destination=destination),
            ),
            (
                "all keywords",
                torch.movedim(
                    input=tensor, source=source, destination=destination
                ),
            ),
            (
                "reordered keywords",
                torch.movedim(
                    destination=destination, input=tensor, source=source
                ),
            ),
        )

    def test_integer_forms_reuse_the_method_view_engine(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        cases = (
            ("scalar", torch.tensor(2.5), 0, -1),
            ("empty", torch.zeros((2, 0, 3)), -1, 0),
            ("offset", base.transpose(0, 2)[1], -1, 0),
            ("noncontiguous", base.transpose(0, 2), 0, 2),
        )
        for case, source_tensor, source, destination in cases:
            expected = source_tensor.movedim(source, destination)
            original = (
                source_tensor.shape,
                source_tensor.stride(),
                source_tensor.storage_offset(),
                source_tensor.data_ptr(),
            )
            for form, actual in self.movedim_calls(
                source_tensor, source, destination
            ):
                with self.subTest(case=case, form=form):
                    self.assertIsNot(actual, source_tensor)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(
                        actual.storage_offset(), expected.storage_offset()
                    )
                    self.assertEqual(actual.data_ptr(), source_tensor.data_ptr())
                    self.assertIs(actual.dtype, torch.float32)
                    self.assertEqual(actual.device, torch.device("cpu"))
                    self.assert_values(actual, expected)
            self.assertEqual(
                (
                    source_tensor.shape,
                    source_tensor.stride(),
                    source_tensor.storage_offset(),
                    source_tensor.data_ptr(),
                ),
                original,
            )

    def test_input_aliases_and_integer_conversion_order_match_the_legacy_binding(self):
        tensor = torch.zeros((2, 3, 4))
        for alias in ("input", "x", "a", "x1"):
            with self.subTest(alias=alias):
                moved = torch.movedim(
                    **{alias: tensor, "source": 0, "destination": -1}
                )
                self.assertEqual(moved.shape, (3, 4, 2))
                self.assertEqual(moved.stride(), (4, 1, 12))
                self.assertEqual(moved.data_ptr(), tensor.data_ptr())

        state = {"destination_converted": False, "calls": []}

        class StatefulInteger(np.int64):
            def __new__(cls, role):
                value = np.int64.__new__(cls, 0)
                value.role = role
                return value

            def __index__(self):
                state["calls"].append(self.role)
                if self.role == "destination":
                    state["destination_converted"] = True
                    return 1
                return 0 if state["destination_converted"] else 2

        moved = torch.movedim(
            tensor,
            StatefulInteger("source"),
            StatefulInteger("destination"),
        )
        self.assertEqual(state["calls"], ["destination", "source"])
        self.assertEqual(moved.shape, (3, 2, 4))
        self.assertEqual(moved.stride(), (4, 12, 1))

    def test_autograd_empty_backward_and_no_grad_match_the_method(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(
            4, 2, 3
        )
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        moved = torch.movedim(leaf, -1, 0)
        self.assertTrue(moved.requires_grad)
        self.assertFalse(moved.is_leaf)
        self.assertEqual(moved.data_ptr(), leaf.data_ptr())
        (moved * torch.tensor(weights.tolist())).sum().backward()
        self.assert_values(leaf.grad, np.moveaxis(weights, 0, -1))

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.movedim(empty, source=0, destination=-1).sum().backward()
        self.assert_values(empty.grad, np.zeros((2, 0, 3), dtype=np.float32))

        with torch.no_grad():
            untracked = torch.movedim(
                input=leaf, source=0, destination=1
            )
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.data_ptr(), leaf.data_ptr())
        self.assertEqual(untracked.shape, (3, 2, 4))
        self.assertEqual(untracked.stride(), (4, 12, 1))

    def test_integer_binding_and_dimension_errors_match_pytorch_shape(self):
        tensor = torch.zeros((2, 3, 4))
        cases = (
            (
                lambda: torch.movedim(),
                "movedim() received an invalid combination of arguments - got "
                f"(), {TOP_LEVEL_OVERLOADS}",
            ),
            (
                lambda: torch.movedim(tensor, 0),
                "movedim() received an invalid combination of arguments - got "
                f"(Tensor, int), {TOP_LEVEL_OVERLOADS}",
            ),
            (
                lambda: torch.movedim(tensor, 0, 1, 2),
                "movedim() received an invalid combination of arguments - got "
                f"(Tensor, int, int, int), {TOP_LEVEL_OVERLOADS}",
            ),
            (
                lambda: torch.movedim(1, 0, 1),
                "movedim() received an invalid combination of arguments - got "
                "(int, int, int), but expected one of:\n"
                " * (Tensor input, int source, int destination)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (!int!, int, int)\n"
                " * (Tensor input, tuple of ints source, tuple of ints destination)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (!int!, !int!, !int!)\n",
            ),
            (
                lambda: torch.movedim(tensor, True, 0),
                "movedim() received an invalid combination of arguments - got "
                "(Tensor, bool, int), but expected one of:\n"
                " * (Tensor input, int source, int destination)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (Tensor, !bool!, int)\n"
                " * (Tensor input, tuple of ints source, tuple of ints destination)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (Tensor, !bool!, !int!)\n",
            ),
            (
                lambda: torch.movedim(tensor, 0, source=1),
                "movedim() received an invalid combination of arguments - got "
                "(Tensor, int, source=int), but expected one of:\n"
                " * (Tensor input, int source, int destination)\n"
                "      didn't match because some of the keywords were incorrect: source\n"
                " * (Tensor input, tuple of ints source, tuple of ints destination)\n"
                "      didn't match because some of the keywords were incorrect: source\n",
            ),
        )
        for call, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), expected)

        for value in (2**63, -(2**63) - 1, np.uint64(2**63)):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(
                    ValueError, "^Overflow when unpacking long long$"
                ):
                    torch.movedim(tensor, 0, value)

        with self.assertRaisesRegex(
            IndexError,
            r"^Dimension out of range \(expected to be in range of "
            r"\[-3, 2\], but got -4\)$",
        ):
            torch.movedim(tensor, 0, -4)

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.zeros((2, 3, 4))
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        positional = RecordingMode(marker)
        with positional:
            result = torch.movedim(tensor, 0, -1)
        self.assertIs(result, marker)
        function, dispatch_types, args, kwargs = positional.calls[0]
        self.assertIs(function, torch.movedim)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor, 0, -1))
        self.assertIsNone(kwargs)

        keyword = RecordingMode(marker)
        with keyword:
            result = torch.movedim(
                destination=-1, input=tensor, source=0
            )
        self.assertIs(result, marker)
        function, dispatch_types, args, kwargs = keyword.calls[0]
        self.assertIs(function, torch.movedim)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(
            kwargs, {"destination": -1, "input": tensor, "source": 0}
        )

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.movedim(
                    tensor, source=0, destination=-1
                )
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.shape, (3, 4, 2))
        self.assertEqual(forwarded.stride(), (4, 1, 12))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        deferred = RecordingMode(marker)
        with deferred:
            self.assertIs(torch.movedim(tensor, 2**100, -4), marker)
        self.assertEqual(len(deferred.calls), 1)

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                torch.movedim(tensor, True, 0)
        self.assertEqual(invalid.calls, [])

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.movedim'; all "
            r"__torch_function__ handlers returned NotImplemented:",
        ):
            with lower:
                with declining:
                    torch.movedim(tensor, 0, 1)
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])

    def test_tensor_like_override_dispatch_uses_the_public_function(self):
        calls = []
        marker = object()

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(torch.movedim(value, 0, 1), marker)
        function, dispatch_types, args, kwargs = calls[0]
        self.assertIs(function, torch.movedim)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (value, 0, 1))
        self.assertIsNone(kwargs)

    def test_callable_metadata_documentation_ownership_and_exports(self):
        function = torch.movedim
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "movedim")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.movedim")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method movedim of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.movedim, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("movedim"), 1)
        self.assertEqual(torch.__all__.count("moveaxis"), 1)
        self.assertTrue(hasattr(torch, "moveaxis"))
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["movedim"], function)
        self.assertIs(wildcard_namespace["moveaxis"], torch.moveaxis)


if __name__ == "__main__":
    unittest.main()
