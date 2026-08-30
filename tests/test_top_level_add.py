import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
add(input, other, *, alpha=1, out=None) -> Tensor

Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.

.. math::
    \\text{{out}}_i = \\text{{input}}_i + \\text{{alpha}} \\times \\text{{other}}_i


Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    input (Tensor): the input tensor.
    other (Tensor or Number): the tensor or number to add to :attr:`input`.

Keyword arguments:
    alpha (Number): the multiplier for :attr:`other`.
    out (Tensor, optional): the output tensor.

Examples::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
    >>> torch.add(a, 20)
    tensor([ 20.0202,  21.0985,  21.3506,  19.3944])

    >>> b = torch.randn(4)
    >>> b
    tensor([-0.9732, -0.3497,  0.6245,  0.4022])
    >>> c = torch.randn(4, 1)
    >>> c
    tensor([[ 0.3743],
            [-1.7724],
            [-0.5811],
            [-0.8017]])
    >>> torch.add(b, c, alpha=10)
    tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
            [-18.6971, -18.0736, -17.0994, -17.3216],
            [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
            [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
"""


class TopLevelAddTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def test_scalar_calls_match_existing_addition_paths(self):
        base = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        offset = base[1]
        scalar_forms = (True, -2, 2.5, np.bool_(False), np.int64(3), np.float32(-0.0))
        for scalar in scalar_forms:
            for form, call, expected in (
                ("tensor/scalar", lambda scalar=scalar: torch.add(offset, scalar), offset + scalar),
                ("scalar/tensor", lambda scalar=scalar: torch.add(scalar, offset), scalar + offset),
                (
                    "canonical tensor/scalar",
                    lambda scalar=scalar: torch.add(input=offset, other=scalar),
                    offset + scalar,
                ),
                (
                    "canonical scalar/tensor",
                    lambda scalar=scalar: torch.add(input=scalar, other=offset),
                    scalar + offset,
                ),
                (
                    "legacy x aliases",
                    lambda scalar=scalar: torch.add(x=offset, x2=scalar),
                    offset + scalar,
                ),
                (
                    "alpha and out defaults",
                    lambda scalar=scalar: torch.add(offset, scalar, alpha=1, out=None),
                    offset + scalar,
                ),
            ):
                self.assert_tensor_matches(
                    call(), expected, case=(form, type(scalar).__name__, scalar)
                )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        self.assert_tensor_matches(
            torch.add(empty, -2.0), empty + -2.0, case="strided empty"
        )

        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF85_4321,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        self.assert_tensor_matches(
            torch.add(-0.0, special), -0.0 + special, case="IEEE special values"
        )

    def test_autograd_empty_views_and_no_grad_reuse_scalar_addition(self):
        for case, make_pair in (
            (
                "offset",
                lambda: (
                    torch.tensor(
                        np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
                        requires_grad=True,
                    ),
                    lambda leaf: leaf.transpose(0, 2)[1],
                ),
            ),
            (
                "empty",
                lambda: (
                    torch.zeros((2, 0, 3), requires_grad=True),
                    lambda leaf: leaf.transpose(0, 2)[1],
                ),
            ),
        ):
            function_leaf, function_view = make_pair()
            operator_leaf, operator_view = make_pair()
            function_output = torch.add(4.0, function_view(function_leaf))
            operator_output = 4.0 + operator_view(operator_leaf)
            self.assert_tensor_matches(function_output, operator_output, case=case)
            function_output.sum().backward()
            operator_output.sum().backward()
            self.assert_tensor_matches(
                function_leaf.grad, operator_leaf.grad, case=f"{case} gradient"
            )

        leaf = torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            output = torch.add(input=leaf.transpose(0, 1), other=2.0, out=None)
        self.assertFalse(output.requires_grad)
        self.assertTrue(output.is_leaf)
        self.assertIsNone(leaf.grad)
        self.assertTrue(torch.add(leaf, 2.0).requires_grad)

    def test_modes_and_overrides_observe_calls_before_native_limits(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        destination = torch.tensor([0.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        calls = (
            (lambda: torch.add(tensor, 2.0), (tensor, 2.0), None),
            (lambda: torch.add(2.0, tensor), (2.0, tensor), None),
            (
                lambda: torch.add(input=2.0, other=tensor, alpha=2, out=destination),
                (),
                ("input", "other", "alpha", "out"),
            ),
            (lambda: torch.add(tensor, tensor), (tensor, tensor), None),
        )
        for call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.add)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            if expected_keywords is None:
                self.assertIsNone(kwargs)
            else:
                self.assertEqual(tuple(kwargs), expected_keywords)

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        for call, expected_keywords in (
            (lambda value: torch.add(value, tensor), None),
            (lambda value: torch.add(tensor, value), None),
            (lambda value: torch.add(tensor, 2.0, alpha=value), ("alpha",)),
            (lambda value: torch.add(tensor, 2.0, out=value), ("out",)),
        ):
            override_calls.clear()
            self.assertIs(call(Override()), marker)
            function, dispatch_types, _, kwargs = override_calls[0]
            self.assertIs(function, torch.add)
            self.assertEqual(dispatch_types, (Override,))
            if expected_keywords is None:
                self.assertIsNone(kwargs)
            else:
                self.assertEqual(tuple(kwargs), expected_keywords)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.add(input=2.0, other=tensor, alpha=1, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(forwarded, 2.0 + tensor, case="forwarded modes")

        for call in (
            lambda: torch.add([], tensor),
            lambda: torch.add(tensor, []),
            lambda: torch.add(tensor, 2.0, alpha="1"),
            lambda: torch.add(tensor, 2.0, unexpected=True),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

    def test_rejected_native_extensions_and_callable_metadata(self):
        tensor = torch.tensor([1.0])
        destination = torch.tensor([17.0])
        cases = (
            (
                lambda: torch.add(),
                r"^add\(\) received an invalid combination of arguments - got "
                r"\(\), but expected \(Tensor input, Tensor other, \*, Number "
                r"alpha = 1, Tensor out = None\)$",
            ),
            (
                lambda: torch.add(tensor, tensor, tensor),
                r"^add\(\) takes 2 positional arguments but 3 were given$",
            ),
            (
                lambda: torch.add([], tensor),
                r"^add\(\): argument 'input' \(position 1\) must be Tensor, not list$",
            ),
            (
                lambda: torch.add(tensor, []),
                r"^add\(\): argument 'other' \(position 2\) must be Tensor, not list$",
            ),
            (
                lambda: torch.add(input=None, other=tensor),
                r"^add\(\): argument 'input' must be Tensor, not NoneType$",
            ),
            (
                lambda: torch.add(tensor, 2.0, alpha=True),
                r"^Boolean alpha only supported for Boolean results\.$",
            ),
            (
                lambda: torch.add(tensor, 2.0, alpha="1"),
                r"^add\(\): argument 'alpha' must be Number, not str$",
            ),
            (
                lambda: torch.add(tensor, 2.0, dtype=torch.float32),
                r"^add\(\) got an unexpected keyword argument 'dtype'$",
            ),
            (
                lambda: torch.add(tensor, 2.0, device=torch.device("cpu")),
                r"^add\(\) got an unexpected keyword argument 'device'$",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, message):
                    call()

        with self.assertRaisesRegex(
            NotImplementedError,
            r"^add\(\): Tensor/Tensor operands are not supported; one operand must be a real scalar$",
        ):
            torch.add(tensor, tensor)
        with self.assertRaisesRegex(
            NotImplementedError, r"^add\(\): non-default alpha is not supported$"
        ):
            torch.add(tensor, 2.0, alpha=2)
        with self.assertRaisesRegex(
            RuntimeError, r"^add\(\): the 'out' argument is not supported$"
        ):
            torch.add(tensor, 2.0, out=destination)
        self.assertEqual(destination.tolist(), [17.0])
        with self.assertRaisesRegex(TypeError, "scalar-scalar addition is not supported"):
            torch.add(2, 3)
        for value in (object(), 1 + 2j, np.complex64(1 + 2j)):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    torch.add(tensor, value)
                with self.assertRaises(TypeError):
                    torch.add(value, tensor)
        self.assertFalse(hasattr(torch.Tensor, "add"))

        function = torch.add
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "add")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.add")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertRegex(
            repr(function), r"^<built-in method add of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.add, function)
        for mutation in (
            lambda: setattr(owner, "add", None),
            lambda: delattr(owner, "add"),
        ):
            with self.assertRaises(TypeError):
                mutation()
            self.assertIs(owner.add, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("add"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["add"], function)


if __name__ == "__main__":
    unittest.main()
