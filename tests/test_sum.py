import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
sum(input, *, dtype=None) -> Tensor

Returns the sum of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

.. note:: Use the `dtype` argument if you need the result in a specific tensor type.
          Otherwise, the result type may be automatically promoted (e.g., from `torch.int32` to `torch.int64`).

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.1133, -0.9567,  0.2958]])
    >>> torch.sum(a)
    tensor(-0.5475)

.. function:: sum(input, dim, keepdim=False, *, dtype=None) -> Tensor
   :noindex:

Returns the sum of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
\x20\x20\x20\x20
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
        If ``None``, all dimensions are reduced.

\x20\x20\x20\x20
    keepdim (bool, optional): whether the output tensor has :attr:`dim` retained or not. Default: ``False``.


Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
            [-0.2993,  0.9138,  0.9337, -1.6864],
            [ 0.1132,  0.7892, -0.1003,  0.5688],
            [ 0.3637, -0.9906, -0.4752, -1.5197]])
    >>> torch.sum(a, 1)
    tensor([-0.4598, -0.1381,  1.3708, -2.6217])
    >>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
    >>> torch.sum(b, (2, 1))
    tensor([  435.,  1335.,  2235.,  3135.])
"""

EXPECTED_OVERLOADS = (
    "but expected one of:\n"
    " * (Tensor input, *, torch.dtype dtype = None)\n"
    " * (Tensor input, tuple of ints dim, bool keepdim = False, *, "
    "torch.dtype dtype = None, Tensor out = None)\n"
)


class TopLevelSumTests(unittest.TestCase):
    def assert_scalar_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, ())
            self.assertEqual(actual.stride(), ())
            self.assertEqual(actual.storage_offset(), 0)
            self.assertEqual(actual.numel(), 1)
            self.assertTrue(actual.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, value=True):
            self.assertEqual(
                np.float32(actual.item()).view(np.uint32).item(),
                np.float32(expected.item()).view(np.uint32).item(),
            )

    @staticmethod
    def supported_calls(source):
        return (
            ("positional", lambda: torch.sum(source)),
            ("input", lambda: torch.sum(input=source)),
            ("x", lambda: torch.sum(x=source)),
            ("a", lambda: torch.sum(a=source)),
            ("x1", lambda: torch.sum(x1=source)),
            ("dtype none", lambda: torch.sum(source, dtype=None)),
            ("dtype float32", lambda: torch.sum(input=source, dtype=torch.float32)),
            ("dtype float alias", lambda: torch.sum(x=source, dtype=torch.float)),
        )

    def test_supported_forms_reuse_full_reduction_values_and_metadata(self):
        dense = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        noncontiguous = dense.transpose(0, 2)
        cases = (
            ("scalar", torch.tensor(-3.5)),
            ("negative zero", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("offset", noncontiguous[1]),
            ("noncontiguous", noncontiguous),
        )

        for case, source in cases:
            expected = source.sum()
            for form, call in self.supported_calls(source):
                self.assert_scalar_matches(call(), expected, case=(case, form))

        empty = torch.sum(torch.zeros((3, 0, 2)))
        self.assertEqual(np.asarray(empty).view(np.uint32).item(), 0)

    def test_autograd_repeated_backward_empty_and_no_grad_reuse_sum_edge(self):
        leaf = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], requires_grad=True
        )
        loss = torch.sum(leaf.transpose(0, 1), dtype=torch.float32)
        self.assertTrue(loss.requires_grad)
        self.assertFalse(loss.is_leaf)
        loss.backward()
        loss.backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.sum(empty.transpose(0, 2)).backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.tolist(), [[], []])

        with torch.no_grad():
            untracked = torch.sum(leaf)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertTrue(torch.sum(leaf).requires_grad)

    def test_torch_function_modes_and_input_overrides_receive_original_calls(self):
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        calls = (
            (lambda: torch.sum(tensor), (tensor,), None),
            (lambda: torch.sum(input=tensor), (), {"input": tensor}),
            (
                lambda: torch.sum(x=tensor, dtype=torch.float32),
                (),
                {"x": tensor, "dtype": torch.float32},
            ),
        )
        for call, expected_args, expected_kwargs in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.sum)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.sum(input=tensor, dtype=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.item(), 3.0)
        forwarded.backward()
        forwarded.backward()
        self.assertEqual(tensor.grad.tolist(), [2.0, 2.0])

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(torch.sum(value, dtype=None), marker)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, torch.sum)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (value,))
        self.assertEqual(kwargs, {"dtype": None})

    def test_callable_ownership_documentation_pickling_and_exports(self):
        function = torch.sum
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "sum")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.sum")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method sum of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.sum, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("sum"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["sum"], function)

    def test_supported_errors_match_and_unsupported_overloads_are_rejected(self):
        tensor = torch.ones((2, 3))
        invalid = "sum() received an invalid combination of arguments - got "
        exact_cases = (
            (lambda: torch.sum(), f"{invalid}(), {EXPECTED_OVERLOADS}"),
            (
                lambda: torch.sum(1),
                "sum(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.sum(input=[]),
                "sum(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.sum(dtype=None),
                'sum() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sum(tensor, input=tensor),
                f"{invalid}(Tensor, input=Tensor), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, dtype=1),
                f"{invalid}(Tensor, dtype=int), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, 0, False, torch.float32),
                "sum() takes from 2 to 3 positional arguments but 4 were given",
            ),
        )
        for call, message in exact_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        unsupported = (
            ("positional dim", lambda: torch.sum(tensor, 0)),
            ("keyword dim", lambda: torch.sum(tensor, dim=0)),
            ("keepdim", lambda: torch.sum(tensor, keepdim=True)),
            ("out", lambda: torch.sum(tensor, out=None)),
            ("other dtype", lambda: torch.sum(tensor, dtype=object())),
        )
        for case, call in unsupported:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^sum\(\) received an invalid combination of arguments",
                ):
                    call()


if __name__ == "__main__":
    unittest.main()
