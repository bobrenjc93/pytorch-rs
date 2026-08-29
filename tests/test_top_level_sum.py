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
    
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
        If ``None``, all dimensions are reduced.

    
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
    def assert_scalar_matches_method(self, output, source, method_output, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertIsNot(output, source)
            self.assertFalse(output.is_set_to(source))
            self.assertEqual(output.shape, method_output.shape)
            self.assertEqual(output.stride(), method_output.stride())
            self.assertEqual(output.storage_offset(), method_output.storage_offset())
            self.assertEqual(output.numel(), method_output.numel())
            self.assertEqual(output.is_contiguous(), method_output.is_contiguous())
            self.assertEqual(output.requires_grad, method_output.requires_grad)
            self.assertEqual(output.is_leaf, method_output.is_leaf)
            self.assertIs(output.dtype, torch.float32)
            self.assertEqual(output.device, torch.device("cpu"))
        with self.subTest(case=case, value=True):
            np.testing.assert_array_equal(
                np.asarray(output).reshape(-1).view(np.uint32),
                np.asarray(method_output).reshape(-1).view(np.uint32),
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

    @staticmethod
    def make_autograd_case(case):
        if case == "scalar":
            leaf = torch.tensor(-3.5, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return leaf, leaf.transpose(0, 2)[1]

        leaf = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        if case == "offset":
            return leaf, leaf.transpose(0, 2)[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 2)
        raise AssertionError(f"unknown autograd case: {case}")

    def test_supported_calls_delegate_to_tensor_sum_values_metadata_and_storage(self):
        dense = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        noncontiguous = dense.transpose(0, 2)
        cases = (
            ("scalar", torch.tensor(-3.5)),
            ("negative zero", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("contiguous offset", dense[1]),
            ("offset", noncontiguous[1]),
            ("noncontiguous", noncontiguous),
        )

        for case, source in cases:
            method_output = source.sum()
            for form, call in self.supported_calls(source):
                self.assert_scalar_matches_method(
                    call(), source, method_output, case=(case, form)
                )

    def test_autograd_empty_offsets_no_grad_and_reuse_match_tensor_sum(self):
        forms = ("positional", "input", "dtype none", "dtype float32")
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form in forms:
                function_leaf, function_source = self.make_autograd_case(case)
                method_leaf, method_source = self.make_autograd_case(case)
                output = dict(self.supported_calls(function_source))[form]()
                method_output = method_source.sum()

                self.assert_scalar_matches_method(
                    output, function_source, method_output, case=(case, form, "output")
                )
                output.backward()
                method_output.backward()
                self.assert_scalar_matches_method(
                    function_leaf.grad,
                    function_leaf,
                    method_leaf.grad,
                    case=(case, form, "gradient"),
                )

        leaf = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], requires_grad=True
        )
        loss = torch.sum(leaf.transpose(0, 1), dtype=torch.float)
        loss.backward()
        loss.backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])

        no_grad_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        with torch.no_grad():
            untracked = torch.sum(input=no_grad_leaf, dtype=torch.float32)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertTrue(torch.sum(no_grad_leaf).requires_grad)

    def test_modes_and_overrides_observe_supported_sum_calls(self):
        tensor = torch.tensor([1.0, 2.0])
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        value = Override()
        calls = (
            ("positional", lambda: torch.sum(value), (value,), None),
            ("input", lambda: torch.sum(input=value), (), {"input": value}),
            ("x", lambda: torch.sum(x=value), (), {"x": value}),
            ("a", lambda: torch.sum(a=value), (), {"a": value}),
            ("x1", lambda: torch.sum(x1=value), (), {"x1": value}),
            (
                "dtype",
                lambda: torch.sum(tensor, dtype=value),
                (tensor,),
                {"dtype": value},
            ),
        )
        for form, call, expected_args, expected_kwargs in calls:
            with self.subTest(form=form):
                Override.calls.clear()
                self.assertIs(call(), marker)
                self.assertEqual(len(Override.calls), 1)
                function, dispatch_types, args, kwargs = Override.calls[0]
                self.assertIs(function, torch.sum)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            self.assertIs(torch.sum(input=tensor, dtype=torch.float32), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.sum)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "dtype": torch.float32})

        mode = RecordingMode(marker)
        with mode:
            self.assertIs(torch.sum(tensor, axis=0, keepdims=True), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.sum)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor,))
        self.assertEqual(kwargs, {"axis": 0, "keepdims": True})

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = torch.sum(tensor)
        self.assertEqual(forwarded.item(), 3.0)

        declining_mode = RecordingMode(NotImplemented)
        Override.calls.clear()
        with declining_mode:
            self.assertIs(torch.sum(value), marker)
        self.assertEqual(len(declining_mode.calls), 1)
        self.assertEqual(len(Override.calls), 1)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        message = (
            "Multiple dispatch failed for 'torch.sum'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            "  - tensor subclass <class "
            f"'{DecliningOverride.__module__}.{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            torch.sum(DecliningOverride())

    def test_unsupported_reduction_forms_remain_outside_the_native_surface(self):
        tensor = torch.ones((2, 3))
        destination = torch.tensor([17.0, 19.0, 23.0])
        unsupported = (
            ("positional dim", lambda: torch.sum(tensor, 0)),
            ("positional none dim", lambda: torch.sum(tensor, None)),
            ("keyword dim", lambda: torch.sum(tensor, dim=0)),
            ("none dim", lambda: torch.sum(tensor, dim=None)),
            ("axis", lambda: torch.sum(tensor, axis=0)),
            ("none axis", lambda: torch.sum(tensor, axis=None)),
            ("positional keepdim", lambda: torch.sum(tensor, 0, False)),
            ("keyword keepdim", lambda: torch.sum(tensor, dim=0, keepdim=True)),
            ("keyword keepdims", lambda: torch.sum(tensor, axis=0, keepdims=True)),
            ("out none without dim", lambda: torch.sum(tensor, out=None)),
            ("out with dim", lambda: torch.sum(tensor, 0, out=destination)),
            ("out with axis", lambda: torch.sum(tensor, axis=0, out=destination)),
        )
        for case, call in unsupported:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^sum\(\) received an invalid combination of arguments",
                ):
                    call()
                self.assertEqual(destination.tolist(), [17.0, 19.0, 23.0])

        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        dimension = Override()
        self.assertIs(torch.sum(tensor, dim=dimension), marker)
        self.assertEqual(len(Override.calls), 1)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, torch.sum)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (tensor,))
        self.assertEqual(kwargs, {"dim": dimension})

        Override.calls.clear()
        axis = Override()
        self.assertIs(torch.sum(tensor, axis=axis), marker)
        self.assertEqual(len(Override.calls), 1)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, torch.sum)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (tensor,))
        self.assertEqual(kwargs, {"axis": axis})

        Override.calls.clear()
        keepdims = Override()
        self.assertIs(torch.sum(tensor, dim=0, keepdims=keepdims), marker)
        self.assertEqual(len(Override.calls), 1)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, torch.sum)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (tensor,))
        self.assertEqual(kwargs, {"dim": 0, "keepdims": keepdims})

        Override.calls.clear()
        tuple_dimension = Override()
        self.assertIs(torch.sum(tensor, dim=(0, tuple_dimension)), marker)
        self.assertEqual(len(Override.calls), 1)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, torch.sum)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (tensor,))
        self.assertEqual(kwargs, {"dim": (0, tuple_dimension)})

    def test_invalid_reduction_arguments_are_rejected_before_dispatch(self):
        tensor = torch.ones((2, 3))
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return marker

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        invalid_mode_calls = (
            ("object dim", lambda: torch.sum(tensor, dim=object())),
            ("string dim", lambda: torch.sum(tensor, dim="0")),
            ("bool dim", lambda: torch.sum(tensor, dim=True)),
            ("tuple bool dim", lambda: torch.sum(tensor, dim=(True,))),
            ("object axis", lambda: torch.sum(tensor, axis=object())),
            ("bad keepdim", lambda: torch.sum(tensor, dim=0, keepdim="x")),
            ("bad keepdims", lambda: torch.sum(tensor, axis=0, keepdims=1)),
            ("bad out", lambda: torch.sum(tensor, dim=0, out=object())),
        )
        for case, call in invalid_mode_calls:
            with self.subTest(case=case):
                mode = RecordingMode()
                with mode:
                    with self.assertRaises(TypeError):
                        call()
                self.assertEqual(mode.calls, [])

        invalid_override_calls = (
            ("input override bad dim", lambda: torch.sum(Override(), dim="0")),
            (
                "input override bad keepdim",
                lambda: torch.sum(Override(), dim=0, keepdim="x"),
            ),
            (
                "input override bad out",
                lambda: torch.sum(Override(), dim=0, out=object()),
            ),
        )
        for case, call in invalid_override_calls:
            with self.subTest(case=case):
                Override.calls.clear()
                with self.assertRaises(TypeError):
                    call()
                self.assertEqual(Override.calls, [])

    def test_binding_dtype_and_type_errors_match_the_supported_boundary(self):
        tensor = torch.ones((2, 3))
        invalid = "sum() received an invalid combination of arguments - got "
        cases = (
            (
                lambda: torch.sum(),
                f"{invalid}(), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(dtype=torch.float32),
                'sum() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sum(tensor, tensor, tensor, tensor),
                "sum() takes from 2 to 3 positional arguments but 4 were given",
            ),
            (
                lambda: torch.sum(1),
                "sum(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.sum(input=[]),
                "sum(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.sum(1, dtype=None),
                f"{invalid}(int, dtype=NoneType), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(input=[], dim=0),
                f"{invalid}(dim=int, input=list, ), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, dtype=1),
                f"{invalid}(Tensor, dtype=int), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, dtype=object()),
                f"{invalid}(Tensor, dtype=object), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, extra=True),
                f"{invalid}(Tensor, extra=bool), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: torch.sum(tensor, dim=0, axis=1),
                "sum() got an unexpected keyword argument 'axis'",
            ),
            (
                lambda: torch.sum(tensor, axis=1, dim=0),
                "sum() got an unexpected keyword argument 'axis'",
            ),
            (
                lambda: torch.sum(tensor, 0, axis=1),
                "sum() got an unexpected keyword argument 'axis'",
            ),
            (
                lambda: torch.sum(tensor, dim=0, keepdim=True, keepdims=False),
                "sum() got an unexpected keyword argument 'keepdims'",
            ),
            (
                lambda: torch.sum(tensor, dim=0, keepdims=False, keepdim=True),
                "sum() got an unexpected keyword argument 'keepdims'",
            ),
            (
                lambda: torch.sum(tensor, 0, True, keepdims=False),
                "sum() got an unexpected keyword argument 'keepdims'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_callable_metadata_documentation_ownership_exports_and_pickling(self):
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
        for action in (
            lambda: setattr(owner, "sum", None),
            lambda: delattr(owner, "sum"),
        ):
            with self.assertRaises(TypeError):
                action()
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


if __name__ == "__main__":
    unittest.main()
