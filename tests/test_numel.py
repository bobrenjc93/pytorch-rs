import copy
import inspect
import pickle
import re
import sys
import types
import unittest

import torch_rs as torch


NUMEL_DOC = (
    "\nnumel(input: Tensor) -> int\n\n"
    "Returns the total number of elements in the :attr:`input` tensor.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n\n"
    "Example::\n\n"
    "    >>> a = torch.randn(1, 2, 3, 4, 5)\n"
    "    >>> torch.numel(a)\n"
    "    120\n"
    "    >>> a = torch.zeros(4,4)\n"
    "    >>> torch.numel(a)\n"
    "    16\n\n"
)


class TopLevelNumelTests(unittest.TestCase):
    def tensor_cases(self):
        base = torch.tensor(
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
                [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]],
                [[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
            ]
        )
        return (
            ("scalar", torch.tensor(-0.0), 1),
            ("empty", torch.zeros((2, 0, 3)), 0),
            ("offset", base[1], 8),
            ("noncontiguous", base.transpose(0, 2), 24),
            (
                "extreme empty",
                torch.zeros((0,))
                .reshape((2, 0, sys.maxsize))
                .transpose(0, 2),
                0,
            ),
        )

    def test_all_call_forms_return_exact_metadata_cardinality(self):
        for case, tensor, expected in self.tensor_cases():
            metadata = (
                tensor.shape,
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
                tensor.requires_grad,
                tensor.is_leaf,
            )
            results = (
                torch.numel(tensor),
                torch.numel(input=tensor),
                torch.numel(x=tensor),
                torch.numel(a=tensor),
                torch.numel(x1=tensor),
            )
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                self.assertEqual(results, (expected,) * len(results))
                self.assertTrue(all(type(result) is int for result in results))
                self.assertEqual(
                    (
                        tensor.shape,
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.data_ptr(),
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                    metadata,
                )

    def test_query_does_not_mutate_a_pending_autograd_graph(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 3.0).transpose(0, 1)[1]
        metadata = (
            tracked.shape,
            tracked.stride(),
            tracked.storage_offset(),
            tracked.data_ptr(),
            tracked.requires_grad,
            tracked.is_leaf,
        )

        self.assertEqual(torch.numel(tracked), 2)
        self.assertEqual(torch.numel(input=tracked), 2)
        self.assertEqual(torch.numel(x=tracked), 2)
        self.assertEqual(
            (
                tracked.shape,
                tracked.stride(),
                tracked.storage_offset(),
                tracked.data_ptr(),
                tracked.requires_grad,
                tracked.is_leaf,
            ),
            metadata,
        )
        self.assertIsNone(leaf.grad)

        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0], [0.0, 3.0]])

    def test_tensor_numel_remains_a_separate_method(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        descriptor = inspect.getattr_static(torch.Tensor, "numel")

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIsNot(descriptor, torch.numel)
        self.assertEqual(tensor.numel(), 4)
        self.assertEqual(tensor.nelement(), 4)
        self.assertEqual(torch.numel(tensor), tensor.numel())

    def test_torch_function_overrides_and_modes_receive_original_calls(self):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        for keyword in (None, "input", "x", "a", "x1"):
            value = Override()
            Override.calls.clear()
            result = (
                torch.numel(value)
                if keyword is None
                else torch.numel(**{keyword: value})
            )
            with self.subTest(kind="override", keyword=keyword):
                self.assertIs(result, marker)
                self.assertEqual(len(Override.calls), 1)
                function, dispatch_types, args, kwargs = Override.calls[0]
                self.assertIs(function, torch.numel)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(args, (value,) if keyword is None else ())
                self.assertEqual(
                    kwargs, None if keyword is None else {keyword: value}
                )

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        tensor = torch.tensor([1.0, 2.0, 3.0])
        for keyword in (None, "input", "x", "a", "x1"):
            mode = RecordingMode()
            with mode:
                result = (
                    torch.numel(tensor)
                    if keyword is None
                    else torch.numel(**{keyword: tensor})
                )
            with self.subTest(kind="mode", keyword=keyword):
                self.assertIs(result, marker)
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs = mode.calls[0]
                self.assertIs(function, torch.numel)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, (tensor,) if keyword is None else ())
                self.assertEqual(
                    kwargs, None if keyword is None else {keyword: tensor}
                )

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.numel(x=tensor)
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded, 3)

        class DecliningMode(RecordingMode):
            def __repr__(self):
                return "declining-numel-mode"

        message = (
            "Multiple dispatch failed for 'torch.numel'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            "  - mode object declining-numel-mode\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented"
        )
        with DecliningMode(NotImplemented):
            with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                torch.numel(tensor)

    def test_binding_and_type_errors_match_the_legacy_schema(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.numel(),
                'numel() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.numel(tensor, tensor),
                "numel() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.numel(tensor, input=tensor),
                "numel() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.numel(tensor, x=tensor),
                "numel() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.numel(foo=tensor),
                'numel() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.numel(None),
                "numel(): argument 'input' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: torch.numel(input=1),
                "numel(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.numel(x=tensor, extra=True),
                "numel() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.numel(extra=True, x=tensor),
                "numel() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.numel(input=tensor, x1=tensor),
                "numel() got an unexpected keyword argument 'x1'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_callable_metadata_exports_copy_and_pickle(self):
        function = torch.numel
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "numel")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.numel")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, NUMEL_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method numel of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.numel, function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("numel"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["numel"], function)


if __name__ == "__main__":
    unittest.main()
