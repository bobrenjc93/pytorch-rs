import gc
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = (
    "\nravel(input) -> Tensor\n\n"
    "Return a contiguous flattened tensor. A copy is made only if needed.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n\n"
    "Example::\n\n"
    "    >>> t = torch.tensor([[[1, 2],\n"
    "    ...                    [3, 4]],\n"
    "    ...                   [[5, 6],\n"
    "    ...                    [7, 8]]])\n"
    "    >>> torch.ravel(t)\n"
    "    tensor([1, 2, 3, 4, 5, 6, 7, 8])\n"
)


class TopLevelRavelTests(unittest.TestCase):
    def assert_matches_method(self, output, source, method_output):
        self.assertIsNot(output, source)
        self.assertEqual(output.shape, method_output.shape)
        self.assertEqual(output.stride(), method_output.stride())
        self.assertEqual(output.storage_offset(), method_output.storage_offset())
        self.assertEqual(output.is_contiguous(), method_output.is_contiguous())
        self.assertEqual(output.requires_grad, method_output.requires_grad)
        self.assertEqual(output.is_leaf, method_output.is_leaf)
        self.assertIs(output.dtype, torch.float32)
        self.assertEqual(output.device, torch.device("cpu"))
        self.assertEqual(
            output.data_ptr() == source.data_ptr(), source.is_contiguous()
        )
        np.testing.assert_array_equal(np.asarray(output), np.asarray(method_output))

    def ravel_calls(self, source):
        return (
            ("positional", torch.ravel(source)),
            ("input", torch.ravel(input=source)),
            ("x", torch.ravel(x=source)),
            ("a", torch.ravel(a=source)),
            ("x1", torch.ravel(x1=source)),
        )

    def test_all_call_forms_delegate_to_the_native_view_or_copy_engine(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist(), requires_grad=True)
        singleton_base = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        cases = (
            ("scalar", base[0][0][0]),
            ("vector", base[0][1]),
            ("ordinary", base),
            ("offset", base[1]),
            ("transpose", base.transpose(0, 2)),
            ("strided-vector", base.transpose(0, 2)[0][0]),
            ("singleton-stride", singleton_base.transpose(0, 1)[2]),
            (
                "empty-offset",
                torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1],
            ),
        )

        for case, source in cases:
            method_output = source.ravel()
            for form, output in self.ravel_calls(source):
                with self.subTest(case=case, form=form):
                    self.assert_matches_method(output, source, method_output)

        negative_zero = torch.ravel(torch.tensor(-0.0))
        self.assertEqual(np.asarray(negative_zero).view(np.uint32).item(), 0x8000_0000)

    def test_view_and_copy_outputs_outlive_their_sources(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retained_view():
            temporary = torch.tensor(values.tolist())
            return torch.ravel(temporary[1])

        def retained_copy():
            temporary = torch.tensor(values.tolist())
            return torch.ravel(temporary.transpose(0, 2))

        view = retained_view()
        copied = retained_copy()
        gc.collect()
        np.testing.assert_array_equal(np.asarray(view), values[1].reshape(-1))
        np.testing.assert_array_equal(
            np.asarray(copied), values.transpose(2, 1, 0).reshape(-1)
        )

    def test_autograd_empty_backward_and_no_grad_match_tensor_ravel(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        output = torch.ravel(leaf.transpose(0, 1))
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        (output * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]],
        )

        scalar = torch.tensor(2.0, requires_grad=True)
        (torch.ravel(input=scalar) * 7.0).sum().backward()
        self.assertEqual(scalar.grad.item(), 7.0)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.ravel(x=empty).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))

        source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        non_contiguous = source.transpose(0, 1)
        with torch.no_grad():
            alias = torch.ravel(a=source)
            copied = torch.ravel(x1=non_contiguous)
        self.assertTrue(alias.requires_grad)
        self.assertTrue(alias.is_leaf)
        self.assertFalse(copied.requires_grad)
        self.assertTrue(copied.is_leaf)

    def test_overrides_and_modes_receive_the_original_top_level_call(self):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        value = Override()
        calls = (
            ("positional", lambda: torch.ravel(value), (value,), None),
            ("input", lambda: torch.ravel(input=value), (), {"input": value}),
            ("x", lambda: torch.ravel(x=value), (), {"x": value}),
            ("a", lambda: torch.ravel(a=value), (), {"a": value}),
            ("x1", lambda: torch.ravel(x1=value), (), {"x1": value}),
        )
        for form, call, expected_args, expected_kwargs in calls:
            with self.subTest(form=form):
                Override.calls.clear()
                self.assertIs(call(), marker)
                self.assertEqual(len(Override.calls), 1)
                function, dispatch_types, args, kwargs = Override.calls[0]
                self.assertIs(function, torch.ravel)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            self.assertIs(torch.ravel(input=tensor), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.ravel)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor})

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = torch.ravel(tensor)
        np.testing.assert_array_equal(np.asarray(forwarded), [1.0, 2.0, 3.0, 4.0])

        declining_mode = RecordingMode(NotImplemented)
        Override.calls.clear()
        with declining_mode:
            self.assertIs(torch.ravel(value), marker)
        self.assertEqual(len(declining_mode.calls), 1)
        self.assertEqual(len(Override.calls), 1)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        message = (
            "Multiple dispatch failed for 'torch.ravel'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            "  - tensor subclass <class "
            f"'{DecliningOverride.__module__}.{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            torch.ravel(DecliningOverride())

    def test_binding_and_type_errors_match_the_legacy_schema(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.ravel(),
                'ravel() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.ravel(tensor, tensor),
                "ravel() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.ravel(tensor, input=tensor),
                "ravel() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.ravel(tensor, x=tensor),
                "ravel() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.ravel(extra=tensor),
                'ravel() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.ravel(1),
                "ravel(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.ravel(input=1),
                "ravel(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.ravel(x=tensor, extra=True),
                "ravel() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.ravel(extra=True, x=tensor),
                "ravel() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.ravel(input=tensor, x=tensor),
                "ravel() got an unexpected keyword argument 'x'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_callable_metadata_documentation_ownership_exports_and_pickling(self):
        function = torch.ravel
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "ravel")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.ravel")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method ravel of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.ravel, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("ravel"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["ravel"], function)

        message = (
            "cannot set 'ravel' attribute of immutable type "
            "'torch_rs._C._VariableFunctionsClass'"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            owner.ravel = None
        self.assertIs(owner.ravel, function)


if __name__ == "__main__":
    unittest.main()
