import gc
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
ravel(input) -> Tensor

Return a contiguous flattened tensor. A copy is made only if needed.

Args:
    input (Tensor): the input tensor.

Example::

    >>> t = torch.tensor([[[1, 2],
    ...                    [3, 4]],
    ...                   [[5, 6],
    ...                    [7, 8]]])
    >>> torch.ravel(t)
    tensor([1, 2, 3, 4, 5, 6, 7, 8])
"""


class TopLevelRavelTests(unittest.TestCase):
    def assert_matches_method(self, output, method_output, source, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertIsNot(output, source)
            self.assertIsNot(output, method_output)
            self.assertEqual(output.shape, method_output.shape)
            self.assertEqual(output.stride(), method_output.stride())
            self.assertEqual(output.storage_offset(), method_output.storage_offset())
            self.assertEqual(output.is_contiguous(), method_output.is_contiguous())
            self.assertEqual(output.requires_grad, method_output.requires_grad)
            self.assertEqual(output.is_leaf, method_output.is_leaf)
            self.assertIs(output.dtype, method_output.dtype)
            self.assertEqual(output.device, method_output.device)
            self.assertEqual(output.is_set_to(method_output), source.is_contiguous())
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(output).reshape(-1).view(np.uint32),
                np.asarray(method_output).reshape(-1).view(np.uint32),
            )

    def call_forms(self, source):
        return (
            ("positional", lambda: torch.ravel(source)),
            ("input", lambda: torch.ravel(input=source)),
            ("x", lambda: torch.ravel(x=source)),
            ("a", lambda: torch.ravel(a=source)),
            ("x1", lambda: torch.ravel(x1=source)),
        )

    def test_delegates_layout_alias_and_copy_behavior_to_tensor_ravel(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        singleton_base = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        cases = (
            ("scalar", base[0][0][0]),
            ("vector", base[0][1]),
            ("ordinary", base),
            ("offset", base[1]),
            ("transpose", base.transpose(0, 2)),
            ("strided vector", base.transpose(0, 2)[0][0]),
            ("singleton stride", singleton_base.transpose(0, 1)[2]),
            ("empty offset", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
        )
        for case, source in cases:
            method_output = source.ravel()
            for form, call in self.call_forms(source):
                self.assert_matches_method(
                    call(), method_output, source, case=(case, form)
                )

    def test_outputs_survive_view_and_copy_sources(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def make_outputs():
            source = torch.tensor(values.tolist())
            return torch.ravel(source[1]), torch.ravel(source.transpose(0, 2))

        view, copied = make_outputs()
        gc.collect()
        np.testing.assert_array_equal(np.asarray(view), values[1].reshape(-1))
        np.testing.assert_array_equal(
            np.asarray(copied), values.transpose(2, 1, 0).reshape(-1)
        )

    def test_autograd_and_no_grad_match_tensor_ravel(self):
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
        (torch.ravel(scalar) * 7.0).sum().backward()
        self.assertEqual(scalar.grad.item(), 7.0)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.ravel(empty).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))

        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with torch.no_grad():
            alias = torch.ravel(source)
            copied = torch.ravel(source.transpose(0, 1))
        self.assertTrue(alias.requires_grad)
        self.assertTrue(alias.is_leaf)
        self.assertFalse(copied.requires_grad)
        self.assertTrue(copied.is_leaf)

    def test_modes_overrides_and_forwarding_use_the_top_level_callable(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for form, call, expected_args, expected_kwargs in (
            ("positional", lambda: torch.ravel(tensor), (tensor,), None),
            ("input", lambda: torch.ravel(input=tensor), (), {"input": tensor}),
            ("x", lambda: torch.ravel(x=tensor), (), {"x": tensor}),
        ):
            with self.subTest(dispatch="mode", form=form):
                mode = RecordingMode()
                with mode:
                    self.assertIs(call(), marker)
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs = mode.calls[0]
                self.assertIs(function, torch.ravel)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for form, call, expected_args, expected_kwargs in (
            ("positional", lambda value: torch.ravel(value), None, None),
            ("input", lambda value: torch.ravel(input=value), (), "input"),
            ("a", lambda value: torch.ravel(a=value), (), "a"),
            ("x1", lambda value: torch.ravel(x1=value), (), "x1"),
        ):
            with self.subTest(dispatch="override", form=form):
                value = Override()
                Override.calls.clear()
                self.assertIs(call(value), marker)
                function, dispatch_types, args, kwargs = Override.calls[0]
                self.assertIs(function, torch.ravel)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(args, (value,) if expected_args is None else expected_args)
                self.assertEqual(
                    kwargs,
                    None if expected_kwargs is None else {expected_kwargs: value},
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
                forwarded = torch.ravel(input=tensor)
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [1.0, 2.0, 3.0, 4.0])

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        qualified_type = (
            f"{DecliningOverride.__module__}.{DecliningOverride.__qualname__}"
        )
        message = (
            "Multiple dispatch failed for 'torch.ravel'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{qualified_type}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            torch.ravel(DecliningOverride())

        invalid_mode = RecordingMode()
        with invalid_mode:
            with self.assertRaisesRegex(
                TypeError,
                r'^ravel\(\) missing 1 required positional arguments: "input"$',
            ):
                torch.ravel()
        self.assertEqual(invalid_mode.calls, [])

    def test_callable_ownership_documentation_exports_and_pickling(self):
        function = torch.ravel
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "ravel")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.ravel")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
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
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("ravel"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["ravel"], function)

        descriptor = inspect.getattr_static(torch.Tensor, "ravel")
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertEqual(
            descriptor.__doc__, "\nravel() -> Tensor\n\nsee :func:`torch.ravel`\n"
        )

    def test_binding_and_type_errors(self):
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
                lambda: torch.ravel([1.0]),
                "ravel(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.ravel(input=1),
                "ravel(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.ravel(tensor, input=tensor),
                "ravel() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.ravel(tensor, extra=True),
                "ravel() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.ravel(extra=tensor, input=tensor),
                "ravel() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.ravel(extra=tensor),
                'ravel() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.ravel(a=tensor, x=tensor),
                "ravel() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.ravel(x=tensor, a=tensor),
                "ravel() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.ravel(input=tensor, x1=tensor),
                "ravel() got an unexpected keyword argument 'x1'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
