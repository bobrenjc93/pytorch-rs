import gc
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


class TopLevelDetachTests(unittest.TestCase):
    def assert_detached(self, source, result):
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.stride(), source.stride())
        self.assertEqual(result.storage_offset(), source.storage_offset())
        self.assertEqual(result.is_contiguous(), source.is_contiguous())
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        self.assertEqual(result.tolist(), source.tolist())
        self.assertTrue(source.is_set_to(result))
        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)
        self.assertFalse((result + 1.0).requires_grad)

    def detach_calls(self, source):
        return (
            ("positional", torch.detach(source)),
            ("input", torch.detach(input=source)),
            ("x", torch.detach(x=source)),
            ("a", torch.detach(a=source)),
            ("x1", torch.detach(x1=source)),
        )

    def test_all_call_forms_use_the_native_detach_engine(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        cases = (
            ("scalar", torch.tensor(-0.0, requires_grad=True)),
            (
                "ordinary",
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True),
            ),
            ("strided-offset", (leaf * 2.0).transpose(0, 1)[1]),
            (
                "empty-offset",
                torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1],
            ),
        )

        for case, source in cases:
            method_output = source.detach()
            for form, output in self.detach_calls(source):
                with self.subTest(case=case, form=form):
                    self.assert_detached(source, output)
                    self.assertEqual(output.tolist(), method_output.tolist())
            self.assertTrue(source.requires_grad)

        detached_zero = torch.detach(torch.tensor(-0.0))
        self.assertEqual(np.asarray(detached_zero).view(np.uint32).item(), 0x8000_0000)

    def test_detached_alias_outlives_temporary_source_owners(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retain_offset_view():
            temporary = torch.tensor(values.tolist(), requires_grad=True)
            return torch.detach(temporary[1])

        def retain_strided_view():
            temporary = torch.tensor(values.tolist(), requires_grad=True)
            return torch.detach(x=temporary.transpose(0, 2)[1])

        offset = retain_offset_view()
        strided = retain_strided_view()
        gc.collect()
        np.testing.assert_array_equal(np.asarray(offset), values[1])
        np.testing.assert_array_equal(np.asarray(strided), values.transpose(2, 1, 0)[1])

    def test_detach_does_not_consume_or_modify_the_source_graph(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        source = (leaf * 3.0).transpose(0, 1)[1]
        detached = torch.detach(source)

        self.assertTrue(source.requires_grad)
        self.assertFalse(detached.requires_grad)
        detached_loss = (detached * detached).sum()
        self.assertFalse(detached_loss.requires_grad)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            detached_loss.backward()

        source.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0], [0.0, 3.0]])

    def test_overrides_and_modes_receive_original_top_level_calls(self):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        value = Override()
        calls = (
            ("positional", lambda: torch.detach(value), (value,), None),
            ("input", lambda: torch.detach(input=value), (), {"input": value}),
            ("x", lambda: torch.detach(x=value), (), {"x": value}),
            ("a", lambda: torch.detach(a=value), (), {"a": value}),
            ("x1", lambda: torch.detach(x1=value), (), {"x1": value}),
        )
        for form, call, expected_args, expected_kwargs in calls:
            with self.subTest(kind="override", form=form):
                Override.calls.clear()
                self.assertIs(call(), marker)
                self.assertEqual(len(Override.calls), 1)
                function, dispatch_types, args, kwargs = Override.calls[0]
                self.assertIs(function, torch.detach)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        tensor = torch.tensor([1.0], requires_grad=True)
        mode_calls = (
            ("positional", lambda: torch.detach(tensor), (tensor,), None),
            ("input", lambda: torch.detach(input=tensor), (), {"input": tensor}),
            ("x", lambda: torch.detach(x=tensor), (), {"x": tensor}),
            ("a", lambda: torch.detach(a=tensor), (), {"a": tensor}),
            ("x1", lambda: torch.detach(x1=tensor), (), {"x1": tensor}),
        )
        for form, call, expected_args, expected_kwargs in mode_calls:
            with self.subTest(kind="mode", form=form):
                mode = RecordingMode()
                with mode:
                    self.assertIs(call(), marker)
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs = mode.calls[0]
                self.assertIs(function, torch.detach)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.detach(x=tensor)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_detached(tensor, forwarded)

        declining_mode = RecordingMode(NotImplemented)
        Override.calls.clear()
        with declining_mode:
            self.assertIs(torch.detach(value), marker)
        self.assertEqual(len(declining_mode.calls), 1)
        self.assertEqual(len(Override.calls), 1)

    def test_not_implemented_errors_name_every_declining_handler(self):
        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        override_message = (
            "Multiple dispatch failed for 'torch.detach'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            "  - tensor subclass <class "
            f"'{DecliningOverride.__module__}.{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(override_message)}$"):
            torch.detach(DecliningOverride())

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __repr__(self):
                return "declining-detach-mode"

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        mode_message = (
            "Multiple dispatch failed for 'torch.detach'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            "  - mode object declining-detach-mode\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented"
        )
        with DecliningMode():
            with self.assertRaisesRegex(TypeError, f"^{re.escape(mode_message)}$"):
                torch.detach(torch.tensor([1.0]))

    def test_binding_and_type_errors_match_the_legacy_schema(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.detach(),
                'detach() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.detach(tensor, tensor),
                "detach() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.detach(tensor, input=tensor),
                "detach() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.detach(tensor, x=tensor),
                "detach() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.detach(foo=tensor),
                'detach() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.detach(None),
                "detach(): argument 'input' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: torch.detach(input=1),
                "detach(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.detach(x=tensor, extra=True),
                "detach() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.detach(extra=True, x=tensor),
                "detach() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.detach(input=tensor, x=tensor),
                "detach() got an unexpected keyword argument 'x'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_callable_ownership_pickling_and_exports(self):
        function = torch.detach
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "detach")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.detach")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__doc__)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method detach of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.detach, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("detach"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["detach"], function)

        message = (
            "cannot set 'detach' attribute of immutable type "
            "'torch_rs._C._VariableFunctionsClass'"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            owner.detach = None
        self.assertIs(owner.detach, function)


if __name__ == "__main__":
    unittest.main()
