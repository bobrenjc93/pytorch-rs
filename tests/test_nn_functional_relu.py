import importlib
import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn as nn
import torch_rs.nn.functional as functional


FUNCTION_DOC = """relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """

if sys.version_info >= (3, 13):
    FUNCTION_DOC = (
        "relu(input, inplace=False) -> Tensor\n\n"
        "Applies the rectified linear unit function element-wise. See\n"
        ":class:`~torch.nn.ReLU` for more details.\n"
    )


class FunctionalReluTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        np.testing.assert_array_equal(
            np.asarray(actual).reshape(-1).view(np.uint32),
            np.asarray(expected).reshape(-1).view(np.uint32),
        )

    def test_imports_signature_and_documentation(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        imported_functional = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn import functional as from_nn
        from torch_rs.nn.functional import relu

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(nn.functional, functional)
        self.assertIs(functional, imported_functional)
        self.assertIs(from_nn, functional)
        self.assertIs(relu, functional.relu)
        self.assertNotIn("nn", torch.__all__)
        self.assertFalse(hasattr(nn, "__all__"))
        self.assertFalse(hasattr(functional, "__all__"))
        self.assertIsNone(nn.__doc__)
        self.assertEqual(functional.__doc__, "Functional interface.")

        function = functional.relu
        signature = inspect.signature(function)
        parameters = tuple(signature.parameters.values())
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "relu")
        self.assertEqual(function.__qualname__, "relu")
        self.assertEqual(function.__module__, "torch_rs.nn.functional")
        self.assertEqual(function.__defaults__, (False,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(tuple(signature.parameters), ("input", "inplace"))
        self.assertIs(parameters[0].annotation, torch.Tensor)
        self.assertIs(parameters[1].annotation, bool)
        self.assertIs(parameters[1].default, False)
        self.assertIs(signature.return_annotation, torch.Tensor)

    def test_out_of_place_forms_delegate_to_native_relu(self):
        storage = torch.tensor(
            [
                [9.0, 9.0, 9.0, 9.0],
                [-1.0, 2.0, -0.0, 3.0],
                [4.0, -5.0, 6.0, -7.0],
            ]
        )
        offset = storage[1]
        cases = (
            torch.tensor(-0.0),
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
            offset,
            offset.reshape(2, 2).transpose(0, 1),
        )
        for case, source in enumerate(cases):
            expected = torch.relu(source)
            calls = (
                lambda: functional.relu(source),
                lambda: functional.relu(source, False),
                lambda: functional.relu(input=source, inplace=False),
            )
            for form, call in enumerate(calls):
                with self.subTest(case=case, form=form):
                    actual = call()
                    self.assertIsNot(actual, source)
                    self.assertFalse(actual.is_set_to(source))
                    self.assert_tensor_matches(actual, expected)

    def test_autograd_and_no_grad_reuse_native_behavior(self):
        leaf = torch.tensor(
            [
                [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
                [[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]],
            ],
            requires_grad=True,
        )
        source = leaf[1].transpose(0, 1)
        output = functional.relu(source)
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.stride(), (1, 3))
        output.sum().backward()
        self.assertEqual(
            leaf.grad.tolist(),
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            ],
        )

        untracked_leaf = torch.tensor(
            [[-1.0, 2.0], [0.0, 3.0]], requires_grad=True
        )
        with torch.no_grad():
            untracked = functional.relu(untracked_leaf.transpose(0, 1))
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.tolist(), [[0.0, 0.0], [2.0, 3.0]])
        self.assertIsNone(untracked_leaf.grad)

    def test_torch_function_overrides_receive_the_public_normalized_call(self):
        replacement = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return replacement

        source = Override()
        calls = (
            (lambda: functional.relu(source), False),
            (lambda: functional.relu(source, True), True),
            (
                lambda: functional.relu(input=source, inplace=True),
                True,
            ),
        )
        for case, (call, expected_inplace) in enumerate(calls):
            with self.subTest(case=case):
                self.assertIs(call(), replacement)
                func, dispatch_types, args, kwargs = Override.calls[-1]
                self.assertIs(func, functional.relu)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(args, (source,))
                self.assertEqual(kwargs, {"inplace": expected_inplace})

        events = []

        class Mode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(("mode", func, types, args, kwargs))
                return NotImplemented

        class ModeFallback:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("override", func, types, args, kwargs))
                return replacement

        mode = Mode()
        fallback = ModeFallback()
        with mode:
            self.assertIs(functional.relu(fallback, inplace=True), replacement)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [mode]
            )
        self.assertEqual(
            tuple(event[0] for event in events), ("mode", "override")
        )
        for _, func, dispatch_types, args, kwargs in events:
            self.assertIs(func, functional.relu)
            self.assertEqual(dispatch_types, (ModeFallback,))
            self.assertEqual(args, (fallback,))
            self.assertEqual(kwargs, {"inplace": True})
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            "^no implementation found for "
            "'torch_rs.nn.functional.relu' on types that implement "
            "__torch_function__:",
        ):
            functional.relu(DecliningOverride())

        expected_error = ValueError("relu override failed")

        class RaisingOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise expected_error

        restoring_mode = Mode()
        with restoring_mode:
            with self.assertRaises(ValueError) as raised:
                functional.relu(RaisingOverride())
            self.assertIs(raised.exception, expected_error)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [restoring_mode],
            )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_torch_function_modes_forward_decline_raise_and_restore_stack(self):
        source = torch.tensor([-1.0, 2.0], requires_grad=True)
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func,
                        types,
                        args,
                        kwargs,
                        len(
                            torch.overrides._get_current_function_mode_stack()
                        ),
                    )
                )
                return self.result

        accepting = RecordingMode(marker)
        with accepting:
            self.assertIs(
                functional.relu(input=source, inplace=True), marker
            )
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [accepting],
            )
        func, dispatch_types, args, kwargs, stack_depth = accepting.calls[0]
        self.assertIs(func, functional.relu)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(args, (source,))
        self.assertEqual(kwargs, {"inplace": True})
        self.assertEqual(stack_depth, 0)

        forwarding_events = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_events.append(
                    (
                        self.label,
                        func,
                        types,
                        args,
                        kwargs,
                        len(
                            torch.overrides._get_current_function_mode_stack()
                        ),
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                forwarded = functional.relu(source, False)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(),
                    [lower, upper],
                )
        self.assertEqual(
            tuple(event[0] for event in forwarding_events),
            ("upper", "lower"),
        )
        self.assertEqual(tuple(event[-1] for event in forwarding_events), (1, 0))
        for _, func, dispatch_types, args, kwargs, _ in forwarding_events:
            self.assertIs(func, functional.relu)
            self.assertEqual(dispatch_types, (torch.Tensor,))
            self.assertEqual(args, (source,))
            self.assertEqual(kwargs, {"inplace": False})
        self.assertEqual(forwarded.tolist(), [0.0, 2.0])
        forwarded.sum().backward()
        self.assertEqual(source.grad.tolist(), [0.0, 1.0])

        declining = RecordingMode(NotImplemented)
        with declining:
            with self.assertRaisesRegex(
                TypeError,
                "^no implementation found for "
                "'torch_rs.nn.functional.relu' on types that implement "
                r"__torch_function__: \[\] nor in mode ",
            ):
                functional.relu(source)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [declining],
            )
        self.assertEqual(
            tuple(call[1] for call in declining.calls),
            ((torch.Tensor,), ()),
        )
        self.assertEqual(tuple(call[-1] for call in declining.calls), (0, 0))

        expected_error = ValueError("relu mode failed")

        class RaisingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.stack_depth = None

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.stack_depth = len(
                    torch.overrides._get_current_function_mode_stack()
                )
                raise expected_error

        recovery_events = []

        class RecoveryMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                recovery_events.append(func)
                return func(*args, **(kwargs or {}))

        recovery = RecoveryMode()
        raising = RaisingMode()
        with recovery:
            with self.assertRaises(ValueError) as raised:
                with raising:
                    functional.relu(source)
            self.assertIs(raised.exception, expected_error)
            self.assertEqual(raising.stack_depth, 1)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [recovery],
            )
            recovered = functional.relu(source)
        self.assertEqual(recovery_events, [functional.relu])
        self.assertEqual(recovered.tolist(), [0.0, 2.0])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_inplace_true_fails_before_mutating_the_input(self):
        leaf = torch.tensor(
            [[9.0, 9.0, 9.0], [-1.0, 2.0, -0.0]], requires_grad=True
        )
        source = leaf[1]
        values_before = np.asarray(leaf.detach()).copy().view(np.uint32)
        metadata_before = (
            source.shape,
            source.stride(),
            source.storage_offset(),
            source.data_ptr(),
            source.requires_grad,
            source.is_leaf,
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "^torch_rs\\.nn\\.functional\\.relu does not support inplace=True$",
        ):
            functional.relu(source, inplace=True)

        np.testing.assert_array_equal(
            np.asarray(leaf.detach()).view(np.uint32), values_before
        )
        self.assertEqual(
            (
                source.shape,
                source.stride(),
                source.storage_offset(),
                source.data_ptr(),
                source.requires_grad,
                source.is_leaf,
            ),
            metadata_before,
        )
        self.assertIsNone(leaf.grad)


if __name__ == "__main__":
    unittest.main()
