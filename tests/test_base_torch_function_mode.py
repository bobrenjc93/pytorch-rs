import copy
import inspect
import pickle
import re
import unittest

import torch_rs as torch


class BaseTorchFunctionModeTests(unittest.TestCase):
    def test_class_is_public_but_not_wildcard_exported(self):
        mode_type = torch.overrides.BaseTorchFunctionMode

        self.assertEqual(mode_type.__name__, "BaseTorchFunctionMode")
        self.assertEqual(mode_type.__qualname__, "BaseTorchFunctionMode")
        self.assertEqual(mode_type.__module__, "torch_rs.overrides")
        self.assertEqual(mode_type.__bases__, (torch.overrides.TorchFunctionMode,))
        self.assertTrue(issubclass(mode_type, torch.overrides.TorchFunctionMode))
        self.assertIsNone(mode_type.__doc__)
        self.assertEqual(str(inspect.signature(mode_type)), "() -> None")
        self.assertEqual(
            str(inspect.signature(mode_type.__torch_function__)),
            "(self, func, types, args=(), kwargs=None)",
        )
        self.assertEqual(mode_type.__torch_function__.__defaults__, ((), None))
        self.assertEqual(mode_type.__torch_function__.__annotations__, {})

        self.assertNotIn("BaseTorchFunctionMode", torch.overrides.__all__)
        namespace = {}
        exec("from torch_rs.overrides import *", namespace)
        self.assertNotIn("BaseTorchFunctionMode", namespace)

        mode = mode_type()
        self.assertIs(type(mode), mode_type)
        self.assertEqual(mode.__dict__, {})
        with self.assertRaisesRegex(
            TypeError,
            re.escape(
                "TorchFunctionMode.__init__() takes 1 positional argument but 2 "
                "were given"
            ),
        ):
            mode_type(1)
        with self.assertRaisesRegex(
            TypeError,
            re.escape(
                "TorchFunctionMode.__init__() got an unexpected keyword argument "
                "'value'"
            ),
        ):
            mode_type(value=1)

    def test_copy_deepcopy_and_pickle_preserve_instance_state(self):
        mode = torch.overrides.BaseTorchFunctionMode()
        mode.payload = [1, 2, 3]

        shallow = copy.copy(mode)
        self.assertIs(type(shallow), type(mode))
        self.assertIsNot(shallow, mode)
        self.assertIs(shallow.payload, mode.payload)

        deep = copy.deepcopy(mode)
        self.assertIs(type(deep), type(mode))
        self.assertIsNot(deep, mode)
        self.assertEqual(deep.payload, mode.payload)
        self.assertIsNot(deep.payload, mode.payload)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(mode, protocol=protocol))
                self.assertIs(type(restored), type(mode))
                self.assertIsNot(restored, mode)
                self.assertEqual(restored.__dict__, mode.__dict__)
                self.assertIsNot(restored.payload, mode.payload)

    def test_direct_handler_transparently_calls_the_supplied_function(self):
        mode = torch.overrides.BaseTorchFunctionMode()
        marker = object()
        calls = []

        def target(*args, **kwargs):
            calls.append((args, kwargs))
            return marker

        self.assertIs(
            mode.__torch_function__(target, (str,), (1, 2), None),
            marker,
        )
        self.assertEqual(calls, [((1, 2), {})])

        self.assertIs(
            mode.__torch_function__(target, (), (), {"answer": 42}),
            marker,
        )
        self.assertEqual(calls[-1], ((), {"answer": 42}))

        error = RuntimeError("target failed")

        def raising_target():
            raise error

        with self.assertRaises(RuntimeError) as raised:
            mode.__torch_function__(raising_target, (), (), None)
        self.assertIs(raised.exception, error)

    def test_function_method_and_property_calls_forward_through_nested_modes(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expected_adjoint = tensor.mH
        events = []

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(
                    (
                        func.__name__,
                        tuple(dispatch_type.__name__ for dispatch_type in types),
                        tuple(argument is tensor for argument in args),
                        kwargs,
                        tuple(torch.overrides._get_current_function_mode_stack()),
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = RecordingMode()
        upper = torch.overrides.BaseTorchFunctionMode()
        with lower:
            with upper:
                self.assertIs(torch.positive(tensor), tensor)
                self.assertTrue(tensor.adjoint().is_set_to(expected_adjoint))
                self.assertIs(tensor.real, tensor)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(),
                    [lower, upper],
                )

        self.assertEqual(
            events,
            [
                ("positive", (), (True,), {}, ()),
                ("adjoint", ("Tensor",), (True,), None, ()),
                ("__get__", ("Tensor",), (True,), None, ()),
            ],
        )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_dispatch_exceptions_restore_every_active_mode(self):
        tensor = torch.tensor([1.0])
        base_mode = torch.overrides.BaseTorchFunctionMode()

        with base_mode:
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape(
                    "tensor.H is only supported on matrices (2-D tensors). Got "
                    "1-D tensor."
                ),
            ):
                tensor.H
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [base_mode]
            )
            self.assertIs(torch.positive(tensor), tensor)

        error = ValueError("lower mode failed")

        class RaisingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise error

        lower = RaisingMode()
        with lower:
            with base_mode:
                with self.assertRaises(ValueError) as raised:
                    torch.positive(tensor)
                self.assertIs(raised.exception, error)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(),
                    [lower, base_mode],
                )
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [lower]
            )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_broader_and_reentrant_dispatch_helpers_remain_unsupported(self):
        unsupported = {
            "enable_reentrant_dispatch",
            "handle_torch_function",
            "has_torch_function",
            "redispatch_function",
        }
        self.assertTrue(unsupported.isdisjoint(torch.overrides.__all__))
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.overrides, name))


if __name__ == "__main__":
    unittest.main()
