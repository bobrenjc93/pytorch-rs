import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


class AsTensorTests(unittest.TestCase):
    def identity_calls(self, source):
        return (
            ("positional", torch.as_tensor(source)),
            ("data keyword", torch.as_tensor(data=source)),
            ("dtype None", torch.as_tensor(source, dtype=None)),
            ("float32 dtype", torch.as_tensor(source, dtype=torch.float32)),
            ("float alias", torch.as_tensor(source, dtype=torch.float)),
            ("device None", torch.as_tensor(source, device=None)),
            ("device string", torch.as_tensor(source, device="cpu")),
            ("device object", torch.as_tensor(source, device=torch.device("cpu"))),
            (
                "dtype and device",
                torch.as_tensor(
                    source, dtype=torch.float32, device=torch.device("cpu")
                ),
            ),
        )

    def assert_identity_calls(self, source):
        detached = source.detach()
        metadata = (
            source.shape,
            source.stride(),
            source.storage_offset(),
            source.dtype,
            source.device,
            source.layout,
            source.requires_grad,
            source.is_leaf,
            source.data_ptr(),
        )
        bits = np.asarray(detached).reshape(-1).view(np.uint32).copy()

        for form, result in self.identity_calls(source):
            with self.subTest(form=form):
                self.assertIs(result, source)
                self.assertTrue(result.is_set_to(detached))
                self.assertEqual(
                    (
                        result.shape,
                        result.stride(),
                        result.storage_offset(),
                        result.dtype,
                        result.device,
                        result.layout,
                        result.requires_grad,
                        result.is_leaf,
                        result.data_ptr(),
                    ),
                    metadata,
                )
                np.testing.assert_array_equal(
                    np.asarray(result.detach()).reshape(-1).view(np.uint32), bits
                )

    def test_supported_tensor_layouts_are_exact_identities(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        cases = (
            ("scalar", torch.tensor(-0.0)),
            ("empty offset", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("offset", strided[1]),
            ("strided", strided),
            ("special values", torch.tensor(memoryview(special_bits.view(np.float32)))),
        )

        for case, source in cases:
            with self.subTest(case=case):
                self.assert_identity_calls(source)

    def test_leaf_and_non_leaf_autograd_state_is_preserved(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]

        for case, source in (("leaf", leaf), ("non-leaf", non_leaf)):
            with self.subTest(case=case):
                self.assert_identity_calls(source)

        torch.as_tensor(non_leaf, dtype=torch.float32, device="cpu").sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        gradient = leaf.grad
        self.assertIs(torch.as_tensor(leaf).grad, gradient)

    def test_callable_metadata_documentation_exports_and_pickling(self):
        function = torch.as_tensor
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "as_tensor")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.as_tensor")
        self.assertEqual(function.__module__, "torch")
        self.assertTrue(
            function.__doc__.startswith(
                "\nas_tensor(data: Any, *, dtype: Optional[dtype] = None, "
                "device: Optional[DeviceLikeType]) -> Tensor\n"
            )
        )
        self.assertIn("sharing data and preserving autograd\nhistory", function.__doc__)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method as_tensor of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.as_tensor, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("as_tensor"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["as_tensor"], function)

    def test_call_binding_and_metadata_type_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.as_tensor(),
                'as_tensor() missing 1 required positional arguments: "data"',
            ),
            (
                lambda: torch.as_tensor(tensor, None),
                "as_tensor() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.as_tensor(tensor, torch.float32),
                "as_tensor() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.as_tensor(tensor, data=tensor),
                "as_tensor() got multiple values for argument 'data'",
            ),
            (
                lambda: torch.as_tensor(input=tensor),
                'as_tensor() missing 1 required positional arguments: "data"',
            ),
            (
                lambda: torch.as_tensor(tensor, unexpected=True),
                "as_tensor() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: torch.as_tensor(tensor, dtype=object()),
                "as_tensor(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: torch.as_tensor(tensor, device=object()),
                "as_tensor(): argument 'device' must be torch.device, not object",
            ),
            (
                lambda: torch.as_tensor(tensor, device=True),
                "as_tensor(): argument 'device' must be torch.device, not bool",
            ),
            (
                lambda: torch.as_tensor(tensor, device=""),
                "Device string must not be empty",
            ),
            (
                lambda: torch.as_tensor(tensor, device="cpu:-1"),
                "Invalid device string: 'cpu:-1'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(Exception) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_non_tensor_and_copying_paths_are_explicitly_rejected(self):
        for value, type_name in (([1.0], "list"), (1.0, "float"), (None, "NoneType")):
            with self.subTest(type_name=type_name):
                with self.assertRaisesRegex(
                    TypeError,
                    "^"
                    + re.escape(
                        "as_tensor(): only existing Tensor inputs are supported; "
                        f"conversion from {type_name} is not implemented"
                    )
                    + "$",
                ):
                    torch.as_tensor(value)

        tensor = torch.tensor([1.0])
        for device in ("cpu:0", torch.device("cpu", 0)):
            with self.subTest(device=device):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^as_tensor\\(\\): indexed CPU targets require a copy, "
                    "but conversions are not supported$",
                ):
                    torch.as_tensor(tensor, device=device)

        for device in ("cuda", "cuda:0", 0):
            with self.subTest(device=device):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^as_tensor\\(\\): non-CPU device targets require a copy, "
                    "but conversions are not supported$",
                ):
                    torch.as_tensor(tensor, device=device)

        self.assertIs(torch.as_tensor(tensor, device="cpu:255"), tensor)

    def test_non_tensor_torch_function_objects_do_not_dispatch(self):
        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        with self.assertRaisesRegex(
            TypeError,
            "^as_tensor\\(\\): only existing Tensor inputs are supported; "
            "conversion from Override is not implemented$",
        ):
            torch.as_tensor(Override())
        self.assertEqual(Override.calls, [])

    def test_torch_function_modes_receive_original_calls(self):
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        tensor = torch.tensor([1.0])
        calls = (
            (lambda: torch.as_tensor(tensor), (tensor,), None),
            (lambda: torch.as_tensor(data=tensor), (), {"data": tensor}),
            (lambda: torch.as_tensor([1.0]), ([1.0],), None),
            (
                lambda: torch.as_tensor(tensor, device="not-a-device"),
                (tensor,),
                {"device": "not-a-device"},
            ),
        )
        for call, expected_args, expected_kwargs in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.as_tensor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

    def test_mode_validation_forwarding_and_not_implemented_behavior(self):
        tensor = torch.tensor([1.0])

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        invalid_calls = (
            lambda: torch.as_tensor(),
            lambda: torch.as_tensor(tensor, tensor),
            lambda: torch.as_tensor(tensor, dtype=object()),
            lambda: torch.as_tensor(tensor, device=object()),
            lambda: torch.as_tensor(tensor, unexpected=True),
        )
        for call in invalid_calls:
            mode = RecordingMode()
            with mode:
                with self.assertRaises(Exception):
                    call()
            self.assertEqual(mode.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                self.assertIs(torch.as_tensor(tensor), tensor)
        self.assertEqual(order, ["upper", "lower"])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        with DecliningMode():
            with self.assertRaisesRegex(
                TypeError,
                "^Multiple dispatch failed for 'torch.as_tensor'; all "
                "__torch_function__ handlers returned NotImplemented:",
            ):
                torch.as_tensor(tensor)


if __name__ == "__main__":
    unittest.main()
