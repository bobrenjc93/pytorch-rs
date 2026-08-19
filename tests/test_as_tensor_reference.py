import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AsTensorReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("as_tensor differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_identity_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
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
        return (
            module.tensor(-0.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            strided[1],
            strided,
            module.tensor(memoryview(special_bits.view(np.float32))),
        )

    def tensor_bits(self, module, tensor):
        if module is reference_torch:
            return tensor.detach().cpu().numpy().reshape(-1).view(np.uint32)
        return np.asarray(tensor.detach()).reshape(-1).view(np.uint32)

    def test_identity_layout_storage_and_bits_match_pytorch_2_13(self):
        actual_cases = self.make_identity_cases(torch)
        expected_cases = self.make_identity_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                actual_pointer = actual.data_ptr()
                expected_pointer = expected.data_ptr()
                actual_result = torch.as_tensor(
                    actual, dtype=torch.float32, device=torch.device("cpu")
                )
                expected_result = reference_torch.as_tensor(
                    expected,
                    dtype=reference_torch.float32,
                    device=reference_torch.device("cpu"),
                )

                self.assertIs(actual_result, actual)
                self.assertIs(expected_result, expected)
                self.assertEqual(
                    (
                        tuple(actual_result.shape),
                        actual_result.stride(),
                        actual_result.storage_offset(),
                        str(actual_result.dtype).replace("torch_rs", "torch"),
                        str(actual_result.device),
                        str(actual_result.layout),
                        actual_result.requires_grad,
                        actual_result.is_leaf,
                    ),
                    (
                        tuple(expected_result.shape),
                        expected_result.stride(),
                        expected_result.storage_offset(),
                        str(expected_result.dtype),
                        str(expected_result.device),
                        str(expected_result.layout),
                        expected_result.requires_grad,
                        expected_result.is_leaf,
                    ),
                )
                self.assertEqual(actual_result.data_ptr(), actual_pointer)
                self.assertEqual(expected_result.data_ptr(), expected_pointer)
                np.testing.assert_array_equal(
                    self.tensor_bits(torch, actual_result),
                    self.tensor_bits(reference_torch, expected_result),
                )

    def autograd_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        source = (leaf * 3.0).transpose(0, 1)[1]
        pointer = source.data_ptr()
        result = module.as_tensor(source, dtype=module.float32, device="cpu")
        identity = result is source
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == pointer,
        )
        result.sum().backward()
        return identity, metadata, leaf.grad.tolist()

    def test_autograd_identity_matches_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_outcome(torch), self.autograd_outcome(reference_torch)
        )

    def callable_contract(self, module):
        function = module.as_tensor
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.as_tensor is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("as_tensor"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["as_tensor"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def test_supported_binding_and_validation_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.as_tensor(), lambda: reference_torch.as_tensor()),
            (
                lambda: torch.as_tensor(actual, None),
                lambda: reference_torch.as_tensor(expected, None),
            ),
            (
                lambda: torch.as_tensor(actual, data=actual),
                lambda: reference_torch.as_tensor(expected, data=expected),
            ),
            (
                lambda: torch.as_tensor(input=actual),
                lambda: reference_torch.as_tensor(input=expected),
            ),
            (
                lambda: torch.as_tensor(actual, unexpected=True),
                lambda: reference_torch.as_tensor(expected, unexpected=True),
            ),
            (
                lambda: torch.as_tensor(actual, dtype=object()),
                lambda: reference_torch.as_tensor(expected, dtype=object()),
            ),
            (
                lambda: torch.as_tensor(actual, device=object()),
                lambda: reference_torch.as_tensor(expected, device=object()),
            ),
            (
                lambda: torch.as_tensor(actual, device=True),
                lambda: reference_torch.as_tensor(expected, device=True),
            ),
            (
                lambda: torch.as_tensor(actual, device=""),
                lambda: reference_torch.as_tensor(expected, device=""),
            ),
            (
                lambda: torch.as_tensor(actual, device="not-a-device"),
                lambda: reference_torch.as_tensor(expected, device="not-a-device"),
            ),
            (
                lambda: torch.as_tensor(actual, device="cpu:-1"),
                lambda: reference_torch.as_tensor(expected, device="cpu:-1"),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        self.assertIs(torch.as_tensor(actual, device="cpu:255"), actual)
        self.assertIs(
            reference_torch.as_tensor(expected, device="cpu:255"), expected
        )

    def mode_observation(self, module, form):
        tensor = module.tensor([1.0])
        marker = object()

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = Mode()
        with mode:
            if form == "positional tensor":
                result = module.as_tensor(tensor)
            elif form == "data keyword":
                result = module.as_tensor(data=tensor)
            elif form == "list":
                result = module.as_tensor([1.0])
            else:
                result = module.as_tensor(tensor, device="not-a-device")
        function, dispatch_types, args, kwargs = mode.calls[0]
        return (
            result is marker,
            function is module.as_tensor,
            dispatch_types,
            tuple(type(argument).__name__ for argument in args),
            None
            if kwargs is None
            else tuple((key, type(value).__name__) for key, value in kwargs.items()),
        )

    def test_mode_dispatch_matches_pytorch_2_13(self):
        for form in ("positional tensor", "data keyword", "list", "bad device"):
            with self.subTest(form=form):
                self.assertEqual(
                    self.mode_observation(torch, form),
                    self.mode_observation(reference_torch, form),
                )

        for module in (torch, reference_torch):
            tensor = module.tensor([1.0])
            order = []

            class ForwardingMode(module.overrides.TorchFunctionMode):
                def __init__(self, label):
                    self.label = label

                def __torch_function__(self, func, types, args=(), kwargs=None):
                    order.append(self.label)
                    return func(*args, **(kwargs or {}))

            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    self.assertIs(module.as_tensor(tensor), tensor)
            self.assertEqual(order, ["upper", "lower"])

    def test_conversion_boundaries_are_explicit(self):
        for data in ([1.0], np.asarray([1.0], dtype=np.float32)):
            expected = reference_torch.as_tensor(data)
            self.assertEqual(expected.tolist(), [1.0])
            with self.assertRaisesRegex(
                TypeError,
                "^as_tensor\\(\\): only existing Tensor inputs are supported; "
                "conversion from (list|numpy\\.ndarray) is not implemented$",
            ):
                torch.as_tensor(data)

        actual = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1)[1]
        expected = reference_torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        ).transpose(0, 1)[1]
        copied = reference_torch.as_tensor(expected, device="cpu:0")
        self.assertIsNot(copied, expected)
        self.assertNotEqual(copied.data_ptr(), expected.data_ptr())
        self.assertTrue(copied.requires_grad)
        self.assertFalse(copied.is_leaf)
        with self.assertRaisesRegex(
            RuntimeError,
            "^as_tensor\\(\\): indexed CPU targets require a copy, "
            "but conversions are not supported$",
        ):
            torch.as_tensor(actual, device="cpu:0")

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "requires an available CUDA device",
    )
    def test_cuda_conversion_boundary_against_reference_runtime(self):
        actual_leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        actual = (actual_leaf * 2.0).transpose(0, 1)[1]
        expected_leaf = reference_torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        expected = (expected_leaf * 2.0).transpose(0, 1)[1]

        converted = reference_torch.as_tensor(expected, device="cuda:0")
        self.assertIsNot(converted, expected)
        self.assertEqual(converted.device.type, "cuda")
        self.assertEqual(converted.device.index, 0)
        self.assertEqual(converted.tolist(), expected.tolist())
        self.assertTrue(converted.requires_grad)
        self.assertFalse(converted.is_leaf)
        converted.sum().backward()
        self.assertEqual(expected_leaf.grad.tolist(), [[0.0, 2.0], [0.0, 2.0]])

        for device in ("cuda", "cuda:0", 0):
            with self.subTest(device=device):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^as_tensor\\(\\): non-CPU device targets require a copy, "
                    "but conversions are not supported$",
                ):
                    torch.as_tensor(actual, device=device)


if __name__ == "__main__":
    unittest.main()
