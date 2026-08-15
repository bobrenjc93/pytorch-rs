import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


def float32_from_bits(bits):
    return np.array(bits, dtype=np.uint32).view(np.float32)[()]


def float32_bits(tensor):
    value = tensor.detach() if tensor.requires_grad else tensor
    return int(np.asarray(value).view(np.uint32).item())


class ScalarTensorTests(unittest.TestCase):
    def assert_error(self, call, error_type, message):
        with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
            call()

    def test_python_and_numpy_real_scalars_support_positional_and_keyword_calls(self):
        cases = (
            (False, 0x00000000),
            (True, 0x3F800000),
            (-2, 0xC0000000),
            (1.25, 0x3FA00000),
            (np.bool_(True), 0x3F800000),
            (np.int8(-3), 0xC0400000),
            (np.uint64(2**63 - 1), 0x5F000000),
            (np.float16(-1.5), 0xBFC00000),
            (np.float64(2.5), 0x40200000),
        )
        for value, expected_bits in cases:
            for form, call in (
                ("positional", lambda value=value: torch.scalar_tensor(value)),
                ("keyword", lambda value=value: torch.scalar_tensor(s=value)),
            ):
                with self.subTest(value=value, form=form):
                    output = call()
                    self.assertEqual(output.shape, ())
                    self.assertIs(output.dtype, torch.float32)
                    self.assertEqual(float32_bits(output), expected_bits)

    def test_float32_signed_zero_and_nonfinite_bits_are_preserved(self):
        cases = (
            (0x00000000, 0x00000000),
            (0x80000000, 0x80000000),
            (0x7F800000, 0x7F800000),
            (0xFF800000, 0xFF800000),
            (0x7FC12345, 0x7FC12345),
            (0xFFC12345, 0xFFC12345),
            (0x7F812345, 0x7FC12345),
            (0xFF812345, 0xFFC12345),
        )
        for input_bits, expected_bits in cases:
            value = float32_from_bits(input_bits)
            for form, output in (
                ("positional", torch.scalar_tensor(value)),
                ("keyword", torch.scalar_tensor(s=value)),
            ):
                with self.subTest(
                    input_bits=f"{input_bits:08x}", form=form
                ):
                    self.assertEqual(float32_bits(output), expected_bits)

    def test_zero_dimensional_tensor_inputs_are_copied_exactly(self):
        for bits in (0x80000000, 0x7F800000, 0xFFC12345):
            source = torch.full((), float32_from_bits(bits))
            for form, output in (
                ("positional", torch.scalar_tensor(source)),
                ("keyword", torch.scalar_tensor(s=source)),
            ):
                with self.subTest(bits=f"{bits:08x}", form=form):
                    self.assertEqual(float32_bits(output), bits)
                    self.assertNotEqual(output.data_ptr(), source.data_ptr())
                    self.assertEqual(output.shape, ())
                    self.assertTrue(output.is_leaf)

    def test_supported_factory_options_create_scalar_leaves(self):
        option_cases = (
            {},
            {"dtype": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"layout": None},
            {"layout": torch.strided},
            {"device": None},
            {"device": "cpu"},
            {"device": "cpu:0"},
            {"device": "cpu:1"},
            {"device": torch.device("cpu")},
            {"pin_memory": None},
            {"pin_memory": False},
            {"requires_grad": None},
            {"requires_grad": False},
        )
        for options in option_cases:
            with self.subTest(options=options):
                output = torch.scalar_tensor(-0.0, **options)
                self.assertEqual(output.shape, ())
                self.assertEqual(output.stride(), ())
                self.assertEqual(output.storage_offset(), 0)
                self.assertEqual(output.numel(), 1)
                self.assertIs(output.dtype, torch.float32)
                self.assertEqual(output.device, torch.device("cpu"))
                self.assertIs(output.layout, torch.strided)
                self.assertTrue(output.is_leaf)
                self.assertFalse(output.requires_grad)
                self.assertEqual(float32_bits(output), 0x80000000)

        leaf = torch.scalar_tensor(2.0, requires_grad=True)
        self.assertTrue(leaf.requires_grad)
        self.assertTrue(leaf.is_leaf)
        self.assertIsNone(leaf.grad)
        (leaf * 3.0).backward()
        self.assertEqual(leaf.grad.item(), 3.0)

    def test_binding_validation_and_error_precedence_match_pytorch(self):
        vector = torch.tensor([1.0])
        grad_scalar = torch.tensor(1.0, requires_grad=True)
        cases = (
            (
                lambda: torch.scalar_tensor(),
                TypeError,
                'scalar_tensor() missing 1 required positional arguments: "s"',
            ),
            (
                lambda: torch.scalar_tensor(dtype=1),
                TypeError,
                'scalar_tensor() missing 1 required positional arguments: "s"',
            ),
            (
                lambda: torch.scalar_tensor(1, 2),
                TypeError,
                "scalar_tensor() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.scalar_tensor(1, s=2),
                TypeError,
                "scalar_tensor() got multiple values for argument 's'",
            ),
            (
                lambda: torch.scalar_tensor(1, unexpected=True),
                TypeError,
                "scalar_tensor() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: torch.scalar_tensor([]),
                TypeError,
                "scalar_tensor(): argument 's' (position 1) must be Number, not list",
            ),
            (
                lambda: torch.scalar_tensor(s=[]),
                TypeError,
                "scalar_tensor(): argument 's' must be Number, not list",
            ),
            (
                lambda: torch.scalar_tensor(vector),
                TypeError,
                "scalar_tensor(): argument 's' (position 1) must be Number, not Tensor",
            ),
            (
                lambda: torch.scalar_tensor(grad_scalar),
                TypeError,
                "scalar_tensor(): argument 's' (position 1) must be Number, not Tensor",
            ),
            (
                lambda: torch.scalar_tensor(1, dtype=1),
                TypeError,
                "scalar_tensor(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: torch.scalar_tensor(1, layout=1),
                TypeError,
                "scalar_tensor(): argument 'layout' must be torch.layout, not int",
            ),
            (
                lambda: torch.scalar_tensor(1, device=1.5),
                TypeError,
                "scalar_tensor(): argument 'device' must be torch.device, not float",
            ),
            (
                lambda: torch.scalar_tensor(1, pin_memory=0),
                TypeError,
                "scalar_tensor(): argument 'pin_memory' must be bool, not int",
            ),
            (
                lambda: torch.scalar_tensor(1, requires_grad=0),
                TypeError,
                "scalar_tensor(): argument 'requires_grad' must be bool, not int",
            ),
            (
                lambda: torch.scalar_tensor(1, device=""),
                RuntimeError,
                "Device string must not be empty",
            ),
            (
                lambda: torch.scalar_tensor(1, device="banana"),
                RuntimeError,
                "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: banana",
            ),
            (
                lambda: torch.scalar_tensor(1, device="cpu:01"),
                RuntimeError,
                "Invalid device string: 'cpu:01'",
            ),
            (
                lambda: torch.scalar_tensor(1e39),
                RuntimeError,
                "value cannot be converted to type float without overflow",
            ),
            (
                lambda: torch.scalar_tensor(2**64),
                OverflowError,
                "int too big to convert",
            ),
            (
                lambda: torch.scalar_tensor(-(2**63) - 1),
                OverflowError,
                "can't convert negative int to unsigned",
            ),
            (
                lambda: torch.scalar_tensor(np.uint64(2**63)),
                TypeError,
                "an integer is required",
            ),
            (
                lambda: torch.scalar_tensor([], dtype=1),
                TypeError,
                "scalar_tensor(): argument 's' (position 1) must be Number, not list",
            ),
            (
                lambda: torch.scalar_tensor(1e39, dtype=1),
                TypeError,
                "scalar_tensor(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: torch.scalar_tensor(1, dtype=1, layout=2),
                TypeError,
                "scalar_tensor(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: torch.scalar_tensor(
                    1, pin_memory=0, requires_grad=0
                ),
                TypeError,
                "scalar_tensor(): argument 'pin_memory' must be bool, not int",
            ),
            (
                lambda: torch.scalar_tensor(
                    1, device="not-a-device", unexpected=True
                ),
                TypeError,
                "scalar_tensor() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: torch.scalar_tensor(1, s=2, dtype=3),
                TypeError,
                "scalar_tensor(): argument 'dtype' must be torch.dtype, not int",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                self.assert_error(call, error_type, message)

    def test_callable_metadata_and_exports_match_pytorch(self):
        function = torch.scalar_tensor
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "scalar_tensor")
        self.assertEqual(
            function.__qualname__, "_VariableFunctionsClass.scalar_tensor"
        )
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__doc__)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method scalar_tensor of type object at 0x[0-9a-f]+>$",
        )
        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.scalar_tensor, function)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

        self.assertEqual(torch.__all__.count("scalar_tensor"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["scalar_tensor"], function)


if __name__ == "__main__":
    unittest.main()
