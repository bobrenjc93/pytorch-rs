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


def float32_from_bits(bits):
    return np.array(bits, dtype=np.uint32).view(np.float32)[()]


def float32_bits(tensor):
    value = tensor.detach() if tensor.requires_grad else tensor
    return int(np.asarray(value).view(np.uint32).item())


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ScalarTensorReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "scalar_tensor differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def tensor_contract(self, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "bits": float32_bits(tensor),
        }

    def test_real_scalars_bits_and_argument_forms_match_pytorch_2_13(self):
        values = (
            False,
            True,
            -2,
            1.25,
            np.bool_(True),
            np.int8(-3),
            np.uint64(2**63 - 1),
            np.float16(-1.5),
            np.float64(2.5),
            *(float32_from_bits(bits) for bits in (
                0x00000000,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7FC12345,
                0xFFC12345,
                0x7F812345,
                0xFF812345,
            )),
        )
        for value in values:
            for form in ("positional", "keyword"):
                with self.subTest(value=value, form=form):
                    if form == "positional":
                        actual = torch.scalar_tensor(value)
                        expected = reference_torch.scalar_tensor(value)
                    else:
                        actual = torch.scalar_tensor(s=value)
                        expected = reference_torch.scalar_tensor(s=value)
                    self.assertEqual(
                        self.tensor_contract(actual),
                        self.tensor_contract(expected),
                    )

    def test_zero_dimensional_inputs_and_supported_options_match(self):
        actual_source = torch.full((), float32_from_bits(0xFFC12345))
        expected_source = reference_torch.full(
            (), float32_from_bits(0xFFC12345), dtype=reference_torch.float32
        )
        actual = torch.scalar_tensor(actual_source)
        expected = reference_torch.scalar_tensor(expected_source)
        self.assertEqual(self.tensor_contract(actual), self.tensor_contract(expected))
        self.assertNotEqual(actual.data_ptr(), actual_source.data_ptr())
        self.assertNotEqual(expected.data_ptr(), expected_source.data_ptr())

        option_pairs = (
            ({}, {}),
            ({"dtype": None}, {"dtype": None}),
            ({"dtype": torch.float32}, {"dtype": reference_torch.float32}),
            ({"layout": None}, {"layout": None}),
            ({"layout": torch.strided}, {"layout": reference_torch.strided}),
            ({"device": None}, {"device": None}),
            ({"device": "cpu"}, {"device": "cpu"}),
            ({"device": "cpu:0"}, {"device": "cpu:0"}),
            ({"device": "cpu:1"}, {"device": "cpu:1"}),
            (
                {"device": torch.device("cpu")},
                {"device": reference_torch.device("cpu")},
            ),
            (
                {"device": torch.device("cpu", 2)},
                {"device": reference_torch.device("cpu", 2)},
            ),
            ({"pin_memory": None}, {"pin_memory": None}),
            ({"pin_memory": False}, {"pin_memory": False}),
            ({"requires_grad": None}, {"requires_grad": None}),
            ({"requires_grad": False}, {"requires_grad": False}),
            ({"requires_grad": True}, {"requires_grad": True}),
        )
        for actual_options, expected_options in option_pairs:
            with self.subTest(options=actual_options):
                actual = torch.scalar_tensor(-0.0, **actual_options)
                expected = reference_torch.scalar_tensor(
                    -0.0, **expected_options
                )
                self.assertEqual(
                    self.tensor_contract(actual), self.tensor_contract(expected)
                )

        actual_leaf = torch.scalar_tensor(2.0, requires_grad=True)
        expected_leaf = reference_torch.scalar_tensor(2.0, requires_grad=True)
        (actual_leaf * 3.0).backward()
        (expected_leaf * 3.0).backward()
        self.assertEqual(
            self.tensor_contract(actual_leaf.grad),
            self.tensor_contract(expected_leaf.grad),
        )

    def test_binding_errors_and_validation_precedence_match_pytorch_2_13(self):
        actual_vector = torch.tensor([1.0])
        expected_vector = reference_torch.tensor(
            [1.0], dtype=reference_torch.float32
        )
        actual_grad = torch.tensor(1.0, requires_grad=True)
        expected_grad = reference_torch.tensor(1.0, requires_grad=True)
        call_pairs = (
            (lambda: torch.scalar_tensor(), lambda: reference_torch.scalar_tensor()),
            (
                lambda: torch.scalar_tensor(dtype=1),
                lambda: reference_torch.scalar_tensor(dtype=1),
            ),
            (
                lambda: torch.scalar_tensor(1, 2),
                lambda: reference_torch.scalar_tensor(1, 2),
            ),
            (
                lambda: torch.scalar_tensor(1, s=2),
                lambda: reference_torch.scalar_tensor(1, s=2),
            ),
            (
                lambda: torch.scalar_tensor(1, unexpected=True),
                lambda: reference_torch.scalar_tensor(1, unexpected=True),
            ),
            (lambda: torch.scalar_tensor([]), lambda: reference_torch.scalar_tensor([])),
            (
                lambda: torch.scalar_tensor(s=[]),
                lambda: reference_torch.scalar_tensor(s=[]),
            ),
            (
                lambda: torch.scalar_tensor(actual_vector),
                lambda: reference_torch.scalar_tensor(expected_vector),
            ),
            (
                lambda: torch.scalar_tensor(actual_grad),
                lambda: reference_torch.scalar_tensor(expected_grad),
            ),
            (
                lambda: torch.scalar_tensor(1, dtype=1),
                lambda: reference_torch.scalar_tensor(1, dtype=1),
            ),
            (
                lambda: torch.scalar_tensor(1, layout=1),
                lambda: reference_torch.scalar_tensor(1, layout=1),
            ),
            (
                lambda: torch.scalar_tensor(1, device=1.5),
                lambda: reference_torch.scalar_tensor(1, device=1.5),
            ),
            (
                lambda: torch.scalar_tensor(1, pin_memory=0),
                lambda: reference_torch.scalar_tensor(1, pin_memory=0),
            ),
            (
                lambda: torch.scalar_tensor(1, requires_grad=0),
                lambda: reference_torch.scalar_tensor(1, requires_grad=0),
            ),
            (
                lambda: torch.scalar_tensor(1, device=""),
                lambda: reference_torch.scalar_tensor(1, device=""),
            ),
            (
                lambda: torch.scalar_tensor(1, device="banana"),
                lambda: reference_torch.scalar_tensor(1, device="banana"),
            ),
            (
                lambda: torch.scalar_tensor(1, device="cpu:01"),
                lambda: reference_torch.scalar_tensor(1, device="cpu:01"),
            ),
            (lambda: torch.scalar_tensor(1e39), lambda: reference_torch.scalar_tensor(1e39)),
            (
                lambda: torch.scalar_tensor(2**64),
                lambda: reference_torch.scalar_tensor(2**64),
            ),
            (
                lambda: torch.scalar_tensor(-(2**63) - 1),
                lambda: reference_torch.scalar_tensor(-(2**63) - 1),
            ),
            (
                lambda: torch.scalar_tensor(np.uint64(2**63)),
                lambda: reference_torch.scalar_tensor(np.uint64(2**63)),
            ),
            (
                lambda: torch.scalar_tensor([], dtype=1),
                lambda: reference_torch.scalar_tensor([], dtype=1),
            ),
            (
                lambda: torch.scalar_tensor(1e39, dtype=1),
                lambda: reference_torch.scalar_tensor(1e39, dtype=1),
            ),
            (
                lambda: torch.scalar_tensor(1, dtype=1, layout=2),
                lambda: reference_torch.scalar_tensor(1, dtype=1, layout=2),
            ),
            (
                lambda: torch.scalar_tensor(
                    1, pin_memory=0, requires_grad=0
                ),
                lambda: reference_torch.scalar_tensor(
                    1, pin_memory=0, requires_grad=0
                ),
            ),
            (
                lambda: torch.scalar_tensor(
                    1, device="not-a-device", unexpected=True
                ),
                lambda: reference_torch.scalar_tensor(
                    1, device="not-a-device", unexpected=True
                ),
            ),
            (
                lambda: torch.scalar_tensor(1, s=2, dtype=3),
                lambda: reference_torch.scalar_tensor(1, s=2, dtype=3),
            ),
        )
        for actual_call, expected_call in call_pairs:
            with self.subTest(actual_call=actual_call):
                self.assert_error_matches(actual_call, expected_call)

    def callable_contract(self, module):
        function = module.scalar_tensor
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
            "owner_callable_identity": owner.scalar_tensor is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("scalar_tensor"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "wildcard_identity": wildcard_namespace["scalar_tensor"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
