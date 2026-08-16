import inspect
import types
import unittest

import numpy as np

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SetDefaultDTypeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "set_default_dtype differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def default_dtype_outcome(self, module, dtype):
        canonical = module.float32
        result = module.set_default_dtype(dtype)
        factories = (
            module.tensor(1.25),
            module.scalar_tensor(1.25),
            module.zeros((2, 0, 3)),
            module.ones((2, 3)),
            module.eye(2, 3),
            module.full((2,), 1.25),
        )
        return (
            result is None,
            dtype is canonical,
            module.float is canonical,
            module.get_default_dtype() is canonical,
            tuple(tensor.dtype is canonical for tensor in factories),
            str(module.get_default_dtype()),
        )

    def test_float32_and_alias_noops_match_pytorch_2_13(self):
        for actual_dtype, expected_dtype in (
            (torch.float32, reference_torch.float32),
            (torch.float, reference_torch.float),
            (torch.get_default_dtype(), reference_torch.get_default_dtype()),
        ):
            with self.subTest(dtype=repr(expected_dtype)):
                self.assertEqual(
                    self.default_dtype_outcome(torch, actual_dtype),
                    self.default_dtype_outcome(reference_torch, expected_dtype),
                )

    def rebound_getter_outcome(self, module):
        canonical = module.float32
        original_get_default_dtype = module.get_default_dtype
        marker = object()
        module.get_default_dtype = lambda: marker
        try:
            try:
                module.set_default_dtype(marker)
            except Exception as error:
                invalid_outcome = (type(error).__name__, str(error))
            else:
                invalid_outcome = None
            valid_outcome = module.set_default_dtype(canonical)
        finally:
            module.get_default_dtype = original_get_default_dtype

        return (
            invalid_outcome,
            valid_outcome is None,
            module.get_default_dtype() is canonical,
            module.tensor(1.25).dtype is canonical,
        )

    def test_rebinding_public_getter_matches_pytorch_2_13(self):
        self.assertEqual(
            self.rebound_getter_outcome(torch),
            self.rebound_getter_outcome(reference_torch),
        )

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual = torch.set_default_dtype
        expected = reference_torch.set_default_dtype
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(
            inspect.signature(actual).parameters["d"].kind,
            inspect.signature(expected).parameters["d"].kind,
        )
        self.assertEqual(
            "set_default_dtype" in torch.__all__,
            "set_default_dtype" in reference_torch.__all__,
        )
        self.assertEqual(torch.__all__.count("set_default_dtype"), 0)

    def test_positional_only_binding_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.set_default_dtype(),
                lambda: reference_torch.set_default_dtype(),
            ),
            (
                lambda: torch.set_default_dtype(torch.float32, torch.float32),
                lambda: reference_torch.set_default_dtype(
                    reference_torch.float32, reference_torch.float32
                ),
            ),
            (
                lambda: torch.set_default_dtype(d=torch.float32),
                lambda: reference_torch.set_default_dtype(d=reference_torch.float32),
            ),
            (
                lambda: torch.set_default_dtype(dtype=torch.float32),
                lambda: reference_torch.set_default_dtype(
                    dtype=reference_torch.float32
                ),
            ),
            (
                lambda: torch.set_default_dtype(
                    torch.float32, dtype=torch.float32
                ),
                lambda: reference_torch.set_default_dtype(
                    reference_torch.float32, dtype=reference_torch.float32
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(torch.get_default_dtype(), torch.float32)
                self.assertIs(
                    reference_torch.get_default_dtype(), reference_torch.float32
                )

    def test_invalid_dtype_object_errors_match_pytorch_2_13(self):
        cases = (
            (None, None),
            (True, True),
            (1, 1),
            (1.0, 1.0),
            ("torch.float32", "torch.float32"),
            (object(), object()),
            (torch.dtype, reference_torch.dtype),
            (torch.tensor(1.0), reference_torch.tensor(1.0)),
            (torch.device("cpu"), reference_torch.device("cpu")),
            (np.dtype("float32"), np.dtype("float32")),
            (np.float32, np.float32),
        )
        for case, (actual_value, expected_value) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda value=actual_value: torch.set_default_dtype(value),
                    lambda value=expected_value: reference_torch.set_default_dtype(value),
                )
                self.assertIs(torch.get_default_dtype(), torch.float32)
                self.assertIs(
                    reference_torch.get_default_dtype(), reference_torch.float32
                )


if __name__ == "__main__":
    unittest.main()
