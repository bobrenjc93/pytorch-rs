import inspect
import types
import unittest

import numpy as np

import torch_rs as torch


FUNCTION_DOC = r"""

    Sets the default floating point dtype to :attr:`d`. Supports floating point dtype
    as inputs. Other dtypes will cause torch to raise an exception.

    When PyTorch is initialized its default floating point dtype is torch.float32,
    and the intent of set_default_dtype(torch.float64) is to facilitate NumPy-like
    type inference. The default floating point dtype is used to:

    1. Implicitly determine the default complex dtype. When the default floating type is float16,
       the default complex dtype is complex32. For float32, the default complex dtype is complex64.
       For float64, it is complex128. For bfloat16, an exception will be raised because
       there is no corresponding complex type for bfloat16.
    2. Infer the dtype for tensors constructed using Python floats or complex Python
       numbers. See examples below.
    3. Determine the result of type promotion between bool and integer tensors and
       Python floats and complex Python numbers.

    Args:
        d (:class:`torch.dtype`): the floating point dtype to make the default.

    Example:
        >>> # xdoctest: +SKIP("Other tests may have changed the default type. Can we reset it?")
        >>> # initial default for floating point is torch.float32
        >>> # Python floats are interpreted as float32
        >>> torch.tensor([1.2, 3]).dtype
        torch.float32
        >>> # initial default for floating point is torch.complex64
        >>> # Complex Python numbers are interpreted as complex64
        >>> torch.tensor([1.2, 3j]).dtype
        torch.complex64

        >>> torch.set_default_dtype(torch.float64)
        >>> # Python floats are now interpreted as float64
        >>> torch.tensor([1.2, 3]).dtype  # a new floating point tensor
        torch.float64
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype  # a new complex tensor
        torch.complex128

        >>> torch.set_default_dtype(torch.float16)
        >>> # Python floats are now interpreted as float16
        >>> torch.tensor([1.2, 3]).dtype  # a new floating point tensor
        torch.float16
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype  # a new complex tensor
        torch.complex32

    """


class SetDefaultDTypeTests(unittest.TestCase):
    def assert_default_factories_are_canonical(self):
        canonical = torch.float32
        self.assertIs(torch.get_default_dtype(), canonical)
        self.assertIs(torch.float, canonical)
        factories = (
            torch.tensor(1.25),
            torch.scalar_tensor(1.25),
            torch.zeros((2, 0, 3)),
            torch.ones((2, 3)),
            torch.eye(2, 3),
            torch.full((2,), 1.25),
        )
        for tensor in factories:
            with self.subTest(shape=tensor.shape):
                self.assertIs(tensor.dtype, canonical)

    def test_float32_and_its_alias_are_stateless_noops(self):
        canonical = torch.float32
        aliases = (
            torch.float32,
            torch.float,
            torch.get_default_dtype(),
            torch.tensor(-0.0).dtype,
        )
        for alias in aliases:
            with self.subTest(alias=repr(alias)):
                self.assertIs(alias, canonical)
                self.assertIsNone(torch.set_default_dtype(alias))
                self.assert_default_factories_are_canonical()

        self.assertIs(torch.float32, canonical)
        self.assertIs(torch.float, canonical)
        self.assertIs(torch.get_default_dtype(), canonical)

    def test_callable_metadata_matches_pytorch_2_13(self):
        function = torch.set_default_dtype
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "set_default_dtype")
        self.assertEqual(function.__qualname__, "set_default_dtype")
        self.assertEqual(function.__module__, torch.__name__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(
            function.__annotations__,
            {"d": "torch.dtype", "return": None},
        )
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})

        signature = inspect.signature(function)
        self.assertEqual(str(signature), "(d: 'torch.dtype', /) -> None")
        self.assertEqual(
            signature.parameters["d"].kind,
            inspect.Parameter.POSITIONAL_ONLY,
        )
        self.assertEqual(signature.parameters["d"].annotation, "torch.dtype")
        self.assertIsNone(signature.return_annotation)

        self.assertTrue(hasattr(torch, "set_default_dtype"))
        self.assertNotIn("set_default_dtype", torch.__all__)
        self.assertEqual(torch.__all__.count("set_default_dtype"), 0)

    def test_positional_only_binding_errors_match_pytorch_2_13(self):
        function = torch.set_default_dtype
        cases = (
            (
                lambda: function(),
                "set_default_dtype() missing 1 required positional argument: 'd'",
            ),
            (
                lambda: function(torch.float32, torch.float32),
                "set_default_dtype() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(d=torch.float32),
                "set_default_dtype() got some positional-only arguments passed as keyword arguments: 'd'",
            ),
            (
                lambda: function(dtype=torch.float32),
                "set_default_dtype() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: function(torch.float32, dtype=torch.float32),
                "set_default_dtype() got an unexpected keyword argument 'dtype'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertIs(torch.get_default_dtype(), torch.float32)

    def test_invalid_dtype_objects_use_the_pytorch_2_13_error(self):
        message = (
            "invalid dtype object: only floating-point types are supported as the "
            "default type"
        )
        invalid_values = (
            None,
            True,
            1,
            1.0,
            "torch.float32",
            object(),
            torch.dtype,
            torch.tensor(1.0),
            torch.device("cpu"),
            np.dtype("float32"),
            np.float32,
        )
        for value in invalid_values:
            with self.subTest(value=repr(value)):
                with self.assertRaises(TypeError) as raised:
                    torch.set_default_dtype(value)
                self.assertEqual(str(raised.exception), message)
                self.assert_default_factories_are_canonical()


if __name__ == "__main__":
    unittest.main()
