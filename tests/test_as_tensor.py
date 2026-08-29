import array
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
as_tensor(data: Any, *, dtype: Optional[dtype] = None, device: Optional[DeviceLikeType]) -> Tensor

Converts :attr:`data` into a tensor, sharing data and preserving autograd
history if possible.

If :attr:`data` is already a tensor with the requested dtype and device
then :attr:`data` itself is returned, but if :attr:`data` is a
tensor with a different dtype or device then it's copied as if using
`data.to(dtype=dtype, device=device)`.

If :attr:`data` is a NumPy array (an ndarray) with the same dtype and device then a
tensor is constructed using :func:`torch.from_numpy`.

If :attr:`data` is a CuPy array, the returned tensor will be located on the same device as the CuPy array unless
specifically overwritten by :attr:`device` or a default device. The device of the CuPy array is inferred from the
pointer of the array using `cudaPointerGetAttributes` unless :attr:`device` is provided with an explicit device index.

.. seealso::

    :func:`torch.tensor` never shares its data and creates a new "leaf tensor" (see :doc:`/notes/autograd`).


Args:
    data (array_like): Initial data for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, infers data type from :attr:`data`.
    device (:class:`torch.device`, optional): the device of the constructed tensor. If None and data is a tensor
        then the device of data is used. If None and data is not a tensor then
        the result tensor is constructed on the current device.


Example::

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a)
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a, device=torch.device('cuda'))
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([1,  2,  3])
"""


class AsTensorTests(unittest.TestCase):
    def tensor_metadata(self, tensor):
        return (
            tensor.tolist(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.layout,
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )

    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        produced = leaf * 2.0
        tracked = produced.transpose(0, 1)
        source = torch.tensor(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float32
        )
        strided = source.transpose(0, 1)[1]
        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        gradient_leaf = torch.tensor([2.0, 3.0], requires_grad=True)
        (gradient_leaf * 4.0).sum().backward()
        return leaf, tracked, (
            ("scalar", torch.tensor(-0.0, dtype=torch.float32)),
            ("empty view", empty),
            ("strided view", strided),
            ("autograd leaf", leaf),
            ("autograd non-leaf", produced),
            ("autograd view", tracked),
            ("accumulated gradient", gradient_leaf.grad),
        )

    def test_exact_native_tensor_supported_metadata_returns_identical_object(self):
        leaf, tracked, cases = self.tensor_cases()
        option_cases = (
            {},
            {"dtype": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"device": None},
            {"device": "cpu"},
            {"device": torch.device("cpu")},
            {"dtype": torch.float32, "device": "cpu"},
            {"dtype": torch.float, "device": torch.device("cpu")},
        )
        for case, tensor in cases:
            for options in option_cases:
                with self.subTest(case=case, options=options):
                    before = self.tensor_metadata(tensor)
                    result = torch.as_tensor(tensor, **options)
                    self.assertIs(result, tensor)
                    self.assertEqual(self.tensor_metadata(tensor), before)

        torch.as_tensor(tracked).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_indexed_cpu_device_requests_copy_exact_native_tensors(self):
        for device in ("cpu:0", "cpu:1", torch.device("cpu", 0)):
            with self.subTest(device=device):
                leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                result = torch.as_tensor(leaf, device=device)
                self.assertIsNot(result, leaf)
                self.assertEqual(result.tolist(), leaf.tolist())
                self.assertIs(result.dtype, torch.float32)
                self.assertEqual(result.device, torch.device("cpu"))
                self.assertTrue(result.requires_grad)
                self.assertFalse(result.is_leaf)
                self.assertNotEqual(result.data_ptr(), leaf.data_ptr())
                result.sum().backward()
                self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])

    def test_supported_non_tensor_inputs_use_tensor_copy_path(self):
        cases = (
            ("scalar", 1.25, (), 1.25),
            ("sequence", [[1, 2, 3], [4, 5, 6]], (2, 3), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            ("array", array.array("i", [-7, 0, 9]), (3,), [-7.0, 0.0, 9.0]),
            ("bytearray", bytearray((1, 2, 3)), (3,), [1.0, 2.0, 3.0]),
        )
        for case, source, shape, expected in cases:
            with self.subTest(case=case):
                result = torch.as_tensor(source, dtype=torch.float32, device="cpu")
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.tolist(), expected)
                self.assertIs(result.dtype, torch.float32)
                self.assertEqual(result.device, torch.device("cpu"))
                self.assertFalse(result.requires_grad)
                self.assertTrue(result.is_leaf)

        exporter = array.array("i", [1, 2, 3])
        buffer_result = torch.as_tensor(memoryview(exporter), dtype=torch.float32)
        exporter[0] = 9
        self.assertEqual(buffer_result.tolist(), [1.0, 2.0, 3.0])

    def test_numpy_arrays_are_copied_not_shared(self):
        source = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        result = torch.as_tensor(source, dtype=torch.float32)
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0])
        self.assertNotEqual(result.data_ptr(), source.__array_interface__["data"][0])

        source[0] = 9.0
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0])

    def test_binding_validation_and_unsupported_metadata_errors(self):
        cases = (
            (
                lambda: torch.as_tensor(),
                TypeError,
                'as_tensor() missing 1 required positional arguments: "data"',
            ),
            (
                lambda: torch.as_tensor([1.0], torch.float32),
                TypeError,
                "as_tensor() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.as_tensor([1.0], data=[2.0]),
                TypeError,
                "as_tensor() got multiple values for argument 'data'",
            ),
            (
                lambda: torch.as_tensor([1.0], requires_grad=True),
                TypeError,
                "as_tensor() got an unexpected keyword argument 'requires_grad'",
            ),
            (
                lambda: torch.as_tensor([1.0], dtype=1),
                TypeError,
                "as_tensor(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: torch.as_tensor([1.0], device=1.5),
                TypeError,
                "as_tensor(): argument 'device' must be torch.device, not float",
            ),
            (
                lambda: torch.as_tensor([1.0], device="cuda:0"),
                RuntimeError,
                "as_tensor(): device 'cuda:0' is not supported; only 'cpu' is implemented",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()

    def test_callable_metadata_and_exports_match_pytorch_2_13(self):
        function = torch.as_tensor
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "as_tensor")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.as_tensor")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method as_tensor of type object at 0x[0-9a-f]+>$",
        )
        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.as_tensor, function)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

        self.assertEqual(torch.__all__.count("as_tensor"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["as_tensor"], function)


if __name__ == "__main__":
    unittest.main()
