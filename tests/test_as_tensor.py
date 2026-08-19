import copy
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

NON_TENSOR_ERROR = (
    "as_tensor(): non-Tensor data is not supported; only existing Tensor "
    "inputs are implemented"
)


class AsTensorTests(unittest.TestCase):
    def assert_error(self, call, error_type, message):
        with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
            call()

    def test_matching_tensor_inputs_return_the_exact_object(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        sources = (
            base,
            base[1, 2, 3],
            base.transpose(0, 2),
            base.transpose(0, 2)[1],
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
        )
        calls = (
            lambda source: torch.as_tensor(source),
            lambda source: torch.as_tensor(data=source),
            lambda source: torch.as_tensor(source, dtype=None),
            lambda source: torch.as_tensor(source, dtype=torch.float32),
            lambda source: torch.as_tensor(source, dtype=torch.float),
            lambda source: torch.as_tensor(source, device=None),
            lambda source: torch.as_tensor(source, device="cpu"),
            lambda source: torch.as_tensor(source, device=torch.device("cpu")),
            lambda source: torch.as_tensor(
                data=source, dtype=torch.float32, device="cpu"
            ),
        )

        for source in sources:
            contract = (
                source.shape,
                source.stride(),
                source.storage_offset(),
                source.data_ptr(),
                source.dtype,
                source.device,
                source.layout,
                source.requires_grad,
                source.is_leaf,
                source.output_nr,
            )
            for call in calls:
                with self.subTest(
                    shape=source.shape, stride=source.stride(), call=call
                ):
                    result = call(source)
                    self.assertIs(result, source)
                    self.assertEqual(
                        (
                            result.shape,
                            result.stride(),
                            result.storage_offset(),
                            result.data_ptr(),
                            result.dtype,
                            result.device,
                            result.layout,
                            result.requires_grad,
                            result.is_leaf,
                            result.output_nr,
                        ),
                        contract,
                    )

    def test_identity_preserves_autograd_and_no_grad_state(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        view = leaf.transpose(0, 1)[1]
        result = torch.as_tensor(view, dtype=torch.float32, device="cpu")
        self.assertIs(result, view)

        weights = torch.tensor([3.0, 7.0])
        (result * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 7.0, 0.0]])

        with torch.no_grad():
            no_grad_result = torch.as_tensor(view)
        self.assertIs(no_grad_result, view)
        self.assertTrue(no_grad_result.requires_grad)
        (no_grad_result * weights).sum().backward()
        self.assertEqual(
            leaf.grad.tolist(), [[0.0, 6.0, 0.0], [0.0, 14.0, 0.0]]
        )

    def test_only_logically_unindexed_cpu_targets_are_identity(self):
        source = torch.tensor([1.0, 2.0])
        unindexed = (
            "cpu",
            "cpu:255",
            "cpu:2147483647",
            torch.device("cpu"),
            torch.device("cpu", 255),
        )
        for device in unindexed:
            with self.subTest(device=device):
                self.assertIs(torch.as_tensor(source, device=device), source)

        indexed = (
            "cpu:0",
            "cpu:1",
            "cpu:128",
            "cpu:256",
            torch.device("cpu", 0),
            torch.device("cpu", 128),
            torch.device("cpu", 256),
        )
        for device in indexed:
            with self.subTest(device=device), self.assertRaisesRegex(
                RuntimeError,
                r"^as_tensor\(\): indexed CPU device '.+' is not supported; "
                r"only unindexed 'cpu' Tensor identity is implemented$",
            ):
                torch.as_tensor(source, device=device)

    def test_non_cpu_targets_are_explicitly_rejected(self):
        source = torch.tensor([1.0, 2.0])
        for device in ("cuda", "cuda:0", "mps", "meta", "xpu:0"):
            with self.subTest(device=device):
                self.assert_error(
                    lambda device=device: torch.as_tensor(source, device=device),
                    RuntimeError,
                    f"as_tensor(): device '{device}' is not supported; only "
                    "unindexed 'cpu' Tensor identity is implemented",
                )

    def test_non_tensor_data_is_rejected_without_conversion_hooks(self):
        events = []

        class ConversionTrap:
            def __iter__(self):
                events.append("iter")
                raise AssertionError("iteration must not run")

            def __array__(self, *args, **kwargs):
                events.append(("array", args, kwargs))
                raise AssertionError("array conversion must not run")

            def __dlpack__(self, *args, **kwargs):
                events.append(("dlpack", args, kwargs))
                raise AssertionError("DLPack conversion must not run")

            def __float__(self):
                events.append("float")
                raise AssertionError("scalar conversion must not run")

        values = (
            [1.0, 2.0],
            (1.0, 2.0),
            np.asarray([1.0, 2.0], dtype=np.float32),
            1.0,
            None,
            object(),
            ConversionTrap(),
        )
        for value in values:
            with self.subTest(type=type(value).__name__):
                self.assert_error(
                    lambda value=value: torch.as_tensor(value),
                    TypeError,
                    NON_TENSOR_ERROR,
                )
        self.assertEqual(events, [])

    def test_binding_validation_and_precedence_match_pytorch(self):
        source = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.as_tensor(),
                TypeError,
                'as_tensor() missing 1 required positional arguments: "data"',
            ),
            (
                lambda: torch.as_tensor(dtype=1),
                TypeError,
                'as_tensor() missing 1 required positional arguments: "data"',
            ),
            (
                lambda: torch.as_tensor(source, None),
                TypeError,
                "as_tensor() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.as_tensor(source, data=source),
                TypeError,
                "as_tensor() got multiple values for argument 'data'",
            ),
            (
                lambda: torch.as_tensor(source, unexpected=True),
                TypeError,
                "as_tensor() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: torch.as_tensor(source, dtype=1),
                TypeError,
                "as_tensor(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: torch.as_tensor(source, device=1.5),
                TypeError,
                "as_tensor(): argument 'device' must be torch.device, not float",
            ),
            (
                lambda: torch.as_tensor(source, dtype=1, unexpected=True),
                TypeError,
                "as_tensor(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: torch.as_tensor(source, device=1.5, unexpected=True),
                TypeError,
                "as_tensor(): argument 'device' must be torch.device, not float",
            ),
            (
                lambda: torch.as_tensor(source, data=source, dtype=1),
                TypeError,
                "as_tensor(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: torch.as_tensor(source, device=""),
                RuntimeError,
                "Device string must not be empty",
            ),
            (
                lambda: torch.as_tensor(source, device="banana"),
                RuntimeError,
                "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, "
                "ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, "
                "mtia, privateuseone device type at start of device string: banana",
            ),
            (
                lambda: torch.as_tensor(source, device="cpu:01"),
                RuntimeError,
                "Invalid device string: 'cpu:01'",
            ),
            (
                lambda: torch.as_tensor(source, device="cpu:2147483648"),
                RuntimeError,
                "Could not parse device index '2147483648' in device string "
                "'cpu:2147483648'",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                self.assert_error(call, error_type, message)

    def test_torch_function_modes_match_creation_dispatch(self):
        source = torch.tensor([1.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        accepting = RecordingMode(marker)
        with accepting:
            result = torch.as_tensor(source)
        self.assertIs(result, marker)
        self.assertEqual(len(accepting.calls), 1)
        function, dispatch_types, args, kwargs = accepting.calls[0]
        self.assertIs(function, torch.as_tensor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (source,))
        self.assertIsNone(kwargs)

        accepting = RecordingMode(marker)
        with accepting:
            result = torch.as_tensor(data=[1.0], device="not-a-device")
        self.assertIs(result, marker)
        function, dispatch_types, args, kwargs = accepting.calls[0]
        self.assertIs(function, torch.as_tensor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"data": [1.0], "device": "not-a-device"})

        invalid = RecordingMode(marker)
        with invalid:
            self.assert_error(
                lambda: torch.as_tensor(source, dtype=1),
                TypeError,
                "as_tensor(): argument 'dtype' must be torch.dtype, not int",
            )
            self.assert_error(
                lambda: torch.as_tensor(source, unexpected=True),
                TypeError,
                "as_tensor() got an unexpected keyword argument 'unexpected'",
            )
        self.assertEqual(invalid.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.as_tensor(
                    source, dtype=torch.float32, device="cpu"
                )
        self.assertIs(forwarded, source)
        self.assertEqual([event[0] for event in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, torch.as_tensor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (source,))
            self.assertEqual(kwargs, {"dtype": torch.float32, "device": "cpu"})

        declining = RecordingMode(NotImplemented)
        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.as_tensor'; all "
            r"__torch_function__ handlers returned NotImplemented:",
        ):
            with declining:
                torch.as_tensor(source)
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_callable_metadata_documentation_exports_and_pickling(self):
        function = torch.as_tensor
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "as_tensor")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.as_tensor")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertIsNone(function.__self__)
        self.assertRegex(
            repr(function),
            r"^<built-in method as_tensor of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        reducer, (owner, name) = function.__reduce__()
        self.assertIs(reducer, getattr)
        self.assertEqual(name, "as_tensor")
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.as_tensor, function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("as_tensor"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["as_tensor"], function)


if __name__ == "__main__":
    unittest.main()
