import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


ZEROS_LIKE_DOC = """
zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor filled with the scalar value `0`, with the same size as
:attr:`input`. ``torch.zeros_like(input)`` is equivalent to
``torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

.. warning::
    As of 0.4, this function does not support an :attr:`out` keyword. As an alternative,
    the old ``torch.zeros_like(input, out=output)`` is equivalent to
    ``torch.zeros(input.size(), out=output)``.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.

Example::

    >>> input = torch.empty(2, 3)
    >>> torch.zeros_like(input)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])
"""


class ZerosLikeTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        values = np.ascontiguousarray(np.asarray(tensor))
        return values.reshape(-1).view(np.uint32)

    def assert_zeros_like_output(self, source, expected_stride):
        original = (
            source.shape,
            source.stride(),
            source.storage_offset(),
            self.tensor_bits(source).copy(),
            source.requires_grad,
            source.is_leaf,
        )
        result = torch.zeros_like(source)

        self.assertIsNot(result, source)
        self.assertFalse(result.is_set_to(source))
        if source.numel() != 0:
            self.assertNotEqual(result.data_ptr(), source.data_ptr())
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.stride(), expected_stride)
        self.assertEqual(result.storage_offset(), 0)
        self.assertIs(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)
        self.assertTrue(
            np.all(self.tensor_bits(result) == np.uint32(0)),
            msg="zeros_like must write positive-zero float32 bits",
        )
        self.assertEqual(source.shape, original[0])
        self.assertEqual(source.stride(), original[1])
        self.assertEqual(source.storage_offset(), original[2])
        np.testing.assert_array_equal(self.tensor_bits(source), original[3])
        self.assertEqual(source.requires_grad, original[4])
        self.assertEqual(source.is_leaf, original[5])
        return result

    def test_default_positional_preserve_format_outputs(self):
        values = np.array(
            [
                0x8000_0000,
                0xBF80_0000,
                0x4000_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x3F80_0000,
            ],
            dtype=np.uint32,
        ).view(np.float32)
        contiguous = torch.tensor(memoryview(values), dtype=torch.float32).reshape(
            (2, 3)
        )
        transposed = contiguous.transpose(0, 1)
        channels_last = torch.zeros((2, 3, 4, 5)).contiguous(
            memory_format=torch.channels_last
        )
        offset = torch.arange(18.0).reshape((3, 2, 3))[1]
        scalar = torch.tensor(-0.0, requires_grad=True)
        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]

        for name, source, expected_stride in (
            ("contiguous", contiguous, (3, 1)),
            ("transposed", transposed, (1, 3)),
            ("channels_last", channels_last, (60, 1, 15, 3)),
            ("offset", offset, (3, 1)),
            ("scalar", scalar, ()),
            ("empty", empty, (3, 3)),
        ):
            with self.subTest(name=name):
                self.assert_zeros_like_output(source, expected_stride)

    def test_input_keyword_matches_positional_form(self):
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1)
        positional = torch.zeros_like(source)
        keyword = torch.zeros_like(input=source)

        self.assertEqual(keyword.shape, positional.shape)
        self.assertEqual(keyword.stride(), positional.stride())
        self.assertEqual(keyword.tolist(), positional.tolist())
        self.assertFalse(keyword.is_set_to(positional))

    def test_explicit_options_are_not_native_supported(self):
        options = (
            ("dtype", {"dtype": None}),
            ("dtype", {"dtype": torch.float32}),
            ("dtype", {"dtype": int}),
            ("layout", {"layout": None}),
            ("layout", {"layout": torch.strided}),
            ("device", {"device": None}),
            ("device", {"device": "cpu"}),
            ("device", {"device": torch.device("cpu")}),
            ("device", {"device": 0}),
            ("requires_grad", {"requires_grad": None}),
            ("requires_grad", {"requires_grad": False}),
            ("requires_grad", {"requires_grad": True}),
            ("memory_format", {"memory_format": None}),
            ("memory_format", {"memory_format": torch.preserve_format}),
            ("memory_format", {"memory_format": torch.contiguous_format}),
        )
        source = torch.zeros((2, 3))
        for argument, kwargs in options:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf"^zeros_like\(\): explicit argument '{argument}' is not supported$",
                ):
                    torch.zeros_like(source, **kwargs)

    def test_argument_errors_match_generated_binding(self):
        source = torch.zeros((2, 3))
        cases = (
            (
                lambda: torch.zeros_like(),
                TypeError,
                'zeros_like() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.zeros_like(source, source),
                TypeError,
                "zeros_like() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.zeros_like(1),
                TypeError,
                "zeros_like(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.zeros_like(input=1),
                TypeError,
                "zeros_like(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.zeros_like(source, input=source),
                TypeError,
                "zeros_like() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.zeros_like(source, unexpected=True),
                TypeError,
                "zeros_like() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: torch.zeros_like(source, dtype=1),
                TypeError,
                "zeros_like(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: torch.zeros_like(source, layout=1),
                TypeError,
                "zeros_like(): argument 'layout' must be torch.layout, not int",
            ),
            (
                lambda: torch.zeros_like(source, device=True),
                TypeError,
                "zeros_like(): argument 'device' must be torch.device, not bool",
            ),
            (
                lambda: torch.zeros_like(source, requires_grad=0),
                TypeError,
                "zeros_like(): argument 'requires_grad' must be bool, not int",
            ),
            (
                lambda: torch.zeros_like(source, memory_format=1),
                TypeError,
                "zeros_like(): argument 'memory_format' must be torch.memory_format, not int",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()

    def test_torch_function_modes_receive_public_callable_and_can_forward(self):
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1)
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func,
                        types,
                        args,
                        kwargs,
                        tuple(torch.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.zeros_like(source), marker)
        self.assertEqual(len(mode.calls), 1)
        function, types_, args, kwargs, stack = mode.calls[0]
        self.assertIs(function, torch.zeros_like)
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.zeros_like")
        self.assertEqual(types_, ())
        self.assertEqual(args, (source,))
        self.assertIsNone(kwargs)
        self.assertEqual(stack, ())

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **({} if kwargs is None else kwargs))

        with ForwardingMode():
            forwarded = torch.zeros_like(source)
        self.assertEqual(forwarded.shape, source.shape)
        self.assertEqual(forwarded.stride(), source.stride())
        self.assertEqual(forwarded.tolist(), [[0.0, 0.0], [0.0, 0.0]])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            "Multiple dispatch failed for 'torch.zeros_like'",
        ):
            with DecliningMode():
                torch.zeros_like(source)

    def test_torch_function_overrides_include_option_values(self):
        marker = object()

        class DTypeOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.observation = (func, types, args, kwargs)
                return marker

        source = torch.zeros((2,))
        self.assertIs(torch.zeros_like(source, dtype=DTypeOverride()), marker)
        function, types_, args, kwargs = DTypeOverride.observation
        self.assertIs(function, torch.zeros_like)
        self.assertEqual(types_, (DTypeOverride,))
        self.assertEqual(args, (source,))
        self.assertEqual(tuple(kwargs), ("dtype",))

    def test_torch_function_override_ordering_propagates_subclasscheck_errors(self):
        class RaisingMeta(type):
            def __subclasscheck__(cls, subclass):
                raise RuntimeError("subclass check failed")

        class FirstOption(metaclass=RaisingMeta):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return "first"

        class SecondOption:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return "second"

        with self.assertRaisesRegex(RuntimeError, "subclass check failed"):
            torch.zeros_like(
                torch.zeros((1,)),
                dtype=FirstOption(),
                layout=SecondOption(),
            )

    def test_option_torch_function_probe_is_not_retried_after_failure(self):
        class FlakyOption:
            probes = 0
            dispatched = False

            def __getattribute__(self, name):
                if name == "__torch_function__":
                    type(self).probes += 1
                    if type(self).probes == 1:
                        raise RuntimeError("hidden probe failure")
                return object.__getattribute__(self, name)

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.dispatched = True
                return object()

        with self.assertRaisesRegex(
            TypeError,
            r"zeros_like\(\): argument 'dtype' must be torch.dtype, not FlakyOption",
        ):
            torch.zeros_like(torch.zeros((1,)), dtype=FlakyOption())
        self.assertEqual(FlakyOption.probes, 1)
        self.assertFalse(FlakyOption.dispatched)

    def test_callable_metadata_exports_and_pickling(self):
        function = torch.zeros_like
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "zeros_like")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.zeros_like")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__.replace("torch_rs._C", "torch._C"), "torch._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.zeros_like, function)
        self.assertEqual(function.__doc__, ZEROS_LIKE_DOC)
        self.assertIsNone(function.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(function)
        self.assertEqual(torch.__all__.count("zeros_like"), 1)
        self.assertIs(wildcard_namespace["zeros_like"], function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )


if __name__ == "__main__":
    unittest.main()
