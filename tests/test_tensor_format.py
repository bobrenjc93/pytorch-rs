import copy
import inspect
import json
import pickle
import re
import subprocess
import sys
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch


SPECIAL_BITS = (
    0xC020_0000,
    0x3F9E_0652,
    0x0000_0000,
    0x8000_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC1_2345,
    0xFFC5_4321,
)
FORMAT_SPECS = ("", ".2f", "+08.2f", "e", "g", "%", "^12")


def float32_from_bits(bits):
    return float(np.asarray((bits,), dtype=np.uint32).view(np.float32)[0])


def scalar_view(module, bits, *, requires_grad=False):
    values = np.asarray((0x3F80_0000, bits), dtype=np.uint32).view(np.float32)
    leaf = module.tensor(
        memoryview(values),
        dtype=module.float32,
        requires_grad=requires_grad,
    )
    return leaf[1], leaf


class TensorFormatTests(unittest.TestCase):
    def metadata(self, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
            tensor.is_leaf,
        )

    def test_rank_zero_float32_values_use_python_scalar_formatting(self):
        for bits in SPECIAL_BITS:
            tensor, _ = scalar_view(torch, bits)
            expected = float32_from_bits(bits)
            self.assertEqual(tensor.shape, ())
            self.assertEqual(tensor.storage_offset(), 1)
            for format_spec in FORMAT_SPECS:
                with self.subTest(bits=f"{bits:#010x}", format_spec=format_spec):
                    actual = format(tensor, format_spec)
                    self.assertIs(type(actual), str)
                    self.assertEqual(actual, format(expected, format_spec))

    def test_active_autograd_is_detached_for_formatting_and_unchanged(self):
        for bits in SPECIAL_BITS:
            tensor, leaf = scalar_view(torch, bits, requires_grad=True)
            tensor_metadata = self.metadata(tensor)
            leaf_metadata = self.metadata(leaf)
            for format_spec in FORMAT_SPECS:
                with self.subTest(bits=f"{bits:#010x}", format_spec=format_spec):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        actual = format(tensor, format_spec)
                    self.assertEqual(
                        actual,
                        format(float32_from_bits(bits), format_spec),
                    )
                    self.assertEqual(caught, [])
                    self.assertEqual(self.metadata(tensor), tensor_metadata)
                    self.assertEqual(self.metadata(leaf), leaf_metadata)
                    self.assertIsNone(leaf.grad)

            tensor.backward()
            self.assertEqual(leaf.grad.tolist(), [0.0, 1.0])

    def test_fresh_process_formatting_emits_no_scalar_conversion_warning(self):
        script = r'''
import json, warnings
import torch_rs as torch

leaf = torch.tensor([1.0, -0.0], requires_grad=True)
value = leaf[1]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = format(value, "+08.2f")
print(json.dumps({"result": result, "warnings": len(caught)}))
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(result.stdout),
            {"result": "-0000.00", "warnings": 0},
        )

    def test_non_scalars_use_object_empty_format_and_reject_nonempty_specs(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        cases = (
            torch.zeros((0,)),
            torch.tensor([1.0]),
            leaf,
            (leaf * 2.0).transpose(0, 1),
            torch.zeros((2, 0, 3)).transpose(0, 2),
        )
        for tensor in cases:
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                metadata = self.metadata(tensor)
                self.assertEqual(format(tensor, ""), str(tensor))
                self.assertEqual(self.metadata(tensor), metadata)
                with self.assertRaises(TypeError) as raised:
                    format(tensor, ".2f")
                self.assertEqual(
                    str(raised.exception),
                    "unsupported format string passed to Tensor.__format__",
                )
                self.assertEqual(self.metadata(tensor), metadata)

        self.assertIsNone(leaf.grad)
        leaf.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])

    def test_callable_metadata_and_pickle_match_python_method_shape(self):
        tensor = torch.tensor(1.25)
        function = inspect.getattr_static(torch.Tensor, "__format__")
        bound = tensor.__format__

        self.assertIs(type(function), types.FunctionType)
        self.assertIs(type(bound), types.MethodType)
        self.assertRegex(
            repr(function),
            r"^<function Tensor\.__format__ at 0x[0-9a-f]+>$",
        )
        self.assertEqual(function.__name__, "__format__")
        self.assertEqual(function.__qualname__, "Tensor.__format__")
        self.assertEqual(function.__module__, "torch_rs._tensor")
        self.assertEqual(bound.__name__, "__format__")
        self.assertEqual(bound.__qualname__, "Tensor.__format__")
        self.assertEqual(bound.__module__, "torch_rs._tensor")
        self.assertIsNone(function.__doc__)
        self.assertIsNone(bound.__doc__)
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(bound.__annotations__, {})
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertFalse(hasattr(bound, "__text_signature__"))
        self.assertEqual(str(inspect.signature(function)), "(self, format_spec)")
        self.assertEqual(str(inspect.signature(bound)), "(format_spec)")
        self.assertIn("__format__", torch.Tensor.__dict__)
        self.assertTrue(
            all(
                "__format__" not in owner.__dict__
                for owner in torch.Tensor.__mro__[1:-1]
            )
        )
        self.assertIn("__format__", object.__dict__)
        self.assertIs(torch._tensor.Tensor, torch.Tensor)
        self.assertIs(torch._tensor.Tensor.__format__, function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_direct_call_binding_and_receiver_errors(self):
        tensor = torch.tensor(1.25)
        function = inspect.getattr_static(torch.Tensor, "__format__")
        bound = tensor.__format__
        self.assertEqual(function(self=tensor, format_spec=".1f"), "1.2")

        calls = (
            (
                lambda: function(),
                TypeError,
                "Tensor.__format__() missing 2 required positional arguments: "
                "'self' and 'format_spec'",
            ),
            (
                lambda: function(tensor),
                TypeError,
                "Tensor.__format__() missing 1 required positional argument: "
                "'format_spec'",
            ),
            (
                lambda: bound(),
                TypeError,
                "Tensor.__format__() missing 1 required positional argument: "
                "'format_spec'",
            ),
            (
                lambda: function(tensor, "", 1),
                TypeError,
                "Tensor.__format__() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: bound("", 1),
                TypeError,
                "Tensor.__format__() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: function(tensor, "", unexpected=True),
                TypeError,
                "Tensor.__format__() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: function(tensor, "", self=tensor),
                TypeError,
                "Tensor.__format__() got multiple values for argument 'self'",
            ),
            (
                lambda: function(tensor, "", format_spec=""),
                TypeError,
                "Tensor.__format__() got multiple values for argument 'format_spec'",
            ),
            (
                lambda: function(1.0, ""),
                AttributeError,
                "'float' object has no attribute 'dim'",
            ),
            (
                lambda: function(tensor, object()),
                TypeError,
                "__format__() argument must be str, not object",
            ),
        )
        for call, error_type, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_override_and_mode_receive_both_arguments(self):
        function = inspect.getattr_static(torch.Tensor, "__format__")
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(function(value, ".2f"), marker)
        called_function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(called_function, function)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (value, ".2f"))
        self.assertEqual(kwargs, {})

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        tensor = torch.tensor(-0.0)
        recording = RecordingMode(marker)
        with recording:
            result = tensor.__format__(format_spec="+08.2f")
        self.assertIs(result, marker)
        self.assertEqual(len(recording.calls), 1)
        called_function, dispatch_types, args, kwargs = recording.calls[0]
        self.assertIs(called_function, function)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(args, (tensor, "+08.2f"))
        self.assertEqual(kwargs, {})

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = format(tensor, "+08.2f")
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded, "-0000.00")

        declining = RecordingMode(NotImplemented)
        with self.assertRaises(TypeError) as raised:
            with declining:
                format(tensor, ".2f")
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^no implementation found for "
                r"'torch_rs\._tensor\.__format__' on types that implement "
                r"__torch_function__: \[\] nor in mode "
                r"<.*RecordingMode object at 0x[0-9a-f]+>$"
            ),
        )
        self.assertEqual(len(declining.calls), 2)
        self.assertEqual(
            [call[1] for call in declining.calls],
            [(torch.Tensor,), ()],
        )
        self.assertTrue(
            all(call[2] == (tensor, ".2f") for call in declining.calls)
        )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_tensor_module_reload_reinstalls_a_picklable_method(self):
        script = r'''
import importlib, inspect, pickle
import torch_rs as torch

module = torch._tensor
tensor_type = torch.Tensor
old_function = inspect.getattr_static(tensor_type, "__format__")
assert format(torch.tensor(-0.0), "+08.2f") == "-0000.00"
assert importlib.reload(module) is module
new_function = inspect.getattr_static(tensor_type, "__format__")
assert module.Tensor is tensor_type
assert torch.Tensor is tensor_type
assert new_function is not old_function
assert module.Tensor.__format__ is new_function
assert format(torch.tensor(1.25), ".1f") == "1.2"
for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    assert pickle.loads(pickle.dumps(new_function, protocol)) is new_function
'''
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
