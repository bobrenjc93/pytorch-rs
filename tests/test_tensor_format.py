import copy
import inspect
import pickle
import re
import subprocess
import sys
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch


class TensorFormatTests(unittest.TestCase):
    @staticmethod
    def float32_from_bits(bits):
        return np.asarray(bits, dtype=np.uint32).view(np.float32).item()

    @staticmethod
    def metadata(tensor):
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

    def test_rank_zero_values_use_python_float_formatting(self):
        values = (
            ("positive zero", self.float32_from_bits(0x0000_0000)),
            ("negative zero", self.float32_from_bits(0x8000_0000)),
            ("ordinary positive", self.float32_from_bits(0x3FA0_0000)),
            ("ordinary negative", self.float32_from_bits(0xC020_0000)),
            ("positive infinity", self.float32_from_bits(0x7F80_0000)),
            ("negative infinity", self.float32_from_bits(0xFF80_0000)),
            ("positive nan", self.float32_from_bits(0x7FC0_0000)),
            ("negative nan", self.float32_from_bits(0xFFC0_0000)),
        )
        format_specs = (
            "",
            ".2f",
            "+.3e",
            " 012.4f",
            "#.0f",
            "%",
            "^15",
            "F",
        )

        for case, value in values:
            tensors = (
                ("owned", torch.tensor(value)),
                ("offset scalar view", torch.tensor([9.0, value])[1]),
            )
            for tensor_case, tensor in tensors:
                before = self.metadata(tensor)
                for format_spec in format_specs:
                    with self.subTest(
                        case=case,
                        tensor_case=tensor_case,
                        format_spec=format_spec,
                    ):
                        formatted = format(tensor, format_spec)
                        self.assertIs(type(formatted), str)
                        self.assertEqual(formatted, format(value, format_spec))
                        self.assertEqual(self.metadata(tensor), before)

    def test_scalar_autograd_is_detached_only_for_formatting(self):
        leaf = torch.tensor(1.25, requires_grad=True)
        output = leaf * -2.0
        leaf_before = self.metadata(leaf)
        output_before = self.metadata(output)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(format(leaf, "+.2f"), "+1.25")
            self.assertEqual(format(output, ".3e"), "-2.500e+00")

        self.assertEqual(caught, [])
        self.assertEqual(self.metadata(leaf), leaf_before)
        self.assertEqual(self.metadata(output), output_before)
        self.assertIsNone(leaf.grad)

        output.backward()
        self.assertEqual(leaf.grad.item(), -2.0)

    def test_non_scalars_delegate_empty_format_and_reject_nonempty_specs(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        cases = (
            ("vector", torch.tensor([1.25])),
            ("empty", torch.zeros((2, 0, 3))),
            (
                "offset view",
                torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                .transpose(0, 1)[1],
            ),
            ("tracked view", tracked),
        )

        for case, tensor in cases:
            before = self.metadata(tensor)
            with self.subTest(case=case, format_spec=""):
                formatted = format(tensor, "")
                self.assertIs(type(formatted), str)
                self.assertEqual(formatted, str(tensor))
            for format_spec in (".2f", "x", " >12"):
                with self.subTest(case=case, format_spec=format_spec):
                    with self.assertRaises(TypeError) as raised:
                        format(tensor, format_spec)
                    self.assertEqual(
                        str(raised.exception),
                        "unsupported format string passed to Tensor.__format__",
                    )
            self.assertEqual(self.metadata(tensor), before)

        self.assertIsNone(leaf.grad)
        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_callable_metadata_binding_errors_copy_and_pickle(self):
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
        self.assertIsNone(function.__doc__)
        self.assertEqual(function.__annotations__, {})
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(str(inspect.signature(function)), "(self, format_spec)")
        self.assertEqual(str(inspect.signature(bound)), "(format_spec)")
        self.assertIn("__format__", torch.Tensor.__dict__)
        self.assertTrue(
            all(
                "__format__" not in owner.__dict__
                for owner in torch.Tensor.__mro__[1:-1]
            )
        )
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

        self.assertEqual(function(self=tensor, format_spec=".2f"), "1.25")
        self.assertEqual(bound(format_spec=".2f"), "1.25")
        invalid_calls = (
            (
                lambda: function(),
                "Tensor.__format__() missing 2 required positional arguments: "
                "'self' and 'format_spec'",
            ),
            (
                lambda: function(tensor),
                "Tensor.__format__() missing 1 required positional argument: "
                "'format_spec'",
            ),
            (
                lambda: function(tensor, ".2f", "extra"),
                "Tensor.__format__() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: function(tensor, unexpected=True),
                "Tensor.__format__() got an unexpected keyword argument "
                "'unexpected'",
            ),
            (
                lambda: bound(self=tensor),
                "Tensor.__format__() got multiple values for argument 'self'",
            ),
        )
        for call, message in invalid_calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        with self.assertRaises(AttributeError) as raised:
            function(1, ".2f")
        self.assertEqual(str(raised.exception), "'int' object has no attribute 'dim'")
        for format_spec, type_name in ((None, "None"), (2, "int")):
            with self.subTest(format_spec=format_spec):
                with self.assertRaises(TypeError) as raised:
                    function(tensor, format_spec)
                self.assertEqual(
                    str(raised.exception),
                    f"__format__() argument must be str, not {type_name}",
                )

    def test_override_and_modes_receive_self_and_format_spec(self):
        function = inspect.getattr_static(torch.Tensor, "__format__")
        marker = "format-marker"

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        override = Override()
        self.assertEqual(function(override, ".3f"), marker)
        called_function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(called_function, function)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (override, ".3f"))
        self.assertEqual(kwargs, {})

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        tensor = torch.tensor(1.25)
        recording = RecordingMode(marker)
        with recording:
            result = format(tensor, ".2f")
        self.assertEqual(result, marker)
        self.assertEqual(len(recording.calls), 1)
        called_function, dispatch_types, args, kwargs = recording.calls[0]
        self.assertIs(called_function, function)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(args, (tensor, ".2f"))
        self.assertEqual(kwargs, {})

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = format(tensor, ".2f")
        self.assertEqual(forwarded, "1.25")
        self.assertEqual([call[0] for call in order], ["upper", "lower"])
        for _, called_function, dispatch_types, args, kwargs in order:
            self.assertIs(called_function, function)
            self.assertEqual(dispatch_types, (torch.Tensor,))
            self.assertEqual(args, (tensor, ".2f"))
            self.assertEqual(kwargs, {})

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
        self.assertEqual(declining.calls[0][1], (torch.Tensor,))
        self.assertEqual(declining.calls[1][1], ())
        for _, _, args, kwargs in declining.calls:
            self.assertEqual(args, (tensor, ".2f"))
            self.assertEqual(kwargs, {})
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        rejected = RecordingMode(marker)
        with rejected:
            with self.assertRaises(TypeError):
                function(tensor)
        self.assertEqual(rejected.calls, [])

    def test_package_and_tensor_module_reload_keep_a_canonical_method(self):
        script = r'''
import importlib
import inspect
import pickle

import torch_rs as torch

tensor_module = torch._tensor
tensor_type = torch.Tensor
old_function = inspect.getattr_static(tensor_type, "__format__")
assert importlib.reload(torch) is torch
assert torch.Tensor is tensor_type
assert torch.Tensor.__format__ is old_function
assert torch._tensor is tensor_module

assert importlib.reload(tensor_module) is tensor_module
function = inspect.getattr_static(tensor_type, "__format__")
assert function is not old_function
assert tensor_module.Tensor is tensor_type
assert torch.Tensor is tensor_type
assert tensor_module.Tensor.__format__ is function
assert format(torch.tensor(1.25), ".2f") == "1.25"
for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    assert pickle.loads(pickle.dumps(function, protocol=protocol)) is function
try:
    pickle.dumps(old_function)
except pickle.PicklingError:
    pass
else:
    raise AssertionError("the stale Tensor.__format__ remained pickleable")

leaf = torch.tensor(2.0, requires_grad=True)
(leaf * leaf).backward()
assert leaf.grad.item() == 4.0
'''
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
