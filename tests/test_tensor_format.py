import copy
import importlib
import inspect
import json
import math
import pickle
import re
import subprocess
import sys
import types
import unittest
import warnings

import torch_rs as torch


class TensorFormatTests(unittest.TestCase):
    def test_scalar_float32_formatting_uses_the_detached_python_value(self):
        cases = (
            (1.23456789, "", "1.2345678806304932"),
            (1.23456789, ".2f", "1.23"),
            (1.23456789, "+08.2f", "+0001.23"),
            (-0.0, "", "-0.0"),
            (-0.0, "+08.2f", "-0000.00"),
            (math.inf, ".2f", "inf"),
            (-math.inf, "+08.2f", "-0000inf"),
            (math.nan, "", "nan"),
            (math.nan, "+08.2f", "+0000nan"),
        )
        for value, format_spec, expected in cases:
            tensor = torch.tensor(value, dtype=torch.float32, requires_grad=True)
            with self.subTest(value=value, format_spec=format_spec):
                self.assertEqual(format(tensor, format_spec), expected)

        tensor = torch.tensor(-0.0, dtype=torch.float32, requires_grad=True)
        if sys.version_info >= (3, 11):
            self.assertEqual(format(tensor, " z"), " 0.0")
        else:
            with self.assertRaisesRegex(
                ValueError,
                "^Unknown format code 'z' for object of type 'float'$",
            ):
                format(tensor, " z")

        tensor = torch.tensor(1.25)
        with self.assertRaisesRegex(
            ValueError,
            "^Unknown format code 's' for object of type 'float'$",
        ):
            format(tensor, "s")

    def test_scalar_formatting_does_not_warn_or_change_the_graph(self):
        leaf = torch.tensor(1.25, requires_grad=True)
        tensor = leaf * 2.0
        before = (
            tensor.const_data_ptr(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.requires_grad,
            tensor.is_leaf,
            leaf.grad,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(format(tensor, ""), "2.5")
            self.assertEqual(format(tensor, ".2f"), "2.50")

        self.assertEqual(caught, [])
        self.assertEqual(
            (
                tensor.const_data_ptr(),
                tensor.shape,
                tensor.stride(),
                tensor.storage_offset(),
                tensor.requires_grad,
                tensor.is_leaf,
                leaf.grad,
            ),
            before,
        )
        tensor.backward()
        self.assertEqual(leaf.grad.item(), 2.0)

    def test_requires_grad_warning_is_absent_in_a_fresh_process(self):
        script = r'''
import json
import warnings
import torch_rs as torch

leaf = torch.tensor(1.25, requires_grad=True)
tensor = leaf * 2.0
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    outputs = [format(tensor, ""), format(tensor, ".2f")]
tensor.backward()
print(json.dumps({
    "outputs": outputs,
    "warnings": [(item.category.__name__, str(item.message)) for item in caught],
    "gradient": leaf.grad.item(),
}))
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(result.stdout),
            {"outputs": ["2.5", "2.50"], "warnings": [], "gradient": 2.0},
        )

    def test_non_scalar_empty_format_uses_str_and_errors_are_non_mutating(self):
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        tensor = leaf * 3.0
        before = (
            tensor.const_data_ptr(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.requires_grad,
            tensor.is_leaf,
            leaf.grad,
        )

        self.assertEqual(format(tensor, ""), str(tensor))
        self.assertEqual(f"{tensor}", str(tensor))
        for format_spec in (".2f", "s"):
            with self.subTest(format_spec=format_spec):
                with self.assertRaises(TypeError) as raised:
                    format(tensor, format_spec)
                self.assertEqual(
                    str(raised.exception),
                    "unsupported format string passed to Tensor.__format__",
                )

        self.assertEqual(
            (
                tensor.const_data_ptr(),
                tensor.shape,
                tensor.stride(),
                tensor.storage_offset(),
                tensor.requires_grad,
                tensor.is_leaf,
                leaf.grad,
            ),
            before,
        )
        tensor.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [3.0, 3.0])

    def test_mode_dispatch_receives_self_and_format_spec(self):
        function = inspect.getattr_static(torch.Tensor, "__format__")
        tensor = torch.tensor(1.25)

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode("formatted-by-mode")
        with mode:
            self.assertEqual(format(tensor, ".2f"), "formatted-by-mode")
        self.assertEqual(len(mode.calls), 1)
        called_function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(called_function, function)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], tensor)
        self.assertEqual(args[1], ".2f")
        self.assertEqual(kwargs, {})

        events = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                self.assertEqual(format(tensor, ".2f"), "1.25")
        self.assertEqual([event[0] for event in events], ["upper", "lower"])
        for _, called_function, dispatch_types, args, kwargs in events:
            self.assertIs(called_function, function)
            self.assertEqual(dispatch_types, (torch.Tensor,))
            self.assertIs(args[0], tensor)
            self.assertEqual(args[1:], (".2f",))
            self.assertEqual(kwargs, {})

        class DecliningMode(RecordingMode):
            def __repr__(self):
                return "declining-format-mode"

        declining = DecliningMode(NotImplemented)
        with self.assertRaisesRegex(
            TypeError,
            re.escape(
                "no implementation found for 'torch_rs._tensor.__format__' on "
                "types that implement __torch_function__: [] nor in mode "
                "declining-format-mode"
            ),
        ):
            with declining:
                format(tensor, ".2f")
        self.assertEqual(len(declining.calls), 2)
        self.assertEqual(
            [dispatch_types for _, dispatch_types, _, _ in declining.calls],
            [(torch.Tensor,), ()],
        )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_unbound_override_receives_the_two_positional_arguments(self):
        function = inspect.getattr_static(torch.Tensor, "__format__")
        marker = "formatted-by-override"

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertEqual(function(value, ".2f"), marker)
        called_function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(called_function, function)
        self.assertEqual(dispatch_types, (Override,))
        self.assertIs(args[0], value)
        self.assertEqual(args[1:], (".2f",))
        self.assertEqual(kwargs, {})

    def test_callable_metadata_direct_errors_and_pickling(self):
        tensor = torch.tensor(1.25)
        function = inspect.getattr_static(torch.Tensor, "__format__")
        bound = tensor.__format__

        self.assertIs(type(function), types.FunctionType)
        self.assertIs(type(bound), types.MethodType)
        self.assertRegex(
            repr(function), r"^<function Tensor\.__format__ at 0x[0-9a-f]+>$"
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

        calls = (
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
                lambda: bound(),
                "Tensor.__format__() missing 1 required positional argument: "
                "'format_spec'",
            ),
            (
                lambda: bound(".2f", "extra"),
                "Tensor.__format__() takes 2 positional arguments but 3 were given",
            ),
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        self.assertEqual(function(tensor, format_spec=".2f"), "1.25")
        self.assertEqual(function(self=tensor, format_spec=".2f"), "1.25")
        with self.assertRaises(AttributeError) as raised:
            function(1, ".2f")
        self.assertEqual(str(raised.exception), "'int' object has no attribute 'dim'")

    def test_z_reload_replaces_the_method_without_breaking_the_native_class(self):
        script = r'''
import importlib
import inspect
import json
import pickle
import torch_rs as torch

tensor_type = torch.Tensor
tensor = torch.tensor(-0.0, requires_grad=True)
module = torch._tensor
old_function = inspect.getattr_static(tensor_type, "__format__")
old_pickle_identity = pickle.loads(pickle.dumps(old_function)) is old_function
package_reload = importlib.reload(torch)
after_package_function = inspect.getattr_static(torch.Tensor, "__format__")
module_reload = importlib.reload(module)
new_function = inspect.getattr_static(torch.Tensor, "__format__")
try:
    pickle.dumps(old_function)
except Exception as error:
    old_pickle_error = [
        type(error).__name__,
        "Tensor.__format__" in str(error),
        "not the same object as torch_rs._tensor.Tensor.__format__" in str(error),
    ]
else:
    old_pickle_error = None
print(json.dumps({
    "old_pickle_identity": old_pickle_identity,
    "package_identity": package_reload is torch,
    "package_preserved_function": after_package_function is old_function,
    "module_identity": module_reload is module,
    "tensor_type_preserved": torch.Tensor is tensor_type and module.Tensor is tensor_type,
    "function_replaced": new_function is not old_function,
    "new_pickle_identity": pickle.loads(pickle.dumps(new_function)) is new_function,
    "old_pickle_error": old_pickle_error,
    "formatted": format(tensor, "+08.2f"),
}))
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        output = json.loads(result.stdout)
        self.assertEqual(
            output,
            {
                "old_pickle_identity": True,
                "package_identity": True,
                "package_preserved_function": True,
                "module_identity": True,
                "tensor_type_preserved": True,
                "function_replaced": True,
                "new_pickle_identity": True,
                "old_pickle_error": ["PicklingError", True, True],
                "formatted": "-0000.00",
            },
        )


if __name__ == "__main__":
    unittest.main()
