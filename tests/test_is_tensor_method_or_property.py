import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import unittest
from collections.abc import Callable

import torch_rs as torch


FUNCTION_DOC = """
    Returns True if the function passed in is a handler for a
    method or property belonging to ``torch.Tensor``, as passed
    into ``__torch_function__``.

    .. note::
       For properties, their ``__get__`` method must be passed in.

    This may be needed, in particular, for the following reasons:

    1. Methods/properties sometimes don't contain a `__module__` slot.
    2. They require that the first passed-in argument is an instance
       of ``torch.Tensor``.

    Examples
    --------
    >>> is_tensor_method_or_property(torch.Tensor.add)
    True
    >>> is_tensor_method_or_property(torch.add)
    False
    """


def tensor_descriptor(name):
    for tensor_type in torch.Tensor.__mro__:
        if name in vars(tensor_type):
            return vars(tensor_type)[name]
    raise AssertionError(f"missing Tensor descriptor {name!r}")


class IsTensorMethodOrPropertyTests(unittest.TestCase):
    def test_tensor_and_tensorbase_methods_are_recognized(self):
        function = torch.overrides.is_tensor_method_or_property
        methods = (
            torch.Tensor.is_shared,
            torch.Tensor.sqrt,
            torch.Tensor.view,
            torch.Tensor.__add__,
            torch.Tensor.__pos__,
        )
        for method in methods:
            with self.subTest(method=method):
                self.assertIs(function(method), True)

        self.assertIs(tensor_descriptor("is_shared"), torch.Tensor.is_shared)
        self.assertIs(tensor_descriptor("sqrt"), torch.Tensor.__base__.sqrt)

    def test_property_get_handlers_are_recognized(self):
        function = torch.overrides.is_tensor_method_or_property
        for name in ("real", "shape", "T", "grad"):
            with self.subTest(name=name):
                descriptor = tensor_descriptor(name)
                self.assertFalse(callable(descriptor))
                self.assertEqual(descriptor.__get__.__name__, "__get__")
                self.assertIs(function(descriptor.__get__), True)
                self.assertIs(function(descriptor), False)

        unrelated_property = property(lambda self: None)
        self.assertIs(function(unrelated_property.__get__), True)

    def test_top_level_bound_and_unrelated_callables_are_rejected(self):
        function = torch.overrides.is_tensor_method_or_property

        def local_function():
            return None

        tensor = torch.tensor([1.0])
        callables = (
            torch.sqrt,
            torch.positive,
            local_function,
            len,
            str.upper,
            object.__str__,
            torch.Tensor.__str__,
            torch.Tensor.stride,
            torch.Tensor.__iter__,
            tensor.sqrt,
        )
        for callable_object in callables:
            with self.subTest(callable_object=callable_object):
                self.assertIs(function(callable_object), False)

    def test_later_monkey_patches_are_not_classified_as_handlers(self):
        function = torch.overrides.is_tensor_method_or_property

        def future_tensor_method(self):
            return self

        name = "_future_tensor_method_for_override_probe"
        self.assertFalse(hasattr(torch.Tensor, name))
        setattr(torch.Tensor, name, future_tensor_method)
        try:
            self.assertIs(getattr(torch.Tensor, name), future_tensor_method)
            self.assertIs(function(future_tensor_method), False)
        finally:
            delattr(torch.Tensor, name)

        self.assertIs(function(future_tensor_method), False)

    def test_callable_metadata_imports_copy_and_pickle(self):
        function = torch.overrides.is_tensor_method_or_property
        self.assertEqual(function.__name__, "is_tensor_method_or_property")
        self.assertEqual(function.__qualname__, "is_tensor_method_or_property")
        self.assertEqual(function.__module__, "torch_rs.overrides")
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertEqual(
            function.__annotations__,
            {"func": Callable, "return": bool},
        )
        self.assertEqual(
            str(inspect.signature(function)),
            "(func: collections.abc.Callable) -> bool",
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__.keys(), {"__wrapped__"})
        self.assertEqual(
            str(inspect.signature(function.__wrapped__)),
            str(inspect.signature(function)),
        )
        self.assertEqual(
            inspect.cleandoc(function.__wrapped__.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )

        self.assertIs(
            importlib.import_module("torch_rs.overrides").is_tensor_method_or_property,
            function,
        )
        direct_namespace = {}
        exec(
            "from torch_rs.overrides import is_tensor_method_or_property",
            direct_namespace,
        )
        self.assertIs(direct_namespace["is_tensor_method_or_property"], function)
        wildcard_namespace = {}
        exec("from torch_rs.overrides import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["is_tensor_method_or_property"], function)
        self.assertIn("is_tensor_method_or_property", torch.overrides.__all__)
        self.assertNotIn("is_tensor_method_or_property", torch.__all__)
        self.assertFalse(hasattr(torch, "is_tensor_method_or_property"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

    def test_argument_and_invalid_input_errors_match_python_contract(self):
        function = torch.overrides.is_tensor_method_or_property
        argument_errors = (
            (
                lambda: function(),
                "is_tensor_method_or_property() missing 1 required positional "
                "argument: 'func'",
            ),
            (
                lambda: function(torch.Tensor.sqrt, torch.Tensor.view),
                "is_tensor_method_or_property() takes 1 positional argument but "
                "2 were given",
            ),
            (
                lambda: function(value=torch.Tensor.sqrt),
                "is_tensor_method_or_property() got an unexpected keyword "
                "argument 'value'",
            ),
        )
        for call, message in argument_errors:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        class CallableWithoutName:
            def __call__(self):
                return None

        class HashFailure:
            def __hash__(self):
                raise RuntimeError("hash failed")

        invalid_inputs = (
            (
                lambda: function(None),
                AttributeError,
                "'NoneType' object has no attribute '__name__'",
            ),
            (
                lambda: function(object()),
                AttributeError,
                "'object' object has no attribute '__name__'",
            ),
            (
                lambda: function(CallableWithoutName()),
                AttributeError,
                "'CallableWithoutName' object has no attribute '__name__'",
            ),
            (lambda: function(HashFailure()), RuntimeError, "hash failed"),
        )
        for call, error_type, message in invalid_inputs:
            with self.subTest(error_type=error_type.__name__, message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        with self.assertRaises(TypeError) as raised:
            function([])
        self.assertIn("unhashable", str(raised.exception))
        self.assertIn("list", str(raised.exception))

        class NamedGet:
            __name__ = "__get__"

            def __call__(self):
                return None

        self.assertIs(function(NamedGet()), True)
        self.assertIs(function(func=torch.Tensor.sqrt), True)

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

function = torch.overrides.is_tensor_method_or_property
assert function(torch.Tensor.is_shared) is True
assert function(torch.Tensor.sqrt) is True
assert function(torch.Tensor.__base__.real.__get__) is True
assert function(torch.sqrt) is False

def unrelated_tensor_method(self):
    return self

torch.Tensor._unrelated_override_probe = unrelated_tensor_method
try:
    assert function(unrelated_tensor_method) is False
finally:
    del torch.Tensor._unrelated_override_probe

assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
