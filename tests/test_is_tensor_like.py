import copy
import importlib
import inspect
import pickle
import re
import sys
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = """
    Returns ``True`` if the passed-in input is a Tensor-like.

    Currently, this occurs whenever there's a ``__torch_function__``
    attribute on the type of the input.

    Examples
    --------
    A subclass of tensor is generally a Tensor-like.

    >>> class SubTensor(torch.Tensor): ...
    >>> is_tensor_like(SubTensor([0]))
    True

    Built-in or user types aren't usually Tensor-like.

    >>> is_tensor_like(6)
    False
    >>> is_tensor_like(None)
    False
    >>> class NotATensor: ...
    >>> is_tensor_like(NotATensor())
    False

    But, they can be made Tensor-like by implementing __torch_function__.

    >>> class TensorLike:
    ...     @classmethod
    ...     def __torch_function__(cls, func, types, args, kwargs):
    ...         return -1
    >>> is_tensor_like(TensorLike())
    True
    """


class IsTensorLikeTests(unittest.TestCase):
    def test_native_tensors_and_torch_function_objects_return_true(self):
        function = torch.overrides.is_tensor_like

        class ClassOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise AssertionError("is_tensor_like invoked the override")

        class FalseyOverride:
            __torch_function__ = None

        class BaseOverride:
            __torch_function__ = False

        class DerivedOverride(BaseOverride):
            pass

        class InstanceOverride:
            pass

        class DynamicOverride:
            def __init__(self):
                self.lookups = []

            def __getattr__(self, name):
                self.lookups.append(name)
                if name == "__torch_function__":
                    return 0
                raise AttributeError(name)

        instance_override = InstanceOverride()
        instance_override.__torch_function__ = "present"
        dynamic_override = DynamicOverride()
        values = (
            torch.tensor(1.0),
            torch.tensor([1.0, 2.0]),
            torch.zeros((2, 3)).transpose(0, 1),
            torch.Tensor,
            ClassOverride(),
            FalseyOverride(),
            DerivedOverride(),
            instance_override,
            dynamic_override,
        )
        for case, value in enumerate(values):
            with self.subTest(case=case, value_type=type(value).__name__):
                result = function(value)
                self.assertIs(type(result), bool)
                self.assertIs(result, True)

        self.assertEqual(dynamic_override.lookups, ["__torch_function__"])

    def test_ordinary_objects_return_false(self):
        function = torch.overrides.is_tensor_like

        class Plain:
            pass

        values = (
            None,
            True,
            1,
            1.5,
            2.0j,
            "tensor",
            b"tensor",
            [],
            (),
            {},
            object(),
            Plain(),
            Plain,
            torch.float32,
            torch.device("cpu"),
            torch.overrides,
        )
        for case, value in enumerate(values):
            with self.subTest(case=case, value_type=type(value).__name__):
                result = function(value)
                self.assertIs(type(result), bool)
                self.assertIs(result, False)

    def test_attribute_lookup_and_descriptor_exceptions_match_hasattr(self):
        function = torch.overrides.is_tensor_like
        events = []

        class PresentDescriptor:
            def __get__(self, instance, owner):
                events.append(("present", instance is not None, owner))
                return None

        class Present:
            __torch_function__ = PresentDescriptor()

        self.assertIs(function(Present()), True)
        self.assertEqual(events, [("present", True, Present)])

        class MissingDescriptor:
            def __get__(self, instance, owner):
                events.append(("missing", instance is not None, owner))
                raise AttributeError("hidden")

        class Missing:
            __torch_function__ = MissingDescriptor()

        self.assertIs(function(Missing()), False)
        self.assertEqual(events[-1], ("missing", True, Missing))

        class HiddenAttributeError(AttributeError):
            pass

        class MissingSubclass:
            @property
            def __torch_function__(self):
                raise HiddenAttributeError("also hidden")

        self.assertIs(function(MissingSubclass()), False)

        class Fallback:
            @property
            def __torch_function__(self):
                events.append("descriptor")
                raise AttributeError("try __getattr__")

            def __getattr__(self, name):
                events.append(("fallback", name))
                if name == "__torch_function__":
                    return None
                raise AttributeError(name)

        self.assertIs(function(Fallback()), True)
        self.assertEqual(
            events[-2:], ["descriptor", ("fallback", "__torch_function__")]
        )

        expected_error = RuntimeError("descriptor failure")

        class Raising:
            @property
            def __torch_function__(self):
                raise expected_error

        with self.assertRaises(RuntimeError) as raised:
            function(Raising())
        self.assertIs(raised.exception, expected_error)

    def test_inheritance_and_live_tensor_binding_are_observed(self):
        function = torch.overrides.is_tensor_like

        class Base:
            __torch_function__ = None

        class Inherited(Base):
            pass

        class Hidden(Base):
            @property
            def __torch_function__(self):
                raise AttributeError("hidden by subclass")

        self.assertIs(function(Inherited()), True)
        self.assertIs(function(Hidden()), False)

        native_tensor_type = torch.Tensor
        native_tensor = torch.tensor([1.0])

        class ReplacementTensor:
            pass

        replacement = ReplacementTensor()
        try:
            torch.Tensor = ReplacementTensor
            self.assertIs(function(replacement), True)
            self.assertIs(function(native_tensor), True)
        finally:
            torch.Tensor = native_tensor_type

        self.assertIs(function(native_tensor), True)

    def test_active_torch_function_modes_do_not_intercept_or_change_results(self):
        function = torch.overrides.is_tensor_like
        tensor = torch.tensor([1.0])

        class Plain:
            pass

        class Override:
            __torch_function__ = None

        plain = Plain()
        override = Override()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return "intercepted"

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.overrides._get_current_function_mode(), mode)
            self.assertEqual(
                (function(tensor), function(plain), function(override)),
                (True, False, True),
            )
            self.assertIs(torch.overrides._get_current_function_mode(), mode)

        self.assertEqual(mode.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_signature_documentation_module_exports_and_pickle(self):
        module = importlib.import_module("torch_rs.overrides")
        function = module.is_tensor_like
        signature = inspect.Signature(
            parameters=(
                inspect.Parameter(
                    "inp", inspect.Parameter.POSITIONAL_OR_KEYWORD
                ),
            )
        )

        self.assertIs(torch.overrides, module)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "is_tensor_like")
        self.assertEqual(function.__qualname__, "is_tensor_like")
        self.assertEqual(function.__module__, module.__name__)
        self.assertIs(sys.modules[function.__module__], module)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(function.__annotations__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(inspect.signature(function), signature)
        self.assertFalse(hasattr(torch, "is_tensor_like"))
        self.assertEqual(module.__all__.count("is_tensor_like"), 1)

        namespace = {}
        exec("from torch_rs.overrides import *", namespace)
        self.assertIs(namespace["is_tensor_like"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIs(pickle.loads(payload), function)

    def test_argument_binding_matches_the_python_signature(self):
        function = torch.overrides.is_tensor_like
        tensor = torch.tensor([1.0])
        self.assertIs(function(inp=tensor), True)

        cases = (
            (
                lambda: function(),
                "is_tensor_like() missing 1 required positional argument: 'inp'",
            ),
            (
                lambda: function(tensor, tensor),
                "is_tensor_like() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(input=tensor),
                "is_tensor_like() got an unexpected keyword argument 'input'",
            ),
            (
                lambda: function(tensor, inp=tensor),
                "is_tensor_like() got multiple values for argument 'inp'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
