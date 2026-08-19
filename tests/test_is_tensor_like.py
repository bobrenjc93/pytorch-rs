import inspect
import pickle
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
    def test_native_tensors_hooks_and_ordinary_objects(self):
        class ClassHook:
            __torch_function__ = None

        class InstanceHook:
            pass

        class DynamicHook:
            def __getattr__(self, name):
                if name == "__torch_function__":
                    return None
                raise AttributeError(name)

        instance_hook = InstanceHook()
        instance_hook.__torch_function__ = None
        tensor = torch.tensor([1.0, 2.0])
        values = (
            (torch.Tensor, True),
            (tensor, True),
            (tensor[0], True),
            (ClassHook(), True),
            (instance_hook, True),
            (DynamicHook(), True),
            (None, False),
            (1, False),
            (object(), False),
        )
        for value, expected in values:
            with self.subTest(value_type=type(value).__name__):
                result = torch.overrides.is_tensor_like(value)
                self.assertIs(type(result), bool)
                self.assertIs(result, expected)

    def test_attribute_lookup_and_descriptor_exceptions(self):
        events = []

        class ReturningDescriptor:
            def __get__(self, instance, owner):
                events.append(("return", instance is not None, owner))
                return None

        class MissingDescriptor:
            def __get__(self, instance, owner):
                events.append(("missing", instance is not None, owner))
                raise AttributeError("hook is hidden")

        class FailingDescriptor:
            def __get__(self, instance, owner):
                events.append(("failing", instance is not None, owner))
                raise RuntimeError("hook lookup failed")

        class ReturningHook:
            __torch_function__ = ReturningDescriptor()

        class MissingHook:
            __torch_function__ = MissingDescriptor()

        class FailingHook:
            __torch_function__ = FailingDescriptor()

        self.assertIs(torch.overrides.is_tensor_like(ReturningHook()), True)
        self.assertIs(torch.overrides.is_tensor_like(MissingHook()), False)
        with self.assertRaisesRegex(RuntimeError, "^hook lookup failed$"):
            torch.overrides.is_tensor_like(FailingHook())
        self.assertEqual(
            events,
            [
                ("return", True, ReturningHook),
                ("missing", True, MissingHook),
                ("failing", True, FailingHook),
            ],
        )

    def test_inherited_and_masked_hooks(self):
        class HookBase:
            __torch_function__ = None

        class InheritedHook(HookBase):
            pass

        class MaskedHook(HookBase):
            @property
            def __torch_function__(self):
                raise AttributeError("masked")

        self.assertIs(torch.overrides.is_tensor_like(InheritedHook()), True)
        self.assertIs(torch.overrides.is_tensor_like(MaskedHook()), False)

    def test_active_mode_does_not_intercept_the_predicate(self):
        calls = []
        tensor = torch.tensor([1.0])

        class Hook:
            __torch_function__ = None

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                calls.append((func, dispatch_types, args, kwargs))
                return "intercepted"

        with RecordingMode():
            results = (
                torch.overrides.is_tensor_like(tensor),
                torch.overrides.is_tensor_like(Hook()),
                torch.overrides.is_tensor_like(object()),
            )
        self.assertEqual(results, (True, True, False))
        self.assertEqual(calls, [])

    def test_predicate_reads_the_live_public_tensor_binding(self):
        native_tensor_type = torch.Tensor
        native_tensor = torch.tensor([1.0])
        try:
            torch.Tensor = int
            self.assertIs(
                torch.overrides.is_tensor_like(native_tensor_type), True
            )
            self.assertIs(torch.overrides.is_tensor_like(native_tensor), True)
            self.assertIs(torch.overrides.is_tensor_like(1), True)
            self.assertIs(torch.overrides.is_tensor_like(int), False)
            self.assertIs(torch.overrides.is_tensor_like("one"), False)

            torch.Tensor = 42
            self.assertIs(
                torch.overrides.is_tensor_like(native_tensor_type), True
            )
            self.assertIs(torch.overrides.is_tensor_like(native_tensor), True)
            self.assertIs(torch.overrides.is_tensor_like(1), False)
        finally:
            torch.Tensor = native_tensor_type

    def test_callable_metadata_wildcard_export_and_pickling(self):
        function = torch.overrides.is_tensor_like
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "is_tensor_like")
        self.assertEqual(function.__qualname__, "is_tensor_like")
        self.assertEqual(function.__module__, "torch_rs.overrides")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(function.__annotations__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(str(inspect.signature(function)), "(inp)")
        self.assertIn("is_tensor_like", torch.overrides.__all__)
        self.assertFalse(hasattr(torch, "is_tensor_like"))

        wildcard_namespace = {}
        exec("from torch_rs.overrides import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["is_tensor_like"], function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )


if __name__ == "__main__":
    unittest.main()
