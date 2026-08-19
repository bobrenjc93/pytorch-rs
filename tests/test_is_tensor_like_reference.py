import copy
import importlib
import inspect
import pickle
import pickletools
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class IsTensorLikeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_tensor_like differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def result_observation(self, module):
        function = module.overrides.is_tensor_like
        dynamic_lookups = []

        class Plain:
            pass

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

        class HiddenOverride(BaseOverride):
            @property
            def __torch_function__(self):
                raise AttributeError("hidden by subclass")

        class InstanceOverride:
            pass

        class DynamicOverride:
            def __getattr__(self, name):
                dynamic_lookups.append(name)
                if name == "__torch_function__":
                    return 0
                raise AttributeError(name)

        instance_override = InstanceOverride()
        instance_override.__torch_function__ = "present"
        values = (
            module.tensor(1.0),
            module.zeros((2, 3)).transpose(0, 1),
            module.Tensor,
            ClassOverride(),
            FalseyOverride(),
            DerivedOverride(),
            HiddenOverride(),
            instance_override,
            DynamicOverride(),
            None,
            1,
            "tensor",
            [],
            object(),
            Plain(),
            Plain,
            module.float32,
            module.device("cpu"),
            module.overrides,
        )
        results = []
        for value in values:
            result = function(value)
            results.append((type(result).__name__, result))
        return tuple(results), tuple(dynamic_lookups)

    def test_results_instance_attributes_and_inheritance_match_pytorch_2_13(self):
        self.assertEqual(
            self.result_observation(torch),
            self.result_observation(reference_torch),
        )

    def descriptor_observation(self, module):
        function = module.overrides.is_tensor_like
        events = []

        class PresentDescriptor:
            def __get__(self, instance, owner):
                events.append(("present", instance is not None, owner.__name__))
                return None

        class Present:
            __torch_function__ = PresentDescriptor()

        class MissingDescriptor:
            def __get__(self, instance, owner):
                events.append(("missing", instance is not None, owner.__name__))
                raise AttributeError("hidden")

        class Missing:
            __torch_function__ = MissingDescriptor()

        class HiddenAttributeError(AttributeError):
            pass

        class MissingSubclass:
            @property
            def __torch_function__(self):
                events.append("attribute-error-subclass")
                raise HiddenAttributeError("also hidden")

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

        expected_error = RuntimeError("descriptor failure")

        class Raising:
            @property
            def __torch_function__(self):
                events.append("raising")
                raise expected_error

        results = (
            function(Present()),
            function(Missing()),
            function(MissingSubclass()),
            function(Fallback()),
        )
        try:
            function(Raising())
        except BaseException as error:
            error_outcome = (
                type(error).__name__,
                str(error),
                error is expected_error,
            )
        else:
            self.fail(f"{module.__name__}.is_tensor_like swallowed RuntimeError")
        return results, tuple(events), error_outcome

    def test_attribute_lookup_and_descriptor_exceptions_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_observation(torch),
            self.descriptor_observation(reference_torch),
        )

    def rebound_tensor_observation(self, module):
        function = module.overrides.is_tensor_like
        native_tensor_type = module.Tensor
        native_tensor = module.tensor([1.0])

        class ReplacementTensor:
            pass

        replacement = ReplacementTensor()
        try:
            module.Tensor = ReplacementTensor
            rebound = (function(replacement), function(native_tensor))
            module.Tensor = 42
            invalid_binding = (function(replacement), function(native_tensor))
        finally:
            module.Tensor = native_tensor_type

        return rebound, invalid_binding, function(native_tensor)

    def test_live_public_tensor_binding_matches_pytorch_2_13(self):
        self.assertEqual(
            self.rebound_tensor_observation(torch),
            self.rebound_tensor_observation(reference_torch),
        )

    def mode_observation(self, module):
        function = module.overrides.is_tensor_like
        tensor = module.tensor([1.0])

        class Plain:
            pass

        class Override:
            __torch_function__ = None

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return "intercepted"

        mode = RecordingMode()
        with mode:
            before = module.overrides._get_current_function_mode() is mode
            results = (function(tensor), function(Plain()), function(Override()))
            after = module.overrides._get_current_function_mode() is mode
        return (
            before,
            results,
            after,
            len(mode.calls),
            len(module.overrides._get_current_function_mode_stack()),
        )

    def test_active_mode_independence_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_observation(torch),
            self.mode_observation(reference_torch),
        )

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def test_signature_documentation_module_exports_and_pickle_match(self):
        actual_module = importlib.import_module("torch_rs.overrides")
        expected_module = importlib.import_module("torch.overrides")
        actual = actual_module.is_tensor_like
        expected = expected_module.is_tensor_like

        self.assertIs(torch.overrides, actual_module)
        self.assertIs(reference_torch.overrides, expected_module)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertIs(sys.modules[actual.__module__], actual_module)
        self.assertIs(sys.modules[expected.__module__], expected_module)
        self.assertFalse(hasattr(torch, "is_tensor_like"))
        self.assertFalse(hasattr(reference_torch, "is_tensor_like"))
        self.assertEqual(
            actual_module.__all__.count("is_tensor_like"),
            expected_module.__all__.count("is_tensor_like"),
        )

        for module, function in (
            (actual_module, actual),
            (expected_module, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["is_tensor_like"], function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol)), expected
                )
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_binding_errors_match_pytorch_2_13(self):
        actual = torch.overrides.is_tensor_like
        expected = reference_torch.overrides.is_tensor_like
        actual_tensor = torch.tensor([1.0])
        expected_tensor = reference_torch.tensor([1.0])
        self.assertIs(actual(inp=actual_tensor), expected(inp=expected_tensor))

        cases = (
            (lambda: actual(), lambda: expected()),
            (
                lambda: actual(actual_tensor, actual_tensor),
                lambda: expected(expected_tensor, expected_tensor),
            ),
            (
                lambda: actual(input=actual_tensor),
                lambda: expected(input=expected_tensor),
            ),
            (
                lambda: actual(actual_tensor, inp=actual_tensor),
                lambda: expected(expected_tensor, inp=expected_tensor),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
