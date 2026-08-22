import copy
import inspect
import pickle
import pickletools
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


def tensor_descriptor(module, name):
    for tensor_type in module.Tensor.__mro__:
        if name in vars(tensor_type):
            return vars(tensor_type)[name]
    raise AssertionError(f"missing Tensor descriptor {name!r}")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class IsTensorMethodOrPropertyReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_tensor_method_or_property differentials require pinned "
                "PyTorch 2.13.0"
            )

    def error_observation(self, call):
        try:
            call()
        except Exception as error:
            return type(error).__name__, str(error), error.args
        return None

    def metadata_observation(self, module):
        function = module.overrides.is_tensor_method_or_property
        direct_namespace = {}
        import_statement = (
            f"from {module.overrides.__name__} import "
            "is_tensor_method_or_property"
        )
        exec(
            import_statement,
            direct_namespace,
        )
        wildcard_namespace = {}
        exec(f"from {module.overrides.__name__} import *", wildcard_namespace)
        wrapped = function.__wrapped__
        return (
            type(function).__name__,
            function.__name__,
            function.__qualname__,
            function.__module__.rsplit(".", 1)[-1],
            function.__doc__,
            str(inspect.signature(function)),
            function.__annotations__,
            function.__defaults__,
            function.__kwdefaults__,
            hasattr(function, "__text_signature__"),
            function.__dict__.keys(),
            wrapped.__name__,
            wrapped.__qualname__,
            wrapped.__module__.rsplit(".", 1)[-1],
            wrapped.__doc__,
            str(inspect.signature(wrapped)),
            wrapped.__annotations__,
            wrapped.__defaults__,
            wrapped.__kwdefaults__,
            direct_namespace["is_tensor_method_or_property"] is function,
            wildcard_namespace["is_tensor_method_or_property"] is function,
            "is_tensor_method_or_property" in module.overrides.__all__,
            hasattr(module, "is_tensor_method_or_property"),
            "is_tensor_method_or_property" in module.__all__,
        )

    def test_signature_annotations_documentation_and_imports_match(self):
        self.assertEqual(
            self.metadata_observation(torch),
            self.metadata_observation(reference_torch),
        )

    def recognition_observation(self, module):
        function = module.overrides.is_tensor_method_or_property
        tensor = module.tensor([1.0])
        method_names = ("is_shared", "sqrt", "view", "__add__", "__pos__")
        property_names = ("real", "shape", "T", "grad")

        def unrelated_function():
            return None

        class NamedGet:
            __name__ = "__get__"

            def __call__(self):
                return None

        return (
            tuple(function(getattr(module.Tensor, name)) for name in method_names),
            tuple(
                function(tensor_descriptor(module, name).__get__)
                for name in property_names
            ),
            tuple(
                function(value)
                for value in (
                    module.sqrt,
                    module.positive,
                    module.negative,
                    unrelated_function,
                    len,
                    str.upper,
                    object.__str__,
                    module.Tensor.__str__,
                    tensor.sqrt,
                )
            ),
            tuple(function(tensor_descriptor(module, name)) for name in property_names),
            function(property(lambda self: None).__get__),
            function(NamedGet()),
        )

    def test_methods_properties_and_unrelated_callables_match_pytorch_2_13(self):
        self.assertEqual(
            self.recognition_observation(torch),
            self.recognition_observation(reference_torch),
        )

    def invalid_input_observation(self, function):
        class CallableWithoutName:
            def __call__(self):
                return None

        class HashFailure:
            def __hash__(self):
                raise RuntimeError("hash failed")

        calls = (
            lambda: function(),
            lambda: function(None, None),
            lambda: function(value=None),
            lambda: function(None),
            lambda: function(object()),
            lambda: function([]),
            lambda: function({}),
            lambda: function(CallableWithoutName()),
            lambda: function(HashFailure()),
        )
        return tuple(self.error_observation(call) for call in calls)

    def test_argument_and_invalid_input_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.invalid_input_observation(
                torch.overrides.is_tensor_method_or_property
            ),
            self.invalid_input_observation(
                reference_torch.overrides.is_tensor_method_or_property
            ),
        )
        self.assertIs(
            torch.overrides.is_tensor_method_or_property(func=torch.Tensor.sqrt),
            reference_torch.overrides.is_tensor_method_or_property(
                func=reference_torch.Tensor.sqrt
            ),
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
        return tuple(shape)

    def test_copy_and_pickle_match_pytorch_2_13(self):
        actual = torch.overrides.is_tensor_method_or_property
        expected = reference_torch.overrides.is_tensor_method_or_property
        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)),
                    actual,
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol=protocol)),
                    expected,
                )
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )


if __name__ == "__main__":
    unittest.main()
