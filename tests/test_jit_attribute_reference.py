import copy
import importlib
import inspect
import pickle
import pickletools
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class PickleMarker:
    pass


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitAttributeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.Attribute differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, value, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(value, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def tuple_outcome(self, module):
        attribute_type = module.jit.Attribute
        value = {"items": [1, 2]}
        marker = object()
        attribute = attribute_type(value=value, type=marker)
        unpacked_value, unpacked_type = attribute
        as_dict = attribute._asdict()
        replacement = attribute._replace(value="replacement")
        made = attribute_type._make((3, str))
        return (
            type(attribute) is attribute_type,
            isinstance(attribute, tuple),
            len(attribute),
            attribute.value is value,
            attribute.type is marker,
            attribute[0] is value,
            attribute[1] is marker,
            attribute[:1] == (value,),
            unpacked_value is value,
            unpacked_type is marker,
            tuple(as_dict),
            as_dict["value"] is value,
            as_dict["type"] is marker,
            replacement == ("replacement", marker),
            replacement.type is marker,
            made == (3, str),
            made.type is str,
            attribute_type(1, int).__getnewargs__(),
            hash(attribute_type(1, str)) == hash((1, str)),
            repr(attribute_type(3, int)),
            str(attribute_type("x", list)),
        )

    def copy_outcome(self, module):
        attribute_type = module.jit.Attribute
        value = {"items": [1, 2]}
        attribute = attribute_type(value, PickleMarker)
        shallow = copy.copy(attribute)
        deep = copy.deepcopy(attribute)
        return (
            copy.copy(attribute_type) is attribute_type,
            copy.deepcopy(attribute_type) is attribute_type,
            shallow is attribute,
            shallow == attribute,
            shallow.value is value,
            shallow.type is PickleMarker,
            deep is attribute,
            deep == attribute,
            deep.value is value,
            deep.type is PickleMarker,
        )

    def test_construction_tuple_behavior_repr_and_copy_match(self):
        self.assertEqual(
            self.tuple_outcome(torch),
            self.tuple_outcome(reference_torch),
        )
        self.assertEqual(
            self.copy_outcome(torch),
            self.copy_outcome(reference_torch),
        )

    def test_signature_documentation_and_module_ownership_match(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual_script = importlib.import_module("torch_rs.jit._script")
        expected_script = importlib.import_module("torch.jit._script")
        actual = actual_jit.Attribute
        expected = expected_jit.Attribute

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(actual_jit._script, actual_script)
        self.assertIs(expected_jit._script, expected_script)
        self.assertIs(actual, actual_script.Attribute)
        self.assertIs(expected, expected_script.Attribute)
        self.assertIs(type(actual), type(expected))
        self.assertEqual(actual.__bases__, expected.__bases__)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertIs(inspect.getmodule(actual), actual_script)
        self.assertIs(inspect.getmodule(expected), expected_script)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(
            str(inspect.signature(actual.__new__)),
            str(inspect.signature(expected.__new__)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__slots__, expected.__slots__)
        self.assertEqual(actual._fields, expected._fields)
        self.assertEqual(actual._field_defaults, expected._field_defaults)
        self.assertEqual(actual.__match_args__, expected.__match_args__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual_script.__doc__, expected_script.__doc__)
        self.assertEqual(
            tuple(actual.__dict__),
            tuple(expected.__dict__),
        )

    def test_exports_and_pickling_match_pytorch_2_13(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        actual = actual_jit.Attribute
        expected = expected_jit.Attribute
        wildcard_supported = {
            "Attribute",
            "annotate",
            "export",
            "ignore",
            "isinstance",
            "onednn_fusion_enabled",
            "script_if_tracing",
            "strict_fusion",
            "unused",
        }

        self.assertEqual(
            actual_jit.__all__,
            [
                name
                for name in expected_jit.__all__
                if name in wildcard_supported
            ],
        )
        self.assertEqual(
            {name for name in vars(actual_jit) if not name.startswith("_")},
            {*wildcard_supported, "is_scripting", "is_tracing"},
        )
        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            wildcard_supported,
        )
        self.assertIs(actual_namespace["Attribute"], actual)
        self.assertIs(expected_namespace["Attribute"], expected)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("Attribute", namespace)
            self.assertFalse(hasattr(module, "Attribute"))

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="class", protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol)),
                    expected,
                )

        actual_value = actual({"items": [1, 2]}, PickleMarker)
        expected_value = expected({"items": [1, 2]}, PickleMarker)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="instance", protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual_value, protocol),
                    self.pickle_shape(expected_value, protocol),
                )
                actual_restored = pickle.loads(
                    pickle.dumps(actual_value, protocol)
                )
                expected_restored = pickle.loads(
                    pickle.dumps(expected_value, protocol)
                )
                self.assertIs(type(actual_restored), actual)
                self.assertIs(type(expected_restored), expected)
                self.assertEqual(actual_restored.value, expected_restored.value)
                self.assertIs(actual_restored.type, PickleMarker)
                self.assertIs(expected_restored.type, PickleMarker)

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.jit.Attribute
        expected = reference_torch.jit.Attribute
        cases = (
            lambda attribute_type: attribute_type(),
            lambda attribute_type: attribute_type(1),
            lambda attribute_type: attribute_type(value=1),
            lambda attribute_type: attribute_type(type=int),
            lambda attribute_type: attribute_type(1, int, 3),
            lambda attribute_type: attribute_type(1, int, other=3),
            lambda attribute_type: attribute_type(1, int, value=2),
            lambda attribute_type: attribute_type(1, int, type=str),
            lambda attribute_type: attribute_type(the_value=1, the_type=int),
        )
        for call in cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

    def test_supported_boundary_remains_eager_jit_helpers_only(self):
        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
        self.assertEqual(
            {name for name in vars(torch.jit) if not name.startswith("_")},
            {
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "is_scripting",
                "is_tracing",
                "onednn_fusion_enabled",
                "script_if_tracing",
                "strict_fusion",
                "unused",
            },
        )
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "interface",
            "script",
            "script_method",
            "trace",
            "trace_module",
        ):
            with self.subTest(name=name):
                self.assertIn(name, expected_public)
                self.assertFalse(hasattr(torch.jit, name))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
