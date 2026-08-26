import copy
import importlib
import inspect
import pickle
import unittest
from collections import OrderedDict

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class StateDict(dict):
    pass


def mapping_snapshot(mapping, values):
    return [(key, values.index(value)) for key, value in mapping.items()]


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ConsumePrefixInStateDictReferenceTests(unittest.TestCase):
    def setUp(self):
        self.actual = torch.nn.modules.utils.consume_prefix_in_state_dict_if_present
        self.expected = (
            reference_torch.nn.modules.utils.consume_prefix_in_state_dict_if_present
        )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def mutation_outcome(self, function, mapping_type):
        values = [object() for _ in range(4)]
        state_dict = mapping_type(
            [
                ("module.encoder.weight", values[0]),
                ("unrelated", values[1]),
                ("module.encoder.bias", values[2]),
                ("encoder.weight", values[3]),
            ]
        )
        identity = id(state_dict)
        result = function(state_dict, "module.")
        return (
            result,
            id(state_dict) == identity,
            mapping_snapshot(state_dict, values),
        )

    def metadata_outcome(self, function, mapping_type):
        state_values = [object(), object()]
        metadata_values = [object() for _ in range(5)]
        state_dict = mapping_type(
            [("module.value", state_values[0]), ("plain", state_values[1])]
        )
        state_dict._metadata = OrderedDict(
            [
                ("", metadata_values[0]),
                ("module", metadata_values[1]),
                ("module.layer", metadata_values[2]),
                ("other", metadata_values[3]),
                ("modulex", metadata_values[4]),
            ]
        )
        metadata = state_dict._metadata
        result = function(state_dict, "module.")
        return (
            result,
            mapping_snapshot(state_dict, state_values),
            state_dict._metadata is metadata,
            mapping_snapshot(metadata, metadata_values),
        )

    def no_op_outcome(self, function, prefix):
        values = [object(), object()]
        metadata_values = [object(), object()]
        state_dict = StateDict([("weight", values[0]), ("bias", values[1])])
        state_dict._metadata = OrderedDict(
            [("", metadata_values[0]), ("layer", metadata_values[1])]
        )
        result = function(state_dict, prefix)
        return (
            result,
            mapping_snapshot(state_dict, values),
            mapping_snapshot(state_dict._metadata, metadata_values),
        )

    def test_in_place_renaming_order_collisions_and_identity_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for mapping_type in (dict, OrderedDict):
            with self.subTest(mapping_type=mapping_type.__name__):
                self.assertEqual(
                    self.mutation_outcome(self.actual, mapping_type),
                    self.mutation_outcome(self.expected, mapping_type),
                )

    def test_metadata_root_and_nested_key_handling_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for mapping_type in (StateDict, OrderedDict):
            with self.subTest(mapping_type=mapping_type.__name__):
                self.assertEqual(
                    self.metadata_outcome(self.actual, mapping_type),
                    self.metadata_outcome(self.expected, mapping_type),
                )

    def test_no_op_and_empty_prefix_behavior_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for prefix in ("missing.", ""):
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    self.no_op_outcome(self.actual, prefix),
                    self.no_op_outcome(self.expected, prefix),
                )

        self.assertIsNone(self.actual({}, None))
        self.assertIsNone(self.expected({}, None))

    def test_errors_and_partial_mutation_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        error_cases = (
            (lambda function: function(None, "module.")),
            (lambda function: function({"value": object()}, None)),
            (lambda function: function({1: object()}, "module.")),
        )
        for case, make_call in enumerate(error_cases):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda make_call=make_call: make_call(self.actual),
                    lambda make_call=make_call: make_call(self.expected),
                )

        def partial_key_failure(function):
            values = [object(), object(), object()]
            state_dict = OrderedDict(
                [
                    ("module.first", values[0]),
                    (7, values[1]),
                    ("module.last", values[2]),
                ]
            )
            try:
                function(state_dict, "module.")
            except Exception as error:
                return (
                    type(error),
                    str(error),
                    error.args,
                    mapping_snapshot(state_dict, values),
                )
            self.fail("expected the non-string key to raise")

        self.assertEqual(
            partial_key_failure(self.actual), partial_key_failure(self.expected)
        )

        def partial_metadata_failure(function):
            values = [object(), object(), object()]
            state_dict = StateDict([("module.value", values[0])])
            state_dict._metadata = OrderedDict(
                [("module.layer", values[1]), (None, values[2])]
            )
            try:
                function(state_dict, "module.")
            except Exception as error:
                return (
                    type(error),
                    str(error),
                    error.args,
                    mapping_snapshot(state_dict, values),
                    mapping_snapshot(state_dict._metadata, values),
                )
            self.fail("expected the invalid metadata key to raise")

        self.assertEqual(
            partial_metadata_failure(self.actual),
            partial_metadata_failure(self.expected),
        )

    def test_signature_documentation_imports_and_pickle_identity_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_utils = importlib.import_module("torch_rs.nn.modules.utils")
        expected_utils = importlib.import_module("torch.nn.modules.utils")
        actual_modules = importlib.import_module("torch_rs.nn.modules")
        expected_modules = importlib.import_module("torch.nn.modules")

        self.assertIs(torch.nn.modules.utils, actual_utils)
        self.assertIs(reference_torch.nn.modules.utils, expected_utils)
        self.assertIs(actual_modules.utils, actual_utils)
        self.assertIs(expected_modules.utils, expected_utils)
        self.assertIs(actual_utils.consume_prefix_in_state_dict_if_present, self.actual)
        self.assertIs(
            expected_utils.consume_prefix_in_state_dict_if_present, self.expected
        )

        self.assertEqual(
            inspect.signature(self.actual), inspect.signature(self.expected)
        )
        self.assertEqual(self.actual.__annotations__, self.expected.__annotations__)
        self.assertEqual(self.actual.__doc__, self.expected.__doc__)
        self.assertEqual(
            self.actual.__module__.replace("torch_rs", "torch", 1),
            self.expected.__module__,
        )
        self.assertEqual(self.actual.__name__, self.expected.__name__)
        self.assertEqual(self.actual.__qualname__, self.expected.__qualname__)
        self.assertEqual(self.actual.__defaults__, self.expected.__defaults__)
        self.assertEqual(self.actual.__kwdefaults__, self.expected.__kwdefaults__)
        self.assertEqual(self.actual.__dict__, self.expected.__dict__)
        self.assertEqual(actual_utils.__all__, expected_utils.__all__)

        for function in (self.actual, self.expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(module=function.__module__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol)), function
                    )

        for modules_module in (actual_modules, expected_modules):
            self.assertNotIn(self.actual.__name__, modules_module.__dict__)
            namespace = {}
            exec(f"from {modules_module.__name__} import *", namespace)
            self.assertNotIn(self.actual.__name__, namespace)
            self.assertNotIn("utils", namespace)

    def test_argument_errors_match_and_larger_serialization_scope_stays_absent(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        calls = (
            lambda function: function(),
            lambda function: function({}),
            lambda function: function({}, "module.", None),
            lambda function: function(mapping={}, prefix="module."),
            lambda function: function({}, "module.", prefix="other."),
        )
        for case, make_call in enumerate(calls):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda make_call=make_call: make_call(self.actual),
                    lambda make_call=make_call: make_call(self.expected),
                )

        self.assertTrue(hasattr(reference_torch.nn, "Module"))
        self.assertFalse(hasattr(torch.nn, "Module"))
        for name in ("save", "load"):
            self.assertTrue(hasattr(reference_torch, name))
            self.assertFalse(hasattr(torch, name))
            self.assertTrue(hasattr(reference_torch.serialization, name))
            self.assertFalse(hasattr(torch.serialization, name))


if __name__ == "__main__":
    unittest.main()
