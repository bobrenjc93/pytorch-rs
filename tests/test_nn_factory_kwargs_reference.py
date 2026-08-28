import copy
import importlib
import inspect
import pickle
import re
import sys
import types
import unittest
from collections import OrderedDict

import torch_rs as torch
import torch_rs.nn as nn

try:
    import torch as reference_torch
    import torch.nn as reference_nn
except ImportError:
    reference_torch = None
    reference_nn = None


class _TracingMapping:
    def __init__(self, items):
        self.values = OrderedDict(items)
        self.accesses = []

    def keys(self):
        self.accesses.append(("keys",))
        return self.values.keys()

    def get(self, key, default=None):
        self.accesses.append(("get", key))
        return self.values.get(key, default)

    def __contains__(self, key):
        self.accesses.append(("contains", key))
        return key in self.values

    def __getitem__(self, key):
        self.accesses.append(("getitem", key))
        return self.values[key]


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class FactoryKwargsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.factory_kwargs differentials require pinned PyTorch 2.13.0"
            )

    def outcome(self, function, kwargs):
        try:
            value = function(kwargs)
        except Exception as error:
            return ("error", type(error).__name__, str(error), error.args)
        return ("return", type(value).__name__, list(value.items()))

    def assert_case_matches(self, actual_kwargs, expected_kwargs, *, case):
        with self.subTest(case=case):
            self.assertEqual(
                self.outcome(nn.factory_kwargs, actual_kwargs),
                self.outcome(reference_nn.factory_kwargs, expected_kwargs),
            )

    def test_none_direct_nested_and_error_behavior_match_pytorch_2_13(self):
        cases = (
            (None, None),
            ({}, {}),
            ({"device": "cpu"}, {"device": "cpu"}),
            ({"dtype": "float32"}, {"dtype": "float32"}),
            (
                {"memory_format": "channels_last"},
                {"memory_format": "channels_last"},
            ),
            (
                OrderedDict(
                    [
                        ("device", "cpu"),
                        ("dtype", "float32"),
                        ("memory_format", "channels_last"),
                    ]
                ),
                OrderedDict(
                    [
                        ("device", "cpu"),
                        ("dtype", "float32"),
                        ("memory_format", "channels_last"),
                    ]
                ),
            ),
            (
                {
                    "factory_kwargs": OrderedDict(
                        [("pin_memory", True), ("layout", "strided")]
                    ),
                    "device": "cpu",
                },
                {
                    "factory_kwargs": OrderedDict(
                        [("pin_memory", True), ("layout", "strided")]
                    ),
                    "device": "cpu",
                },
            ),
            (
                {"factory_kwargs": {"unexpected_nested": 1}},
                {"factory_kwargs": {"unexpected_nested": 1}},
            ),
            ({"unexpected": 1}, {"unexpected": 1}),
            ({1: "unexpected"}, {1: "unexpected"}),
            (
                {"factory_kwargs": {"device": "nested"}, "device": "direct"},
                {"factory_kwargs": {"device": "nested"}, "device": "direct"},
            ),
            (
                {"factory_kwargs": {"dtype": "nested"}, "dtype": "direct"},
                {"factory_kwargs": {"dtype": "nested"}, "dtype": "direct"},
            ),
            (
                {
                    "factory_kwargs": {"memory_format": "nested"},
                    "memory_format": "direct",
                },
                {
                    "factory_kwargs": {"memory_format": "nested"},
                    "memory_format": "direct",
                },
            ),
            ({"factory_kwargs": None}, {"factory_kwargs": None}),
            ({"factory_kwargs": 1}, {"factory_kwargs": 1}),
            ([], []),
            (object(), object()),
        )
        for case, (actual_kwargs, expected_kwargs) in enumerate(cases):
            self.assert_case_matches(actual_kwargs, expected_kwargs, case=case)

    def test_fresh_result_identity_and_input_preservation_match(self):
        def contract(function):
            nested_value = object()
            direct_value = object()
            nested = OrderedDict(
                [("pin_memory", nested_value), ("dtype", "float32")]
            )
            outer = OrderedDict(
                [("factory_kwargs", nested), ("device", direct_value)]
            )
            nested_snapshot = list(nested.items())
            outer_snapshot = list(outer.items())
            first_none = function(None)
            second_none = function(None)
            result = function(outer)
            result["pin_memory"] = object()
            return (
                type(first_none) is dict,
                first_none == {},
                first_none is not second_none,
                type(result) is dict,
                result is not nested,
                result["device"] is direct_value,
                list(nested.items()) == nested_snapshot,
                list(outer.items()) == outer_snapshot,
                nested["pin_memory"] is nested_value,
            )

        self.assertEqual(
            contract(nn.factory_kwargs), contract(reference_nn.factory_kwargs)
        )

    def mapping_contract(self, function):
        nested = _TracingMapping(
            [("pin_memory", "nested"), ("layout", "strided")]
        )
        outer = _TracingMapping(
            [
                ("factory_kwargs", nested),
                ("device", "cpu"),
                ("dtype", "float32"),
                ("memory_format", "channels_last"),
            ]
        )
        outcome = self.outcome(function, outer)
        return outcome, outer.accesses, nested.accesses

    def invalid_mapping_contract(self, function):
        mapping = _TracingMapping([("unexpected", 1)])
        outcome = self.outcome(function, mapping)
        return outcome, mapping.accesses

    def duplicate_mapping_contract(self, function, duplicate):
        nested = _TracingMapping([(duplicate, "nested")])
        outer = _TracingMapping(
            [("factory_kwargs", nested), (duplicate, "direct")]
        )
        outcome = self.outcome(function, outer)
        return outcome, outer.accesses, nested.accesses

    def test_mapping_access_copying_and_failure_order_match(self):
        self.assertEqual(
            self.mapping_contract(nn.factory_kwargs),
            self.mapping_contract(reference_nn.factory_kwargs),
        )
        self.assertEqual(
            self.invalid_mapping_contract(nn.factory_kwargs),
            self.invalid_mapping_contract(reference_nn.factory_kwargs),
        )
        for duplicate in ("device", "dtype", "memory_format"):
            with self.subTest(duplicate=duplicate):
                self.assertEqual(
                    self.duplicate_mapping_contract(nn.factory_kwargs, duplicate),
                    self.duplicate_mapping_contract(
                        reference_nn.factory_kwargs, duplicate
                    ),
                )

    def test_signature_documentation_metadata_imports_and_pickle_match(self):
        actual = nn.factory_kwargs
        expected = reference_nn.factory_kwargs

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch", 1),
            expected.__module__,
        )
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertIs(inspect.getmodule(actual), nn)
        self.assertIs(inspect.getmodule(expected), reference_nn)
        self.assertEqual(hasattr(nn, "__all__"), hasattr(reference_nn, "__all__"))

        actual_imported = importlib.import_module("torch_rs.nn")
        expected_imported = importlib.import_module("torch.nn")
        self.assertIs(actual_imported, nn)
        self.assertIs(expected_imported, reference_nn)

        for module, function in ((nn, actual), (reference_nn, expected)):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["factory_kwargs"], function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(module=module.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol)), function
                    )

        self.assertFalse(hasattr(torch, "factory_kwargs"))
        self.assertFalse(hasattr(reference_torch, "factory_kwargs"))

    def reload_contract(self, root):
        module = root.nn
        namespace = module.__dict__
        children = (module.functional, module.init, module.modules)
        old_function = module.factory_kwargs
        old_payload = pickle.dumps(old_function)
        reloaded = importlib.reload(module)
        new_function = module.factory_kwargs

        try:
            pickle.dumps(old_function)
        except Exception as error:
            stale_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                    "torch_rs", "torch"
                ),
            )
        else:
            self.fail("a stale factory_kwargs function remained pickleable")

        return (
            reloaded is module,
            module.__dict__ is namespace,
            root.nn is module,
            sys.modules[module.__name__] is module,
            (module.functional, module.init, module.modules) == children,
            new_function is not old_function,
            old_function(None) == {},
            new_function(None) == {},
            pickle.loads(old_payload) is new_function,
            pickle.loads(pickle.dumps(new_function)) is new_function,
            copy.copy(new_function) is new_function,
            copy.deepcopy(new_function) is new_function,
            stale_error,
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch), self.reload_contract(reference_torch)
        )

    def test_python_call_validation_matches(self):
        calls = (
            lambda function: function(),
            lambda function: function({}, {}),
            lambda function: function(options={}),
            lambda function: function({}, kwargs={}),
        )
        for case, make_call in enumerate(calls):
            with self.subTest(case=case):
                actual = self.outcome_from_call(lambda: make_call(nn.factory_kwargs))
                expected = self.outcome_from_call(
                    lambda: make_call(reference_nn.factory_kwargs)
                )
                self.assertEqual(actual, expected)

    def outcome_from_call(self, call):
        try:
            call()
        except Exception as error:
            return (type(error).__name__, str(error), error.args)
        self.fail("expected call validation to fail")


if __name__ == "__main__":
    unittest.main()
