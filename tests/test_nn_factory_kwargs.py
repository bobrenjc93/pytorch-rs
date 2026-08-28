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


class FactoryKwargsTests(unittest.TestCase):
    def test_canonical_imports_and_exports(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        from torch_rs.nn import factory_kwargs

        self.assertIs(torch.nn, nn)
        self.assertIs(imported_nn, nn)
        self.assertIs(factory_kwargs, nn.factory_kwargs)
        self.assertFalse(hasattr(nn, "__all__"))
        self.assertFalse(hasattr(torch, "factory_kwargs"))
        self.assertNotIn("nn", torch.__all__)

        namespace = {}
        exec("from torch_rs.nn import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")},
            {"factory_kwargs", "functional", "init", "modules"},
        )
        self.assertIs(namespace["factory_kwargs"], nn.factory_kwargs)

        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("nn", top_level_namespace)
        self.assertNotIn("factory_kwargs", top_level_namespace)

    def test_signature_metadata_documentation_copying_and_pickling(self):
        function = nn.factory_kwargs

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "factory_kwargs")
        self.assertEqual(function.__qualname__, "factory_kwargs")
        self.assertEqual(function.__module__, "torch_rs.nn")
        self.assertIs(inspect.getmodule(function), nn)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(str(inspect.signature(function)), "(kwargs)")
        self.assertIn(
            "Return a canonicalized dict of factory kwargs.", function.__doc__
        )
        self.assertIn("unexpected kwargs", function.__doc__)
        self.assertIn('factory_kwargs={"dtype": dtype2}', function.__doc__)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.nn", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_none_and_direct_values_return_fresh_plain_dicts(self):
        first_empty = nn.factory_kwargs(None)
        second_empty = nn.factory_kwargs(None)
        self.assertEqual(first_empty, {})
        self.assertEqual(second_empty, {})
        self.assertIs(type(first_empty), dict)
        self.assertIsNot(first_empty, second_empty)

        values = {
            "device": torch.device("cpu"),
            "dtype": torch.float32,
            "memory_format": torch.channels_last,
        }
        original = OrderedDict(values.items())
        snapshot = list(original.items())
        result = nn.factory_kwargs(original)

        self.assertIs(type(result), dict)
        self.assertIsNot(result, original)
        self.assertEqual(result, values)
        self.assertEqual(list(original.items()), snapshot)
        for key, value in values.items():
            self.assertIs(result[key], value)

        result["device"] = object()
        self.assertIs(original["device"], values["device"])

    def test_nested_values_are_shallow_copied_and_merged_without_mutation(self):
        nested_value = object()
        direct_value = object()
        nested = OrderedDict(
            [
                ("pin_memory", nested_value),
                ("dtype", torch.float32),
            ]
        )
        outer = OrderedDict(
            [
                ("factory_kwargs", nested),
                ("device", torch.device("cpu")),
                ("memory_format", direct_value),
            ]
        )
        nested_snapshot = list(nested.items())
        outer_snapshot = list(outer.items())

        result = nn.factory_kwargs(outer)

        self.assertIs(type(result), dict)
        self.assertIsNot(result, nested)
        self.assertIs(result["pin_memory"], nested_value)
        self.assertIs(result["dtype"], torch.float32)
        self.assertIs(result["device"], outer["device"])
        self.assertIs(result["memory_format"], direct_value)
        self.assertEqual(list(nested.items()), nested_snapshot)
        self.assertEqual(list(outer.items()), outer_snapshot)

        result["pin_memory"] = object()
        result["new"] = object()
        self.assertEqual(list(nested.items()), nested_snapshot)
        self.assertEqual(list(outer.items()), outer_snapshot)

        nested_only = {"factory_kwargs": {"unexpected_nested_key": nested_value}}
        copied = nn.factory_kwargs(nested_only)
        self.assertEqual(copied, nested_only["factory_kwargs"])
        self.assertIsNot(copied, nested_only["factory_kwargs"])

    def test_unexpected_and_duplicate_keys_raise_without_mutating_inputs(self):
        unexpected = OrderedDict(
            [("device", torch.device("cpu")), ("unexpected", object())]
        )
        unexpected_snapshot = list(unexpected.items())
        with self.assertRaises(TypeError) as raised:
            nn.factory_kwargs(unexpected)
        self.assertEqual(str(raised.exception), "unexpected kwargs {'unexpected'}")
        self.assertEqual(raised.exception.args, ("unexpected kwargs {'unexpected'}",))
        self.assertEqual(list(unexpected.items()), unexpected_snapshot)

        values = {
            "device": (torch.device("cpu"), "cuda"),
            "dtype": (torch.float32, object()),
            "memory_format": (torch.contiguous_format, torch.channels_last),
        }
        for key, (nested_value, direct_value) in values.items():
            nested = OrderedDict([(key, nested_value)])
            outer = OrderedDict(
                [("factory_kwargs", nested), (key, direct_value)]
            )
            nested_snapshot = list(nested.items())
            outer_snapshot = list(outer.items())
            with self.subTest(key=key):
                with self.assertRaises(TypeError) as raised:
                    nn.factory_kwargs(outer)
                self.assertEqual(
                    str(raised.exception),
                    f"{key} specified twice, in **kwargs and in factory_kwargs",
                )
                self.assertEqual(list(nested.items()), nested_snapshot)
                self.assertEqual(list(outer.items()), outer_snapshot)

    def test_mapping_access_order_and_nested_copy_protocol(self):
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

        result = nn.factory_kwargs(outer)

        self.assertEqual(
            result,
            {
                "pin_memory": "nested",
                "layout": "strided",
                "device": "cpu",
                "dtype": "float32",
                "memory_format": "channels_last",
            },
        )
        expected_outer_accesses = [("keys",), ("get", "factory_kwargs")]
        for key in {"device", "dtype", "memory_format"}:
            expected_outer_accesses.extend(
                [("contains", key), ("getitem", key)]
            )
        self.assertEqual(outer.accesses, expected_outer_accesses)
        self.assertEqual(
            nested.accesses,
            [
                ("keys",),
                ("getitem", "pin_memory"),
                ("getitem", "layout"),
            ],
        )

        invalid = _TracingMapping([("unexpected", 1)])
        with self.assertRaisesRegex(TypeError, "^unexpected kwargs"):
            nn.factory_kwargs(invalid)
        self.assertEqual(invalid.accesses, [("keys",), ("keys",)])

    def test_python_call_validation(self):
        calls = (
            lambda: nn.factory_kwargs(),
            lambda: nn.factory_kwargs({}, {}),
            lambda: nn.factory_kwargs(options={}),
            lambda: nn.factory_kwargs({}, kwargs={}),
        )
        for case, call in enumerate(calls):
            with self.subTest(case=case):
                with self.assertRaises(TypeError):
                    call()

    def test_reload_replaces_function_and_preserves_import_graph(self):
        namespace = nn.__dict__
        children = (nn.functional, nn.init, nn.modules)
        old_function = nn.factory_kwargs
        old_payload = pickle.dumps(old_function)

        reloaded = importlib.reload(nn)
        new_function = nn.factory_kwargs

        self.assertIs(reloaded, nn)
        self.assertIs(nn.__dict__, namespace)
        self.assertIs(torch.nn, nn)
        self.assertIs(sys.modules["torch_rs.nn"], nn)
        self.assertEqual((nn.functional, nn.init, nn.modules), children)
        self.assertIsNot(new_function, old_function)
        self.assertEqual(old_function(None), {})
        self.assertEqual(new_function(None), {})
        self.assertIs(pickle.loads(old_payload), new_function)
        self.assertIs(pickle.loads(pickle.dumps(new_function)), new_function)

        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function factory_kwargs at 0x...>: "
            "it's not the same object as torch_rs.nn.factory_kwargs",
        )


if __name__ == "__main__":
    unittest.main()
