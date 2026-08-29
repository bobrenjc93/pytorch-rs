import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import unittest
from collections import OrderedDict, UserDict

import torch_rs as torch
import torch_rs.nn as nn


class _TrackingOuterMapping:
    def __init__(self, values):
        self.values = values
        self.events = []

    def keys(self):
        self.events.append(("keys",))
        return self.values.keys()

    def get(self, key, default=None):
        self.events.append(("get", key, default))
        return self.values.get(key, default)

    def __contains__(self, key):
        self.events.append(("contains", key))
        return key in self.values

    def __getitem__(self, key):
        self.events.append(("getitem", key))
        return self.values[key]


class _TrackingNestedMapping:
    def __init__(self, values):
        self.values = values
        self.events = []

    def keys(self):
        self.events.append(("keys",))
        return self.values.keys()

    def __getitem__(self, key):
        self.events.append(("getitem", key))
        return self.values[key]


class FactoryKwargsTests(unittest.TestCase):
    def test_canonical_imports_and_exports(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        from torch_rs import nn as from_torch
        from torch_rs.nn import factory_kwargs

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(from_torch, nn)
        self.assertIs(factory_kwargs, nn.factory_kwargs)
        self.assertFalse(hasattr(torch, "factory_kwargs"))
        self.assertNotIn("nn", torch.__all__)
        self.assertFalse(hasattr(nn, "__all__"))
        self.assertIsNone(nn.__doc__)

        wildcard_namespace = {}
        exec("from torch_rs.nn import *", wildcard_namespace)
        self.assertEqual(
            {name for name in wildcard_namespace if not name.startswith("_")},
            {"Parameter", "factory_kwargs", "functional", "init", "modules"},
        )
        self.assertIs(wildcard_namespace["factory_kwargs"], nn.factory_kwargs)

    def test_signature_metadata_documentation_copying_and_pickle(self):
        function = nn.factory_kwargs

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "factory_kwargs")
        self.assertEqual(function.__qualname__, "factory_kwargs")
        self.assertEqual(function.__module__, "torch_rs.nn")
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(str(inspect.signature(function)), "(kwargs)")
        self.assertIs(inspect.getmodule(function), nn)
        documentation = inspect.cleandoc(function.__doc__)
        self.assertTrue(
            documentation.startswith(
                "Return a canonicalized dict of factory kwargs."
            )
        )
        self.assertIn("factory_kwargs = torch.nn.factory_kwargs(kwargs)", documentation)
        self.assertIn("This function does error validation", documentation)
        self.assertIn(
            'f(dtype1, factory_kwargs={"dtype": dtype2})', documentation
        )

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.nn", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_none_empty_and_direct_kwargs_return_fresh_exact_dicts(self):
        first_none = nn.factory_kwargs(None)
        second_none = nn.factory_kwargs(None)
        empty = {}
        from_empty = nn.factory_kwargs(empty)

        for result in (first_none, second_none, from_empty):
            self.assertIs(type(result), dict)
            self.assertEqual(result, {})
        self.assertIsNot(first_none, second_none)
        self.assertIsNot(from_empty, empty)

        device = torch.device("cpu")
        dtype = torch.float32
        memory_format = torch.channels_last
        kwargs = {
            "device": device,
            "dtype": dtype,
            "memory_format": memory_format,
        }
        original = kwargs.copy()

        result = nn.factory_kwargs(kwargs)

        self.assertIs(type(result), dict)
        self.assertEqual(result, original)
        self.assertIsNot(result, kwargs)
        self.assertEqual(kwargs, original)
        for key, value in original.items():
            self.assertIs(result[key], value)

        user_dict = UserDict(kwargs)
        from_user_dict = nn.factory_kwargs(user_dict)
        self.assertIs(type(from_user_dict), dict)
        self.assertEqual(from_user_dict, kwargs)
        self.assertIsNot(from_user_dict, user_dict)

    def test_nested_factory_kwargs_are_shallow_copied_and_combined(self):
        device = torch.device("cpu")
        mutable_value = []
        nested = OrderedDict(
            (("pin_memory", True), ("custom_factory_option", mutable_value))
        )
        kwargs = {"device": device, "factory_kwargs": nested}
        original_outer = kwargs.copy()
        original_nested = nested.copy()

        result = nn.factory_kwargs(kwargs)

        self.assertIs(type(result), dict)
        self.assertEqual(
            result,
            {
                "pin_memory": True,
                "custom_factory_option": mutable_value,
                "device": device,
            },
        )
        self.assertEqual(
            list(result), ["pin_memory", "custom_factory_option", "device"]
        )
        self.assertIs(result["custom_factory_option"], mutable_value)
        self.assertIsNot(result, nested)
        self.assertEqual(kwargs, original_outer)
        self.assertEqual(nested, original_nested)

        result["pin_memory"] = False
        result["added"] = object()
        self.assertEqual(nested, original_nested)
        nested["later"] = 1
        self.assertNotIn("later", result)

        pairs = [("layout", torch.strided), ("requires_grad", False)]
        from_pairs = nn.factory_kwargs({"factory_kwargs": pairs})
        self.assertIs(type(from_pairs), dict)
        self.assertEqual(from_pairs, dict(pairs))
        self.assertEqual(pairs, [("layout", torch.strided), ("requires_grad", False)])

    def test_unexpected_and_duplicate_keys_raise_without_mutation(self):
        kwargs = {"unexpected": object()}
        original = kwargs.copy()
        with self.assertRaisesRegex(TypeError, r"^unexpected kwargs \{'unexpected'\}$"):
            nn.factory_kwargs(kwargs)
        self.assertEqual(kwargs, original)

        for key in ("device", "dtype", "memory_format"):
            direct_value = object()
            nested_value = object()
            nested = {key: nested_value}
            kwargs = {key: direct_value, "factory_kwargs": nested}
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{key} specified twice, in \*\*kwargs and in factory_kwargs$",
                ):
                    nn.factory_kwargs(kwargs)
                self.assertIs(kwargs[key], direct_value)
                self.assertIs(kwargs["factory_kwargs"], nested)
                self.assertIs(nested[key], nested_value)

        nested_extra = {"unexpected": object()}
        result = nn.factory_kwargs({"factory_kwargs": nested_extra})
        self.assertEqual(result, nested_extra)
        self.assertIsNot(result, nested_extra)

    def test_outer_and_nested_mapping_access_protocol(self):
        nested = _TrackingNestedMapping(
            OrderedDict((("pin_memory", True), ("requires_grad", False)))
        )
        direct_values = {
            "device": "cpu",
            "dtype": "float32",
            "memory_format": "contiguous",
            "factory_kwargs": nested,
        }
        outer = _TrackingOuterMapping(direct_values)

        result = nn.factory_kwargs(outer)

        self.assertEqual(
            result,
            {
                "pin_memory": True,
                "requires_grad": False,
                "device": "cpu",
                "dtype": "float32",
                "memory_format": "contiguous",
            },
        )
        self.assertEqual(outer.events[0], ("keys",))
        self.assertEqual(outer.events[1][:2], ("get", "factory_kwargs"))
        self.assertIs(type(outer.events[1][2]), dict)
        self.assertEqual(outer.events[1][2], {})

        remaining_events = iter(outer.events[2:])
        visited = []
        for event in remaining_events:
            self.assertEqual(event[0], "contains")
            key = event[1]
            visited.append(key)
            self.assertEqual(next(remaining_events), ("getitem", key))
        self.assertEqual(set(visited), {"device", "dtype", "memory_format"})
        self.assertEqual(list(result)[2:], visited)
        self.assertEqual(
            nested.events,
            [("keys",), ("getitem", "pin_memory"), ("getitem", "requires_grad")],
        )

        unexpected = _TrackingOuterMapping({"unexpected": 1})
        with self.assertRaisesRegex(TypeError, "unexpected kwargs"):
            nn.factory_kwargs(unexpected)
        self.assertEqual(unexpected.events, [("keys",), ("keys",)])

        duplicate = _TrackingOuterMapping(
            {"device": "outer", "factory_kwargs": {"device": "nested"}}
        )
        with self.assertRaisesRegex(TypeError, "device specified twice"):
            nn.factory_kwargs(duplicate)
        self.assertEqual(duplicate.events[0], ("keys",))
        self.assertEqual(duplicate.events[1][:2], ("get", "factory_kwargs"))
        self.assertNotIn("getitem", {event[0] for event in duplicate.events})
        self.assertEqual(duplicate.events[-1], ("contains", "device"))

    def test_invalid_call_and_mapping_errors(self):
        invalid_calls = (
            lambda: nn.factory_kwargs(),
            lambda: nn.factory_kwargs({}, {}),
            lambda: nn.factory_kwargs(options={}),
            lambda: nn.factory_kwargs({}, kwargs={}),
            lambda: nn.factory_kwargs(kwargs={}, unexpected=True),
        )
        for case, call in enumerate(invalid_calls):
            with self.subTest(case=case):
                with self.assertRaises(TypeError):
                    call()

        for value in (0, [], object()):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(AttributeError):
                    nn.factory_kwargs(value)

        with self.assertRaisesRegex(
            TypeError, "'NoneType' object is not iterable"
        ):
            nn.factory_kwargs({"factory_kwargs": None})

    def test_reload_replaces_only_the_function_definition(self):
        old_function = nn.factory_kwargs
        namespace = nn.__dict__
        functional = nn.functional
        init = nn.init
        modules = nn.modules

        reloaded = importlib.reload(nn)

        self.assertIs(reloaded, nn)
        self.assertIs(nn.__dict__, namespace)
        self.assertIs(torch.nn, nn)
        self.assertIs(nn.functional, functional)
        self.assertIs(nn.init, init)
        self.assertIs(nn.modules, modules)
        self.assertIsNot(nn.factory_kwargs, old_function)
        self.assertEqual(str(inspect.signature(nn.factory_kwargs)), "(kwargs)")
        self.assertEqual(nn.factory_kwargs({"device": "cpu"}), {"device": "cpu"})

        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(nn.factory_kwargs, protocol)),
                    nn.factory_kwargs,
                )

    def test_import_reload_and_use_do_not_import_pytorch(self):
        code = r"""
import copy
import importlib
import pickle
import sys

assert "torch" not in sys.modules
import torch_rs
import torch_rs.nn as nn
from torch_rs.nn import factory_kwargs
assert "torch" not in sys.modules
assert torch_rs.nn is nn
assert factory_kwargs is nn.factory_kwargs
assert copy.copy(factory_kwargs) is factory_kwargs
assert pickle.loads(pickle.dumps(factory_kwargs)) is factory_kwargs
old_function = factory_kwargs
assert importlib.reload(nn) is nn
assert nn.factory_kwargs is not old_function
assert nn.factory_kwargs(None) == {}
assert "torch" not in sys.modules
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
