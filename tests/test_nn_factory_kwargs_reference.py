import copy
import importlib
import inspect
import pickle
import pickletools
import re
import types
import unittest
from collections import OrderedDict, UserDict

import torch_rs as torch
import torch_rs.nn as nn

try:
    import torch as reference_torch
    import torch.nn as reference_nn
except ImportError:
    reference_torch = None
    reference_nn = None


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


class _ListKeysMapping:
    def keys(self):
        return ["device"]


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class FactoryKwargsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.factory_kwargs differentials require pinned PyTorch 2.13.0"
            )

    def assert_errors_match(self, actual_call, expected_call):
        with self.assertRaises(BaseException) as actual_raised:
            actual_call()
        with self.assertRaises(BaseException) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

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

    def test_canonical_values_freshness_and_input_preservation_match(self):
        device = object()
        dtype = object()
        memory_format = object()
        extra = object()
        cases = (
            lambda: None,
            lambda: {},
            lambda: {"device": device},
            lambda: {"dtype": dtype},
            lambda: {"memory_format": memory_format},
            lambda: {
                "device": device,
                "dtype": dtype,
                "memory_format": memory_format,
            },
            lambda: {
                "factory_kwargs": OrderedDict(
                    (
                        ("device", device),
                        ("dtype", dtype),
                        ("memory_format", memory_format),
                        ("extra", extra),
                    )
                )
            },
            lambda: {
                "device": device,
                "factory_kwargs": OrderedDict(
                    (("dtype", dtype), ("extra", extra))
                ),
            },
            lambda: {
                "factory_kwargs": [("dtype", dtype), ("extra", extra)]
            },
        )

        for case, make_input in enumerate(cases):
            actual_input = make_input()
            expected_input = make_input()
            actual_outer_snapshot = copy.copy(actual_input)
            expected_outer_snapshot = copy.copy(expected_input)
            actual_nested_snapshot = (
                copy.copy(actual_input.get("factory_kwargs"))
                if isinstance(actual_input, dict)
                else None
            )
            expected_nested_snapshot = (
                copy.copy(expected_input.get("factory_kwargs"))
                if isinstance(expected_input, dict)
                else None
            )

            with self.subTest(case=case):
                actual = nn.factory_kwargs(actual_input)
                expected = reference_nn.factory_kwargs(expected_input)

                self.assertIs(type(actual), dict)
                self.assertIs(type(expected), dict)
                self.assertEqual(actual, expected)
                self.assertEqual(list(actual), list(expected))
                if actual_input is not None:
                    self.assertIsNot(actual, actual_input)
                    self.assertEqual(actual_input, actual_outer_snapshot)
                    self.assertEqual(expected_input, expected_outer_snapshot)
                if actual_nested_snapshot is not None:
                    self.assertEqual(
                        actual_input["factory_kwargs"], actual_nested_snapshot
                    )
                    self.assertEqual(
                        expected_input["factory_kwargs"], expected_nested_snapshot
                    )

        actual_first = nn.factory_kwargs(None)
        actual_second = nn.factory_kwargs(None)
        expected_first = reference_nn.factory_kwargs(None)
        expected_second = reference_nn.factory_kwargs(None)
        self.assertIsNot(actual_first, actual_second)
        self.assertIsNot(expected_first, expected_second)

        mutable_value = []
        actual_nested = {"value": mutable_value}
        expected_nested = {"value": mutable_value}
        actual = nn.factory_kwargs({"factory_kwargs": actual_nested})
        expected = reference_nn.factory_kwargs({"factory_kwargs": expected_nested})
        self.assertIs(actual["value"], mutable_value)
        self.assertIs(expected["value"], mutable_value)
        self.assertIsNot(actual, actual_nested)
        self.assertIsNot(expected, expected_nested)

        actual_user_dict = UserDict(device=device, dtype=dtype)
        expected_user_dict = UserDict(device=device, dtype=dtype)
        actual = nn.factory_kwargs(actual_user_dict)
        expected = reference_nn.factory_kwargs(expected_user_dict)
        self.assertEqual(actual, expected)
        self.assertEqual(list(actual), list(expected))
        self.assertIs(type(actual), dict)
        self.assertIs(type(expected), dict)

    def test_errors_and_call_validation_match(self):
        error_cases = (
            (
                lambda: nn.factory_kwargs({"unexpected": 1}),
                lambda: reference_nn.factory_kwargs({"unexpected": 1}),
            ),
            (
                lambda: nn.factory_kwargs({"alpha": 1, "beta": 2}),
                lambda: reference_nn.factory_kwargs({"alpha": 1, "beta": 2}),
            ),
            (
                lambda: nn.factory_kwargs(
                    {"device": "outer", "factory_kwargs": {"device": "nested"}}
                ),
                lambda: reference_nn.factory_kwargs(
                    {"device": "outer", "factory_kwargs": {"device": "nested"}}
                ),
            ),
            (
                lambda: nn.factory_kwargs(
                    {"dtype": "outer", "factory_kwargs": {"dtype": "nested"}}
                ),
                lambda: reference_nn.factory_kwargs(
                    {"dtype": "outer", "factory_kwargs": {"dtype": "nested"}}
                ),
            ),
            (
                lambda: nn.factory_kwargs(
                    {
                        "memory_format": "outer",
                        "factory_kwargs": {"memory_format": "nested"},
                    }
                ),
                lambda: reference_nn.factory_kwargs(
                    {
                        "memory_format": "outer",
                        "factory_kwargs": {"memory_format": "nested"},
                    }
                ),
            ),
            (
                lambda: nn.factory_kwargs({"factory_kwargs": None}),
                lambda: reference_nn.factory_kwargs({"factory_kwargs": None}),
            ),
            (
                lambda: nn.factory_kwargs({"factory_kwargs": 1}),
                lambda: reference_nn.factory_kwargs({"factory_kwargs": 1}),
            ),
            (lambda: nn.factory_kwargs(0), lambda: reference_nn.factory_kwargs(0)),
            (lambda: nn.factory_kwargs([]), lambda: reference_nn.factory_kwargs([])),
            (
                lambda: nn.factory_kwargs(_ListKeysMapping()),
                lambda: reference_nn.factory_kwargs(_ListKeysMapping()),
            ),
            (lambda: nn.factory_kwargs(), lambda: reference_nn.factory_kwargs()),
            (
                lambda: nn.factory_kwargs({}, {}),
                lambda: reference_nn.factory_kwargs({}, {}),
            ),
            (
                lambda: nn.factory_kwargs(options={}),
                lambda: reference_nn.factory_kwargs(options={}),
            ),
            (
                lambda: nn.factory_kwargs({}, kwargs={}),
                lambda: reference_nn.factory_kwargs({}, kwargs={}),
            ),
            (
                lambda: nn.factory_kwargs(kwargs={}, unexpected=True),
                lambda: reference_nn.factory_kwargs(
                    kwargs={}, unexpected=True
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(error_cases):
            with self.subTest(case=case):
                self.assert_errors_match(actual_call, expected_call)

    def test_imports_metadata_documentation_copy_and_pickle_match(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        reference_imported_nn = importlib.import_module("torch.nn")
        from torch_rs import nn as from_torch
        from torch_rs.nn import factory_kwargs

        actual = nn.factory_kwargs
        expected = reference_nn.factory_kwargs
        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(from_torch, nn)
        self.assertIs(factory_kwargs, actual)
        self.assertIs(reference_torch.nn, reference_nn)
        self.assertIs(reference_nn, reference_imported_nn)
        self.assertFalse(hasattr(torch, "factory_kwargs"))
        self.assertFalse(hasattr(reference_torch, "factory_kwargs"))
        self.assertFalse(hasattr(nn, "__all__"))
        self.assertFalse(hasattr(reference_nn, "__all__"))
        self.assertEqual(nn.__doc__, reference_nn.__doc__)

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.nn import *", actual_wildcard)
        exec("from torch.nn import *", expected_wildcard)
        self.assertIs(actual_wildcard["factory_kwargs"], actual)
        self.assertIs(expected_wildcard["factory_kwargs"], expected)

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertIs(inspect.getmodule(actual), nn)
        self.assertIs(inspect.getmodule(expected), reference_nn)

        for operation in (copy.copy, copy.deepcopy):
            with self.subTest(operation=operation.__name__):
                self.assertIs(operation(actual), actual)
                self.assertIs(operation(expected), expected)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol)), expected
                )

    def test_mapping_access_and_failure_order_match(self):
        actual_nested = _TrackingNestedMapping(
            OrderedDict((("pin_memory", True), ("requires_grad", False)))
        )
        expected_nested = _TrackingNestedMapping(
            OrderedDict((("pin_memory", True), ("requires_grad", False)))
        )
        actual_outer = _TrackingOuterMapping(
            {
                "device": "cpu",
                "dtype": "float32",
                "memory_format": "contiguous",
                "factory_kwargs": actual_nested,
            }
        )
        expected_outer = _TrackingOuterMapping(
            {
                "device": "cpu",
                "dtype": "float32",
                "memory_format": "contiguous",
                "factory_kwargs": expected_nested,
            }
        )

        actual = nn.factory_kwargs(actual_outer)
        expected = reference_nn.factory_kwargs(expected_outer)

        self.assertEqual(actual, expected)
        self.assertEqual(list(actual), list(expected))
        self.assertEqual(actual_outer.events, expected_outer.events)
        self.assertEqual(actual_nested.events, expected_nested.events)

        actual_unexpected = _TrackingOuterMapping({"unexpected": 1})
        expected_unexpected = _TrackingOuterMapping({"unexpected": 1})
        self.assert_errors_match(
            lambda: nn.factory_kwargs(actual_unexpected),
            lambda: reference_nn.factory_kwargs(expected_unexpected),
        )
        self.assertEqual(actual_unexpected.events, expected_unexpected.events)
        self.assertEqual(actual_unexpected.events, [("keys",), ("keys",)])

        actual_duplicate = _TrackingOuterMapping(
            {"device": "outer", "factory_kwargs": {"device": "nested"}}
        )
        expected_duplicate = _TrackingOuterMapping(
            {"device": "outer", "factory_kwargs": {"device": "nested"}}
        )
        self.assert_errors_match(
            lambda: nn.factory_kwargs(actual_duplicate),
            lambda: reference_nn.factory_kwargs(expected_duplicate),
        )
        self.assertEqual(actual_duplicate.events, expected_duplicate.events)

    def test_reload_behavior_matches(self):
        def reload_outcome(module):
            old_function = module.factory_kwargs
            namespace = module.__dict__
            reloaded = importlib.reload(module)
            new_function = module.factory_kwargs

            try:
                pickle.dumps(old_function)
            except Exception as error:
                stale_pickle_error = (
                    type(error),
                    re.sub(
                        r"0x[0-9a-fA-F]+",
                        "0x...",
                        str(error).replace("torch_rs", "torch"),
                    ),
                )
            else:
                stale_pickle_error = None

            return (
                reloaded is module,
                module.__dict__ is namespace,
                new_function is not old_function,
                str(inspect.signature(new_function)),
                new_function.__annotations__,
                new_function.__doc__,
                module.__doc__,
                hasattr(module, "__all__"),
                new_function(None),
                new_function({"device": "cpu"}),
                stale_pickle_error,
                tuple(
                    pickle.loads(pickle.dumps(new_function, protocol))
                    is new_function
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ),
            )

        self.assertEqual(reload_outcome(nn), reload_outcome(reference_nn))
        self.assertIs(torch.nn, nn)
        self.assertIs(reference_torch.nn, reference_nn)


if __name__ == "__main__":
    unittest.main()
