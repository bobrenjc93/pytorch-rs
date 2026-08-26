import copy
import importlib
import inspect
import pickle
import types
import typing
import unittest
from collections import OrderedDict

import torch_rs as torch

from torch_rs.nn.modules.utils import consume_prefix_in_state_dict_if_present


class StateDict(dict):
    pass


class ConsumePrefixInStateDictTests(unittest.TestCase):
    def test_dict_and_ordered_dict_are_renamed_in_place(self):
        for mapping_type in (dict, OrderedDict):
            values = [object() for _ in range(4)]
            state_dict = mapping_type(
                [
                    ("module.encoder.weight", values[0]),
                    ("unrelated", values[1]),
                    ("module.encoder.bias", values[2]),
                    ("encoder.weight", values[3]),
                ]
            )

            with self.subTest(mapping_type=mapping_type.__name__):
                self.assertIsNone(
                    consume_prefix_in_state_dict_if_present(state_dict, "module.")
                )
                self.assertIs(type(state_dict), mapping_type)
                self.assertEqual(
                    list(state_dict),
                    ["unrelated", "encoder.weight", "encoder.bias"],
                )
                self.assertIs(state_dict["unrelated"], values[1])
                self.assertIs(state_dict["encoder.weight"], values[0])
                self.assertIs(state_dict["encoder.bias"], values[2])
                self.assertNotIn(values[3], state_dict.values())

    def test_metadata_is_renamed_in_place_for_dict_and_ordered_dict(self):
        for mapping_type in (StateDict, OrderedDict):
            state_values = [object(), object()]
            metadata_values = [object() for _ in range(5)]
            state_dict = mapping_type(
                [
                    ("module.value", state_values[0]),
                    ("plain", state_values[1]),
                ]
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

            with self.subTest(mapping_type=mapping_type.__name__):
                self.assertIsNone(
                    consume_prefix_in_state_dict_if_present(state_dict, "module.")
                )
                self.assertEqual(list(state_dict), ["plain", "value"])
                self.assertIs(state_dict["plain"], state_values[1])
                self.assertIs(state_dict["value"], state_values[0])

                self.assertIs(state_dict._metadata, metadata)
                self.assertEqual(
                    list(metadata), ["", "other", "modulex", "layer"]
                )
                self.assertIs(metadata[""], metadata_values[1])
                self.assertIs(metadata["other"], metadata_values[3])
                self.assertIs(metadata["modulex"], metadata_values[4])
                self.assertIs(metadata["layer"], metadata_values[2])
                self.assertNotIn(metadata_values[0], metadata.values())

    def test_no_match_and_empty_prefix_preserve_contents(self):
        for mapping_type in (StateDict, OrderedDict):
            values = [object(), object()]
            metadata_values = [object(), object()]
            state_dict = mapping_type([("weight", values[0]), ("bias", values[1])])
            state_dict._metadata = OrderedDict(
                [("", metadata_values[0]), ("layer", metadata_values[1])]
            )
            state_items = list(state_dict.items())
            metadata_items = list(state_dict._metadata.items())

            with self.subTest(mapping_type=mapping_type.__name__, prefix="missing."):
                self.assertIsNone(
                    consume_prefix_in_state_dict_if_present(state_dict, "missing.")
                )
                self.assertEqual(list(state_dict.items()), state_items)
                self.assertEqual(list(state_dict._metadata.items()), metadata_items)
                for (_, actual), (_, expected) in zip(
                    state_dict.items(), state_items, strict=True
                ):
                    self.assertIs(actual, expected)
                for (_, actual), (_, expected) in zip(
                    state_dict._metadata.items(), metadata_items, strict=True
                ):
                    self.assertIs(actual, expected)

            with self.subTest(mapping_type=mapping_type.__name__, prefix=""):
                self.assertIsNone(
                    consume_prefix_in_state_dict_if_present(state_dict, "")
                )
                self.assertEqual(list(state_dict.items()), state_items)
                self.assertEqual(list(state_dict._metadata.items()), metadata_items)

        self.assertIsNone(consume_prefix_in_state_dict_if_present({}, None))

    def test_errors_match_python_operations_and_keep_partial_mutations(self):
        with self.assertRaisesRegex(
            AttributeError, "^'NoneType' object has no attribute 'keys'$"
        ):
            consume_prefix_in_state_dict_if_present(None, "module.")

        state_dict = {"value": object()}
        with self.assertRaisesRegex(
            TypeError,
            "^startswith first arg must be str or a tuple of str, not NoneType$",
        ):
            consume_prefix_in_state_dict_if_present(state_dict, None)
        self.assertEqual(list(state_dict), ["value"])

        first = object()
        invalid = object()
        untouched = object()
        state_dict = OrderedDict(
            [("module.first", first), (7, invalid), ("module.last", untouched)]
        )
        with self.assertRaisesRegex(
            AttributeError, "^'int' object has no attribute 'startswith'$"
        ):
            consume_prefix_in_state_dict_if_present(state_dict, "module.")
        self.assertEqual(list(state_dict), [7, "module.last", "first"])
        self.assertIs(state_dict["first"], first)
        self.assertIs(state_dict[7], invalid)
        self.assertIs(state_dict["module.last"], untouched)

        state_dict = StateDict([("module.value", first)])
        state_dict._metadata = OrderedDict(
            [("module.layer", invalid), (None, untouched)]
        )
        with self.assertRaisesRegex(TypeError, "object of type 'NoneType' has no len"):
            consume_prefix_in_state_dict_if_present(state_dict, "module.")
        self.assertEqual(list(state_dict), ["value"])
        self.assertEqual(list(state_dict._metadata), [None, "layer"])
        self.assertIs(state_dict._metadata["layer"], invalid)
        self.assertIs(state_dict._metadata[None], untouched)

    def test_signature_documentation_and_import_identity(self):
        nn_module = importlib.import_module("torch_rs.nn")
        modules_module = importlib.import_module("torch_rs.nn.modules")
        utils_module = importlib.import_module("torch_rs.nn.modules.utils")
        function = consume_prefix_in_state_dict_if_present

        self.assertIs(torch.nn, nn_module)
        self.assertIs(nn_module.modules, modules_module)
        self.assertIs(modules_module.utils, utils_module)
        self.assertIs(utils_module.consume_prefix_in_state_dict_if_present, function)

        direct_import = {}
        exec(
            "from torch_rs.nn.modules.utils import "
            "consume_prefix_in_state_dict_if_present",
            direct_import,
        )
        self.assertIs(direct_import[function.__name__], function)

        wildcard_import = {}
        exec("from torch_rs.nn.modules.utils import *", wildcard_import)
        self.assertEqual(
            {name for name in wildcard_import if not name.startswith("__")},
            {"consume_prefix_in_state_dict_if_present"},
        )
        self.assertIs(wildcard_import[function.__name__], function)
        self.assertEqual(utils_module.__all__, [function.__name__])

        modules_wildcard = {}
        exec("from torch_rs.nn.modules import *", modules_wildcard)
        self.assertNotIn(function.__name__, modules_wildcard)
        self.assertNotIn("utils", modules_wildcard)
        self.assertEqual(modules_module.__all__, [])

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__module__, "torch_rs.nn.modules.utils")
        self.assertIs(inspect.getmodule(function), utils_module)
        self.assertEqual(function.__name__, "consume_prefix_in_state_dict_if_present")
        self.assertEqual(function.__qualname__, function.__name__)
        self.assertEqual(
            str(inspect.signature(function)),
            "(state_dict: dict[str, typing.Any], prefix: str) -> None",
        )
        self.assertEqual(
            function.__annotations__,
            {"state_dict": dict[str, typing.Any], "prefix": str, "return": None},
        )
        self.assertEqual(
            typing.get_type_hints(function),
            {
                "state_dict": dict[str, typing.Any],
                "prefix": str,
                "return": type(None),
            },
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertIn("Strip the prefix in state_dict in place", function.__doc__)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)

    def test_argument_errors_and_unsupported_serialization_scope(self):
        function = consume_prefix_in_state_dict_if_present
        cases = (
            (
                lambda: function(),
                "consume_prefix_in_state_dict_if_present() missing 2 required "
                "positional arguments: 'state_dict' and 'prefix'",
            ),
            (
                lambda: function({}),
                "consume_prefix_in_state_dict_if_present() missing 1 required "
                "positional argument: 'prefix'",
            ),
            (
                lambda: function({}, "module.", None),
                "consume_prefix_in_state_dict_if_present() takes 2 positional "
                "arguments but 3 were given",
            ),
            (
                lambda: function(mapping={}, prefix="module."),
                "consume_prefix_in_state_dict_if_present() got an unexpected "
                "keyword argument 'mapping'",
            ),
            (
                lambda: function({}, "module.", prefix="other."),
                "consume_prefix_in_state_dict_if_present() got multiple values "
                "for argument 'prefix'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        self.assertFalse(hasattr(torch.nn, "Module"))
        self.assertFalse(hasattr(torch.nn.modules, "Module"))
        self.assertFalse(hasattr(torch, "save"))
        self.assertFalse(hasattr(torch, "load"))
        self.assertFalse(hasattr(torch.serialization, "save"))
        self.assertFalse(hasattr(torch.serialization, "load"))


if __name__ == "__main__":
    unittest.main()
