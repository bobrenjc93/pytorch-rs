import copy
import importlib
import inspect
import pickle
import pickletools
import types
import typing
import unittest
import warnings

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _Base:
    pass


class _Child(_Base):
    pass


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitIsinstanceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.isinstance differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def warning_outcome(self, module, obj, target_type):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = module.jit.isinstance(obj, target_type)
        return (
            result,
            type(result).__name__,
            tuple(
                (warning.category.__name__, str(warning.message))
                for warning in caught
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
        return shape

    def test_plain_and_tuple_type_semantics_match_pytorch_2_13(self):
        cases = (
            (1, int),
            (True, int),
            (1, bool),
            ("value", str),
            (_Child(), _Base),
            ("value", (int, str)),
            (1.5, (int, str)),
            ([1, 2], (dict[str, int], list[int])),
            (1, ()),
        )
        for obj, target_type in cases:
            with self.subTest(obj=obj, target_type=target_type):
                self.assertEqual(
                    self.warning_outcome(torch, obj, target_type),
                    self.warning_outcome(reference_torch, obj, target_type),
                )

        actual_tensor = torch.tensor([1.0, 2.0])
        expected_tensor = reference_torch.tensor([1.0, 2.0])
        self.assertEqual(
            self.warning_outcome(torch, actual_tensor, torch.Tensor),
            self.warning_outcome(
                reference_torch, expected_tensor, reference_torch.Tensor
            ),
        )

    def test_parameterized_container_semantics_match_pytorch_2_13(self):
        cases = (
            ([1, 2], list[int]),
            ([1, 2], typing.List[int]),
            ([1, "two"], list[int]),
            ([[1], [2, 3]], list[list[int]]),
            ([[1], ["two"]], typing.List[typing.List[int]]),
            ({"one": 1}, dict[str, int]),
            ({"one": [1, 2]}, typing.Dict[str, typing.List[int]]),
            ({1: 1}, dict[str, int]),
            ({"one": "1"}, typing.Dict[str, int]),
            ((1, "two"), tuple[int, str]),
            ((1, "two"), typing.Tuple[int, str]),
            ((1,), tuple[int, str]),
            (("one", 2), typing.Tuple[int, str]),
            (([1], {"two": 2}), tuple[list[int], dict[str, int]]),
        )
        for obj, target_type in cases:
            with self.subTest(obj=obj, target_type=target_type):
                self.assertEqual(
                    self.warning_outcome(torch, obj, target_type),
                    self.warning_outcome(reference_torch, obj, target_type),
                )

    def test_optional_and_union_semantics_match_pytorch_2_13(self):
        cases = (
            (None, typing.Optional[int]),
            (1, typing.Optional[int]),
            ("one", typing.Optional[int]),
            ([1, 2], typing.Optional[list[int]]),
            ([1, "two"], typing.Optional[list[int]]),
            ((1, "two"), typing.Optional[tuple[int, str]]),
            (1, typing.Union[int, str]),
            ("one", typing.Union[int, str]),
            (1.5, typing.Union[int, str]),
            (1, int | str),
            ("one", int | str),
            (1.5, int | str),
            ([1, 2], typing.Union[str, list[int]]),
            ("one", typing.Union[list[int], str]),
            (None, typing.Union[int, str]),
        )
        for obj, target_type in cases:
            with self.subTest(obj=obj, target_type=target_type):
                self.assertEqual(
                    self.warning_outcome(torch, obj, target_type),
                    self.warning_outcome(reference_torch, obj, target_type),
                )

    def test_empty_container_warnings_match_pytorch_2_13(self):
        cases = (
            ([], list[int]),
            ([], list[float]),
            ([], typing.List[str]),
            ({}, dict[str, int]),
            ((), tuple[()]),
            ({}, list[int]),
            ([[], []], list[list[int]]),
            ({"left": [], "right": []}, dict[str, list[int]]),
            ({}, (list[int], dict[str, int])),
        )
        for obj, target_type in cases:
            with self.subTest(obj=obj, target_type=target_type):
                self.assertEqual(
                    self.warning_outcome(torch, obj, target_type),
                    self.warning_outcome(reference_torch, obj, target_type),
                )

    def test_raw_invalid_and_call_shape_errors_match_pytorch_2_13(self):
        for target_type in (
            list,
            dict,
            tuple,
            typing.List,
            typing.Dict,
            typing.Tuple,
            typing.Optional,
            "int",
            [int],
            {int},
            {"value": int},
            None,
            typing.Any,
            typing.Literal[1],
        ):
            with self.subTest(target_type=target_type):
                self.assert_error_matches(
                    lambda target_type=target_type: torch.jit.isinstance(
                        1, target_type
                    ),
                    lambda target_type=target_type: reference_torch.jit.isinstance(
                        1, target_type
                    ),
                )

        actual = torch.jit.isinstance
        expected = reference_torch.jit.isinstance
        calls = (
            lambda function: function(),
            lambda function: function(1),
            lambda function: function(target_type=int),
            lambda function: function(1, int, str),
            lambda function: function(object=1, target_type=int),
            lambda function: function(1, int, obj=2),
            lambda function: function(1, int, target_type=str),
        )
        for call in calls:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda call=call: call(actual),
                    lambda call=call: call(expected),
                )

    def test_signature_documentation_and_ownership_match_pytorch_2_13(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")
        actual = actual_jit.isinstance
        expected = expected_jit.isinstance

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_jit)
        self.assertIs(inspect.getmodule(expected), expected_jit)
        self.assertIs(actual.__globals__["_isinstance"], actual_internal._isinstance)
        self.assertIs(
            expected.__globals__["_isinstance"], expected_internal._isinstance
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual_jit.__doc__, expected_jit.__doc__)

        self.assertEqual(
            str(inspect.signature(actual_internal._isinstance)),
            str(inspect.signature(expected_internal._isinstance)),
        )
        self.assertEqual(
            actual_internal._isinstance.__annotations__,
            expected_internal._isinstance.__annotations__,
        )

    def test_exports_copying_and_pickling_match_supported_scope(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        actual = actual_jit.isinstance
        expected = expected_jit.isinstance
        wildcard_supported = {
            "Attribute",
            "annotate",
            "export",
            "ignore",
            "isinstance",
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
        self.assertEqual(
            torch.__all__.count("isinstance"),
            reference_torch.__all__.count("isinstance"),
        )

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            wildcard_supported,
        )
        self.assertIs(actual_namespace["isinstance"], actual)
        self.assertIs(expected_namespace["isinstance"], expected)

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
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

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("jit", namespace)
            self.assertNotIn("isinstance", namespace)
            self.assertFalse(hasattr(module, "isinstance"))

    def test_compilation_surface_remains_unsupported(self):
        for name in ("script", "trace", "trace_module"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(reference_torch.jit, name))
                self.assertFalse(hasattr(torch.jit, name))

        self.assertIs(torch.jit.is_scripting(), False)
        self.assertIs(torch.jit.is_tracing(), False)
        self.assertTrue(hasattr(reference_torch, "compile"))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
