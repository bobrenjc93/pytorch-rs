import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest
import warnings

import torch_rs as torch


FUNCTION_DOC = """
    Provide container type refinement in TorchScript.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    It can refine parameterized containers of the List, Dict, Tuple, and Optional types. E.g. ``List[str]``,
    ``Dict[str, List[torch.Tensor]]``, ``Optional[Tuple[int,str,int]]``. It can also
    refine basic types such as bools and ints that are available in TorchScript.

    Args:
        obj: object to refine the type of
        target_type: type to try to refine obj to
    Returns:
        ``bool``: True if obj was successfully refined to the type of target_type,
            False otherwise with no new type refinement


    Example (using ``torch.jit.isinstance`` for type refinement):
    .. testcode::

        import torch
        from typing import Any, Dict, List

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, input: Any): # note the Any type
                if torch.jit.isinstance(input, List[torch.Tensor]):
                    for t in input:
                        y = t.clamp(0, 0.5)
                elif torch.jit.isinstance(input, Dict[str, str]):
                    for val in input.values():
                        print(val)

        m = torch.jit.script(MyModule())
        x = [torch.rand(3,3), torch.rand(4,3)]
        m(x)
        y = {"key1":"val1","key2":"val2"}
        m(y)
    """

EMPTY_CONTAINER_WARNING = (
    "The inner type of a container is lost when calling "
    "torch.jit.isinstance in eager mode. For example, List[int] would become "
    "list and therefore falsely return True for List[float] or List[str]."
)


class _Base:
    pass


class _Child(_Base):
    pass


class JitIsinstanceTests(unittest.TestCase):
    def assert_warning_outcome(self, obj, target_type, expected, count=1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = torch.jit.isinstance(obj, target_type)

        self.assertIs(result, expected)
        self.assertEqual(len(caught), count)
        for warning in caught:
            self.assertIs(warning.category, UserWarning)
            self.assertEqual(str(warning.message), EMPTY_CONTAINER_WARNING)

    def test_plain_types_and_tuples_of_types(self):
        tensor = torch.tensor([1.0, 2.0])
        cases = (
            (1, int, True),
            (True, int, True),
            (1, bool, False),
            ("value", str, True),
            (_Child(), _Base, True),
            (tensor, torch.Tensor, True),
            ("value", (int, str), True),
            (1.5, (int, str), False),
            ([1, 2], (dict[str, int], list[int]), True),
            (1, (), False),
        )
        for obj, target_type, expected in cases:
            with self.subTest(obj=obj, target_type=target_type):
                result = torch.jit.isinstance(obj, target_type)
                self.assertIs(type(result), bool)
                self.assertIs(result, expected)

    def test_parameterized_lists_dicts_and_fixed_tuples(self):
        cases = (
            ([1, 2], list[int], True),
            ([1, 2], typing.List[int], True),
            ([1, "two"], list[int], False),
            ([[1], [2, 3]], list[list[int]], True),
            ([[1], ["two"]], typing.List[typing.List[int]], False),
            ({"one": 1}, dict[str, int], True),
            ({"one": [1, 2]}, typing.Dict[str, typing.List[int]], True),
            ({1: 1}, dict[str, int], False),
            ({"one": "1"}, typing.Dict[str, int], False),
            ((1, "two"), tuple[int, str], True),
            ((1, "two"), typing.Tuple[int, str], True),
            ((1,), tuple[int, str], False),
            (("one", 2), typing.Tuple[int, str], False),
            (([1], {"two": 2}), tuple[list[int], dict[str, int]], True),
        )
        for obj, target_type, expected in cases:
            with self.subTest(obj=obj, target_type=target_type):
                self.assertIs(torch.jit.isinstance(obj, target_type), expected)

    def test_optional_and_union_types(self):
        cases = (
            (None, typing.Optional[int], True),
            (1, typing.Optional[int], True),
            ("one", typing.Optional[int], False),
            ([1, 2], typing.Optional[list[int]], True),
            ([1, "two"], typing.Optional[list[int]], False),
            ((1, "two"), typing.Optional[tuple[int, str]], True),
            (1, typing.Union[int, str], True),
            ("one", typing.Union[int, str], True),
            (1.5, typing.Union[int, str], False),
            (1, int | str, True),
            ("one", int | str, True),
            (1.5, int | str, False),
            ([1, 2], typing.Union[str, list[int]], True),
            ("one", typing.Union[list[int], str], False),
            (None, typing.Union[int, str], True),
        )
        for obj, target_type, expected in cases:
            with self.subTest(obj=obj, target_type=target_type):
                self.assertIs(torch.jit.isinstance(obj, target_type), expected)

    def test_empty_containers_warn_and_retain_eager_container_semantics(self):
        for target_type in (list[int], list[float], typing.List[str]):
            with self.subTest(target_type=target_type):
                self.assert_warning_outcome([], target_type, True)

        self.assert_warning_outcome({}, dict[str, int], True)
        self.assert_warning_outcome((), tuple[()], True)
        self.assert_warning_outcome({}, list[int], False)
        self.assert_warning_outcome([[], []], list[list[int]], True, count=2)
        self.assert_warning_outcome(
            {"left": [], "right": []}, dict[str, list[int]], True, count=2
        )

    def test_raw_and_invalid_target_errors_match_pytorch_2_13(self):
        raw_cases = (
            (
                list,
                "Attempted to use list without a contained type. Please add a "
                "contained type, e.g. list[int]",
            ),
            (
                dict,
                "Attempted to use dict without contained types. Please add "
                "contained type, e.g. dict[int, int]",
            ),
            (
                tuple,
                "Attempted to use tuple without a contained type. Please add a "
                "contained type, e.g. tuple[int]",
            ),
            (
                typing.List,
                "Attempted to use List without a contained type. Please add a "
                "contained type, e.g. List[int]",
            ),
            (
                typing.Dict,
                "Attempted to use Dict without contained types. Please add "
                "contained type, e.g. Dict[int, int]",
            ),
            (
                typing.Tuple,
                "Attempted to use Tuple without a contained type. Please add a "
                "contained type, e.g. Tuple[int]",
            ),
            (
                typing.Optional,
                "Attempted to use Optional without a contained type. Please add a "
                "contained type, e.g. Optional[int]",
            ),
        )
        for target_type, message in raw_cases:
            with self.subTest(target_type=target_type):
                with self.assertRaises(RuntimeError) as raised:
                    torch.jit.isinstance([], target_type)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        invalid_container_message = (
            "The second argument to `torch.jit.isinstance` must be a type or a "
            "tuple of types"
        )
        for target_type in ("int", [int], {int}, {"value": int}):
            with self.subTest(target_type=target_type):
                with self.assertRaises(RuntimeError) as raised:
                    torch.jit.isinstance(1, target_type)
                self.assertEqual(str(raised.exception), invalid_container_message)
                self.assertEqual(
                    raised.exception.args, (invalid_container_message,)
                )

        with self.assertRaises(TypeError) as raised:
            torch.jit.isinstance(1, None)
        self.assertEqual(
            str(raised.exception),
            "isinstance() arg 2 must be a type, a tuple of types, or a union",
        )
        self.assertEqual(
            raised.exception.args,
            ("isinstance() arg 2 must be a type, a tuple of types, or a union",),
        )

    def test_signature_documentation_and_ownership(self):
        jit = importlib.import_module("torch_rs.jit")
        internal = importlib.import_module("torch_rs._jit_internal")
        function = jit.isinstance

        self.assertIs(torch.jit, jit)
        self.assertIs(torch._jit_internal, internal)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(obj, target_type)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "isinstance")
        self.assertEqual(function.__qualname__, "isinstance")
        self.assertEqual(function.__module__, "torch_rs.jit")
        self.assertIs(inspect.getmodule(function), jit)
        self.assertIs(function.__globals__["_isinstance"], internal._isinstance)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertIsNone(jit.__doc__)

        self.assertIs(function(obj=1, target_type=int), True)

    def test_exports_copying_and_pickling_use_the_canonical_module(self):
        jit = torch.jit
        function = jit.isinstance
        supported = {
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
            jit.__all__,
            [
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "onednn_fusion_enabled",
                "script_if_tracing",
                "strict_fusion",
                "unused",
            ],
        )
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {*supported, "is_scripting", "is_tracing"},
        )
        jit_namespace = {}
        exec("from torch_rs.jit import *", jit_namespace)
        self.assertEqual(
            {name for name in jit_namespace if not name.startswith("__")},
            supported,
        )
        self.assertIs(jit_namespace["isinstance"], function)

        self.assertNotIn("jit", torch.__all__)
        self.assertNotIn("isinstance", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("jit", top_level_namespace)
        self.assertNotIn("isinstance", top_level_namespace)
        self.assertFalse(hasattr(torch, "isinstance"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.jit", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_invalid_call_shapes_use_python_function_errors(self):
        function = torch.jit.isinstance
        cases = (
            (
                lambda: function(),
                "isinstance() missing 2 required positional arguments: 'obj' and "
                "'target_type'",
            ),
            (
                lambda: function(1),
                "isinstance() missing 1 required positional argument: "
                "'target_type'",
            ),
            (
                lambda: function(target_type=int),
                "isinstance() missing 1 required positional argument: 'obj'",
            ),
            (
                lambda: function(1, int, str),
                "isinstance() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: function(object=1, target_type=int),
                "isinstance() got an unexpected keyword argument 'object'",
            ),
            (
                lambda: function(1, int, obj=2),
                "isinstance() got multiple values for argument 'obj'",
            ),
            (
                lambda: function(1, int, target_type=str),
                "isinstance() got multiple values for argument 'target_type'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_scripting_tracing_and_compilation_remain_unsupported(self):
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "script",
            "script_method",
            "trace",
            "trace_module",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.jit, name))
        self.assertIs(torch.jit.is_scripting(), False)
        self.assertIs(torch.jit.is_tracing(), False)
        self.assertFalse(hasattr(torch, "compile"))

    def test_import_and_eager_checks_do_not_import_pytorch(self):
        script = r"""
import sys
import typing
import warnings

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

assert torch.jit.isinstance([1, 2], list[int]) is True
assert torch.jit.isinstance({"value": [1]}, typing.Dict[str, typing.List[int]]) is True
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    assert torch.jit.isinstance([], list[str]) is True
assert len(caught) == 1
assert not hasattr(torch.jit, "script")
assert not hasattr(torch.jit, "trace")
assert not hasattr(torch, "compile")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
