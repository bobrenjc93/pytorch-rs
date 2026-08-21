import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch


FUNCTION_DOC = """Use to give type of `the_value` in TorchScript compiler.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    This method is a pass-through function that returns `the_value`, used to hint TorchScript
    compiler the type of `the_value`. It is a no-op when running outside of TorchScript.

    Though TorchScript can infer correct type for most Python expressions, there are some cases where
    type inference can be wrong, including:

    - Empty containers like `[]` and `{}`, which TorchScript assumes to be container of `Tensor`
    - Optional types like `Optional[T]` but assigned a valid value of type `T`, TorchScript would assume
      it is type `T` rather than `Optional[T]`

    Note that `annotate()` does not help in `__init__` method of `torch.nn.Module` subclasses because it
    is executed in eager mode. To annotate types of `torch.nn.Module` attributes,
    use :meth:`~torch.jit.Attribute` instead.

    Example:

    .. testcode::

        import torch
        from typing import Dict

        @torch.jit.script
        def fn():
            # Telling TorchScript that this empty dictionary is a (str -> int) dictionary
            # instead of default dictionary type of (str -> Tensor).
            d = torch.jit.annotate(Dict[str, int], {})

            # Without `torch.jit.annotate` above, following statement would fail because of
            # type mismatch.
            d["name"] = 20

    .. testcleanup::

        del fn

    Args:
        the_type: Python type that should be passed to TorchScript compiler as type hint for `the_value`
        the_value: Value or expression to hint type for.

    Returns:
        `the_value` is passed back as return value.
    """


class ExplodingTypeHint:
    def __getattribute__(self, name):
        raise AssertionError(f"type hint attribute was accessed: {name}")


class JitAnnotateTests(unittest.TestCase):
    def test_tensors_and_views_are_exact_identities(self):
        base = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        transposed = base.transpose(0, 1)
        offset = transposed[1]

        for name, value, type_hint in (
            ("base", base, torch.Tensor),
            ("transposed view", transposed, list[int]),
            ("offset view", offset, str),
        ):
            with self.subTest(name=name):
                metadata = (
                    value.shape,
                    value.stride(),
                    value.storage_offset(),
                    value.data_ptr(),
                    value.requires_grad,
                    value.is_leaf,
                )
                result = torch.jit.annotate(type_hint, value)
                self.assertIs(result, value)
                self.assertEqual(
                    (
                        result.shape,
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr(),
                        result.requires_grad,
                        result.is_leaf,
                    ),
                    metadata,
                )

    def test_leaf_nonleaf_and_existing_gradient_state_are_unchanged(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        self.assertIs(torch.jit.annotate(dict[str, int], leaf), leaf)

        nonleaf = (leaf * 3.0).transpose(0, 1)[1]
        result = torch.jit.annotate(None, nonleaf)
        self.assertIs(result, nonleaf)
        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_leaf)
        result.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])

        gradient = leaf.grad
        self.assertIs(torch.jit.annotate(int, gradient), gradient)
        self.assertIs(leaf.grad, gradient)

    def test_containers_and_arbitrary_values_are_not_copied_or_inspected(self):
        custom_value = object()
        values = (
            [],
            {},
            set(),
            ([], {"nested": [1, 2]}),
            custom_value,
            None,
        )
        type_hints = (
            list[int],
            dict[str, int],
            tuple[str, ...],
            torch.Tensor,
            ExplodingTypeHint(),
            "not a type",
        )

        for type_hint, value in zip(type_hints, values, strict=True):
            with self.subTest(value_type=type(value).__name__):
                self.assertIs(torch.jit.annotate(type_hint, value), value)

        nested = values[3]
        result = torch.jit.annotate(float, nested)
        result[0].append(3)
        self.assertEqual(nested, ([3], {"nested": [1, 2]}))

    def test_signature_documentation_and_module_identity(self):
        jit = importlib.import_module("torch_rs.jit")
        function = jit.annotate

        self.assertIs(torch.jit, jit)
        self.assertIs(sys.modules["torch_rs.jit"], jit)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(the_type, the_value)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "annotate")
        self.assertEqual(function.__qualname__, "annotate")
        self.assertEqual(function.__module__, "torch_rs.jit")
        self.assertIs(inspect.getmodule(function), jit)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertIsNone(jit.__doc__)

        keyword_value = {"items": [1, 2]}
        self.assertIs(
            function(the_type=int, the_value=keyword_value), keyword_value
        )

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        jit = torch.jit
        function = jit.annotate

        self.assertEqual(
            jit.__all__,
            [
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "script_if_tracing",
                "unused",
            ],
        )
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {
                "Attribute",
                "Final",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "is_scripting",
                "is_tracing",
                "script_if_tracing",
                "unused",
            },
        )
        jit_namespace = {}
        exec("from torch_rs.jit import *", jit_namespace)
        self.assertEqual(
            {name for name in jit_namespace if not name.startswith("__")},
            {
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "script_if_tracing",
                "unused",
            },
        )
        self.assertIs(jit_namespace["annotate"], function)

        self.assertNotIn("jit", torch.__all__)
        self.assertNotIn("annotate", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("jit", top_level_namespace)
        self.assertNotIn("annotate", top_level_namespace)

        explicit_namespace = {}
        exec("from torch_rs import jit", explicit_namespace)
        self.assertIs(explicit_namespace["jit"], jit)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.jit", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_invalid_calls_with_pytorch_2_13_errors(self):
        function = torch.jit.annotate
        cases = (
            (
                lambda: function(),
                "annotate() missing 2 required positional arguments: "
                "'the_type' and 'the_value'",
            ),
            (
                lambda: function(int),
                "annotate() missing 1 required positional argument: 'the_value'",
            ),
            (
                lambda: function(the_value=1),
                "annotate() missing 1 required positional argument: 'the_type'",
            ),
            (
                lambda: function(int, 1, 2),
                "annotate() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: function(type=int, the_value=1),
                "annotate() got an unexpected keyword argument 'type'",
            ),
            (
                lambda: function(int, 1, the_type=str),
                "annotate() got multiple values for argument 'the_type'",
            ),
            (
                lambda: function(int, 1, the_value=2),
                "annotate() got multiple values for argument 'the_value'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_scripting_tracing_and_compilation_remain_unsupported(self):
        unsupported_jit_names = (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "script",
            "script_method",
            "trace",
            "trace_module",
        )
        for name in unsupported_jit_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.jit, name))
        self.assertFalse(hasattr(torch, "compile"))

    def test_importing_the_package_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

value = {"items": [1, 2]}
assert torch.jit.annotate(list[int], value) is value
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
