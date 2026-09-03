import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import unittest

import torch_rs as torch


ATTRIBUTE_DOC = """
    This method is a pass-through function that returns `value`, mostly
    used to indicate to the TorchScript compiler that the left-hand side
    expression is a class instance attribute with type of `type`. Note that
    `torch.jit.Attribute` should only be used in `__init__` method of `jit.ScriptModule`
    subclasses.

    Though TorchScript can infer correct type for most Python expressions, there are some cases where
    type inference can be wrong, including:

    - Empty containers like `[]` and `{}`, which TorchScript assumes to be container of `Tensor`
    - Optional types like `Optional[T]` but assigned a valid value of type `T`, TorchScript would assume
      it is type `T` rather than `Optional[T]`

    In eager mode, it is simply a pass-through function that returns `value`
    without other implications.

    Example:

    .. testcode::

        import torch
        from typing import Dict

        class AttributeModule(torch.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.jit.Attribute(0.1, float)

                # we should be able to use self.foo as a float here
                assert 0.0 < self.foo

                self.names_ages = torch.jit.Attribute({}, Dict[str, int])
                self.names_ages["someone"] = 20
                assert isinstance(self.names_ages["someone"], int)

        m = AttributeModule()
        # m will contain two attributes
        # 1. foo of type float
        # 2. names_ages of type Dict[str, int]

    .. testcleanup::

        del AttributeModule
        del m

    Note: it's now preferred to instead use type annotations instead of `torch.jit.Attribute`:

    .. testcode::

        import torch
        from typing import Dict

        class AttributeModule(torch.nn.Module):
            names: Dict[str, int]

            def __init__(self) -> None:
                super().__init__()
                self.names = {}

        m = AttributeModule()

    .. testcleanup::

        del AttributeModule
        del m

    Args:
        value: An initial value to be assigned to attribute.
        type: A Python type

    Returns:
        Returns `value`
"""

SCRIPT_MODULE_DOC = """TorchScript.

This module contains functionality to support the JIT's scripting frontend, notably:
    - torch.jit.script

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""


class PickleMarker:
    pass


class ExplodingMarker:
    def __getattribute__(self, name):
        raise AssertionError(f"type marker attribute was accessed: {name}")


class JitAttributeTests(unittest.TestCase):
    def test_construction_preserves_exact_objects_and_tuple_behavior(self):
        attribute_type = torch.jit.Attribute
        value = {"items": [1, 2]}
        type_marker = ExplodingMarker()
        attribute = attribute_type(value, type_marker)

        self.assertIs(type(attribute), attribute_type)
        self.assertIsInstance(attribute, tuple)
        self.assertEqual(len(attribute), 2)
        self.assertIs(attribute.value, value)
        self.assertIs(attribute.type, type_marker)
        self.assertIs(attribute[0], value)
        self.assertIs(attribute[1], type_marker)
        self.assertEqual(attribute[:1], (value,))
        unpacked_value, unpacked_type = attribute
        self.assertIs(unpacked_value, value)
        self.assertIs(unpacked_type, type_marker)
        self.assertEqual(attribute.count(value), 1)
        self.assertEqual(attribute.index(type_marker), 1)

        as_dict = attribute._asdict()
        self.assertEqual(tuple(as_dict), ("value", "type"))
        self.assertIs(as_dict["value"], value)
        self.assertIs(as_dict["type"], type_marker)
        self.assertEqual(attribute._make((1, str)), attribute_type(1, str))
        self.assertEqual(attribute_type(1, str)._replace(value=2), (2, str))
        self.assertEqual(attribute_type(1, str).__getnewargs__(), (1, str))
        self.assertEqual(hash(attribute_type(1, str)), hash((1, str)))

    def test_repr_signature_documentation_and_module_ownership(self):
        jit = importlib.import_module("torch_rs.jit")
        script = importlib.import_module("torch_rs.jit._script")
        attribute_type = jit.Attribute

        self.assertIs(torch.jit, jit)
        self.assertIs(jit._script, script)
        self.assertIs(attribute_type, script.Attribute)
        self.assertIs(sys.modules["torch_rs.jit._script"], script)
        self.assertIs(type(attribute_type), type)
        self.assertEqual(attribute_type.__bases__, (tuple,))
        self.assertEqual(attribute_type.__name__, "Attribute")
        self.assertEqual(attribute_type.__qualname__, "Attribute")
        self.assertEqual(attribute_type.__module__, "torch_rs.jit._script")
        self.assertIs(inspect.getmodule(attribute_type), script)
        self.assertEqual(str(inspect.signature(attribute_type)), "(value, type)")
        self.assertEqual(
            str(inspect.signature(attribute_type.__new__)),
            "(_cls, value, type)",
        )
        self.assertEqual(attribute_type.__annotations__, {})
        self.assertEqual(attribute_type.__slots__, ())
        self.assertEqual(attribute_type._fields, ("value", "type"))
        self.assertEqual(attribute_type._field_defaults, {})
        self.assertEqual(attribute_type.__match_args__, ("value", "type"))
        self.assertEqual(attribute_type.__doc__, ATTRIBUTE_DOC)
        self.assertEqual(script.__doc__, SCRIPT_MODULE_DOC)
        self.assertEqual(
            repr(attribute_type(value=3, type=int)),
            "Attribute(value=3, type=<class 'int'>)",
        )
        self.assertEqual(str(attribute_type("x", list)), repr(attribute_type("x", list)))

    def test_exports_copying_and_pickling_use_the_canonical_class(self):
        jit = torch.jit
        script = jit._script
        attribute_type = jit.Attribute
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
            {
                *supported,
                "is_scripting",
                "is_tracing",
                "optimized_execution",
            },
        )
        self.assertFalse(hasattr(script, "__all__"))
        self.assertEqual(
            {name for name in vars(script) if not name.startswith("_")},
            {"Attribute", "collections"},
        )

        direct_namespace = {}
        exec("from torch_rs.jit._script import Attribute", direct_namespace)
        self.assertIs(direct_namespace["Attribute"], attribute_type)
        wildcard_namespace = {}
        exec("from torch_rs.jit import *", wildcard_namespace)
        self.assertEqual(
            {name for name in wildcard_namespace if not name.startswith("__")},
            supported,
        )
        self.assertIs(wildcard_namespace["Attribute"], attribute_type)

        self.assertNotIn("Attribute", torch.__all__)
        self.assertFalse(hasattr(torch, "Attribute"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("Attribute", top_level_namespace)

        self.assertIs(copy.copy(attribute_type), attribute_type)
        self.assertIs(copy.deepcopy(attribute_type), attribute_type)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="class", protocol=protocol):
                payload = pickle.dumps(attribute_type, protocol=protocol)
                self.assertIn(b"torch_rs.jit._script", payload)
                self.assertIs(pickle.loads(payload), attribute_type)

        value = {"items": [1, 2]}
        attribute = attribute_type(value, PickleMarker)
        shallow = copy.copy(attribute)
        self.assertIsNot(shallow, attribute)
        self.assertEqual(shallow, attribute)
        self.assertIs(shallow.value, value)
        self.assertIs(shallow.type, PickleMarker)

        deep = copy.deepcopy(attribute)
        self.assertIsNot(deep, attribute)
        self.assertEqual(deep, attribute)
        self.assertIsNot(deep.value, value)
        self.assertIs(deep.type, PickleMarker)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="instance", protocol=protocol):
                payload = pickle.dumps(attribute, protocol=protocol)
                self.assertIn(b"torch_rs.jit._script", payload)
                restored = pickle.loads(payload)
                self.assertIs(type(restored), attribute_type)
                self.assertEqual(restored.value, value)
                self.assertIsNot(restored.value, value)
                self.assertIs(restored.type, PickleMarker)

    def test_argument_errors_match_pytorch_2_13(self):
        attribute_type = torch.jit.Attribute
        cases = (
            (
                lambda: attribute_type(),
                "Attribute.__new__() missing 2 required positional arguments: "
                "'value' and 'type'",
            ),
            (
                lambda: attribute_type(1),
                "Attribute.__new__() missing 1 required positional argument: 'type'",
            ),
            (
                lambda: attribute_type(value=1),
                "Attribute.__new__() missing 1 required positional argument: 'type'",
            ),
            (
                lambda: attribute_type(type=int),
                "Attribute.__new__() missing 1 required positional argument: 'value'",
            ),
            (
                lambda: attribute_type(1, int, 3),
                "Attribute.__new__() takes 3 positional arguments but 4 were given",
            ),
            (
                lambda: attribute_type(1, int, other=3),
                "Attribute.__new__() got an unexpected keyword argument 'other'",
            ),
            (
                lambda: attribute_type(1, int, value=2),
                "Attribute.__new__() got multiple values for argument 'value'",
            ),
            (
                lambda: attribute_type(1, int, type=str),
                "Attribute.__new__() got multiple values for argument 'type'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_script_modules_interfaces_and_compilation_remain_unsupported(self):
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
                self.assertFalse(hasattr(torch.jit, name))
        self.assertTrue(callable(torch.compile))

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
marker = object()
attribute = torch.jit.Attribute(value, marker)
assert torch.jit.Attribute is torch.jit._script.Attribute
assert attribute.value is value
assert attribute.type is marker
assert not hasattr(torch.jit, "ScriptModule")
assert not hasattr(torch.jit, "script")
assert not hasattr(torch.jit, "interface")
assert callable(torch.compile)
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
