import importlib.util
import json
import subprocess
import sys
import unittest


REFERENCE_AVAILABLE = importlib.util.find_spec("torch") is not None

CONTRACT_SCRIPT = r"""
import copy
import importlib
import inspect
import json
import pickle
import pickletools
import sys
import types

import numpy as np

module = importlib.import_module(sys.argv[1])
setter_name = sys.argv[2]
if module.__name__ == "torch" and module.__version__.split("+")[0] != "2.13.0":
    raise AssertionError("thread setter differentials require pinned PyTorch 2.13.0")

function = getattr(module, setter_name)
getter = getattr(module, setter_name.replace("set_", "get_"))


class IndexOnly:
    def __index__(self):
        raise AssertionError("thread setters must not call __index__")


def outcome(call):
    try:
        result = call()
    except Exception as error:
        return ["raise", type(error).__name__, str(error), list(error.args)]
    return ["return", result is None, type(result).__name__, repr(result)]


def signature_outcome(function):
    try:
        return ["return", str(inspect.signature(function))]
    except Exception as error:
        message = str(error).replace("torch_rs.torch_rs", "torch")
        args = [
            argument.replace("torch_rs.torch_rs", "torch")
            if isinstance(argument, str)
            else argument
            for argument in error.args
        ]
        return ["raise", type(error).__name__, message, args]


def pickle_shape(function, protocol):
    shape = []
    for opcode, argument, _ in pickletools.genops(
        pickle.dumps(function, protocol=protocol)
    ):
        if opcode.name == "FRAME":
            argument = "<frame length>"
        elif isinstance(argument, str):
            argument = argument.replace("torch_rs.torch_rs", "torch")
            argument = argument.replace("torch_rs", "torch")
        shape.append([opcode.name, argument])
    return shape


wildcard_namespace = {}
exec(f"from {module.__name__} import *", wildcard_namespace)
invalid_values = [
    ("none", None),
    ("true", True),
    ("false", False),
    ("float", 1.0),
    ("str", "1"),
    ("bytes", b"1"),
    ("object", object()),
    ("index_only", IndexOnly()),
    ("numpy_bool", np.bool_(True)),
    ("zero", 0),
    ("negative", -1),
    ("overflow_positive", 10**100),
    ("overflow_negative", -(10**100)),
]
one_result = outcome(lambda: function(1))

print(
    json.dumps(
        {
            "metadata": {
                "type_is_builtin": type(function) is types.BuiltinFunctionType,
                "name": function.__name__,
                "qualname": function.__qualname__,
                "module": function.__module__.replace("torch_rs.torch_rs", "torch"),
                "doc": function.__doc__,
                "text_signature": function.__text_signature__,
                "signature": signature_outcome(function),
                "repr": repr(function),
                "self_is_c": function.__self__ is module._C,
                "c_identity": getattr(module._C, setter_name) is function,
                "all_count": module.__all__.count(setter_name),
                "wildcard_identity": wildcard_namespace[setter_name] is function,
                "copy_identity": copy.copy(function) is function,
                "deepcopy_identity": copy.deepcopy(function) is function,
                "pickle_identity": [
                    pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ],
                "pickle_shapes": [
                    pickle_shape(function, protocol)
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ],
            },
            "binding_errors": [
                outcome(lambda: function()),
                outcome(lambda: function(1, 1)),
                outcome(lambda: function(threads=1)),
                outcome(lambda: function(1, threads=1)),
            ],
            "invalid_value_errors": [
                [label, outcome(lambda value=value: function(value))]
                for label, value in invalid_values
            ],
            "one_result": one_result,
            "getter_after_one": getter(),
        }
    )
)
"""


@unittest.skipUnless(REFERENCE_AVAILABLE, "install the reference dependency group")
class ThreadSetterReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def contract(module_name, setter_name):
            completed = subprocess.run(
                [sys.executable, "-I", "-c", CONTRACT_SCRIPT, module_name, setter_name],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise AssertionError(completed.stdout + completed.stderr)
            return json.loads(completed.stdout)

        cls.actual = {
            name: contract("torch_rs", name)
            for name in ("set_num_threads", "set_num_interop_threads")
        }
        cls.reference = {
            name: contract("torch", name)
            for name in ("set_num_threads", "set_num_interop_threads")
        }

    def test_callable_metadata_exports_copying_and_pickling_match_pytorch_2_13(self):
        for name in ("set_num_threads", "set_num_interop_threads"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.actual[name]["metadata"],
                    self.reference[name]["metadata"],
                )

    def test_binding_and_invalid_value_errors_match_pytorch_2_13(self):
        for name in ("set_num_threads", "set_num_interop_threads"):
            with self.subTest(name=name, errors="binding"):
                self.assertEqual(
                    self.actual[name]["binding_errors"],
                    self.reference[name]["binding_errors"],
                )
            with self.subTest(name=name, errors="values"):
                self.assertEqual(
                    self.actual[name]["invalid_value_errors"],
                    self.reference[name]["invalid_value_errors"],
                )

    def test_single_worker_one_call_matches_pytorch_2_13_return_and_getter(self):
        for name in ("set_num_threads", "set_num_interop_threads"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.actual[name]["one_result"],
                    ["return", True, "NoneType", "None"],
                )
                self.assertEqual(
                    self.actual[name]["one_result"],
                    self.reference[name]["one_result"],
                )
                self.assertEqual(self.actual[name]["getter_after_one"], 1)
                self.assertEqual(
                    self.actual[name]["getter_after_one"],
                    self.reference[name]["getter_after_one"],
                )


if __name__ == "__main__":
    unittest.main()
