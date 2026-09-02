import importlib.util
import json
import subprocess
import sys
import unittest

import torch_rs as torch


REFERENCE_AVAILABLE = importlib.util.find_spec("torch") is not None

METADATA_SCRIPT = r"""
import copy
import importlib
import inspect
import json
import pickle
import sys
import types

module = importlib.import_module(sys.argv[1])
if module.__name__ == "torch" and module.__version__.split("+")[0] != "2.13.0":
    raise AssertionError("thread setter differentials require pinned PyTorch 2.13.0")


def signature_outcome(function):
    try:
        return ["return", str(inspect.signature(function))]
    except Exception as error:
        return ["raise", type(error).__name__, str(error), list(error.args)]


metadata = {}
for name in ("set_num_threads", "set_num_interop_threads"):
    function = getattr(module, name)
    wildcard_namespace = {}
    exec(f"from {module.__name__} import *", wildcard_namespace)
    native_namespace = {}
    exec(f"from {module.__name__}._C import {name}", native_namespace)
    metadata[name] = {
        "type_is_builtin": type(function) is types.BuiltinFunctionType,
        "name": function.__name__,
        "qualname": function.__qualname__,
        "module": function.__module__.replace("torch_rs.torch_rs", "torch"),
        "doc": function.__doc__,
        "text_signature": function.__text_signature__,
        "signature": signature_outcome(function),
        "repr": repr(function),
        "self_is_c": function.__self__ is module._C,
        "c_identity": getattr(module._C, name) is function,
        "all_count": module.__all__.count(name),
        "wildcard_identity": wildcard_namespace[name] is function,
        "native_import_identity": native_namespace[name] is function,
        "copy_identity": copy.copy(function) is function,
        "deepcopy_identity": copy.deepcopy(function) is function,
        "reduce": function.__reduce__(),
        "pickle_identity": [
            pickle.loads(pickle.dumps(function, protocol=protocol)) is function
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
        ],
    }
print(json.dumps(metadata))
"""

CALL_SCRIPT = r"""
import importlib
import json
import pickle
import sys

import numpy as np

module = importlib.import_module(sys.argv[1])
if module.__name__ == "torch" and module.__version__.split("+")[0] != "2.13.0":
    raise AssertionError("thread setter differentials require pinned PyTorch 2.13.0")
name = sys.argv[2]
case = sys.argv[3]
function = getattr(module, name)


class OneInt(int):
    pass


class IndexLike:
    def __index__(self):
        return 1


values = {
    "int_one": 1,
    "int_subclass_one": OneInt(1),
    "numpy_int32_one": np.int32(1),
    "numpy_int64_one": np.int64(1),
    "numpy_uint8_one": np.uint8(1),
    "bool_true": True,
    "float_one": 1.0,
    "string_one": "1",
    "none": None,
    "object": object(),
    "index_like": IndexLike(),
    "numpy_array_one": np.array(1),
    "numpy_bool_true": np.bool_(True),
    "numpy_float64_one": np.float64(1.0),
    "zero": 0,
    "negative": -1,
}


def call():
    if case == "no_args":
        return function()
    if case == "two_args":
        return function(1, 1)
    if case == "keyword":
        return function(threads=1)
    if case == "positional_keyword":
        return function(1, threads=1)
    return function(values[case])


try:
    result = call()
except Exception as error:
    print(
        json.dumps(
            {
                "kind": "raise",
                "type": type(error).__name__,
                "message": str(error),
                "args": list(error.args),
                "target_getter": (
                    module.get_num_threads()
                    if name == "set_num_threads"
                    else module.get_num_interop_threads()
                ),
            }
        )
    )
else:
    print(
        json.dumps(
            {
                "kind": "return",
                "is_none": result is None,
                "target_getter_type_is_int": (
                    type(
                        module.get_num_threads()
                        if name == "set_num_threads"
                        else module.get_num_interop_threads()
                    )
                    is int
                ),
                "target_getter": (
                    module.get_num_threads()
                    if name == "set_num_threads"
                    else module.get_num_interop_threads()
                ),
            }
        )
    )
"""

UNSUPPORTED_SCRIPT = r"""
import importlib
import json
import sys

import numpy as np

module = importlib.import_module("torch_rs")
name = sys.argv[1]
case = sys.argv[2]
function = getattr(module, name)
values = {"two": 2, "numpy_int64_two": np.int64(2)}

try:
    function(values[case])
except Exception as error:
    outcome = {
        "kind": "raise",
        "type": type(error).__name__,
        "message": str(error),
        "args": list(error.args),
        "getters": [module.get_num_threads(), module.get_num_interop_threads()],
    }
else:
    outcome = {
        "kind": "return",
        "getters": [module.get_num_threads(), module.get_num_interop_threads()],
    }
print(json.dumps(outcome))
"""


@unittest.skipUnless(REFERENCE_AVAILABLE, "install the reference dependency group")
class ThreadSettersReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actual_metadata = cls.run_script(METADATA_SCRIPT, "torch_rs")
        cls.reference_metadata = cls.run_script(METADATA_SCRIPT, "torch")

    @classmethod
    def run_script(cls, script, *arguments):
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script, *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        return json.loads(completed.stdout)

    def test_builtin_metadata_exports_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(self.actual_metadata, self.reference_metadata)

    def test_integer_scalar_one_forms_match_pytorch_2_13(self):
        cases = (
            "int_one",
            "int_subclass_one",
            "numpy_int32_one",
            "numpy_int64_one",
            "numpy_uint8_one",
        )
        for name in ("set_num_threads", "set_num_interop_threads"):
            for case in cases:
                with self.subTest(name=name, case=case):
                    actual = self.run_script(CALL_SCRIPT, "torch_rs", name, case)
                    reference = self.run_script(CALL_SCRIPT, "torch", name, case)
                    self.assertEqual(actual, reference)
                    self.assertEqual(
                        actual,
                        {
                            "kind": "return",
                            "is_none": True,
                            "target_getter_type_is_int": True,
                            "target_getter": 1,
                        },
                    )

    def test_overlapping_invalid_values_match_pytorch_2_13(self):
        cases = (
            "bool_true",
            "float_one",
            "string_one",
            "none",
            "object",
            "index_like",
            "numpy_array_one",
            "numpy_bool_true",
            "numpy_float64_one",
            "zero",
            "negative",
            "no_args",
            "two_args",
            "keyword",
            "positional_keyword",
        )
        for name in ("set_num_threads", "set_num_interop_threads"):
            for case in cases:
                with self.subTest(name=name, case=case):
                    actual = self.run_script(CALL_SCRIPT, "torch_rs", name, case)
                    reference = self.run_script(CALL_SCRIPT, "torch", name, case)
                    self.assertEqual(actual["kind"], "raise")
                    self.assertEqual(actual["type"], reference["type"])
                    self.assertEqual(actual["message"], reference["message"])
                    self.assertEqual(actual["args"], reference["args"])
                    self.assertEqual(actual["target_getter"], 1)

    def test_multiworker_counts_remain_explicitly_unsupported(self):
        for name in ("set_num_threads", "set_num_interop_threads"):
            for case in ("two", "numpy_int64_two"):
                with self.subTest(name=name, case=case):
                    outcome = self.run_script(UNSUPPORTED_SCRIPT, name, case)
                    message = (
                        f"{name}(): mutable thread pools are not supported; "
                        "only 1 worker is implemented"
                    )
                    self.assertEqual(
                        outcome,
                        {
                            "kind": "raise",
                            "type": "NotImplementedError",
                            "message": message,
                            "args": [message],
                            "getters": [1, 1],
                        },
                    )

    def test_torch_rs_reload_keeps_native_setter_identity_and_noop_behavior(self):
        import importlib

        old_functions = {
            name: getattr(torch, name)
            for name in ("set_num_threads", "set_num_interop_threads")
        }
        native = torch._C

        self.assertIs(importlib.reload(native), native)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch._C, native)
        for name, function in old_functions.items():
            with self.subTest(name=name):
                self.assertIs(getattr(torch, name), function)
                self.assertIs(getattr(torch._C, name), function)
                self.assertIs(function(1), None)
                self.assertIs(torch.get_num_threads(), 1)
                self.assertIs(torch.get_num_interop_threads(), 1)


if __name__ == "__main__":
    unittest.main()
