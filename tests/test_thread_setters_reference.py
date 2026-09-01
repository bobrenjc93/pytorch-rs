import importlib.util
import json
import subprocess
import sys
import unittest

import torch_rs as torch


REFERENCE_AVAILABLE = importlib.util.find_spec("torch") is not None


CONTRACT_SCRIPT = r"""
import copy
import importlib
import inspect
import json
import pickle
import sys
import types

import numpy as np

module = importlib.import_module(sys.argv[1])
if module.__name__ == "torch" and module.__version__.split("+")[0] != "2.13.0":
    raise AssertionError(
        "thread setter differentials require pinned PyTorch 2.13.0"
    )


class IntSubclass(int):
    def __int__(self):
        raise AssertionError("int subclass conversion must not dispatch __int__")


class IndexOnly:
    def __index__(self):
        raise AssertionError("thread setters must reject index-only objects")


class IntOnly:
    def __int__(self):
        raise AssertionError("thread setters must reject int-only objects")


def signature_outcome(function):
    try:
        return ["return", str(inspect.signature(function))]
    except Exception as error:
        return ["raise", type(error).__name__, str(error), list(error.args)]


def call_outcome(name, expression, getter_name):
    function = getattr(module, name)
    namespace = {
        "function": function,
        "IntSubclass": IntSubclass,
        "IndexOnly": IndexOnly,
        "IntOnly": IntOnly,
        "np": np,
    }
    try:
        result = eval(expression, namespace)
    except Exception as error:
        return {
            "kind": "raise",
            "type": type(error).__name__,
            "message": str(error),
            "args": list(error.args),
        }
    return {
        "kind": "return",
        "result_is_none": result is None,
        "getter": getattr(module, getter_name)(),
    }


name = sys.argv[2]
mode = sys.argv[3]
getter_name = {
    "set_num_threads": "get_num_threads",
    "set_num_interop_threads": "get_num_interop_threads",
}[name]

if mode == "metadata":
    function = getattr(module, name)
    wildcard_namespace = {}
    exec(f"from {module.__name__} import *", wildcard_namespace)
    print(
        json.dumps(
            {
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
                "copy_identity": copy.copy(function) is function,
                "deepcopy_identity": copy.deepcopy(function) is function,
                "pickle_identity": [
                    pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ],
            }
        )
    )
else:
    cases = {
        "one": "function(1)",
        "int_subclass_one": "function(IntSubclass(1))",
        "np_int8_one": "function(np.int8(1))",
        "np_int64_one": "function(np.int64(1))",
        "np_uint32_one": "function(np.uint32(1))",
        "no_args": "function()",
        "two_args": "function(1, 1)",
        "kw_threads": "function(threads=1)",
        "kw_num_threads": "function(num_threads=1)",
        "none": "function(None)",
        "bool_true": "function(True)",
        "bool_false": "function(False)",
        "np_bool": "function(np.bool_(True))",
        "float_one": "function(1.0)",
        "np_float": "function(np.float64(1.0))",
        "str_one": "function('1')",
        "index_only": "function(IndexOnly())",
        "int_only": "function(IntOnly())",
        "zero": "function(0)",
        "minus_one": "function(-1)",
        "overflow_int32_high": "function(2**31)",
        "overflow_int32_low": "function(-(2**31) - 1)",
        "overflow_long_high": "function(2**63)",
        "overflow_long_low": "function(-(2**63) - 1)",
        "overflow_np_uint64": "function(np.uint64(2**63))",
    }
    print(json.dumps(call_outcome(name, cases[mode], getter_name)))
"""


@unittest.skipUnless(REFERENCE_AVAILABLE, "install the reference dependency group")
class ThreadSetterReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._actual = {}
        cls._reference = {}
        cls._cases = (
            "metadata",
            "one",
            "int_subclass_one",
            "np_int8_one",
            "np_int64_one",
            "np_uint32_one",
            "no_args",
            "two_args",
            "kw_threads",
            "kw_num_threads",
            "none",
            "bool_true",
            "bool_false",
            "np_bool",
            "float_one",
            "np_float",
            "str_one",
            "index_only",
            "int_only",
            "zero",
            "minus_one",
            "overflow_int32_high",
            "overflow_int32_low",
            "overflow_long_high",
            "overflow_long_low",
            "overflow_np_uint64",
        )
        for module_name, storage in (
            ("torch_rs", cls._actual),
            ("torch", cls._reference),
        ):
            for name in ("set_num_threads", "set_num_interop_threads"):
                storage[name] = {}
                for case in cls._cases:
                    completed = subprocess.run(
                        [sys.executable, "-I", "-c", CONTRACT_SCRIPT, module_name, name, case],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if completed.returncode != 0:
                        raise AssertionError(completed.stdout + completed.stderr)
                    storage[name][case] = json.loads(completed.stdout)

    def test_builtin_contract_matches_pytorch_2_13(self):
        for name in ("set_num_threads", "set_num_interop_threads"):
            with self.subTest(name=name):
                self.assertEqual(
                    self._actual[name]["metadata"],
                    self._reference[name]["metadata"],
                )

    def test_singleton_integer_conversions_match_pytorch_2_13(self):
        for name in ("set_num_threads", "set_num_interop_threads"):
            for case in (
                "one",
                "int_subclass_one",
                "np_int8_one",
                "np_int64_one",
                "np_uint32_one",
            ):
                with self.subTest(name=name, case=case):
                    self.assertEqual(
                        self._actual[name][case],
                        self._reference[name][case],
                    )

    def test_argument_and_invalid_value_errors_match_pytorch_2_13(self):
        for name in ("set_num_threads", "set_num_interop_threads"):
            for case in (
                "no_args",
                "two_args",
                "kw_threads",
                "kw_num_threads",
                "none",
                "bool_true",
                "bool_false",
                "np_bool",
                "float_one",
                "np_float",
                "str_one",
                "index_only",
                "int_only",
                "zero",
                "minus_one",
                "overflow_int32_high",
                "overflow_int32_low",
                "overflow_long_high",
                "overflow_long_low",
                "overflow_np_uint64",
            ):
                with self.subTest(name=name, case=case):
                    self.assertEqual(
                        self._actual[name][case],
                        self._reference[name][case],
                    )

    def test_non_singleton_positive_counts_remain_unsupported(self):
        for name in ("set_num_threads", "set_num_interop_threads"):
            setter = getattr(torch, name)
            for value in (2, 2**31 - 1):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(NotImplementedError):
                        setter(value)
                    self.assertEqual(torch.get_num_threads(), 1)
                    self.assertEqual(torch.get_num_interop_threads(), 1)


if __name__ == "__main__":
    unittest.main()
