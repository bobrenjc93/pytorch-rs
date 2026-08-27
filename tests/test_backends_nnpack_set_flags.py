import copy
import importlib
import inspect
import json
import pickle
import re
import subprocess
import sys
import threading
import types
import unittest

import numpy as np

import torch_rs as torch


SET_FLAGS_DOC = "Set if nnpack is enabled globally"


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("set_flags must not request truthiness")


class NnpackSetFlagsTests(unittest.TestCase):
    def setUp(self):
        self.nnpack = importlib.import_module("torch_rs.backends.nnpack")
        self.original = torch._C._get_nnpack_enabled()
        self.nnpack.set_flags(True)

    def tearDown(self):
        self.nnpack.set_flags(self.original)

    def test_fresh_process_defaults_to_enabled_while_build_is_unavailable(self):
        script = r'''
import json

import torch_rs as torch

nnpack = torch.backends.nnpack
print(json.dumps({
    "available": nnpack.is_available(),
    "enabled": torch._C._get_nnpack_enabled(),
    "first": nnpack.set_flags(False),
    "second": nnpack.set_flags(True),
    "flags": hasattr(nnpack, "flags"),
    "execution": hasattr(torch, "_nnpack_spatial_convolution"),
}))
'''
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
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "available": False,
                "enabled": True,
                "first": [True],
                "second": [False],
                "flags": False,
                "execution": False,
            },
        )

    def test_repeated_exact_bool_updates_return_previous_exact_bool(self):
        previous = True
        for enabled in (False, True, True, False, False, True):
            with self.subTest(enabled=enabled):
                result = self.nnpack.set_flags(enabled)
                self.assertIs(type(result), tuple)
                self.assertEqual(len(result), 1)
                self.assertIs(type(result[0]), bool)
                self.assertIs(result[0], previous)
                self.assertIs(torch._C._get_nnpack_enabled(), enabled)
                self.assertIs(self.nnpack.is_available(), False)
                previous = enabled

        self.assertEqual(self.nnpack.set_flags(_enabled=False), (True,))
        self.assertIs(torch._C._get_nnpack_enabled(), False)

    def test_non_bool_values_are_rejected_without_coercion_or_state_change(self):
        invalid_values = (
            (None, "NoneType"),
            (0, "int"),
            (1, "int"),
            (0.0, "float"),
            (np.bool_(True), "numpy.bool"),
            ("", "str"),
            ([], "list"),
            (object(), "object"),
            (_RejectTruthiness(), "_RejectTruthiness"),
            (torch.tensor(True), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
            (torch.strided, "torch.layout"),
            (torch.Size([1]), "torch.Size"),
            (torch.finfo(torch.float32), "torch.finfo"),
        )
        for state in (False, True):
            self.nnpack.set_flags(state)
            for value, type_name in invalid_values:
                with self.subTest(state=state, value_type=type_name):
                    message = (
                        "set_enabled_NNPACK expects a bool, but got "
                        f"{type_name}"
                    )
                    with self.assertRaises(RuntimeError) as raised:
                        self.nnpack.set_flags(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(torch._C._get_nnpack_enabled(), state)
                    self.assertIs(self.nnpack.is_available(), False)

    def test_state_is_process_global_across_threads(self):
        nnpack = self.nnpack
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(nnpack.set_flags(False))
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(nnpack.set_flags(False))
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(torch._C._get_nnpack_enabled(), False)
        self.assertEqual(nnpack.set_flags(True), (False,))
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [(True,), (True,)])
        self.assertIs(torch._C._get_nnpack_enabled(), False)

    def test_reload_preserves_state_for_old_and_new_setters(self):
        nnpack = self.nnpack
        old_is_available = nnpack.is_available
        old_setter = nnpack.set_flags
        namespace = nnpack.__dict__

        self.assertEqual(old_setter(False), (True,))
        reloaded = importlib.reload(nnpack)

        self.assertIs(reloaded, nnpack)
        self.assertIs(nnpack.__dict__, namespace)
        self.assertIs(torch.backends.nnpack, nnpack)
        self.assertIs(sys.modules[nnpack.__name__], nnpack)
        self.assertIsNot(nnpack.is_available, old_is_available)
        self.assertIsNot(nnpack.set_flags, old_setter)
        self.assertIs(torch._C._get_nnpack_enabled(), False)
        self.assertEqual(nnpack.set_flags(True), (False,))
        self.assertEqual(old_setter(False), (True,))
        self.assertEqual(nnpack.set_flags(True), (False,))

        for name, old_function in (
            ("is_available", old_is_available),
            ("set_flags", old_setter),
        ):
            with self.subTest(stale_function=name):
                with self.assertRaises(pickle.PicklingError) as raised:
                    pickle.dumps(old_function)
                message = re.sub(
                    r"0x[0-9a-fA-F]+",
                    "0x...",
                    str(raised.exception),
                )
                self.assertEqual(
                    message,
                    f"Can't pickle <function {name} at 0x...>: "
                    "it's not the same object as "
                    f"torch_rs.backends.nnpack.{name}",
                )

    def test_metadata_imports_copying_and_pickling_match_public_contract(self):
        nnpack = self.nnpack
        function = nnpack.set_flags

        self.assertIs(torch.backends.nnpack, nnpack)
        self.assertIs(sys.modules["torch_rs.backends.nnpack"], nnpack)
        self.assertIs(type(nnpack), types.ModuleType)
        self.assertIsNone(nnpack.__doc__)
        self.assertEqual(nnpack.__all__, ["is_available", "set_flags"])
        self.assertEqual(
            {name for name in vars(nnpack) if not name.startswith("_")},
            {"is_available", "set_flags", "torch"},
        )
        self.assertEqual(nnpack.__annotations__, {})
        self.assertIs(nnpack.torch, torch)

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(_enabled)")
        self.assertEqual(inspect.get_annotations(function), {})
        self.assertEqual(function.__name__, "set_flags")
        self.assertEqual(function.__qualname__, "set_flags")
        self.assertEqual(function.__module__, "torch_rs.backends.nnpack")
        self.assertIs(inspect.getmodule(function), nnpack)
        self.assertEqual(function.__doc__, SET_FLAGS_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(
            function.__code__.co_names,
            (
                "torch",
                "_C",
                "_get_nnpack_enabled",
                "_set_nnpack_enabled",
            ),
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        backend_import = {}
        function_import = {}
        child_wildcard = {}
        exec("from torch_rs.backends import nnpack", backend_import)
        exec(
            "from torch_rs.backends.nnpack import set_flags",
            function_import,
        )
        exec("from torch_rs.backends.nnpack import *", child_wildcard)
        self.assertIs(backend_import["nnpack"], nnpack)
        self.assertIs(function_import["set_flags"], function)
        self.assertIs(child_wildcard["set_flags"], function)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"is_available", "set_flags"},
        )

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.nnpack", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_call_binding_errors_leave_state_unchanged(self):
        self.nnpack.set_flags(True)
        unexpected_keyword = (
            "set_flags() got an unexpected keyword argument 'enabled'"
        )
        if sys.version_info >= (3, 13):
            unexpected_keyword += ". Did you mean '_enabled'?"
        cases = (
            (
                lambda: self.nnpack.set_flags(),
                "set_flags() missing 1 required positional argument: '_enabled'",
            ),
            (
                lambda: self.nnpack.set_flags(True, False),
                "set_flags() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: self.nnpack.set_flags(enabled=True),
                unexpected_keyword,
            ),
            (
                lambda: self.nnpack.set_flags(True, _enabled=False),
                "set_flags() got multiple values for argument '_enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(torch._C._get_nnpack_enabled(), True)

    def test_private_state_accessors_are_not_public_backend_capabilities(self):
        self.assertTrue(hasattr(torch._C, "_get_nnpack_enabled"))
        self.assertTrue(hasattr(torch._C, "_set_nnpack_enabled"))
        self.assertFalse(hasattr(torch, "_get_nnpack_enabled"))
        self.assertFalse(hasattr(torch, "_set_nnpack_enabled"))
        self.assertNotIn("_get_nnpack_enabled", torch._C.__all__)
        self.assertNotIn("_set_nnpack_enabled", torch._C.__all__)
        self.assertFalse(hasattr(self.nnpack, "flags"))
        self.assertFalse(hasattr(torch, "_nnpack_spatial_convolution"))
        self.assertIs(self.nnpack.is_available(), False)


if __name__ == "__main__":
    unittest.main()
