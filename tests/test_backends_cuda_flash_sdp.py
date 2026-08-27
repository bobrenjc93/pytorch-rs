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
import typing
import unittest

import numpy as np

import torch_rs as torch


FLASH_SDP_ENABLED_DOC = """
    .. warning:: This flag is beta and subject to change.

    Returns whether flash scaled dot product attention is enabled or not.
    """

ENABLE_FLASH_SDP_DOC = """
    .. warning:: This flag is beta and subject to change.

    Enables or disables flash scaled dot product attention.
    """


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("enable_flash_sdp must not request truthiness")


class CudaFlashSdpTests(unittest.TestCase):
    def setUp(self):
        self.cuda = importlib.import_module("torch_rs.backends.cuda")
        self.original = torch._C._get_flash_sdp_enabled()
        self.cuda.enable_flash_sdp(True)

    def tearDown(self):
        self.cuda.enable_flash_sdp(self.original)

    def test_fresh_process_defaults_to_exact_true_without_flash_attention(self):
        script = r'''
import json

import torch_rs as torch

cuda = torch.backends.cuda
initial = cuda.flash_sdp_enabled()
first = cuda.enable_flash_sdp(False)
disabled = cuda.flash_sdp_enabled()
second = cuda.enable_flash_sdp(True)
print(json.dumps({
    "initial": initial,
    "initial_type": type(initial).__name__,
    "first": first,
    "disabled": disabled,
    "second": second,
    "restored": cuda.flash_sdp_enabled(),
    "available": cuda.is_flash_attention_available(),
    "cuda": hasattr(torch, "cuda"),
    "execution": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
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
                "initial": True,
                "initial_type": "bool",
                "first": None,
                "disabled": False,
                "second": None,
                "restored": True,
                "available": False,
                "cuda": False,
                "execution": False,
            },
        )

    def test_repeated_exact_bool_updates_are_preference_only(self):
        cuda = self.cuda

        self.assertIs(cuda.flash_sdp_enabled(), True)
        self.assertIs(type(cuda.flash_sdp_enabled()), bool)
        for enabled in (False, True, True, False, False, True):
            with self.subTest(enabled=enabled):
                self.assertIs(cuda.enable_flash_sdp(enabled), None)
                self.assertIs(cuda.flash_sdp_enabled(), enabled)
                self.assertIs(torch._C._get_flash_sdp_enabled(), enabled)
                self.assertIs(cuda.is_flash_attention_available(), False)

    def test_non_bool_values_are_rejected_without_state_change(self):
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
            self.cuda.enable_flash_sdp(state)
            for value, type_name in invalid_values:
                with self.subTest(state=state, value_type=type_name):
                    message = (
                        "set_sdp_use_math expects a bool, but got "
                        f"{type_name}"
                    )
                    with self.assertRaises(RuntimeError) as raised:
                        self.cuda.enable_flash_sdp(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(self.cuda.flash_sdp_enabled(), state)
                    self.assertIs(torch._C._get_flash_sdp_enabled(), state)
                    self.assertIs(
                        self.cuda.is_flash_attention_available(),
                        False,
                    )

    def test_state_is_process_global_across_threads_and_aliases(self):
        cuda = self.cuda
        imported = importlib.import_module("torch_rs.backends.cuda")
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        self.assertIs(imported, cuda)

        def worker():
            try:
                observations.append(cuda.flash_sdp_enabled())
                observations.append(imported.enable_flash_sdp(False))
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(imported.flash_sdp_enabled())
                observations.append(cuda.enable_flash_sdp(False))
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(cuda.flash_sdp_enabled(), False)
        self.assertIs(cuda.enable_flash_sdp(True), None)
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, None, True, None])
        self.assertIs(cuda.flash_sdp_enabled(), False)
        self.assertIs(torch._C._get_flash_sdp_enabled(), False)

    def test_reload_preserves_state_and_replaces_public_functions(self):
        cuda = self.cuda
        old_getter = cuda.flash_sdp_enabled
        old_setter = cuda.enable_flash_sdp
        namespace = cuda.__dict__

        self.assertIs(old_setter(False), None)
        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIsNot(cuda.flash_sdp_enabled, old_getter)
        self.assertIsNot(cuda.enable_flash_sdp, old_setter)
        self.assertIs(cuda.flash_sdp_enabled(), False)
        self.assertIs(cuda.enable_flash_sdp(True), None)
        self.assertIs(old_getter(), True)
        self.assertIs(old_setter(False), None)
        self.assertIs(cuda.flash_sdp_enabled(), False)
        self.assertIs(cuda.enable_flash_sdp(True), None)

        for name, old_function in (
            ("flash_sdp_enabled", old_getter),
            ("enable_flash_sdp", old_setter),
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
                    f"torch_rs.backends.cuda.{name}",
                )

    def test_metadata_exports_copying_and_pickling_match_public_contract(self):
        cuda = self.cuda
        getter = cuda.flash_sdp_enabled
        setter = cuda.enable_flash_sdp

        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules["torch_rs.backends.cuda"], cuda)
        self.assertIs(type(cuda), types.ModuleType)
        self.assertIsNone(cuda.__doc__)
        self.assertEqual(
            cuda.__all__,
            [
                "is_built",
                "is_ck_sdpa_available",
                "enable_flash_sdp",
                "flash_sdp_enabled",
                "is_flash_attention_available",
            ],
        )
        self.assertEqual(
            {name for name in vars(cuda) if not name.startswith("_")},
            {
                "enable_flash_sdp",
                "flash_sdp_enabled",
                "is_built",
                "is_ck_sdpa_available",
                "is_flash_attention_available",
                "torch",
            },
        )
        self.assertIs(cuda.torch, torch)

        cases = (
            (
                getter,
                "flash_sdp_enabled",
                "()",
                {},
                FLASH_SDP_ENABLED_DOC,
                ("torch", "_C", "_get_flash_sdp_enabled"),
            ),
            (
                setter,
                "enable_flash_sdp",
                "(enabled: bool)",
                {"enabled": bool},
                ENABLE_FLASH_SDP_DOC,
                ("torch", "_C", "_set_sdp_use_flash"),
            ),
        )
        for function, name, signature, annotations, doc, code_names in cases:
            with self.subTest(function=name):
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(inspect.get_annotations(function), annotations)
                self.assertEqual(typing.get_type_hints(function), annotations)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(
                    function.__module__,
                    "torch_rs.backends.cuda",
                )
                self.assertIs(inspect.getmodule(function), cuda)
                self.assertEqual(function.__doc__, doc)
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))
                self.assertEqual(function.__code__.co_names, code_names)
                self.assertEqual(function.__code__.co_freevars, ())
                self.assertEqual(function.__code__.co_cellvars, ())

        backend_import = {}
        getter_import = {}
        setter_import = {}
        child_wildcard = {}
        exec("from torch_rs.backends import cuda", backend_import)
        exec(
            "from torch_rs.backends.cuda import flash_sdp_enabled",
            getter_import,
        )
        exec(
            "from torch_rs.backends.cuda import enable_flash_sdp",
            setter_import,
        )
        exec("from torch_rs.backends.cuda import *", child_wildcard)
        self.assertIs(backend_import["cuda"], cuda)
        self.assertIs(getter_import["flash_sdp_enabled"], getter)
        self.assertIs(setter_import["enable_flash_sdp"], setter)
        self.assertIs(child_wildcard["flash_sdp_enabled"], getter)
        self.assertIs(child_wildcard["enable_flash_sdp"], setter)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {
                "enable_flash_sdp",
                "flash_sdp_enabled",
                "is_built",
                "is_ck_sdpa_available",
                "is_flash_attention_available",
            },
        )

        for function in (getter, setter):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function.__name__, protocol=protocol):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.backends.cuda", payload)
                    self.assertIs(pickle.loads(payload), function)

    def test_binding_errors_leave_state_unchanged(self):
        cuda = self.cuda
        cuda.enable_flash_sdp(True)
        cases = (
            (
                lambda: cuda.flash_sdp_enabled(None),
                "flash_sdp_enabled() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: cuda.flash_sdp_enabled(enabled=True),
                "flash_sdp_enabled() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: cuda.enable_flash_sdp(),
                "enable_flash_sdp() missing 1 required positional argument: 'enabled'",
            ),
            (
                lambda: cuda.enable_flash_sdp(True, False),
                "enable_flash_sdp() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: cuda.enable_flash_sdp(_enabled=False),
                "enable_flash_sdp() got an unexpected keyword argument '_enabled'",
            ),
            (
                lambda: cuda.enable_flash_sdp(True, enabled=False),
                "enable_flash_sdp() got multiple values for argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(cuda.flash_sdp_enabled(), True)

        self.assertIs(cuda.enable_flash_sdp(enabled=False), None)
        self.assertIs(cuda.flash_sdp_enabled(), False)

    def test_private_accessors_and_sdp_execution_boundary(self):
        self.assertTrue(hasattr(torch._C, "_get_flash_sdp_enabled"))
        self.assertTrue(hasattr(torch._C, "_set_sdp_use_flash"))
        self.assertFalse(hasattr(torch, "_get_flash_sdp_enabled"))
        self.assertFalse(hasattr(torch, "_set_sdp_use_flash"))
        self.assertNotIn("_get_flash_sdp_enabled", torch._C.__all__)
        self.assertNotIn("_set_sdp_use_flash", torch._C.__all__)

        self.assertIs(torch._C._set_sdp_use_flash(False), None)
        self.assertIs(torch._C._get_flash_sdp_enabled(), False)
        self.assertIs(self.cuda.flash_sdp_enabled(), False)
        self.assertIs(self.cuda.is_flash_attention_available(), False)
        self.assertFalse(
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
        )
        self.assertFalse(hasattr(torch, "cuda"))


if __name__ == "__main__":
    unittest.main()
