import contextlib
import copy
import importlib
import inspect
import pickle
import re
import sys
import threading
import types
import unittest

import numpy as np

import torch_rs as torch


FLAGS_DOC = "Context manager for setting if nnpack is enabled globally"


class _BodyError(Exception):
    pass


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("flags must not request truthiness")


class NnpackFlagsTests(unittest.TestCase):
    def setUp(self):
        self.nnpack = importlib.import_module("torch_rs.backends.nnpack")
        self.original = torch._C._get_nnpack_enabled()
        self.nnpack.set_flags(True)

    def tearDown(self):
        self.nnpack.set_flags(self.original)

    def test_default_explicit_nested_and_exception_contexts_restore_state(self):
        with self.nnpack.flags() as entered:
            self.assertIsNone(entered)
            self.assertIs(torch._C._get_nnpack_enabled(), False)
            self.assertIs(self.nnpack.is_available(), False)
            with self.nnpack.flags(True) as nested_entered:
                self.assertIsNone(nested_entered)
                self.assertIs(torch._C._get_nnpack_enabled(), True)
            self.assertIs(torch._C._get_nnpack_enabled(), False)
        self.assertIs(torch._C._get_nnpack_enabled(), True)

        self.nnpack.set_flags(False)
        with self.assertRaises(_BodyError):
            with self.nnpack.flags(enabled=True) as entered:
                self.assertIsNone(entered)
                self.assertIs(torch._C._get_nnpack_enabled(), True)
                raise _BodyError("body failed")
        self.assertIs(torch._C._get_nnpack_enabled(), False)

    def test_invalid_values_are_validated_only_when_the_context_is_entered(self):
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
            for value, type_name in invalid_values:
                with self.subTest(state=state, value_type=type_name):
                    self.nnpack.set_flags(state)
                    context = self.nnpack.flags(value)
                    self.assertIs(
                        type(context),
                        contextlib._GeneratorContextManager,
                    )
                    self.assertIs(torch._C._get_nnpack_enabled(), state)

                    message = (
                        "set_enabled_NNPACK expects a bool, but got "
                        f"{type_name}"
                    )
                    with self.assertRaises(RuntimeError) as raised:
                        context.__enter__()
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(torch._C._get_nnpack_enabled(), state)

    def test_call_binding_errors_are_immediate_and_leave_state_unchanged(self):
        cases = (
            lambda: self.nnpack.flags(True, False),
            lambda: self.nnpack.flags(_enabled=True),
            lambda: self.nnpack.flags(True, enabled=False),
        )
        for call in cases:
            with self.subTest(call=call):
                self.nnpack.set_flags(True)
                with self.assertRaises(TypeError):
                    call()
                self.assertIs(torch._C._get_nnpack_enabled(), True)

    def test_context_state_is_process_global_across_threads(self):
        entered = threading.Event()
        leave = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                with self.nnpack.flags(False) as value:
                    observations.append((value, torch._C._get_nnpack_enabled()))
                    entered.set()
                    if not leave.wait(timeout=10):
                        raise RuntimeError("timed out waiting to leave context")
                observations.append(torch._C._get_nnpack_enabled())
            except BaseException as error:
                errors.append(error)
                entered.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(entered.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(torch._C._get_nnpack_enabled(), False)
        with self.nnpack.flags(True) as value:
            self.assertIsNone(value)
            self.assertIs(torch._C._get_nnpack_enabled(), True)
        self.assertIs(torch._C._get_nnpack_enabled(), False)
        leave.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [(None, False), True])
        self.assertIs(torch._C._get_nnpack_enabled(), True)

    def test_metadata_imports_copying_and_pickling_match_public_contract(self):
        nnpack = self.nnpack
        function = nnpack.flags
        wrapped = function.__wrapped__

        self.assertEqual(nnpack.__all__, ["is_available", "flags", "set_flags"])
        self.assertEqual(
            {name for name in vars(nnpack) if not name.startswith("_")},
            {"contextmanager", "flags", "is_available", "set_flags", "torch"},
        )
        self.assertIs(nnpack.contextmanager, contextlib.contextmanager)

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(enabled=False)")
        self.assertEqual(inspect.get_annotations(function), {})
        self.assertEqual(function.__name__, "flags")
        self.assertEqual(function.__qualname__, "flags")
        self.assertEqual(function.__module__, "torch_rs.backends.nnpack")
        self.assertIs(inspect.getmodule(function), nnpack)
        self.assertEqual(function.__doc__, FLAGS_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(set(function.__dict__), {"__wrapped__"})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_names, ("_GeneratorContextManager",))
        self.assertEqual(function.__code__.co_freevars, ("func",))
        self.assertEqual(function.__code__.co_cellvars, ())
        self.assertFalse(inspect.isgeneratorfunction(function))

        self.assertIs(type(wrapped), types.FunctionType)
        self.assertEqual(str(inspect.signature(wrapped)), "(enabled=False)")
        self.assertEqual(wrapped.__name__, "flags")
        self.assertEqual(wrapped.__qualname__, "flags")
        self.assertEqual(wrapped.__module__, "torch_rs.backends.nnpack")
        self.assertEqual(wrapped.__doc__, FLAGS_DOC)
        self.assertEqual(wrapped.__defaults__, (False,))
        self.assertIsNone(wrapped.__kwdefaults__)
        self.assertEqual(wrapped.__dict__, {})
        self.assertEqual(
            wrapped.__code__.co_names,
            ("__allow_nonbracketed_mutation", "set_flags"),
        )
        self.assertEqual(wrapped.__code__.co_freevars, ())
        self.assertEqual(wrapped.__code__.co_cellvars, ())
        self.assertTrue(inspect.isgeneratorfunction(wrapped))

        backend_import = {}
        function_import = {}
        child_wildcard = {}
        exec("from torch_rs.backends import nnpack", backend_import)
        exec("from torch_rs.backends.nnpack import flags", function_import)
        exec("from torch_rs.backends.nnpack import *", child_wildcard)
        self.assertIs(backend_import["nnpack"], nnpack)
        self.assertIs(function_import["flags"], function)
        self.assertIs(child_wildcard["flags"], function)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"flags", "is_available", "set_flags"},
        )

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.nnpack", payload)
                self.assertIs(pickle.loads(payload), function)

        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(wrapped)
        with self.assertRaises(TypeError) as raised:
            pickle.dumps(function())
        self.assertEqual(str(raised.exception), "cannot pickle 'generator' object")

    def test_reload_preserves_active_and_pending_contexts(self):
        nnpack = self.nnpack
        old_flags = nnpack.flags
        active = old_flags(False)
        pending = old_flags(False)
        namespace = nnpack.__dict__

        self.assertIsNone(active.__enter__())
        self.assertIs(torch._C._get_nnpack_enabled(), False)
        reloaded = importlib.reload(nnpack)

        self.assertIs(reloaded, nnpack)
        self.assertIs(nnpack.__dict__, namespace)
        self.assertIs(torch.backends.nnpack, nnpack)
        self.assertIs(sys.modules[nnpack.__name__], nnpack)
        self.assertIsNot(nnpack.flags, old_flags)
        self.assertIs(torch._C._get_nnpack_enabled(), False)

        with nnpack.flags(True) as entered:
            self.assertIsNone(entered)
            self.assertIs(torch._C._get_nnpack_enabled(), True)
        self.assertIs(torch._C._get_nnpack_enabled(), False)

        self.assertIs(active.__exit__(None, None, None), False)
        self.assertIs(torch._C._get_nnpack_enabled(), True)
        self.assertIsNone(pending.__enter__())
        self.assertIs(torch._C._get_nnpack_enabled(), False)
        self.assertIs(pending.__exit__(None, None, None), False)
        self.assertIs(torch._C._get_nnpack_enabled(), True)

        with old_flags(False):
            self.assertIs(torch._C._get_nnpack_enabled(), False)
        self.assertIs(torch._C._get_nnpack_enabled(), True)

        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_flags)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function flags at 0x...>: it's not the same "
            "object as torch_rs.backends.nnpack.flags",
        )
        self.assertIs(pickle.loads(pickle.dumps(nnpack.flags)), nnpack.flags)

    def test_context_manager_does_not_add_nnpack_execution(self):
        self.assertFalse(hasattr(torch, "_nnpack_spatial_convolution"))
        with self.nnpack.flags(True):
            self.assertIs(self.nnpack.is_available(), False)
            self.assertFalse(hasattr(torch, "_nnpack_spatial_convolution"))


if __name__ == "__main__":
    unittest.main()
