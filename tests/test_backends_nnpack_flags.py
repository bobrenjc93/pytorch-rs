import copy
import importlib
import inspect
import pickle
import re
import sys
import threading
import types
import unittest

import torch_rs as torch


FLAGS_DOC = "Context manager for setting if nnpack is enabled globally"


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("flags must not request truthiness")


class _ContextBodyError(Exception):
    pass


class NnpackFlagsTests(unittest.TestCase):
    def setUp(self):
        self.nnpack = importlib.import_module("torch_rs.backends.nnpack")
        self.original = torch._C._get_nnpack_enabled()
        self.nnpack.set_flags(True)

    def tearDown(self):
        self.nnpack.set_flags(self.original)

    def test_default_explicit_nested_and_exceptional_restoration(self):
        nnpack = self.nnpack

        for initial in (False, True):
            with self.subTest(initial=initial, enabled="default"):
                nnpack.set_flags(initial)
                context = nnpack.flags()
                self.assertIs(torch._C._get_nnpack_enabled(), initial)
                with context as entered:
                    self.assertIsNone(entered)
                    self.assertIs(torch._C._get_nnpack_enabled(), False)
                    self.assertIs(nnpack.is_available(), False)
                self.assertIs(torch._C._get_nnpack_enabled(), initial)

            for enabled in (False, True):
                with self.subTest(initial=initial, enabled=enabled):
                    nnpack.set_flags(initial)
                    context = nnpack.flags(enabled=enabled)
                    self.assertIs(torch._C._get_nnpack_enabled(), initial)
                    self.assertIsNone(context.__enter__())
                    self.assertIs(torch._C._get_nnpack_enabled(), enabled)
                    self.assertIs(context.__exit__(None, None, None), False)
                    self.assertIs(torch._C._get_nnpack_enabled(), initial)

        nnpack.set_flags(True)
        with nnpack.flags(False) as outer:
            self.assertIsNone(outer)
            self.assertIs(torch._C._get_nnpack_enabled(), False)
            with nnpack.flags(True) as inner:
                self.assertIsNone(inner)
                self.assertIs(torch._C._get_nnpack_enabled(), True)
            self.assertIs(torch._C._get_nnpack_enabled(), False)
        self.assertIs(torch._C._get_nnpack_enabled(), True)

        marker = _ContextBodyError("body failed")
        with self.assertRaises(_ContextBodyError) as raised:
            with nnpack.flags(False) as entered:
                self.assertIsNone(entered)
                self.assertIs(torch._C._get_nnpack_enabled(), False)
                raise marker
        self.assertIs(raised.exception, marker)
        self.assertIs(torch._C._get_nnpack_enabled(), True)

    def test_strict_boolean_validation_is_deferred_until_entry(self):
        nnpack = self.nnpack
        invalid_values = (
            (None, "NoneType"),
            (0, "int"),
            (1, "int"),
            (object(), "object"),
            (_RejectTruthiness(), "_RejectTruthiness"),
        )

        for state in (False, True):
            for value, type_name in invalid_values:
                with self.subTest(state=state, value_type=type_name):
                    nnpack.set_flags(state)
                    context = nnpack.flags(value)
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

        nnpack.set_flags(True)
        for call in (
            lambda: nnpack.flags(True, False),
            lambda: nnpack.flags(_enabled=True),
            lambda: nnpack.flags(False, enabled=True),
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()
                self.assertIs(torch._C._get_nnpack_enabled(), True)

    def test_context_is_a_reusable_decorator_factory(self):
        nnpack = self.nnpack
        observations = []

        @nnpack.flags(False)
        def decorated(value):
            observations.append(torch._C._get_nnpack_enabled())
            if value == "raise":
                raise _ContextBodyError("decorated body failed")
            return value

        self.assertEqual(decorated("first"), "first")
        self.assertIs(torch._C._get_nnpack_enabled(), True)
        self.assertEqual(decorated("second"), "second")
        self.assertIs(torch._C._get_nnpack_enabled(), True)
        with self.assertRaisesRegex(_ContextBodyError, "decorated body failed"):
            decorated("raise")
        self.assertEqual(observations, [False, False, False])
        self.assertIs(torch._C._get_nnpack_enabled(), True)

    def test_state_changes_are_process_global_across_threads(self):
        nnpack = self.nnpack
        worker_entered = threading.Event()
        main_context_exited = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                with nnpack.flags(False) as entered:
                    observations.append(
                        ("worker-enter", entered, torch._C._get_nnpack_enabled())
                    )
                    worker_entered.set()
                    if not main_context_exited.wait(timeout=10):
                        raise RuntimeError("timed out waiting for main context")
                    observations.append(
                        ("worker-resume", torch._C._get_nnpack_enabled())
                    )
                observations.append(
                    ("worker-exit", torch._C._get_nnpack_enabled())
                )
            except BaseException as error:
                errors.append(error)
                worker_entered.set()

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(worker_entered.wait(timeout=10))
            self.assertEqual(errors, [])
            self.assertIs(torch._C._get_nnpack_enabled(), False)
            with nnpack.flags(True) as entered:
                self.assertIsNone(entered)
                self.assertIs(torch._C._get_nnpack_enabled(), True)
            self.assertIs(torch._C._get_nnpack_enabled(), False)
        finally:
            main_context_exited.set()
            thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            observations,
            [
                ("worker-enter", None, False),
                ("worker-resume", False),
                ("worker-exit", True),
            ],
        )
        self.assertIs(torch._C._get_nnpack_enabled(), True)

    def test_metadata_imports_copying_and_pickling(self):
        nnpack = self.nnpack
        function = nnpack.flags
        wrapped = function.__wrapped__

        self.assertIs(torch.backends.nnpack, nnpack)
        self.assertIs(sys.modules["torch_rs.backends.nnpack"], nnpack)
        self.assertIs(type(nnpack), types.ModuleType)
        self.assertIsNone(nnpack.__doc__)
        self.assertEqual(
            nnpack.__all__,
            ["is_available", "flags", "set_flags"],
        )
        self.assertEqual(
            {name for name in vars(nnpack) if not name.startswith("_")},
            {
                "contextmanager",
                "flags",
                "is_available",
                "set_flags",
                "torch",
            },
        )

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
        self.assertEqual(function.__dict__, {"__wrapped__": wrapped})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_freevars, ("func",))
        self.assertEqual(function.__code__.co_cellvars, ())

        self.assertIs(type(wrapped), types.FunctionType)
        self.assertEqual(str(inspect.signature(wrapped)), "(enabled=False)")
        self.assertEqual(inspect.get_annotations(wrapped), {})
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
            with self.subTest(kind="function", protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.nnpack", payload)
                self.assertIs(pickle.loads(payload), function)

        context = function(False)
        self.assertEqual(type(context).__module__, "contextlib")
        self.assertEqual(type(context).__qualname__, "_GeneratorContextManager")
        self.assertEqual(context.__doc__, FLAGS_DOC)
        self.assertIs(context.func, wrapped)
        self.assertEqual(context.args, (False,))
        self.assertEqual(context.kwds, {})
        copied = copy.copy(context)
        self.assertIsNot(copied, context)
        self.assertIs(copied.gen, context.gen)
        with self.assertRaisesRegex(TypeError, "cannot pickle 'generator' object"):
            copy.deepcopy(context)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="context", protocol=protocol):
                with self.assertRaisesRegex(
                    TypeError,
                    "cannot pickle 'generator' object",
                ):
                    pickle.dumps(context, protocol=protocol)

    def test_reload_preserves_state_and_existing_contexts(self):
        nnpack = self.nnpack
        old_flags = nnpack.flags
        old_wrapped = old_flags.__wrapped__
        namespace = nnpack.__dict__
        active_context = old_flags(False)

        self.assertIsNone(active_context.__enter__())
        self.assertIs(torch._C._get_nnpack_enabled(), False)
        reloaded = importlib.reload(nnpack)

        self.assertIs(reloaded, nnpack)
        self.assertIs(nnpack.__dict__, namespace)
        self.assertIs(torch.backends.nnpack, nnpack)
        self.assertIs(sys.modules[nnpack.__name__], nnpack)
        self.assertIsNot(nnpack.flags, old_flags)
        self.assertIsNot(nnpack.flags.__wrapped__, old_wrapped)
        self.assertIs(torch._C._get_nnpack_enabled(), False)
        self.assertIs(active_context.__exit__(None, None, None), False)
        self.assertIs(torch._C._get_nnpack_enabled(), True)

        for function in (old_flags, nnpack.flags):
            with self.subTest(function=function):
                with function(False) as entered:
                    self.assertIsNone(entered)
                    self.assertIs(torch._C._get_nnpack_enabled(), False)
                self.assertIs(torch._C._get_nnpack_enabled(), True)

        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_flags)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function flags at 0x...>: it's not the same object "
            "as torch_rs.backends.nnpack.flags",
        )
        self.assertIs(pickle.loads(pickle.dumps(nnpack.flags)), nnpack.flags)

    def test_context_management_does_not_add_nnpack_execution(self):
        self.assertIs(self.nnpack.is_available(), False)
        with self.nnpack.flags(False):
            self.assertIs(self.nnpack.is_available(), False)
            self.assertFalse(hasattr(torch, "_nnpack_spatial_convolution"))
        self.assertIs(self.nnpack.is_available(), False)


if __name__ == "__main__":
    unittest.main()
