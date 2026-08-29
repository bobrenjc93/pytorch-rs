import copy
import importlib
import inspect
import pickle
import re
import sys
import threading
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


GET_FASTPATH_DOC = (
    "Returns whether fast path for TransformerEncoder and MultiHeadAttention\n"
    "    is enabled, or ``True`` if jit is scripting.\n"
    "\n"
    "    .. note::\n"
    "        The fastpath might not be run even if ``get_fastpath_enabled`` returns\n"
    "        ``True`` unless all conditions on inputs are met.\n"
    "    "
)
SET_FASTPATH_DOC = "Sets whether fast path is enabled"


class MhaFastpathTests(unittest.TestCase):
    def setUp(self):
        self.mha = importlib.reload(
            importlib.import_module("torch_rs.backends.mha")
        )

    def tearDown(self):
        importlib.reload(self.mha)

    def test_default_setter_identity_and_scripting_override(self):
        mha = self.mha

        self.assertIs(mha.get_fastpath_enabled(), True)
        self.assertIs(mha._is_fastpath_enabled, True)

        values = (False, None, 0, [], object())
        for value in values:
            with self.subTest(value_type=type(value).__name__):
                self.assertIsNone(mha.set_fastpath_enabled(value))
                self.assertIs(mha._is_fastpath_enabled, value)
                self.assertIs(mha.get_fastpath_enabled(), value)

        stored = object()
        self.assertIsNone(mha.set_fastpath_enabled(stored))
        with mock.patch.object(torch.jit, "is_scripting", return_value=True) as probe:
            self.assertIs(mha.get_fastpath_enabled(), True)
            probe.assert_called_once_with()
        self.assertIs(mha._is_fastpath_enabled, stored)
        self.assertIs(mha.get_fastpath_enabled(), stored)

    def test_state_is_shared_and_visible_across_threads(self):
        mha = self.mha
        initial = object()
        worker_value = object()
        main_value = object()
        worker_written = threading.Event()
        main_written = threading.Event()
        outcomes = {}
        errors = []

        mha.set_fastpath_enabled(initial)

        def worker():
            try:
                outcomes["initial"] = mha.get_fastpath_enabled()
                outcomes["setter_result"] = mha.set_fastpath_enabled(worker_value)
                worker_written.set()
                if not main_written.wait(timeout=10):
                    raise TimeoutError("main thread did not publish its state")
                outcomes["final"] = mha.get_fastpath_enabled()
            except BaseException as error:
                errors.append(error)
                worker_written.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_written.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(mha.get_fastpath_enabled(), worker_value)
        self.assertIsNone(mha.set_fastpath_enabled(main_value))
        main_written.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertIs(outcomes["initial"], initial)
        self.assertIsNone(outcomes["setter_result"])
        self.assertIs(outcomes["final"], main_value)

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        mha = self.mha
        get_fastpath_enabled = mha.get_fastpath_enabled
        set_fastpath_enabled = mha.set_fastpath_enabled

        self.assertIs(torch.backends.mha, mha)
        self.assertIs(sys.modules["torch_rs.backends.mha"], mha)
        self.assertIs(type(mha), types.ModuleType)
        self.assertIsNone(mha.__doc__)
        self.assertFalse(hasattr(mha, "__all__"))
        self.assertEqual(
            {name for name in vars(mha) if not name.startswith("_")},
            {"get_fastpath_enabled", "set_fastpath_enabled", "torch"},
        )
        self.assertEqual(mha.__annotations__, {"_is_fastpath_enabled": bool})
        self.assertIs(mha.torch, torch)

        expected = (
            (
                get_fastpath_enabled,
                "get_fastpath_enabled",
                "() -> bool",
                {"return": bool},
                GET_FASTPATH_DOC,
                ("torch", "jit", "is_scripting", "_is_fastpath_enabled"),
            ),
            (
                set_fastpath_enabled,
                "set_fastpath_enabled",
                "(value: bool) -> None",
                {"value": bool, "return": None},
                SET_FASTPATH_DOC,
                ("_is_fastpath_enabled",),
            ),
        )
        for function, name, signature, annotations, doc, code_names in expected:
            with self.subTest(function=name):
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(function.__annotations__, annotations)
                self.assertEqual(inspect.get_annotations(function), annotations)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, "torch_rs.backends.mha")
                self.assertIs(inspect.getmodule(function), mha)
                self.assertEqual(
                    inspect.cleandoc(function.__doc__),
                    inspect.cleandoc(doc),
                )
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))
                self.assertEqual(function.__code__.co_names, code_names)
                self.assertEqual(function.__code__.co_freevars, ())
                self.assertEqual(function.__code__.co_cellvars, ())

        self.assertEqual(
            typing.get_type_hints(get_fastpath_enabled),
            {"return": bool},
        )
        self.assertEqual(
            typing.get_type_hints(set_fastpath_enabled),
            {"value": bool, "return": type(None)},
        )

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        mha = self.mha
        functions = {
            "get_fastpath_enabled": mha.get_fastpath_enabled,
            "set_fastpath_enabled": mha.set_fastpath_enabled,
        }

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.mha, mha)
        self.assertEqual(
            {name for name in vars(backends) if not name.startswith("_")},
            {
                "cpu",
                "cuda",
                "cusparselt",
                "cudnn",
                "kleidiai",
                "m",
                "mha",
                "mkl",
                "nnpack",
                "openmp",
            },
        )

        package_import = {}
        backend_import = {}
        function_imports = {}
        parent_wildcard = {}
        child_wildcard = {}
        exec("from torch_rs import backends", package_import)
        exec("from torch_rs.backends import mha", backend_import)
        for name in functions:
            namespace = {}
            exec(f"from torch_rs.backends.mha import {name}", namespace)
            function_imports[name] = namespace[name]
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.mha import *", child_wildcard)

        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["mha"], mha)
        self.assertIs(parent_wildcard["mha"], mha)
        for name, function in functions.items():
            self.assertIs(function_imports[name], function)
            self.assertIs(child_wildcard[name], function)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"get_fastpath_enabled", "set_fastpath_enabled", "torch"},
        )
        self.assertIs(child_wildcard["torch"], torch)

        self.assertNotIn("backends", torch.__all__)
        self.assertNotIn("mha", torch.__all__)
        self.assertFalse(hasattr(torch, "mha"))
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("backends", top_level_wildcard)
        self.assertNotIn("mha", top_level_wildcard)

        for name, function in functions.items():
            with self.subTest(function=name):
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.backends.mha", payload)
                    self.assertIs(pickle.loads(payload), function)

    def test_reload_resets_state_and_replaces_functions(self):
        mha = self.mha
        stored = object()
        old_functions = {
            "get_fastpath_enabled": mha.get_fastpath_enabled,
            "set_fastpath_enabled": mha.set_fastpath_enabled,
        }
        namespace = mha.__dict__
        mha.set_fastpath_enabled(stored)

        reloaded = importlib.reload(mha)

        self.assertIs(reloaded, mha)
        self.assertIs(mha.__dict__, namespace)
        self.assertIs(torch.backends.mha, mha)
        self.assertIs(sys.modules[mha.__name__], mha)
        self.assertIs(mha._is_fastpath_enabled, True)
        self.assertIs(mha.get_fastpath_enabled(), True)
        for name, old_function in old_functions.items():
            new_function = getattr(mha, name)
            self.assertIsNot(new_function, old_function)
            self.assertIs(copy.copy(new_function), new_function)
            self.assertIs(copy.deepcopy(new_function), new_function)
            self.assertIs(pickle.loads(pickle.dumps(new_function)), new_function)
            with self.assertRaises(pickle.PicklingError) as raised:
                pickle.dumps(old_function)
            message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
            self.assertEqual(
                message,
                f"Can't pickle <function {name} at 0x...>: "
                "it's not the same object as "
                f"torch_rs.backends.mha.{name}",
            )

    def test_argument_forms_and_errors_match_pytorch_2_13(self):
        mha = self.mha
        value = object()
        self.assertIsNone(mha.set_fastpath_enabled(value=value))
        self.assertIs(mha.get_fastpath_enabled(), value)

        cases = (
            (
                lambda: mha.get_fastpath_enabled(None),
                "get_fastpath_enabled() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: mha.get_fastpath_enabled(enabled=True),
                "get_fastpath_enabled() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: mha.set_fastpath_enabled(),
                "set_fastpath_enabled() missing 1 required positional "
                "argument: 'value'",
            ),
            (
                lambda: mha.set_fastpath_enabled(True, False),
                "set_fastpath_enabled() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: mha.set_fastpath_enabled(enabled=True),
                "set_fastpath_enabled() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: mha.set_fastpath_enabled(True, value=False),
                "set_fastpath_enabled() got multiple values for argument 'value'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_attention_kernels_and_transformer_modules_remain_unsupported(self):
        self.assertIs(self.mha.get_fastpath_enabled(), True)
        for name in (
            "MultiheadAttention",
            "Transformer",
            "TransformerDecoder",
            "TransformerEncoder",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.nn, name))

        self.assertFalse(
            hasattr(torch.nn.functional, "multi_head_attention_forward")
        )
        self.assertFalse(hasattr(torch, "_native_multi_head_attention"))
        for module_name in (
            "torch_rs.nn.modules.activation",
            "torch_rs.nn.modules.transformer",
        ):
            with self.subTest(module=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
