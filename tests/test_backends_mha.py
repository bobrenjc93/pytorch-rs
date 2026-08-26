import copy
import importlib
import inspect
import pickle
import re
import subprocess
import sys
import threading
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


GETTER_DOC = """Returns whether fast path for TransformerEncoder and MultiHeadAttention
    is enabled, or ``True`` if jit is scripting.

    .. note::
        The fastpath might not be run even if ``get_fastpath_enabled`` returns
        ``True`` unless all conditions on inputs are met.
    """
SETTER_DOC = "Sets whether fast path is enabled"


class _ExplodingTruth:
    def __bool__(self):
        raise AssertionError("the stored fastpath value was coerced to bool")


class MhaFastpathTests(unittest.TestCase):
    def setUp(self):
        self.original_value = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(True)

    def tearDown(self):
        torch.backends.mha.set_fastpath_enabled(self.original_value)

    def test_default_and_setter_preserve_exact_objects(self):
        mha = torch.backends.mha
        self.assertIs(mha.get_fastpath_enabled(), True)

        mutable = []
        values = (False, None, 0, "", mutable, {}, object(), _ExplodingTruth())
        for value in values:
            with self.subTest(value=value):
                self.assertIs(mha.set_fastpath_enabled(value), None)
                self.assertIs(mha.get_fastpath_enabled(), value)

        mutable.append("updated")
        mha.set_fastpath_enabled(mutable)
        self.assertIs(mha.get_fastpath_enabled(), mutable)
        self.assertEqual(mha.get_fastpath_enabled(), ["updated"])

        marker = object()
        self.assertIs(mha.set_fastpath_enabled(value=marker), None)
        self.assertIs(mha.get_fastpath_enabled(), marker)

    def test_scripting_always_reports_true_without_changing_stored_value(self):
        mha = torch.backends.mha
        first = object()
        second = object()
        mha.set_fastpath_enabled(first)

        with mock.patch.object(torch.jit, "is_scripting", return_value=True) as probe:
            self.assertIs(mha.get_fastpath_enabled(), True)
            self.assertIs(mha.set_fastpath_enabled(second), None)
            self.assertIs(mha.get_fastpath_enabled(), True)
            self.assertEqual(probe.call_count, 2)

        self.assertIs(mha.get_fastpath_enabled(), second)

    def test_updates_are_visible_across_threads(self):
        mha = torch.backends.mha
        initial = object()
        updated = object()
        worker_value = object()
        ready = threading.Event()
        continue_reading = threading.Event()
        observations = []
        errors = []

        mha.set_fastpath_enabled(initial)

        def observer():
            try:
                observations.append(mha.get_fastpath_enabled() is initial)
                ready.set()
                if not continue_reading.wait(timeout=10):
                    raise RuntimeError("timed out waiting for the fastpath update")
                observations.append(mha.get_fastpath_enabled() is updated)
                observations.append(mha.set_fastpath_enabled(worker_value) is None)
                observations.append(mha.get_fastpath_enabled() is worker_value)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=observer)
        thread.start()
        self.assertTrue(ready.wait(timeout=10))
        self.assertIs(mha.set_fastpath_enabled(updated), None)
        continue_reading.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True, True, True])
        self.assertIs(mha.get_fastpath_enabled(), worker_value)

    def test_signature_annotations_documentation_and_module_identity(self):
        mha = importlib.import_module("torch_rs.backends.mha")
        getter = mha.get_fastpath_enabled
        setter = mha.set_fastpath_enabled

        self.assertIs(torch.backends.mha, mha)
        self.assertIs(sys.modules["torch_rs.backends.mha"], mha)
        self.assertIsNone(mha.__doc__)
        self.assertFalse(hasattr(mha, "__all__"))
        self.assertEqual(
            {name for name in vars(mha) if not name.startswith("_")},
            {"get_fastpath_enabled", "set_fastpath_enabled", "torch"},
        )
        self.assertIs(mha.torch, torch)

        expected = {
            "get_fastpath_enabled": (
                inspect.Signature(return_annotation=bool),
                {"return": bool},
                {"return": bool},
                GETTER_DOC,
                ("torch", "jit", "is_scripting", "_is_fastpath_enabled"),
            ),
            "set_fastpath_enabled": (
                inspect.Signature(
                    [
                        inspect.Parameter(
                            "value",
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=bool,
                        )
                    ],
                    return_annotation=None,
                ),
                {"value": bool, "return": None},
                {"value": bool, "return": type(None)},
                SETTER_DOC,
                ("_is_fastpath_enabled",),
            ),
        }
        for function in (getter, setter):
            with self.subTest(function=function.__name__):
                signature, annotations, hints, doc, names = expected[function.__name__]
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(inspect.signature(function), signature)
                self.assertEqual(inspect.get_annotations(function), annotations)
                self.assertEqual(typing.get_type_hints(function), hints)
                self.assertEqual(function.__qualname__, function.__name__)
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
                self.assertEqual(function.__code__.co_names, names)
                self.assertEqual(function.__code__.co_freevars, ())
                self.assertEqual(function.__code__.co_cellvars, ())

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        mha = importlib.import_module("torch_rs.backends.mha")
        functions = (mha.get_fastpath_enabled, mha.set_fastpath_enabled)

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.mha, mha)
        self.assertEqual(
            {name for name in vars(backends) if not name.startswith("_")},
            {"cuda", "cudnn", "mha", "mkl", "nnpack", "openmp"},
        )

        package_import = {}
        backend_import = {}
        function_import = {}
        parent_wildcard = {}
        child_wildcard = {}
        exec("from torch_rs import backends", package_import)
        exec("from torch_rs.backends import mha", backend_import)
        exec(
            "from torch_rs.backends.mha import "
            "get_fastpath_enabled, set_fastpath_enabled",
            function_import,
        )
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.mha import *", child_wildcard)
        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["mha"], mha)
        self.assertIs(function_import["get_fastpath_enabled"], functions[0])
        self.assertIs(function_import["set_fastpath_enabled"], functions[1])
        self.assertIs(parent_wildcard["mha"], mha)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"get_fastpath_enabled", "set_fastpath_enabled", "torch"},
        )
        self.assertIs(child_wildcard["torch"], torch)

        self.assertNotIn("backends", torch.__all__)
        self.assertFalse(hasattr(torch, "mha"))
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("backends", top_level_wildcard)
        self.assertNotIn("mha", top_level_wildcard)

        for function in functions:
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function.__name__, protocol=protocol):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.backends.mha", payload)
                    self.assertIs(pickle.loads(payload), function)

    def test_reload_resets_state_and_replaces_callables(self):
        backends = torch.backends
        mha = backends.mha
        namespace = mha.__dict__
        old_getter = mha.get_fastpath_enabled
        old_setter = mha.set_fastpath_enabled
        first = object()
        second = object()
        mha.set_fastpath_enabled(first)

        reloaded = importlib.reload(mha)

        self.assertIs(reloaded, mha)
        self.assertIs(mha.__dict__, namespace)
        self.assertIs(backends.mha, mha)
        self.assertIs(sys.modules[mha.__name__], mha)
        self.assertIsNot(mha.get_fastpath_enabled, old_getter)
        self.assertIsNot(mha.set_fastpath_enabled, old_setter)
        self.assertIs(mha._is_fastpath_enabled, True)
        self.assertIs(mha.get_fastpath_enabled(), True)
        self.assertIs(old_getter(), True)

        self.assertIs(old_setter(first), None)
        self.assertIs(mha.get_fastpath_enabled(), first)
        self.assertIs(mha.set_fastpath_enabled(second), None)
        self.assertIs(old_getter(), second)

        for function in (mha.get_fastpath_enabled, mha.set_fastpath_enabled):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            self.assertIs(pickle.loads(pickle.dumps(function)), function)
        for function in (old_getter, old_setter):
            with self.assertRaises(pickle.PicklingError) as raised:
                pickle.dumps(function)
            message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
            self.assertEqual(
                message,
                f"Can't pickle <function {function.__name__} at 0x...>: "
                "it's not the same object as "
                f"torch_rs.backends.mha.{function.__name__}",
            )

    def test_call_shape_errors_preserve_state(self):
        mha = torch.backends.mha
        getter = mha.get_fastpath_enabled
        setter = mha.set_fastpath_enabled
        marker = object()
        setter(marker)
        cases = (
            (
                lambda: getter(None),
                "get_fastpath_enabled() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: getter(enabled=True),
                "get_fastpath_enabled() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: setter(),
                "set_fastpath_enabled() missing 1 required positional argument: "
                "'value'",
            ),
            (
                lambda: setter(None, None),
                "set_fastpath_enabled() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: setter(enabled=True),
                "set_fastpath_enabled() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: setter(None, value=True),
                "set_fastpath_enabled() got multiple values for argument 'value'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(getter(), marker)

    def test_attention_and_transformer_execution_remain_unsupported(self):
        mha = torch.backends.mha
        mha.set_fastpath_enabled(False)
        self.assertIs(mha.get_fastpath_enabled(), False)

        for name in (
            "MultiheadAttention",
            "Transformer",
            "TransformerDecoder",
            "TransformerDecoderLayer",
            "TransformerEncoder",
            "TransformerEncoderLayer",
        ):
            with self.subTest(module_name=name):
                self.assertFalse(hasattr(torch.nn, name))

        for name in (
            "multi_head_attention_forward",
            "scaled_dot_product_attention",
        ):
            with self.subTest(functional_name=name):
                self.assertFalse(hasattr(torch.nn.functional, name))

        for name in ("_native_multi_head_attention", "_transformer_encoder_layer_fwd"):
            with self.subTest(native_name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertFalse(hasattr(torch._C, name))

        for module_name in (
            "torch_rs.nn.modules.activation",
            "torch_rs.nn.modules.transformer",
        ):
            with self.subTest(import_name=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)

    def test_importing_and_calling_does_not_import_pytorch_or_attention_code(self):
        script = r'''
import importlib
import sys

class RejectExternalImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"external PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalImport())
import torch_rs as torch
from torch_rs.backends import mha
from torch_rs.backends.mha import get_fastpath_enabled, set_fastpath_enabled

assert torch.backends.mha is mha
assert mha.get_fastpath_enabled is get_fastpath_enabled
assert mha.set_fastpath_enabled is set_fastpath_enabled
assert get_fastpath_enabled() is True
marker = object()
assert set_fastpath_enabled(marker) is None
assert get_fastpath_enabled() is marker
torch.jit.is_scripting = lambda: True
assert get_fastpath_enabled() is True
assert importlib.reload(mha) is mha
assert get_fastpath_enabled() is True
assert not hasattr(torch.nn, "MultiheadAttention")
assert not hasattr(torch.nn, "Transformer")
assert not hasattr(torch.nn.functional, "multi_head_attention_forward")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
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


if __name__ == "__main__":
    unittest.main()
