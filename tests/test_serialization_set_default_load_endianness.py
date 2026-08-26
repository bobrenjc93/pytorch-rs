import copy
import enum
import importlib
import inspect
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


SETTER_DOC = """
    Set fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Args:
        endianness: the new fallback byte order
    """


class SerializationSetDefaultLoadEndiannessTests(unittest.TestCase):
    def setUp(self):
        self.serialization = torch.serialization
        self.original = self.serialization.get_default_load_endianness()
        self.serialization.set_default_load_endianness(None)

    def tearDown(self):
        self.serialization._state.default_load_endianness = self.original

    def test_none_and_current_enum_members_update_the_getter_by_identity(self):
        setter = self.serialization.set_default_load_endianness
        getter = self.serialization.get_default_load_endianness

        for value in (None, *self.serialization.LoadEndianness):
            with self.subTest(value=value):
                self.assertIsNone(setter(value))
                self.assertIs(getter(), value)

        value = self.serialization.LoadEndianness.LITTLE
        self.assertIsNone(setter(endianness=value))
        self.assertIs(getter(), value)

    def test_updates_are_process_global_and_visible_across_threads(self):
        setter = self.serialization.set_default_load_endianness
        getter = self.serialization.get_default_load_endianness
        load_endianness = self.serialization.LoadEndianness
        worker_ready = threading.Event()
        read_updated = threading.Event()
        observations = []
        errors = []

        setter(load_endianness.NATIVE)

        def observer():
            try:
                observations.append(getter() is load_endianness.NATIVE)
                worker_ready.set()
                if not read_updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for the updated state")
                observations.append(getter() is load_endianness.LITTLE)
            except BaseException as error:
                errors.append(error)
                worker_ready.set()

        thread = threading.Thread(target=observer)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        setter(load_endianness.LITTLE)
        read_updated.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True])

        worker_results = []

        def writer():
            worker_results.append(setter(load_endianness.BIG))
            worker_results.append(getter() is load_endianness.BIG)

        thread = threading.Thread(target=writer)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(worker_results, [None, True])
        self.assertIs(getter(), load_endianness.BIG)

    def test_signature_metadata_documentation_and_module_identity(self):
        serialization = importlib.import_module("torch_rs.serialization")
        setter = serialization.set_default_load_endianness

        self.assertIs(torch.serialization, serialization)
        self.assertIs(sys.modules["torch_rs.serialization"], serialization)
        self.assertIsNone(serialization.__doc__)
        self.assertIs(type(setter), types.FunctionType)
        self.assertEqual(setter.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(setter), serialization)
        self.assertEqual(str(inspect.signature(setter)), "(endianness)")
        self.assertEqual(setter.__annotations__, {})
        self.assertEqual(typing.get_type_hints(setter), {})
        self.assertIsNone(setter.__defaults__)
        self.assertIsNone(setter.__kwdefaults__)
        self.assertEqual(setter.__dict__, {})
        self.assertFalse(hasattr(setter, "__text_signature__"))
        self.assertEqual(setter.__name__, "set_default_load_endianness")
        self.assertEqual(setter.__qualname__, "set_default_load_endianness")
        self.assertEqual(setter.__code__.co_freevars, ())
        self.assertEqual(setter.__code__.co_cellvars, ())
        self.assertEqual(
            inspect.cleandoc(setter.__doc__),
            inspect.cleandoc(SETTER_DOC),
        )

    def test_imports_exports_copy_and_pickle_use_the_canonical_function(self):
        serialization = self.serialization
        setter = serialization.set_default_load_endianness
        exported_names = [
            "check_module_version_greater_or_equal",
            "LoadEndianness",
            "get_crc32_options",
            "set_crc32_options",
            "get_default_load_endianness",
            "set_default_load_endianness",
            "get_default_mmap_options",
            "set_default_mmap_options",
        ]

        self.assertEqual(serialization.__all__, exported_names)
        self.assertEqual(serialization.__all__.count(setter.__name__), 1)

        package_import = {}
        exec("from torch_rs import serialization", package_import)
        self.assertIs(package_import["serialization"], serialization)

        direct_import = {}
        exec(
            "from torch_rs.serialization import set_default_load_endianness",
            direct_import,
        )
        self.assertIs(direct_import[setter.__name__], setter)

        namespace = {}
        exec("from torch_rs.serialization import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")},
            set(exported_names),
        )
        self.assertIs(namespace[setter.__name__], setter)

        self.assertFalse(hasattr(torch, setter.__name__))
        self.assertNotIn(setter.__name__, torch.__all__)
        self.assertIs(copy.copy(setter), setter)
        self.assertIs(copy.deepcopy(setter), setter)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(setter, protocol=protocol)
                self.assertIn(b"torch_rs.serialization", payload)
                self.assertIs(pickle.loads(payload), setter)

    def test_invalid_types_and_call_errors_preserve_the_current_state(self):
        setter = self.serialization.set_default_load_endianness
        getter = self.serialization.get_default_load_endianness
        load_endianness = self.serialization.LoadEndianness
        other_endianness = enum.Enum(
            "LoadEndianness",
            {"NATIVE": 1, "LITTLE": 2, "BIG": 3},
        )
        setter(load_endianness.LITTLE)

        invalid_values = (
            1,
            2,
            3,
            True,
            False,
            "native",
            "little",
            "big",
            other_endianness.NATIVE,
            load_endianness,
            object(),
        )
        message = "Invalid argument type in function set_default_load_endianness"
        for value in invalid_values:
            with self.subTest(value=repr(value)):
                before = getter()
                with self.assertRaises(TypeError) as raised:
                    setter(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(getter(), before)

        call_errors = (
            (
                lambda: setter(),
                "set_default_load_endianness() missing 1 required positional "
                "argument: 'endianness'",
            ),
            (
                lambda: setter(None, None),
                "set_default_load_endianness() takes 1 positional argument but 2 "
                "were given",
            ),
            (
                lambda: setter(enabled=True),
                "set_default_load_endianness() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: setter(None, endianness=load_endianness.NATIVE),
                "set_default_load_endianness() got multiple values for argument "
                "'endianness'",
            ),
        )
        for call, expected_message in call_errors:
            with self.subTest(message=expected_message):
                before = getter()
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), expected_message)
                self.assertEqual(raised.exception.args, (expected_message,))
                self.assertIs(getter(), before)

    def test_reload_and_reimport_preserve_state_and_reject_stale_members(self):
        original_module = self.serialization
        original_class = original_module.LoadEndianness
        original_getter = original_module.get_default_load_endianness
        original_setter = original_module.set_default_load_endianness
        module_name = original_module.__name__

        original_setter(original_class.LITTLE)
        self.assertIs(importlib.reload(original_module), original_module)
        reloaded_class = original_module.LoadEndianness
        reloaded_getter = original_module.get_default_load_endianness
        reloaded_setter = original_module.set_default_load_endianness

        self.assertIsNot(reloaded_class, original_class)
        self.assertIs(original_getter(), original_class.LITTLE)
        self.assertIs(reloaded_getter(), original_class.LITTLE)
        with self.assertRaisesRegex(
            TypeError,
            "^Invalid argument type in function set_default_load_endianness$",
        ):
            original_setter(original_class.BIG)
        self.assertIsNone(original_setter(reloaded_class.BIG))
        self.assertIs(reloaded_getter(), reloaded_class.BIG)

        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)
            replacement_class = replacement_module.LoadEndianness

            self.assertIsNot(replacement_module, original_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertIsNot(replacement_class, reloaded_class)
            self.assertIs(
                replacement_module.get_default_load_endianness(),
                reloaded_class.BIG,
            )

            for setter, stale_member in (
                (original_setter, replacement_class.NATIVE),
                (
                    replacement_module.set_default_load_endianness,
                    reloaded_class.NATIVE,
                ),
            ):
                with self.assertRaisesRegex(
                    TypeError,
                    "^Invalid argument type in function "
                    "set_default_load_endianness$",
                ):
                    setter(stale_member)
                self.assertIs(
                    replacement_module.get_default_load_endianness(),
                    reloaded_class.BIG,
                )

            self.assertIsNone(
                replacement_module.set_default_load_endianness(
                    replacement_class.NATIVE
                )
            )
            self.assertIs(original_getter(), replacement_class.NATIVE)
            self.assertIs(reloaded_getter(), replacement_class.NATIVE)

            self.assertIsNone(original_setter(reloaded_class.LITTLE))
            self.assertIs(
                replacement_module.get_default_load_endianness(),
                reloaded_class.LITTLE,
            )
        finally:
            sys.modules[module_name] = original_module
            torch.serialization = original_module

    def test_save_and_load_remain_unsupported(self):
        for name in ("save", "load"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(self.serialization, name))
                self.assertNotIn(name, self.serialization.__all__)

    def test_importing_reloading_and_calling_does_not_import_pytorch(self):
        script = r"""
import importlib
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

serialization = torch.serialization
load_endianness = serialization.LoadEndianness
getter = serialization.get_default_load_endianness
setter = serialization.set_default_load_endianness
assert setter(load_endianness.LITTLE) is None
assert getter() is load_endianness.LITTLE
try:
    setter(2)
except TypeError as error:
    assert str(error) == "Invalid argument type in function set_default_load_endianness"
else:
    raise AssertionError("integer enum values must be rejected")
assert getter() is load_endianness.LITTLE

assert importlib.reload(serialization) is serialization
reloaded_class = serialization.LoadEndianness
assert reloaded_class is not load_endianness
assert getter() is load_endianness.LITTLE
try:
    setter(load_endianness.BIG)
except TypeError:
    pass
else:
    raise AssertionError("members from before reload must be rejected")
assert setter(reloaded_class.BIG) is None
assert getter() is reloaded_class.BIG

del sys.modules["torch_rs.serialization"]
replacement = importlib.import_module("torch_rs.serialization")
assert replacement is not serialization
assert torch.serialization is replacement
assert replacement.get_default_load_endianness() is reloaded_class.BIG
try:
    replacement.set_default_load_endianness(reloaded_class.NATIVE)
except TypeError:
    pass
else:
    raise AssertionError("members from another module instance must be rejected")
assert replacement.set_default_load_endianness(None) is None
assert getter() is None
assert not hasattr(replacement, "save")
assert not hasattr(replacement, "load")
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
