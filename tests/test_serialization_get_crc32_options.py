import contextlib
import copy
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


GETTER_DOC = """
    Get whether :func:`torch.save` computes and writes crc32 for each record.

    Defaults to ``True``.
    """

SETTER_DOC = """
    Set whether :func:`torch.save` computes and writes crc32 for each record.

    .. note::
        Setting this to ``False`` may make unzipping of the ``torch.save`` output
        fail or warn due to corrupted CRC32. However ``torch.load`` will be
        able to load the file.

    Args:
        compute_crc32 (bool): set crc32 computation flag
    """


class SerializationCrc32OptionsTests(unittest.TestCase):
    def setUp(self):
        self.original_crc32_option = torch.serialization.get_crc32_options()
        torch.serialization.set_crc32_options(True)

    def tearDown(self):
        torch.serialization.set_crc32_options(self.original_crc32_option)

    def test_default_true_is_exact_and_preserves_grad_mode(self):
        function = torch.serialization.get_crc32_options
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(), True)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

    def test_setter_stores_every_runtime_value_without_coercion(self):
        class RejectBoolConversion:
            def __bool__(self):
                raise AssertionError("set_crc32_options must not call bool")

        mutable = []
        marker = object()
        values = (
            False,
            True,
            None,
            0,
            1,
            2,
            0.0,
            float("nan"),
            "",
            "false",
            mutable,
            {},
            marker,
            RejectBoolConversion,
            RejectBoolConversion(),
        )
        for value in values:
            with self.subTest(type=type(value).__name__, value=repr(value)):
                self.assertIsNone(torch.serialization.set_crc32_options(value))
                self.assertIs(torch.serialization.get_crc32_options(), value)

        self.assertIsNone(
            torch.serialization.set_crc32_options(compute_crc32=mutable)
        )
        self.assertIs(torch.serialization.get_crc32_options(), mutable)
        mutable.append("updated")
        self.assertEqual(torch.serialization.get_crc32_options(), ["updated"])

    def test_updates_are_process_global_and_visible_across_threads(self):
        serialization = torch.serialization
        initial = object()
        updated = object()
        worker_ready = threading.Event()
        read_updated = threading.Event()
        observations = []
        errors = []

        serialization.set_crc32_options(initial)

        def observer():
            try:
                observations.append(serialization.get_crc32_options() is initial)
                worker_ready.set()
                if not read_updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for the updated state")
                observations.append(serialization.get_crc32_options() is updated)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=observer)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        serialization.set_crc32_options(updated)
        read_updated.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True])

        worker_value = object()
        worker_results = []

        def writer():
            worker_results.append(serialization.set_crc32_options(worker_value))
            worker_results.append(serialization.get_crc32_options() is worker_value)

        thread = threading.Thread(target=writer)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(worker_results, [None, True])
        self.assertIs(serialization.get_crc32_options(), worker_value)

    def test_state_is_stable_across_threads_and_grad_modes(self):
        function = torch.serialization.get_crc32_options
        value = object()
        torch.serialization.set_crc32_options(value)
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        function() is value,
                        torch.is_grad_enabled(),
                        function() is value,
                        torch.is_grad_enabled(),
                    )
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (
                    expected_grad_state,
                    True,
                    expected_grad_state,
                    True,
                    expected_grad_state,
                ),
            )

    def test_signature_annotations_documentation_and_module_identity(self):
        serialization = importlib.import_module("torch_rs.serialization")
        getter = serialization.get_crc32_options
        setter = serialization.set_crc32_options

        self.assertIs(torch.serialization, serialization)
        self.assertIs(sys.modules["torch_rs.serialization"], serialization)
        self.assertIsNone(serialization.__doc__)
        for function in (getter, setter):
            self.assertIs(type(function), types.FunctionType)
            self.assertEqual(function.__module__, "torch_rs.serialization")
            self.assertIs(inspect.getmodule(function), serialization)
            self.assertIsNone(function.__defaults__)
            self.assertIsNone(function.__kwdefaults__)
            self.assertEqual(function.__dict__, {})
            self.assertFalse(hasattr(function, "__text_signature__"))

        self.assertEqual(str(inspect.signature(getter)), "() -> bool")
        self.assertEqual(getter.__annotations__, {"return": bool})
        self.assertEqual(typing.get_type_hints(getter), {"return": bool})
        self.assertEqual(getter.__name__, "get_crc32_options")
        self.assertEqual(getter.__qualname__, "get_crc32_options")
        self.assertEqual(inspect.cleandoc(getter.__doc__), inspect.cleandoc(GETTER_DOC))

        self.assertEqual(str(inspect.signature(setter)), "(compute_crc32: bool)")
        self.assertEqual(setter.__annotations__, {"compute_crc32": bool})
        self.assertEqual(
            typing.get_type_hints(setter),
            {"compute_crc32": bool},
        )
        self.assertEqual(setter.__name__, "set_crc32_options")
        self.assertEqual(setter.__qualname__, "set_crc32_options")
        self.assertEqual(inspect.cleandoc(setter.__doc__), inspect.cleandoc(SETTER_DOC))

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        serialization = torch.serialization
        exports = {
            "check_module_version_greater_or_equal": (
                serialization.check_module_version_greater_or_equal
            ),
            "LoadEndianness": serialization.LoadEndianness,
            "get_crc32_options": serialization.get_crc32_options,
            "set_crc32_options": serialization.set_crc32_options,
            "get_default_load_endianness": (
                serialization.get_default_load_endianness
            ),
            "set_default_load_endianness": (
                serialization.set_default_load_endianness
            ),
            "get_default_mmap_options": serialization.get_default_mmap_options,
            "set_default_mmap_options": serialization.set_default_mmap_options,
        }

        self.assertEqual(serialization.__all__, list(exports))

        package_import = {}
        exec("from torch_rs import serialization", package_import)
        self.assertIs(package_import["serialization"], serialization)

        direct_import = {}
        exec(
            "from torch_rs.serialization import "
            "check_module_version_greater_or_equal, LoadEndianness, "
            "get_crc32_options, set_crc32_options, "
            "get_default_load_endianness, set_default_load_endianness, "
            "get_default_mmap_options, "
            "set_default_mmap_options",
            direct_import,
        )
        for name, value in exports.items():
            self.assertIs(direct_import[name], value)

        serialization_namespace = {}
        exec("from torch_rs.serialization import *", serialization_namespace)
        self.assertEqual(
            {
                name
                for name in serialization_namespace
                if not name.startswith("__")
            },
            set(exports),
        )
        for name, value in exports.items():
            self.assertIs(serialization_namespace[name], value)

        self.assertNotIn("serialization", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        for name in ("serialization", *exports):
            self.assertNotIn(name, torch.__all__)
            self.assertNotIn(name, top_level_namespace)

        for name, value in exports.items():
            self.assertIs(copy.copy(value), value)
            self.assertIs(copy.deepcopy(value), value)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIn(b"torch_rs.serialization", payload)
                    self.assertIs(pickle.loads(payload), value)

    def test_argument_errors_match_pytorch_2_13(self):
        getter = torch.serialization.get_crc32_options
        setter = torch.serialization.set_crc32_options
        cases = (
            (
                lambda: getter(None),
                "get_crc32_options() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: getter(None, None),
                "get_crc32_options() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: getter(enabled=True),
                "get_crc32_options() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: getter(None, enabled=True),
                "get_crc32_options() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: setter(),
                "set_crc32_options() missing 1 required positional argument: "
                "'compute_crc32'",
            ),
            (
                lambda: setter(True, False),
                "set_crc32_options() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: setter(enabled=True),
                "set_crc32_options() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: setter(True, compute_crc32=False),
                "set_crc32_options() got multiple values for argument "
                "'compute_crc32'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                before = getter()
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(getter(), before)

    def test_reload_preserves_the_process_state(self):
        serialization = torch.serialization
        value = object()
        serialization.set_crc32_options(value)

        self.assertIs(importlib.reload(serialization), serialization)
        self.assertIs(torch.serialization, serialization)
        self.assertIs(serialization.get_crc32_options(), value)

    def test_reimported_submodule_shares_state_with_existing_functions(self):
        original_module = torch.serialization
        original_getter = original_module.get_crc32_options
        original_setter = original_module.set_crc32_options
        module_name = original_module.__name__
        first_value = object()
        replacement_value = object()

        original_setter(first_value)
        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)

            self.assertIsNot(replacement_module, original_module)
            self.assertIs(sys.modules[module_name], replacement_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertIs(original_getter(), first_value)
            self.assertIs(replacement_module.get_crc32_options(), first_value)

            self.assertIsNone(
                replacement_module.set_crc32_options(replacement_value)
            )
            self.assertIs(original_getter(), replacement_value)
            self.assertIs(
                replacement_module.get_crc32_options(),
                replacement_value,
            )

            self.assertIsNone(original_setter(None))
            self.assertIsNone(original_getter())
            self.assertIsNone(replacement_module.get_crc32_options())
        finally:
            sys.modules[module_name] = original_module
            torch.serialization = original_module

    def test_save_load_and_other_serialization_apis_remain_unsupported(self):
        serialization = torch.serialization

        self.assertEqual(
            {name for name in vars(serialization) if not name.startswith("_")},
            {
                "check_module_version_greater_or_equal",
                "LoadEndianness",
                "get_crc32_options",
                "set_crc32_options",
                "get_default_load_endianness",
                "set_default_load_endianness",
                "get_default_mmap_options",
                "set_default_mmap_options",
            },
        )
        for name in ("save", "load"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(serialization, name))
                self.assertNotIn(name, serialization.__all__)

        for name in (
            "check_module_version_greater_or_equal",
            "get_crc32_options",
            "set_crc32_options",
            "LoadEndianness",
            "get_default_load_endianness",
            "set_default_load_endianness",
            "save",
            "load",
        ):
            with self.subTest(top_level_name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)

    def test_importing_reloading_and_calling_does_not_import_pytorch(self):
        script = r"""
import importlib
import mmap
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

serialization = torch.serialization
assert [member.value for member in serialization.LoadEndianness] == [1, 2, 3]
assert serialization.get_default_load_endianness() is None
assert serialization.get_crc32_options() is True
assert serialization.set_crc32_options(False) is None
assert serialization.get_crc32_options() is False
assert importlib.import_module("torch_rs.serialization") is serialization
assert importlib.reload(serialization) is serialization
assert serialization.get_crc32_options() is False
old_getter = serialization.get_crc32_options
old_setter = serialization.set_crc32_options
del sys.modules["torch_rs.serialization"]
replacement = importlib.import_module("torch_rs.serialization")
assert replacement is not serialization
assert torch.serialization is replacement
assert old_getter() is False
assert replacement.get_crc32_options() is False
assert replacement.set_crc32_options("replacement") is None
assert old_getter() == "replacement"
assert old_setter(None) is None
assert replacement.get_crc32_options() is None
assert (
    serialization.get_default_mmap_options()
    == replacement.get_default_mmap_options()
)
assert serialization.__all__ == [
    "check_module_version_greater_or_equal",
    "LoadEndianness",
    "get_crc32_options",
    "set_crc32_options",
    "get_default_load_endianness",
    "set_default_load_endianness",
    "get_default_mmap_options",
    "set_default_mmap_options",
]
assert replacement.__all__ == [
    "check_module_version_greater_or_equal",
    "LoadEndianness",
    "get_crc32_options",
    "set_crc32_options",
    "get_default_load_endianness",
    "set_default_load_endianness",
    "get_default_mmap_options",
    "set_default_mmap_options",
]
assert replacement.get_default_load_endianness() is None
assert hasattr(replacement, "set_default_load_endianness")
if hasattr(mmap, "MAP_PRIVATE") and hasattr(mmap, "MAP_SHARED"):
    replacement.set_default_mmap_options(mmap.MAP_SHARED)
    assert old_getter() is None
    assert replacement.get_default_mmap_options() == mmap.MAP_SHARED
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
