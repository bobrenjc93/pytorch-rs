import copy
import importlib
import inspect
import mmap
import pickle
import sys
import threading
import typing
import unittest
from unittest import mock

import torch_rs as torch


SETTER_DOC = """
    Context manager or function to set default mmap options for :func:`torch.load` with ``mmap=True`` to flags.

    For now, only either ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED`` are supported.
    Please open an issue if you need any other option to be added here.

    .. note::
        This feature is currently not supported for Windows.

    Args:
        flags: ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED``
    """

HAS_MMAP_FLAGS = (
    sys.platform != "win32"
    and hasattr(mmap, "MAP_PRIVATE")
    and hasattr(mmap, "MAP_SHARED")
)


class SerializationSetDefaultMmapOptionsTests(unittest.TestCase):
    def setUp(self):
        self.original = torch.serialization.get_default_mmap_options()
        if HAS_MMAP_FLAGS:
            torch.serialization.set_default_mmap_options(mmap.MAP_PRIVATE)

    def tearDown(self):
        if HAS_MMAP_FLAGS:
            torch.serialization.set_default_mmap_options(self.original)

    @unittest.skipUnless(HAS_MMAP_FLAGS, "requires POSIX mmap flags")
    def test_calls_update_the_process_global_state_immediately(self):
        serialization = torch.serialization
        setter = serialization.set_default_mmap_options

        context = setter(mmap.MAP_SHARED)
        self.assertIs(type(context), setter)
        self.assertEqual(context.prev, mmap.MAP_PRIVATE)
        self.assertEqual(serialization.get_default_mmap_options(), mmap.MAP_SHARED)

        replacement = setter(flags=mmap.MAP_PRIVATE)
        self.assertIs(type(replacement), setter)
        self.assertEqual(replacement.prev, mmap.MAP_SHARED)
        self.assertEqual(serialization.get_default_mmap_options(), mmap.MAP_PRIVATE)

        equivalent_flag = float(mmap.MAP_SHARED)
        setter(equivalent_flag)
        self.assertIs(serialization.get_default_mmap_options(), equivalent_flag)

    @unittest.skipUnless(HAS_MMAP_FLAGS, "requires POSIX mmap flags")
    def test_nested_contexts_and_exceptional_exits_restore_prior_state(self):
        serialization = torch.serialization
        setter = serialization.set_default_mmap_options
        getter = serialization.get_default_mmap_options

        outer = setter(mmap.MAP_SHARED)
        self.assertEqual(getter(), mmap.MAP_SHARED)
        with outer as outer_value:
            self.assertIsNone(outer_value)
            self.assertEqual(getter(), mmap.MAP_SHARED)
            with setter(mmap.MAP_PRIVATE) as inner_value:
                self.assertIsNone(inner_value)
                self.assertEqual(getter(), mmap.MAP_PRIVATE)
            self.assertEqual(getter(), mmap.MAP_SHARED)
        self.assertEqual(getter(), mmap.MAP_PRIVATE)

        marker = RuntimeError("context body failed")
        with self.assertRaises(RuntimeError) as raised:
            with setter(mmap.MAP_SHARED):
                self.assertEqual(getter(), mmap.MAP_SHARED)
                raise marker
        self.assertIs(raised.exception, marker)
        self.assertEqual(getter(), mmap.MAP_PRIVATE)

    @unittest.skipUnless(HAS_MMAP_FLAGS, "requires POSIX mmap flags")
    def test_updates_from_either_thread_are_visible_process_wide(self):
        serialization = torch.serialization
        setter = serialization.set_default_mmap_options
        getter = serialization.get_default_mmap_options
        worker_ready = threading.Event()
        read_shared = threading.Event()
        worker_finished = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(getter() == mmap.MAP_PRIVATE)
                worker_ready.set()
                if not read_shared.wait(timeout=10):
                    raise RuntimeError("timed out waiting for MAP_SHARED")
                observations.append(getter() == mmap.MAP_SHARED)
                setter(mmap.MAP_PRIVATE)
                worker_finished.set()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        setter(mmap.MAP_SHARED)
        read_shared.set()
        self.assertTrue(worker_finished.wait(timeout=10))
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True])
        self.assertEqual(getter(), mmap.MAP_PRIVATE)

    @unittest.skipUnless(HAS_MMAP_FLAGS, "requires POSIX mmap flags")
    def test_invalid_flags_raise_without_changing_state(self):
        serialization = torch.serialization
        setter = serialization.set_default_mmap_options
        getter = serialization.get_default_mmap_options

        for value in (None, False, 0, "shared"):
            with self.subTest(value=value):
                before = getter()
                message = (
                    "Invalid argument in function set_default_mmap_options, "
                    "expected mmap.MAP_PRIVATE or mmap.MAP_SHARED, but got "
                    f"{value}"
                )
                with self.assertRaises(ValueError) as raised:
                    setter(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(getter(), before)

    def test_windows_check_precedes_flag_validation(self):
        serialization = torch.serialization
        getter = serialization.get_default_mmap_options
        message = (
            "Changing the default mmap options is currently not supported for Windows"
        )

        with mock.patch.object(serialization, "_IS_WINDOWS", True):
            for value in (getattr(mmap, "MAP_PRIVATE", None), None, "invalid"):
                with self.subTest(value=value):
                    before = getter()
                    with self.assertRaises(RuntimeError) as raised:
                        serialization.set_default_mmap_options(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(getter(), before)

    def test_signature_annotations_documentation_and_module_identity(self):
        serialization = importlib.import_module("torch_rs.serialization")
        setter = serialization.set_default_mmap_options

        self.assertIs(torch.serialization, serialization)
        self.assertIs(sys.modules["torch_rs.serialization"], serialization)
        self.assertIsNone(serialization.__doc__)
        self.assertIs(type(setter), type)
        self.assertEqual(setter.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(setter), serialization)
        self.assertEqual(str(inspect.signature(setter)), "(flags: int) -> None")
        self.assertEqual(setter.__annotations__, {})
        self.assertEqual(typing.get_type_hints(setter), {})
        self.assertEqual(setter.__name__, "set_default_mmap_options")
        self.assertEqual(setter.__qualname__, "set_default_mmap_options")
        self.assertEqual(inspect.cleandoc(setter.__doc__), inspect.cleandoc(SETTER_DOC))

        methods = {
            "__init__": "(self, flags: int) -> None",
            "__enter__": "(self) -> None",
            "__exit__": (
                "(self, exc_type: Any, exc_value: Any, traceback: Any) -> None"
            ),
        }
        expected_annotations = {
            "__init__": {"flags": int, "return": None},
            "__enter__": {"return": None},
            "__exit__": {
                "exc_type": typing.Any,
                "exc_value": typing.Any,
                "traceback": typing.Any,
                "return": None,
            },
        }
        for name, signature in methods.items():
            with self.subTest(name=name):
                method = getattr(setter, name)
                self.assertEqual(str(inspect.signature(method)), signature)
                self.assertEqual(method.__annotations__, expected_annotations[name])
                self.assertIsNone(method.__defaults__)
                self.assertIsNone(method.__kwdefaults__)

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        serialization = torch.serialization
        setter = serialization.set_default_mmap_options
        exported_names = [
            "get_crc32_options",
            "set_crc32_options",
            "get_default_mmap_options",
            "set_default_mmap_options",
        ]
        self.assertEqual(serialization.__all__, exported_names)

        direct_import = {}
        exec(
            "from torch_rs.serialization import set_default_mmap_options",
            direct_import,
        )
        self.assertIs(direct_import["set_default_mmap_options"], setter)

        namespace = {}
        exec("from torch_rs.serialization import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")},
            set(exported_names),
        )
        self.assertIs(namespace["set_default_mmap_options"], setter)

        self.assertNotIn("set_default_mmap_options", torch.__all__)
        self.assertFalse(hasattr(torch, "set_default_mmap_options"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("set_default_mmap_options", top_level_namespace)

        self.assertIs(copy.copy(setter), setter)
        self.assertIs(copy.deepcopy(setter), setter)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(setter, protocol=protocol)
                self.assertIn(b"torch_rs.serialization", payload)
                self.assertIs(pickle.loads(payload), setter)

    @unittest.skipUnless(HAS_MMAP_FLAGS, "requires POSIX mmap flags")
    def test_argument_errors_match_pytorch_2_13(self):
        setter = torch.serialization.set_default_mmap_options
        getter = torch.serialization.get_default_mmap_options
        cases = (
            (
                lambda: setter(),
                "set_default_mmap_options.__init__() missing 1 required "
                "positional argument: 'flags'",
            ),
            (
                lambda: setter(mmap.MAP_PRIVATE, mmap.MAP_SHARED),
                "set_default_mmap_options.__init__() takes 2 positional "
                "arguments but 3 were given",
            ),
            (
                lambda: setter(enabled=True),
                "set_default_mmap_options.__init__() got an unexpected keyword "
                "argument 'enabled'",
            ),
            (
                lambda: setter(mmap.MAP_PRIVATE, flags=mmap.MAP_SHARED),
                "set_default_mmap_options.__init__() got multiple values for "
                "argument 'flags'",
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

    @unittest.skipUnless(HAS_MMAP_FLAGS, "requires POSIX mmap flags")
    def test_reload_and_reimport_share_state_with_existing_contexts(self):
        original_module = torch.serialization
        original_getter = original_module.get_default_mmap_options
        original_setter = original_module.set_default_mmap_options

        active_context = original_setter(mmap.MAP_SHARED)
        self.assertIs(importlib.reload(original_module), original_module)
        self.assertIs(torch.serialization, original_module)
        self.assertIsNot(original_module.set_default_mmap_options, original_setter)
        self.assertEqual(original_getter(), mmap.MAP_SHARED)
        self.assertEqual(original_module.get_default_mmap_options(), mmap.MAP_SHARED)
        self.assertIsNone(active_context.__exit__(None, None, None))
        self.assertEqual(original_module.get_default_mmap_options(), mmap.MAP_PRIVATE)

        module_name = original_module.__name__
        old_setter = original_module.set_default_mmap_options
        old_getter = original_module.get_default_mmap_options
        old_setter(mmap.MAP_SHARED)
        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)
            self.assertIsNot(replacement_module, original_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertEqual(old_getter(), mmap.MAP_SHARED)
            self.assertEqual(
                replacement_module.get_default_mmap_options(), mmap.MAP_SHARED
            )

            replacement_module.set_default_mmap_options(mmap.MAP_PRIVATE)
            self.assertEqual(old_getter(), mmap.MAP_PRIVATE)
            old_setter(mmap.MAP_SHARED)
            self.assertEqual(
                replacement_module.get_default_mmap_options(), mmap.MAP_SHARED
            )
        finally:
            old_setter(self.original)
            sys.modules[module_name] = original_module
            torch.serialization = original_module


if __name__ == "__main__":
    unittest.main()
