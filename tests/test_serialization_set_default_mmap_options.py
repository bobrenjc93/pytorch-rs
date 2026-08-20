import copy
import importlib
import inspect
import mmap
import pickle
import sys
import threading
import typing
import unittest

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


class SerializationSetDefaultMmapOptionsTests(unittest.TestCase):
    def setUp(self):
        self.serialization = torch.serialization
        self.getter = self.serialization.get_default_mmap_options
        self.setter = self.serialization.set_default_mmap_options
        self.private = getattr(mmap, "MAP_PRIVATE", None)
        self.shared = getattr(mmap, "MAP_SHARED", None)
        self.supported = (
            sys.platform != "win32"
            and self.private is not None
            and self.shared is not None
        )
        self.original = self.getter()
        if self.supported:
            self.setter(self.private)

    def tearDown(self):
        if self.supported:
            self.serialization.set_default_mmap_options(self.original)

    def require_supported(self):
        if not self.supported:
            self.skipTest("mmap.MAP_PRIVATE and mmap.MAP_SHARED are required")

    def test_signature_annotations_documentation_and_class_identity(self):
        setter = self.setter

        self.assertIs(type(setter), type)
        self.assertEqual(setter.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(setter), self.serialization)
        self.assertEqual(setter.__name__, "set_default_mmap_options")
        self.assertEqual(setter.__qualname__, "set_default_mmap_options")
        self.assertEqual(setter.__bases__, (object,))
        self.assertEqual(str(inspect.signature(setter)), "(flags: int) -> None")
        self.assertEqual(setter.__annotations__, {})
        self.assertEqual(typing.get_type_hints(setter), {})
        self.assertEqual(
            inspect.cleandoc(setter.__doc__),
            inspect.cleandoc(SETTER_DOC),
        )

        methods = {
            "__init__": (
                "(self, flags: int) -> None",
                {"flags": int, "return": None},
                {"flags": int, "return": type(None)},
            ),
            "__enter__": (
                "(self) -> None",
                {"return": None},
                {"return": type(None)},
            ),
            "__exit__": (
                "(self, exc_type: Any, exc_value: Any, traceback: Any) -> None",
                {
                    "exc_type": typing.Any,
                    "exc_value": typing.Any,
                    "traceback": typing.Any,
                    "return": None,
                },
                {
                    "exc_type": typing.Any,
                    "exc_value": typing.Any,
                    "traceback": typing.Any,
                    "return": type(None),
                },
            ),
        }
        for name, (signature, annotations, type_hints) in methods.items():
            with self.subTest(name=name):
                method = getattr(setter, name)
                self.assertEqual(str(inspect.signature(method)), signature)
                self.assertEqual(method.__annotations__, annotations)
                self.assertEqual(typing.get_type_hints(method), type_hints)
                self.assertIsNone(method.__defaults__)
                self.assertIsNone(method.__kwdefaults__)
                self.assertEqual(method.__dict__, {})
                self.assertFalse(hasattr(method, "__text_signature__"))

    def test_imports_exports_copy_and_pickle_use_the_canonical_class(self):
        setter = self.setter

        self.assertIn("set_default_mmap_options", self.serialization.__all__)
        self.assertEqual(self.serialization.__all__.count(setter.__name__), 1)
        direct_import = {}
        exec(
            "from torch_rs.serialization import set_default_mmap_options",
            direct_import,
        )
        self.assertIs(direct_import[setter.__name__], setter)

        namespace = {}
        exec("from torch_rs.serialization import *", namespace)
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

    def test_direct_calls_update_immediately_and_instances_restore(self):
        self.require_supported()

        context = self.setter(self.shared)
        self.assertIs(type(context), self.setter)
        self.assertEqual(context.__dict__, {"prev": self.private})
        self.assertIs(self.getter(), self.shared)
        self.assertIsNone(context.__enter__())
        self.assertIs(self.getter(), self.shared)
        self.assertIsNone(context.__exit__(None, None, None))
        self.assertIs(self.getter(), self.private)

        context = self.setter(flags=self.shared)
        self.assertIs(self.getter(), self.shared)
        del context
        self.assertIs(self.getter(), self.shared)
        self.setter(self.private)
        self.assertIs(self.getter(), self.private)

    def test_nested_contexts_and_exceptional_exits_restore_prior_state(self):
        self.require_supported()

        class MarkerError(Exception):
            pass

        self.assertIs(self.getter(), self.private)
        with self.setter(self.shared) as outer_value:
            self.assertIsNone(outer_value)
            self.assertIs(self.getter(), self.shared)
            with self.setter(self.private) as inner_value:
                self.assertIsNone(inner_value)
                self.assertIs(self.getter(), self.private)
            self.assertIs(self.getter(), self.shared)

            with self.assertRaisesRegex(MarkerError, "boom"):
                with self.setter(self.private):
                    self.assertIs(self.getter(), self.private)
                    raise MarkerError("boom")
            self.assertIs(self.getter(), self.shared)
        self.assertIs(self.getter(), self.private)

    def test_updates_and_context_restoration_are_process_global(self):
        self.require_supported()

        entered = threading.Event()
        leave = threading.Event()
        exited = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                with self.setter(self.shared):
                    observations.append(self.getter() is self.shared)
                    entered.set()
                    if not leave.wait(timeout=10):
                        raise RuntimeError("timed out waiting to leave mmap context")
                observations.append(self.getter() is self.private)
                exited.set()
            except BaseException as error:
                errors.append(error)
                entered.set()
                exited.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(entered.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(self.getter(), self.shared)
        leave.set()
        self.assertTrue(exited.wait(timeout=10))
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True])
        self.assertIs(self.getter(), self.private)

    def test_invalid_flags_and_call_errors_preserve_state(self):
        self.require_supported()

        invalid_flags = (0, -1, self.private | self.shared, False, None, "invalid")
        for flags in invalid_flags:
            with self.subTest(flags=flags):
                message = (
                    "Invalid argument in function set_default_mmap_options, "
                    "expected mmap.MAP_PRIVATE or mmap.MAP_SHARED, but got "
                    f"{flags}"
                )
                with self.assertRaises(ValueError) as raised:
                    self.setter(flags)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(self.getter(), self.private)

        cases = (
            (
                lambda: self.setter(),
                "set_default_mmap_options.__init__() missing 1 required "
                "positional argument: 'flags'",
            ),
            (
                lambda: self.setter(self.private, self.shared),
                "set_default_mmap_options.__init__() takes 2 positional "
                "arguments but 3 were given",
            ),
            (
                lambda: self.setter(enabled=True),
                "set_default_mmap_options.__init__() got an unexpected keyword "
                "argument 'enabled'",
            ),
            (
                lambda: self.setter(self.private, flags=self.shared),
                "set_default_mmap_options.__init__() got multiple values for "
                "argument 'flags'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(self.getter(), self.private)

        if True == self.shared:
            true_context = self.setter(True)
            self.assertIs(self.getter(), True)
            true_context.__exit__(None, None, None)
            self.assertIs(self.getter(), self.private)

    def test_windows_platform_guard_matches_pytorch_error(self):
        before = self.getter()
        original = self.serialization._IS_WINDOWS
        self.serialization._IS_WINDOWS = True
        try:
            message = (
                "Changing the default mmap options is currently not supported "
                "for Windows"
            )
            with self.assertRaises(RuntimeError) as raised:
                self.setter(self.private)
            self.assertEqual(str(raised.exception), message)
            self.assertEqual(raised.exception.args, (message,))
            self.assertIs(self.getter(), before)
        finally:
            self.serialization._IS_WINDOWS = original

    def test_reload_and_reimport_share_persistent_state_with_old_classes(self):
        self.require_supported()

        original_module = self.serialization
        original_setter = self.setter
        original_getter = self.getter
        module_name = original_module.__name__

        outer = original_setter(self.shared)
        try:
            self.assertIs(importlib.reload(original_module), original_module)
            reloaded_setter = original_module.set_default_mmap_options
            self.assertIsNot(reloaded_setter, original_setter)
            self.assertIs(original_getter(), self.shared)
            self.assertIs(original_module.get_default_mmap_options(), self.shared)

            with reloaded_setter(self.private):
                self.assertIs(original_getter(), self.private)
            self.assertIs(original_getter(), self.shared)

            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)
            self.assertIsNot(replacement_module, original_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertIs(replacement_module.get_default_mmap_options(), self.shared)

            with replacement_module.set_default_mmap_options(self.private):
                self.assertIs(original_getter(), self.private)
            self.assertIs(original_getter(), self.shared)

            outer.__exit__(None, None, None)
            outer = None
            self.assertIs(original_getter(), self.private)
            self.assertIs(replacement_module.get_default_mmap_options(), self.private)
        finally:
            if outer is not None:
                outer.__exit__(None, None, None)
            sys.modules[module_name] = original_module
            torch.serialization = original_module


if __name__ == "__main__":
    unittest.main()
