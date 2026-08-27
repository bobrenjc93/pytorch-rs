import copy
import importlib
import inspect
import pickle
import sys
import threading
import types
import typing
import unittest
from collections.abc import Callable

import torch_rs as torch


class FirstGlobal:
    pass


class SecondGlobal:
    pass


def function_global():
    pass


class EqualGlobal:
    def __init__(self, label):
        self.label = label

    def __hash__(self):
        return 17

    def __eq__(self, other):
        return isinstance(other, EqualGlobal)


class SerializationSafeGlobalsTests(unittest.TestCase):
    def setUp(self):
        self.serialization = torch.serialization
        self.original_safe_globals = self.serialization.get_safe_globals()
        self.serialization.clear_safe_globals()

    def tearDown(self):
        self.serialization.clear_safe_globals()
        self.serialization.add_safe_globals(self.original_safe_globals)

    def test_clear_return_values_and_fresh_list_reads(self):
        serialization = self.serialization

        first_empty = serialization.get_safe_globals()
        second_empty = serialization.get_safe_globals()
        self.assertIs(type(first_empty), list)
        self.assertEqual(first_empty, [])
        self.assertEqual(second_empty, [])
        self.assertIsNot(first_empty, second_empty)

        self.assertIsNone(serialization.add_safe_globals([FirstGlobal]))
        first_read = serialization.get_safe_globals()
        second_read = serialization.get_safe_globals()
        self.assertEqual(first_read, [FirstGlobal])
        self.assertEqual(second_read, [FirstGlobal])
        self.assertIsNot(first_read, second_read)

        first_read.clear()
        first_read.append(SecondGlobal)
        self.assertEqual(serialization.get_safe_globals(), [FirstGlobal])

        self.assertIsNone(serialization.clear_safe_globals())
        self.assertEqual(serialization.get_safe_globals(), [])

    def test_iterable_consumption_deduplication_and_tuple_entries(self):
        serialization = self.serialization
        tuple_entry = (function_global, "legacy.module.function_global")
        yielded = [
            FirstGlobal,
            FirstGlobal,
            tuple_entry,
            tuple_entry,
            function_global,
        ]
        consumed = []

        def entries():
            for value in yielded:
                consumed.append(value)
                yield value
            consumed.append("finished")

        self.assertIsNone(serialization.add_safe_globals(entries()))
        self.assertEqual(consumed, [*yielded, "finished"])

        stored = serialization.get_safe_globals()
        self.assertEqual(len(stored), 3)
        self.assertEqual(set(stored), {FirstGlobal, tuple_entry, function_global})
        self.assertTrue(any(value is tuple_entry for value in stored))

        equal_tuple = tuple([function_global, "legacy.module.function_global"])
        self.assertIsNot(equal_tuple, tuple_entry)
        self.assertIsNone(serialization.add_safe_globals((equal_tuple,)))
        stored = serialization.get_safe_globals()
        self.assertEqual(len(stored), 3)
        self.assertTrue(any(value is tuple_entry for value in stored))
        self.assertFalse(any(value is equal_tuple for value in stored))

        serialization.clear_safe_globals()
        first_equal = EqualGlobal("first")
        second_equal = EqualGlobal("second")
        self.assertIsNone(
            serialization.add_safe_globals([first_equal, second_equal])
        )
        stored = serialization.get_safe_globals()
        self.assertEqual(len(stored), 1)
        self.assertIs(stored[0], first_equal)
        self.assertIsNone(serialization.add_safe_globals([second_equal]))
        self.assertIs(serialization.get_safe_globals()[0], first_equal)

    def test_hashability_errors_preserve_state_and_stop_consumption(self):
        serialization = self.serialization
        serialization.add_safe_globals([FirstGlobal])
        events = []

        def invalid_entries():
            events.append("first")
            yield function_global
            events.append("unhashable")
            yield []
            events.append("not-consumed")
            yield SecondGlobal

        with self.assertRaises(TypeError) as raised:
            serialization.add_safe_globals(invalid_entries())
        self.assertEqual(str(raised.exception), "unhashable type: 'list'")
        self.assertEqual(raised.exception.args, ("unhashable type: 'list'",))
        self.assertEqual(events, ["first", "unhashable"])
        self.assertEqual(serialization.get_safe_globals(), [FirstGlobal])

        with self.assertRaises(TypeError) as raised:
            serialization.add_safe_globals([(function_global, [])])
        self.assertEqual(str(raised.exception), "unhashable type: 'list'")
        self.assertEqual(serialization.get_safe_globals(), [FirstGlobal])

        for value, message in (
            (None, "'NoneType' object is not iterable"),
            (1, "'int' object is not iterable"),
        ):
            with self.subTest(value=value):
                with self.assertRaises(TypeError) as raised:
                    serialization.add_safe_globals(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(serialization.get_safe_globals(), [FirstGlobal])

    def test_updates_are_process_global_and_visible_across_threads(self):
        serialization = self.serialization
        tuple_entry = (function_global, "thread.function_global")
        worker_ready = threading.Event()
        main_updated = threading.Event()
        worker_updated = threading.Event()
        main_cleared = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(serialization.get_safe_globals() == [])
                worker_ready.set()
                if not main_updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(
                    set(serialization.get_safe_globals())
                    == {FirstGlobal, tuple_entry}
                )
                observations.append(
                    serialization.add_safe_globals([SecondGlobal]) is None
                )
                worker_updated.set()
                if not main_cleared.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread clear")
                observations.append(serialization.get_safe_globals() == [])
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        self.assertIsNone(
            serialization.add_safe_globals([FirstGlobal, tuple_entry])
        )
        main_updated.set()
        self.assertTrue(worker_updated.wait(timeout=10))
        self.assertEqual(
            set(serialization.get_safe_globals()),
            {FirstGlobal, tuple_entry, SecondGlobal},
        )
        self.assertIsNone(serialization.clear_safe_globals())
        main_cleared.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True, True, True])

    def test_signature_annotations_documentation_and_identity(self):
        serialization = self.serialization
        entry_type = Callable | tuple[Callable, str]
        expected = {
            "clear_safe_globals": (
                "() -> None",
                {"return": None},
                {"return": type(None)},
                "Clears the list of globals that are safe for ``weights_only`` load.",
            ),
            "get_safe_globals": (
                "() -> list[collections.abc.Callable | "
                "tuple[collections.abc.Callable, str]]",
                {"return": list[entry_type]},
                {"return": list[entry_type]},
                "Returns the list of user-added globals that are safe for "
                "``weights_only`` load.",
            ),
            "add_safe_globals": (
                "(safe_globals: list[collections.abc.Callable | "
                "tuple[collections.abc.Callable, str]]) -> None",
                {"safe_globals": list[entry_type], "return": None},
                {"safe_globals": list[entry_type], "return": type(None)},
                None,
            ),
        }

        for name, (signature, annotations, type_hints, doc) in expected.items():
            with self.subTest(name=name):
                function = getattr(serialization, name)
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(function.__module__, "torch_rs.serialization")
                self.assertIs(inspect.getmodule(function), serialization)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(function.__annotations__, annotations)
                self.assertEqual(typing.get_type_hints(function), type_hints)
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))
                if doc is not None:
                    self.assertEqual(inspect.cleandoc(function.__doc__), doc)

        self.assertIn(
            "Each item in the list can either be a function/class or a tuple",
            serialization.add_safe_globals.__doc__,
        )

    def test_exports_copy_and_pickle_use_the_canonical_functions(self):
        serialization = self.serialization
        expected_exports = [
            "LoadEndianness",
            "get_crc32_options",
            "set_crc32_options",
            "get_default_load_endianness",
            "set_default_load_endianness",
            "get_default_mmap_options",
            "set_default_mmap_options",
            "clear_safe_globals",
            "get_safe_globals",
            "add_safe_globals",
        ]
        functions = {
            name: getattr(serialization, name)
            for name in (
                "clear_safe_globals",
                "get_safe_globals",
                "add_safe_globals",
            )
        }

        self.assertEqual(serialization.__all__, expected_exports)
        self.assertEqual(
            {name for name in vars(serialization) if not name.startswith("_")},
            set(expected_exports),
        )

        direct_import = {}
        exec(
            "from torch_rs.serialization import "
            "clear_safe_globals, get_safe_globals, add_safe_globals",
            direct_import,
        )
        wildcard_import = {}
        exec("from torch_rs.serialization import *", wildcard_import)
        for name, function in functions.items():
            self.assertIs(direct_import[name], function)
            self.assertIs(wildcard_import[name], function)
            self.assertFalse(hasattr(torch, name))
            self.assertNotIn(name, torch.__all__)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.serialization", payload)
                    self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_preserve_state(self):
        serialization = self.serialization
        serialization.add_safe_globals([FirstGlobal])
        cases = (
            (
                lambda: serialization.clear_safe_globals(None),
                "clear_safe_globals() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: serialization.clear_safe_globals(enabled=True),
                "clear_safe_globals() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: serialization.get_safe_globals(None),
                "get_safe_globals() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: serialization.get_safe_globals(enabled=True),
                "get_safe_globals() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: serialization.add_safe_globals(),
                "add_safe_globals() missing 1 required positional argument: "
                "'safe_globals'",
            ),
            (
                lambda: serialization.add_safe_globals([], []),
                "add_safe_globals() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: serialization.add_safe_globals(enabled=[]),
                "add_safe_globals() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: serialization.add_safe_globals(
                    [], safe_globals=[]
                ),
                "add_safe_globals() got multiple values for argument "
                "'safe_globals'",
            ),
        )

        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(serialization.get_safe_globals(), [FirstGlobal])

        self.assertIsNone(
            serialization.add_safe_globals(safe_globals=[SecondGlobal])
        )

    def test_reload_and_reimport_preserve_process_state(self):
        original_module = self.serialization
        original_getter = original_module.get_safe_globals
        original_adder = original_module.add_safe_globals
        original_clearer = original_module.clear_safe_globals
        tuple_entry = (function_global, "reload.function_global")

        original_adder([FirstGlobal])
        self.assertIs(importlib.reload(original_module), original_module)
        self.assertIs(torch.serialization, original_module)
        self.assertIsNot(original_module.get_safe_globals, original_getter)
        self.assertEqual(original_getter(), [FirstGlobal])
        self.assertEqual(original_module.get_safe_globals(), [FirstGlobal])

        module_name = original_module.__name__
        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)
            self.assertIsNot(replacement_module, original_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertEqual(original_getter(), [FirstGlobal])
            self.assertEqual(replacement_module.get_safe_globals(), [FirstGlobal])

            self.assertIsNone(replacement_module.add_safe_globals([tuple_entry]))
            self.assertEqual(
                set(original_getter()),
                {FirstGlobal, tuple_entry},
            )
            self.assertIsNone(original_clearer())
            self.assertEqual(replacement_module.get_safe_globals(), [])
        finally:
            sys.modules[module_name] = original_module
            torch.serialization = original_module

    def test_safe_globals_save_and_load_remain_unsupported(self):
        serialization = self.serialization

        for name in ("safe_globals", "save", "load"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(serialization, name))
                self.assertNotIn(name, serialization.__all__)


if __name__ == "__main__":
    unittest.main()
