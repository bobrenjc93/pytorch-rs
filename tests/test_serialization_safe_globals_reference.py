import copy
import importlib
import inspect
import pickle
import pickletools
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


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
        return 23

    def __eq__(self, other):
        return isinstance(other, EqualGlobal)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SerializationSafeGlobalsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "serialization safe-global differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.original_safe_globals = {
            module: module.serialization.get_safe_globals()
            for module in (torch, reference_torch)
        }
        for module in self.original_safe_globals:
            module.serialization.clear_safe_globals()

    def tearDown(self):
        for module, values in self.original_safe_globals.items():
            module.serialization.clear_safe_globals()
            module.serialization.add_safe_globals(values)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def registry_outcome(self, module):
        serialization = module.serialization
        serialization.clear_safe_globals()
        tuple_entry = (function_global, "legacy.module.function_global")
        values = [
            FirstGlobal,
            FirstGlobal,
            tuple_entry,
            tuple_entry,
            function_global,
        ]
        consumed = []

        def entries():
            for value in values:
                consumed.append(value)
                yield value
            consumed.append("finished")

        add_result = serialization.add_safe_globals(entries())
        first_read = serialization.get_safe_globals()
        second_read = serialization.get_safe_globals()
        tuple_identity = any(value is tuple_entry for value in first_read)
        first_read.clear()
        first_read.append(SecondGlobal)
        after_mutation = serialization.get_safe_globals()

        serialization.clear_safe_globals()
        first_equal = EqualGlobal("first")
        second_equal = EqualGlobal("second")
        equal_add_result = serialization.add_safe_globals(
            [first_equal, second_equal]
        )
        equal_read = serialization.get_safe_globals()
        duplicate_add_result = serialization.add_safe_globals([second_equal])
        duplicate_read = serialization.get_safe_globals()
        clear_result = serialization.clear_safe_globals()
        first_empty = serialization.get_safe_globals()
        second_empty = serialization.get_safe_globals()

        return (
            add_result is None,
            consumed == [*values, "finished"],
            set(second_read) == {FirstGlobal, tuple_entry, function_global},
            len(second_read),
            first_read is not second_read,
            tuple_identity,
            set(after_mutation) == {FirstGlobal, tuple_entry, function_global},
            equal_add_result is None,
            len(equal_read),
            equal_read[0] is first_equal,
            duplicate_add_result is None,
            duplicate_read[0] is first_equal,
            clear_result is None,
            first_empty == [],
            type(first_empty) is list,
            first_empty is not second_empty,
        )

    def hashability_outcome(self, module):
        serialization = module.serialization
        serialization.clear_safe_globals()
        serialization.add_safe_globals([FirstGlobal])
        events = []

        def invalid_entries():
            events.append("first")
            yield function_global
            events.append("unhashable")
            yield []
            events.append("not-consumed")
            yield SecondGlobal

        errors = []
        for value in (invalid_entries(), [(function_global, [])], None, 1):
            try:
                serialization.add_safe_globals(value)
            except Exception as error:
                errors.append(
                    (type(error).__module__, type(error).__qualname__, str(error), error.args)
                )
            else:
                errors.append(None)

        return (
            errors,
            events,
            serialization.get_safe_globals() == [FirstGlobal],
        )

    def thread_outcome(self, module):
        serialization = module.serialization
        serialization.clear_safe_globals()
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
                errors.append((type(error).__name__, str(error)))

        thread = threading.Thread(target=worker)
        thread.start()
        if not worker_ready.wait(timeout=10):
            raise RuntimeError("timed out waiting for worker startup")
        main_result = serialization.add_safe_globals([FirstGlobal, tuple_entry])
        main_updated.set()
        if not worker_updated.wait(timeout=10):
            raise RuntimeError("timed out waiting for worker update")
        main_read = set(serialization.get_safe_globals())
        clear_result = serialization.clear_safe_globals()
        main_cleared.set()
        thread.join(timeout=10)

        return (
            main_result is None,
            main_read == {FirstGlobal, tuple_entry, SecondGlobal},
            clear_result is None,
            not thread.is_alive(),
            errors,
            observations,
        )

    def reload_outcome(self, module):
        original_module = module.serialization
        original_getter = original_module.get_safe_globals
        original_adder = original_module.add_safe_globals
        original_clearer = original_module.clear_safe_globals
        module_name = original_module.__name__
        tuple_entry = (function_global, "reload.function_global")

        original_clearer()
        original_adder([FirstGlobal])
        try:
            reloaded = importlib.reload(original_module)
            after_reload = (
                reloaded is original_module,
                module.serialization is original_module,
                reloaded.get_safe_globals is not original_getter,
                original_getter() == [FirstGlobal],
                reloaded.get_safe_globals() == [FirstGlobal],
            )

            removed = sys.modules.pop(module_name)
            replacement = importlib.import_module(module_name)
            after_reimport = (
                removed is original_module,
                replacement is not original_module,
                sys.modules[module_name] is replacement,
                module.serialization is replacement,
                original_getter() == [FirstGlobal],
                replacement.get_safe_globals() == [FirstGlobal],
            )
            add_result = replacement.add_safe_globals([tuple_entry])
            shared_after_add = set(original_getter()) == {FirstGlobal, tuple_entry}
            clear_result = original_clearer()
            shared_after_clear = replacement.get_safe_globals() == []
            return (
                after_reload,
                after_reimport,
                add_result is None,
                shared_after_add,
                clear_result is None,
                shared_after_clear,
            )
        finally:
            sys.modules[module_name] = original_module
            module.serialization = original_module

    def test_registry_semantics_match_pytorch_2_13(self):
        actual = self.registry_outcome(torch)
        expected = self.registry_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertTrue(all(actual))

    def test_hashability_errors_and_consumption_match_pytorch_2_13(self):
        actual = self.hashability_outcome(torch)
        expected = self.hashability_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(actual[1], ["first", "unhashable"])
        self.assertTrue(actual[2])

    def test_thread_visibility_matches_pytorch_2_13(self):
        actual = self.thread_outcome(torch)
        expected = self.thread_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, (True, True, True, True, [], [True] * 4))

    def test_metadata_copying_and_pickling_match_pytorch_2_13(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization

        for name in (
            "clear_safe_globals",
            "get_safe_globals",
            "add_safe_globals",
        ):
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual)),
                    str(inspect.signature(expected)),
                )
                self.assertEqual(actual.__annotations__, expected.__annotations__)
                self.assertEqual(
                    typing.get_type_hints(actual),
                    typing.get_type_hints(expected),
                )
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(
                    actual.__module__.replace("torch_rs", "torch"),
                    expected.__module__,
                )
                self.assertIs(inspect.getmodule(actual), actual_module)
                self.assertIs(inspect.getmodule(expected), expected_module)
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.deepcopy(expected), expected)

                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(name=name, protocol=protocol):
                        self.assertIs(
                            pickle.loads(pickle.dumps(actual, protocol)),
                            actual,
                        )
                        self.assertIs(
                            pickle.loads(pickle.dumps(expected, protocol)),
                            expected,
                        )
                        self.assertEqual(
                            self.pickle_shape(actual, protocol),
                            self.pickle_shape(expected, protocol),
                        )

    def test_exports_and_unsupported_scope_match_pytorch_2_13(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        supported_names = {
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
        }

        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported_names],
        )
        self.assertEqual(
            {name for name in vars(actual_module) if not name.startswith("_")},
            supported_names,
        )

        for name in ("clear_safe_globals", "get_safe_globals", "add_safe_globals"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertFalse(hasattr(reference_torch, name))
                self.assertNotIn(name, torch.__all__)
                self.assertEqual(
                    actual_module.__all__.count(name),
                    expected_module.__all__.count(name),
                )

        for name in ("safe_globals", "save", "load"):
            with self.subTest(unsupported=name):
                self.assertTrue(hasattr(expected_module, name))
                self.assertFalse(hasattr(actual_module, name))
                self.assertNotIn(name, actual_module.__all__)

    def test_argument_errors_and_state_preservation_match_pytorch_2_13(self):
        actual = torch.serialization
        expected = reference_torch.serialization
        actual.add_safe_globals([FirstGlobal])
        expected.add_safe_globals([FirstGlobal])
        cases = (
            (
                lambda: actual.clear_safe_globals(None),
                lambda: expected.clear_safe_globals(None),
            ),
            (
                lambda: actual.clear_safe_globals(enabled=True),
                lambda: expected.clear_safe_globals(enabled=True),
            ),
            (
                lambda: actual.get_safe_globals(None),
                lambda: expected.get_safe_globals(None),
            ),
            (
                lambda: actual.get_safe_globals(enabled=True),
                lambda: expected.get_safe_globals(enabled=True),
            ),
            (
                lambda: actual.add_safe_globals(),
                lambda: expected.add_safe_globals(),
            ),
            (
                lambda: actual.add_safe_globals([], []),
                lambda: expected.add_safe_globals([], []),
            ),
            (
                lambda: actual.add_safe_globals(enabled=[]),
                lambda: expected.add_safe_globals(enabled=[]),
            ),
            (
                lambda: actual.add_safe_globals([], safe_globals=[]),
                lambda: expected.add_safe_globals([], safe_globals=[]),
            ),
        )

        for actual_call, expected_call in cases:
            self.assert_error_matches(actual_call, expected_call)
            self.assertEqual(actual.get_safe_globals(), [FirstGlobal])
            self.assertEqual(expected.get_safe_globals(), [FirstGlobal])

        self.assertIsNone(actual.add_safe_globals(safe_globals=[SecondGlobal]))
        self.assertIsNone(expected.add_safe_globals(safe_globals=[SecondGlobal]))

    def test_reload_and_reimport_state_match_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
