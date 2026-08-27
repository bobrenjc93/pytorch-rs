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


def _first_safe_global():
    pass


def _second_safe_global():
    pass


class _EqualRegistrant:
    def __hash__(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, _EqualRegistrant)


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

    def mutation_outcome(self, module):
        serialization = module.serialization
        tuple_entry = (_second_safe_global, "renamed.module.SecondGlobal")

        class OnePassIterable:
            def __init__(self):
                self.iterations = 0
                self.yield_count = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations != 1:
                    raise AssertionError("safe globals iterable was consumed twice")
                for value in (
                    _first_safe_global,
                    tuple_entry,
                    _first_safe_global,
                    tuple_entry,
                ):
                    self.yield_count += 1
                    yield value

        values = OnePassIterable()
        add_result = serialization.add_safe_globals(values)
        first_read = serialization.get_safe_globals()
        second_read = serialization.get_safe_globals()
        first_read.clear()
        after_mutation = serialization.get_safe_globals()

        equal_first = _EqualRegistrant()
        equal_second = _EqualRegistrant()
        serialization.clear_safe_globals()
        serialization.add_safe_globals([equal_first, equal_second])
        equal_initial = serialization.get_safe_globals()
        serialization.add_safe_globals([equal_second])
        equal_updated = serialization.get_safe_globals()

        clear_result = serialization.clear_safe_globals()
        return (
            add_result is None,
            values.iterations,
            values.yield_count,
            type(first_read).__module__,
            type(first_read).__qualname__,
            first_read is not second_read,
            len(second_read),
            set(second_read) == {_first_safe_global, tuple_entry},
            any(value is tuple_entry for value in second_read),
            set(after_mutation) == {_first_safe_global, tuple_entry},
            len(equal_initial),
            equal_initial[0] is equal_first,
            len(equal_updated),
            equal_updated[0] is equal_first,
            clear_result is None,
            serialization.get_safe_globals() == [],
        )

    def thread_outcome(self, module):
        serialization = module.serialization
        tuple_entry = (_second_safe_global, "renamed.module.SecondGlobal")
        worker_ready = threading.Event()
        read_update = threading.Event()
        observations = []
        errors = []

        serialization.clear_safe_globals()
        serialization.add_safe_globals([_first_safe_global])

        def observer():
            try:
                observations.append(
                    set(serialization.get_safe_globals()) == {_first_safe_global}
                )
                worker_ready.set()
                if not read_update.wait(timeout=10):
                    raise RuntimeError("timed out waiting for registry update")
                observations.append(
                    set(serialization.get_safe_globals())
                    == {_first_safe_global, tuple_entry}
                )
                observations.append(
                    serialization.add_safe_globals([_second_safe_global]) is None
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        thread = threading.Thread(target=observer)
        thread.start()
        if not worker_ready.wait(timeout=10):
            raise RuntimeError("timed out waiting for registry observer")
        main_result = serialization.add_safe_globals([tuple_entry])
        read_update.set()
        thread.join(timeout=10)

        return (
            thread.is_alive(),
            errors,
            observations,
            main_result is None,
            set(serialization.get_safe_globals())
            == {_first_safe_global, _second_safe_global, tuple_entry},
        )

    def reload_outcome(self, module):
        serialization = module.serialization
        old_add = serialization.add_safe_globals
        old_get = serialization.get_safe_globals
        old_clear = serialization.clear_safe_globals
        serialization.clear_safe_globals()
        old_add([_first_safe_global])

        reload_result = importlib.reload(serialization)
        state_after_reload = set(serialization.get_safe_globals())
        old_add([_second_safe_global])
        state_after_old_add = set(serialization.get_safe_globals())
        new_clear_result = serialization.clear_safe_globals()

        return (
            reload_result is serialization,
            module.serialization is serialization,
            serialization.add_safe_globals is not old_add,
            serialization.get_safe_globals is not old_get,
            serialization.clear_safe_globals is not old_clear,
            state_after_reload == {_first_safe_global},
            state_after_old_add == {_first_safe_global, _second_safe_global},
            set(old_get()) == {_first_safe_global, _second_safe_global},
            new_clear_result is None,
            old_get() == [],
            old_clear() is None,
        )

    def test_iterables_deduplication_tuple_entries_and_return_values_match(self):
        self.assertEqual(
            self.mutation_outcome(torch),
            self.mutation_outcome(reference_torch),
        )

    def test_hashability_iteration_and_argument_errors_match(self):
        class RaisingIterable:
            def __iter__(self):
                yield _second_safe_global
                raise RuntimeError("iteration failed")

        class RaisingHash:
            def __hash__(self):
                raise LookupError("hash failed")

        value_pairs = (
            ([[]], [[]]),
            (
                [(_second_safe_global, [])],
                [(_second_safe_global, [])],
            ),
            (1, 1),
            (None, None),
            (RaisingIterable(), RaisingIterable()),
            ([RaisingHash()], [RaisingHash()]),
        )
        for actual_values, expected_values in value_pairs:
            with self.subTest(values=repr(actual_values)):
                for module in (torch, reference_torch):
                    module.serialization.clear_safe_globals()
                    module.serialization.add_safe_globals([_first_safe_global])
                self.assert_error_matches(
                    lambda: torch.serialization.add_safe_globals(actual_values),
                    lambda: reference_torch.serialization.add_safe_globals(
                        expected_values
                    ),
                )
                for module in (torch, reference_torch):
                    self.assertEqual(
                        set(module.serialization.get_safe_globals()),
                        {_first_safe_global},
                    )

        actual = torch.serialization
        expected = reference_torch.serialization
        argument_cases = (
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
                lambda: actual.add_safe_globals(globals=[]),
                lambda: expected.add_safe_globals(globals=[]),
            ),
            (
                lambda: actual.add_safe_globals([], safe_globals=[]),
                lambda: expected.add_safe_globals([], safe_globals=[]),
            ),
        )
        for actual_call, expected_call in argument_cases:
            self.assert_error_matches(actual_call, expected_call)

    def test_thread_visibility_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_outcome(torch),
            self.thread_outcome(reference_torch),
        )

    def test_serialization_module_reload_visibility_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )

    def test_metadata_matches_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.serialization")
        expected_module = importlib.import_module("torch.serialization")

        self.assertIs(torch.serialization, actual_module)
        self.assertIs(reference_torch.serialization, expected_module)
        self.assertIsNone(actual_module.__doc__)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)

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

    def test_imports_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        supported_names = (
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
        )

        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported_names],
        )
        self.assertIs(sys.modules["torch_rs.serialization"], actual_module)
        self.assertIs(sys.modules["torch.serialization"], expected_module)

        actual_direct_import = {}
        expected_direct_import = {}
        exec(
            "from torch_rs.serialization import "
            "clear_safe_globals, get_safe_globals, add_safe_globals",
            actual_direct_import,
        )
        exec(
            "from torch.serialization import "
            "clear_safe_globals, get_safe_globals, add_safe_globals",
            expected_direct_import,
        )

        for name in supported_names[-3:]:
            actual = getattr(actual_module, name)
            expected = getattr(expected_module, name)
            self.assertIs(actual_direct_import[name], actual)
            self.assertIs(expected_direct_import[name], expected)
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

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in supported_names[-3:]:
                self.assertFalse(hasattr(module, name))
                self.assertNotIn(name, module.__all__)
                self.assertNotIn(name, namespace)

    def test_context_manager_save_and_load_remain_out_of_scope(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        for name in ("safe_globals", "save", "load"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual_module, name))
                self.assertNotIn(name, actual_module.__all__)
                self.assertTrue(hasattr(expected_module, name))
                self.assertIn(name, expected_module.__all__)


if __name__ == "__main__":
    unittest.main()
