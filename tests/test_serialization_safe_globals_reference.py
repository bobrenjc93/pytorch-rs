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


class _EqualCallable:
    def __init__(self, name):
        self.name = name

    def __call__(self):
        return self.name

    def __hash__(self):
        return 23

    def __eq__(self, other):
        return isinstance(other, _EqualCallable)


class _OneShotIterable:
    def __init__(self, values):
        self.values = values
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations != 1:
            raise AssertionError("safe globals iterable was consumed more than once")
        return iter(self.values)


class _FailingIterable:
    def __iter__(self):
        yield len
        raise RuntimeError("safe globals iteration failed")


class _PickleTarget:
    pass


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
        for module, entries in self.original_safe_globals.items():
            module.serialization.clear_safe_globals()
            module.serialization.add_safe_globals(entries)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def registry_outcome(self, module):
        serialization = module.serialization
        first = _EqualCallable("first")
        equal_but_distinct = _EqualCallable("second")
        first_tuple = tuple([len, "custom.len"])
        equal_tuple = tuple([len, "custom.len"])

        add_result = serialization.add_safe_globals(
            [first, first, equal_but_distinct, len, first_tuple, equal_tuple]
        )
        snapshot = serialization.get_safe_globals()
        fresh_snapshot = serialization.get_safe_globals()
        initial = (
            add_result is None,
            type(snapshot) is list,
            snapshot is not fresh_snapshot,
            len(snapshot),
            any(entry is first for entry in snapshot),
            any(entry is equal_but_distinct for entry in snapshot),
            any(entry is len for entry in snapshot),
            any(entry is first_tuple for entry in snapshot),
            any(entry is equal_tuple for entry in snapshot),
        )

        snapshot.clear()
        after_snapshot_mutation = (
            len(serialization.get_safe_globals()),
            any(entry is first for entry in serialization.get_safe_globals()),
        )

        second_result = serialization.add_safe_globals(
            [equal_but_distinct, equal_tuple, str]
        )
        updated = serialization.get_safe_globals()
        after_second_add = (
            second_result is None,
            len(updated),
            any(entry is first for entry in updated),
            any(entry is equal_but_distinct for entry in updated),
            any(entry is first_tuple for entry in updated),
            any(entry is equal_tuple for entry in updated),
            any(entry is str for entry in updated),
        )

        retained_snapshot = fresh_snapshot
        clear_result = serialization.clear_safe_globals()
        after_clear = (
            clear_result is None,
            serialization.get_safe_globals(),
            len(retained_snapshot),
        )
        return initial, after_snapshot_mutation, after_second_add, after_clear

    def iterable_outcome(self, module):
        serialization = module.serialization
        serialization.clear_safe_globals()
        one_shot = _OneShotIterable((len, str, len))
        one_shot_result = serialization.add_safe_globals(one_shot)
        one_shot_snapshot = serialization.get_safe_globals()

        serialization.clear_safe_globals()
        generator_result = serialization.add_safe_globals(iter((int, float)))
        generator_snapshot = serialization.get_safe_globals()

        serialization.clear_safe_globals()
        string_result = serialization.add_safe_globals("aba")
        string_snapshot = serialization.get_safe_globals()

        serialization.clear_safe_globals()
        dict_result = serialization.add_safe_globals({"left": 1, "right": 2})
        dict_snapshot = serialization.get_safe_globals()

        marker = object()
        serialization.clear_safe_globals()
        keyword_result = serialization.add_safe_globals(safe_globals=[marker])
        keyword_snapshot = serialization.get_safe_globals()
        return (
            (
                one_shot_result is None,
                one_shot.iterations,
                set(one_shot_snapshot) == {len, str},
            ),
            (
                generator_result is None,
                set(generator_snapshot) == {int, float},
            ),
            (string_result is None, set(string_snapshot) == {"a", "b"}),
            (
                dict_result is None,
                set(dict_snapshot) == {"left", "right"},
            ),
            (
                keyword_result is None,
                len(keyword_snapshot),
                keyword_snapshot[0] is marker,
            ),
        )

    def thread_outcome(self, module):
        serialization = module.serialization
        serialization.clear_safe_globals()
        first = object()
        second = object()
        observations = []
        errors = []
        serialization.add_safe_globals([first])

        def worker():
            try:
                initial = serialization.get_safe_globals()
                serialization.add_safe_globals([second])
                updated = serialization.get_safe_globals()
                observations.append(
                    (
                        any(entry is first for entry in initial),
                        any(entry is first for entry in updated),
                        any(entry is second for entry in updated),
                    )
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        current = serialization.get_safe_globals()
        return (
            thread.is_alive(),
            errors,
            observations,
            any(entry is first for entry in current),
            any(entry is second for entry in current),
        )

    def failure_outcome(self, module, value):
        serialization = module.serialization
        serialization.clear_safe_globals()
        marker = object()
        serialization.add_safe_globals([marker])
        try:
            serialization.add_safe_globals(value)
        except Exception as error:
            current = serialization.get_safe_globals()
            return (
                type(error).__module__,
                type(error).__qualname__,
                str(error),
                error.args,
                len(current),
                current[0] is marker,
            )
        return ("accepted",)

    def reimport_outcome(self, module):
        original_module = module.serialization
        original_add = original_module.add_safe_globals
        original_get = original_module.get_safe_globals
        original_clear = original_module.clear_safe_globals
        module_name = original_module.__name__
        first = object()
        second = object()
        third = object()

        original_clear()
        original_add([first])
        reloaded = importlib.reload(original_module)
        reload_state = (
            reloaded is original_module,
            module.serialization is original_module,
            original_get is not reloaded.get_safe_globals,
            any(entry is first for entry in original_get()),
            any(entry is first for entry in reloaded.get_safe_globals()),
        )
        reloaded.add_safe_globals([second])
        cross_reload_state = (
            len(original_get()),
            any(entry is second for entry in original_get()),
        )

        try:
            removed = sys.modules.pop(module_name)
            replacement = importlib.import_module(module_name)
            replacement_state = (
                removed is original_module,
                replacement is not original_module,
                sys.modules[module_name] is replacement,
                module.serialization is replacement,
                any(entry is first for entry in replacement.get_safe_globals()),
                any(entry is second for entry in replacement.get_safe_globals()),
            )
            replacement.add_safe_globals([third])
            cross_reimport_state = (
                any(entry is third for entry in original_get()),
                original_clear() is None,
                replacement.get_safe_globals(),
            )
            return (
                reload_state,
                cross_reload_state,
                replacement_state,
                cross_reimport_state,
            )
        finally:
            sys.modules[module_name] = original_module
            module.serialization = original_module

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

    def test_registry_semantics_match_pytorch_2_13(self):
        actual = self.registry_outcome(torch)
        expected = self.registry_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(
            actual,
            (
                (True, True, True, 3, True, False, True, True, False),
                (3, True),
                (True, 4, True, False, True, False, True),
                (True, [], 3),
            ),
        )

    def test_runtime_iterable_behavior_matches_pytorch_2_13(self):
        actual = self.iterable_outcome(torch)
        expected = self.iterable_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(
            actual,
            (
                (True, 1, True),
                (True, True),
                (True, True),
                (True, True),
                (True, 1, True),
            ),
        )

    def test_iterable_errors_and_atomic_state_preservation_match(self):
        factories = (
            lambda: None,
            lambda: 3,
            lambda: 1.5,
            lambda: [len, []],
            lambda: [(len, [])],
            _FailingIterable,
        )
        for case, factory in enumerate(factories):
            with self.subTest(case=case):
                self.assertEqual(
                    self.failure_outcome(torch, factory()),
                    self.failure_outcome(reference_torch, factory()),
                )

    def test_process_global_thread_visibility_matches_pytorch_2_13(self):
        actual = self.thread_outcome(torch)
        expected = self.thread_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, (False, [], [(True, True, True)], True, True))

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs.serialization")
        expected_module = importlib.import_module("torch.serialization")

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

    def test_exports_copy_and_pickle_match_supported_scope(self):
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
        self.assertEqual(
            {name for name in vars(actual_module) if not name.startswith("_")},
            set(supported_names),
        )

        actual_direct = {}
        expected_direct = {}
        exec(
            "from torch_rs.serialization import "
            "clear_safe_globals, get_safe_globals, add_safe_globals",
            actual_direct,
        )
        exec(
            "from torch.serialization import "
            "clear_safe_globals, get_safe_globals, add_safe_globals",
            expected_direct,
        )
        for name in (
            "clear_safe_globals",
            "get_safe_globals",
            "add_safe_globals",
        ):
            actual = getattr(actual_module, name)
            expected = getattr(expected_module, name)
            self.assertIs(actual_direct[name], actual)
            self.assertIs(expected_direct[name], expected)
            self.assertIs(copy.copy(actual), actual)
            self.assertIs(copy.copy(expected), expected)
            self.assertIs(copy.deepcopy(actual), actual)
            self.assertIs(copy.deepcopy(expected), expected)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)), expected
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

        for module in (torch, reference_torch):
            top_level = {}
            exec(f"from {module.__name__} import *", top_level)
            for name in (
                "clear_safe_globals",
                "get_safe_globals",
                "add_safe_globals",
            ):
                self.assertFalse(hasattr(module, name))
                self.assertNotIn(name, module.__all__)
                self.assertNotIn(name, top_level)

        self.assertFalse(hasattr(actual_module, "safe_globals"))
        self.assertNotIn("safe_globals", actual_module.__all__)
        self.assertTrue(hasattr(expected_module, "safe_globals"))

    def test_snapshot_copy_and_pickle_behavior_matches_pytorch_2_13(self):
        def outcome(module, protocol):
            module.serialization.clear_safe_globals()
            custom_path = (_PickleTarget, "custom.PickleTarget")
            module.serialization.add_safe_globals([_PickleTarget, custom_path])
            snapshot = module.serialization.get_safe_globals()
            shallow = copy.copy(snapshot)
            restored = pickle.loads(pickle.dumps(snapshot, protocol=protocol))
            return (
                type(snapshot) is list,
                shallow is not snapshot,
                any(entry is _PickleTarget for entry in shallow),
                any(entry is custom_path for entry in shallow),
                set(restored) == {_PickleTarget, custom_path},
                any(entry is _PickleTarget for entry in restored),
            )

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                actual = outcome(torch, protocol)
                expected = outcome(reference_torch, protocol)
                self.assertEqual(actual, expected)
                self.assertEqual(actual, (True, True, True, True, True, True))

    def test_argument_binding_errors_match_pytorch_2_13(self):
        actual = torch.serialization
        expected = reference_torch.serialization
        cases = (
            (
                lambda: actual.get_safe_globals(None),
                lambda: expected.get_safe_globals(None),
            ),
            (
                lambda: actual.get_safe_globals(value=None),
                lambda: expected.get_safe_globals(value=None),
            ),
            (
                lambda: actual.clear_safe_globals(None),
                lambda: expected.clear_safe_globals(None),
            ),
            (
                lambda: actual.clear_safe_globals(value=None),
                lambda: expected.clear_safe_globals(value=None),
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
                lambda: actual.add_safe_globals(values=[]),
                lambda: expected.add_safe_globals(values=[]),
            ),
            (
                lambda: actual.add_safe_globals([], safe_globals=[]),
                lambda: expected.add_safe_globals([], safe_globals=[]),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_reload_and_reimport_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reimport_outcome(torch),
            self.reimport_outcome(reference_torch),
        )

    def test_context_manager_and_weights_only_loading_remain_unsupported(self):
        actual = torch.serialization
        expected = reference_torch.serialization
        self.assertFalse(hasattr(actual, "safe_globals"))
        self.assertNotIn("safe_globals", actual.__all__)
        self.assertTrue(hasattr(expected, "safe_globals"))
        self.assertIn("safe_globals", expected.__all__)
        for name in ("save", "load", "get_unsafe_globals_in_checkpoint"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual, name))
                self.assertNotIn(name, actual.__all__)


if __name__ == "__main__":
    unittest.main()
