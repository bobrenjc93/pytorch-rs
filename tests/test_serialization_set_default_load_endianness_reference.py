import copy
import enum
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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SerializationSetDefaultLoadEndiannessReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "serialization load-endianness setter differentials require "
                "pinned PyTorch 2.13.0"
            )

    def setUp(self):
        for module in (torch, reference_torch):
            module.serialization.set_default_load_endianness(None)

    def tearDown(self):
        for module in (torch, reference_torch):
            module.serialization.set_default_load_endianness(None)

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
        setter = serialization.set_default_load_endianness
        getter = serialization.get_default_load_endianness
        outcomes = []
        for value in (None, *serialization.LoadEndianness):
            result = setter(value)
            observed = getter()
            outcomes.append(
                (
                    result is None,
                    observed is value,
                    None if observed is None else observed.name,
                    None if observed is None else observed.value,
                )
            )

        value = serialization.LoadEndianness.LITTLE
        result = setter(endianness=value)
        outcomes.append((result is None, getter() is value))
        return outcomes

    def threaded_outcome(self, module):
        serialization = module.serialization
        setter = serialization.set_default_load_endianness
        getter = serialization.get_default_load_endianness
        load_endianness = serialization.LoadEndianness
        worker_count = 8
        start = threading.Barrier(worker_count)
        finish = threading.Barrier(worker_count)
        observations = [None] * worker_count
        errors = []

        def worker(index):
            try:
                start.wait(timeout=10)
                result = setter(load_endianness.BIG)
                finish.wait(timeout=10)
                observations[index] = (
                    result is None,
                    getter() is load_endianness.BIG,
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        return observations, errors

    def reload_outcome(self, module):
        original_module = module.serialization
        original_class = original_module.LoadEndianness
        original_getter = original_module.get_default_load_endianness
        original_setter = original_module.set_default_load_endianness
        module_name = original_module.__name__

        original_setter(original_class.LITTLE)
        reloaded = importlib.reload(original_module)
        reloaded_class = reloaded.LoadEndianness
        reloaded_getter = reloaded.get_default_load_endianness
        reloaded_setter = reloaded.set_default_load_endianness
        after_reload = (
            reloaded is original_module,
            module.serialization is original_module,
            reloaded_class is not original_class,
            reloaded_getter is not original_getter,
            reloaded_setter is not original_setter,
            original_getter() is original_class.LITTLE,
            reloaded_getter() is original_class.LITTLE,
        )

        stale_error = None
        try:
            original_setter(original_class.BIG)
        except Exception as error:
            stale_error = (type(error).__name__, str(error), error.args)
        state_after_stale_error = reloaded_getter() is original_class.LITTLE
        original_result = original_setter(reloaded_class.BIG)
        state_after_original_setter = reloaded_getter() is reloaded_class.BIG

        try:
            removed = sys.modules.pop(module_name)
            replacement_module = importlib.import_module(module_name)
            replacement_class = replacement_module.LoadEndianness
            after_reimport = (
                removed is original_module,
                replacement_module is not original_module,
                sys.modules[module_name] is replacement_module,
                module.serialization is replacement_module,
                replacement_class is not reloaded_class,
                original_getter() is reloaded_class.BIG,
                replacement_module.get_default_load_endianness()
                is reloaded_class.BIG,
            )

            replacement_stale_error = None
            try:
                replacement_module.set_default_load_endianness(
                    reloaded_class.NATIVE
                )
            except Exception as error:
                replacement_stale_error = (
                    type(error).__name__,
                    str(error),
                    error.args,
                )
            state_after_replacement_stale_error = (
                original_getter() is reloaded_class.BIG
            )
            replacement_result = replacement_module.set_default_load_endianness(
                replacement_class.NATIVE
            )
            shared_replacement_state = (
                original_getter() is replacement_class.NATIVE,
                reloaded_getter() is replacement_class.NATIVE,
                replacement_module.get_default_load_endianness()
                is replacement_class.NATIVE,
            )
        finally:
            sys.modules[module_name] = original_module
            module.serialization = original_module
            original_module.set_default_load_endianness(None)

        return (
            after_reload,
            stale_error,
            state_after_stale_error,
            original_result is None,
            state_after_original_setter,
            after_reimport,
            replacement_stale_error,
            state_after_replacement_stale_error,
            replacement_result is None,
            shared_replacement_state,
        )

    def test_mutation_and_identity_match_pytorch_2_13(self):
        self.assertEqual(
            self.mutation_outcome(torch),
            self.mutation_outcome(reference_torch),
        )

    def test_threaded_setter_calls_match_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

    def test_signature_metadata_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs.serialization")
        expected_module = importlib.import_module("torch.serialization")
        actual = actual_module.set_default_load_endianness
        expected = expected_module.set_default_load_endianness

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
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

    def test_imports_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        actual = actual_module.set_default_load_endianness
        expected = expected_module.set_default_load_endianness

        self.assertEqual(
            actual_module.__all__,
            [
                name
                for name in expected_module.__all__
                if name
                in {
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
            ],
        )

        actual_direct_import = {}
        expected_direct_import = {}
        exec(
            "from torch_rs.serialization import set_default_load_endianness",
            actual_direct_import,
        )
        exec(
            "from torch.serialization import set_default_load_endianness",
            expected_direct_import,
        )
        self.assertIs(actual_direct_import[actual.__name__], actual)
        self.assertIs(expected_direct_import[expected.__name__], expected)

        for module, function in (
            (actual_module, actual),
            (expected_module, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace[function.__name__], function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        self.assertFalse(hasattr(torch, actual.__name__))
        self.assertFalse(hasattr(reference_torch, expected.__name__))
        self.assertNotIn(actual.__name__, torch.__all__)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol)),
                    expected,
                )
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_invalid_types_call_errors_and_state_preservation_match(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        actual = actual_module.set_default_load_endianness
        expected = expected_module.set_default_load_endianness
        actual_getter = actual_module.get_default_load_endianness
        expected_getter = expected_module.get_default_load_endianness
        actual_other = enum.Enum("LoadEndianness", {"NATIVE": 1}).NATIVE
        expected_other = enum.Enum("LoadEndianness", {"NATIVE": 1}).NATIVE

        actual(actual_module.LoadEndianness.LITTLE)
        expected(expected_module.LoadEndianness.LITTLE)
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(
                    None,
                    endianness=actual_module.LoadEndianness.NATIVE,
                ),
                lambda: expected(
                    None,
                    endianness=expected_module.LoadEndianness.NATIVE,
                ),
            ),
            (lambda: actual(1), lambda: expected(1)),
            (lambda: actual(2), lambda: expected(2)),
            (lambda: actual(3), lambda: expected(3)),
            (lambda: actual(True), lambda: expected(True)),
            (lambda: actual(False), lambda: expected(False)),
            (lambda: actual("native"), lambda: expected("native")),
            (lambda: actual(actual_other), lambda: expected(expected_other)),
            (
                lambda: actual(actual_module.LoadEndianness),
                lambda: expected(expected_module.LoadEndianness),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                actual_before = actual_getter()
                expected_before = expected_getter()
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(actual_getter(), actual_before)
                self.assertIs(expected_getter(), expected_before)

    def test_reload_and_reimport_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )

    def test_save_and_load_remain_unsupported(self):
        for name in ("save", "load"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(reference_torch.serialization, name))
                self.assertFalse(hasattr(torch.serialization, name))
                self.assertNotIn(name, torch.serialization.__all__)


if __name__ == "__main__":
    unittest.main()
