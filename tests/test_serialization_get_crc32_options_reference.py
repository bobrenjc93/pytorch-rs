import contextlib
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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SerializationCrc32OptionsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "serialization CRC32 option differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.original_crc32_options = {
            module: module.serialization.get_crc32_options()
            for module in (torch, reference_torch)
        }
        for module in self.original_crc32_options:
            module.serialization.set_crc32_options(True)

    def tearDown(self):
        for module, value in self.original_crc32_options.items():
            module.serialization.set_crc32_options(value)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def supported_state_outcome(self, module):
        function = module.serialization.get_crc32_options

        def query_outcome():
            before = module.is_grad_enabled()
            result = function()
            after = module.is_grad_enabled()
            return before, result is True, after

        states = [query_outcome()]
        with module.no_grad():
            states.append(query_outcome())
            with module.no_grad():
                states.append(query_outcome())
            states.append(query_outcome())
        states.append(query_outcome())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = query_outcome()
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
        self.assertEqual(errors, [])
        return states, worker_states

    def mutation_outcome(self, module, values):
        setter = module.serialization.set_crc32_options
        getter = module.serialization.get_crc32_options
        outcomes = []
        for value in values:
            result = setter(value)
            observed = getter()
            outcomes.append(
                (
                    result is None,
                    observed is value,
                    type(observed).__module__,
                    type(observed).__qualname__,
                )
            )

        keyword_value = []
        result = setter(compute_crc32=keyword_value)
        observed = getter()
        keyword_value.append("updated")
        outcomes.append(
            (
                result is None,
                observed is keyword_value,
                observed == ["updated"],
            )
        )
        return outcomes

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

    def test_default_threaded_and_grad_states_match_pytorch_2_13(self):
        self.assertEqual(
            self.supported_state_outcome(torch),
            self.supported_state_outcome(reference_torch),
        )
        self.assertIs(torch.serialization.get_crc32_options(), True)
        self.assertIs(reference_torch.serialization.get_crc32_options(), True)

    def test_runtime_value_acceptance_and_identity_match_pytorch_2_13(self):
        class RejectBoolConversion:
            def __bool__(self):
                raise AssertionError("set_crc32_options must not call bool")

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
            [],
            {},
            object(),
            RejectBoolConversion,
            RejectBoolConversion(),
        )
        actual = self.mutation_outcome(torch, values)
        expected = self.mutation_outcome(reference_torch, values)
        self.assertEqual(actual, expected)
        self.assertTrue(all(outcome[0] and outcome[1] for outcome in actual))

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_serialization = importlib.import_module("torch_rs.serialization")
        expected_serialization = importlib.import_module("torch.serialization")

        self.assertIs(torch.serialization, actual_serialization)
        self.assertIs(reference_torch.serialization, expected_serialization)
        self.assertIsNone(actual_serialization.__doc__)
        self.assertEqual(actual_serialization.__doc__, expected_serialization.__doc__)
        for name in ("get_crc32_options", "set_crc32_options"):
            with self.subTest(name=name):
                actual = getattr(actual_serialization, name)
                expected = getattr(expected_serialization, name)
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
                self.assertIs(inspect.getmodule(actual), actual_serialization)
                self.assertIs(inspect.getmodule(expected), expected_serialization)
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )

    def test_imports_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_serialization = torch.serialization
        expected_serialization = reference_torch.serialization
        supported_names = ("get_crc32_options", "set_crc32_options")

        self.assertIs(sys.modules["torch_rs.serialization"], actual_serialization)
        self.assertIs(sys.modules["torch.serialization"], expected_serialization)
        self.assertEqual(
            actual_serialization.__all__,
            [
                name
                for name in expected_serialization.__all__
                if name in supported_names
            ],
        )
        self.assertEqual(
            torch.__all__.count("serialization"),
            reference_torch.__all__.count("serialization"),
        )
        for name in supported_names:
            self.assertEqual(
                torch.__all__.count(name),
                reference_torch.__all__.count(name),
            )

        actual_package_import = {}
        expected_package_import = {}
        exec("from torch_rs import serialization", actual_package_import)
        exec("from torch import serialization", expected_package_import)
        self.assertIs(actual_package_import["serialization"], actual_serialization)
        self.assertIs(
            expected_package_import["serialization"], expected_serialization
        )

        actual_direct_import = {}
        expected_direct_import = {}
        exec(
            "from torch_rs.serialization import "
            "get_crc32_options, set_crc32_options",
            actual_direct_import,
        )
        exec(
            "from torch.serialization import "
            "get_crc32_options, set_crc32_options",
            expected_direct_import,
        )
        for name in supported_names:
            self.assertIs(
                actual_direct_import[name],
                getattr(actual_serialization, name),
            )
            self.assertIs(
                expected_direct_import[name],
                getattr(expected_serialization, name),
            )

        for module in (actual_serialization, expected_serialization):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in supported_names:
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("serialization", namespace)
            for name in supported_names:
                self.assertNotIn(name, namespace)

        for name in supported_names:
            actual = getattr(actual_serialization, name)
            expected = getattr(expected_serialization, name)
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

    def test_argument_errors_and_state_preservation_match_pytorch_2_13(self):
        actual_getter = torch.serialization.get_crc32_options
        expected_getter = reference_torch.serialization.get_crc32_options
        actual_setter = torch.serialization.set_crc32_options
        expected_setter = reference_torch.serialization.set_crc32_options
        cases = (
            (lambda: actual_getter(None), lambda: expected_getter(None)),
            (
                lambda: actual_getter(None, None),
                lambda: expected_getter(None, None),
            ),
            (
                lambda: actual_getter(enabled=True),
                lambda: expected_getter(enabled=True),
            ),
            (
                lambda: actual_getter(None, enabled=True),
                lambda: expected_getter(None, enabled=True),
            ),
            (lambda: actual_setter(), lambda: expected_setter()),
            (
                lambda: actual_setter(True, False),
                lambda: expected_setter(True, False),
            ),
            (
                lambda: actual_setter(enabled=True),
                lambda: expected_setter(enabled=True),
            ),
            (
                lambda: actual_setter(True, compute_crc32=False),
                lambda: expected_setter(True, compute_crc32=False),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                actual_before = actual_getter()
                expected_before = expected_getter()
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(actual_getter(), actual_before)
                self.assertIs(expected_getter(), expected_before)

    def test_save_load_and_other_serialization_apis_remain_unsupported(self):
        actual_serialization = torch.serialization
        expected_serialization = reference_torch.serialization
        actual_public = {
            name for name in vars(actual_serialization) if not name.startswith("_")
        }

        self.assertEqual(
            actual_public,
            {"get_crc32_options", "set_crc32_options"},
        )
        unsupported = set(expected_serialization.__all__) - actual_public
        self.assertTrue(
            {
                "get_default_load_endianness",
                "set_default_load_endianness",
                "get_default_mmap_options",
                "set_default_mmap_options",
                "save",
                "load",
            }.issubset(unsupported)
        )
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual_serialization, name))

        for name in ("save", "load"):
            with self.subTest(top_level_name=name):
                self.assertTrue(hasattr(reference_torch, name))
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
        for name in ("get_crc32_options", "set_crc32_options"):
            with self.subTest(top_level_name=name):
                self.assertFalse(hasattr(reference_torch, name))
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)


if __name__ == "__main__":
    unittest.main()
