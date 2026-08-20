import contextlib
import copy
import importlib
import inspect
import mmap
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
class SerializationDefaultMmapOptionsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "serialization mmap option differentials require pinned "
                "PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def supported_state_outcome(self, module):
        function = module.serialization.get_default_mmap_options

        def query_outcome():
            before = module.is_grad_enabled()
            result = function()
            after = module.is_grad_enabled()
            return (
                before,
                result,
                type(result).__module__,
                type(result).__qualname__,
                after,
            )

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

    def test_supported_default_threaded_and_grad_states_match_pytorch_2_13(self):
        self.assertEqual(
            self.supported_state_outcome(torch),
            self.supported_state_outcome(reference_torch),
        )

        expected = getattr(mmap, "MAP_PRIVATE", None)
        self.assertIs(torch.serialization.get_default_mmap_options(), expected)
        self.assertIs(
            reference_torch.serialization.get_default_mmap_options(),
            expected,
        )

    def test_map_private_and_map_shared_mutation_match_pytorch_2_13(self):
        actual = torch.serialization.get_default_mmap_options
        expected = reference_torch.serialization.get_default_mmap_options
        actual_setter = torch.serialization.set_default_mmap_options
        expected_setter = reference_torch.serialization.set_default_mmap_options
        private = getattr(mmap, "MAP_PRIVATE", None)
        shared = getattr(mmap, "MAP_SHARED", None)

        if private is None or shared is None:
            self.assertIs(actual(), None)
            self.assertIs(expected(), None)
            return

        actual_original = actual()
        expected_original = expected()
        try:
            actual_setter(private)
            expected_setter(private)
            actual_states = [actual()]
            expected_states = [expected()]

            actual_setter(shared)
            expected_setter(shared)
            actual_states.append(actual())
            expected_states.append(expected())

            actual_setter(private)
            expected_setter(private)
            actual_states.append(actual())
            expected_states.append(expected())
        finally:
            actual_setter(actual_original)
            expected_setter(expected_original)

        self.assertEqual(actual_states, [private, shared, private])
        self.assertEqual(expected_states, [private, shared, private])
        self.assertIs(torch.is_grad_enabled(), True)
        self.assertIs(reference_torch.is_grad_enabled(), True)

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs.serialization")
        expected_module = importlib.import_module("torch.serialization")
        actual = actual_module.get_default_mmap_options
        expected = expected_module.get_default_mmap_options

        self.assertIs(torch.serialization, actual_module)
        self.assertIs(reference_torch.serialization, expected_module)
        self.assertIsNone(actual_module.__doc__)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
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
        actual = actual_module.get_default_mmap_options
        expected = expected_module.get_default_mmap_options
        supported_names = (
            "get_crc32_options",
            "set_crc32_options",
            "get_default_mmap_options",
            "set_default_mmap_options",
        )

        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported_names],
        )

        actual_package_import = {}
        expected_package_import = {}
        exec("from torch_rs import serialization", actual_package_import)
        exec("from torch import serialization", expected_package_import)
        self.assertIs(actual_package_import["serialization"], actual_module)
        self.assertIs(expected_package_import["serialization"], expected_module)

        actual_direct_import = {}
        expected_direct_import = {}
        exec(
            "from torch_rs.serialization import get_default_mmap_options",
            actual_direct_import,
        )
        exec(
            "from torch.serialization import get_default_mmap_options",
            expected_direct_import,
        )
        self.assertIs(actual_direct_import["get_default_mmap_options"], actual)
        self.assertIs(expected_direct_import["get_default_mmap_options"], expected)

        for module, function in (
            (actual_module, actual),
            (expected_module, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["get_default_mmap_options"], function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("get_default_mmap_options", namespace)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.serialization.get_default_mmap_options
        expected = reference_torch.serialization.get_default_mmap_options
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                actual_before = actual()
                expected_before = expected()
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(actual(), actual_before)
                self.assertIs(expected(), expected_before)

        self.assertEqual(actual(**{}), expected(**{}))

    def test_setter_is_supported_while_save_and_load_remain_unsupported(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization

        self.assertTrue(hasattr(expected_module, "set_default_mmap_options"))
        self.assertTrue(hasattr(actual_module, "set_default_mmap_options"))
        self.assertIn("set_default_mmap_options", actual_module.__all__)
        for name in ("save", "load"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(expected_module, name))
                self.assertFalse(hasattr(actual_module, name))
                self.assertNotIn(name, actual_module.__all__)

        self.assertFalse(hasattr(torch, "get_default_mmap_options"))
        self.assertFalse(hasattr(reference_torch, "get_default_mmap_options"))
        self.assertNotIn("get_default_mmap_options", torch.__all__)


if __name__ == "__main__":
    unittest.main()
