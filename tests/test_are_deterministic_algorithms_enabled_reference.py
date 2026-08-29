import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
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
class AreDeterministicAlgorithmsEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "are_deterministic_algorithms_enabled differentials require "
                "pinned PyTorch 2.13.0"
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
        function = module.are_deterministic_algorithms_enabled

        def query_outcome():
            before = module.is_grad_enabled()
            result = function()
            after = module.is_grad_enabled()
            return before, result is False, after

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

    def test_reference_only_mutation_bounds_the_unsupported_true_state(self):
        actual = torch.are_deterministic_algorithms_enabled
        expected = reference_torch.are_deterministic_algorithms_enabled
        original_enabled = expected()
        original_warn_only = (
            reference_torch.is_deterministic_algorithms_warn_only_enabled()
        )

        try:
            reference_torch.use_deterministic_algorithms(False)
            actual_states = [actual()]
            expected_states = [expected()]

            reference_torch.use_deterministic_algorithms(True)
            actual_states.append(actual())
            expected_states.append(expected())

            reference_torch.use_deterministic_algorithms(False)
            actual_states.append(actual())
            expected_states.append(expected())
        finally:
            reference_torch.use_deterministic_algorithms(
                original_enabled,
                warn_only=original_warn_only,
            )

        for state in actual_states:
            self.assertIs(state, False)
        self.assertEqual(expected_states, [False, True, False])
        for state in expected_states:
            self.assertIs(type(state), bool)

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs")
        expected_module = importlib.import_module("torch")
        actual = actual_module.are_deterministic_algorithms_enabled
        expected = expected_module.are_deterministic_algorithms_enabled

        self.assertIs(torch, actual_module)
        self.assertIs(reference_torch, expected_module)
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

    def test_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual = torch.are_deterministic_algorithms_enabled
        expected = reference_torch.are_deterministic_algorithms_enabled

        self.assertEqual(
            torch.__all__.count("are_deterministic_algorithms_enabled"),
            reference_torch.__all__.count("are_deterministic_algorithms_enabled"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["are_deterministic_algorithms_enabled"], function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.are_deterministic_algorithms_enabled
        expected = reference_torch.are_deterministic_algorithms_enabled
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
                self.assert_error_matches(actual_call, expected_call)

    def test_default_only_deterministic_setters_support_boundary(self):
        self.assertTrue(hasattr(reference_torch, "use_deterministic_algorithms"))
        self.assertTrue(hasattr(torch, "use_deterministic_algorithms"))
        self.assertEqual(
            torch.__all__.count("use_deterministic_algorithms"),
            reference_torch.__all__.count("use_deterministic_algorithms"),
        )
        self.assertTrue(hasattr(reference_torch, "set_deterministic_debug_mode"))
        self.assertTrue(hasattr(torch, "set_deterministic_debug_mode"))
        self.assertEqual(torch.__all__.count("set_deterministic_debug_mode"), 1)


if __name__ == "__main__":
    unittest.main()
