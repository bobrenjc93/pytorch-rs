import copy
import inspect
import pickle
import pickletools
import threading
import types
import unittest
import warnings

import numpy as np

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class UseDeterministicAlgorithmsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "use_deterministic_algorithms differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.original_states = {
            module: (
                module.are_deterministic_algorithms_enabled(),
                module.is_deterministic_algorithms_warn_only_enabled(),
            )
            for module in (torch, reference_torch)
        }
        for module in self.original_states:
            module.use_deterministic_algorithms(False, warn_only=False)

    def tearDown(self):
        for module, (mode, warn_only) in self.original_states.items():
            module.use_deterministic_algorithms(mode, warn_only=warn_only)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def state(self, module):
        return (
            module.are_deterministic_algorithms_enabled(),
            module.is_deterministic_algorithms_warn_only_enabled(),
        )

    def state_transitions(self, module):
        transitions = []
        for mode, warn_only, omit_warn_only in (
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (False, True, False),
            (True, False, True),
            (False, False, True),
        ):
            if omit_warn_only:
                result = module.use_deterministic_algorithms(mode)
            else:
                result = module.use_deterministic_algorithms(
                    mode,
                    warn_only=warn_only,
                )
            state = self.state(module)
            transitions.append(
                (
                    result is None,
                    tuple(type(value) is bool for value in state),
                    state,
                )
            )
        return transitions

    def test_state_transitions_match_pytorch_2_13(self):
        actual = self.state_transitions(torch)
        expected = self.state_transitions(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(
            [entry[-1] for entry in actual],
            [
                (False, False),
                (True, False),
                (True, True),
                (False, True),
                (True, False),
                (False, False),
            ],
        )

    def threaded_outcome(self, module):
        module.use_deterministic_algorithms(False, warn_only=True)
        observations = []
        errors = []

        def worker():
            try:
                observations.append(self.state(module))
                result = module.use_deterministic_algorithms(
                    True,
                    warn_only=False,
                )
                observations.append((result is None, self.state(module)))
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        return observations, errors, self.state(module)

    def test_cross_thread_visibility_matches_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

    def cpu_outcome(self, module, mode, warn_only):
        module.use_deterministic_algorithms(mode, warn_only=warn_only)

        def run_once():
            left = module.tensor(
                [[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]],
                requires_grad=True,
            )
            right = module.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            weights = module.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
            product = (left + 1.5).relu().matmul(right)
            zeroed = module.nn.functional.dropout(
                product,
                p=1.0,
                training=True,
                inplace=False,
            )
            (left * weights).sum().backward()
            return product.tolist(), zeroed.tolist(), left.grad.tolist()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = run_once()
            second = run_once()
        return (
            first,
            second,
            tuple((warning.category.__name__, str(warning.message)) for warning in caught),
        )

    def test_supported_cpu_operations_match_and_remain_deterministic(self):
        for mode, warn_only in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            with self.subTest(mode=mode, warn_only=warn_only):
                actual = self.cpu_outcome(torch, mode, warn_only)
                expected = self.cpu_outcome(reference_torch, mode, warn_only)
                self.assertEqual(actual, expected)
                self.assertEqual(actual[0], actual[1])
                self.assertEqual(actual[2], ())

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

    def test_callable_metadata_documentation_and_exports_match(self):
        actual = torch.use_deterministic_algorithms
        expected = reference_torch.use_deterministic_algorithms

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(
            torch.__all__.count("use_deterministic_algorithms"),
            reference_torch.__all__.count("use_deterministic_algorithms"),
        )

        for module, function in ((torch, actual), (reference_torch, expected)):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["use_deterministic_algorithms"], function)
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

    def test_python_argument_binding_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.use_deterministic_algorithms(),
                lambda: reference_torch.use_deterministic_algorithms(),
            ),
            (
                lambda: torch.use_deterministic_algorithms(True, False),
                lambda: reference_torch.use_deterministic_algorithms(True, False),
            ),
            (
                lambda: torch.use_deterministic_algorithms(warn_only=True),
                lambda: reference_torch.use_deterministic_algorithms(warn_only=True),
            ),
            (
                lambda: torch.use_deterministic_algorithms(True, mode=False),
                lambda: reference_torch.use_deterministic_algorithms(True, mode=False),
            ),
            (
                lambda: torch.use_deterministic_algorithms(True, enabled=False),
                lambda: reference_torch.use_deterministic_algorithms(
                    True,
                    enabled=False,
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertEqual(self.state(torch), (False, False))
                self.assertEqual(self.state(reference_torch), (False, False))

    def test_bool_validation_errors_and_state_preservation_match(self):
        cases = (
            (None, None),
            (1, 1),
            (1.0, 1.0),
            ("true", "true"),
            (np.bool_(True), np.bool_(True)),
            (torch.tensor(1.0), reference_torch.tensor(1.0)),
            (torch.float32, reference_torch.float32),
            (torch.device("cpu"), reference_torch.device("cpu")),
        )
        for case, (actual_value, expected_value) in enumerate(cases):
            for argument in ("mode", "warn_only"):
                with self.subTest(case=case, argument=argument):
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    reference_torch.use_deterministic_algorithms(
                        True,
                        warn_only=True,
                    )
                    if argument == "mode":
                        actual_call = lambda value=actual_value: (
                            torch.use_deterministic_algorithms(
                                value,
                                warn_only=False,
                            )
                        )
                        expected_call = lambda value=expected_value: (
                            reference_torch.use_deterministic_algorithms(
                                value,
                                warn_only=False,
                            )
                        )
                    else:
                        actual_call = lambda value=actual_value: (
                            torch.use_deterministic_algorithms(
                                False,
                                warn_only=value,
                            )
                        )
                        expected_call = lambda value=expected_value: (
                            reference_torch.use_deterministic_algorithms(
                                False,
                                warn_only=value,
                            )
                        )
                    self.assert_error_matches(actual_call, expected_call)
                    self.assertEqual(self.state(torch), (True, True))
                    self.assertEqual(self.state(reference_torch), (True, True))

    def test_debug_mode_apis_remain_deliberately_unsupported(self):
        for name in (
            "set_deterministic_debug_mode",
            "get_deterministic_debug_mode",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
                self.assertTrue(hasattr(reference_torch, name))
                self.assertIn(name, reference_torch.__all__)


if __name__ == "__main__":
    unittest.main()
