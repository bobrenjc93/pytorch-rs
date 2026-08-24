import copy
import importlib
import inspect
import pickle
import pickletools
import types
import typing
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _RejectTruthiness:
    calls = 0

    def __bool__(self):
        type(self).calls += 1
        raise AssertionError("deterministic mode must not request truthiness")


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
        self.actual_original = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        self.expected_original = (
            reference_torch.are_deterministic_algorithms_enabled(),
            reference_torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        torch.use_deterministic_algorithms(False, warn_only=False)
        reference_torch.use_deterministic_algorithms(False, warn_only=False)

    def tearDown(self):
        torch.use_deterministic_algorithms(
            self.actual_original[0],
            warn_only=self.actual_original[1],
        )
        reference_torch.use_deterministic_algorithms(
            self.expected_original[0],
            warn_only=self.expected_original[1],
        )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def state_and_operation_outcome(self, module, mode, warn_only):
        before = module.is_grad_enabled()
        returned = module.use_deterministic_algorithms(mode, warn_only=warn_only)
        after = module.is_grad_enabled()
        values = module.tensor([-2.0, 1.0, 3.0], requires_grad=True)
        result = (values * 2.0).relu()
        total = result.sum()
        total.backward()
        with module.no_grad():
            no_grad_before = module.is_grad_enabled()
            no_grad_returned = module.use_deterministic_algorithms(
                mode,
                warn_only=warn_only,
            )
            no_grad_values = module.relu(module.tensor([-1.0, 2.0])).tolist()
            no_grad_after = module.is_grad_enabled()
        return (
            returned,
            before,
            after,
            module.are_deterministic_algorithms_enabled(),
            module.is_deterministic_algorithms_warn_only_enabled(),
            module.get_deterministic_debug_mode(),
            result.tolist(),
            total.item(),
            values.grad.tolist(),
            no_grad_before,
            no_grad_returned,
            no_grad_values,
            no_grad_after,
            module.is_grad_enabled(),
        )

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

    def test_state_grad_and_deterministic_operations_match_pytorch_2_13(self):
        for mode, warn_only in (
            (False, False),
            (False, True),
            (True, True),
            (True, False),
        ):
            with self.subTest(mode=mode, warn_only=warn_only):
                self.assertEqual(
                    self.state_and_operation_outcome(torch, mode, warn_only),
                    self.state_and_operation_outcome(
                        reference_torch,
                        mode,
                        warn_only,
                    ),
                )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs")
        expected_module = importlib.import_module("torch")
        actual = actual_module.use_deterministic_algorithms
        expected = expected_module.use_deterministic_algorithms

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
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
        actual = torch.use_deterministic_algorithms
        expected = reference_torch.use_deterministic_algorithms
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

    def test_binding_and_strict_boolean_errors_match_pytorch_2_13(self):
        binding_cases = (
            (
                lambda: torch.use_deterministic_algorithms(),
                lambda: reference_torch.use_deterministic_algorithms(),
            ),
            (
                lambda: torch.use_deterministic_algorithms(True, False),
                lambda: reference_torch.use_deterministic_algorithms(True, False),
            ),
            (
                lambda: torch.use_deterministic_algorithms(True, enabled=False),
                lambda: reference_torch.use_deterministic_algorithms(
                    True,
                    enabled=False,
                ),
            ),
            (
                lambda: torch.use_deterministic_algorithms(True, mode=False),
                lambda: reference_torch.use_deterministic_algorithms(
                    True,
                    mode=False,
                ),
            ),
            (
                lambda: torch.use_deterministic_algorithms(warn_only=True),
                lambda: reference_torch.use_deterministic_algorithms(warn_only=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(binding_cases):
            with self.subTest(kind="binding", case=case):
                self.assert_error_matches(actual_call, expected_call)

        invalid_pairs = (
            (None, None),
            (0, 0),
            (1, 1),
            (0.0, 0.0),
            ("", ""),
            ([], []),
            (np.bool_(True), np.bool_(True)),
            (_RejectTruthiness(), _RejectTruthiness()),
            (torch.tensor(True), reference_torch.tensor(True)),
            (torch.float32, reference_torch.float32),
            (torch.device("cpu"), reference_torch.device("cpu")),
        )
        _RejectTruthiness.calls = 0
        for case, (actual_value, expected_value) in enumerate(invalid_pairs):
            with self.subTest(kind="mode", case=case):
                self.assert_error_matches(
                    lambda actual_value=actual_value: torch.use_deterministic_algorithms(
                        actual_value
                    ),
                    lambda expected_value=expected_value: reference_torch.use_deterministic_algorithms(
                        expected_value
                    ),
                )
                self.assertEqual(
                    (
                        torch.are_deterministic_algorithms_enabled(),
                        torch.is_deterministic_algorithms_warn_only_enabled(),
                    ),
                    (False, False),
                )
                self.assertEqual(
                    (
                        reference_torch.are_deterministic_algorithms_enabled(),
                        reference_torch.is_deterministic_algorithms_warn_only_enabled(),
                    ),
                    (False, False),
                )

            with self.subTest(kind="warn_only", case=case):
                self.assert_error_matches(
                    lambda actual_value=actual_value: torch.use_deterministic_algorithms(
                        True,
                        warn_only=actual_value,
                    ),
                    lambda expected_value=expected_value: reference_torch.use_deterministic_algorithms(
                        True,
                        warn_only=expected_value,
                    ),
                )
                self.assertEqual(
                    (
                        torch.are_deterministic_algorithms_enabled(),
                        torch.is_deterministic_algorithms_warn_only_enabled(),
                    ),
                    (False, False),
                )
                self.assertEqual(
                    (
                        reference_torch.are_deterministic_algorithms_enabled(),
                        reference_torch.is_deterministic_algorithms_warn_only_enabled(),
                    ),
                    (False, False),
                )

        self.assertEqual(_RejectTruthiness.calls, 0)


if __name__ == "__main__":
    unittest.main()
