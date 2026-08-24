import copy
import importlib
import inspect
import pickle
import pickletools
import types
import typing
import unittest

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
        self.original_actual = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        self.original_expected = (
            reference_torch.are_deterministic_algorithms_enabled(),
            reference_torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        torch.use_deterministic_algorithms(False)
        reference_torch.use_deterministic_algorithms(False)

    def tearDown(self):
        torch.use_deterministic_algorithms(
            self.original_actual[0],
            warn_only=self.original_actual[1],
        )
        reference_torch.use_deterministic_algorithms(
            self.original_expected[0],
            warn_only=self.original_expected[1],
        )

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
            module.get_deterministic_debug_mode(),
        )

    def operation_outcome(self, module):
        leaf = module.tensor([1.0, -2.0, 3.0], requires_grad=True)
        output = ((leaf + 2.0) * leaf).sum()
        output.backward()
        return output.item(), leaf.grad.tolist()

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

    def test_state_transitions_and_supported_operations_match_pytorch_2_13(self):
        actual_baseline = self.operation_outcome(torch)
        expected_baseline = self.operation_outcome(reference_torch)
        self.assertEqual(actual_baseline, expected_baseline)

        for enabled, warn_only in (
            (False, False),
            (False, True),
            (True, True),
            (True, False),
            (False, False),
        ):
            with self.subTest(enabled=enabled, warn_only=warn_only):
                actual_result = torch.use_deterministic_algorithms(
                    enabled,
                    warn_only=warn_only,
                )
                expected_result = reference_torch.use_deterministic_algorithms(
                    enabled,
                    warn_only=warn_only,
                )
                self.assertIs(actual_result, expected_result)
                self.assertEqual(self.state(torch), self.state(reference_torch))
                self.assertEqual(self.operation_outcome(torch), actual_baseline)
                self.assertEqual(
                    self.operation_outcome(reference_torch),
                    expected_baseline,
                )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs")
        expected_module = importlib.import_module("torch")
        actual = actual_module.use_deterministic_algorithms
        expected = expected_module.use_deterministic_algorithms

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

    def test_binding_and_strict_bool_errors_match_pytorch_2_13(self):
        actual = torch.use_deterministic_algorithms
        expected = reference_torch.use_deterministic_algorithms
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(True, False), lambda: expected(True, False)),
            (
                lambda: actual(True, unknown=False),
                lambda: expected(True, unknown=False),
            ),
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(1), lambda: expected(1)),
            (lambda: actual("true"), lambda: expected("true")),
            (
                lambda: actual(True, warn_only=None),
                lambda: expected(True, warn_only=None),
            ),
            (
                lambda: actual(True, warn_only=0),
                lambda: expected(True, warn_only=0),
            ),
            (
                lambda: actual(True, warn_only="false"),
                lambda: expected(True, warn_only="false"),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertEqual(self.state(torch), (False, False, 0))
                self.assertEqual(self.state(reference_torch), (False, False, 0))

        self.assertIs(actual(mode=True, warn_only=True), None)
        self.assertIs(expected(mode=True, warn_only=True), None)
        self.assertEqual(self.state(torch), self.state(reference_torch))


if __name__ == "__main__":
    unittest.main()
