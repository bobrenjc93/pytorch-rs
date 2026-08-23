import copy
import inspect
import json
import pickle
import pickletools
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("set_warn_always must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SetWarnAlwaysReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "set_warn_always differentials require pinned PyTorch 2.13.0"
            )

    def setUp(self):
        self.original_actual = torch.is_warn_always_enabled()
        self.original_expected = reference_torch.is_warn_always_enabled()
        torch.set_warn_always(False)
        reference_torch.set_warn_always(False)

    def tearDown(self):
        torch.set_warn_always(self.original_actual)
        reference_torch.set_warn_always(self.original_expected)

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

    def test_state_transitions_and_return_values_match_pytorch_2_13(self):
        for state in (False, True, True, False, False, True, False):
            with self.subTest(state=state):
                actual_result = torch.set_warn_always(state)
                expected_result = reference_torch.set_warn_always(state)
                self.assertIs(actual_result, expected_result)
                self.assertIs(actual_result, None)
                self.assertIs(
                    torch.is_warn_always_enabled(),
                    reference_torch.is_warn_always_enabled(),
                )
                self.assertIs(torch.is_warn_always_enabled(), state)

    def test_call_shape_and_invalid_value_errors_match_pytorch_2_13(self):
        actual = torch.set_warn_always
        expected = reference_torch.set_warn_always
        call_cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(False, True), lambda: expected(False, True)),
            (lambda: actual(b=True), lambda: expected(b=True)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(False, enabled=True),
                lambda: expected(False, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(call_cases):
            with self.subTest(kind="call", case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(torch.is_warn_always_enabled(), False)
                self.assertIs(reference_torch.is_warn_always_enabled(), False)

        actual_values = (
            None,
            0,
            1,
            0.0,
            "",
            [],
            object(),
            _RejectTruthiness(),
            torch.tensor(True),
            torch.float32,
            torch.device("cpu"),
        )
        expected_values = (
            None,
            0,
            1,
            0.0,
            "",
            [],
            object(),
            _RejectTruthiness(),
            reference_torch.tensor(True),
            reference_torch.float32,
            reference_torch.device("cpu"),
        )
        for state in (False, True):
            torch.set_warn_always(state)
            reference_torch.set_warn_always(state)
            for case, (actual_value, expected_value) in enumerate(
                zip(actual_values, expected_values)
            ):
                with self.subTest(kind="value", state=state, case=case):
                    self.assert_error_matches(
                        lambda value=actual_value: actual(value),
                        lambda value=expected_value: expected(value),
                    )
                    self.assertIs(torch.is_warn_always_enabled(), state)
                    self.assertIs(reference_torch.is_warn_always_enabled(), state)

    def test_metadata_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual = torch.set_warn_always
        expected = reference_torch.set_warn_always

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
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

        self.assertEqual(
            torch.__all__.count("set_warn_always"),
            reference_torch.__all__.count("set_warn_always"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["set_warn_always"], function)
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

    def test_native_warning_policy_matches_pytorch_2_13(self):
        script = r'''
import importlib
import json
import sys
import warnings

module = importlib.import_module(sys.argv[1])


def warning_count(callback, count):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(count):
            callback()
    return len(caught)


scalar = module.tensor(1.0)
module.set_warn_always(False)
false_first = warning_count(lambda: scalar.T, 3)
module.set_warn_always(True)
true_after_consumed = warning_count(lambda: scalar.T, 3)
module.set_warn_always(False)
false_after_consumed = warning_count(lambda: scalar.T, 3)

module.set_warn_always(True)
true_before_consumed = warning_count(lambda: scalar.H, 3)
module.set_warn_always(False)
false_after_true_only = warning_count(lambda: scalar.H, 3)

module.set_warn_always(True)
python_true = warning_count(lambda: warnings.warn("ordinary", UserWarning), 3)
module.set_warn_always(False)
python_false = warning_count(lambda: warnings.warn("ordinary", UserWarning), 3)

print(json.dumps({
    "counts": [
        false_first,
        true_after_consumed,
        false_after_consumed,
        true_before_consumed,
        false_after_true_only,
        python_true,
        python_false,
    ],
    "state": module.is_warn_always_enabled(),
}))
'''

        def outcome(module_name):
            completed = subprocess.run(
                [sys.executable, "-c", script, module_name],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            return json.loads(completed.stdout)

        self.assertEqual(outcome("torch_rs"), outcome("torch"))


if __name__ == "__main__":
    unittest.main()
