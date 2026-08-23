import contextlib
import copy
import importlib
import inspect
import json
import os
import pickle
import pickletools
import subprocess
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
class IsWarnAlwaysEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "warn_always differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def state_outcome(self, module):
        function = module.is_warn_always_enabled

        def query_outcome():
            before = module.is_grad_enabled()
            result = function()
            after = module.is_grad_enabled()
            return before, result, after

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

    def test_mutable_threaded_and_grad_states_match_pytorch_2_13(self):
        actual_original = torch.is_warn_always_enabled()
        expected_original = reference_torch.is_warn_always_enabled()
        try:
            for enabled in (False, True, False):
                with self.subTest(enabled=enabled):
                    self.assertIs(torch.set_warn_always(enabled), None)
                    self.assertIs(reference_torch.set_warn_always(enabled), None)
                    self.assertEqual(
                        self.state_outcome(torch),
                        self.state_outcome(reference_torch),
                    )
        finally:
            torch.set_warn_always(actual_original)
            reference_torch.set_warn_always(expected_original)

        self.assertIs(torch.is_warn_always_enabled(), actual_original)
        self.assertIs(reference_torch.is_warn_always_enabled(), expected_original)

    def test_native_state_and_native_module_reload_match_pytorch_2_13(self):
        actual_original = torch.is_warn_always_enabled()
        expected_original = reference_torch.is_warn_always_enabled()

        try:
            actual_results = [torch._C._set_warnAlways(True)]
            expected_results = [reference_torch._C._set_warnAlways(True)]
            actual_states = [torch.is_warn_always_enabled()]
            expected_states = [reference_torch.is_warn_always_enabled()]
            actual_native = torch._C
            expected_native = reference_torch._C
            self.assertIs(importlib.reload(actual_native), actual_native)
            self.assertIs(importlib.reload(expected_native), expected_native)
            actual_states.append(torch._C._get_warnAlways())
            expected_states.append(reference_torch._C._get_warnAlways())
        finally:
            torch.set_warn_always(actual_original)
            reference_torch.set_warn_always(expected_original)

        self.assertEqual(actual_results, expected_results)
        self.assertEqual(actual_states, expected_states)
        self.assertEqual(actual_states, [True, True])
        for state in (*actual_states, *expected_states):
            self.assertIs(type(state), bool)

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs")
        expected_module = importlib.import_module("torch")
        self.assertIs(torch, actual_module)
        self.assertIs(reference_torch, expected_module)
        for name in ("is_warn_always_enabled", "set_warn_always"):
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
                    typing.get_type_hints(actual), typing.get_type_hints(expected)
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
                self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )

    def test_exports_copy_and_pickle_match_pytorch_2_13(self):
        for name in ("is_warn_always_enabled", "set_warn_always"):
            actual = getattr(torch, name)
            expected = getattr(reference_torch, name)
            self.assertEqual(
                torch.__all__.count(name), reference_torch.__all__.count(name)
            )
            for module, function in ((torch, actual), (reference_torch, expected)):
                namespace = {}
                exec(f"from {module.__name__} import *", namespace)
                self.assertIs(namespace[name], function)
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)

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

    def test_argument_errors_match_pytorch_2_13(self):
        actual_getter = torch.is_warn_always_enabled
        expected_getter = reference_torch.is_warn_always_enabled
        getter_cases = (
            (lambda: actual_getter(None), lambda: expected_getter(None)),
            (lambda: actual_getter(None, None), lambda: expected_getter(None, None)),
            (lambda: actual_getter(enabled=True), lambda: expected_getter(enabled=True)),
            (
                lambda: actual_getter(None, enabled=True),
                lambda: expected_getter(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(getter_cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        actual_setter = torch.set_warn_always
        expected_setter = reference_torch.set_warn_always
        setter_cases = (
            (lambda: actual_setter(), lambda: expected_setter()),
            (
                lambda: actual_setter(True, False),
                lambda: expected_setter(True, False),
            ),
            (lambda: actual_setter(b=True), lambda: expected_setter(b=True)),
            (
                lambda: actual_setter(enabled=True),
                lambda: expected_setter(enabled=True),
            ),
            (
                lambda: actual_setter(True, b=False),
                lambda: expected_setter(True, b=False),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(setter_cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        actual_original = torch.is_warn_always_enabled()
        expected_original = reference_torch.is_warn_always_enabled()
        try:
            invalid_pairs = (
                (None, None),
                (0, 0),
                (1.0, 1.0),
                ("", ""),
                (torch.tensor(1.0), reference_torch.tensor(1.0)),
                (torch.float32, reference_torch.float32),
                (torch.device("cpu"), reference_torch.device("cpu")),
                (torch.contiguous_format, reference_torch.contiguous_format),
                (torch.strided, reference_torch.strided),
                (torch.Size([1]), reference_torch.Size([1])),
                (
                    torch.finfo(torch.float32),
                    reference_torch.finfo(reference_torch.float32),
                ),
            )
            for actual_value, expected_value in invalid_pairs:
                with self.subTest(value=actual_value):
                    self.assert_error_matches(
                        lambda value=actual_value: actual_setter(value),
                        lambda value=expected_value: expected_setter(value),
                    )
        finally:
            torch.set_warn_always(actual_original)
            reference_torch.set_warn_always(expected_original)

    def test_native_warning_transitions_and_python_warnings_match_pytorch_2_13(self):
        source = r'''
import importlib
import json
import os
import warnings

torch = importlib.import_module(os.environ["WARN_ALWAYS_MODULE"])


def native_warning_count(attribute, calls):
    tensor = torch.tensor(1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(calls):
            getattr(tensor, attribute)
    return {
        "count": len(caught),
        "categories": [warning.category.__name__ for warning in caught],
        "messages": [
            str(warning.message).split(" (Triggered internally at ", 1)[0]
            for warning in caught
        ],
    }


def ordinary_warning():
    warnings.warn("ordinary Python warning", UserWarning)


outputs = {"native": [], "ordinary": []}
for enabled, attribute in (
    (False, "T"),
    (True, "T"),
    (False, "T"),
    (True, "mT"),
    (False, "mT"),
):
    outputs["native"].append(
        {
            "setter_result_is_none": torch.set_warn_always(enabled) is None,
            "state": torch.is_warn_always_enabled(),
            "warning": native_warning_count(attribute, 2),
        }
    )

for enabled in (False, True):
    torch.set_warn_always(enabled)
    globals().pop("__warningregistry__", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        ordinary_warning()
        ordinary_warning()
    outputs["ordinary"].append(len(caught))

torch.set_warn_always(False)
print(json.dumps(outputs, sort_keys=True))
'''

        def outcome(module):
            environment = os.environ.copy()
            environment["WARN_ALWAYS_MODULE"] = module
            completed = subprocess.run(
                [sys.executable, "-c", source],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
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
