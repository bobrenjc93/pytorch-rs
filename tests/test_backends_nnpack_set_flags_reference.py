import copy
import importlib
import inspect
import pickle
import pickletools
import re
import sys
import threading
import types
import unittest

import numpy as np

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("set_flags must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class NnpackSetFlagsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.nnpack.set_flags differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = importlib.import_module("torch_rs.backends.nnpack")
        self.expected = importlib.import_module("torch.backends.nnpack")
        self.original_actual = torch._C._get_nnpack_enabled()
        self.original_expected = reference_torch._C._get_nnpack_enabled()
        self.actual.set_flags(True)
        self.expected.set_flags(True)

    def tearDown(self):
        self.actual.set_flags(self.original_actual)
        self.expected.set_flags(self.original_expected)

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

    def transition_contract(self, root, module):
        outcomes = []
        for enabled in (False, True, True, False, False, True):
            result = module.set_flags(enabled)
            outcomes.append(
                (
                    type(result) is tuple,
                    len(result),
                    type(result[0]) is bool,
                    result,
                    root._C._get_nnpack_enabled() is enabled,
                )
            )
        outcomes.append(module.set_flags(_enabled=False))
        outcomes.append(root._C._get_nnpack_enabled() is False)
        return outcomes

    def thread_contract(self, root, module):
        module.set_flags(True)
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(module.set_flags(False))
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(module.set_flags(False))
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_changed.wait(timeout=10)
        state_after_worker = root._C._get_nnpack_enabled()
        main_result = module.set_flags(True)
        main_changed.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            state_after_worker,
            main_result,
            not thread.is_alive(),
            errors,
            observations,
            root._C._get_nnpack_enabled(),
        )

    def reload_contract(self, root, module):
        old_is_available = module.is_available
        old_setter = module.set_flags
        namespace = module.__dict__
        module.set_flags(False)
        reloaded = importlib.reload(module)
        preserved_state = root._C._get_nnpack_enabled()
        new_result = module.set_flags(True)
        old_result = old_setter(False)
        final_result = module.set_flags(True)

        stale_errors = []
        for old_function in (old_is_available, old_setter):
            try:
                pickle.dumps(old_function)
            except Exception as error:
                stale_errors.append(
                    (
                        type(error).__name__,
                        re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                            "torch_rs", "torch"
                        ),
                    )
                )
            else:
                self.fail("a stale NNPACK function remained pickleable")

        return (
            reloaded is module,
            module.__dict__ is namespace,
            root.backends.nnpack is module,
            sys.modules[module.__name__] is module,
            module.is_available is not old_is_available,
            module.set_flags is not old_setter,
            preserved_state,
            new_result,
            old_result,
            final_result,
            stale_errors,
        )

    def test_state_transitions_and_threads_match_pytorch_2_13(self):
        self.assertEqual(
            self.transition_contract(torch, self.actual),
            self.transition_contract(reference_torch, self.expected),
        )
        self.assertEqual(
            self.thread_contract(torch, self.actual),
            self.thread_contract(reference_torch, self.expected),
        )

    def test_invalid_values_and_call_errors_match_pytorch_2_13(self):
        actual_values = (
            None,
            0,
            1,
            0.0,
            np.bool_(True),
            "",
            [],
            object(),
            _RejectTruthiness(),
            torch.tensor(True),
            torch.float32,
            torch.device("cpu"),
            torch.strided,
            torch.Size([1]),
            torch.finfo(torch.float32),
        )
        expected_values = (
            None,
            0,
            1,
            0.0,
            np.bool_(True),
            "",
            [],
            object(),
            _RejectTruthiness(),
            reference_torch.tensor(True),
            reference_torch.float32,
            reference_torch.device("cpu"),
            reference_torch.strided,
            reference_torch.Size([1]),
            reference_torch.finfo(reference_torch.float32),
        )
        for state in (False, True):
            self.actual.set_flags(state)
            self.expected.set_flags(state)
            for case, (actual_value, expected_value) in enumerate(
                zip(actual_values, expected_values)
            ):
                with self.subTest(kind="value", state=state, case=case):
                    self.assert_error_matches(
                        lambda value=actual_value: self.actual.set_flags(value),
                        lambda value=expected_value: self.expected.set_flags(value),
                    )
                    self.assertIs(torch._C._get_nnpack_enabled(), state)
                    self.assertIs(
                        reference_torch._C._get_nnpack_enabled(),
                        state,
                    )

        cases = (
            (
                lambda: self.actual.set_flags(),
                lambda: self.expected.set_flags(),
            ),
            (
                lambda: self.actual.set_flags(True, False),
                lambda: self.expected.set_flags(True, False),
            ),
            (
                lambda: self.actual.set_flags(enabled=True),
                lambda: self.expected.set_flags(enabled=True),
            ),
            (
                lambda: self.actual.set_flags(True, _enabled=False),
                lambda: self.expected.set_flags(True, _enabled=False),
            ),
        )
        self.actual.set_flags(True)
        self.expected.set_flags(True)
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(kind="binding", case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(torch._C._get_nnpack_enabled(), True)
                self.assertIs(reference_torch._C._get_nnpack_enabled(), True)

    def test_metadata_imports_copying_and_pickling_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected
        actual_function = actual.set_flags
        expected_function = expected.set_flags

        self.assertIs(torch.backends.nnpack, actual)
        self.assertIs(reference_torch.backends.nnpack, expected)
        self.assertIs(type(actual), types.ModuleType)
        self.assertIs(type(expected), types.ModuleType)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__all__, expected.__all__)
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {
                name
                for name in vars(expected)
                if name
                in {
                    "contextmanager",
                    "flags",
                    "is_available",
                    "set_flags",
                    "torch",
                }
            },
        )
        self.assertIs(actual.torch, torch)
        self.assertIs(expected.torch, reference_torch)

        self.assertIs(type(actual_function), types.FunctionType)
        self.assertIs(type(expected_function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual_function)),
            str(inspect.signature(expected_function)),
        )
        self.assertEqual(
            actual_function.__annotations__,
            expected_function.__annotations__,
        )
        self.assertEqual(actual_function.__name__, expected_function.__name__)
        self.assertEqual(
            actual_function.__qualname__,
            expected_function.__qualname__,
        )
        self.assertEqual(
            actual_function.__module__.replace("torch_rs", "torch"),
            expected_function.__module__,
        )
        self.assertIs(inspect.getmodule(actual_function), actual)
        self.assertIs(inspect.getmodule(expected_function), expected)
        self.assertEqual(actual_function.__doc__, expected_function.__doc__)
        self.assertEqual(
            actual_function.__defaults__,
            expected_function.__defaults__,
        )
        self.assertEqual(
            actual_function.__kwdefaults__,
            expected_function.__kwdefaults__,
        )
        self.assertEqual(actual_function.__dict__, expected_function.__dict__)
        self.assertEqual(
            hasattr(actual_function, "__text_signature__"),
            hasattr(expected_function, "__text_signature__"),
        )
        self.assertEqual(
            actual_function.__code__.co_names,
            expected_function.__code__.co_names,
        )
        self.assertEqual(
            actual_function.__code__.co_freevars,
            expected_function.__code__.co_freevars,
        )
        self.assertEqual(
            actual_function.__code__.co_cellvars,
            expected_function.__code__.co_cellvars,
        )

        for package_name, module in (
            ("torch_rs", actual),
            ("torch", expected),
        ):
            backend_import = {}
            function_import = {}
            wildcard = {}
            exec(f"from {package_name}.backends import nnpack", backend_import)
            exec(
                f"from {package_name}.backends.nnpack import set_flags",
                function_import,
            )
            exec(f"from {package_name}.backends.nnpack import *", wildcard)
            self.assertIs(backend_import["nnpack"], module)
            self.assertIs(function_import["set_flags"], module.set_flags)
            self.assertIs(wildcard["set_flags"], module.set_flags)
            supported = {
                name
                for name in wildcard
                if name in {"flags", "is_available", "set_flags"}
            }
            self.assertEqual(
                supported,
                {"flags", "is_available", "set_flags"},
            )

        for function in (actual_function, expected_function):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual_function, protocol)),
                    actual_function,
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected_function, protocol)),
                    expected_function,
                )
                self.assertEqual(
                    self.pickle_shape(actual_function, protocol),
                    self.pickle_shape(expected_function, protocol),
                )

    def test_reload_stability_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_supported_scope_keeps_availability_flags_and_execution_separate(self):
        for module in (self.actual, self.expected):
            module.set_flags(False)
            self.assertEqual(module.set_flags(True), (False,))

        self.assertIs(self.actual.is_available(), False)
        self.assertTrue(hasattr(self.actual, "flags"))
        self.assertTrue(hasattr(self.expected, "flags"))
        self.assertFalse(hasattr(torch, "_nnpack_spatial_convolution"))
        self.assertTrue(hasattr(reference_torch, "_nnpack_spatial_convolution"))


if __name__ == "__main__":
    unittest.main()
