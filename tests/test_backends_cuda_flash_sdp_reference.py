import copy
import importlib
import inspect
import pickle
import pickletools
import re
import sys
import threading
import types
import typing
import unittest

import numpy as np

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_NAMES = {
    "enable_flash_sdp",
    "flash_sdp_enabled",
    "is_built",
    "is_flash_attention_available",
}


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("enable_flash_sdp must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaFlashSdpReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cuda Flash SDP differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = importlib.import_module("torch_rs.backends.cuda")
        self.expected = importlib.import_module("torch.backends.cuda")
        self.original_actual = self.actual.flash_sdp_enabled()
        self.original_expected = self.expected.flash_sdp_enabled()
        self.actual.enable_flash_sdp(True)
        self.expected.enable_flash_sdp(True)

    def tearDown(self):
        self.actual.enable_flash_sdp(self.original_actual)
        self.expected.enable_flash_sdp(self.original_expected)

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

    def state_contract(self, root, module):
        outcomes = [
            (
                type(module.flash_sdp_enabled()) is bool,
                module.flash_sdp_enabled() is True,
                root._C._get_flash_sdp_enabled() is True,
            )
        ]
        for enabled in (False, True, True, False, False, True):
            outcomes.append(
                (
                    module.enable_flash_sdp(enabled) is None,
                    module.flash_sdp_enabled() is enabled,
                    root._C._get_flash_sdp_enabled() is enabled,
                )
            )
        outcomes.append(module.enable_flash_sdp(enabled=False) is None)
        outcomes.append(module.flash_sdp_enabled() is False)
        return outcomes

    def thread_contract(self, module):
        module.enable_flash_sdp(True)
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(("initial", module.flash_sdp_enabled()))
                observations.append(
                    ("setter", module.enable_flash_sdp(False))
                )
                observations.append(("worker", module.flash_sdp_enabled()))
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(("main", module.flash_sdp_enabled()))
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_changed.wait(timeout=10)
        state_after_worker = module.flash_sdp_enabled()
        main_result = module.enable_flash_sdp(True)
        main_changed.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            state_after_worker,
            main_result,
            not thread.is_alive(),
            errors,
            observations,
            module.flash_sdp_enabled(),
        )

    def reload_contract(self, root, module):
        old_getter = module.flash_sdp_enabled
        old_setter = module.enable_flash_sdp
        namespace = module.__dict__
        old_setter(False)
        reloaded = importlib.reload(module)
        preserved_state = module.flash_sdp_enabled()
        new_result = module.enable_flash_sdp(True)
        old_getter_state = old_getter()
        old_result = old_setter(False)
        final_state = module.flash_sdp_enabled()

        stale_errors = []
        for old_function in (old_getter, old_setter):
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
                self.fail("a stale Flash SDP function remained pickleable")

        return (
            reloaded is module,
            module.__dict__ is namespace,
            root.backends.cuda is module,
            sys.modules[module.__name__] is module,
            module.flash_sdp_enabled is not old_getter,
            module.enable_flash_sdp is not old_setter,
            preserved_state,
            new_result,
            old_getter_state,
            old_result,
            final_state,
            stale_errors,
            copy.copy(module.flash_sdp_enabled) is module.flash_sdp_enabled,
            copy.deepcopy(module.enable_flash_sdp) is module.enable_flash_sdp,
            pickle.loads(pickle.dumps(module.flash_sdp_enabled))
            is module.flash_sdp_enabled,
            pickle.loads(pickle.dumps(module.enable_flash_sdp))
            is module.enable_flash_sdp,
        )

    def test_state_transitions_threads_and_reload_match_pytorch_2_13(self):
        self.assertEqual(
            self.state_contract(torch, self.actual),
            self.state_contract(reference_torch, self.expected),
        )
        self.assertEqual(
            self.thread_contract(self.actual),
            self.thread_contract(self.expected),
        )
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
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
            self.actual.enable_flash_sdp(state)
            self.expected.enable_flash_sdp(state)
            for case, (actual_value, expected_value) in enumerate(
                zip(actual_values, expected_values)
            ):
                with self.subTest(kind="value", state=state, case=case):
                    self.assert_error_matches(
                        lambda value=actual_value: self.actual.enable_flash_sdp(
                            value
                        ),
                        lambda value=expected_value: self.expected.enable_flash_sdp(
                            value
                        ),
                    )
                    self.assertIs(self.actual.flash_sdp_enabled(), state)
                    self.assertIs(self.expected.flash_sdp_enabled(), state)

        cases = (
            (
                lambda: self.actual.flash_sdp_enabled(None),
                lambda: self.expected.flash_sdp_enabled(None),
            ),
            (
                lambda: self.actual.flash_sdp_enabled(enabled=True),
                lambda: self.expected.flash_sdp_enabled(enabled=True),
            ),
            (
                lambda: self.actual.enable_flash_sdp(),
                lambda: self.expected.enable_flash_sdp(),
            ),
            (
                lambda: self.actual.enable_flash_sdp(True, False),
                lambda: self.expected.enable_flash_sdp(True, False),
            ),
            (
                lambda: self.actual.enable_flash_sdp(True, enabled=False),
                lambda: self.expected.enable_flash_sdp(True, enabled=False),
            ),
            (
                lambda: self.actual.enable_flash_sdp(value=True),
                lambda: self.expected.enable_flash_sdp(value=True),
            ),
        )
        self.actual.enable_flash_sdp(True)
        self.expected.enable_flash_sdp(True)
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(kind="binding", case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(self.actual.flash_sdp_enabled(), True)
                self.assertIs(self.expected.flash_sdp_enabled(), True)

    def test_metadata_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected

        self.assertIsNone(actual.__doc__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            actual.__all__,
            [name for name in expected.__all__ if name in SUPPORTED_NAMES],
        )
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {
                name
                for name in vars(expected)
                if name in SUPPORTED_NAMES | {"torch"}
            },
        )

        for name in ("flash_sdp_enabled", "enable_flash_sdp"):
            actual_function = getattr(actual, name)
            expected_function = getattr(expected, name)
            with self.subTest(function=name):
                self.assertIs(type(actual_function), types.FunctionType)
                self.assertIs(type(expected_function), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual_function)),
                    str(inspect.signature(expected_function)),
                )
                self.assertEqual(
                    inspect.get_annotations(actual_function),
                    inspect.get_annotations(expected_function),
                )
                self.assertEqual(
                    typing.get_type_hints(actual_function),
                    typing.get_type_hints(expected_function),
                )
                self.assertEqual(actual_function.__name__, expected_function.__name__)
                self.assertEqual(
                    actual_function.__qualname__, expected_function.__qualname__
                )
                self.assertEqual(
                    actual_function.__module__.replace("torch_rs", "torch"),
                    expected_function.__module__,
                )
                self.assertIs(inspect.getmodule(actual_function), actual)
                self.assertIs(inspect.getmodule(expected_function), expected)
                self.assertEqual(actual_function.__doc__, expected_function.__doc__)
                self.assertEqual(
                    actual_function.__defaults__, expected_function.__defaults__
                )
                self.assertEqual(
                    actual_function.__kwdefaults__, expected_function.__kwdefaults__
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
                self.assertIs(copy.copy(actual_function), actual_function)
                self.assertIs(copy.copy(expected_function), expected_function)
                self.assertIs(copy.deepcopy(actual_function), actual_function)
                self.assertIs(copy.deepcopy(expected_function), expected_function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
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

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.backends.cuda import *", actual_wildcard)
        exec("from torch.backends.cuda import *", expected_wildcard)
        self.assertEqual(
            {name for name in actual_wildcard if not name.startswith("__")},
            {
                name
                for name in expected_wildcard
                if name in SUPPORTED_NAMES
            },
        )

    def test_preference_is_independent_of_availability_and_execution(self):
        actual_available = self.actual.is_flash_attention_available()
        expected_available = self.expected.is_flash_attention_available()
        self.assertIs(actual_available, False)

        for module, available in (
            (self.actual, actual_available),
            (self.expected, expected_available),
        ):
            for enabled in (False, True):
                with self.subTest(module=module.__name__, enabled=enabled):
                    self.assertIsNone(module.enable_flash_sdp(enabled))
                    self.assertIs(module.flash_sdp_enabled(), enabled)
                    self.assertIs(module.is_flash_attention_available(), available)

        self.assertFalse(hasattr(self.actual, "sdp_kernel"))
        self.assertFalse(
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
        )
        self.assertTrue(
            hasattr(
                reference_torch.nn.functional,
                "scaled_dot_product_attention",
            )
        )


if __name__ == "__main__":
    unittest.main()
