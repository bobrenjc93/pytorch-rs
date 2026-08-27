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


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("enable_flash_sdp must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaFlashSdpPreferenceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Flash SDP preference differentials require pinned PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = importlib.import_module("torch_rs.backends.cuda")
        self.expected = importlib.import_module("torch.backends.cuda")
        self.original_actual = torch._C._get_flash_sdp_enabled()
        self.original_expected = reference_torch._C._get_flash_sdp_enabled()
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

    def transition_contract(self, root, module):
        outcomes = []
        for enabled in (False, True, True, False, False, True):
            result = module.enable_flash_sdp(enabled)
            value = module.flash_sdp_enabled()
            outcomes.append(
                (
                    result is None,
                    type(value) is bool,
                    value is enabled,
                    root._C._get_flash_sdp_enabled() is enabled,
                )
            )
        result = module.enable_flash_sdp(enabled=False)
        outcomes.append((result is None, module.flash_sdp_enabled() is False))
        return outcomes

    def thread_contract(self, module):
        module.enable_flash_sdp(True)
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(module.flash_sdp_enabled())
                observations.append(module.enable_flash_sdp(False))
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(module.flash_sdp_enabled())
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
        module.enable_flash_sdp(False)
        reloaded = importlib.reload(module)
        preserved_state = module.flash_sdp_enabled()
        new_result = module.enable_flash_sdp(True)
        old_result = old_setter(False)
        old_value = old_getter()
        final_result = module.enable_flash_sdp(True)

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
                self.fail("a stale Flash SDP preference function remained pickleable")

        return (
            reloaded is module,
            module.__dict__ is namespace,
            root.backends.cuda is module,
            sys.modules[module.__name__] is module,
            module.flash_sdp_enabled is not old_getter,
            module.enable_flash_sdp is not old_setter,
            preserved_state,
            new_result,
            old_result,
            old_value,
            final_result,
            stale_errors,
        )

    def test_state_transitions_and_threads_match_pytorch_2_13(self):
        self.assertEqual(
            self.transition_contract(torch, self.actual),
            self.transition_contract(reference_torch, self.expected),
        )
        self.assertEqual(
            self.thread_contract(self.actual),
            self.thread_contract(self.expected),
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
                lambda: self.actual.enable_flash_sdp(value=True),
                lambda: self.expected.enable_flash_sdp(value=True),
            ),
            (
                lambda: self.actual.enable_flash_sdp(True, enabled=False),
                lambda: self.expected.enable_flash_sdp(True, enabled=False),
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
        supported = {
            "enable_flash_sdp",
            "flash_sdp_enabled",
            "is_built",
            "is_flash_attention_available",
        }
        self.assertEqual(
            self.actual.__all__,
            [name for name in self.expected.__all__ if name in supported],
        )
        self.assertEqual(self.actual.__doc__, self.expected.__doc__)
        self.assertEqual(self.actual.__annotations__, self.expected.__annotations__)
        self.assertEqual(
            {name for name in vars(self.actual) if not name.startswith("_")},
            {
                name
                for name in vars(self.expected)
                if name in supported | {"torch"}
            },
        )

        for name in ("flash_sdp_enabled", "enable_flash_sdp"):
            actual = getattr(self.actual, name)
            expected = getattr(self.expected, name)
            with self.subTest(function=name):
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual)),
                    str(inspect.signature(expected)),
                )
                self.assertEqual(
                    inspect.get_annotations(actual),
                    inspect.get_annotations(expected),
                )
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
                self.assertIs(inspect.getmodule(actual), self.actual)
                self.assertIs(inspect.getmodule(expected), self.expected)
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )
                self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
                self.assertEqual(
                    actual.__code__.co_freevars,
                    expected.__code__.co_freevars,
                )
                self.assertEqual(
                    actual.__code__.co_cellvars,
                    expected.__code__.co_cellvars,
                )

            for package_name, module in (
                ("torch_rs", self.actual),
                ("torch", self.expected),
            ):
                function_import = {}
                wildcard = {}
                exec(
                    f"from {package_name}.backends.cuda import {name}",
                    function_import,
                )
                exec(f"from {package_name}.backends.cuda import *", wildcard)
                self.assertIs(function_import[name], getattr(module, name))
                self.assertIs(wildcard[name], getattr(module, name))

            self.assertIs(copy.copy(actual), actual)
            self.assertIs(copy.copy(expected), expected)
            self.assertIs(copy.deepcopy(actual), actual)
            self.assertIs(copy.deepcopy(expected), expected)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=name, protocol=protocol):
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)), expected
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

    def test_reload_stability_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_build_availability_and_execution_remain_separate(self):
        for module in (self.actual, self.expected):
            module.enable_flash_sdp(False)
            self.assertIs(module.flash_sdp_enabled(), False)
            module.enable_flash_sdp(True)
            self.assertIs(module.flash_sdp_enabled(), True)

        self.assertIs(self.actual.is_flash_attention_available(), False)
        self.assertIs(torch.backends.cuda.is_built(), False)
        self.assertFalse(hasattr(torch, "cuda"))
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
