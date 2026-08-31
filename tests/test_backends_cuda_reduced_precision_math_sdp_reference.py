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
        raise AssertionError("the reduction setter must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaReducedPrecisionMathSdpReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "reduced-precision math SDP differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = importlib.import_module("torch_rs.backends.cuda")
        self.expected = importlib.import_module("torch.backends.cuda")
        self.original_actual = (
            torch._C._get_math_sdp_allow_fp16_bf16_reduction()
        )
        self.original_expected = (
            reference_torch._C._get_math_sdp_allow_fp16_bf16_reduction()
        )
        self.original_actual_math = torch._C._get_math_sdp_enabled()
        self.original_expected_math = reference_torch._C._get_math_sdp_enabled()
        self.original_actual_mem_efficient = (
            torch._C._get_mem_efficient_sdp_enabled()
        )
        self.original_expected_mem_efficient = (
            reference_torch._C._get_mem_efficient_sdp_enabled()
        )
        self.actual.allow_fp16_bf16_reduction_math_sdp(False)
        self.expected.allow_fp16_bf16_reduction_math_sdp(False)

    def tearDown(self):
        self.actual.allow_fp16_bf16_reduction_math_sdp(self.original_actual)
        self.expected.allow_fp16_bf16_reduction_math_sdp(self.original_expected)
        self.actual.enable_math_sdp(self.original_actual_math)
        self.expected.enable_math_sdp(self.original_expected_math)
        self.actual.enable_mem_efficient_sdp(self.original_actual_mem_efficient)
        self.expected.enable_mem_efficient_sdp(
            self.original_expected_mem_efficient
        )

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
        for enabled in (True, False, False, True, True, False):
            result = module.allow_fp16_bf16_reduction_math_sdp(enabled)
            state = module.fp16_bf16_reduction_math_sdp_allowed()
            outcomes.append(
                (
                    result,
                    type(state) is bool,
                    state is enabled,
                    root._C._get_math_sdp_allow_fp16_bf16_reduction()
                    is enabled,
                )
            )
        outcomes.append(
            module.allow_fp16_bf16_reduction_math_sdp(enabled=True)
        )
        outcomes.append(module.fp16_bf16_reduction_math_sdp_allowed() is True)
        return outcomes

    def thread_contract(self, root, module):
        module.allow_fp16_bf16_reduction_math_sdp(False)
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(
                    module.fp16_bf16_reduction_math_sdp_allowed()
                )
                observations.append(
                    module.allow_fp16_bf16_reduction_math_sdp(True)
                )
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(
                    module.fp16_bf16_reduction_math_sdp_allowed()
                )
                observations.append(
                    module.allow_fp16_bf16_reduction_math_sdp(True)
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_changed.wait(timeout=10)
        state_after_worker = (
            root._C._get_math_sdp_allow_fp16_bf16_reduction()
        )
        main_result = module.allow_fp16_bf16_reduction_math_sdp(False)
        main_changed.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            state_after_worker,
            main_result,
            not thread.is_alive(),
            errors,
            observations,
            root._C._get_math_sdp_allow_fp16_bf16_reduction(),
        )

    def reload_contract(self, root, module):
        old_getter = module.fp16_bf16_reduction_math_sdp_allowed
        old_setter = module.allow_fp16_bf16_reduction_math_sdp
        namespace = module.__dict__
        module.allow_fp16_bf16_reduction_math_sdp(True)
        reloaded = importlib.reload(module)
        preserved_state = module.fp16_bf16_reduction_math_sdp_allowed()
        new_result = module.allow_fp16_bf16_reduction_math_sdp(False)
        old_getter_result = old_getter()
        old_result = old_setter(True)
        final_result = module.allow_fp16_bf16_reduction_math_sdp(False)

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
                self.fail("a stale reduced-precision SDP function remained pickleable")

        return (
            reloaded is module,
            module.__dict__ is namespace,
            root.backends.cuda is module,
            sys.modules[module.__name__] is module,
            module.fp16_bf16_reduction_math_sdp_allowed is not old_getter,
            module.allow_fp16_bf16_reduction_math_sdp is not old_setter,
            preserved_state,
            new_result,
            old_getter_result,
            old_result,
            final_result,
            stale_errors,
        )

    def test_state_transitions_threads_and_reload_match_pytorch_2_13(self):
        self.assertEqual(
            self.transition_contract(torch, self.actual),
            self.transition_contract(reference_torch, self.expected),
        )
        self.assertEqual(
            self.thread_contract(torch, self.actual),
            self.thread_contract(reference_torch, self.expected),
        )
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_invalid_values_and_binding_errors_match_pytorch_2_13(self):
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
            self.actual.allow_fp16_bf16_reduction_math_sdp(state)
            self.expected.allow_fp16_bf16_reduction_math_sdp(state)
            for case, (actual_value, expected_value) in enumerate(
                zip(actual_values, expected_values)
            ):
                with self.subTest(kind="value", state=state, case=case):
                    self.assert_error_matches(
                        lambda value=actual_value: (
                            self.actual.allow_fp16_bf16_reduction_math_sdp(value)
                        ),
                        lambda value=expected_value: (
                            self.expected.allow_fp16_bf16_reduction_math_sdp(value)
                        ),
                    )
                    self.assertIs(
                        self.actual.fp16_bf16_reduction_math_sdp_allowed(),
                        state,
                    )
                    self.assertIs(
                        self.expected.fp16_bf16_reduction_math_sdp_allowed(),
                        state,
                    )

        cases = (
            (
                lambda: self.actual.fp16_bf16_reduction_math_sdp_allowed(None),
                lambda: self.expected.fp16_bf16_reduction_math_sdp_allowed(None),
            ),
            (
                lambda: self.actual.fp16_bf16_reduction_math_sdp_allowed(
                    enabled=True
                ),
                lambda: self.expected.fp16_bf16_reduction_math_sdp_allowed(
                    enabled=True
                ),
            ),
            (
                lambda: self.actual.allow_fp16_bf16_reduction_math_sdp(),
                lambda: self.expected.allow_fp16_bf16_reduction_math_sdp(),
            ),
            (
                lambda: self.actual.allow_fp16_bf16_reduction_math_sdp(
                    True, False
                ),
                lambda: self.expected.allow_fp16_bf16_reduction_math_sdp(
                    True, False
                ),
            ),
            (
                lambda: self.actual.allow_fp16_bf16_reduction_math_sdp(
                    _enabled=False
                ),
                lambda: self.expected.allow_fp16_bf16_reduction_math_sdp(
                    _enabled=False
                ),
            ),
            (
                lambda: self.actual.allow_fp16_bf16_reduction_math_sdp(
                    True, enabled=False
                ),
                lambda: self.expected.allow_fp16_bf16_reduction_math_sdp(
                    True, enabled=False
                ),
            ),
        )
        self.actual.allow_fp16_bf16_reduction_math_sdp(False)
        self.expected.allow_fp16_bf16_reduction_math_sdp(False)
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(kind="binding", case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(
                    self.actual.fp16_bf16_reduction_math_sdp_allowed(),
                    False,
                )
                self.assertIs(
                    self.expected.fp16_bf16_reduction_math_sdp_allowed(),
                    False,
                )

    def test_metadata_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected
        supported = {
            "allow_fp16_bf16_reduction_math_sdp",
            "cuBLASModule",
            "enable_flash_sdp",
            "enable_math_sdp",
            "enable_mem_efficient_sdp",
            "flash_sdp_enabled",
            "fp16_bf16_reduction_math_sdp_allowed",
            "is_built",
            "is_ck_sdpa_available",
            "is_flash_attention_available",
            "math_sdp_enabled",
            "matmul",
            "mem_efficient_sdp_enabled",
            "sdp_kernel",
        }

        self.assertIs(torch.backends.cuda, actual)
        self.assertIs(reference_torch.backends.cuda, expected)
        self.assertIs(type(actual), types.ModuleType)
        self.assertIs(type(expected), types.ModuleType)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            actual.__all__,
            [name for name in expected.__all__ if name in supported],
        )
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {
                name
                for name in vars(expected)
                if name in supported | {"torch"}
            },
        )

        for name in (
            "fp16_bf16_reduction_math_sdp_allowed",
            "allow_fp16_bf16_reduction_math_sdp",
        ):
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
                self.assertEqual(actual_function.__code__.co_freevars, ())
                self.assertEqual(actual_function.__code__.co_cellvars, ())
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

        for package_name, module in (("torch_rs", actual), ("torch", expected)):
            getter_import = {}
            setter_import = {}
            wildcard = {}
            exec(
                f"from {package_name}.backends.cuda import "
                "fp16_bf16_reduction_math_sdp_allowed",
                getter_import,
            )
            exec(
                f"from {package_name}.backends.cuda import "
                "allow_fp16_bf16_reduction_math_sdp",
                setter_import,
            )
            exec(f"from {package_name}.backends.cuda import *", wildcard)
            self.assertIs(
                getter_import["fp16_bf16_reduction_math_sdp_allowed"],
                module.fp16_bf16_reduction_math_sdp_allowed,
            )
            self.assertIs(
                setter_import["allow_fp16_bf16_reduction_math_sdp"],
                module.allow_fp16_bf16_reduction_math_sdp,
            )
            self.assertEqual(
                {name for name in wildcard if name in supported},
                supported,
            )

    def test_preference_is_independent_from_other_flags_and_execution_support(self):
        actual_other_states = {
            "flash": self.actual.flash_sdp_enabled(),
            "math": self.actual.math_sdp_enabled(),
            "mem_efficient": self.actual.mem_efficient_sdp_enabled(),
        }
        expected_other_states = {
            "cudnn": self.expected.cudnn_sdp_enabled(),
            "flash": self.expected.flash_sdp_enabled(),
            "math": self.expected.math_sdp_enabled(),
            "mem_efficient": self.expected.mem_efficient_sdp_enabled(),
        }
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                self.assertIs(
                    self.actual.allow_fp16_bf16_reduction_math_sdp(enabled),
                    None,
                )
                self.assertIs(
                    self.expected.allow_fp16_bf16_reduction_math_sdp(enabled),
                    None,
                )
                self.assertIs(
                    self.actual.fp16_bf16_reduction_math_sdp_allowed(),
                    enabled,
                )
                self.assertIs(
                    self.expected.fp16_bf16_reduction_math_sdp_allowed(),
                    enabled,
                )
                self.assertEqual(
                    {
                        "flash": self.actual.flash_sdp_enabled(),
                        "math": self.actual.math_sdp_enabled(),
                        "mem_efficient": self.actual.mem_efficient_sdp_enabled(),
                    },
                    actual_other_states,
                )
                self.assertEqual(
                    {
                        "cudnn": self.expected.cudnn_sdp_enabled(),
                        "flash": self.expected.flash_sdp_enabled(),
                        "math": self.expected.math_sdp_enabled(),
                        "mem_efficient": self.expected.mem_efficient_sdp_enabled(),
                    },
                    expected_other_states,
                )

        self.actual.allow_fp16_bf16_reduction_math_sdp(True)
        self.expected.allow_fp16_bf16_reduction_math_sdp(True)
        for enabled in (False, True):
            with self.subTest(math_enabled=enabled):
                self.assertIs(self.actual.enable_math_sdp(enabled), None)
                self.assertIs(self.expected.enable_math_sdp(enabled), None)
                self.assertIs(
                    self.actual.fp16_bf16_reduction_math_sdp_allowed(), True
                )
                self.assertIs(
                    self.expected.fp16_bf16_reduction_math_sdp_allowed(), True
                )
            with self.subTest(mem_efficient_enabled=enabled):
                self.assertIs(self.actual.enable_mem_efficient_sdp(enabled), None)
                self.assertIs(self.expected.enable_mem_efficient_sdp(enabled), None)
                self.assertIs(
                    self.actual.fp16_bf16_reduction_math_sdp_allowed(), True
                )
                self.assertIs(
                    self.expected.fp16_bf16_reduction_math_sdp_allowed(), True
                )

        for name in (
            "cudnn_sdp_enabled",
            "enable_cudnn_sdp",
        ):
            with self.subTest(unsupported_preference=name):
                self.assertFalse(hasattr(self.actual, name))
                self.assertTrue(hasattr(self.expected, name))

        self.assertFalse(hasattr(torch, "float16"))
        self.assertFalse(hasattr(torch, "bfloat16"))
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
        )
        self.assertTrue(hasattr(reference_torch, "float16"))
        self.assertTrue(hasattr(reference_torch, "bfloat16"))
        self.assertTrue(hasattr(reference_torch, "cuda"))
        self.assertTrue(
            hasattr(reference_torch.nn.functional, "scaled_dot_product_attention")
        )


if __name__ == "__main__":
    unittest.main()
