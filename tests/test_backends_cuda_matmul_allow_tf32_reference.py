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
        raise AssertionError("cuda.matmul.allow_tf32 must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaMatmulAllowTf32ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cuda.matmul.allow_tf32 differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = importlib.import_module("torch_rs.backends.cuda.matmul")
        self.expected = reference_torch.backends.cuda.matmul
        self.original_actual = torch._C._get_cublas_allow_tf32()
        self.original_expected = reference_torch._C._get_cublas_allow_tf32()
        self.actual.allow_tf32 = False
        self.expected.allow_tf32 = False

    def tearDown(self):
        actual = importlib.import_module("torch_rs.backends.cuda.matmul")
        actual.allow_tf32 = self.original_actual
        self.expected.allow_tf32 = self.original_expected
        torch.set_float32_matmul_precision("highest")
        reference_torch.set_float32_matmul_precision("highest")

    def normalize(self, value):
        if isinstance(value, str):
            return value.replace("torch_rs.torch_rs", "torch._C").replace(
                "torch_rs",
                "torch",
            )
        if isinstance(value, tuple):
            return tuple(self.normalize(item) for item in value)
        return value

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(
            self.normalize(str(actual_raised.exception)),
            str(expected_raised.exception),
        )
        self.assertEqual(
            self.normalize(actual_raised.exception.args),
            expected_raised.exception.args,
        )

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs.torch_rs", "torch._C")
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def transition_contract(self, root, module):
        outcomes = []
        for allow_tf32 in (True, False, False, True, True, False):
            module.allow_tf32 = allow_tf32
            state = module.allow_tf32
            outcomes.append(
                (
                    type(state) is bool,
                    state is allow_tf32,
                    root._C._get_cublas_allow_tf32() is allow_tf32,
                )
            )
        return outcomes

    def thread_contract(self, root, module):
        module.allow_tf32 = False
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(module.allow_tf32)
                module.allow_tf32 = True
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(module.allow_tf32)
                module.allow_tf32 = True
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_changed.wait(timeout=10)
        state_after_worker = root._C._get_cublas_allow_tf32()
        module.allow_tf32 = False
        main_changed.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            state_after_worker,
            not thread.is_alive(),
            errors,
            observations,
            root._C._get_cublas_allow_tf32(),
        )

    def test_default_transitions_threads_and_invalid_values_match_pytorch_2_13(self):
        self.assertIs(self.original_actual, self.original_expected)
        self.assertEqual(
            self.transition_contract(torch, self.actual),
            self.transition_contract(reference_torch, self.expected),
        )
        self.assertEqual(
            self.thread_contract(torch, self.actual),
            self.thread_contract(reference_torch, self.expected),
        )

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
            self.actual.allow_tf32 = state
            self.expected.allow_tf32 = state
            for case, (actual_value, expected_value) in enumerate(
                zip(actual_values, expected_values)
            ):
                with self.subTest(kind="value", state=state, case=case):
                    self.assert_error_matches(
                        lambda value=actual_value: setattr(
                            self.actual,
                            "allow_tf32",
                            value,
                        ),
                        lambda value=expected_value: setattr(
                            self.expected,
                            "allow_tf32",
                            value,
                        ),
                    )
                    self.assertIs(self.actual.allow_tf32, state)
                    self.assertIs(self.expected.allow_tf32, state)

    def test_import_reload_and_metadata_match_supported_contract(self):
        actual = self.actual
        expected = self.expected

        self.assertIs(torch.backends.cuda.matmul, actual)
        self.assertIsInstance(actual, types.ModuleType)
        self.assertIs(sys.modules["torch_rs.backends.cuda.matmul"], actual)
        self.assertFalse(isinstance(expected, types.ModuleType))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch.backends.cuda.matmul")

        descriptor = vars(type(actual))["allow_tf32"]
        self.assertEqual(set(vars(descriptor)), {"getter", "setter"})
        self.assertEqual(descriptor.__doc__, None)
        self.assertIs(descriptor.getter, torch._C._get_cublas_allow_tf32)
        self.assertIs(descriptor.setter, torch._C._set_cublas_allow_tf32)
        self.assertIs(actual.m.__annotations__["allow_tf32"], bool)
        self.assertNotIn("allow_tf32", vars(actual))
        self.assertNotIn("allow_tf32", vars(actual.m))
        self.assertNotIn("allow_tf32", dir(actual))

        direct = {}
        wildcard = {}
        exec("from torch_rs.backends.cuda import matmul", direct)
        exec("from torch_rs.backends.cuda.matmul import allow_tf32", direct)
        exec("from torch_rs.backends.cuda.matmul import *", wildcard)
        self.assertIs(direct["matmul"], actual)
        self.assertIs(direct["allow_tf32"], actual.allow_tf32)
        self.assertNotIn("allow_tf32", wildcard)
        self.assertIn("matmul", torch.backends.cuda.__all__)

        namespace = actual.__dict__
        actual.allow_tf32 = True
        reloaded = importlib.reload(actual)
        self.assertIsNot(reloaded, actual)
        self.assertIs(actual.__dict__, namespace)
        self.assertIs(torch.backends.cuda.matmul, actual)
        self.assertIs(sys.modules[actual.__name__], reloaded)
        self.assertIs(reloaded.m, actual)
        self.assertIs(actual.allow_tf32, True)
        self.assertIs(reloaded.allow_tf32, True)

    def test_private_accessor_metadata_and_pickle_match_pytorch_2_13(self):
        for name in ("_get_cublas_allow_tf32", "_set_cublas_allow_tf32"):
            actual = getattr(torch._C, name)
            expected = getattr(reference_torch._C, name)
            with self.subTest(name=name):
                self.assertIs(type(actual), types.BuiltinFunctionType)
                self.assertIs(type(expected), types.BuiltinFunctionType)
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(
                    self.normalize(actual.__module__),
                    expected.__module__,
                )
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(
                    actual.__text_signature__,
                    expected.__text_signature__,
                )
                try:
                    actual_signature = ("return", str(inspect.signature(actual)))
                except BaseException as error:
                    actual_signature = (
                        "error",
                        type(error).__name__,
                        str(error),
                    )
                try:
                    expected_signature = ("return", str(inspect.signature(expected)))
                except BaseException as error:
                    expected_signature = (
                        "error",
                        type(error).__name__,
                        str(error),
                    )
                self.assertEqual(
                    self.normalize(actual_signature),
                    expected_signature,
                )
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.deepcopy(expected), expected)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual, protocol)),
                        actual,
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)),
                        expected,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

        binding_calls = (
            lambda root: root._C._get_cublas_allow_tf32(None),
            lambda root: root._C._get_cublas_allow_tf32(value=None),
            lambda root: root._C._set_cublas_allow_tf32(),
            lambda root: root._C._set_cublas_allow_tf32(True, False),
            lambda root: root._C._set_cublas_allow_tf32(object=False),
        )
        self.actual.allow_tf32 = True
        self.expected.allow_tf32 = True
        for case, call in enumerate(binding_calls):
            with self.subTest(kind="binding", case=case):
                self.assert_error_matches(
                    lambda call=call: call(torch),
                    lambda call=call: call(reference_torch),
                )
                self.assertIs(self.actual.allow_tf32, True)
                self.assertIs(self.expected.allow_tf32, True)

        for name in ("_get_cublas_allow_tf32", "_set_cublas_allow_tf32"):
            direct = {}
            wildcard = {}
            exec(f"from torch_rs._C import {name}", direct)
            exec("from torch_rs._C import *", wildcard)
            self.assertIs(direct[name], getattr(torch._C, name))
            self.assertNotIn(name, torch.__all__)
            self.assertNotIn(name, torch._C.__all__)
            self.assertNotIn(name, wildcard)

    def test_independence_and_unsupported_surface_are_explicit(self):
        actual_cuda = torch.backends.cuda
        expected_cuda = reference_torch.backends.cuda
        actual = self.actual
        expected = self.expected

        actual_other_states = {
            "cudnn": torch.backends.cudnn.allow_tf32,
            "precision": torch.get_float32_matmul_precision(),
            "flash": actual_cuda.flash_sdp_enabled(),
            "math": actual_cuda.math_sdp_enabled(),
            "mem_efficient": actual_cuda.mem_efficient_sdp_enabled(),
            "reduction": actual_cuda.fp16_bf16_reduction_math_sdp_allowed(),
        }
        expected_other_states = {
            "cudnn": reference_torch.backends.cudnn.allow_tf32,
            "flash": expected_cuda.flash_sdp_enabled(),
            "math": expected_cuda.math_sdp_enabled(),
            "mem_efficient": expected_cuda.mem_efficient_sdp_enabled(),
            "reduction": expected_cuda.fp16_bf16_reduction_math_sdp_allowed(),
        }

        for allow_tf32 in (False, True):
            with self.subTest(allow_tf32=allow_tf32):
                actual.allow_tf32 = allow_tf32
                expected.allow_tf32 = allow_tf32
                self.assertIs(actual.allow_tf32, expected.allow_tf32)
                self.assertEqual(
                    {
                        "cudnn": torch.backends.cudnn.allow_tf32,
                        "precision": torch.get_float32_matmul_precision(),
                        "flash": actual_cuda.flash_sdp_enabled(),
                        "math": actual_cuda.math_sdp_enabled(),
                        "mem_efficient": actual_cuda.mem_efficient_sdp_enabled(),
                        "reduction": (
                            actual_cuda.fp16_bf16_reduction_math_sdp_allowed()
                        ),
                    },
                    actual_other_states,
                )
                self.assertEqual(
                    {
                        "cudnn": reference_torch.backends.cudnn.allow_tf32,
                        "flash": expected_cuda.flash_sdp_enabled(),
                        "math": expected_cuda.math_sdp_enabled(),
                        "mem_efficient": expected_cuda.mem_efficient_sdp_enabled(),
                        "reduction": (
                            expected_cuda.fp16_bf16_reduction_math_sdp_allowed()
                        ),
                    },
                    expected_other_states,
                )

        actual.allow_tf32 = True
        torch.set_float32_matmul_precision("highest")
        self.assertIs(actual.allow_tf32, True)
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")

        for enabled in (False, True):
            with self.subTest(cudnn=enabled):
                torch.backends.cudnn.allow_tf32 = enabled
                self.assertIs(actual.allow_tf32, True)
            with self.subTest(flash=enabled):
                actual_cuda.enable_flash_sdp(enabled)
                self.assertIs(actual.allow_tf32, True)
            with self.subTest(math=enabled):
                actual_cuda.enable_math_sdp(enabled)
                self.assertIs(actual.allow_tf32, True)
            with self.subTest(mem_efficient=enabled):
                actual_cuda.enable_mem_efficient_sdp(enabled)
                self.assertIs(actual.allow_tf32, True)
            with self.subTest(reduction=enabled):
                actual_cuda.allow_fp16_bf16_reduction_math_sdp(enabled)
                self.assertIs(actual.allow_tf32, True)

        for name in (
            "allow_fp16_accumulation",
            "allow_fp16_reduced_precision_reduction",
            "allow_fp16_reduced_precision_reduction_split_k",
            "allow_bf16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction_split_k",
            "fp32_precision",
        ):
            with self.subTest(unsupported=name):
                self.assertFalse(hasattr(actual, name))
                self.assertTrue(hasattr(expected, name))
                with self.assertRaises(AttributeError):
                    setattr(actual, name, True)

        self.assertIs(actual_cuda.is_built(), False)
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        self.assertTrue(reference_torch.backends.cuda.is_built())
        self.assertTrue(hasattr(reference_torch, "cuda"))


if __name__ == "__main__":
    unittest.main()
