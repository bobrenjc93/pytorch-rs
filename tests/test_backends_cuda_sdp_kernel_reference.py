import copy
import importlib
import inspect
import pickle
import pickletools
import re
import sys
import types
import unittest
import warnings

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_CUDA_NAMES = {
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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaSdpKernelReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cuda sdp_kernel differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = importlib.import_module("torch_rs.backends.cuda")
        self.expected = importlib.import_module("torch.backends.cuda")
        self.original_actual = self.states(self.actual)
        self.original_expected = self.states(self.expected)
        self.set_states(self.actual, (True, True, True))
        self.set_states(self.expected, (True, True, True))

    def tearDown(self):
        self.set_states(self.actual, self.original_actual)
        self.set_states(self.expected, self.original_expected)

    def states(self, module):
        return (
            module.flash_sdp_enabled(),
            module.math_sdp_enabled(),
            module.mem_efficient_sdp_enabled(),
        )

    def set_states(self, module, states):
        flash, math, mem_efficient = states
        module.enable_flash_sdp(flash)
        module.enable_math_sdp(math)
        module.enable_mem_efficient_sdp(mem_efficient)

    def context_result(self, module, initial, target, raise_body=False):
        self.set_states(module, initial)
        observations = [("before", self.states(module))]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                context = module.sdp_kernel(*target)
            observations.append(("created", self.states(module)))
            with context as entered:
                observations.append(("enter", entered, self.states(module)))
                if raise_body:
                    raise ValueError("body failed")
            observations.append(("exit", self.states(module)))
        except BaseException as error:
            observations.append(
                ("error", type(error).__name__, str(error), self.states(module))
            )
        return observations

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

    def test_valid_three_backend_context_behavior_matches_pytorch_2_13(self):
        cases = (
            ((True, True, True), (False, True, False), False),
            ((False, False, False), (True, True, True), False),
            ((True, False, True), (False, False, True), True),
        )
        for initial, target, raise_body in cases:
            with self.subTest(initial=initial, target=target, raise_body=raise_body):
                self.assertEqual(
                    self.context_result(self.actual, initial, target, raise_body),
                    self.context_result(self.expected, initial, target, raise_body),
                )

    def test_signature_exports_copying_and_pickling_match_supported_scope(self):
        actual = self.actual.sdp_kernel
        expected = self.expected.sdp_kernel
        actual_signature = inspect.signature(actual)
        expected_signature = inspect.signature(expected)

        self.assertEqual(
            list(actual_signature.parameters),
            ["enable_flash", "enable_math", "enable_mem_efficient"],
        )
        self.assertEqual(
            list(actual_signature.parameters),
            list(expected_signature.parameters)[:3],
        )
        for name in actual_signature.parameters:
            self.assertEqual(
                actual_signature.parameters[name].default,
                expected_signature.parameters[name].default,
            )
            self.assertEqual(
                actual_signature.parameters[name].annotation,
                expected_signature.parameters[name].annotation,
            )
        self.assertNotIn("enable_cudnn", actual_signature.parameters)

        self.assertEqual(
            self.actual.__all__,
            [name for name in self.expected.__all__ if name in SUPPORTED_CUDA_NAMES],
        )
        self.assertEqual(
            {name for name in vars(self.actual) if not name.startswith("_")},
            {
                name
                for name in vars(self.expected)
                if name in SUPPORTED_CUDA_NAMES | {"torch"}
            },
        )
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )

        for package_name, module, function in (
            ("torch_rs", self.actual, actual),
            ("torch", self.expected, expected),
        ):
            backend_import = {}
            function_import = {}
            wildcard = {}
            exec(f"from {package_name}.backends import cuda", backend_import)
            exec(
                f"from {package_name}.backends.cuda import sdp_kernel",
                function_import,
            )
            exec(f"from {package_name}.backends.cuda import *", wildcard)
            self.assertIs(backend_import["cuda"], module)
            self.assertIs(function_import["sdp_kernel"], function)
            self.assertIs(wildcard["sdp_kernel"], function)
            self.assertEqual(
                {name for name in wildcard if name in SUPPORTED_CUDA_NAMES},
                SUPPORTED_CUDA_NAMES,
            )

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_cudnn_and_execution_controls_remain_out_of_scope(self):
        for name in (
            "SDPAParams",
            "can_use_cudnn_attention",
            "can_use_efficient_attention",
            "can_use_flash_attention",
            "cudnn_sdp_enabled",
            "enable_cudnn_sdp",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(self.actual, name))
                self.assertTrue(hasattr(self.expected, name))

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch.nn.functional, "scaled_dot_product_attention"))
        self.assertTrue(
            hasattr(reference_torch.nn.functional, "scaled_dot_product_attention")
        )


if __name__ == "__main__":
    unittest.main()
