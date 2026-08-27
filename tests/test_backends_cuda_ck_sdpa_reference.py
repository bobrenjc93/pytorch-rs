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

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_CUDA_EXPORTS = {
    "is_built",
    "is_ck_sdpa_available",
    "is_flash_attention_available",
}


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaCkSdpaAvailabilityReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Composable Kernel SDPA availability differentials require "
                "pinned PyTorch 2.13.0"
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

    def threaded_contract(self, root):
        function = root.backends.cuda.is_ck_sdpa_available
        native_function = root._C._is_ck_sdpa_available
        barrier = threading.Barrier(16)
        results = [None] * 16
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=5)
                value = function()
                results[index] = (
                    value,
                    type(value) is bool,
                    value is native_function(),
                    function is root.backends.cuda.is_ck_sdpa_available,
                    native_function is root._C._is_ck_sdpa_available,
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(16)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        return (
            [thread.is_alive() for thread in threads],
            errors,
            results,
        )

    def reload_contract(self, root):
        parent = root.backends
        module = parent.cuda
        native = root._C
        old_function = module.is_ck_sdpa_available
        native_function = native._is_ck_sdpa_available
        namespace = module.__dict__
        reloaded = importlib.reload(module)
        new_function = module.is_ck_sdpa_available

        try:
            pickle.dumps(old_function)
        except Exception as error:
            stale_pickle_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                    "torch_rs", "torch"
                ),
            )
        else:
            self.fail("a stale Composable Kernel SDPA build query remained pickleable")

        native_reloaded = importlib.reload(native)
        return (
            reloaded is module,
            module.__dict__ is namespace,
            parent.cuda is module,
            sys.modules[module.__name__] is module,
            old_function is not new_function,
            copy.copy(new_function) is new_function,
            copy.deepcopy(new_function) is new_function,
            pickle.loads(pickle.dumps(new_function)) is new_function,
            stale_pickle_error,
            native_reloaded is native,
            root._C is native,
            native._is_ck_sdpa_available is native_function,
            native_function() is False,
        )

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.cuda")
        expected_module = importlib.import_module("torch.backends.cuda")
        actual = actual_module.is_ck_sdpa_available
        expected = expected_module.is_ck_sdpa_available

        self.assertIsNone(actual_module.__doc__)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertEqual(
            actual_module.__all__,
            [
                name
                for name in expected_module.__all__
                if name in SUPPORTED_CUDA_EXPORTS
            ],
        )
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
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
        self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)

    def test_imports_copying_and_pickling_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.cuda")
        expected_module = importlib.import_module("torch.backends.cuda")
        actual = actual_module.is_ck_sdpa_available
        expected = expected_module.is_ck_sdpa_available

        for package_name, module, function in (
            ("torch_rs", actual_module, actual),
            ("torch", expected_module, expected),
        ):
            function_import = {}
            exec(
                f"from {package_name}.backends.cuda import is_ck_sdpa_available",
                function_import,
            )
            self.assertIs(function_import["is_ck_sdpa_available"], function)

        actual_child_wildcard = {}
        expected_child_wildcard = {}
        exec("from torch_rs.backends.cuda import *", actual_child_wildcard)
        exec("from torch.backends.cuda import *", expected_child_wildcard)
        self.assertEqual(
            {name for name in actual_child_wildcard if not name.startswith("__")},
            {
                name
                for name in expected_child_wildcard
                if name in SUPPORTED_CUDA_EXPORTS
            },
        )
        self.assertIs(actual_child_wildcard["is_ck_sdpa_available"], actual)
        self.assertIs(expected_child_wildcard["is_ck_sdpa_available"], expected)

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

    def test_thread_behavior_matches_pytorch_2_13(self):
        actual = self.threaded_contract(torch)
        expected = self.threaded_contract(reference_torch)

        self.assertEqual(actual, expected)
        self.assertEqual(
            actual,
            ([False] * 16, [], [(False, True, True, True, True)] * 16),
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )
        actual = torch.backends.cuda.is_ck_sdpa_available
        expected = reference_torch.backends.cuda.is_ck_sdpa_available
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.backends.cuda.is_ck_sdpa_available
        expected = reference_torch.backends.cuda.is_ck_sdpa_available
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_cuda_enabled_h100_reports_ck_unavailable_and_execution_boundary(self):
        if not reference_torch.backends.cuda.is_built():
            self.skipTest("requires a CUDA-built reference PyTorch")
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        self.assertIs(reference_torch.backends.cuda.is_built(), True)
        self.assertIs(reference_torch.backends.cuda.is_ck_sdpa_available(), False)
        self.assertIs(reference_torch._C._is_ck_sdpa_available(), False)
        self.assertIs(torch.backends.cuda.is_ck_sdpa_available(), False)
        self.assertIs(torch._C._is_ck_sdpa_available(), False)
        self.assertIs(torch.backends.cuda.is_built(), False)

        device = reference_torch.device("cuda", 0)
        query = reference_torch.ones((1, 1, 2, 8), device=device)
        result = reference_torch.nn.functional.scaled_dot_product_attention(
            query,
            query,
            query,
        )
        reference_torch.cuda.synchronize(device)
        self.assertEqual(result.cpu().tolist(), [[[[1.0] * 8] * 2]])

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        self.assertFalse(
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
        )
        self.assertTrue(
            hasattr(
                reference_torch.nn.functional,
                "scaled_dot_product_attention",
            )
        )
        for name in (
            "SDPAParams",
            "enable_cudnn_sdp",
            "enable_flash_sdp",
            "enable_math_sdp",
            "enable_mem_efficient_sdp",
            "preferred_rocm_fa_library",
            "sdp_kernel",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.backends.cuda, name))
                self.assertTrue(hasattr(reference_torch.backends.cuda, name))

        with self.assertRaises(RuntimeError):
            torch.tensor([1.0], device="cuda:0")


if __name__ == "__main__":
    unittest.main()
