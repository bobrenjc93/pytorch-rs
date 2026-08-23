import contextlib
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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaIsAvailableReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "cuda.is_available differentials require pinned PyTorch 2.13.0"
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

    def threaded_outcome(self, module):
        function = module.cuda.is_available
        baseline = function()
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = (
                        module.is_grad_enabled(),
                        function(),
                        module.is_grad_enabled(),
                        function(),
                        module.is_grad_enabled(),
                    )
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
        return baseline, worker_states

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_cuda = importlib.import_module("torch_rs.cuda")
        expected_cuda = importlib.import_module("torch.cuda")
        actual = actual_cuda.is_available
        expected = expected_cuda.is_available

        self.assertIs(torch.cuda, actual_cuda)
        self.assertIs(reference_torch.cuda, expected_cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_cuda)
        self.assertIs(sys.modules["torch.cuda"], expected_cuda)
        self.assertIs(type(actual_cuda), types.ModuleType)
        self.assertIs(type(expected_cuda), types.ModuleType)
        self.assertEqual(
            actual_cuda.__name__.replace("torch_rs", "torch"),
            expected_cuda.__name__,
        )
        self.assertEqual(
            actual_cuda.__package__.replace("torch_rs", "torch"),
            expected_cuda.__package__,
        )
        self.assertEqual(
            actual_cuda.__spec__.name.replace("torch_rs", "torch"),
            expected_cuda.__spec__.name,
        )
        self.assertEqual(actual_cuda.__doc__, expected_cuda.__doc__)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_cuda)
        self.assertIs(inspect.getmodule(expected), expected_cuda)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_imports_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_cuda = torch.cuda
        expected_cuda = reference_torch.cuda
        actual = actual_cuda.is_available
        expected = expected_cuda.is_available

        self.assertEqual(
            actual_cuda.__all__,
            [name for name in expected_cuda.__all__ if name == "is_available"],
        )
        self.assertEqual(
            torch.__all__.count("cuda"), reference_torch.__all__.count("cuda")
        )
        self.assertEqual(
            torch.__all__.count("is_available"),
            reference_torch.__all__.count("is_available"),
        )

        actual_package_import = {}
        expected_package_import = {}
        exec("from torch_rs import cuda", actual_package_import)
        exec("from torch import cuda", expected_package_import)
        self.assertIs(actual_package_import["cuda"], actual_cuda)
        self.assertIs(expected_package_import["cuda"], expected_cuda)

        actual_direct_import = {}
        expected_direct_import = {}
        exec("from torch_rs.cuda import is_available", actual_direct_import)
        exec("from torch.cuda import is_available", expected_direct_import)
        self.assertIs(actual_direct_import["is_available"], actual)
        self.assertIs(expected_direct_import["is_available"], expected)

        actual_cuda_namespace = {}
        expected_cuda_namespace = {}
        exec("from torch_rs.cuda import *", actual_cuda_namespace)
        exec("from torch.cuda import *", expected_cuda_namespace)
        self.assertEqual(
            {name for name in actual_cuda_namespace if not name.startswith("__")},
            {"is_available"},
        )
        self.assertIs(actual_cuda_namespace["is_available"], actual)
        self.assertIs(expected_cuda_namespace["is_available"], expected)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("cuda", namespace)
            self.assertNotIn("is_available", namespace)

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

    def reload_contract(self, root):
        module = root.cuda
        old_all = module.__all__
        old_function = module.is_available
        namespace = module.__dict__
        reloaded = importlib.reload(module)
        new_function = module.is_available

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
            self.fail("a stale CUDA availability query remained pickleable")

        return (
            reloaded is module,
            module.__dict__ is namespace,
            root.cuda is module,
            sys.modules[module.__name__] is module,
            module.__all__ is not old_all,
            old_function is not new_function,
            copy.copy(new_function) is new_function,
            copy.deepcopy(new_function) is new_function,
            pickle.loads(pickle.dumps(new_function)) is new_function,
            stale_pickle_error,
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )
        self.assertEqual(
            torch.cuda.__all__,
            [
                name
                for name in reference_torch.cuda.__all__
                if name == "is_available"
            ],
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(torch.cuda.is_available, protocol),
                    self.pickle_shape(reference_torch.cuda.is_available, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.cuda.is_available
        expected = reference_torch.cuda.is_available
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(device=True), lambda: expected(device=True)),
            (
                lambda: actual(None, device=True),
                lambda: expected(None, device=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_results_are_stable_across_threads_and_grad_modes(self):
        actual_baseline, actual_workers = self.threaded_outcome(torch)
        expected_baseline, expected_workers = self.threaded_outcome(reference_torch)

        self.assertIs(actual_baseline, False)
        self.assertIs(type(expected_baseline), bool)
        for baseline, worker_states in (
            (actual_baseline, actual_workers),
            (expected_baseline, expected_workers),
        ):
            for index, state in enumerate(worker_states):
                expected_grad_state = index % 2 == 0
                self.assertIs(state[0], expected_grad_state)
                self.assertIs(state[1], baseline)
                self.assertIs(state[2], expected_grad_state)
                self.assertIs(state[3], baseline)
                self.assertIs(state[4], expected_grad_state)

    def test_cuda_visible_h100_reference_and_unsupported_execution_boundary(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch build")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        self.assertIs(reference_torch.cuda.is_available(), True)
        self.assertGreaterEqual(reference_torch.cuda.device_count(), 1)
        self.assertIn("H100", device_name)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertIs(torch.cuda.is_available(), torch._C._has_cuda)
        self.assertIs(torch.backends.cuda.is_built(), False)

        probe = reference_torch.ones(1, device=reference_torch.device("cuda", 0))
        self.assertEqual(probe.item(), 1.0)
        reference_torch.cuda.synchronize(0)
        self.assertIs(reference_torch.cuda.is_available(), True)
        self.assertIs(torch.cuda.is_available(), False)

        actual_public = {
            name for name in vars(torch.cuda) if not name.startswith("_")
        }
        expected_exports = set(reference_torch.cuda.__all__)
        self.assertEqual(actual_public, {"is_available"})
        self.assertEqual(set(torch.cuda.__all__), {"is_available"})
        self.assertTrue(expected_exports - {"is_available"})
        for name in expected_exports - {"is_available"}:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.cuda, name))

        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        with self.assertRaises(RuntimeError):
            torch.tensor([1.0], device="cuda:0")


if __name__ == "__main__":
    unittest.main()
