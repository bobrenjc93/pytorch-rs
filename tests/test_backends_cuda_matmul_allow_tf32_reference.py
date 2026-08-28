import copy
import importlib
import inspect
import json
import os
import pickle
import pickletools
import re
import subprocess
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

    def fresh_cuda_module(self, root):
        module_name = f"{root.__name__}.backends.cuda"
        sys.modules.pop(module_name, None)
        if hasattr(root.backends, "cuda"):
            del root.backends.cuda
        module = importlib.import_module(module_name)
        root.backends.cuda = module
        return module

    def setUp(self):
        self.actual = self.fresh_cuda_module(torch)
        self.expected = self.fresh_cuda_module(reference_torch)
        self.actual_original = self.actual.matmul.allow_tf32
        self.expected_original = self.expected.matmul.allow_tf32
        self.actual.matmul.allow_tf32 = False
        self.expected.matmul.allow_tf32 = False

    def tearDown(self):
        actual = self.fresh_cuda_module(torch)
        expected = self.fresh_cuda_module(reference_torch)
        actual.matmul.allow_tf32 = self.actual_original
        expected.matmul.allow_tf32 = self.expected_original

    def normalize(self, value):
        if isinstance(value, str):
            return re.sub(r"0x[0-9a-fA-F]+", "0x...", value).replace(
                "torch_rs.torch_rs",
                "torch._C",
            ).replace("torch_rs", "torch")
        if isinstance(value, tuple):
            return tuple(self.normalize(item) for item in value)
        if isinstance(value, list):
            return [self.normalize(item) for item in value]
        return value

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(
            self.normalize(str(actual_raised.exception)),
            self.normalize(str(expected_raised.exception)),
        )
        self.assertEqual(
            self.normalize(actual_raised.exception.args),
            self.normalize(expected_raised.exception.args),
        )

    def pickle_shape(self, value, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(value, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs.torch_rs", "torch._C").replace(
                    "torch_rs",
                    "torch",
                )
            shape.append((opcode.name, argument))
        return shape

    def test_exact_bool_updates_errors_and_deletion_match_pytorch_2_13(self):
        actual = self.actual.matmul
        expected = self.expected.matmul

        for allow_tf32 in (True, False, False, True, True, False):
            with self.subTest(allow_tf32=allow_tf32):
                actual.allow_tf32 = allow_tf32
                expected.allow_tf32 = allow_tf32
                self.assertIs(actual.allow_tf32, expected.allow_tf32)
                self.assertIs(type(actual.allow_tf32), type(expected.allow_tf32))

        invalid_values = (
            None,
            0,
            1,
            0.0,
            np.bool_(True),
            "",
            [],
            object(),
            _RejectTruthiness(),
        )
        for state in (False, True):
            actual.allow_tf32 = state
            expected.allow_tf32 = state
            for value in invalid_values:
                with self.subTest(state=state, value_type=type(value).__name__):
                    self.assert_error_matches(
                        lambda value=value: setattr(actual, "allow_tf32", value),
                        lambda value=value: setattr(expected, "allow_tf32", value),
                    )
                    self.assertIs(actual.allow_tf32, state)
                    self.assertIs(expected.allow_tf32, state)

        for actual_value, expected_value in (
            (torch.tensor(True), reference_torch.tensor(True)),
            (torch.float32, reference_torch.float32),
            (torch.device("cpu"), reference_torch.device("cpu")),
            (torch.strided, reference_torch.strided),
            (torch.Size([1]), reference_torch.Size([1])),
            (torch.finfo(torch.float32), reference_torch.finfo(reference_torch.float32)),
        ):
            with self.subTest(value_type=type(actual_value).__name__):
                self.assert_error_matches(
                    lambda: setattr(actual, "allow_tf32", actual_value),
                    lambda: setattr(expected, "allow_tf32", expected_value),
                )
                self.assertIs(actual.allow_tf32, state)
                self.assertIs(expected.allow_tf32, state)

        self.assert_error_matches(
            lambda: delattr(actual, "allow_tf32"),
            lambda: delattr(expected, "allow_tf32"),
        )
        self.assertIs(actual.allow_tf32, state)
        self.assertIs(expected.allow_tf32, state)

        for name in ("unknown", "_unknown"):
            with self.subTest(name=name, operation="get"):
                self.assert_error_matches(
                    lambda name=name: getattr(actual, name),
                    lambda name=name: getattr(expected, name),
                )
            with self.subTest(name=name, operation="set"):
                self.assert_error_matches(
                    lambda name=name: setattr(actual, name, False),
                    lambda name=name: setattr(expected, name, False),
                )

    def thread_contract(self, module):
        matmul = module.matmul
        copied = copy.copy(matmul)
        restored = pickle.loads(pickle.dumps(matmul))
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []
        matmul.allow_tf32 = True

        def worker():
            try:
                observations.append(restored.allow_tf32)
                copied.allow_tf32 = False
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(restored.allow_tf32)
                restored.allow_tf32 = False
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_changed.wait(timeout=10)
        main_saw_worker = matmul.allow_tf32 is False
        matmul.allow_tf32 = True
        main_changed.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            main_saw_worker,
            not thread.is_alive(),
            errors,
            observations,
            matmul.allow_tf32,
            copied.allow_tf32,
            restored.allow_tf32,
        )

    def test_process_global_thread_visibility_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_contract(self.actual),
            self.thread_contract(self.expected),
        )

    def cross_api_contract(self, root):
        matmul = root.backends.cuda.matmul
        root.set_float32_matmul_precision("highest")
        initial = (matmul.allow_tf32, root.get_float32_matmul_precision())
        matmul.allow_tf32 = True
        enabled = (matmul.allow_tf32, root.get_float32_matmul_precision())
        result = root.set_float32_matmul_precision("highest")
        disabled = (matmul.allow_tf32, root.get_float32_matmul_precision())
        return initial, enabled, result is None, disabled

    def test_shared_float32_matmul_precision_state_matches_pytorch_2_13(self):
        self.assertEqual(
            self.cross_api_contract(torch),
            self.cross_api_contract(reference_torch),
        )

    def reload_contract(self, root, module):
        parent = root.backends
        namespace = module.__dict__
        old_proxy = module.matmul
        old_type = type(old_proxy)
        old_proxy.allow_tf32 = False

        reloaded = importlib.reload(module)
        new_proxy = reloaded.matmul
        initial = (
            reloaded is module,
            module.__dict__ is namespace,
            parent.cuda is module,
            sys.modules[module.__name__] is module,
            new_proxy is not old_proxy,
            type(new_proxy) is not old_type,
            type(new_proxy) is module.cuBLASModule,
            old_proxy.allow_tf32,
            new_proxy.allow_tf32,
        )
        new_proxy.allow_tf32 = True
        old_saw_new = old_proxy.allow_tf32
        old_proxy.allow_tf32 = False
        new_saw_old = new_proxy.allow_tf32

        try:
            pickle.dumps(old_proxy)
        except Exception as error:
            stale_pickle_error = (
                type(error).__name__,
                self.normalize(str(error)),
            )
        else:
            self.fail("a stale CUDA matmul proxy remained pickleable")

        restored = pickle.loads(pickle.dumps(new_proxy))
        current_pickle = (
            type(restored) is type(new_proxy),
            restored is not new_proxy,
            restored.allow_tf32,
        )
        fresh = self.fresh_cuda_module(root)
        fresh_contract = (
            fresh is not module,
            fresh.matmul is not new_proxy,
            fresh.matmul.allow_tf32,
        )
        return (
            initial,
            old_saw_new,
            new_saw_old,
            stale_pickle_error,
            current_pickle,
            fresh_contract,
        )

    def test_reload_and_fresh_import_state_match_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def signature_outcome(self, value):
        try:
            return "return", str(inspect.signature(value))
        except BaseException as error:
            return "error", type(error).__name__, self.normalize(str(error))

    def test_proxy_metadata_imports_copy_and_pickle_match_pytorch_2_13(self):
        actual_module = self.actual
        expected_module = self.expected
        actual = actual_module.matmul
        expected = expected_module.matmul
        actual_type = type(actual)
        expected_type = type(expected)

        self.assertEqual(set(vars(actual_type)), set(vars(expected_type)))
        self.assertEqual(actual_type.__name__, expected_type.__name__)
        self.assertEqual(actual_type.__qualname__, expected_type.__qualname__)
        self.assertEqual(
            self.normalize(actual_type.__module__),
            expected_type.__module__,
        )
        self.assertEqual(actual_type.__doc__, expected_type.__doc__)
        self.assertEqual(actual_type.__annotations__, expected_type.__annotations__)
        self.assertEqual(
            inspect.get_annotations(actual_type),
            inspect.get_annotations(expected_type),
        )
        self.assertEqual(
            self.signature_outcome(actual_type),
            self.signature_outcome(expected_type),
        )
        self.assertEqual(vars(actual), vars(expected))
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            self.normalize(actual.__module__),
            expected.__module__,
        )
        self.assertEqual(
            "allow_tf32" in dir(actual),
            "allow_tf32" in dir(expected),
        )
        self.assertEqual(
            self.signature_outcome(actual),
            self.signature_outcome(expected),
        )

        actual_import = {}
        expected_import = {}
        actual_wildcard = {}
        expected_wildcard = {}
        exec(
            "from torch_rs.backends.cuda import cuBLASModule, matmul",
            actual_import,
        )
        exec(
            "from torch.backends.cuda import cuBLASModule, matmul",
            expected_import,
        )
        exec("from torch_rs.backends.cuda import *", actual_wildcard)
        exec("from torch.backends.cuda import *", expected_wildcard)
        self.assertIs(actual_import["cuBLASModule"], actual_type)
        self.assertIs(actual_import["matmul"], actual)
        self.assertIs(expected_import["cuBLASModule"], expected_type)
        self.assertIs(expected_import["matmul"], expected)
        self.assertIs(actual_wildcard["cuBLASModule"], actual_type)
        self.assertIs(actual_wildcard["matmul"], actual)
        self.assertIs(expected_wildcard["cuBLASModule"], expected_type)
        self.assertIs(expected_wildcard["matmul"], expected)

        for package_name in ("torch_rs", "torch"):
            with self.subTest(package_name=package_name):
                with self.assertRaises(ModuleNotFoundError) as raised:
                    importlib.import_module(f"{package_name}.backends.cuda.matmul")
                self.assertEqual(
                    self.normalize(str(raised.exception)),
                    "No module named 'torch.backends.cuda.matmul'",
                )

        for actual_copier, expected_copier in (
            (copy.copy, copy.copy),
            (copy.deepcopy, copy.deepcopy),
        ):
            actual_copy = actual_copier(actual)
            expected_copy = expected_copier(expected)
            self.assertEqual(
                (
                    actual_copy is not actual,
                    type(actual_copy) is actual_type,
                    vars(actual_copy),
                    actual_copy.allow_tf32,
                ),
                (
                    expected_copy is not expected,
                    type(expected_copy) is expected_type,
                    vars(expected_copy),
                    expected_copy.allow_tf32,
                ),
            )

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                actual_restored = pickle.loads(pickle.dumps(actual, protocol))
                expected_restored = pickle.loads(pickle.dumps(expected, protocol))
                self.assertEqual(
                    (
                        actual_restored is not actual,
                        type(actual_restored) is actual_type,
                        vars(actual_restored),
                        actual_restored.allow_tf32,
                    ),
                    (
                        expected_restored is not expected,
                        type(expected_restored) is expected_type,
                        vars(expected_restored),
                        expected_restored.allow_tf32,
                    ),
                )
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_private_accessor_contract_matches_pytorch_2_13(self):
        for name in (
            "_get_cublas_allow_tf32",
            "_set_cublas_allow_tf32",
        ):
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
                self.assertEqual(actual.__text_signature__, expected.__text_signature__)
                self.assertEqual(
                    self.signature_outcome(actual),
                    self.signature_outcome(expected),
                )
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.deepcopy(expected), expected)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)),
                        expected,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

        self.assertIs(
            torch._C._set_cublas_allow_tf32(False),
            reference_torch._C._set_cublas_allow_tf32(False),
        )
        self.assertIs(
            torch._C._get_cublas_allow_tf32(),
            reference_torch._C._get_cublas_allow_tf32(),
        )

        binding_calls = (
            lambda root: root._C._get_cublas_allow_tf32(None),
            lambda root: root._C._get_cublas_allow_tf32(value=None),
            lambda root: root._C._set_cublas_allow_tf32(),
            lambda root: root._C._set_cublas_allow_tf32(True, False),
            lambda root: root._C._set_cublas_allow_tf32(object=False),
        )
        self.actual.matmul.allow_tf32 = True
        self.expected.matmul.allow_tf32 = True
        for case, call in enumerate(binding_calls):
            with self.subTest(kind="binding", case=case):
                self.assert_error_matches(
                    lambda call=call: call(torch),
                    lambda call=call: call(reference_torch),
                )
                self.assertIs(self.actual.matmul.allow_tf32, True)
                self.assertIs(self.expected.matmul.allow_tf32, True)

    def test_supported_namespace_does_not_claim_cuda_execution(self):
        actual_public = {
            name for name in vars(self.actual) if not name.startswith("_")
        }
        expected_public = {
            name for name in vars(self.expected) if not name.startswith("_")
        }
        self.assertTrue(actual_public.issubset(expected_public))
        self.assertIn("cuBLASModule", actual_public)
        self.assertIn("matmul", actual_public)
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertTrue(hasattr(reference_torch, "cuda"))
        self.assertIs(self.actual.is_built(), False)
        for name in (
            "allow_bf16_reduced_precision_reduction",
            "allow_fp16_accumulation",
            "allow_fp16_reduced_precision_reduction",
            "fp32_precision",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(self.actual.matmul, name))
                self.assertTrue(hasattr(self.expected.matmul, name))

    def test_environment_override_and_one_time_snapshot_match_pytorch_2_13(self):
        script = r'''
import importlib
import json
import os

import torch_rs as actual
import torch as expected

def state(root):
    return [
        root.backends.cuda.matmul.allow_tf32,
        root.get_float32_matmul_precision(),
    ]

initial = [state(actual), state(expected)]
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = (
    "0" if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1" else "1"
)
importlib.reload(actual.backends.cuda)
importlib.reload(expected.backends.cuda)
after_reload = [state(actual), state(expected)]
actual.set_float32_matmul_precision("highest")
expected.set_float32_matmul_precision("highest")
after_highest = [state(actual), state(expected)]
print(json.dumps([initial, after_reload, after_highest]))
'''
        for value, initial in (
            (None, [False, "highest"]),
            ("0", [False, "highest"]),
            ("1", [True, "high"]),
        ):
            with self.subTest(value=value):
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = "0"
                if value is None:
                    environment.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
                else:
                    environment["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = value
                completed = subprocess.run(
                    [sys.executable, "-c", script],
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
                self.assertEqual(
                    json.loads(completed.stdout),
                    [
                        [initial, initial],
                        [initial, initial],
                        [[False, "highest"], [False, "highest"]],
                    ],
                )


if __name__ == "__main__":
    unittest.main()
