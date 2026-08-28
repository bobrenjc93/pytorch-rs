import copy
import importlib
import inspect
import json
import pickle
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
        raise AssertionError("cudnn.allow_tf32 must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudnnAllowTf32ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cudnn.allow_tf32 differentials require pinned "
                "PyTorch 2.13.0"
            )

    def fresh_cudnn_module(self, root):
        module_name = f"{root.__name__}.backends.cudnn"
        sys.modules.pop(module_name, None)
        if hasattr(root.backends, "cudnn"):
            del root.backends.cudnn
        module = importlib.import_module(module_name)
        root.backends.cudnn = module
        return module

    def setUp(self):
        self.actual = self.fresh_cudnn_module(torch)
        self.expected = self.fresh_cudnn_module(reference_torch)
        self.actual_original = (
            self.actual.allow_tf32,
            self.actual.enabled,
            self.actual.benchmark,
            self.actual.deterministic,
        )
        self.expected_original = (
            self.expected.allow_tf32,
            self.expected.enabled,
            self.expected.benchmark,
            self.expected.deterministic,
        )
        for module in (self.actual, self.expected):
            module.allow_tf32 = True
            module.enabled = True
            module.benchmark = False
            module.deterministic = False

    def tearDown(self):
        actual = self.fresh_cudnn_module(torch)
        expected = self.fresh_cudnn_module(reference_torch)
        (
            actual.allow_tf32,
            actual.enabled,
            actual.benchmark,
            actual.deterministic,
        ) = self.actual_original
        (
            expected.allow_tf32,
            expected.enabled,
            expected.benchmark,
            expected.deterministic,
        ) = self.expected_original

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

    def test_fresh_process_default_matches_pytorch_2_13(self):
        script = r'''
import json
import torch
import torch_rs

print(json.dumps({
    "actual": torch_rs.backends.cudnn.allow_tf32,
    "actual_type": type(torch_rs.backends.cudnn.allow_tf32).__name__,
    "expected": torch.backends.cudnn.allow_tf32,
    "expected_type": type(torch.backends.cudnn.allow_tf32).__name__,
}))
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "actual": True,
                "actual_type": "bool",
                "expected": True,
                "expected_type": "bool",
            },
        )

    def test_updates_errors_and_independence_match_pytorch_2_13(self):
        for module in (self.actual, self.expected):
            module.enabled = False
            module.benchmark = True
            module.deterministic = True

        for allow_tf32 in (False, True, True, False, False, True):
            with self.subTest(allow_tf32=allow_tf32):
                self.actual.allow_tf32 = allow_tf32
                self.expected.allow_tf32 = allow_tf32
                self.assertIs(self.actual.allow_tf32, self.expected.allow_tf32)
                self.assertIs(
                    type(self.actual.allow_tf32),
                    type(self.expected.allow_tf32),
                )
                self.assertIs(self.actual.enabled, self.expected.enabled)
                self.assertIs(self.actual.benchmark, self.expected.benchmark)
                self.assertIs(
                    self.actual.deterministic,
                    self.expected.deterministic,
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
            for module in (self.actual, self.expected):
                module.allow_tf32 = state
                module.enabled = not state
                module.benchmark = state
                module.deterministic = not state
            for case, (actual_value, expected_value) in enumerate(
                zip(actual_values, expected_values)
            ):
                with self.subTest(state=state, case=case):
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
                    self.assertIs(self.actual.enabled, self.expected.enabled)
                    self.assertIs(self.actual.benchmark, self.expected.benchmark)
                    self.assertIs(
                        self.actual.deterministic,
                        self.expected.deterministic,
                    )

                    self.assert_error_matches(
                        lambda value=actual_value: (
                            torch._C._set_cudnn_allow_tf32(value)
                        ),
                        lambda value=expected_value: (
                            reference_torch._C._set_cudnn_allow_tf32(value)
                        ),
                    )
                    self.assertIs(self.actual.allow_tf32, state)
                    self.assertIs(self.expected.allow_tf32, state)

    def thread_contract(self, module):
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []
        module.allow_tf32 = False

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
        main_saw_worker = module.allow_tf32 is True
        module.allow_tf32 = False
        main_changed.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            main_saw_worker,
            not thread.is_alive(),
            errors,
            observations,
            module.allow_tf32,
        )

    def test_process_global_thread_visibility_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_contract(self.actual),
            self.thread_contract(self.expected),
        )

    def reload_contract(self, root, module):
        parent = root.backends
        namespace = module.__dict__
        module.allow_tf32 = False

        reloaded = importlib.reload(module)
        initial = (
            reloaded is module,
            module.__dict__ is namespace,
            parent.cudnn is module,
            sys.modules[module.__name__] is reloaded,
            reloaded.m is module,
            module.allow_tf32,
            reloaded.allow_tf32,
        )
        reloaded.allow_tf32 = True
        old_saw_new = module.allow_tf32
        module.allow_tf32 = False
        new_saw_old = reloaded.allow_tf32
        fresh = self.fresh_cudnn_module(root)
        fresh_state = fresh.allow_tf32
        return initial, old_saw_new, new_saw_old, fresh_state

    def test_reload_and_deletion_match_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

        actual = self.fresh_cudnn_module(torch)
        expected = self.fresh_cudnn_module(reference_torch)
        actual.allow_tf32 = False
        expected.allow_tf32 = False
        self.assert_error_matches(
            lambda: delattr(actual, "allow_tf32"),
            lambda: delattr(expected, "allow_tf32"),
        )
        self.assertIs(actual.allow_tf32, False)
        self.assertIs(expected.allow_tf32, False)

    def signature_outcome(self, function):
        try:
            return "return", str(inspect.signature(function))
        except BaseException as error:
            return "error", type(error).__name__, str(error)

    def test_proxy_and_private_accessor_contract_matches_pytorch_2_13(self):
        actual_descriptor = vars(type(self.actual))["allow_tf32"]
        expected_descriptor = vars(type(self.expected))["allow_tf32"]

        self.assertEqual(type(self.actual).__name__, type(self.expected).__name__)
        self.assertEqual(
            self.normalize(type(self.actual).__module__),
            type(self.expected).__module__,
        )
        self.assertIs(self.actual.m.__annotations__["allow_tf32"], bool)
        self.assertIs(self.expected.m.__annotations__["allow_tf32"], bool)
        for module in (self.actual, self.expected):
            self.assertNotIn("allow_tf32", vars(module))
            self.assertNotIn("allow_tf32", vars(module.m))
            self.assertNotIn("allow_tf32", dir(module))
        self.assertEqual(
            set(vars(actual_descriptor)),
            set(vars(expected_descriptor)),
        )
        self.assertIs(
            actual_descriptor.__get__(self.actual, type(self.actual)),
            expected_descriptor.__get__(self.expected, type(self.expected)),
        )

        actual_import = {}
        expected_import = {}
        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.backends.cudnn import allow_tf32", actual_import)
        exec("from torch.backends.cudnn import allow_tf32", expected_import)
        exec("from torch_rs.backends.cudnn import *", actual_wildcard)
        exec("from torch.backends.cudnn import *", expected_wildcard)
        self.assertIs(actual_import["allow_tf32"], expected_import["allow_tf32"])
        self.assertEqual(
            "allow_tf32" in actual_wildcard,
            "allow_tf32" in expected_wildcard,
        )

        for name in ("_get_cudnn_allow_tf32", "_set_cudnn_allow_tf32"):
            actual = getattr(torch._C, name)
            expected = getattr(reference_torch._C, name)
            with self.subTest(name=name):
                self.assertIs(type(actual), types.BuiltinFunctionType)
                self.assertIs(type(expected), types.BuiltinFunctionType)
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(self.normalize(actual.__module__), expected.__module__)
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__text_signature__, expected.__text_signature__)
                self.assertEqual(repr(actual), repr(expected))
                self.assertIs(actual.__self__, torch._C)
                self.assertIs(expected.__self__, reference_torch._C)
                self.assertEqual(actual.__reduce__(), expected.__reduce__())
                self.assertEqual(
                    self.normalize(self.signature_outcome(actual)),
                    self.signature_outcome(expected),
                )
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(pickle.loads(pickle.dumps(actual)), actual)

        binding_calls = (
            lambda root: root._C._get_cudnn_allow_tf32(None),
            lambda root: root._C._get_cudnn_allow_tf32(value=None),
            lambda root: root._C._set_cudnn_allow_tf32(),
            lambda root: root._C._set_cudnn_allow_tf32(True, False),
            lambda root: root._C._set_cudnn_allow_tf32(object=False),
        )
        self.actual.allow_tf32 = False
        self.expected.allow_tf32 = False
        for case, call in enumerate(binding_calls):
            with self.subTest(kind="binding", case=case):
                self.assert_error_matches(
                    lambda call=call: call(torch),
                    lambda call=call: call(reference_torch),
                )
                self.assertIs(self.actual.allow_tf32, False)
                self.assertIs(self.expected.allow_tf32, False)

        for root in (torch, reference_torch):
            with self.subTest(package=root.__name__):
                self.assertTrue(hasattr(root._C, "_get_cudnn_allow_tf32"))
                self.assertTrue(hasattr(root._C, "_set_cudnn_allow_tf32"))
                self.assertFalse(hasattr(root, "_get_cudnn_allow_tf32"))
                self.assertFalse(hasattr(root, "_set_cudnn_allow_tf32"))


if __name__ == "__main__":
    unittest.main()
