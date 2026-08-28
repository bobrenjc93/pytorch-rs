import copy
import importlib
import inspect
import json
import pickle
import pickletools
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
class CudnnAllowTF32ReferenceTests(unittest.TestCase):
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
            self.actual.benchmark,
            self.actual.deterministic,
            self.actual.enabled,
        )
        self.expected_original = (
            self.expected.allow_tf32,
            self.expected.benchmark,
            self.expected.deterministic,
            self.expected.enabled,
        )
        self.actual.allow_tf32 = True
        self.expected.allow_tf32 = True
        self.actual.benchmark = False
        self.expected.benchmark = False
        self.actual.deterministic = False
        self.expected.deterministic = False
        self.actual.enabled = True
        self.expected.enabled = True

    def tearDown(self):
        actual = self.fresh_cudnn_module(torch)
        expected = self.fresh_cudnn_module(reference_torch)
        (
            actual.allow_tf32,
            actual.benchmark,
            actual.deterministic,
            actual.enabled,
        ) = self.actual_original
        (
            expected.allow_tf32,
            expected.benchmark,
            expected.deterministic,
            expected.enabled,
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

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = self.normalize(argument)
            shape.append((opcode.name, argument))
        return shape

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

    def test_exact_bool_updates_errors_and_independence_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected

        actual.enabled = False
        expected.enabled = False
        actual.benchmark = True
        expected.benchmark = True
        actual.deterministic = True
        expected.deterministic = True
        for allow_tf32 in (True, False, False, True, True, False):
            with self.subTest(allow_tf32=allow_tf32):
                actual.allow_tf32 = allow_tf32
                expected.allow_tf32 = allow_tf32
                self.assertIs(actual.allow_tf32, expected.allow_tf32)
                self.assertIs(
                    type(actual.allow_tf32),
                    type(expected.allow_tf32),
                )
                self.assertIs(actual.enabled, expected.enabled)
                self.assertIs(actual.benchmark, expected.benchmark)
                self.assertIs(actual.deterministic, expected.deterministic)
                self.assertIs(actual.is_available(), False)
                self.assertIs(actual.version(), None)

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
            actual.allow_tf32 = state
            expected.allow_tf32 = state
            actual.enabled = not state
            expected.enabled = not state
            actual.benchmark = state
            expected.benchmark = state
            actual.deterministic = not state
            expected.deterministic = not state
            for case, (actual_value, expected_value) in enumerate(
                zip(actual_values, expected_values)
            ):
                with self.subTest(state=state, case=case):
                    self.assert_error_matches(
                        lambda value=actual_value: setattr(
                            actual,
                            "allow_tf32",
                            value,
                        ),
                        lambda value=expected_value: setattr(
                            expected,
                            "allow_tf32",
                            value,
                        ),
                    )
                    self.assertIs(actual.allow_tf32, state)
                    self.assertIs(expected.allow_tf32, state)
                    self.assertIs(actual.enabled, expected.enabled)
                    self.assertIs(actual.benchmark, expected.benchmark)
                    self.assertIs(actual.deterministic, expected.deterministic)

                    self.assert_error_matches(
                        lambda value=actual_value: (
                            torch._C._set_cudnn_allow_tf32(value)
                        ),
                        lambda value=expected_value: (
                            reference_torch._C._set_cudnn_allow_tf32(value)
                        ),
                    )
                    self.assertIs(actual.allow_tf32, state)
                    self.assertIs(expected.allow_tf32, state)
                    self.assertIs(actual.enabled, expected.enabled)
                    self.assertIs(actual.benchmark, expected.benchmark)
                    self.assertIs(actual.deterministic, expected.deterministic)

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
        module.allow_tf32 = True

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
        reloaded.allow_tf32 = False
        old_saw_new = module.allow_tf32
        module.allow_tf32 = True
        new_saw_old = reloaded.allow_tf32
        fresh = self.fresh_cudnn_module(root)
        fresh_state = fresh.allow_tf32
        return initial, old_saw_new, new_saw_old, fresh_state

    def test_reload_and_fresh_import_state_match_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_proxy_metadata_imports_deletion_copying_and_pickling_match(self):
        actual = self.actual
        expected = self.expected
        actual_descriptor = vars(type(actual))["allow_tf32"]
        expected_descriptor = vars(type(expected))["allow_tf32"]

        self.assertEqual(type(actual).__name__, type(expected).__name__)
        self.assertEqual(
            self.normalize(type(actual).__module__),
            type(expected).__module__,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(hasattr(actual, "__all__"), hasattr(expected, "__all__"))
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {name for name in vars(expected) if not name.startswith("_")},
        )
        self.assertIs(type(actual.m), types.ModuleType)
        self.assertIs(type(expected.m), types.ModuleType)
        self.assertIs(actual.m.__annotations__["allow_tf32"], bool)
        self.assertIs(expected.m.__annotations__["allow_tf32"], bool)
        for module in (actual, expected):
            self.assertNotIn("allow_tf32", vars(module))
            self.assertNotIn("allow_tf32", vars(module.m))
            self.assertNotIn("allow_tf32", dir(module))

        self.assertEqual(
            set(vars(actual_descriptor)),
            set(vars(expected_descriptor)),
        )
        self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)
        self.assertIs(
            actual_descriptor.getter,
            torch._C._get_cudnn_allow_tf32,
        )
        self.assertIs(
            actual_descriptor.setter,
            torch._C._set_cudnn_allow_tf32,
        )
        self.assertIs(
            expected_descriptor.getter,
            reference_torch._C._get_cudnn_allow_tf32,
        )
        self.assertIs(
            expected_descriptor.setter,
            reference_torch._C._set_cudnn_allow_tf32,
        )
        self.assertIs(
            actual_descriptor.__get__(actual, type(actual)),
            expected_descriptor.__get__(expected, type(expected)),
        )

        actual_import = {}
        expected_import = {}
        actual_wildcard = {}
        expected_wildcard = {}
        exec(
            "from torch_rs.backends.cudnn import allow_tf32",
            actual_import,
        )
        exec(
            "from torch.backends.cudnn import allow_tf32",
            expected_import,
        )
        exec("from torch_rs.backends.cudnn import *", actual_wildcard)
        exec("from torch.backends.cudnn import *", expected_wildcard)
        self.assertIs(
            actual_import["allow_tf32"],
            expected_import["allow_tf32"],
        )
        self.assertEqual(
            "allow_tf32" in actual_wildcard,
            "allow_tf32" in expected_wildcard,
        )

        actual.allow_tf32 = True
        expected.allow_tf32 = True
        self.assert_error_matches(
            lambda: delattr(actual, "allow_tf32"),
            lambda: delattr(expected, "allow_tf32"),
        )
        self.assertIs(actual.allow_tf32, True)
        self.assertIs(expected.allow_tf32, True)

        for module in (actual, expected):
            for state in (False, True):
                module.allow_tf32 = state
                self.assertIs(copy.copy(module.allow_tf32), state)
                self.assertIs(copy.deepcopy(module.allow_tf32), state)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(
                            pickle.dumps(module.allow_tf32, protocol=protocol)
                        ),
                        state,
                    )

        for descriptor in (actual_descriptor, expected_descriptor):
            for copier in (copy.copy, copy.deepcopy):
                copied = copier(descriptor)
                self.assertIsNot(copied, descriptor)
                self.assertIs(type(copied), type(descriptor))
                self.assertIs(copied.getter, descriptor.getter)
                self.assertIs(copied.setter, descriptor.setter)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                restored = pickle.loads(pickle.dumps(descriptor, protocol))
                self.assertIsNot(restored, descriptor)
                self.assertIs(type(restored), type(descriptor))
                self.assertIs(restored.getter, descriptor.getter)
                self.assertIs(restored.setter, descriptor.setter)

        for copier in (copy.copy, copy.deepcopy):
            errors = []
            for module in (actual, expected):
                try:
                    copier(module)
                except Exception as error:
                    errors.append((type(error), str(error), error.args))
                else:
                    self.fail("a cuDNN module proxy unexpectedly supported copying")
            self.assertEqual(errors[0], errors[1])
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            errors = []
            for module in (actual, expected):
                try:
                    pickle.dumps(module, protocol)
                except Exception as error:
                    errors.append((type(error), str(error), error.args))
                else:
                    self.fail("a cuDNN module proxy unexpectedly pickled")
            self.assertEqual(errors[0], errors[1])

    def signature_outcome(self, function):
        try:
            return "return", str(inspect.signature(function))
        except BaseException as error:
            return "error", type(error).__name__, str(error)

    def test_private_accessor_metadata_exports_errors_and_pickle_match(self):
        for name in (
            "_get_cudnn_allow_tf32",
            "_set_cudnn_allow_tf32",
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
                self.assertEqual(
                    actual.__text_signature__,
                    expected.__text_signature__,
                )
                self.assertEqual(
                    hasattr(actual, "__annotations__"),
                    hasattr(expected, "__annotations__"),
                )
                self.assertEqual(repr(actual), repr(expected))
                self.assertIs(actual.__self__, torch._C)
                self.assertIs(expected.__self__, reference_torch._C)
                self.assertEqual(actual.__reduce__(), expected.__reduce__())
                self.assertEqual(
                    self.normalize(self.signature_outcome(actual)),
                    self.signature_outcome(expected),
                )
                for function in (actual, expected):
                    self.assertIs(copy.copy(function), function)
                    self.assertIs(copy.deepcopy(function), function)
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

        for root in (torch, reference_torch):
            with self.subTest(package=root.__name__):
                self.assertTrue(hasattr(root._C, "_get_cudnn_allow_tf32"))
                self.assertTrue(hasattr(root._C, "_set_cudnn_allow_tf32"))
                self.assertFalse(hasattr(root, "_get_cudnn_allow_tf32"))
                self.assertFalse(hasattr(root, "_set_cudnn_allow_tf32"))

        self.assertNotIn("_get_cudnn_allow_tf32", torch._C.__all__)
        self.assertNotIn("_set_cudnn_allow_tf32", torch._C.__all__)
        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs._C import *", actual_wildcard)
        exec("from torch._C import *", expected_wildcard)
        for name in (
            "_get_cudnn_allow_tf32",
            "_set_cudnn_allow_tf32",
        ):
            self.assertNotIn(name, actual_wildcard)
            self.assertNotIn(name, expected_wildcard)

        binding_calls = (
            lambda root: root._C._get_cudnn_allow_tf32(None),
            lambda root: root._C._get_cudnn_allow_tf32(value=None),
            lambda root: root._C._set_cudnn_allow_tf32(),
            lambda root: root._C._set_cudnn_allow_tf32(True, False),
            lambda root: root._C._set_cudnn_allow_tf32(object=False),
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


if __name__ == "__main__":
    unittest.main()
