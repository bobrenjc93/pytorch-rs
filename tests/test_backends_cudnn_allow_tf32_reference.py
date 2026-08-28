import importlib
import inspect
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
        self.actual_original = self.states(self.actual)
        self.expected_original = self.states(self.expected)
        self.set_states(self.actual, (True, True, False, False))
        self.set_states(self.expected, (True, True, False, False))

    def tearDown(self):
        actual = self.fresh_cudnn_module(torch)
        expected = self.fresh_cudnn_module(reference_torch)
        self.set_states(actual, self.actual_original)
        self.set_states(expected, self.expected_original)

    def states(self, module):
        return (
            module.allow_tf32,
            module.enabled,
            module.benchmark,
            module.deterministic,
        )

    def set_states(self, module, states):
        (
            module.allow_tf32,
            module.enabled,
            module.benchmark,
            module.deterministic,
        ) = states

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

    def test_exact_bool_updates_errors_and_independence_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected
        self.set_states(actual, (True, False, True, True))
        self.set_states(expected, (True, False, True, True))

        for allow_tf32 in (False, True, True, False, False, True):
            with self.subTest(allow_tf32=allow_tf32):
                actual.allow_tf32 = allow_tf32
                expected.allow_tf32 = allow_tf32
                self.assertEqual(self.states(actual), self.states(expected))
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
            states = (state, not state, state, not state)
            self.set_states(actual, states)
            self.set_states(expected, states)
            for value in invalid_values:
                with self.subTest(state=state, value_type=type(value).__name__):
                    self.assert_error_matches(
                        lambda value=value: setattr(actual, "allow_tf32", value),
                        lambda value=value: setattr(expected, "allow_tf32", value),
                    )
                    self.assertEqual(self.states(actual), states)
                    self.assertEqual(self.states(expected), states)

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
                self.assertEqual(self.states(actual), states)
                self.assertEqual(self.states(expected), states)

        self.assert_error_matches(
            lambda: delattr(actual, "allow_tf32"),
            lambda: delattr(expected, "allow_tf32"),
        )
        self.assertEqual(self.states(actual), states)
        self.assertEqual(self.states(expected), states)

    def thread_contract(self, module):
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []
        self.set_states(module, (True, False, True, True))

        def worker():
            try:
                observations.append(self.states(module))
                module.allow_tf32 = False
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(self.states(module))
                module.allow_tf32 = False
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_changed.wait(timeout=10)
        main_saw_worker = module.allow_tf32 is False
        module.allow_tf32 = True
        main_changed.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            main_saw_worker,
            not thread.is_alive(),
            errors,
            observations,
            self.states(module),
        )

    def test_process_global_thread_visibility_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_contract(self.actual),
            self.thread_contract(self.expected),
        )

    def reload_contract(self, root, module):
        parent = root.backends
        namespace = module.__dict__
        self.set_states(module, (False, False, True, True))

        reloaded = importlib.reload(module)
        initial = (
            reloaded is module,
            module.__dict__ is namespace,
            parent.cudnn is module,
            sys.modules[module.__name__] is reloaded,
            reloaded.m is module,
            self.states(module),
            self.states(reloaded),
        )
        reloaded.allow_tf32 = True
        old_saw_new = self.states(module)
        module.allow_tf32 = False
        new_saw_old = self.states(reloaded)
        fresh = self.fresh_cudnn_module(root)
        fresh_state = self.states(fresh)
        return initial, old_saw_new, new_saw_old, fresh_state

    def test_reload_and_fresh_import_state_match_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def signature_outcome(self, function):
        try:
            return "return", str(inspect.signature(function))
        except BaseException as error:
            return "error", type(error).__name__, str(error)

    def test_proxy_and_private_accessor_contract_match_pytorch_2_13(self):
        actual_descriptor = vars(type(self.actual))["allow_tf32"]
        expected_descriptor = vars(type(self.expected))["allow_tf32"]

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
        for module in (self.actual, self.expected):
            self.assertIs(module.m.__annotations__["allow_tf32"], bool)
            self.assertNotIn("allow_tf32", vars(module))
            self.assertNotIn("allow_tf32", vars(module.m))
            self.assertNotIn("allow_tf32", dir(module))

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
        self.assertIs(actual_import["allow_tf32"], expected_import["allow_tf32"])
        self.assertEqual(
            "allow_tf32" in actual_wildcard,
            "allow_tf32" in expected_wildcard,
        )

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
                    self.normalize(self.signature_outcome(actual)),
                    self.signature_outcome(expected),
                )

        self.assertIs(
            torch._C._set_cudnn_allow_tf32(False),
            reference_torch._C._set_cudnn_allow_tf32(False),
        )
        self.assertIs(
            torch._C._get_cudnn_allow_tf32(),
            reference_torch._C._get_cudnn_allow_tf32(),
        )

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
