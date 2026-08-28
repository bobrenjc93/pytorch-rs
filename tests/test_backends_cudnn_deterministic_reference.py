import copy
import importlib
import inspect
import pickle
import pickletools
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
        raise AssertionError("cudnn.deterministic must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudnnDeterministicReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cudnn.deterministic differentials require pinned "
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
            self.actual.deterministic,
            self.actual.benchmark,
            self.actual.enabled,
        )
        self.expected_original = (
            self.expected.deterministic,
            self.expected.benchmark,
            self.expected.enabled,
        )
        self.actual.deterministic = False
        self.expected.deterministic = False
        self.actual.benchmark = False
        self.expected.benchmark = False
        self.actual.enabled = True
        self.expected.enabled = True

    def tearDown(self):
        actual = self.fresh_cudnn_module(torch)
        expected = self.fresh_cudnn_module(reference_torch)
        (
            actual.deterministic,
            actual.benchmark,
            actual.enabled,
        ) = self.actual_original
        (
            expected.deterministic,
            expected.benchmark,
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

    def test_exact_bool_updates_errors_and_independence_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected

        actual.enabled = False
        expected.enabled = False
        actual.benchmark = True
        expected.benchmark = True
        for deterministic in (True, False, False, True, True, False):
            with self.subTest(deterministic=deterministic):
                actual.deterministic = deterministic
                expected.deterministic = deterministic
                self.assertIs(actual.deterministic, expected.deterministic)
                self.assertIs(
                    type(actual.deterministic),
                    type(expected.deterministic),
                )
                self.assertIs(actual.enabled, expected.enabled)
                self.assertIs(actual.benchmark, expected.benchmark)
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
            actual.deterministic = state
            expected.deterministic = state
            actual.enabled = not state
            expected.enabled = not state
            actual.benchmark = state
            expected.benchmark = state
            for case, (actual_value, expected_value) in enumerate(
                zip(actual_values, expected_values)
            ):
                with self.subTest(state=state, case=case):
                    self.assert_error_matches(
                        lambda value=actual_value: setattr(
                            actual,
                            "deterministic",
                            value,
                        ),
                        lambda value=expected_value: setattr(
                            expected,
                            "deterministic",
                            value,
                        ),
                    )
                    self.assertIs(actual.deterministic, state)
                    self.assertIs(expected.deterministic, state)
                    self.assertIs(actual.enabled, expected.enabled)
                    self.assertIs(actual.benchmark, expected.benchmark)

                    self.assert_error_matches(
                        lambda value=actual_value: (
                            torch._C._set_cudnn_deterministic(value)
                        ),
                        lambda value=expected_value: (
                            reference_torch._C._set_cudnn_deterministic(value)
                        ),
                    )
                    self.assertIs(actual.deterministic, state)
                    self.assertIs(expected.deterministic, state)

    def thread_contract(self, module):
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []
        module.deterministic = False

        def worker():
            try:
                observations.append(module.deterministic)
                module.deterministic = True
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(module.deterministic)
                module.deterministic = True
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_changed.wait(timeout=10)
        main_saw_worker = module.deterministic is True
        module.deterministic = False
        main_changed.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            main_saw_worker,
            not thread.is_alive(),
            errors,
            observations,
            module.deterministic,
        )

    def test_process_global_thread_visibility_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_contract(self.actual),
            self.thread_contract(self.expected),
        )

    def reload_contract(self, root, module):
        parent = root.backends
        namespace = module.__dict__
        module.deterministic = True

        reloaded = importlib.reload(module)
        initial = (
            reloaded is module,
            module.__dict__ is namespace,
            parent.cudnn is module,
            sys.modules[module.__name__] is reloaded,
            reloaded.m is module,
            module.deterministic,
            reloaded.deterministic,
        )
        reloaded.deterministic = False
        old_saw_new = module.deterministic
        module.deterministic = True
        new_saw_old = reloaded.deterministic
        fresh = self.fresh_cudnn_module(root)
        fresh_state = fresh.deterministic
        return initial, old_saw_new, new_saw_old, fresh_state

    def test_reload_and_fresh_import_state_match_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_proxy_metadata_imports_deletion_copying_and_pickling_match(self):
        actual = self.actual
        expected = self.expected
        actual_descriptor = vars(type(actual))["deterministic"]
        expected_descriptor = vars(type(expected))["deterministic"]

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
        self.assertIs(actual.m.__annotations__["deterministic"], bool)
        self.assertIs(expected.m.__annotations__["deterministic"], bool)
        for module in (actual, expected):
            self.assertNotIn("deterministic", vars(module))
            self.assertNotIn("deterministic", vars(module.m))
            self.assertNotIn("deterministic", dir(module))

        self.assertEqual(
            set(vars(actual_descriptor)),
            set(vars(expected_descriptor)),
        )
        self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)
        self.assertIs(
            actual_descriptor.getter,
            torch._C._get_cudnn_deterministic,
        )
        self.assertIs(
            actual_descriptor.setter,
            torch._C._set_cudnn_deterministic,
        )
        self.assertIs(
            expected_descriptor.getter,
            reference_torch._C._get_cudnn_deterministic,
        )
        self.assertIs(
            expected_descriptor.setter,
            reference_torch._C._set_cudnn_deterministic,
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
            "from torch_rs.backends.cudnn import deterministic",
            actual_import,
        )
        exec(
            "from torch.backends.cudnn import deterministic",
            expected_import,
        )
        exec("from torch_rs.backends.cudnn import *", actual_wildcard)
        exec("from torch.backends.cudnn import *", expected_wildcard)
        self.assertIs(
            actual_import["deterministic"],
            expected_import["deterministic"],
        )
        self.assertEqual(
            "deterministic" in actual_wildcard,
            "deterministic" in expected_wildcard,
        )

        actual.deterministic = True
        expected.deterministic = True
        self.assert_error_matches(
            lambda: delattr(actual, "deterministic"),
            lambda: delattr(expected, "deterministic"),
        )
        self.assertIs(actual.deterministic, True)
        self.assertIs(expected.deterministic, True)

        for module in (actual, expected):
            for state in (False, True):
                module.deterministic = state
                self.assertIs(copy.copy(module.deterministic), state)
                self.assertIs(copy.deepcopy(module.deterministic), state)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(
                            pickle.dumps(module.deterministic, protocol=protocol)
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
            "_get_cudnn_deterministic",
            "_set_cudnn_deterministic",
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
                self.assertTrue(hasattr(root._C, "_get_cudnn_deterministic"))
                self.assertTrue(hasattr(root._C, "_set_cudnn_deterministic"))
                self.assertFalse(hasattr(root, "_get_cudnn_deterministic"))
                self.assertFalse(hasattr(root, "_set_cudnn_deterministic"))

        self.assertNotIn("_get_cudnn_deterministic", torch._C.__all__)
        self.assertNotIn("_set_cudnn_deterministic", torch._C.__all__)
        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs._C import *", actual_wildcard)
        exec("from torch._C import *", expected_wildcard)
        for name in (
            "_get_cudnn_deterministic",
            "_set_cudnn_deterministic",
        ):
            self.assertNotIn(name, actual_wildcard)
            self.assertNotIn(name, expected_wildcard)

        binding_calls = (
            lambda root: root._C._get_cudnn_deterministic(None),
            lambda root: root._C._get_cudnn_deterministic(value=None),
            lambda root: root._C._set_cudnn_deterministic(),
            lambda root: root._C._set_cudnn_deterministic(True, False),
            lambda root: root._C._set_cudnn_deterministic(object=False),
        )
        self.actual.deterministic = True
        self.expected.deterministic = True
        for case, call in enumerate(binding_calls):
            with self.subTest(kind="binding", case=case):
                self.assert_error_matches(
                    lambda call=call: call(torch),
                    lambda call=call: call(reference_torch),
                )
                self.assertIs(self.actual.deterministic, True)
                self.assertIs(self.expected.deterministic, True)


if __name__ == "__main__":
    unittest.main()
