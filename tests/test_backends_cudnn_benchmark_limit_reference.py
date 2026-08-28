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


class _IntSubclass(int):
    pass


class _RejectIndex:
    def __index__(self):
        raise AssertionError("cudnn.benchmark_limit must reject __index__ providers")


class _RejectNumpyIndex(np.int64):
    def __index__(self):
        raise RuntimeError("numpy index failed")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudnnBenchmarkLimitReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cudnn.benchmark_limit differentials require pinned "
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
        self.set_states(self.actual, (10, True, False, False, True))
        self.set_states(self.expected, (10, True, False, False, True))

    def tearDown(self):
        actual = self.fresh_cudnn_module(torch)
        expected = self.fresh_cudnn_module(reference_torch)
        self.set_states(actual, self.actual_original)
        self.set_states(expected, self.expected_original)

    def states(self, module):
        return (
            module.benchmark_limit,
            module.enabled,
            module.benchmark,
            module.deterministic,
            module.allow_tf32,
        )

    def set_states(self, module, states):
        (
            module.benchmark_limit,
            module.enabled,
            module.benchmark,
            module.deterministic,
            module.allow_tf32,
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

    def test_narrowing_validation_and_independence_match_pytorch_2_13(self):
        cases = (
            (0, 0),
            (-1, -1),
            (2**31 - 1, 2**31 - 1),
            (-(2**31), -(2**31)),
            (2**31, -(2**31)),
            (-(2**31) - 1, 2**31 - 1),
            (2**32 + 17, 17),
            (-(2**32) + 17, 17),
            (2**63 - 1, -1),
            (-(2**63), 0),
            (_IntSubclass(23), 23),
            (np.int8(-7), -7),
            (np.int64(29), 29),
            (np.uint64(2**63 - 1), -1),
        )
        self.set_states(self.actual, (10, False, True, True, False))
        self.set_states(self.expected, (10, False, True, True, False))
        for value, narrowed in cases:
            with self.subTest(value=value):
                self.actual.benchmark_limit = value
                self.expected.benchmark_limit = value
                self.assertEqual(self.states(self.actual), self.states(self.expected))
                self.assertEqual(self.actual.benchmark_limit, narrowed)
                self.assertIs(type(self.actual.benchmark_limit), int)
                self.assertIs(
                    torch._C._cuda_set_cudnn_benchmark_limit(value),
                    reference_torch._C._cuda_set_cudnn_benchmark_limit(value),
                )
                self.assertEqual(
                    torch._C._cuda_get_cudnn_benchmark_limit(),
                    reference_torch._C._cuda_get_cudnn_benchmark_limit(),
                )
                self.assertEqual(self.states(self.actual), self.states(self.expected))

        invalid_values = (
            True,
            False,
            np.bool_(True),
            None,
            1.0,
            "1",
            [],
            object(),
            _RejectIndex(),
        )
        for value in invalid_values:
            with self.subTest(kind="invalid", value_type=type(value).__name__):
                states = (123, False, True, True, False)
                self.set_states(self.actual, states)
                self.set_states(self.expected, states)
                self.assert_error_matches(
                    lambda value=value: setattr(
                        self.actual,
                        "benchmark_limit",
                        value,
                    ),
                    lambda value=value: setattr(
                        self.expected,
                        "benchmark_limit",
                        value,
                    ),
                )
                self.assertEqual(self.states(self.actual), states)
                self.assertEqual(self.states(self.expected), states)

        for actual_value, expected_value in (
            (torch.tensor(1), reference_torch.tensor(1)),
            (torch.float32, reference_torch.float32),
            (torch.device("cpu"), reference_torch.device("cpu")),
            (torch.strided, reference_torch.strided),
            (torch.Size([1]), reference_torch.Size([1])),
            (torch.finfo(torch.float32), reference_torch.finfo(reference_torch.float32)),
        ):
            with self.subTest(kind="native", value_type=type(actual_value).__name__):
                self.assert_error_matches(
                    lambda: setattr(
                        self.actual,
                        "benchmark_limit",
                        actual_value,
                    ),
                    lambda: setattr(
                        self.expected,
                        "benchmark_limit",
                        expected_value,
                    ),
                )
                self.assertEqual(self.states(self.actual), states)
                self.assertEqual(self.states(self.expected), states)

        for value in (
            2**63,
            -(2**63) - 1,
            _IntSubclass(2**63),
            np.uint64(2**63),
            np.uint64(2**64 - 1),
        ):
            with self.subTest(kind="overflow", value=value):
                self.assert_error_matches(
                    lambda value=value: setattr(
                        self.actual,
                        "benchmark_limit",
                        value,
                    ),
                    lambda value=value: setattr(
                        self.expected,
                        "benchmark_limit",
                        value,
                    ),
                )
                self.assertEqual(self.states(self.actual), states)
                self.assertEqual(self.states(self.expected), states)

        value = _RejectNumpyIndex(31)
        self.assert_error_matches(
            lambda: setattr(self.actual, "benchmark_limit", value),
            lambda: setattr(self.expected, "benchmark_limit", value),
        )
        self.assertEqual(self.states(self.actual), states)
        self.assertEqual(self.states(self.expected), states)

    def thread_contract(self, module):
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []
        self.set_states(module, (10, False, True, True, False))

        def worker():
            try:
                observations.append(self.states(module))
                module.benchmark_limit = 2**32 + 11
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(self.states(module))
                module.benchmark_limit = -13
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_changed.wait(timeout=10)
        main_saw_worker = module.benchmark_limit == 11
        module.benchmark_limit = 12
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
        self.set_states(module, (19, False, True, True, False))

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
        reloaded.benchmark_limit = 23
        old_saw_new = self.states(module)
        module.benchmark_limit = -29
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

    def test_proxy_private_accessors_and_deletion_match_pytorch_2_13(self):
        actual_descriptor = vars(type(self.actual))["benchmark_limit"]
        expected_descriptor = vars(type(self.expected))["benchmark_limit"]

        self.assertEqual(set(vars(actual_descriptor)), set(vars(expected_descriptor)))
        self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)
        self.assertIs(
            actual_descriptor.getter,
            torch._C._cuda_get_cudnn_benchmark_limit,
        )
        self.assertIs(
            actual_descriptor.setter,
            torch._C._cuda_set_cudnn_benchmark_limit,
        )
        self.assertIs(
            expected_descriptor.getter,
            reference_torch._C._cuda_get_cudnn_benchmark_limit,
        )
        self.assertIs(
            expected_descriptor.setter,
            reference_torch._C._cuda_set_cudnn_benchmark_limit,
        )
        for module in (self.actual, self.expected):
            self.assertIs(module.m.__annotations__["benchmark_limit"], int)
            self.assertNotIn("benchmark_limit", vars(module))
            self.assertNotIn("benchmark_limit", vars(module.m))
            self.assertNotIn("benchmark_limit", dir(module))

        actual_import = {}
        expected_import = {}
        actual_wildcard = {}
        expected_wildcard = {}
        exec(
            "from torch_rs.backends.cudnn import benchmark_limit",
            actual_import,
        )
        exec(
            "from torch.backends.cudnn import benchmark_limit",
            expected_import,
        )
        exec("from torch_rs.backends.cudnn import *", actual_wildcard)
        exec("from torch.backends.cudnn import *", expected_wildcard)
        self.assertEqual(
            actual_import["benchmark_limit"],
            expected_import["benchmark_limit"],
        )
        self.assertEqual(
            "benchmark_limit" in actual_wildcard,
            "benchmark_limit" in expected_wildcard,
        )

        self.actual.benchmark_limit = 17
        self.expected.benchmark_limit = 17
        self.assert_error_matches(
            lambda: delattr(self.actual, "benchmark_limit"),
            lambda: delattr(self.expected, "benchmark_limit"),
        )
        self.assertEqual(self.actual.benchmark_limit, 17)
        self.assertEqual(self.expected.benchmark_limit, 17)

        for name in (
            "_cuda_get_cudnn_benchmark_limit",
            "_cuda_set_cudnn_benchmark_limit",
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
                    self.normalize(self.signature_outcome(actual)),
                    self.signature_outcome(expected),
                )

        binding_calls = (
            lambda root: root._C._cuda_get_cudnn_benchmark_limit(None),
            lambda root: root._C._cuda_get_cudnn_benchmark_limit(value=None),
            lambda root: root._C._cuda_set_cudnn_benchmark_limit(),
            lambda root: root._C._cuda_set_cudnn_benchmark_limit(1, 2),
            lambda root: root._C._cuda_set_cudnn_benchmark_limit(object=1),
        )
        self.actual.benchmark_limit = 41
        self.expected.benchmark_limit = 41
        for case, call in enumerate(binding_calls):
            with self.subTest(kind="binding", case=case):
                self.assert_error_matches(
                    lambda call=call: call(torch),
                    lambda call=call: call(reference_torch),
                )
                self.assertEqual(self.actual.benchmark_limit, 41)
                self.assertEqual(self.expected.benchmark_limit, 41)


if __name__ == "__main__":
    unittest.main()
