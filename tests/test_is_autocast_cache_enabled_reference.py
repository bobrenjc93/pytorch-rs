import contextlib
import copy
import pickle
import threading
import types
import unittest

import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class IsAutocastCacheEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_autocast_cache_enabled differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def supported_state_outcome(self, module):
        function = module.is_autocast_cache_enabled

        def query_around_cpu_execution():
            before_grad = module.is_grad_enabled()
            first = function()
            values = module.relu(
                module.tensor([[-1.0, 2.0], [3.0, -4.0]])
                + module.ones((2, 2))
            ).tolist()
            second = module._C.is_autocast_cache_enabled()
            after_grad = module.is_grad_enabled()
            return (
                before_grad,
                first is True,
                values,
                second is True,
                after_grad,
            )

        main_states = [query_around_cpu_execution()]
        with module.no_grad():
            main_states.append(query_around_cpu_execution())
            with module.no_grad():
                main_states.append(query_around_cpu_execution())
            main_states.append(query_around_cpu_execution())
        main_states.append(query_around_cpu_execution())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = query_around_cpu_execution()
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
        return main_states, worker_states

    def test_supported_default_grad_thread_and_cpu_states_match_pytorch_2_13(self):
        self.assertIs(torch.is_autocast_cache_enabled(), True)
        self.assertIs(reference_torch.is_autocast_cache_enabled(), True)
        self.assertEqual(
            self.supported_state_outcome(torch),
            self.supported_state_outcome(reference_torch),
        )

    def test_reference_only_setter_bounds_the_unsupported_false_state(self):
        actual = torch.is_autocast_cache_enabled
        expected = reference_torch.is_autocast_cache_enabled
        previous_reference_state = expected()

        self.assertFalse(hasattr(torch, "set_autocast_cache_enabled"))
        self.assertFalse(hasattr(torch._C, "set_autocast_cache_enabled"))
        try:
            self.assertIs(reference_torch.set_autocast_cache_enabled(False), None)
            self.assertIs(actual(), True)
            self.assertIs(expected(), False)
            with torch.no_grad(), reference_torch.no_grad():
                self.assertIs(actual(), True)
                self.assertIs(expected(), False)
                self.assertIs(torch.is_grad_enabled(), False)
                self.assertIs(reference_torch.is_grad_enabled(), False)

            self.assertIs(reference_torch._C.set_autocast_cache_enabled(True), None)
            self.assertIs(actual(), True)
            self.assertIs(expected(), True)
        finally:
            reference_torch.set_autocast_cache_enabled(previous_reference_state)

        self.assertIs(actual(), True)
        self.assertIs(expected(), previous_reference_state)

    def assert_reference_only_autocast_context(self, device_type):
        actual = torch.is_autocast_cache_enabled
        expected = reference_torch.is_autocast_cache_enabled
        previous_reference_state = expected()

        try:
            reference_torch.set_autocast_cache_enabled(True)
            with reference_torch.autocast(
                device_type=device_type,
                enabled=True,
                cache_enabled=False,
            ):
                self.assertIs(actual(), True)
                self.assertIs(expected(), False)
                self.assertIs(reference_torch.is_autocast_enabled(device_type), True)
                with reference_torch.autocast(
                    device_type=device_type,
                    enabled=False,
                    cache_enabled=True,
                ):
                    self.assertIs(actual(), True)
                    self.assertIs(expected(), True)
                    self.assertIs(
                        reference_torch.is_autocast_enabled(device_type), False
                    )
                self.assertIs(actual(), True)
                self.assertIs(expected(), False)
                self.assertIs(reference_torch.is_autocast_enabled(device_type), True)
            self.assertIs(actual(), True)
            self.assertIs(expected(), True)
            self.assertIs(reference_torch.is_autocast_enabled(device_type), False)
        finally:
            reference_torch.set_autocast_cache_enabled(previous_reference_state)

    def test_reference_only_cpu_autocast_context_bounds_unsupported_state(self):
        self.assert_reference_only_autocast_context("cpu")

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "CUDA is unavailable",
    )
    def test_reference_only_cuda_autocast_context_bounds_unsupported_state(self):
        self.assert_reference_only_autocast_context("cuda")

    def test_builtin_contract_matches_pytorch_2_13(self):
        actual = torch.is_autocast_cache_enabled
        expected = reference_torch.is_autocast_cache_enabled

        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs.torch_rs", "torch"),
            expected.__module__,
        )
        self.assertIsNone(actual.__doc__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(
            hasattr(actual, "__annotations__"),
            hasattr(expected, "__annotations__"),
        )
        self.assertFalse(hasattr(actual, "__annotations__"))
        self.assertEqual(repr(actual), repr(expected))
        self.assertIs(actual.__self__, torch._C)
        self.assertIs(expected.__self__, reference_torch._C)
        self.assertIs(torch._C.is_autocast_cache_enabled, actual)
        self.assertIs(reference_torch._C.is_autocast_cache_enabled, expected)
        for function in (actual, expected):
            assert_no_argument_signature(self, function, "()")
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        self.assertEqual(
            torch.__all__.count("is_autocast_cache_enabled"),
            reference_torch.__all__.count("is_autocast_cache_enabled"),
        )
        self.assertEqual(torch._C.__all__.count("is_autocast_cache_enabled"), 1)
        for module, function in ((torch, actual), (reference_torch, expected)):
            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertIs(wildcard_namespace["is_autocast_cache_enabled"], function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(module=module.__name__, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    self.assertIs(restored, function)

        actual_native_namespace = {}
        exec(
            "from torch_rs._C import is_autocast_cache_enabled",
            actual_native_namespace,
        )
        self.assertIs(actual_native_namespace["is_autocast_cache_enabled"], actual)
        expected_native_namespace = {}
        exec(
            "from torch._C import is_autocast_cache_enabled",
            expected_native_namespace,
        )
        self.assertIs(
            expected_native_namespace["is_autocast_cache_enabled"], expected
        )

    def test_no_argument_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.is_autocast_cache_enabled(None),
                lambda: reference_torch.is_autocast_cache_enabled(None),
            ),
            (
                lambda: torch.is_autocast_cache_enabled(None, None),
                lambda: reference_torch.is_autocast_cache_enabled(None, None),
            ),
            (
                lambda: torch.is_autocast_cache_enabled(enabled=True),
                lambda: reference_torch.is_autocast_cache_enabled(enabled=True),
            ),
            (
                lambda: torch.is_autocast_cache_enabled(None, enabled=True),
                lambda: reference_torch.is_autocast_cache_enabled(
                    None, enabled=True
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        self.assertIs(torch.is_autocast_cache_enabled(**{}), True)
        self.assertIs(reference_torch.is_autocast_cache_enabled(**{}), True)

    def test_mutation_and_autocast_surfaces_remain_deliberately_absent(self):
        for owner, reference_owner in (
            (torch, reference_torch),
            (torch._C, reference_torch._C),
        ):
            with self.subTest(owner=owner.__name__):
                self.assertFalse(hasattr(owner, "set_autocast_cache_enabled"))
                self.assertTrue(
                    hasattr(reference_owner, "set_autocast_cache_enabled")
                )

        for name in ("autocast", "is_autocast_enabled"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertTrue(hasattr(reference_torch, name))
                self.assertNotIn(name, torch.__all__)


if __name__ == "__main__":
    unittest.main()
