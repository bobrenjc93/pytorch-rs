import contextlib
import inspect
import pickle
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


class _IndexOne:
    def __index__(self):
        return 1


class _IntOne:
    def __int__(self):
        return 1


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SetNumThreadsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "set_num_threads differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def computation_outcome(self, module):
        grad_enabled = module.is_grad_enabled()
        leaf = module.tensor([1.0, -2.0, 3.0], requires_grad=True)

        first = module.set_num_threads(1)
        output = (leaf * leaf).sum()
        second = module.set_num_threads(1)
        if grad_enabled:
            output.backward()
        third = module.set_num_threads(1)

        return (
            grad_enabled,
            first is None,
            second is None,
            third is None,
            module.is_grad_enabled(),
            type(module.get_num_threads()) is int,
            module.get_num_threads(),
            output.item(),
            output.requires_grad,
            None if leaf.grad is None else leaf.grad.tolist(),
        )

    def threaded_outcome(self, module):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = self.computation_outcome(module)
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

        return (
            results,
            errors,
            tuple(thread.is_alive() for thread in threads),
            module.is_grad_enabled(),
            module.get_num_threads(),
        )

    def test_one_computation_grad_and_thread_states_match_pytorch_2_13(self):
        original_reference_threads = reference_torch.get_num_threads()
        try:
            reference_torch.set_num_threads(1)
            self.assertEqual(
                self.computation_outcome(torch),
                self.computation_outcome(reference_torch),
            )
            with torch.no_grad(), reference_torch.no_grad():
                self.assertEqual(
                    self.computation_outcome(torch),
                    self.computation_outcome(reference_torch),
                )
            self.assertEqual(
                self.threaded_outcome(torch),
                self.threaded_outcome(reference_torch),
            )
        finally:
            reference_torch.set_num_threads(original_reference_threads)

        self.assertIs(torch.get_num_threads(), 1)

    def test_supported_integer_forms_match_pytorch_2_13(self):
        original_reference_threads = reference_torch.get_num_threads()
        try:
            cases = (
                (1, 1),
                (_IntSubclass(1), _IntSubclass(1)),
                (np.int32(1), np.int32(1)),
                (np.int64(1), np.int64(1)),
                (np.uint64(1), np.uint64(1)),
            )
            for actual_value, expected_value in cases:
                with self.subTest(value=repr(expected_value)):
                    actual_result = torch.set_num_threads(actual_value)
                    expected_result = reference_torch.set_num_threads(expected_value)
                    self.assertIs(actual_result, expected_result)
                    self.assertEqual(
                        (type(torch.get_num_threads()) is int, torch.get_num_threads()),
                        (
                            type(reference_torch.get_num_threads()) is int,
                            reference_torch.get_num_threads(),
                        ),
                    )
        finally:
            reference_torch.set_num_threads(original_reference_threads)

    def test_builtin_contract_matches_pytorch_2_13(self):
        actual = torch.set_num_threads
        expected = reference_torch.set_num_threads

        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs.torch_rs", "torch"),
            expected.__module__,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(repr(actual), repr(expected))
        self.assertIs(actual.__self__, torch._C)
        self.assertIs(expected.__self__, reference_torch._C)
        self.assertIs(torch._C.set_num_threads, actual)
        self.assertIs(reference_torch._C.set_num_threads, expected)

        for function in (actual, expected):
            with self.assertRaises(ValueError) as raised:
                inspect.signature(function)
            self.assertEqual(
                str(raised.exception),
                "no signature found for builtin <built-in function set_num_threads>",
            )

        self.assertEqual(
            torch.__all__.count("set_num_threads"),
            reference_torch.__all__.count("set_num_threads"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertIs(wildcard_namespace["set_num_threads"], function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(module=module.__name__, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    self.assertIs(restored, function)

    def test_positional_binding_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.set_num_threads(),
                lambda: reference_torch.set_num_threads(),
            ),
            (
                lambda: torch.set_num_threads(1, 1),
                lambda: reference_torch.set_num_threads(1, 1),
            ),
            (
                lambda: torch.set_num_threads(num_threads=1),
                lambda: reference_torch.set_num_threads(num_threads=1),
            ),
            (
                lambda: torch.set_num_threads(thread_count=1),
                lambda: reference_torch.set_num_threads(thread_count=1),
            ),
            (
                lambda: torch.set_num_threads(1, num_threads=1),
                lambda: reference_torch.set_num_threads(1, num_threads=1),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(torch.get_num_threads(), 1)

    def test_type_nonpositive_and_overflow_errors_match_pytorch_2_13(self):
        cases = (
            (True, True),
            (False, False),
            (None, None),
            (1.0, 1.0),
            ("1", "1"),
            (b"1", b"1"),
            ([], []),
            (object(), object()),
            (_IndexOne(), _IndexOne()),
            (_IntOne(), _IntOne()),
            (np.bool_(True), np.bool_(True)),
            (np.float32(1), np.float32(1)),
            (torch.tensor(1.0), reference_torch.tensor(1.0)),
            (torch.float32, reference_torch.float32),
            (torch.device("cpu"), reference_torch.device("cpu")),
            (0, 0),
            (-1, -1),
            (_IntSubclass(0), _IntSubclass(0)),
            (np.int64(-1), np.int64(-1)),
            (-(2**31), -(2**31)),
            (2**31, 2**31),
            (-(2**31) - 1, -(2**31) - 1),
            (np.int64(2**31), np.int64(2**31)),
            (np.int64(-(2**31) - 1), np.int64(-(2**31) - 1)),
            (np.uint32(2**31), np.uint32(2**31)),
            (2**100, 2**100),
            (-(2**100), -(2**100)),
            (np.uint64(2**63), np.uint64(2**63)),
        )
        original_reference_threads = reference_torch.get_num_threads()
        try:
            reference_torch.set_num_threads(1)
            for case, (actual_value, expected_value) in enumerate(cases):
                with self.subTest(case=case):
                    self.assert_error_matches(
                        lambda value=actual_value: torch.set_num_threads(value),
                        lambda value=expected_value: reference_torch.set_num_threads(
                            value
                        ),
                    )
                    self.assertIs(torch.get_num_threads(), 1)
                    self.assertEqual(reference_torch.get_num_threads(), 1)
        finally:
            reference_torch.set_num_threads(original_reference_threads)

    def test_numpy_integer_classification_ignores_mutable_module_attributes(self):
        original_integer = np.integer
        original_reference_threads = reference_torch.get_num_threads()
        numpy_one = np.int64(1)
        try:
            reference_torch.set_num_threads(1)

            np.integer = int
            self.assertIs(torch.set_num_threads(numpy_one), None)
            self.assertIs(reference_torch.set_num_threads(numpy_one), None)
            self.assertIs(torch.get_num_threads(), 1)
            self.assertEqual(reference_torch.get_num_threads(), 1)

            np.integer = _IndexOne
            self.assert_error_matches(
                lambda: torch.set_num_threads(_IndexOne()),
                lambda: reference_torch.set_num_threads(_IndexOne()),
            )
            self.assertIs(torch.get_num_threads(), 1)
            self.assertEqual(reference_torch.get_num_threads(), 1)
        finally:
            np.integer = original_integer
            reference_torch.set_num_threads(original_reference_threads)

    def test_multiworker_values_are_the_deliberate_single_worker_boundary(self):
        original_reference_threads = reference_torch.get_num_threads()
        try:
            for thread_count in (2, 8):
                with self.subTest(thread_count=thread_count):
                    self.assertIs(reference_torch.set_num_threads(thread_count), None)
                    self.assertEqual(reference_torch.get_num_threads(), thread_count)

                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"^set_num_threads only supports the single-worker value 1$",
                    ):
                        torch.set_num_threads(thread_count)
                    self.assertIs(torch.get_num_threads(), 1)
        finally:
            reference_torch.set_num_threads(original_reference_threads)

    def test_interop_setter_remains_deliberately_unsupported(self):
        self.assertFalse(hasattr(torch, "set_num_interop_threads"))
        self.assertFalse(hasattr(torch._C, "set_num_interop_threads"))
        self.assertNotIn("set_num_interop_threads", torch.__all__)

        self.assertTrue(hasattr(reference_torch, "set_num_interop_threads"))
        self.assertTrue(hasattr(reference_torch._C, "set_num_interop_threads"))
        self.assertIn("set_num_interop_threads", reference_torch.__all__)


if __name__ == "__main__":
    unittest.main()
