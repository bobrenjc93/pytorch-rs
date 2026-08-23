import contextlib
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
class GetNumThreadsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "get_num_threads differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def threaded_outcome(self, module):
        function = module.get_num_threads

        def query():
            before = module.is_grad_enabled()
            first = function()
            middle = module.is_grad_enabled()
            second = function()
            after = module.is_grad_enabled()
            return (
                before,
                type(first) is int,
                first,
                middle,
                type(second) is int,
                second,
                after,
            )

        main_states = [query()]
        with module.no_grad():
            main_states.append(query())
            with module.no_grad():
                main_states.append(query())
            main_states.append(query())
        main_states.append(query())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = query()
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

    def test_single_thread_grad_and_worker_states_match_pytorch_2_13(self):
        original_reference_threads = reference_torch.get_num_threads()
        try:
            reference_torch.set_num_threads(1)
            self.assertEqual(reference_torch.get_num_threads(), 1)
            self.assertEqual(
                self.threaded_outcome(torch),
                self.threaded_outcome(reference_torch),
            )
            self.assertEqual(torch.get_num_threads(), 1)
        finally:
            reference_torch.set_num_threads(original_reference_threads)

        self.assertEqual(
            reference_torch.get_num_threads(), original_reference_threads
        )
        self.assertEqual(torch.get_num_threads(), 1)

    def test_builtin_contract_matches_pytorch_2_13(self):
        actual = torch.get_num_threads
        expected = reference_torch.get_num_threads

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
        self.assertIs(torch._C.get_num_threads, actual)
        self.assertIs(reference_torch._C.get_num_threads, expected)
        for function in (actual, expected):
            assert_no_argument_signature(self, function, "()")

        self.assertEqual(
            torch.__all__.count("get_num_threads"),
            reference_torch.__all__.count("get_num_threads"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertIs(wildcard_namespace["get_num_threads"], function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(module=module.__name__, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    self.assertIs(restored, function)

    def test_no_argument_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.get_num_threads(None),
                lambda: reference_torch.get_num_threads(None),
            ),
            (
                lambda: torch.get_num_threads(None, None),
                lambda: reference_torch.get_num_threads(None, None),
            ),
            (
                lambda: torch.get_num_threads(threads=None),
                lambda: reference_torch.get_num_threads(threads=None),
            ),
            (
                lambda: torch.get_num_threads(None, threads=None),
                lambda: reference_torch.get_num_threads(None, threads=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_single_worker_setter_matches_exports_and_interop_remains_unsupported(self):
        self.assertTrue(hasattr(torch, "set_num_threads"))
        self.assertTrue(hasattr(torch._C, "set_num_threads"))
        self.assertEqual(
            torch.__all__.count("set_num_threads"),
            reference_torch.__all__.count("set_num_threads"),
        )

        name = "set_num_interop_threads"
        self.assertFalse(hasattr(torch, name))
        self.assertFalse(hasattr(torch._C, name))
        self.assertNotIn(name, torch.__all__)
        self.assertTrue(hasattr(reference_torch, name))
        self.assertTrue(hasattr(reference_torch._C, name))
        self.assertIn(name, reference_torch.__all__)


if __name__ == "__main__":
    unittest.main()
