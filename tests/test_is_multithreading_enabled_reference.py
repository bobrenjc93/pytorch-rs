import copy
import inspect
import pickle
import threading
import types
import unittest

import numpy as np
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
class IsMultithreadingEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_multithreading_enabled differentials require pinned PyTorch 2.13.0"
            )

    def setUp(self):
        self.previous_states = tuple(
            module.autograd.is_multithreading_enabled()
            for module in (torch, reference_torch)
        )
        for module in (torch, reference_torch):
            module._C._set_multithreading_enabled(True)

    def tearDown(self):
        for module, previous in zip(
            (torch, reference_torch), self.previous_states, strict=True
        ):
            module._C._set_multithreading_enabled(previous)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def thread_local_outcome(self, module):
        query = module.autograd.is_multithreading_enabled
        setter = module._C._set_multithreading_enabled
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        setter(False)

        def worker(index):
            try:
                initial = query()
                enabled = index % 2 == 0
                setter(enabled)
                barrier.wait(timeout=10)
                results[index] = (initial, query())
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

        outcome = (
            results,
            sorted(errors),
            [thread.is_alive() for thread in threads],
            query(),
        )
        setter(True)
        return outcome

    def context_outcome(self, module):
        query = module.autograd.is_multithreading_enabled
        context_type = module.autograd.set_multithreading_enabled
        states = [query()]

        outer = context_type(False)
        states.append(query())
        outer_dict = outer.__dict__.copy()
        with outer as outer_entered:
            states.append(query())
            inner = context_type(True)
            states.append(query())
            inner_dict = inner.__dict__.copy()
            with inner as inner_entered:
                states.append(query())
            states.append(query())
        states.append(query())

        error = RuntimeError("restore multithreading state")
        try:
            with context_type(False):
                states.append(query())
                raise error
        except RuntimeError as raised:
            propagated = raised is error
        else:
            propagated = False
        states.append(query())

        return (
            states,
            outer_dict,
            inner_dict,
            outer_entered,
            inner_entered,
            propagated,
        )

    def decorator_outcome(self, module):
        query = module.autograd.is_multithreading_enabled
        setter = module._C._set_multithreading_enabled
        context = module.autograd.set_multithreading_enabled(False)
        immediate = query()

        @context
        def decorated():
            return query()

        @context
        def generate():
            request = yield query()
            yield request, query()

        setter(True)
        decorated_state = decorated()
        after_decorated = query()
        generator = generate()
        first = next(generator)
        after_first = query()
        second = generator.send("resume")
        after_second = query()
        return (
            immediate,
            decorated_state,
            after_decorated,
            first,
            after_first,
            second,
            after_second,
        )

    def backward_outcome(self, module, enabled):
        values = module.tensor([-2.0, 0.5, 3.0], requires_grad=True)
        with module.autograd.set_multithreading_enabled(enabled):
            state = module.autograd.is_multithreading_enabled()
            loss = (values * values).sum()
            requires_grad = loss.requires_grad
            loss.backward()
        return (
            state,
            requires_grad,
            np.asarray(values.grad).copy(),
            module.autograd.is_multithreading_enabled(),
        )

    def test_mutable_thread_local_state_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_local_outcome(torch),
            self.thread_local_outcome(reference_torch),
        )

    def test_native_builtin_contracts_match_pytorch_2_13(self):
        actual_query = torch.autograd.is_multithreading_enabled
        expected_query = reference_torch.autograd.is_multithreading_enabled
        actual_setter = torch._C._set_multithreading_enabled
        expected_setter = reference_torch._C._set_multithreading_enabled

        for actual, expected in (
            (actual_query, expected_query),
            (actual_setter, expected_setter),
        ):
            self.assertIs(type(actual), types.BuiltinFunctionType)
            self.assertIs(type(expected), types.BuiltinFunctionType)
            self.assertEqual(actual.__name__, expected.__name__)
            self.assertEqual(actual.__qualname__, expected.__qualname__)
            self.assertEqual(
                actual.__module__.replace("torch_rs.torch_rs", "torch._C"),
                expected.__module__,
            )
            self.assertEqual(actual.__doc__, expected.__doc__)
            self.assertEqual(actual.__text_signature__, expected.__text_signature__)
            self.assertEqual(
                hasattr(actual, "__annotations__"),
                hasattr(expected, "__annotations__"),
            )
            self.assertEqual(repr(actual), repr(expected))
            self.assertIs(copy.copy(actual), actual)
            self.assertIs(copy.deepcopy(actual), actual)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=actual, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual, protocol=protocol)),
                        actual,
                    )

        self.assertIs(actual_query.__self__, torch._C)
        self.assertIs(actual_setter.__self__, torch._C)
        assert_no_argument_signature(self, actual_query, "()")
        assert_no_argument_signature(self, expected_query, "()")

    def test_setter_errors_match_pytorch_2_13(self):
        actual = torch._C._set_multithreading_enabled
        expected = reference_torch._C._set_multithreading_enabled
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(True, False), lambda: expected(True, False)),
            (
                lambda: actual(True, enabled=False),
                lambda: expected(True, enabled=False),
            ),
            (
                lambda: actual(True, unexpected=False),
                lambda: expected(True, unexpected=False),
            ),
            (lambda: actual(foo=True), lambda: expected(foo=True)),
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(enabled=None), lambda: expected(enabled=None)),
            (
                lambda: actual(enabled=None, unexpected=True),
                lambda: expected(enabled=None, unexpected=True),
            ),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
            (
                lambda: actual(None, unexpected=True),
                lambda: expected(None, unexpected=True),
            ),
            (lambda: actual(1), lambda: expected(1)),
            (lambda: actual(enabled=1), lambda: expected(enabled=1)),
            (lambda: actual(np.bool_(True)), lambda: expected(np.bool_(True))),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(torch.autograd.is_multithreading_enabled(), True)
                self.assertIs(
                    reference_torch.autograd.is_multithreading_enabled(), True
                )

        self.assertIs(actual(enabled=False), None)
        self.assertIs(expected(enabled=False), None)
        self.assertIs(torch.autograd.is_multithreading_enabled(), False)
        self.assertIs(
            reference_torch.autograd.is_multithreading_enabled(), False
        )

    def test_context_alias_metadata_and_behavior_match_pytorch_2_13(self):
        actual = torch.autograd.set_multithreading_enabled
        expected = reference_torch.autograd.set_multithreading_enabled

        self.assertIs(actual, torch.autograd.grad_mode.set_multithreading_enabled)
        self.assertIs(
            expected,
            reference_torch.autograd.grad_mode.set_multithreading_enabled,
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(
            self.context_outcome(torch),
            self.context_outcome(reference_torch),
        )
        self.assertEqual(
            self.decorator_outcome(torch),
            self.decorator_outcome(reference_torch),
        )

    def test_import_and_export_behavior_matches_pytorch_2_13(self):
        for module in (torch, reference_torch):
            self.assertNotIn(
                "is_multithreading_enabled", module.autograd.__all__
            )
            self.assertIn(
                "set_multithreading_enabled", module.autograd.__all__
            )
            self.assertIn(
                "set_multithreading_enabled", module.autograd.grad_mode.__all__
            )

            wildcard_namespace = {}
            exec(f"from {module.autograd.__name__} import *", wildcard_namespace)
            self.assertNotIn("is_multithreading_enabled", wildcard_namespace)
            self.assertIs(
                wildcard_namespace["set_multithreading_enabled"],
                module.autograd.set_multithreading_enabled,
            )

            self.assertFalse(hasattr(module, "is_multithreading_enabled"))
            self.assertFalse(hasattr(module, "set_multithreading_enabled"))
            self.assertNotIn("is_multithreading_enabled", module.__all__)
            self.assertNotIn("set_multithreading_enabled", module.__all__)

        self.assertNotIn("_is_multithreading_enabled", torch._C.__all__)
        self.assertNotIn("_set_multithreading_enabled", torch._C.__all__)

    def test_backward_results_match_in_both_states(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                actual = self.backward_outcome(torch, enabled)
                expected = self.backward_outcome(reference_torch, enabled)
                self.assertEqual(actual[:2], expected[:2])
                np.testing.assert_allclose(
                    actual[2], expected[2], rtol=1.0e-6, atol=1.0e-6
                )
                self.assertEqual(actual[3], expected[3])


if __name__ == "__main__":
    unittest.main()
