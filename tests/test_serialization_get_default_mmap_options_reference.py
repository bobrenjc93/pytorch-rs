import contextlib
import inspect
import mmap
import threading
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SerializationGetDefaultMmapOptionsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "serialization.get_default_mmap_options differentials require "
                "pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def state_outcome(self, module, platform_default):
        function = module.serialization.get_default_mmap_options

        def query_outcome():
            before = module.is_grad_enabled()
            first = function()
            middle = module.is_grad_enabled()
            second = function()
            after = module.is_grad_enabled()
            return (
                before,
                first is platform_default,
                middle,
                second is platform_default,
                after,
            )

        states = [query_outcome()]
        with module.no_grad():
            states.append(query_outcome())
            with module.no_grad():
                states.append(query_outcome())
            states.append(query_outcome())
        states.append(query_outcome())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = query_outcome()
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
        return states, worker_states

    def test_platform_default_threaded_and_grad_states_match_pytorch_2_13(self):
        platform_default = getattr(mmap, "MAP_PRIVATE", None)
        self.assertEqual(
            self.state_outcome(torch, platform_default),
            self.state_outcome(reference_torch, platform_default),
        )
        self.assertIs(
            torch.serialization.get_default_mmap_options(), platform_default
        )
        self.assertIs(
            reference_torch.serialization.get_default_mmap_options(),
            platform_default,
        )

    def test_reference_only_map_shared_mutation_bounds_unsupported_state(self):
        actual_serialization = torch.serialization
        expected_serialization = reference_torch.serialization
        platform_default = getattr(mmap, "MAP_PRIVATE", None)
        shared = getattr(mmap, "MAP_SHARED", None)

        self.assertFalse(hasattr(actual_serialization, "set_default_mmap_options"))
        self.assertIs(actual_serialization.get_default_mmap_options(), platform_default)
        self.assertIs(
            expected_serialization.get_default_mmap_options(), platform_default
        )

        if platform_default is None or shared is None:
            self.assertIsNone(actual_serialization.get_default_mmap_options())
            self.assertIsNone(expected_serialization.get_default_mmap_options())
            return

        actual_states = [actual_serialization.get_default_mmap_options()]
        expected_states = [expected_serialization.get_default_mmap_options()]
        with expected_serialization.set_default_mmap_options(shared):
            actual_states.append(actual_serialization.get_default_mmap_options())
            expected_states.append(expected_serialization.get_default_mmap_options())
        actual_states.append(actual_serialization.get_default_mmap_options())
        expected_states.append(expected_serialization.get_default_mmap_options())

        self.assertEqual(actual_states, [platform_default] * 3)
        self.assertEqual(
            expected_states,
            [platform_default, shared, platform_default],
        )

    def test_signature_annotations_and_documentation_match_pytorch_2_13(self):
        actual = torch.serialization.get_default_mmap_options
        expected = reference_torch.serialization.get_default_mmap_options

        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.serialization.get_default_mmap_options
        expected = reference_torch.serialization.get_default_mmap_options
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_setter_save_and_load_remain_deliberately_unsupported(self):
        actual_serialization = torch.serialization
        expected_serialization = reference_torch.serialization
        for name in ("set_default_mmap_options", "save", "load"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(expected_serialization, name))
                self.assertFalse(hasattr(actual_serialization, name))
                self.assertNotIn(name, actual_serialization.__all__)


if __name__ == "__main__":
    unittest.main()
