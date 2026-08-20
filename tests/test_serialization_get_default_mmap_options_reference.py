import contextlib
import mmap
import threading
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


def platform_default_mmap_options():
    return getattr(mmap, "MAP_PRIVATE", None)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SerializationGetDefaultMmapOptionsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "serialization mmap option differentials require pinned "
                "PyTorch 2.13.0"
            )

    def threaded_outcome(self, module):
        function = module.serialization.get_default_mmap_options

        def query():
            before = module.is_grad_enabled()
            first = function()
            middle = module.is_grad_enabled()
            second = function()
            after = module.is_grad_enabled()
            return (
                before,
                first,
                type(first).__module__,
                type(first).__qualname__,
                middle,
                second,
                type(second).__module__,
                type(second).__qualname__,
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

    def test_platform_default_threading_and_grad_modes_match_pytorch_2_13(self):
        expected = platform_default_mmap_options()
        if expected is None:
            reference_default = contextlib.nullcontext()
        else:
            reference_default = (
                reference_torch.serialization.set_default_mmap_options(expected)
            )

        with reference_default:
            self.assertEqual(
                self.threaded_outcome(torch),
                self.threaded_outcome(reference_torch),
            )
            actual = torch.serialization.get_default_mmap_options()
            reference = reference_torch.serialization.get_default_mmap_options()
            self.assertEqual(actual, expected)
            self.assertEqual(reference, expected)
            self.assertIs(type(actual), type(expected))
            self.assertIs(type(reference), type(expected))

    def test_reference_map_shared_mutation_is_outside_the_supported_scope(self):
        if not hasattr(mmap, "MAP_PRIVATE") or not hasattr(mmap, "MAP_SHARED"):
            self.skipTest("mmap flag mutation is unavailable on this platform")

        actual_getter = torch.serialization.get_default_mmap_options
        reference_getter = reference_torch.serialization.get_default_mmap_options
        original_reference = reference_getter()
        self.assertFalse(
            hasattr(torch.serialization, "set_default_mmap_options")
        )

        with reference_torch.serialization.set_default_mmap_options(
            mmap.MAP_PRIVATE
        ):
            self.assertEqual(actual_getter(), mmap.MAP_PRIVATE)
            self.assertEqual(reference_getter(), mmap.MAP_PRIVATE)
            with reference_torch.serialization.set_default_mmap_options(
                mmap.MAP_SHARED
            ):
                self.assertEqual(reference_getter(), mmap.MAP_SHARED)
                self.assertEqual(actual_getter(), mmap.MAP_PRIVATE)
            self.assertEqual(reference_getter(), mmap.MAP_PRIVATE)
            self.assertEqual(actual_getter(), mmap.MAP_PRIVATE)

        self.assertEqual(reference_getter(), original_reference)
        self.assertEqual(actual_getter(), mmap.MAP_PRIVATE)


if __name__ == "__main__":
    unittest.main()
