"""Independent output partitioning must preserve serial numerical ordering."""
import concurrent.futures
import unittest

import torch_rs as native

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


class ThreadBudgetTests(unittest.TestCase):
    def tearDown(self):
        native.set_num_threads(1)

    def test_configurable_shared_budget_and_interop_independence(self):
        for count in (1, 2, 4, 8, 2, 1):
            self.assertIsNone(native.set_num_threads(count))
            self.assertEqual(native.get_num_threads(), count)
            with concurrent.futures.ThreadPoolExecutor(2) as workers:
                self.assertEqual(list(workers.map(lambda _: native.get_num_threads(), range(2))), [count] * 2)
            self.assertEqual(native.get_num_interop_threads(), 1)
        self.assertIn('set_num_threads', native.__all__)
        self.assertIs(native.set_num_threads, native._C.set_num_threads)

    def test_invalid_budget_does_not_change_pool(self):
        native.set_num_threads(2)
        for value in (0, -1, True, False, 2.5, '4', None):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                native.set_num_threads(value)
            self.assertEqual(native.get_num_threads(), 2)
        for args, kwargs in [((), {}), ((1, 2), {}), ((), {'threads': 2})]:
            with self.assertRaises(TypeError):
                native.set_num_threads(*args, **kwargs)
            self.assertEqual(native.get_num_threads(), 2)


@unittest.skipIf(torch is None, 'install the reference dependency group')
class ParallelReductionTests(unittest.TestCase):
    def tearDown(self):
        native.set_num_threads(1)

    def test_axes_transposes_offsets_and_cancellation_tails(self):
        rng = np.random.default_rng(302781)
        for shape in ((8193, 35), (8193, 37), (4097, 65), (2049, 137), (513, 1025)):
            values = rng.choice(np.array([1e8, -1e8, 1., -2., 3.], dtype=np.float32), size=shape)
            # Offset view remains contiguous before transpose; parent may die.
            x = native.tensor([[99.] * shape[1]] + values.tolist())[1:]
            for transposed in (False, True):
                view = x.t() if transposed else x
                for axis in (0, 1):
                    for operation in ('sum', 'mean'):
                        native.set_num_threads(1)
                        expected = np.array(getattr(view, operation)(axis).tolist(), dtype=np.float32)
                        for threads in (2, 4, 8):
                            native.set_num_threads(threads)
                            actual = np.array(getattr(view, operation)(axis).tolist(), dtype=np.float32)
                            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_nonunit_strides_on_both_axes(self):
        rng = np.random.default_rng(77821)
        values = rng.integers(-16, 16, size=(129, 2051, 2)).astype(np.float32)
        x = native.tensor(values.tolist()).transpose(0, 2)[1]
        reference = torch.tensor(values).transpose(0, 2)[1]
        self.assertTrue(all(stride > 1 for stride in x.stride()))
        for axis in (0, 1):
            native.set_num_threads(1)
            serial = np.array(x.mean(axis).tolist(), dtype=np.float32)
            for count in (2, 4, 8):
                native.set_num_threads(count)
                actual = np.array(x.mean(axis).tolist(), dtype=np.float32)
                np.testing.assert_array_equal(actual.view(np.uint32), serial.view(np.uint32))
                np.testing.assert_allclose(actual, reference.mean(axis).numpy(), rtol=2e-6, atol=2e-7)

    def test_reference_values_keepdim_and_backward(self):
        rng = np.random.default_rng(5804)
        values = rng.integers(-100, 100, size=(513, 1025)).astype(np.float32) / 64
        for threads in (1, 2, 4, 8):
            native.set_num_threads(threads)
            old = torch.get_num_threads()
            torch.set_num_threads(threads)
            try:
                for transposed in (False, True):
                    for axis in (0, 1):
                        x = native.tensor(values.tolist(), requires_grad=True)
                        ref = torch.tensor(values, requires_grad=True)
                        y = (x.t() if transposed else x).mean(axis, keepdim=True)
                        expected = (ref.t() if transposed else ref).mean(axis, keepdim=True)
                        np.testing.assert_allclose(y.tolist(), expected.tolist(), rtol=2e-6, atol=2e-7)
                        self.assertEqual(tuple(y.shape), tuple(expected.shape))
                        y.sum().backward()
                        expected.sum().backward()
                        np.testing.assert_allclose(x.grad.tolist(), ref.grad.tolist(), rtol=1e-7, atol=0)
            finally:
                torch.set_num_threads(old)

    def test_pool_replacement_during_native_work(self):
        # Pool replacement and calls from different Python threads must be safe.
        # Rust also tests replacement while an old pool is actively executing.
        x = native.ones((1025, 1025))
        def reduce_many():
            for _ in range(20):
                self.assertEqual(x.mean(0).tolist(), [1.] * 1025)
        def reconfigure():
            for count in (4, 2, 8, 1) * 5:
                native.set_num_threads(count)
        with concurrent.futures.ThreadPoolExecutor(3) as workers:
            futures = [workers.submit(reduce_many) for _ in range(2)] + [workers.submit(reconfigure)]
            for future in futures:
                future.result(timeout=60)


if __name__ == '__main__':
    unittest.main()
