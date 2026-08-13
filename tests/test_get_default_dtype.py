import contextlib
import inspect
import threading
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = """
get_default_dtype() -> torch.dtype

Get the current default floating point :class:`torch.dtype`.

Example::

    >>> torch.get_default_dtype()  # initial default for floating point is torch.float32
    torch.float32
    >>> torch.set_default_dtype(torch.float64)
    >>> torch.get_default_dtype()  # default is now changed to torch.float64
    torch.float64

"""


class GetDefaultDTypeTests(unittest.TestCase):
    def test_returns_the_canonical_float32_used_by_default_factories(self):
        default_dtype = torch.get_default_dtype()
        self.assertIs(default_dtype, torch.float32)
        self.assertIs(default_dtype, torch.float)
        self.assertIs(torch.get_default_dtype(), default_dtype)

        tensors = (
            torch.tensor(1.25),
            torch.zeros((2, 0, 3)),
            torch.ones((2, 3)),
            torch.eye(3),
            torch.full((2,), 1.25),
        )
        for tensor in tensors:
            with self.subTest(shape=tensor.shape):
                self.assertIs(tensor.dtype, default_dtype)

    def test_identity_is_stable_in_grad_mode_contexts_and_threads(self):
        canonical = torch.float32
        self.assertIs(torch.get_default_dtype(), canonical)
        with torch.no_grad():
            self.assertIs(torch.get_default_dtype(), canonical)
            self.assertIs(torch.ones((1,)).dtype, canonical)
            with torch.no_grad():
                self.assertIs(torch.get_default_dtype(), canonical)
        self.assertIs(torch.get_default_dtype(), canonical)

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.get_default_dtype(),
                        torch.tensor(1.25).dtype,
                        torch.zeros((0,)).dtype,
                        torch.get_default_dtype(),
                    )
            except BaseException as error:
                errors.append(error)

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
        for result in results:
            for value in result:
                self.assertIs(value, canonical)

    def test_callable_metadata_matches_pytorch_2_13(self):
        function = torch.get_default_dtype
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "get_default_dtype")
        self.assertEqual(function.__qualname__, "get_default_dtype")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(function)
        self.assertIn("get_default_dtype", torch.__all__)

    def test_rejects_all_arguments_with_pytorch_2_13_errors(self):
        function = torch.get_default_dtype
        cases = (
            (
                lambda: function(None),
                "torch.get_default_dtype() takes no arguments (1 given)",
            ),
            (
                lambda: function(None, None),
                "torch.get_default_dtype() takes no arguments (2 given)",
            ),
            (
                lambda: function(dtype=None),
                "torch.get_default_dtype() takes no keyword arguments",
            ),
            (
                lambda: function(None, dtype=None),
                "torch.get_default_dtype() takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
