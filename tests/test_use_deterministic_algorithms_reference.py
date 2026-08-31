import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
import threading
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _DefaultInt(int):
    pass


class _NonzeroIntReportsZero(int):
    def __new__(cls):
        return int.__new__(cls, 1)

    def __eq__(self, other):
        return other == 0

    def __ne__(self, other):
        return False


class _CustomMode:
    pass


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class UseDeterministicAlgorithmsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "use_deterministic_algorithms differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.reference_original_enabled = (
            reference_torch.are_deterministic_algorithms_enabled()
        )
        self.reference_original_warn_only = (
            reference_torch.is_deterministic_algorithms_warn_only_enabled()
        )
        torch.use_deterministic_algorithms(False)
        reference_torch.use_deterministic_algorithms(False, warn_only=False)

    def tearDown(self):
        torch.use_deterministic_algorithms(False)
        reference_torch.use_deterministic_algorithms(
            self.reference_original_enabled,
            warn_only=self.reference_original_warn_only,
        )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def state(self, module):
        debug_mode = module.get_deterministic_debug_mode()
        enabled = module.are_deterministic_algorithms_enabled()
        warn_only = module.is_deterministic_algorithms_warn_only_enabled()
        return (
            type(debug_mode) is int,
            debug_mode,
            enabled is False,
            warn_only is False,
            module.is_grad_enabled(),
        )

    def disabled_false_outcome(self, module, *, keyword=False):
        before = module.is_grad_enabled()
        if keyword:
            result = module.use_deterministic_algorithms(mode=False)
        else:
            result = module.use_deterministic_algorithms(False, warn_only=False)
        return result is None, before, self.state(module)

    def threaded_disabled_false_outcomes(self, module):
        worker_count = 10
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = self.disabled_false_outcome(
                        module,
                        keyword=bool(index % 3),
                    )
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
        return results

    def test_disabled_false_grad_modes_and_threads_match_pytorch_2_13(self):
        for keyword in (False, True):
            with self.subTest(keyword=keyword):
                self.assertEqual(
                    self.disabled_false_outcome(torch, keyword=keyword),
                    self.disabled_false_outcome(reference_torch, keyword=keyword),
                )

        with torch.no_grad(), reference_torch.no_grad():
            self.assertEqual(
                self.disabled_false_outcome(torch),
                self.disabled_false_outcome(reference_torch),
            )

        self.assertEqual(
            self.threaded_disabled_false_outcomes(torch),
            self.threaded_disabled_false_outcomes(reference_torch),
        )

    def test_callable_metadata_matches_pytorch_2_13_boundary(self):
        actual_module = importlib.import_module("torch_rs")
        expected_module = importlib.import_module("torch")
        actual = actual_module.use_deterministic_algorithms
        expected = expected_module.use_deterministic_algorithms

        self.assertIs(torch, actual_module)
        self.assertIs(reference_torch, expected_module)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertIs(inspect.getmodule(expected), expected_module)
        self.assertIn("only the disabled deterministic policy", actual.__doc__)
        self.assertIn("fill_uninitialized_memory", expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual = torch.use_deterministic_algorithms
        expected = reference_torch.use_deterministic_algorithms

        self.assertEqual(
            torch.__all__.count("use_deterministic_algorithms"),
            reference_torch.__all__.count("use_deterministic_algorithms"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["use_deterministic_algorithms"], function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

        self.assertFalse(hasattr(torch._C, "_set_deterministic_algorithms"))
        self.assertTrue(
            hasattr(reference_torch._C, "_set_deterministic_algorithms")
        )

    def test_argument_binding_errors_match_pytorch_2_13(self):
        actual = torch.use_deterministic_algorithms
        expected = reference_torch.use_deterministic_algorithms
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(False, False), lambda: expected(False, False)),
            (lambda: actual(enabled=False), lambda: expected(enabled=False)),
            (
                lambda: actual(False, mode=False),
                lambda: expected(False, mode=False),
            ),
            (lambda: actual(False, warn=False), lambda: expected(False, warn=False)),
            (
                lambda: actual(False, warn_only=False, extra=1),
                lambda: expected(False, warn_only=False, extra=1),
            ),
            (
                lambda: actual(False, fill_uninitialized_memory=False),
                lambda: expected(False, fill_uninitialized_memory=False),
            ),
            (
                lambda: actual(False, False, warn_only=False),
                lambda: expected(False, False, warn_only=False),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))
                self.assertEqual(
                    self.state(reference_torch)[:4],
                    (True, 0, True, True),
                )

    def test_invalid_type_errors_match_pytorch_2_13_where_not_extended(self):
        for value in (None, 0.0, b"", [], object(), _CustomMode(), "default"):
            with self.subTest(mode=ascii(value)):
                self.assert_error_matches(
                    lambda value=value: torch.use_deterministic_algorithms(value),
                    lambda value=value: reference_torch.use_deterministic_algorithms(
                        value
                    ),
                )
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))
                self.assertEqual(
                    self.state(reference_torch)[:4],
                    (True, 0, True, True),
                )

        for warn_only in (0, _DefaultInt(0), None, 0.0, [], object(), _CustomMode()):
            with self.subTest(warn_only=ascii(warn_only)):
                self.assert_error_matches(
                    lambda warn_only=warn_only: torch.use_deterministic_algorithms(
                        False,
                        warn_only=warn_only,
                    ),
                    lambda warn_only=warn_only: (
                        reference_torch.use_deterministic_algorithms(
                            False,
                            warn_only=warn_only,
                        )
                    ),
                )
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))
                self.assertEqual(
                    self.state(reference_torch)[:4],
                    (True, 0, True, True),
                )

    def test_integer_zero_is_supported_extension_and_enabling_is_rejected(self):
        self.assertIs(torch.use_deterministic_algorithms(0), None)
        self.assertIs(torch.use_deterministic_algorithms(_DefaultInt(0)), None)
        self.assertEqual(self.state(torch)[:4], (True, 0, True, True))

        for value in (0, _DefaultInt(0)):
            with self.subTest(reference_value=value):
                with self.assertRaises(TypeError):
                    reference_torch.use_deterministic_algorithms(value)
                self.assertEqual(
                    self.state(reference_torch)[:4],
                    (True, 0, True, True),
                )

        for mode, expected_state in (
            (True, (2, True, False)),
            (False, (0, False, False)),
        ):
            with self.subTest(reference_mode=mode):
                self.assertIsNone(reference_torch.use_deterministic_algorithms(mode))
                self.assertEqual(
                    (
                        reference_torch.get_deterministic_debug_mode(),
                        reference_torch.are_deterministic_algorithms_enabled(),
                        reference_torch.is_deterministic_algorithms_warn_only_enabled(),
                    ),
                    expected_state,
                )
                reference_torch.use_deterministic_algorithms(False)

        for mode in (True, 1, _DefaultInt(1), _NonzeroIntReportsZero(), 2, -1):
            with self.subTest(actual_mode=mode):
                with self.assertRaises(NotImplementedError):
                    torch.use_deterministic_algorithms(mode)
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))

        with self.assertRaises(NotImplementedError):
            torch.use_deterministic_algorithms(False, warn_only=True)
        self.assertEqual(self.state(torch)[:4], (True, 0, True, True))

        self.assertIsNone(
            reference_torch.use_deterministic_algorithms(False, warn_only=True)
        )
        self.assertEqual(
            (
                reference_torch.get_deterministic_debug_mode(),
                reference_torch.are_deterministic_algorithms_enabled(),
                reference_torch.is_deterministic_algorithms_warn_only_enabled(),
            ),
            (0, False, True),
        )

    def test_fill_uninitialized_memory_and_cuda_surfaces_remain_boundaries(self):
        self.assertFalse(hasattr(torch.utils, "deterministic"))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.utils.deterministic")

        import torch.utils.deterministic as expected_deterministic

        self.assertTrue(hasattr(expected_deterministic, "fill_uninitialized_memory"))
        self.assertFalse(hasattr(torch._C, "_set_deterministic_algorithms"))
        self.assertTrue(
            hasattr(reference_torch._C, "_set_deterministic_algorithms")
        )

        original_cudnn = torch.backends.cudnn.deterministic
        original_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = True
            self.assertIs(torch.use_deterministic_algorithms(False), None)
            self.assertIs(torch.backends.cudnn.deterministic, True)
            self.assertIs(torch.backends.cuda.matmul.allow_tf32, True)
        finally:
            torch.backends.cudnn.deterministic = original_cudnn
            torch.backends.cuda.matmul.allow_tf32 = original_tf32


if __name__ == "__main__":
    unittest.main()
