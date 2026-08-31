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

    def false_outcome(self, module, *, keyword_mode=False, keyword_warn_only=False):
        before = module.is_grad_enabled()
        if keyword_mode and keyword_warn_only:
            result = module.use_deterministic_algorithms(
                mode=False,
                warn_only=False,
            )
        elif keyword_mode:
            result = module.use_deterministic_algorithms(mode=False)
        elif keyword_warn_only:
            result = module.use_deterministic_algorithms(False, warn_only=False)
        else:
            result = module.use_deterministic_algorithms(False)
        return result is None, before, self.state(module)

    def threaded_false_outcomes(self, module):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = self.false_outcome(
                        module,
                        keyword_warn_only=index % 3 == 0,
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

    def test_false_forms_grad_modes_and_threads_match_pytorch_2_13(self):
        for keyword_mode, keyword_warn_only in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                keyword_mode=keyword_mode,
                keyword_warn_only=keyword_warn_only,
            ):
                self.assertEqual(
                    self.false_outcome(
                        torch,
                        keyword_mode=keyword_mode,
                        keyword_warn_only=keyword_warn_only,
                    ),
                    self.false_outcome(
                        reference_torch,
                        keyword_mode=keyword_mode,
                        keyword_warn_only=keyword_warn_only,
                    ),
                )

        with torch.no_grad(), reference_torch.no_grad():
            self.assertEqual(
                self.false_outcome(torch, keyword_warn_only=True),
                self.false_outcome(reference_torch, keyword_warn_only=True),
            )

        self.assertEqual(
            self.threaded_false_outcomes(torch),
            self.threaded_false_outcomes(reference_torch),
        )

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs")
        expected_module = importlib.import_module("torch")
        actual = actual_module.use_deterministic_algorithms
        expected = expected_module.use_deterministic_algorithms

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
        headline = 'Sets whether PyTorch operations must use "deterministic"'
        self.assertTrue(actual.__doc__.startswith(headline))
        self.assertTrue(expected.__doc__.startswith(headline))
        self.assertIn(
            "only supports requests that leave deterministic",
            actual.__doc__,
        )
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
            direct_namespace = {}
            exec(
                f"from {module.__name__} import use_deterministic_algorithms",
                direct_namespace,
            )
            self.assertIs(direct_namespace["use_deterministic_algorithms"], function)
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
            (lambda: actual(warn_only=False), lambda: expected(warn_only=False)),
            (lambda: actual(False, False), lambda: expected(False, False)),
            (
                lambda: actual(False, warn_only=False, extra=False),
                lambda: expected(False, warn_only=False, extra=False),
            ),
            (lambda: actual(False, mode=False), lambda: expected(False, mode=False)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))
                self.assertEqual(
                    self.state(reference_torch)[:4],
                    (True, 0, True, True),
                )

    def test_invalid_mode_and_warn_only_type_errors_match_pytorch_2_13(self):
        shared_modes = (
            None,
            0.0,
            b"",
            bytearray(b""),
            memoryview(b""),
            [],
            object(),
            _CustomMode(),
            "default",
            "",
        )
        for value in shared_modes:
            with self.subTest(mode=ascii(value)):
                self.assert_error_matches(
                    lambda value=value: torch.use_deterministic_algorithms(value),
                    lambda value=value: (
                        reference_torch.use_deterministic_algorithms(value)
                    ),
                )
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))
                self.assertEqual(
                    self.state(reference_torch)[:4],
                    (True, 0, True, True),
                )

        native_modes = (
            (torch.tensor(0.0), reference_torch.tensor(0.0)),
            (torch.float32, reference_torch.float32),
            (torch.device("cpu"), reference_torch.device("cpu")),
            (torch.contiguous_format, reference_torch.contiguous_format),
            (torch.strided, reference_torch.strided),
            (torch.Size([]), reference_torch.Size([])),
            (
                torch.finfo(torch.float32),
                reference_torch.finfo(reference_torch.float32),
            ),
        )
        for case, (actual_value, expected_value) in enumerate(native_modes):
            with self.subTest(native_mode=case):
                self.assert_error_matches(
                    lambda value=actual_value: torch.use_deterministic_algorithms(
                        value
                    ),
                    lambda value=expected_value: (
                        reference_torch.use_deterministic_algorithms(value)
                    ),
                )
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))
                self.assertEqual(
                    self.state(reference_torch)[:4],
                    (True, 0, True, True),
                )

        shared_warn_only_values = (0, 1, None, "", object(), _CustomMode())
        for value in shared_warn_only_values:
            with self.subTest(warn_only=ascii(value)):
                self.assert_error_matches(
                    lambda value=value: torch.use_deterministic_algorithms(
                        False,
                        warn_only=value,
                    ),
                    lambda value=value: reference_torch.use_deterministic_algorithms(
                        False,
                        warn_only=value,
                    ),
                )
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))
                self.assertEqual(
                    self.state(reference_torch)[:4],
                    (True, 0, True, True),
                )

    def test_disabled_integer_zero_is_an_explicit_local_extension(self):
        self.assertIs(torch.use_deterministic_algorithms(0), None)
        self.assertEqual(self.state(torch)[:4], (True, 0, True, True))

        with self.assertRaises(TypeError) as raised:
            reference_torch.use_deterministic_algorithms(0)
        self.assertEqual(
            str(raised.exception),
            "_set_deterministic_algorithms(): argument 'mode' (position 1) "
            "must be bool, not int",
        )
        self.assertEqual(
            self.state(reference_torch)[:4],
            (True, 0, True, True),
        )

    def test_true_modes_and_warn_only_remain_explicit_differences(self):
        with self.assertRaises(NotImplementedError):
            torch.use_deterministic_algorithms(True)
        self.assertEqual(self.state(torch)[:4], (True, 0, True, True))

        self.assertIsNone(reference_torch.use_deterministic_algorithms(True))
        self.assertIs(reference_torch.are_deterministic_algorithms_enabled(), True)
        self.assertIs(
            reference_torch.is_deterministic_algorithms_warn_only_enabled(),
            False,
        )
        reference_torch.use_deterministic_algorithms(False, warn_only=False)

        with self.assertRaises(NotImplementedError):
            torch.use_deterministic_algorithms(False, warn_only=True)
        self.assertEqual(self.state(torch)[:4], (True, 0, True, True))

        self.assertIsNone(
            reference_torch.use_deterministic_algorithms(
                False,
                warn_only=True,
            )
        )
        self.assertIs(reference_torch.are_deterministic_algorithms_enabled(), False)
        self.assertIs(
            reference_torch.is_deterministic_algorithms_warn_only_enabled(),
            True,
        )
        reference_torch.use_deterministic_algorithms(False, warn_only=False)


if __name__ == "__main__":
    unittest.main()
