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

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("the setter must not request truthiness")


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
        self.reference_original = (
            reference_torch.are_deterministic_algorithms_enabled(),
            reference_torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        torch.use_deterministic_algorithms(False, warn_only=False)
        reference_torch.use_deterministic_algorithms(False, warn_only=False)

    def tearDown(self):
        torch.use_deterministic_algorithms(False, warn_only=False)
        reference_torch.use_deterministic_algorithms(
            self.reference_original[0],
            warn_only=self.reference_original[1],
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

    def default_outcome(self, module, *, keyword_mode=False, keyword_warn=False):
        before = module.is_grad_enabled()
        if keyword_mode and keyword_warn:
            result = module.use_deterministic_algorithms(
                mode=False,
                warn_only=False,
            )
        elif keyword_mode:
            result = module.use_deterministic_algorithms(mode=False)
        elif keyword_warn:
            result = module.use_deterministic_algorithms(False, warn_only=False)
        else:
            result = module.use_deterministic_algorithms(False)
        return result is None, before, self.state(module)

    def threaded_default_outcomes(self, module):
        call_options = (
            {},
            {"keyword_mode": True},
            {"keyword_warn": True},
            {"keyword_mode": True, "keyword_warn": True},
        )
        worker_count = 12
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = self.default_outcome(
                        module,
                        **call_options[index % len(call_options)],
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

    def test_disabled_default_forms_grad_modes_and_threads_match_pytorch_2_13(self):
        for options in (
            {},
            {"keyword_mode": True},
            {"keyword_warn": True},
            {"keyword_mode": True, "keyword_warn": True},
        ):
            with self.subTest(options=options):
                self.assertEqual(
                    self.default_outcome(torch, **options),
                    self.default_outcome(reference_torch, **options),
                )

        with torch.no_grad(), reference_torch.no_grad():
            self.assertEqual(
                self.default_outcome(torch),
                self.default_outcome(reference_torch),
            )

        self.assertEqual(
            self.threaded_default_outcomes(torch),
            self.threaded_default_outcomes(reference_torch),
        )

    def test_default_use_and_debug_setter_interaction_matches_pytorch_2_13(self):
        for module in (torch, reference_torch):
            with self.subTest(module=module.__name__):
                self.assertIs(
                    module.use_deterministic_algorithms(False, warn_only=False),
                    None,
                )
                self.assertIs(module.set_deterministic_debug_mode(0), None)
                self.assertIs(module.use_deterministic_algorithms(mode=False), None)
                self.assertIs(module.set_deterministic_debug_mode("default"), None)
                self.assertEqual(self.state(module)[:4], (True, 0, True, True))

    def test_callable_metadata_and_documented_boundary(self):
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
        self.assertIn("Deterministic algorithm enforcement", actual.__doc__)
        self.assertIn("remain unsupported", actual.__doc__)
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
            (
                lambda: actual(False, mode=False),
                lambda: expected(False, mode=False),
            ),
            (
                lambda: actual(False, foo=False),
                lambda: expected(False, foo=False),
            ),
            (
                lambda: actual(False, warn_only=False, extra=False),
                lambda: expected(False, warn_only=False, extra=False),
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

    def test_non_bool_argument_errors_match_pytorch_2_13(self):
        shared_values = (
            None,
            0,
            1,
            0.0,
            b"",
            bytearray(b""),
            memoryview(b""),
            [],
            object(),
            _RejectTruthiness(),
            np.bool_(False),
            np.bool_(True),
        )
        for value in shared_values:
            with self.subTest(argument="mode", value=ascii(value)):
                self.assert_error_matches(
                    lambda value=value: torch.use_deterministic_algorithms(
                        value,
                        warn_only=False,
                    ),
                    lambda value=value: reference_torch.use_deterministic_algorithms(
                        value,
                        warn_only=False,
                    ),
                )
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))
                self.assertEqual(
                    self.state(reference_torch)[:4],
                    (True, 0, True, True),
                )
            with self.subTest(argument="warn_only", value=ascii(value)):
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

        native_values = (
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
        for case, (actual_value, expected_value) in enumerate(native_values):
            with self.subTest(argument="mode", native_case=case):
                self.assert_error_matches(
                    lambda value=actual_value: torch.use_deterministic_algorithms(
                        value,
                        warn_only=False,
                    ),
                    lambda value=expected_value: (
                        reference_torch.use_deterministic_algorithms(
                            value,
                            warn_only=False,
                        )
                    ),
                )
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))
                self.assertEqual(
                    self.state(reference_torch)[:4],
                    (True, 0, True, True),
                )
            with self.subTest(argument="warn_only", native_case=case):
                self.assert_error_matches(
                    lambda value=actual_value: torch.use_deterministic_algorithms(
                        False,
                        warn_only=value,
                    ),
                    lambda value=expected_value: (
                        reference_torch.use_deterministic_algorithms(
                            False,
                            warn_only=value,
                        )
                    ),
                )
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))
                self.assertEqual(
                    self.state(reference_torch)[:4],
                    (True, 0, True, True),
                )

        self.assert_error_matches(
            lambda: torch.use_deterministic_algorithms(True, warn_only=0),
            lambda: reference_torch.use_deterministic_algorithms(True, warn_only=0),
        )
        self.assertEqual(self.state(torch)[:4], (True, 0, True, True))
        self.assertEqual(self.state(reference_torch)[:4], (True, 0, True, True))

    def test_nondefault_modes_remain_an_explicit_difference(self):
        cases = (
            (
                lambda: torch.use_deterministic_algorithms(True),
                lambda: reference_torch.use_deterministic_algorithms(True),
                (2, True, False),
            ),
            (
                lambda: torch.use_deterministic_algorithms(True, warn_only=True),
                lambda: reference_torch.use_deterministic_algorithms(
                    True,
                    warn_only=True,
                ),
                (1, True, True),
            ),
            (
                lambda: torch.use_deterministic_algorithms(False, warn_only=True),
                lambda: reference_torch.use_deterministic_algorithms(
                    False,
                    warn_only=True,
                ),
                (0, False, True),
            ),
        )
        for actual_call, expected_call, expected_state in cases:
            with self.subTest(expected_state=expected_state):
                with self.assertRaises(NotImplementedError):
                    actual_call()
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))

                self.assertIsNone(expected_call())
                self.assertEqual(
                    (
                        reference_torch.get_deterministic_debug_mode(),
                        reference_torch.are_deterministic_algorithms_enabled(),
                        reference_torch.is_deterministic_algorithms_warn_only_enabled(),
                    ),
                    expected_state,
                )
                reference_torch.use_deterministic_algorithms(
                    False,
                    warn_only=False,
                )


if __name__ == "__main__":
    unittest.main()
