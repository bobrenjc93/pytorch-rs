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


class _DefaultString(str):
    pass


class _CustomMode:
    pass


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SetDeterministicDebugModeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "set_deterministic_debug_mode differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.reference_original = reference_torch.get_deterministic_debug_mode()
        torch.set_deterministic_debug_mode(0)
        reference_torch.set_deterministic_debug_mode(0)

    def tearDown(self):
        torch.set_deterministic_debug_mode(0)
        reference_torch.set_deterministic_debug_mode(self.reference_original)

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

    def default_outcome(self, module, mode, *, keyword=False):
        before = module.is_grad_enabled()
        if keyword:
            result = module.set_deterministic_debug_mode(debug_mode=mode)
        else:
            result = module.set_deterministic_debug_mode(mode)
        return result is None, before, self.state(module)

    def threaded_default_outcomes(self, module):
        modes = (0, False, "default", _DefaultInt(0), _DefaultString("default"))
        worker_count = 10
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
                        modes[index % len(modes)],
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

    def test_default_forms_grad_modes_and_threads_match_pytorch_2_13(self):
        for mode, keyword in (
            (0, False),
            (0, True),
            (False, False),
            ("default", False),
            (_DefaultInt(0), False),
            (_DefaultString("default"), False),
        ):
            with self.subTest(mode=mode, keyword=keyword):
                self.assertEqual(
                    self.default_outcome(torch, mode, keyword=keyword),
                    self.default_outcome(
                        reference_torch,
                        mode,
                        keyword=keyword,
                    ),
                )

        with torch.no_grad(), reference_torch.no_grad():
            self.assertEqual(
                self.default_outcome(torch, "default"),
                self.default_outcome(reference_torch, "default"),
            )

        self.assertEqual(
            self.threaded_default_outcomes(torch),
            self.threaded_default_outcomes(reference_torch),
        )

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs")
        expected_module = importlib.import_module("torch")
        actual = actual_module.set_deterministic_debug_mode
        expected = expected_module.set_deterministic_debug_mode

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
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual = torch.set_deterministic_debug_mode
        expected = reference_torch.set_deterministic_debug_mode

        self.assertEqual(
            torch.__all__.count("set_deterministic_debug_mode"),
            reference_torch.__all__.count("set_deterministic_debug_mode"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["set_deterministic_debug_mode"], function)
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
        actual = torch.set_deterministic_debug_mode
        expected = reference_torch.set_deterministic_debug_mode
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(0, 0), lambda: expected(0, 0)),
            (lambda: actual(mode=0), lambda: expected(mode=0)),
            (
                lambda: actual(0, debug_mode=0),
                lambda: expected(0, debug_mode=0),
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

    def test_malformed_input_errors_match_pytorch_2_13(self):
        shared_values = (
            None,
            0.0,
            b"default",
            bytearray(b"default"),
            memoryview(b"default"),
            [],
            object(),
            _CustomMode(),
            -1,
            3,
            10**100,
            "",
            "DEFAULT",
            " default",
            "warning",
        )
        for value in shared_values:
            with self.subTest(value=ascii(value)):
                self.assert_error_matches(
                    lambda value=value: torch.set_deterministic_debug_mode(value),
                    lambda value=value: reference_torch.set_deterministic_debug_mode(
                        value
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
            with self.subTest(native_case=case):
                self.assert_error_matches(
                    lambda value=actual_value: torch.set_deterministic_debug_mode(
                        value
                    ),
                    lambda value=expected_value: (
                        reference_torch.set_deterministic_debug_mode(value)
                    ),
                )
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))
                self.assertEqual(
                    self.state(reference_torch)[:4],
                    (True, 0, True, True),
                )

    def test_nondefault_modes_remain_an_explicit_difference(self):
        for mode, expected_state in (
            (1, (1, True, True)),
            (True, (1, True, True)),
            (2, (2, True, False)),
            ("warn", (1, True, True)),
            ("error", (2, True, False)),
        ):
            with self.subTest(mode=mode):
                with self.assertRaises(NotImplementedError):
                    torch.set_deterministic_debug_mode(mode)
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))

                self.assertIsNone(
                    reference_torch.set_deterministic_debug_mode(mode)
                )
                self.assertEqual(
                    (
                        reference_torch.get_deterministic_debug_mode(),
                        reference_torch.are_deterministic_algorithms_enabled(),
                        reference_torch.is_deterministic_algorithms_warn_only_enabled(),
                    ),
                    expected_state,
                )
                reference_torch.set_deterministic_debug_mode(0)


if __name__ == "__main__":
    unittest.main()
