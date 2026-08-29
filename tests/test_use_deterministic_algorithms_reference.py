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
        self.reference_original = (
            reference_torch.are_deterministic_algorithms_enabled(),
            reference_torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        torch.use_deterministic_algorithms(False)
        reference_torch.use_deterministic_algorithms(False, warn_only=False)

    def tearDown(self):
        torch.use_deterministic_algorithms(False)
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

    def default_outcome(self, module, mode, *, keyword=False):
        before = module.is_grad_enabled()
        if keyword:
            result = module.use_deterministic_algorithms(mode=mode)
        else:
            result = module.use_deterministic_algorithms(mode)
        return result is None, before, self.state(module)

    def threaded_default_outcomes(self, module):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = self.default_outcome(module, False)
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

    def test_false_bool_forms_grad_modes_and_threads_match_pytorch_2_13(self):
        for mode, keyword in ((False, False), (False, True)):
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
                self.default_outcome(torch, False),
                self.default_outcome(reference_torch, False),
            )

        self.assertEqual(
            self.threaded_default_outcomes(torch),
            self.threaded_default_outcomes(reference_torch),
        )

    def test_zero_integer_forms_are_a_default_only_extension(self):
        for mode, warn_only in (
            (0, False),
            (False, 0),
            (0, 0),
            (_DefaultInt(0), _DefaultInt(0)),
        ):
            with self.subTest(mode=mode, warn_only=warn_only):
                self.assertIs(
                    torch.use_deterministic_algorithms(mode, warn_only=warn_only),
                    None,
                )
                self.assertEqual(self.state(torch)[:4], (True, 0, True, True))

                with self.assertRaises(TypeError):
                    reference_torch.use_deterministic_algorithms(
                        mode,
                        warn_only=warn_only,
                    )
                self.assertEqual(
                    self.state(reference_torch)[:4],
                    (True, 0, True, True),
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
        self.assertEqual(actual.__doc__, expected.__doc__)
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
        self.assertTrue(hasattr(reference_torch._C, "_set_deterministic_algorithms"))

    def test_argument_binding_and_invalid_types_match_pytorch_2_13(self):
        actual = torch.use_deterministic_algorithms
        expected = reference_torch.use_deterministic_algorithms
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(False, False), lambda: expected(False, False)),
            (
                lambda: actual(False, False, False),
                lambda: expected(False, False, False),
            ),
            (lambda: actual(enabled=False), lambda: expected(enabled=False)),
            (
                lambda: actual(False, warnonly=False),
                lambda: expected(False, warnonly=False),
            ),
            (
                lambda: actual(False, mode=False),
                lambda: expected(False, mode=False),
            ),
            (
                lambda: actual(False, False, warn_only=False),
                lambda: expected(False, False, warn_only=False),
            ),
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(0.0), lambda: expected(0.0)),
            (lambda: actual(""), lambda: expected("")),
            (lambda: actual(_CustomMode()), lambda: expected(_CustomMode())),
            (
                lambda: actual(False, warn_only=None),
                lambda: expected(False, warn_only=None),
            ),
            (
                lambda: actual(False, warn_only=0.0),
                lambda: expected(False, warn_only=0.0),
            ),
            (
                lambda: actual(False, warn_only=""),
                lambda: expected(False, warn_only=""),
            ),
            (
                lambda: actual(False, warn_only=_CustomMode()),
                lambda: expected(False, warn_only=_CustomMode()),
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

    def test_unsupported_enabled_and_warn_only_modes_are_atomic(self):
        for actual_call, expected_call, expected_state in (
            (
                lambda: torch.use_deterministic_algorithms(True),
                lambda: reference_torch.use_deterministic_algorithms(True),
                (2, True, False),
            ),
            (
                lambda: torch.use_deterministic_algorithms(False, warn_only=True),
                lambda: reference_torch.use_deterministic_algorithms(
                    False,
                    warn_only=True,
                ),
                (0, False, True),
            ),
            (
                lambda: torch.use_deterministic_algorithms(True, warn_only=True),
                lambda: reference_torch.use_deterministic_algorithms(
                    True,
                    warn_only=True,
                ),
                (1, True, True),
            ),
        ):
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
                reference_torch.use_deterministic_algorithms(False, warn_only=False)


if __name__ == "__main__":
    unittest.main()
