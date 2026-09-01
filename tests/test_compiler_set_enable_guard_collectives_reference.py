import copy
import importlib
import inspect
import pickle
import pickletools
import signal
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _TruthValue:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __bool__(self):
        self.calls += 1
        return self.value


class _LengthValue:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __len__(self):
        self.calls += 1
        return self.value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerSetEnableGuardCollectivesReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.set_enable_guard_collectives differentials require "
                "pinned PyTorch 2.13.0"
            )

    def setUp(self):
        self.original_actual = torch.compiler.set_enable_guard_collectives(False)
        self.original_expected = (
            reference_torch.compiler.set_enable_guard_collectives(False)
        )

    def tearDown(self):
        torch.compiler.set_enable_guard_collectives(self.original_actual)
        reference_torch.compiler.set_enable_guard_collectives(
            self.original_expected
        )

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

    def state_sequence(self, module):
        function = module.compiler.set_enable_guard_collectives
        values = (
            False,
            True,
            0,
            1,
            -1,
            None,
            "",
            "enabled",
            [],
            [0],
            (),
            (0,),
            object(),
        )
        function(False)
        results = []
        for value in values:
            result = function(value)
            results.append((type(result) is bool, result))
        final = function(False)
        return results, (type(final) is bool, final)

    def truthiness_outcome(self, module, value, use_length=False):
        function = module.compiler.set_enable_guard_collectives
        token = _LengthValue(value) if use_length else _TruthValue(value)
        function(True)
        result = function(token)
        current = function(False)
        return result, current, token.calls

    def error_outcome(self, module, factory):
        function = module.compiler.set_enable_guard_collectives
        token = factory()
        function(True)
        try:
            function(token)
        except Exception as error:
            outcome = type(error), str(error), error.args
        else:
            outcome = None
        current = function(False)
        return outcome, current, getattr(token, "calls", None)

    def test_values_previous_results_and_truthiness_match_pytorch_2_13(self):
        self.assertEqual(
            self.state_sequence(torch),
            self.state_sequence(reference_torch),
        )

        for value in (False, True):
            with self.subTest(protocol="bool", value=value):
                self.assertEqual(
                    self.truthiness_outcome(torch, value),
                    self.truthiness_outcome(reference_torch, value),
                )

        for value in (0, 2):
            with self.subTest(protocol="len", value=value):
                self.assertEqual(
                    self.truthiness_outcome(torch, value, use_length=True),
                    self.truthiness_outcome(
                        reference_torch,
                        value,
                        use_length=True,
                    ),
                )

    def test_truthiness_errors_and_state_preservation_match_pytorch_2_13(self):
        class RaisingTruth:
            def __init__(self):
                self.calls = 0

            def __bool__(self):
                self.calls += 1
                raise RuntimeError("guard collective truthiness failed")

        class InvalidTruth:
            def __bool__(self):
                return 1

        for factory in (RaisingTruth, InvalidTruth):
            with self.subTest(factory=factory.__name__):
                self.assertEqual(
                    self.error_outcome(torch, factory),
                    self.error_outcome(reference_torch, factory),
                )

    def binding_outcome(self, module, case):
        function = module.compiler.set_enable_guard_collectives
        function(True)
        calls = (
            lambda: function(),
            lambda: function(True, False),
            lambda: function(value=True),
            lambda: function(False, enabled=True),
        )
        try:
            calls[case]()
        except Exception as error:
            outcome = type(error), str(error), error.args
        else:
            outcome = None
        return outcome, function(False)

    def test_binding_and_keyword_behavior_match_pytorch_2_13(self):
        for case in range(4):
            with self.subTest(case=case):
                self.assertEqual(
                    self.binding_outcome(torch, case),
                    self.binding_outcome(reference_torch, case),
                )

        torch.compiler.set_enable_guard_collectives(True)
        reference_torch.compiler.set_enable_guard_collectives(True)
        self.assertIs(
            torch.compiler.set_enable_guard_collectives(enabled=False),
            reference_torch.compiler.set_enable_guard_collectives(enabled=False),
        )

    def threaded_outcome(self, module):
        function = module.compiler.set_enable_guard_collectives
        worker_updated = threading.Event()
        main_updated = threading.Event()
        results = []
        errors = []
        function(False)

        def worker():
            try:
                results.append(function(True))
                worker_updated.set()
                if not main_updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for the main thread")
                results.append(function(True))
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_updated.wait(timeout=10))
        main_result = function(False)
        main_updated.set()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        final_result = function(False)
        return results, errors, main_result, final_result

    def test_process_global_thread_state_matches_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

    def overlapping_outcome(self, module):
        function = module.compiler.set_enable_guard_collectives
        truth_barrier = threading.Barrier(3)
        results = []
        errors = []

        class OverlappingTruth:
            def __init__(self):
                self.calls = 0

            def __bool__(self):
                self.calls += 1
                truth_barrier.wait(timeout=10)
                return True

        tokens = [OverlappingTruth(), OverlappingTruth()]
        function(False)

        def worker(token):
            try:
                results.append(function(token))
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [
            threading.Thread(target=worker, args=(token,)) for token in tokens
        ]
        for thread in threads:
            thread.start()
        truth_barrier.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        final_result = function(False)
        return (
            results.count(False),
            results.count(True),
            all(type(result) is bool for result in results),
            errors,
            [token.calls for token in tokens],
            final_result,
        )

    def test_overlapping_exchange_matches_pytorch_2_13(self):
        self.assertEqual(
            self.overlapping_outcome(torch),
            self.overlapping_outcome(reference_torch),
        )

    def signal_reentry_outcome(self, module_name):
        script = r"""
import inspect
import signal
import sys

module = __import__(sys.argv[1])
function = module.compiler.set_enable_guard_collectives
if module.__name__ == "torch_rs":
    exchange = module.compiler._state.exchange_enable_guard_collectives
    if hasattr(exchange, "__code__"):
        target = exchange
        line_marker = "previous_enabled ="
    else:
        target = function
        line_marker = "return _state.exchange_enable_guard_collectives"
else:
    target = function
    line_marker = "return set_guard_complete_hook(guard_collectives_hook)"
source, first_line = inspect.getsourcelines(target)
target_line = first_line + next(
    index for index, line in enumerate(source) if line_marker in line
)
handler_results = []

def handler(signum, frame):
    handler_results.append(function(False))

def tracer(frame, event, arg):
    if (
        frame.f_code is target.__code__
        and event == "line"
        and frame.f_lineno == target_line
    ):
        sys.settrace(None)
        signal.raise_signal(signal.SIGUSR1)
    return tracer

signal.signal(signal.SIGUSR1, handler)
function(True)
sys.settrace(tracer)
outer_result = function(True)
sys.settrace(None)
final_result = function(False)
print(repr((handler_results, outer_result, final_result)))
"""
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script, module_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            self.fail(f"{module_name} deadlocked during signal-handler re-entry")
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        return completed.stdout.strip()

    @unittest.skipUnless(hasattr(signal, "SIGUSR1"), "requires SIGUSR1")
    def test_signal_handler_reentry_matches_pytorch_2_13(self):
        self.assertEqual(
            self.signal_reentry_outcome("torch_rs"),
            self.signal_reentry_outcome("torch"),
        )

    def reload_outcome(self, package, module_name):
        original_module = package.compiler
        old_function = original_module.set_enable_guard_collectives
        original_state = old_function(False)

        try:
            old_function(True)
            reset_result = original_module.reset()
            after_reset = old_function(False)

            old_function(True)
            reloaded = importlib.reload(original_module)
            reload_result = (
                reloaded is original_module,
                package.compiler is original_module,
                reloaded.set_enable_guard_collectives is old_function,
                old_function(False),
                reloaded.set_enable_guard_collectives(True),
            )

            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)
            reimport_result = (
                replacement_module is original_module,
                package.compiler is replacement_module,
                replacement_module.set_enable_guard_collectives(False),
                old_function(True),
                replacement_module.set_enable_guard_collectives(False),
            )
            return reset_result, after_reset, reload_result, reimport_result
        finally:
            old_function(original_state)
            sys.modules[module_name] = original_module
            package.compiler = original_module

    def test_reset_reload_and_reimport_match_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch, "torch_rs.compiler"),
            self.reload_outcome(reference_torch, "torch.compiler"),
        )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.set_enable_guard_collectives
        expected = expected_compiler.set_enable_guard_collectives

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertIs(inspect.getmodule(actual), actual_compiler)
        self.assertIs(inspect.getmodule(expected), expected_compiler)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.set_enable_guard_collectives
        expected = expected_compiler.set_enable_guard_collectives
        supported = {
            "assume_constant_result",
            "reset",
            "allow_in_graph",
            "list_backends",
            "disable",
            "set_default_backend",
            "get_default_backend",
            "set_enable_guard_collectives",
            "is_compiling",
            "is_dynamo_compiling",
            "is_exporting",
            "keep_portable_guards_unsafe",
            "skip_guard_on_inbuilt_nn_modules_unsafe",
            "skip_guard_on_all_nn_modules_unsafe",
            "keep_tensor_guards_unsafe",
            "skip_guard_on_globals_unsafe",
            "skip_all_guards_unsafe",
        }
        self.assertEqual(
            actual_compiler.__all__,
            [name for name in expected_compiler.__all__ if name in supported],
        )

        for module, function in (
            (actual_compiler, actual),
            (expected_compiler, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace[function.__name__], function)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("set_enable_guard_collectives", namespace)

        for function in (actual, expected):
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


if __name__ == "__main__":
    unittest.main()
