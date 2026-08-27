import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


FUNCTION_DOC = """
    Enables use of collectives *during* guard evaluation to synchronize behavior
    across ranks.  This is expensive: we have to issue a collective every time
    we enter a compiled code region, even if no rank actually would need to
    compile.  This can help prevent NCCL hangs by ensuring that we never have a
    situation where one rank starts recompiling while other ranks don't compile;
    it is especially useful in conjunction with enable_compiler_collectives
    where such a situation would immediately cause a hang (as it is necessary
    for all ranks to compile at the same time to run compiler collectives).  Like
    compiler collectives, you can only run this on SPMD programs; you will hang
    otherwise.  Note that a guard collective is only issued if there is any
    compiled code to guard on; if this the first time we encounter a frame or
    the frame is skipped, we don't issue collectives.

    Returns the previous setting of enabled.
    """


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


class CompilerSetEnableGuardCollectivesTests(unittest.TestCase):
    def setUp(self):
        self.original_enabled = torch.compiler.set_enable_guard_collectives(False)

    def tearDown(self):
        torch.compiler.set_enable_guard_collectives(self.original_enabled)

    def test_truthy_and_falsy_values_return_previous_exact_bool(self):
        function = torch.compiler.set_enable_guard_collectives
        values = (
            (False, False),
            (True, True),
            (0, False),
            (1, True),
            (-1, True),
            (None, False),
            ("", False),
            ("enabled", True),
            ([], False),
            ([0], True),
            ((), False),
            ((0,), True),
            (object(), True),
        )

        previous = False
        for value, expected_enabled in values:
            with self.subTest(value=value):
                result = function(value)
                self.assertIs(type(result), bool)
                self.assertIs(result, previous)
                previous = expected_enabled

        self.assertIs(function(False), previous)

    def test_truthiness_is_evaluated_once_before_state_changes(self):
        function = torch.compiler.set_enable_guard_collectives

        function(True)
        falsy = _TruthValue(False)
        self.assertIs(function(falsy), True)
        self.assertEqual(falsy.calls, 1)
        self.assertIs(function(True), False)

        truthy = _TruthValue(True)
        self.assertIs(function(truthy), True)
        self.assertEqual(truthy.calls, 1)
        self.assertIs(function(False), True)

        empty = _LengthValue(0)
        self.assertIs(function(empty), False)
        self.assertEqual(empty.calls, 1)
        self.assertIs(function(True), False)

        nonempty = _LengthValue(2)
        self.assertIs(function(nonempty), True)
        self.assertEqual(nonempty.calls, 1)
        self.assertIs(function(False), True)

    def test_truthiness_errors_preserve_the_previous_state(self):
        function = torch.compiler.set_enable_guard_collectives

        class RaisingTruth:
            def __init__(self):
                self.calls = 0

            def __bool__(self):
                self.calls += 1
                raise RuntimeError("guard collective truthiness failed")

        class InvalidTruth:
            def __bool__(self):
                return 1

        function(True)
        raising = RaisingTruth()
        with self.assertRaisesRegex(
            RuntimeError, "^guard collective truthiness failed$"
        ):
            function(raising)
        self.assertEqual(raising.calls, 1)
        self.assertIs(function(False), True)

        function(True)
        with self.assertRaisesRegex(
            TypeError, "^__bool__ should return bool, returned int$"
        ):
            function(InvalidTruth())
        self.assertIs(function(False), True)

    def test_state_is_process_global_across_threads(self):
        function = torch.compiler.set_enable_guard_collectives
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
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_updated.wait(timeout=10))
        self.assertIs(function(False), True)
        main_updated.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [False, False])
        self.assertIs(function(False), True)

    def test_overlapping_calls_exchange_state_atomically(self):
        compiler = torch.compiler
        function = compiler.set_enable_guard_collectives
        state = compiler._state
        read_barrier = threading.Barrier(2)
        results = []
        errors = []

        class InterleavingState:
            def __getattr__(self, name):
                if name == "enable_guard_collectives":
                    value = state.enable_guard_collectives
                    read_barrier.wait(timeout=10)
                    return value
                return getattr(state, name)

            def __setattr__(self, name, value):
                setattr(state, name, value)

        function(False)
        compiler._state = InterleavingState()

        def worker():
            try:
                results.append(function(True))
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        finally:
            compiler._state = state

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(results.count(False), 1)
        self.assertEqual(results.count(True), 1)
        self.assertTrue(all(type(result) is bool for result in results))
        self.assertIs(function(False), True)

    def test_reset_preserves_state_and_grad_mode(self):
        function = torch.compiler.set_enable_guard_collectives

        def assert_reset_preserves_state(expected_grad_state):
            function(True)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(torch.compiler.reset(), None)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(False), True)

        assert_reset_preserves_state(True)
        with torch.no_grad():
            assert_reset_preserves_state(False)
        assert_reset_preserves_state(True)

    def test_reload_and_reimport_share_state_with_old_functions(self):
        original_module = torch.compiler
        old_function = original_module.set_enable_guard_collectives
        old_function(False)

        try:
            self.assertIs(importlib.reload(original_module), original_module)
            self.assertIs(torch.compiler, original_module)
            self.assertIsNot(
                original_module.set_enable_guard_collectives,
                old_function,
            )
            self.assertIs(old_function(True), False)
            self.assertIs(
                original_module.set_enable_guard_collectives(False),
                True,
            )

            old_function(True)
            module_name = original_module.__name__
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)
            self.assertIsNot(replacement_module, original_module)
            self.assertIs(torch.compiler, replacement_module)
            self.assertIs(
                replacement_module.set_enable_guard_collectives(False),
                True,
            )
            self.assertIs(old_function(True), False)
            self.assertIs(
                replacement_module.set_enable_guard_collectives(False),
                True,
            )
        finally:
            sys.modules[original_module.__name__] = original_module
            torch.compiler = original_module

    def test_signature_metadata_exports_copy_and_pickle(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.set_enable_guard_collectives

        self.assertIs(torch.compiler, compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            inspect.signature(function),
            inspect.Signature(
                [
                    inspect.Parameter(
                        "enabled",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=bool,
                    )
                ]
            ),
        )
        self.assertEqual(function.__annotations__, {"enabled": bool})
        self.assertEqual(typing.get_type_hints(function), {"enabled": bool})
        self.assertEqual(function.__name__, "set_enable_guard_collectives")
        self.assertEqual(function.__qualname__, "set_enable_guard_collectives")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {"_dynamo_forbidden": True})
        self.assertFalse(hasattr(function, "__text_signature__"))

        self.assertIn("set_enable_guard_collectives", compiler.__all__)
        namespace = {}
        exec("from torch_rs.compiler import *", namespace)
        self.assertIs(namespace[function.__name__], function)

        self.assertNotIn("set_enable_guard_collectives", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("set_enable_guard_collectives", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_binding_errors_preserve_state(self):
        function = torch.compiler.set_enable_guard_collectives
        cases = (
            (
                lambda: function(),
                "set_enable_guard_collectives() missing 1 required positional "
                "argument: 'enabled'",
            ),
            (
                lambda: function(True, False),
                "set_enable_guard_collectives() takes 1 positional argument but "
                "2 were given",
            ),
            (
                lambda: function(value=True),
                "set_enable_guard_collectives() got an unexpected keyword "
                "argument 'value'",
            ),
            (
                lambda: function(False, enabled=True),
                "set_enable_guard_collectives() got multiple values for argument "
                "'enabled'",
            ),
        )

        for call, message in cases:
            with self.subTest(message=message):
                function(True)
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(function(False), True)

        function(True)
        self.assertIs(function(enabled=False), True)
        self.assertIs(function(False), False)

    def test_calls_do_not_initialize_compilation_or_distributed_execution(self):
        script = r"""
import importlib
import sys

class RejectCompilerImport:
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "torch"
            or fullname.startswith("torch.")
            or fullname == "torch_rs._dynamo"
            or fullname.startswith("torch_rs._dynamo.")
            or fullname.startswith("torch_rs.compiler.backends")
            or fullname == "torch_rs.compiler.registry"
        ):
            raise RuntimeError(f"compiler import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectCompilerImport())
import torch_rs as torch

compiler = torch.compiler
modules_before_call = set(sys.modules)
backend = lambda graph_module, example_inputs: graph_module.forward
compiler.set_default_backend(backend)
assert torch.distributed.is_initialized() is False
assert compiler.set_enable_guard_collectives(False) is False
assert compiler.set_enable_guard_collectives(True) is False
assert compiler.reset() is None
assert compiler.set_enable_guard_collectives(False) is True
assert compiler.get_default_backend() is backend
assert torch.distributed.is_initialized() is False
assert set(sys.modules) == modules_before_call
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name == "torch_rs._dynamo"
    or name.startswith("torch_rs._dynamo.")
    or name.startswith("torch_rs.compiler.backends")
    or name == "torch_rs.compiler.registry"
    for name in sys.modules
)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
