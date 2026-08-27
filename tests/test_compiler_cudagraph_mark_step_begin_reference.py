import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
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


SUPPORTED_COMPILER_EXPORTS = {
    "assume_constant_result",
    "reset",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "cudagraph_mark_step_begin",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "keep_portable_guards_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
}


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerCudagraphMarkStepBeginReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.cudagraph_mark_step_begin differentials require "
                "pinned PyTorch 2.13.0"
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

    def state_outcome(self, module):
        compiler = module.compiler
        function = compiler.cudagraph_mark_step_begin
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            expected_compiler_state = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )

            states = []
            for context in (contextlib.nullcontext(), module.no_grad()):
                with context:
                    before = module.is_grad_enabled()
                    first = function()
                    middle = module.is_grad_enabled()
                    second = function()
                    states.append(
                        (
                            before,
                            first,
                            middle,
                            second,
                            module.is_grad_enabled(),
                            compiler.get_default_backend() is backend,
                            (
                                compiler.is_compiling(),
                                compiler.is_dynamo_compiling(),
                                compiler.is_exporting(),
                            ),
                        )
                    )

            worker_count = 8
            barrier = threading.Barrier(worker_count)
            worker_states = [None] * worker_count
            errors = []

            def worker(index):
                try:
                    context = (
                        module.no_grad() if index % 2 else contextlib.nullcontext()
                    )
                    with context:
                        barrier.wait(timeout=10)
                        before = module.is_grad_enabled()
                        result = function()
                        worker_states[index] = (
                            before,
                            result,
                            module.is_grad_enabled(),
                            compiler.get_default_backend() is backend,
                            (
                                compiler.is_compiling(),
                                compiler.is_dynamo_compiling(),
                                compiler.is_exporting(),
                            ),
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
            return states, worker_states, expected_compiler_state
        finally:
            compiler.set_default_backend(original_backend)

    def test_eager_repeated_threaded_and_grad_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.state_outcome(torch),
            self.state_outcome(reference_torch),
        )

    def test_signature_metadata_and_identity_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.cudagraph_mark_step_begin
        expected = expected_compiler.cudagraph_mark_step_begin

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
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
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)
        self.assertEqual(actual.__code__.co_argcount, expected.__code__.co_argcount)
        self.assertEqual(
            actual.__code__.co_posonlyargcount,
            expected.__code__.co_posonlyargcount,
        )
        self.assertEqual(
            actual.__code__.co_kwonlyargcount,
            expected.__code__.co_kwonlyargcount,
        )
        self.assertEqual(actual.__code__.co_names, ())

    def test_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.cudagraph_mark_step_begin
        expected = expected_compiler.cudagraph_mark_step_begin

        self.assertEqual(
            actual_compiler.__all__,
            [
                name
                for name in expected_compiler.__all__
                if name in SUPPORTED_COMPILER_EXPORTS
            ],
        )
        self.assertEqual(
            torch.__all__.count("cudagraph_mark_step_begin"),
            reference_torch.__all__.count("cudagraph_mark_step_begin"),
        )

        for compiler, function in (
            (actual_compiler, actual),
            (expected_compiler, expected),
        ):
            namespace = {}
            exec(f"from {compiler.__name__} import *", namespace)
            self.assertIs(namespace[function.__name__], function)
            self.assertEqual(compiler.__all__.count(function.__name__), 1)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("cudagraph_mark_step_begin", namespace)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.cudagraph_mark_step_begin
        expected = reference_torch.compiler.cudagraph_mark_step_begin
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(step=None), lambda: expected(step=None)),
            (
                lambda: actual(None, step=None),
                lambda: expected(None, step=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_reload_behavior_matches_pytorch_2_13(self):
        script = r'''
import importlib
import pickle
import sys

outcomes = []
for package_name in ("torch_rs", "torch"):
    package = importlib.import_module(package_name)
    compiler = package.compiler
    original_backend = compiler.get_default_backend()

    def backend(graph_module, example_inputs):
        return graph_module.forward

    try:
        compiler.set_default_backend(backend)
        old_function = compiler.cudagraph_mark_step_begin
        old_exports = compiler.__all__
        first_result = old_function()
        reloaded = importlib.reload(compiler)
        new_function = reloaded.cudagraph_mark_step_begin
        try:
            pickle.dumps(old_function)
        except pickle.PicklingError:
            old_pickles = False
        else:
            old_pickles = True
        outcomes.append(
            (
                first_result,
                reloaded is compiler,
                package.compiler is compiler,
                sys.modules[compiler.__name__] is compiler,
                new_function is old_function,
                compiler.__all__ is old_exports,
                compiler.get_default_backend() is backend,
                old_function(),
                new_function(),
                old_pickles,
                all(
                    pickle.loads(pickle.dumps(new_function, protocol)) is new_function
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ),
            )
        )
    finally:
        compiler.set_default_backend(original_backend)

assert outcomes[0] == outcomes[1], outcomes
'''
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

    def test_compilation_remains_deliberately_unsupported(self):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(callable(reference_torch.compiler.compile))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch.compiler, "compile"))


if __name__ == "__main__":
    unittest.main()
