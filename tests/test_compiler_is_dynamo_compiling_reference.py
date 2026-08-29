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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerIsDynamoCompilingReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.is_dynamo_compiling differentials require pinned "
                "PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def supported_state_outcome(self, module):
        function = module.compiler.is_dynamo_compiling

        def query_outcome():
            before = module.is_grad_enabled()
            result = function()
            after = module.is_grad_enabled()
            return before, result is False, after

        states = [query_outcome()]
        with module.no_grad():
            states.append(query_outcome())
            with module.no_grad():
                states.append(query_outcome())
            states.append(query_outcome())
        states.append(query_outcome())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = query_outcome()
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
        return states, worker_states

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

    def test_supported_eager_threaded_and_grad_states_match_pytorch_2_13(self):
        self.assertEqual(
            self.supported_state_outcome(torch),
            self.supported_state_outcome(reference_torch),
        )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.is_dynamo_compiling
        expected = expected_compiler.is_dynamo_compiling

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
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

    def test_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.is_dynamo_compiling
        expected = expected_compiler.is_dynamo_compiling

        self.assertEqual(
            actual_compiler.__all__,
            [
                name
                for name in expected_compiler.__all__
                if name
                in {
                    "assume_constant_result",
                    "reset",
                    "disable",
                    "set_default_backend",
                    "get_default_backend",
                    "set_enable_guard_collectives",
                    "cudagraph_mark_step_begin",
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
            ],
        )
        self.assertEqual(
            torch.__all__.count("compiler"),
            reference_torch.__all__.count("compiler"),
        )
        self.assertEqual(
            torch.__all__.count("assume_constant_result"),
            reference_torch.__all__.count("assume_constant_result"),
        )
        self.assertEqual(
            torch.__all__.count("is_dynamo_compiling"),
            reference_torch.__all__.count("is_dynamo_compiling"),
        )
        self.assertEqual(
            torch.__all__.count("is_exporting"),
            reference_torch.__all__.count("is_exporting"),
        )

        for module, function in (
            (actual_compiler, actual),
            (expected_compiler, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(
                namespace["assume_constant_result"],
                module.assume_constant_result,
            )
            self.assertIs(namespace["is_dynamo_compiling"], function)
            self.assertIs(namespace["is_exporting"], module.is_exporting)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("assume_constant_result", namespace)
            self.assertNotIn("is_dynamo_compiling", namespace)
            self.assertNotIn("is_exporting", namespace)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.is_dynamo_compiling
        expected = reference_torch.compiler.is_dynamo_compiling
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_reference_eager_backend_compile_bounds_the_true_state(self):
        actual_is_dynamo_compiling = torch.compiler.is_dynamo_compiling
        expected_is_dynamo_compiling = reference_torch.compiler.is_dynamo_compiling
        actual_states = [actual_is_dynamo_compiling()]
        expected_states = [expected_is_dynamo_compiling()]

        def forward(value):
            return (
                value + 1,
                expected_is_dynamo_compiling(),
                actual_is_dynamo_compiling(),
            )

        compiled = reference_torch.compile(
            forward,
            backend="eager",
            fullgraph=True,
        )
        result, expected_inside, actual_inside = compiled(reference_torch.tensor(1.0))
        actual_states.extend((actual_inside, actual_is_dynamo_compiling()))
        expected_states.extend((expected_inside, expected_is_dynamo_compiling()))

        self.assertEqual(result.item(), 2.0)
        for state in actual_states:
            self.assertIs(state, False)
        self.assertEqual(expected_states, [False, True, False])
        for state in expected_states:
            self.assertIs(type(state), bool)

    def test_compilation_remains_unsupported(self):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(hasattr(reference_torch, "export"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "is_dynamo_compiling"))


if __name__ == "__main__":
    unittest.main()
