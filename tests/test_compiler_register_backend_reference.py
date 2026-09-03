import contextlib
import copy
import importlib
import pickle
import unittest

import torch_rs as torch
import torch_rs._compiler_state as compiler_state

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerRegisterBackendReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.register_backend differentials require pinned "
                "PyTorch 2.13.0"
            )

    @contextlib.contextmanager
    def isolated_actual_registry(self):
        original_backends = compiler_state.backends.copy()
        original_compiler_fns = compiler_state.compiler_fns.copy()
        compiler_state.backends.clear()
        compiler_state.compiler_fns.clear()
        try:
            yield
        finally:
            compiler_state.backends.clear()
            compiler_state.backends.update(original_backends)
            compiler_state.compiler_fns.clear()
            compiler_state.compiler_fns.update(original_compiler_fns)

    @contextlib.contextmanager
    def isolated_reference_registry(self):
        registry = importlib.import_module("torch._dynamo.backends.registry")
        original_backends = registry._BACKENDS
        original_compiler_fns = registry._COMPILER_FNS
        original_lazy_import = registry._lazy_import
        registry._BACKENDS = {}
        registry._COMPILER_FNS = {}
        registry._lazy_import = lambda: None
        try:
            yield
        finally:
            registry._BACKENDS = original_backends
            registry._COMPILER_FNS = original_compiler_fns
            registry._lazy_import = original_lazy_import

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def registration_outcome(self, register_backend, list_backends):
        events = []

        def implicit_backend(graph_module, example_inputs):
            events.append("implicit")
            return graph_module.forward

        def explicit_backend(graph_module, example_inputs):
            events.append("explicit")
            return graph_module.forward

        def empty_name_backend(graph_module, example_inputs):
            events.append("empty")
            return graph_module.forward

        implicit_result = register_backend(implicit_backend)
        explicit_result = register_backend(explicit_backend, "explicit_backend_name")
        empty_name_result = register_backend(empty_name_backend, "")

        debug_decorator = register_backend(name="debug_backend", tags=("debug",))
        experimental_decorator = register_backend(
            name="experimental_backend",
            tags=("experimental",),
        )

        @register_backend()
        def factory_backend(graph_module, example_inputs):
            events.append("factory")
            return graph_module.forward

        @debug_decorator
        def debug_backend(graph_module, example_inputs):
            events.append("debug")
            return graph_module.forward

        @experimental_decorator
        def experimental_backend(graph_module, example_inputs):
            events.append("experimental")
            return graph_module.forward

        default_result = list_backends()
        empty_exclusion_result = list_backends(())
        none_exclusion_result = list_backends(None)
        debug_exclusion_result = list_backends(("debug",))
        experimental_exclusion_result = list_backends(("experimental",))
        string_exclusion_result = list_backends("debug")

        return (
            implicit_result is implicit_backend,
            explicit_result is explicit_backend,
            empty_name_result is empty_name_backend,
            factory_backend._tags,
            debug_backend._tags,
            experimental_backend._tags,
            events,
            default_result,
            empty_exclusion_result,
            none_exclusion_result,
            debug_exclusion_result,
            experimental_exclusion_result,
            "debug_backend" in string_exclusion_result,
            default_result is not list_backends(),
            empty_exclusion_result is not list_backends(()),
        )

    def test_registration_forms_and_backend_lists_match_pytorch_2_13_registry(self):
        with self.isolated_actual_registry():
            actual = self.registration_outcome(
                torch.compiler.register_backend,
                torch.compiler.list_backends,
            )
        with self.isolated_reference_registry():
            expected = self.registration_outcome(
                reference_torch._dynamo.register_backend,
                reference_torch.compiler.list_backends,
            )
        self.assertEqual(actual, expected)

    def invalid_registration_outcome(self, register_backend, list_backends):
        def backend(graph_module, example_inputs):
            return graph_module.forward

        class NamelessBackend:
            def __call__(self, graph_module, example_inputs):
                return graph_module.forward

        register_backend(backend, "duplicate_backend")
        before = list_backends(())

        outcomes = []
        cases = (
            lambda: register_backend(backend, "duplicate_backend"),
            lambda: register_backend(42, "numeric_backend"),
            lambda: register_backend(NamelessBackend()),
            lambda: register_backend(backend, "bad_tags", tags=None),
        )
        for call in cases:
            try:
                call()
            except BaseException as error:
                outcomes.append((type(error).__name__, str(error), error.args))
            else:
                outcomes.append(None)
        return before, outcomes

    def test_duplicate_and_invalid_inputs_match_pytorch_2_13_registry(self):
        with self.isolated_actual_registry():
            actual_before, actual_outcomes = self.invalid_registration_outcome(
                torch.compiler.register_backend,
                torch.compiler.list_backends,
            )
        with self.isolated_reference_registry():
            expected_before, expected_outcomes = self.invalid_registration_outcome(
                reference_torch._dynamo.register_backend,
                reference_torch.compiler.list_backends,
            )

        self.assertEqual(actual_before, expected_before)
        self.assertEqual(actual_outcomes, expected_outcomes)

    def test_argument_errors_copy_and_pickle_match_pytorch_2_13_registry(self):
        actual = torch.compiler.register_backend
        expected = reference_torch._dynamo.register_backend

        def backend(graph_module, example_inputs):
            return graph_module.forward

        cases = (
            (lambda: actual(backend, "x", (), 1), lambda: expected(backend, "x", (), 1)),
            (
                lambda: actual(backend, name="x", unexpected=1),
                lambda: expected(backend, name="x", unexpected=1),
            ),
            (lambda: actual(fn=backend), lambda: expected(fn=backend)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function.__module__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol)),
                        function,
                    )

    def test_compile_execution_paths_remain_unsupported(self):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(callable(reference_torch.compiler.compile))
        self.assertFalse(hasattr(reference_torch.compiler, "register_backend"))
        self.assertTrue(callable(reference_torch._dynamo.register_backend))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertTrue(callable(torch.compiler.register_backend))


if __name__ == "__main__":
    unittest.main()
