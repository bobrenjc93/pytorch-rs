import copy
import importlib
import inspect
import pickle
import pickletools
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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerListBackendsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.list_backends differentials require pinned PyTorch 2.13.0"
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

    def backend_list_outcome(self, function):
        default = function()
        second_default = function()
        positional_tuple = function(("debug", "experimental"))
        keyword_tuple = function(exclude_tags=("debug", "experimental"))
        empty_tags = function(())
        none_tags = function(None)

        results = (
            default,
            second_default,
            positional_tuple,
            keyword_tuple,
            empty_tags,
            none_tags,
        )
        return {
            "result_types": tuple(type(result).__name__ for result in results),
            "all_strings": tuple(
                all(type(value) is str for value in result) for result in results
            ),
            "fresh": (
                default is not second_default,
                default is not positional_tuple,
                positional_tuple is not keyword_tuple,
                empty_tags is not none_tags,
            ),
            "default_matches_explicit_default": default == positional_tuple,
            "keyword_matches_positional": positional_tuple == keyword_tuple,
        }

    def test_empty_registry_behavior_is_a_valid_pytorch_2_13_list_backends_shape(self):
        actual_default = torch.compiler.list_backends()
        actual_empty_tags = torch.compiler.list_backends(())
        actual_none_tags = torch.compiler.list_backends(None)

        self.assertIs(type(actual_default), list)
        self.assertEqual(actual_default, [])
        self.assertEqual(actual_empty_tags, [])
        self.assertEqual(actual_none_tags, [])
        self.assertIsNot(actual_default, actual_empty_tags)
        self.assertIsNot(actual_empty_tags, actual_none_tags)

        self.assertEqual(
            self.backend_list_outcome(torch.compiler.list_backends),
            self.backend_list_outcome(reference_torch.compiler.list_backends),
        )

    def test_signature_documentation_and_metadata_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.list_backends
        expected = expected_compiler.list_backends

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            typing.get_type_hints(actual),
            typing.get_type_hints(expected),
        )
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

    def test_exports_direct_import_wildcard_copy_and_pickle_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.list_backends
        expected = expected_compiler.list_backends

        self.assertEqual(
            actual_compiler.__all__,
            [
                name
                for name in expected_compiler.__all__
                if name in SUPPORTED_COMPILER_EXPORTS
            ],
        )
        self.assertEqual(
            torch.__all__.count("compiler"),
            reference_torch.__all__.count("compiler"),
        )
        self.assertEqual(
            torch.__all__.count("list_backends"),
            reference_torch.__all__.count("list_backends"),
        )

        for module in (actual_compiler, expected_compiler):
            namespace = {}
            exec(f"from {module.__name__} import list_backends", namespace)
            self.assertIs(namespace["list_backends"], module.list_backends)

            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in SUPPORTED_COMPILER_EXPORTS:
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("list_backends", namespace)

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

    def reload_outcome(self, module, compiler_module_name):
        compiler = importlib.import_module(compiler_module_name)
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            old_function = compiler.list_backends
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.list_backends

            try:
                pickle.dumps(old_function)
            except BaseException as error:
                old_pickle_error = (
                    type(error).__name__,
                    "not the same object" in str(error),
                )
            else:
                old_pickle_error = None

            new_pickle_results = tuple(
                pickle.loads(pickle.dumps(new_function, protocol)) is new_function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            )
            query_result = new_function()
            return (
                reloaded is compiler,
                module.compiler is compiler,
                old_function is new_function,
                old_exports is compiler.__all__,
                compiler.get_default_backend() is backend,
                type(query_result).__name__,
                all(type(value) is str for value in query_result),
                copy.copy(old_function) is old_function,
                copy.deepcopy(old_function) is old_function,
                old_pickle_error,
                new_pickle_results,
            )
        finally:
            compiler.set_default_backend(original_backend)

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch, "torch_rs.compiler"),
            self.reload_outcome(reference_torch, "torch.compiler"),
        )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.list_backends
        expected = reference_torch.compiler.list_backends
        cases = (
            (lambda: actual([], []), lambda: expected([], [])),
            (
                lambda: actual((), exclude_tags=()),
                lambda: expected((), exclude_tags=()),
            ),
            (lambda: actual(tags=()), lambda: expected(tags=())),
            (
                lambda: actual(exclude_tags=(), unexpected=()),
                lambda: expected(exclude_tags=(), unexpected=()),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_compile_registration_and_execution_paths_remain_unsupported(self):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(callable(reference_torch.compiler.compile))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "list_backends"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertTrue(callable(torch.compiler.register_backend))


if __name__ == "__main__":
    unittest.main()
