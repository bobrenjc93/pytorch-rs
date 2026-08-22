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

    def test_backend_free_results_preserve_list_and_filtering_invariants(self):
        actual = torch.compiler.list_backends
        expected = reference_torch.compiler.list_backends

        actual_default = actual()
        actual_default_again = actual()
        actual_unfiltered = actual(())
        actual_unfiltered_keyword = actual(exclude_tags=[])
        actual_unknown_tag = actual(("torch_rs_unknown_tag",))
        actual_debug_only = actual(("debug",))

        self.assertEqual(
            [
                actual_default,
                actual_default_again,
                actual_unfiltered,
                actual_unfiltered_keyword,
                actual_unknown_tag,
                actual_debug_only,
            ],
            [[], [], [], [], [], []],
        )
        self.assertIsNot(actual_default, actual_default_again)
        self.assertIsNot(actual_unfiltered, actual_unfiltered_keyword)

        expected_default = expected()
        expected_default_again = expected()
        expected_unfiltered = expected(())
        expected_unfiltered_keyword = expected(exclude_tags=[])
        expected_unknown_tag = expected(("torch_rs_unknown_tag",))
        expected_debug_only = expected(("debug",))

        for result in (
            expected_default,
            expected_default_again,
            expected_unfiltered,
            expected_unfiltered_keyword,
            expected_unknown_tag,
            expected_debug_only,
        ):
            self.assertIs(type(result), list)
            self.assertEqual(result, sorted(result))
            self.assertTrue(all(type(name) is str for name in result))

        self.assertIsNot(expected_default, expected_default_again)
        self.assertIsNot(expected_unfiltered, expected_unfiltered_keyword)
        self.assertEqual(expected_unfiltered, expected_unfiltered_keyword)
        self.assertEqual(expected_unknown_tag, expected_unfiltered)
        self.assertLessEqual(set(expected_default), set(expected_unfiltered))
        self.assertLessEqual(set(expected_debug_only), set(expected_unfiltered))
        self.assertTrue(expected_unfiltered)

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.list_backends
        expected = expected_compiler.list_backends

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

    def test_exports_imports_copy_and_pickle_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.list_backends
        expected = expected_compiler.list_backends
        supported = {
            "assume_constant_result",
            "list_backends",
            "get_default_backend",
            "is_compiling",
            "is_dynamo_compiling",
            "is_exporting",
        }

        self.assertEqual(
            actual_compiler.__all__,
            [name for name in expected_compiler.__all__ if name in supported],
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
            direct_namespace = {}
            exec(
                f"from {module.__name__} import list_backends",
                direct_namespace,
            )
            self.assertIs(direct_namespace["list_backends"], module.list_backends)

            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            for name in supported:
                self.assertIs(wildcard_namespace[name], getattr(module, name))

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

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.list_backends
        expected = reference_torch.compiler.list_backends
        cases = (
            (lambda: actual(None, None), lambda: expected(None, None)),
            (
                lambda: actual((), exclude_tags=()),
                lambda: expected((), exclude_tags=()),
            ),
            (lambda: actual(tags=()), lambda: expected(tags=())),
            (
                lambda: actual((), extra=True),
                lambda: expected((), extra=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_backend_registration_and_compilation_remain_unsupported(self):
        reference_torch.compiler.list_backends()
        self.assertTrue(callable(reference_torch._dynamo.register_backend))
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(callable(reference_torch.compiler.compile))

        self.assertFalse(hasattr(torch, "_dynamo"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))
        self.assertFalse(hasattr(torch.compiler, "set_default_backend"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch, "compile"))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs._dynamo")


if __name__ == "__main__":
    unittest.main()
