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
class CompiledWithCxx11AbiReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiled_with_cxx11_abi differentials require pinned PyTorch 2.13.0"
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

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs")
        expected_module = importlib.import_module("torch")
        actual = actual_module.compiled_with_cxx11_abi
        expected = expected_module.compiled_with_cxx11_abi

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))
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
        self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)

    def test_build_specific_values_are_exact_native_booleans(self):
        actual = torch.compiled_with_cxx11_abi()
        expected = reference_torch.compiled_with_cxx11_abi()

        self.assertIs(type(actual), bool)
        self.assertIs(type(expected), bool)
        self.assertIs(actual, torch._C._GLIBCXX_USE_CXX11_ABI)
        self.assertIs(expected, reference_torch._C._GLIBCXX_USE_CXX11_ABI)
        self.assertIs(actual, False)

    def test_direct_import_and_private_flag_placement_match_pytorch_2_13(self):
        for module in (torch, reference_torch):
            with self.subTest(module=module.__name__):
                function = module.compiled_with_cxx11_abi
                flag = module._C._GLIBCXX_USE_CXX11_ABI
                direct_function = {}
                direct_flag = {}
                package_wildcard = {}
                native_wildcard = {}

                exec(
                    f"from {module.__name__} import compiled_with_cxx11_abi",
                    direct_function,
                )
                exec(
                    f"from {module.__name__}._C import _GLIBCXX_USE_CXX11_ABI",
                    direct_flag,
                )
                exec(f"from {module.__name__} import *", package_wildcard)
                exec(f"from {module.__name__}._C import *", native_wildcard)

                self.assertIs(direct_function["compiled_with_cxx11_abi"], function)
                self.assertIs(direct_flag["_GLIBCXX_USE_CXX11_ABI"], flag)
                self.assertNotIn("compiled_with_cxx11_abi", module.__all__)
                self.assertNotIn("compiled_with_cxx11_abi", package_wildcard)
                self.assertFalse(hasattr(module, "_GLIBCXX_USE_CXX11_ABI"))
                self.assertIn("_GLIBCXX_USE_CXX11_ABI", vars(module._C))
                self.assertNotIn("_GLIBCXX_USE_CXX11_ABI", package_wildcard)
                self.assertNotIn("_GLIBCXX_USE_CXX11_ABI", native_wildcard)
                if hasattr(module._C, "__all__"):
                    self.assertNotIn("_GLIBCXX_USE_CXX11_ABI", module._C.__all__)

    def threaded_contract(self, module):
        function = module.compiled_with_cxx11_abi
        native_value = module._C._GLIBCXX_USE_CXX11_ABI
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    expected_grad_state = module.is_grad_enabled()
                    barrier.wait(timeout=10)
                    first = function()
                    second = function()
                    results[index] = (
                        module.is_grad_enabled() is expected_grad_state,
                        type(first) is bool,
                        first is native_value,
                        second is first,
                        module.is_grad_enabled() is expected_grad_state,
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

    def test_threaded_query_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_contract(torch),
            self.threaded_contract(reference_torch),
        )

    def test_copying_and_pickling_match_pytorch_2_13(self):
        actual = torch.compiled_with_cxx11_abi
        expected = reference_torch.compiled_with_cxx11_abi

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
        actual = torch.compiled_with_cxx11_abi
        expected = reference_torch.compiled_with_cxx11_abi
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


if __name__ == "__main__":
    unittest.main()
