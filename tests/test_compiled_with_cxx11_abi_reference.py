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


FLAG_NAME = "_GLIBCXX_USE_CXX11_ABI"
FUNCTION_NAME = "compiled_with_cxx11_abi"


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

    def test_build_specific_values_are_exact_booleans(self):
        self.assertIs(torch.compiled_with_cxx11_abi(), False)
        self.assertIs(getattr(torch._C, FLAG_NAME), False)
        self.assertIs(type(torch.compiled_with_cxx11_abi()), bool)
        self.assertIs(type(reference_torch.compiled_with_cxx11_abi()), bool)
        self.assertIs(type(getattr(reference_torch._C, FLAG_NAME)), bool)
        self.assertIs(
            torch.compiled_with_cxx11_abi(),
            getattr(torch._C, FLAG_NAME),
        )
        self.assertIs(
            reference_torch.compiled_with_cxx11_abi(),
            getattr(reference_torch._C, FLAG_NAME),
        )

    def test_metadata_matches_pytorch_2_13(self):
        actual = torch.compiled_with_cxx11_abi
        expected = reference_torch.compiled_with_cxx11_abi

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
        self.assertIs(inspect.getmodule(actual), torch)
        self.assertIs(inspect.getmodule(expected), reference_torch)
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

    def test_direct_import_and_wildcard_placement_matches_pytorch_2_13(self):
        for module in (torch, reference_torch):
            with self.subTest(module=module.__name__):
                function = getattr(module, FUNCTION_NAME)
                native = module._C
                package_direct = {}
                native_direct = {}
                package_wildcard = {}
                native_wildcard = {}

                exec(
                    f"from {module.__name__} import {FUNCTION_NAME}",
                    package_direct,
                )
                exec(
                    f"from {module.__name__}._C import {FLAG_NAME}",
                    native_direct,
                )
                exec(f"from {module.__name__} import *", package_wildcard)
                exec(f"from {module.__name__}._C import *", native_wildcard)

                self.assertIs(package_direct[FUNCTION_NAME], function)
                self.assertIs(native_direct[FLAG_NAME], getattr(native, FLAG_NAME))
                self.assertFalse(hasattr(module, FLAG_NAME))
                self.assertFalse(hasattr(native, FUNCTION_NAME))
                self.assertEqual(module.__all__.count(FUNCTION_NAME), 0)
                self.assertNotIn(FUNCTION_NAME, package_wildcard)
                self.assertNotIn(FLAG_NAME, package_wildcard)
                self.assertNotIn(FLAG_NAME, native_wildcard)

    def test_copying_and_pickling_match_pytorch_2_13(self):
        actual = torch.compiled_with_cxx11_abi
        expected = reference_torch.compiled_with_cxx11_abi

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

    def threaded_contract(self, module):
        function = module.compiled_with_cxx11_abi
        native_value = getattr(module._C, FLAG_NAME)
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                first = function()
                second = function()
                results[index] = (
                    type(first) is bool,
                    type(second) is bool,
                    first is native_value,
                    second is native_value,
                    first is second,
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

    def test_thread_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_contract(torch),
            self.threaded_contract(reference_torch),
        )

    def native_reload_contract(self, module):
        native = module._C
        old_function = module.compiled_with_cxx11_abi
        old_value = getattr(native, FLAG_NAME)
        reloaded = importlib.reload(native)
        return (
            reloaded is native,
            module._C is native,
            module.compiled_with_cxx11_abi is old_function,
            getattr(native, FLAG_NAME) is old_value,
            old_function() is getattr(native, FLAG_NAME),
            importlib.import_module(module.__name__) is module,
        )

    def test_native_reload_and_package_reimport_match_pytorch_2_13(self):
        self.assertEqual(
            self.native_reload_contract(torch),
            self.native_reload_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
