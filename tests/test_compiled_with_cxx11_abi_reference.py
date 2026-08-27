import copy
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

    def test_metadata_matches_pytorch_2_13(self):
        actual = torch.compiled_with_cxx11_abi
        expected = reference_torch.compiled_with_cxx11_abi

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(
            inspect.get_annotations(actual),
            inspect.get_annotations(expected),
        )
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

    def placement_contract(self, module):
        native = module._C
        package_import = {}
        native_import = {}
        package_wildcard = {}
        native_wildcard = {}
        exec(
            f"from {module.__name__} import compiled_with_cxx11_abi",
            package_import,
        )
        exec(
            f"from {native.__name__} import _GLIBCXX_USE_CXX11_ABI",
            native_import,
        )
        exec(f"from {module.__name__} import *", package_wildcard)
        exec(f"from {native.__name__} import *", native_wildcard)
        function = module.compiled_with_cxx11_abi
        flag = native._GLIBCXX_USE_CXX11_ABI
        return (
            "compiled_with_cxx11_abi" in vars(module),
            "compiled_with_cxx11_abi" in vars(native),
            "_GLIBCXX_USE_CXX11_ABI" in vars(module),
            "_GLIBCXX_USE_CXX11_ABI" in vars(native),
            module.__all__.count("compiled_with_cxx11_abi"),
            module.__all__.count("_GLIBCXX_USE_CXX11_ABI"),
            package_import["compiled_with_cxx11_abi"] is function,
            native_import["_GLIBCXX_USE_CXX11_ABI"] is flag,
            "compiled_with_cxx11_abi" in package_wildcard,
            "_GLIBCXX_USE_CXX11_ABI" in package_wildcard,
            "_GLIBCXX_USE_CXX11_ABI" in native_wildcard,
            type(flag) is bool,
            function() is flag,
        )

    def test_direct_import_and_wildcard_placement_match_pytorch_2_13(self):
        self.assertEqual(
            self.placement_contract(torch),
            self.placement_contract(reference_torch),
        )

    def test_values_are_build_specific_exact_booleans(self):
        actual = torch.compiled_with_cxx11_abi()
        expected = reference_torch.compiled_with_cxx11_abi()

        self.assertIs(actual, torch._C._GLIBCXX_USE_CXX11_ABI)
        self.assertIs(expected, reference_torch._C._GLIBCXX_USE_CXX11_ABI)
        self.assertIs(type(actual), bool)
        self.assertIs(type(expected), bool)
        self.assertIs(actual, False)

    def mutation_contract(self, module):
        function = module.compiled_with_cxx11_abi
        native = module._C
        original = native._GLIBCXX_USE_CXX11_ABI
        observations = []

        try:
            for replacement in (None, not original, 1, "cxx11", object()):
                native._GLIBCXX_USE_CXX11_ABI = replacement
                result = function()
                observations.append(
                    (
                        type(result) is bool,
                        result is original,
                    )
                )

            del native._GLIBCXX_USE_CXX11_ABI
            result = function()
            observations.append(
                (
                    not hasattr(native, "_GLIBCXX_USE_CXX11_ABI"),
                    type(result) is bool,
                    result is original,
                )
            )
        finally:
            native._GLIBCXX_USE_CXX11_ABI = original

        return observations

    def test_native_flag_mutation_and_deletion_match_pytorch_2_13(self):
        actual = self.mutation_contract(torch)
        expected = self.mutation_contract(reference_torch)

        self.assertEqual(actual, expected)
        self.assertEqual(
            actual,
            [(True, True)] * 5 + [(True, True, True)],
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
            ((None,), {}),
            ((None, None), {}),
            ((), {"enabled": True}),
            ((None,), {"enabled": True}),
        )

        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs):
                self.assert_error_matches(
                    lambda: actual(*args, **kwargs),
                    lambda: expected(*args, **kwargs),
                )

    def threaded_contract(self, module):
        function = module.compiled_with_cxx11_abi
        flag = module._C._GLIBCXX_USE_CXX11_ABI
        barrier = threading.Barrier(16)
        results = [None] * 16
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=5)
                value = function()
                results[index] = (
                    type(value) is bool,
                    value is flag,
                    function is module.compiled_with_cxx11_abi,
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(16)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        return (
            [thread.is_alive() for thread in threads],
            errors,
            results,
        )

    def test_thread_behavior_matches_pytorch_2_13(self):
        actual = self.threaded_contract(torch)
        expected = self.threaded_contract(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, ([False] * 16, [], [(True, True, True)] * 16))


if __name__ == "__main__":
    unittest.main()
