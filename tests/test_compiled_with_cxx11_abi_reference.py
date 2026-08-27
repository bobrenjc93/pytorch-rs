import copy
import importlib
import inspect
import pickle
import pickletools
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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompiledWithCxx11AbiReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiled_with_cxx11_abi differentials require pinned PyTorch "
                "2.13.0"
            )

    def callable_contract(self, module):
        function = module.compiled_with_cxx11_abi
        direct = {}
        wildcard = {}
        exec(
            f"from {module.__name__} import compiled_with_cxx11_abi",
            direct,
        )
        exec(f"from {module.__name__} import *", wildcard)
        return {
            "type": type(function).__name__,
            "is_function": type(function) is types.FunctionType,
            "signature": str(inspect.signature(function)),
            "annotations": inspect.get_annotations(function),
            "type_hints": typing.get_type_hints(function),
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__.replace("torch_rs", "torch"),
            "getmodule_identity": inspect.getmodule(function) is module,
            "doc": function.__doc__,
            "defaults": function.__defaults__,
            "kwdefaults": function.__kwdefaults__,
            "dict": function.__dict__,
            "has_text_signature": hasattr(function, "__text_signature__"),
            "direct_identity": direct["compiled_with_cxx11_abi"] is function,
            "in_vars": "compiled_with_cxx11_abi" in vars(module),
            "all_count": module.__all__.count("compiled_with_cxx11_abi"),
            "excluded_from_wildcard": "compiled_with_cxx11_abi" not in wildcard,
        }

    def test_callable_metadata_and_placement_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def private_flag_contract(self, module):
        native = module._C
        direct = {}
        wildcard = {}
        exec(
            f"from {native.__name__} import _GLIBCXX_USE_CXX11_ABI",
            direct,
        )
        exec(f"from {native.__name__} import *", wildcard)
        flag = native._GLIBCXX_USE_CXX11_ABI
        return {
            "type": type(flag).__name__,
            "is_exact_bool": type(flag) is bool,
            "in_native_vars": "_GLIBCXX_USE_CXX11_ABI" in vars(native),
            "absent_from_package": not hasattr(
                module, "_GLIBCXX_USE_CXX11_ABI"
            ),
            "function_absent_from_native": not hasattr(
                native, "compiled_with_cxx11_abi"
            ),
            "direct_identity": direct["_GLIBCXX_USE_CXX11_ABI"] is flag,
            "excluded_from_native_wildcard": (
                "_GLIBCXX_USE_CXX11_ABI" not in wildcard
            ),
            "function_returns_flag": module.compiled_with_cxx11_abi() is flag,
        }

    def test_private_native_flag_placement_matches_pytorch_2_13(self):
        self.assertEqual(
            self.private_flag_contract(torch),
            self.private_flag_contract(reference_torch),
        )
        self.assertIs(torch._C._GLIBCXX_USE_CXX11_ABI, False)
        self.assertIs(torch.compiled_with_cxx11_abi(), False)
        self.assertIs(
            type(reference_torch._C._GLIBCXX_USE_CXX11_ABI),
            bool,
        )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

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
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol)), expected
                )
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def thread_contract(self, module):
        function = module.compiled_with_cxx11_abi
        native = module._C
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count

        def worker(index):
            barrier.wait(timeout=10)
            value = function()
            results[index] = (
                type(value) is bool,
                value is native._GLIBCXX_USE_CXX11_ABI,
                function is module.compiled_with_cxx11_abi,
            )

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        return (
            not any(thread.is_alive() for thread in threads),
            tuple(results),
        )

    def test_thread_observations_match_pytorch_2_13(self):
        self.assertEqual(
            self.thread_contract(torch),
            self.thread_contract(reference_torch),
        )

    def native_reload_contract(self, module):
        native = module._C
        function = module.compiled_with_cxx11_abi
        flag = native._GLIBCXX_USE_CXX11_ABI

        reloaded = importlib.reload(native)

        return (
            reloaded is native,
            module._C is native,
            module.compiled_with_cxx11_abi is function,
            native._GLIBCXX_USE_CXX11_ABI is flag,
            module.compiled_with_cxx11_abi()
            is native._GLIBCXX_USE_CXX11_ABI,
        )

    def test_native_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.native_reload_contract(torch),
            self.native_reload_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
