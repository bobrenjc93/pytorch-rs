import copy
import gc
import importlib
import inspect
import pickle
import pickletools
import re
import sys
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AcceleratorEmptyCacheReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "accelerator.empty_cache differentials require pinned PyTorch 2.13.0"
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

    def cpu_build_state(self):
        accelerator = torch.accelerator
        return (
            accelerator.current_accelerator(),
            accelerator.is_available(),
            accelerator.device_count(),
            tuple(accelerator.empty_cache() for _ in range(4)),
            accelerator._discover_accelerator(),
            torch._C._has_cuda,
            torch.version.cuda,
        )

    def test_signature_documentation_exports_and_identity_match_pytorch_2_13(self):
        actual_parent = importlib.import_module("torch_rs.accelerator")
        expected_parent = importlib.import_module("torch.accelerator")
        actual_module = importlib.import_module("torch_rs.accelerator.memory")
        expected_module = importlib.import_module("torch.accelerator.memory")
        actual = actual_module.empty_cache
        expected = expected_module.empty_cache

        self.assertIs(actual_parent.memory, actual_module)
        self.assertIs(expected_parent.memory, expected_module)
        self.assertIs(actual_parent.empty_cache, actual)
        self.assertIs(expected_parent.empty_cache, expected)
        self.assertIs(sys.modules[actual_module.__name__], actual_module)
        self.assertIs(sys.modules[expected_module.__name__], expected_module)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name == "empty_cache"],
        )
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(
            inspect.get_annotations(actual), inspect.get_annotations(expected)
        )
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
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

        for package_name, parent, module, function in (
            ("torch_rs", actual_parent, actual_module, actual),
            ("torch", expected_parent, expected_module, expected),
        ):
            parent_import = {}
            module_import = {}
            parent_wildcard = {}
            module_wildcard = {}
            exec(
                f"from {package_name}.accelerator import empty_cache",
                parent_import,
            )
            exec(
                f"from {package_name}.accelerator.memory import empty_cache",
                module_import,
            )
            exec(f"from {package_name}.accelerator import *", parent_wildcard)
            exec(
                f"from {package_name}.accelerator.memory import *",
                module_wildcard,
            )
            self.assertIs(parent_import["empty_cache"], function)
            self.assertIs(module_import["empty_cache"], function)
            self.assertIs(parent_wildcard["empty_cache"], function)
            self.assertIs(module_wildcard["empty_cache"], function)
            self.assertIs(parent.memory, module)

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

    def reload_contract(self, root):
        parent = root.accelerator
        memory = parent.memory
        parent_namespace = parent.__dict__
        memory_namespace = memory.__dict__
        old_parent_all = parent.__all__
        old_memory_all = memory.__all__
        old_function = memory.empty_cache

        parent_reloaded = importlib.reload(parent)
        parent_kept_function = parent.empty_cache is old_function
        memory_reloaded = importlib.reload(memory)
        new_function = memory.empty_cache
        parent_is_stale = parent.empty_cache is old_function

        try:
            pickle.dumps(old_function)
        except Exception as error:
            stale_pickle_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                    "torch_rs", "torch"
                ),
            )
        else:
            self.fail("a stale accelerator memory function remained pickleable")

        final_parent_reload = importlib.reload(parent)
        return (
            parent_reloaded is parent,
            final_parent_reload is parent,
            parent.__dict__ is parent_namespace,
            memory_reloaded is memory,
            memory.__dict__ is memory_namespace,
            parent.memory is memory,
            sys.modules[parent.__name__] is parent,
            sys.modules[memory.__name__] is memory,
            parent.__all__ is not old_parent_all,
            memory.__all__ is not old_memory_all,
            parent_kept_function,
            old_function is not new_function,
            parent_is_stale,
            parent.empty_cache is new_function,
            old_function() is None,
            new_function() is None,
            copy.copy(new_function) is new_function,
            copy.deepcopy(new_function) is new_function,
            pickle.loads(pickle.dumps(new_function)) is new_function,
            stale_pickle_error,
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(torch.accelerator.empty_cache, protocol),
                    self.pickle_shape(
                        reference_torch.accelerator.empty_cache, protocol
                    ),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.accelerator.empty_cache
        expected = reference_torch.accelerator.empty_cache
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(device=True), lambda: expected(device=True)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_cuda_allocation_exercises_reference_while_torch_rs_stays_cpu_only(self):
        cpu_build_state = self.cpu_build_state()
        self.assertEqual(
            cpu_build_state,
            (
                None,
                False,
                0,
                (None, None, None, None),
                (None, False, 0, None),
                False,
                None,
            ),
        )

        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")

        device_index = reference_torch.cuda.current_device()
        device = reference_torch.device("cuda", device_index)
        self.assertEqual(
            reference_torch.accelerator.current_accelerator(),
            reference_torch.device("cuda"),
        )
        self.assertIs(reference_torch.accelerator.empty_cache(), None)
        reference_torch.cuda.synchronize(device)
        allocated_before = reference_torch.accelerator.memory_allocated(device_index)

        allocation = reference_torch.full(
            (1 << 20,), 3.0, dtype=reference_torch.float32, device=device
        )
        reference_torch.cuda.synchronize(device)
        allocation_bytes = allocation.numel() * allocation.element_size()
        allocated_live = reference_torch.accelerator.memory_allocated(device_index)
        self.assertGreaterEqual(allocated_live, allocated_before + allocation_bytes)

        self.assertIs(reference_torch.accelerator.empty_cache(), None)
        reference_torch.cuda.synchronize(device)
        self.assertEqual(allocation[0].item(), 3.0)
        self.assertEqual(
            reference_torch.accelerator.memory_allocated(device_index),
            allocated_live,
        )
        self.assertEqual(self.cpu_build_state(), cpu_build_state)

        del allocation
        gc.collect()
        reference_torch.cuda.synchronize(device)
        allocated_released = reference_torch.accelerator.memory_allocated(device_index)
        self.assertLessEqual(allocated_released, allocated_live - allocation_bytes)
        reserved_before_empty = reference_torch.accelerator.memory_reserved(
            device_index
        )
        self.assertIs(reference_torch.accelerator.empty_cache(), None)
        reference_torch.cuda.synchronize(device)
        self.assertLessEqual(
            reference_torch.accelerator.memory_reserved(device_index),
            reserved_before_empty,
        )
        self.assertEqual(self.cpu_build_state(), cpu_build_state)
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertNotIn("torch_rs.cuda", sys.modules)


if __name__ == "__main__":
    unittest.main()
