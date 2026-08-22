import contextlib
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
class CompilerResetReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.reset differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def compiler_state(self, module):
        return (
            module.compiler.get_default_backend(),
            module.compiler.is_compiling(),
            module.compiler.is_dynamo_compiling(),
            module.compiler.is_exporting(),
        )

    def eager_tensor_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        view = leaf.view(4)
        weights = module.tensor([2.0, -1.0, 0.5, 3.0])
        loss = (view * weights).sum()
        leaf_identity = id(leaf)
        view_identity = id(view)
        leaf_pointer = leaf.data_ptr()
        view_offset = view.storage_offset()
        state = self.compiler_state(module)

        returns_none = [module.compiler.reset() is None for _ in range(5)]
        state_after = self.compiler_state(module)
        tensor_before_backward = (
            id(leaf) == leaf_identity,
            id(view) == view_identity,
            leaf.data_ptr() == leaf_pointer,
            view.data_ptr() == leaf_pointer,
            view.storage_offset() == view_offset,
            leaf.tolist(),
            view.tolist(),
            leaf.grad is None,
        )
        loss.backward()
        return (
            returns_none,
            state,
            state_after,
            tensor_before_backward,
            leaf.grad.tolist(),
        )

    def grad_mode_outcome(self, module):
        def observe():
            before_grad = module.is_grad_enabled()
            before_state = self.compiler_state(module)
            result = module.compiler.reset()
            after_state = self.compiler_state(module)
            after_grad = module.is_grad_enabled()
            return (
                before_grad,
                result is None,
                before_state,
                after_state,
                after_grad,
            )

        states = [observe()]
        with module.no_grad():
            states.append(observe())
            with module.no_grad():
                states.append(observe())
            states.append(observe())
        states.append(observe())
        return states

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

    def test_eager_tensor_graph_and_grad_mode_behavior_match_pytorch_2_13(self):
        self.assertEqual(
            self.eager_tensor_outcome(torch),
            self.eager_tensor_outcome(reference_torch),
        )
        self.assertEqual(
            self.grad_mode_outcome(torch),
            self.grad_mode_outcome(reference_torch),
        )

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.reset
        expected = expected_compiler.reset

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
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

    def test_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.reset
        expected = expected_compiler.reset
        supported = {
            "assume_constant_result",
            "reset",
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
            torch.__all__.count("reset"),
            reference_torch.__all__.count("reset"),
        )

        for module in (actual_compiler, expected_compiler):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in supported:
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("reset", namespace)

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
        actual = torch.compiler.reset
        expected = reference_torch.compiler.reset
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(state=None), lambda: expected(state=None)),
            (
                lambda: actual(None, state=None),
                lambda: expected(None, state=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_compilation_and_cache_serialization_remain_unsupported(self):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(callable(reference_torch.compiler.compile))
        self.assertTrue(callable(reference_torch.compiler.save_cache_artifacts))
        self.assertTrue(callable(reference_torch.compiler.load_cache_artifacts))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        for name in (
            "compile",
            "load_compiled_function",
            "save_cache_artifacts",
            "load_cache_artifacts",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.compiler, name))
                self.assertNotIn(name, torch.compiler.__all__)


if __name__ == "__main__":
    unittest.main()
