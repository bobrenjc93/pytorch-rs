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
            module.is_grad_enabled(),
            module.compiler.get_default_backend(),
            module.compiler.is_compiling(),
            module.compiler.is_dynamo_compiling(),
            module.compiler.is_exporting(),
        )

    def tensor_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            requires_grad=True,
        )
        view = leaf.transpose(0, 1)[1]
        loss = (view * module.tensor([5.0, 7.0])).sum()
        leaf_pointer = leaf.data_ptr()
        view_pointer = view.data_ptr()

        before = (
            leaf.tolist(),
            view.tolist(),
            leaf.requires_grad,
            view.requires_grad,
            leaf.grad is None,
            view.storage_offset(),
            view.stride(),
        )
        first = module.compiler.reset()
        second = module.compiler.reset()
        after = (
            leaf.tolist(),
            view.tolist(),
            leaf.requires_grad,
            view.requires_grad,
            leaf.grad is None,
            view.storage_offset(),
            view.stride(),
        )
        pointers_preserved = (
            leaf.data_ptr() == leaf_pointer,
            view.data_ptr() == view_pointer,
        )
        loss.backward()
        return (
            first is None,
            second is None,
            before,
            after,
            pointers_preserved,
            leaf.grad.tolist(),
        )

    def reset_state_outcome(self, module):
        def reset_once():
            before = self.compiler_state(module)
            first = module.compiler.reset()
            second = module.compiler.reset()
            after = self.compiler_state(module)
            return first is None, second is None, before, after

        states = [reset_once()]
        with module.no_grad():
            states.append(reset_once())
            with module.no_grad():
                states.append(reset_once())
            states.append(reset_once())
        states.append(reset_once())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = (
                    module.no_grad() if index % 2 else contextlib.nullcontext()
                )
                with context:
                    tensor = module.tensor(
                        [float(index), float(index + 1)],
                        requires_grad=True,
                    )
                    pointer = tensor.data_ptr()
                    before = self.compiler_state(module)
                    barrier.wait(timeout=10)
                    first = module.compiler.reset()
                    second = module.compiler.reset()
                    after = self.compiler_state(module)
                    worker_states[index] = (
                        first is None,
                        second is None,
                        before,
                        after,
                        tensor.data_ptr() == pointer,
                        tensor.tolist(),
                        tensor.requires_grad,
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

    def test_repeatable_tensor_and_autograd_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.tensor_outcome(torch),
            self.tensor_outcome(reference_torch),
        )

    def test_grad_query_and_threaded_behavior_matches_pytorch_2_13(self):
        getter = reference_torch.compiler.get_default_backend
        setter = reference_torch.compiler.set_default_backend
        original_backend = getter()

        try:
            setter(None)
            self.assertEqual(
                self.reset_state_outcome(torch),
                self.reset_state_outcome(reference_torch),
            )
        finally:
            setter(original_backend)

        self.assertEqual(torch.compiler.get_default_backend(), "inductor")
        self.assertIs(getter(), original_backend)

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.reset
        expected = expected_compiler.reset

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
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

    def test_exports_copy_and_pickle_match_pytorch_2_13(self):
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
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_compilation_and_cache_serialization_remain_unsupported(self):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(hasattr(reference_torch, "export"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))

        for name in ("compile", "save_cache_artifacts", "load_cache_artifacts"):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(reference_torch.compiler, name)))
                self.assertIn(name, reference_torch.compiler.__all__)
                self.assertFalse(hasattr(torch.compiler, name))
                self.assertNotIn(name, torch.compiler.__all__)


if __name__ == "__main__":
    unittest.main()
