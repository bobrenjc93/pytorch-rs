import copy
import importlib
import inspect
import pickle
import threading
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class NoGradNamespaceTests(unittest.TestCase):
    def test_canonical_imports_are_identical_and_minimal(self):
        autograd = importlib.import_module("torch_rs.autograd")
        grad_mode = importlib.import_module("torch_rs.autograd.grad_mode")
        from torch_rs.autograd import no_grad as autograd_no_grad
        from torch_rs.autograd.grad_mode import no_grad as grad_mode_no_grad

        self.assertIs(torch.autograd, autograd)
        self.assertIs(autograd.grad_mode, grad_mode)
        self.assertIs(torch.no_grad, autograd.no_grad)
        self.assertIs(torch.no_grad, grad_mode.no_grad)
        self.assertIs(torch.no_grad, autograd_no_grad)
        self.assertIs(torch.no_grad, grad_mode_no_grad)
        self.assertEqual(autograd.__all__, ["grad_mode", "no_grad"])
        self.assertEqual(grad_mode.__all__, ["no_grad"])
        self.assertNotIn("autograd", torch.__all__)

        for module in (autograd, grad_mode):
            for unsupported in ("enable_grad", "grad", "backward"):
                with self.subTest(module=module.__name__, unsupported=unsupported):
                    self.assertFalse(hasattr(module, unsupported))

    def test_metadata_copy_and_pickle_resolve_through_grad_mode(self):
        grad_mode = torch.autograd.grad_mode
        context_type = torch.no_grad

        self.assertEqual(context_type.__name__, "no_grad")
        self.assertEqual(context_type.__qualname__, "no_grad")
        self.assertEqual(context_type.__module__, "torch_rs.autograd.grad_mode")
        self.assertIs(inspect.getmodule(context_type), grad_mode)
        self.assertIs(copy.copy(context_type), context_type)
        self.assertIs(copy.deepcopy(context_type), context_type)

        instance = context_type()
        for operation in (copy.copy, copy.deepcopy):
            with self.subTest(operation=operation.__name__):
                restored = operation(instance)
                self.assertIsNot(restored, instance)
                self.assertIs(type(restored), context_type)
                with restored:
                    self.assertFalse(torch.is_grad_enabled())
                self.assertTrue(torch.is_grad_enabled())

        canonical_path = b"torch_rs.autograd.grad_mode"
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol, value="class"):
                payload = pickle.dumps(context_type, protocol=protocol)
                self.assertIn(canonical_path, payload)
                self.assertIs(pickle.loads(payload), context_type)

            with self.subTest(protocol=protocol, value="instance"):
                payload = pickle.dumps(instance, protocol=protocol)
                self.assertIn(canonical_path, payload)
                restored = pickle.loads(payload)
                self.assertIsNot(restored, instance)
                self.assertIs(type(restored), context_type)
                with restored:
                    self.assertFalse(torch.is_grad_enabled())
                self.assertTrue(torch.is_grad_enabled())

    def test_aliases_preserve_context_decorator_generator_and_thread_behavior(self):
        autograd_no_grad = torch.autograd.no_grad
        grad_mode_no_grad = torch.autograd.grad_mode.no_grad
        value = torch.tensor([2.0], requires_grad=True)

        with autograd_no_grad():
            self.assertFalse((value * value).requires_grad)
            with grad_mode_no_grad():
                self.assertFalse((value * value).requires_grad)
            self.assertFalse((value * value).requires_grad)
        self.assertTrue((value * value).requires_grad)

        @autograd_no_grad()
        def decorated():
            return (value * value).requires_grad

        @grad_mode_no_grad()
        def generate():
            request = yield (value * value).requires_grad
            yield request, (value * value).requires_grad

        self.assertFalse(decorated())
        generator = generate()
        self.assertFalse(next(generator))
        self.assertTrue((value * value).requires_grad)
        self.assertEqual(generator.send("resume"), ("resume", False))
        self.assertTrue((value * value).requires_grad)

        worker_states = []

        def worker():
            worker_states.append(torch.is_grad_enabled())
            with grad_mode_no_grad():
                worker_states.append(torch.is_grad_enabled())
            worker_states.append(torch.is_grad_enabled())

        with autograd_no_grad():
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            self.assertFalse(torch.is_grad_enabled())

        self.assertEqual(worker_states, [True, False, True])
        self.assertTrue(torch.is_grad_enabled())


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class NoGradNamespaceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "no-grad namespace differentials require pinned PyTorch 2.13.0"
            )

    def contract(self, module):
        context_type = module.no_grad
        instance = context_type()
        module_prefix = module.__name__
        return {
            "aliases": (
                module.autograd.no_grad is context_type,
                module.autograd.grad_mode.no_grad is context_type,
            ),
            "metadata": (
                context_type.__name__,
                context_type.__qualname__,
                context_type.__module__.removeprefix(module_prefix),
                inspect.getmodule(context_type) is module.autograd.grad_mode,
            ),
            "class_copy": (
                copy.copy(context_type) is context_type,
                copy.deepcopy(context_type) is context_type,
            ),
            "instance_copy": tuple(
                (copied is instance, type(copied) is context_type)
                for copied in (copy.copy(instance), copy.deepcopy(instance))
            ),
            "pickle": tuple(
                (
                    pickle.loads(pickle.dumps(context_type, protocol=protocol))
                    is context_type,
                    type(
                        pickle.loads(pickle.dumps(instance, protocol=protocol))
                    )
                    is context_type,
                )
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_namespace_metadata_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(self.contract(torch), self.contract(reference_torch))


if __name__ == "__main__":
    unittest.main()
