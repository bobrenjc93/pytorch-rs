import copy
import importlib
import inspect
import operator
import pickle
import re
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch


class AutogradBackwardTests(unittest.TestCase):
    def test_single_root_calls_return_none_and_accumulate_gradients(self):
        calls = (
            lambda loss: torch.autograd.backward(loss),
            lambda loss: torch.autograd.backward(tensors=loss),
            lambda loss: torch.autograd.backward(
                loss,
                grad_tensors=None,
                retain_graph=None,
                create_graph=False,
                grad_variables=None,
                inputs=None,
            ),
            lambda loss: torch.autograd.backward(
                loss, None, False, False, None, None
            ),
            lambda loss: torch.autograd.backward(
                loss, None, operator.index(False), 0
            ),
        )

        for case, call in enumerate(calls):
            with self.subTest(case=case):
                leaf = torch.tensor([2.0, -3.0], requires_grad=True)
                loss = (leaf * leaf).sum()
                self.assertIsNone(call(loss))
                self.assertEqual(leaf.grad.tolist(), [4.0, -6.0])

    def test_graph_reuse_freeing_and_accumulation_follow_tensor_backward(self):
        reusable_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        reusable_loss = reusable_leaf.transpose(0, 0).sum()
        torch.autograd.backward(reusable_loss)
        torch.autograd.backward(reusable_loss)
        self.assertEqual(reusable_leaf.grad.tolist(), [2.0, 2.0])

        scalar_leaf = torch.tensor(7.0, requires_grad=True)
        torch.autograd.backward(scalar_leaf)
        torch.autograd.backward(scalar_leaf)
        self.assertEqual(scalar_leaf.grad.item(), 2.0)

        freed_leaf = torch.tensor([2.0, 3.0], requires_grad=True)
        freed_loss = (freed_leaf * freed_leaf).sum()
        torch.autograd.backward(freed_loss)
        self.assertEqual(freed_leaf.grad.tolist(), [4.0, 6.0])
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            torch.autograd.backward(freed_loss)

        torch.autograd.backward((freed_leaf * freed_leaf).sum())
        self.assertEqual(freed_leaf.grad.tolist(), [8.0, 12.0])

    def test_tensor_backward_errors_are_preserved(self):
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.autograd.backward(torch.tensor(1.0))
        with self.assertRaisesRegex(
            RuntimeError, "implicitly created only for scalar"
        ):
            torch.autograd.backward(
                torch.tensor([1.0, 2.0], requires_grad=True)
            )

    def test_unsupported_forms_fail_before_gradients_or_graph_state_change(self):
        unsupported = (
            (
                "tuple roots",
                TypeError,
                "torch_rs.autograd.backward only supports one exact native Tensor",
                lambda leaf, loss: torch.autograd.backward((loss,)),
            ),
            (
                "list roots",
                TypeError,
                "torch_rs.autograd.backward only supports one exact native Tensor",
                lambda leaf, loss: torch.autograd.backward([loss]),
            ),
            (
                "explicit gradient",
                NotImplementedError,
                "torch_rs.autograd.backward does not support explicit gradients",
                lambda leaf, loss: torch.autograd.backward(
                    loss, grad_tensors=torch.tensor(1.0)
                ),
            ),
            (
                "retained graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support retain_graph=True",
                lambda leaf, loss: torch.autograd.backward(
                    loss, retain_graph=True
                ),
            ),
            (
                "higher-order graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support create_graph=True",
                lambda leaf, loss: torch.autograd.backward(
                    loss, create_graph=True
                ),
            ),
            (
                "grad_variables",
                NotImplementedError,
                "torch_rs.autograd.backward does not support grad_variables",
                lambda leaf, loss: torch.autograd.backward(
                    loss, grad_variables=torch.tensor(1.0)
                ),
            ),
            (
                "inputs",
                NotImplementedError,
                "torch_rs.autograd.backward does not support inputs",
                lambda leaf, loss: torch.autograd.backward(loss, inputs=leaf),
            ),
        )

        for label, error_type, message, call in unsupported:
            with self.subTest(label=label):
                leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                loss = (leaf * leaf).sum()
                with self.assertRaisesRegex(
                    error_type, f"^{re.escape(message)}$"
                ):
                    call(leaf, loss)
                self.assertIsNone(leaf.grad)

                loss.backward()
                self.assertEqual(leaf.grad.tolist(), [4.0, 6.0])

    def test_graph_option_conversion_errors_are_non_mutating(self):
        cases = (
            ("retain_graph", 0.5),
            ("create_graph", None),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                leaf = torch.tensor(2.0, requires_grad=True)
                loss = leaf * leaf
                with self.assertRaises(TypeError) as raised:
                    torch.autograd.backward(loss, **{name: value})
                self.assertEqual(
                    str(raised.exception),
                    f"'{type(value).__name__}' object cannot be interpreted as "
                    "an integer",
                )
                self.assertIsNone(leaf.grad)
                loss.backward()
                self.assertEqual(leaf.grad.item(), 4.0)

    def test_metadata_imports_copying_pickling_and_exports(self):
        module = importlib.import_module("torch_rs.autograd")
        function = module.backward

        self.assertIs(module, torch.autograd)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "backward")
        self.assertEqual(function.__qualname__, "backward")
        self.assertEqual(function.__module__, "torch_rs.autograd")
        self.assertEqual(
            tuple(inspect.signature(function).parameters),
            (
                "tensors",
                "grad_tensors",
                "retain_graph",
                "create_graph",
                "grad_variables",
                "inputs",
            ),
        )
        self.assertEqual(function.__defaults__, (None, None, False, None, None))
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(
            tuple(function.__annotations__),
            (
                "tensors",
                "grad_tensors",
                "retain_graph",
                "create_graph",
                "grad_variables",
                "inputs",
                "return",
            ),
        )
        self.assertIs(
            typing.get_args(function.__annotations__["tensors"])[0],
            torch.Tensor,
        )
        self.assertIs(function.__annotations__["create_graph"], bool)
        self.assertIsNone(function.__annotations__["return"])
        self.assertIn("Compute the sum of gradients", function.__doc__)

        self.assertEqual(module.__all__.count("backward"), 1)
        wildcard_namespace = {}
        exec("from torch_rs.autograd import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["backward"], function)
        explicit_namespace = {}
        exec("from torch_rs.autograd import backward", explicit_namespace)
        self.assertIs(explicit_namespace["backward"], function)

        self.assertFalse(hasattr(torch, "backward"))
        self.assertNotIn("backward", torch.__all__)
        self.assertFalse(hasattr(module, "grad"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_signature_binding_errors_are_non_mutating(self):
        leaf = torch.tensor(2.0, requires_grad=True)
        loss = leaf * leaf
        calls = (
            lambda: torch.autograd.backward(),
            lambda: torch.autograd.backward(loss, tensors=loss),
            lambda: torch.autograd.backward(loss, unexpected=True),
            lambda: torch.autograd.backward(
                loss, None, None, False, None, None, None
            ),
        )
        for case, call in enumerate(calls):
            with self.subTest(case=case), self.assertRaises(TypeError):
                call()
        self.assertIsNone(leaf.grad)
        loss.backward()
        self.assertEqual(leaf.grad.item(), 4.0)

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch
from torch_rs.autograd import backward

leaf = torch.tensor(2.0, requires_grad=True)
assert backward(leaf * leaf) is None
assert leaf.grad.item() == 4.0
assert not hasattr(torch.autograd, "grad")
assert not hasattr(torch, "backward")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
