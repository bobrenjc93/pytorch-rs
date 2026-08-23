import copy
import inspect
import operator
import pickle
import re
import types
import unittest

import torch_rs as torch


class IndexValue:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class TensorBackwardTests(unittest.TestCase):
    def test_default_equivalent_positional_and_keyword_forms(self):
        calls = (
            lambda loss: loss.backward(),
            lambda loss: loss.backward(None),
            lambda loss: loss.backward(gradient=None),
            lambda loss: loss.backward(None, None, False, None),
            lambda loss: loss.backward(None, False, False, None),
            lambda loss: loss.backward(
                gradient=None,
                retain_graph=None,
                create_graph=False,
                inputs=None,
            ),
            lambda loss: loss.backward(
                None, IndexValue(0), IndexValue(0), None
            ),
            lambda loss: torch.Tensor.backward(
                self=loss,
                gradient=None,
                retain_graph=operator.index(False),
                create_graph=0,
                inputs=None,
            ),
        )

        for case, call in enumerate(calls):
            with self.subTest(case=case):
                leaf = torch.tensor([2.0, -3.0], requires_grad=True)
                loss = (leaf * leaf).sum()
                self.assertIsNone(call(loss))
                self.assertEqual(leaf.grad.tolist(), [4.0, -6.0])

    def test_scalar_validation_is_preserved(self):
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.tensor(1.0).backward(
                gradient=None,
                retain_graph=False,
                create_graph=False,
                inputs=None,
            )

        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "implicitly created only for scalar"
        ):
            leaf.backward(None, None, False, None)
        self.assertIsNone(leaf.grad)
        leaf.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])

    def test_accumulation_graph_reuse_and_freeing_are_preserved(self):
        reusable_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        reusable_loss = reusable_leaf.transpose(0, 0).sum()
        self.assertIsNone(reusable_loss.backward(retain_graph=False))
        self.assertIsNone(
            reusable_loss.backward(
                gradient=None,
                retain_graph=None,
                create_graph=False,
                inputs=None,
            )
        )
        self.assertEqual(reusable_leaf.grad.tolist(), [2.0, 2.0])

        scalar_leaf = torch.tensor(7.0, requires_grad=True)
        scalar_leaf.backward(None, False, False, None)
        scalar_leaf.backward()
        self.assertEqual(scalar_leaf.grad.item(), 2.0)

        freed_leaf = torch.tensor([2.0, 3.0], requires_grad=True)
        freed_loss = (freed_leaf * freed_leaf).sum()
        freed_loss.backward(gradient=None, retain_graph=None)
        self.assertEqual(freed_leaf.grad.tolist(), [4.0, 6.0])
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            freed_loss.backward(None, False, False, None)

        (freed_leaf * freed_leaf).sum().backward(inputs=None)
        self.assertEqual(freed_leaf.grad.tolist(), [8.0, 12.0])

    def test_unsupported_options_fail_before_gradients_or_graph_state_change(self):
        unsupported = (
            (
                "tensor gradient",
                "torch_rs.Tensor.backward does not support explicit gradients",
                lambda leaf, loss: loss.backward(torch.tensor(1.0)),
            ),
            (
                "non-tensor gradient",
                "torch_rs.Tensor.backward does not support explicit gradients",
                lambda leaf, loss: loss.backward(gradient=object()),
            ),
            (
                "retained graph",
                "torch_rs.Tensor.backward does not support retain_graph=True",
                lambda leaf, loss: loss.backward(retain_graph=True),
            ),
            (
                "integer retained graph",
                "torch_rs.Tensor.backward does not support retain_graph=True",
                lambda leaf, loss: loss.backward(retain_graph=IndexValue(1)),
            ),
            (
                "higher-order graph",
                "torch_rs.Tensor.backward does not support create_graph=True",
                lambda leaf, loss: loss.backward(create_graph=True),
            ),
            (
                "integer higher-order graph",
                "torch_rs.Tensor.backward does not support create_graph=True",
                lambda leaf, loss: loss.backward(create_graph=1),
            ),
            (
                "tensor inputs",
                "torch_rs.Tensor.backward does not support inputs",
                lambda leaf, loss: loss.backward(inputs=leaf),
            ),
            (
                "sequence inputs",
                "torch_rs.Tensor.backward does not support inputs",
                lambda leaf, loss: loss.backward(inputs=[leaf]),
            ),
        )

        for label, message, call in unsupported:
            with self.subTest(label=label):
                leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                leaf.sum().backward()
                loss = (leaf * leaf).sum()
                with self.assertRaisesRegex(
                    NotImplementedError, f"^{re.escape(message)}$"
                ):
                    call(leaf, loss)
                self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])

                loss.backward()
                self.assertEqual(leaf.grad.tolist(), [5.0, 7.0])

    def test_option_conversion_errors_are_non_mutating(self):
        for name, value in (("retain_graph", 0.5), ("create_graph", None)):
            with self.subTest(name=name):
                leaf = torch.tensor(2.0, requires_grad=True)
                loss = leaf * leaf
                with self.assertRaises(TypeError) as raised:
                    loss.backward(**{name: value})
                self.assertEqual(
                    str(raised.exception),
                    f"'{type(value).__name__}' object cannot be interpreted "
                    "as an integer",
                )
                self.assertIsNone(leaf.grad)
                loss.backward()
                self.assertEqual(leaf.grad.item(), 4.0)

    def test_python_metadata_matches_the_public_method_shape(self):
        tensor = torch.tensor(1.0, requires_grad=True)
        function = inspect.getattr_static(torch.Tensor, "backward")
        bound = tensor.backward

        self.assertIs(type(function), types.FunctionType)
        self.assertIs(type(bound), types.MethodType)
        self.assertRegex(
            repr(function), r"^<function Tensor\.backward at 0x[0-9a-f]+>$"
        )
        self.assertEqual(function.__name__, "backward")
        self.assertEqual(function.__qualname__, "Tensor.backward")
        self.assertEqual(function.__module__, "torch_rs._tensor")
        self.assertEqual(bound.__name__, "backward")
        self.assertEqual(bound.__qualname__, "Tensor.backward")
        self.assertEqual(bound.__module__, "torch_rs._tensor")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(bound.__annotations__, {})
        self.assertEqual(function.__defaults__, (None, None, False, None))
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertFalse(hasattr(bound, "__text_signature__"))
        self.assertEqual(
            str(inspect.signature(function)),
            "(self, gradient=None, retain_graph=None, create_graph=False, "
            "inputs=None)",
        )
        self.assertEqual(
            str(inspect.signature(bound)),
            "(gradient=None, retain_graph=None, create_graph=False, inputs=None)",
        )
        self.assertIn("Computes the gradient", function.__doc__)
        self.assertEqual(bound.__doc__, function.__doc__)
        self.assertIn("backward", torch.Tensor.__dict__)
        self.assertTrue(
            all(
                "backward" not in owner.__dict__
                for owner in torch.Tensor.__mro__[1:]
            )
        )
        self.assertIs(torch._tensor.Tensor, torch.Tensor)
        self.assertIs(torch._tensor.Tensor.backward, function)
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
        function = inspect.getattr_static(torch.Tensor, "backward")
        calls = (
            lambda: function(),
            lambda: loss.backward(None, gradient=None),
            lambda: loss.backward(unexpected=True),
            lambda: loss.backward(None, None, False, None, None),
        )
        for case, call in enumerate(calls):
            with self.subTest(case=case), self.assertRaises(TypeError):
                call()
        self.assertIsNone(leaf.grad)
        loss.backward()
        self.assertEqual(leaf.grad.item(), 4.0)

    def test_torch_function_mode_dispatch_remains_unsupported(self):
        leaf = torch.tensor(2.0, requires_grad=True)
        loss = leaf * leaf

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        mode = RecordingMode()
        with mode:
            result = loss.backward(
                gradient=None,
                retain_graph=None,
                create_graph=False,
                inputs=None,
            )
        self.assertIsNone(result)
        self.assertEqual(mode.calls, [])
        self.assertEqual(leaf.grad.item(), 4.0)


if __name__ == "__main__":
    unittest.main()
