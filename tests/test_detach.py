import inspect
import re
import types
import unittest

import torch_rs as torch


class TopLevelDetachTests(unittest.TestCase):
    def assert_detached(self, source, result):
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.stride(), source.stride())
        self.assertEqual(result.storage_offset(), source.storage_offset())
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        self.assertEqual(result.tolist(), source.tolist())
        self.assertFalse(result.requires_grad)
        self.assertFalse((result + 1.0).requires_grad)

    def test_positional_and_keyword_calls_preserve_layout_metadata(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        cases = (
            torch.tensor(3.0, requires_grad=True),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True),
            (leaf * 2.0).transpose(0, 1)[1],
            torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1],
        )

        for case, source in enumerate(cases):
            with self.subTest(case=case, call="positional"):
                self.assert_detached(source, torch.detach(source))
            with self.subTest(case=case, call="keyword"):
                self.assert_detached(source, torch.detach(input=source))
            self.assertTrue(source.requires_grad)

    def test_detach_does_not_consume_or_modify_the_source_graph(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        source = (leaf * 3.0).transpose(0, 1)[1]
        detached = torch.detach(source)

        self.assertTrue(source.requires_grad)
        self.assertFalse(detached.requires_grad)
        detached_loss = (detached * detached).sum()
        self.assertFalse(detached_loss.requires_grad)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            detached_loss.backward()

        source.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0], [0.0, 3.0]])

    def test_callable_metadata(self):
        self.assertIs(type(torch.detach), types.BuiltinFunctionType)
        self.assertTrue(callable(torch.detach))
        self.assertEqual(torch.detach.__name__, "detach")
        self.assertIsNone(torch.detach.__text_signature__)
        self.assertIsNone(torch.detach.__doc__)
        with self.assertRaises(ValueError):
            inspect.signature(torch.detach)
        self.assertIn("detach", torch.__all__)

    def test_binding_and_non_tensor_errors(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.detach(),
                'detach() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.detach(tensor, tensor),
                "detach() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.detach(tensor, input=tensor),
                "detach() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.detach(foo=tensor),
                'detach() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.detach(tensor, extra=True),
                "detach() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.detach(None),
                "detach(): argument 'input' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: torch.detach(input=1),
                "detach(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.detach([]),
                "detach(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.detach(1, extra=True),
                "detach(): argument 'input' (position 1) must be Tensor, not int",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
