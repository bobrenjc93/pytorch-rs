import inspect
import sys
import types
import unittest

import torch_rs as torch


class TensorLayoutTests(unittest.TestCase):
    def test_strided_is_the_canonical_layout_singleton(self):
        self.assertIs(type(torch.strided), torch.layout)
        self.assertIsInstance(torch.strided, torch.layout)
        self.assertIs(torch.strided, torch.strided)
        self.assertEqual(repr(torch.strided), "torch.strided")
        self.assertEqual(str(torch.strided), "torch.strided")

        self.assertEqual(torch.layout.__module__, "torch_rs")
        self.assertEqual(torch.layout.__name__, "layout")
        self.assertEqual(torch.layout.__qualname__, "layout")
        self.assertIsNone(torch.layout.__doc__)
        self.assertIn("layout", torch.__all__)
        self.assertIn("strided", torch.__all__)

        for arguments in ((), ("strided",), (1, 2)):
            with self.subTest(arguments=arguments):
                with self.assertRaises(TypeError) as raised:
                    torch.layout(*arguments)
                self.assertEqual(
                    str(raised.exception),
                    "cannot create 'torch_rs.layout' instances",
                )

        assignments = (
            (lambda: setattr(torch.strided, "label", "strided"), "'label'"),
            (lambda: delattr(torch.strided, "label"), "'label'"),
            (lambda: setattr(torch.strided, "__doc__", "layout"), "'__doc__'"),
            (lambda: delattr(torch.strided, "__doc__"), "'__doc__'"),
        )
        for action, attribute in assignments:
            with self.subTest(attribute=attribute):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertIn("'torch_rs.layout' object", str(raised.exception))
                self.assertIn(attribute, str(raised.exception))

    def test_layout_uses_identity_equality_and_hashing(self):
        layout = torch.strided
        self.assertIs(layout == layout, True)
        self.assertIs(layout != layout, False)
        self.assertIs(torch.layout.__eq__(layout, layout), True)
        self.assertIs(torch.layout.__ne__(layout, layout), False)
        self.assertIs(type(hash(layout)), int)
        self.assertEqual(hash(layout), object.__hash__(layout))
        self.assertEqual(hash(layout), hash(torch.strided))
        self.assertEqual({layout: "strided"}[torch.strided], "strided")
        self.assertEqual({layout}, {torch.strided})

        for other in (
            None,
            0,
            "torch.strided",
            torch.float32,
            torch.contiguous_format,
            object(),
        ):
            with self.subTest(other=other):
                self.assertIs(layout == other, False)
                self.assertIs(layout != other, True)
                self.assertIs(torch.layout.__eq__(layout, other), NotImplemented)
                self.assertIs(torch.layout.__ne__(layout, other), NotImplemented)

    def test_scalar_empty_views_and_autograd_return_one_singleton(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        offset_view = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        ).transpose(0, 1)[1]
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        extreme_offset_empty = torch.zeros((sys.maxsize, 0))[sys.maxsize - 1]

        cases = (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3))),
            ("offset strided view", offset_view),
            ("extreme empty view", extreme_empty),
            ("extreme offset empty", extreme_offset_empty),
            ("autograd leaf", leaf),
            ("autograd non-leaf view", tracked),
            ("detached view", tracked.detach()),
        )
        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.dtype,
                    tensor.device,
                    tensor.requires_grad,
                    tensor.is_leaf,
                )
                for _ in range(3):
                    self.assertIs(tensor.layout, torch.strided)
                self.assertEqual(
                    (
                        tensor.shape,
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.dtype,
                        tensor.device,
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                    metadata,
                )

        self.assertIsNone(leaf.grad)
        self.assertTrue(leaf.is_leaf)
        self.assertFalse(tracked.is_leaf)
        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        self.assertIs(leaf.grad.layout, torch.strided)

    def test_tensorbase_layout_descriptor_is_read_only(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "layout")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "layout")
        self.assertEqual(descriptor.__qualname__, "TensorBase.layout")
        self.assertIsNone(descriptor.__doc__)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertIs(torch.Tensor.layout, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(descriptor.__get__(tensor, torch.Tensor), torch.strided)

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'layout' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

        actions = (
            lambda: setattr(tensor, "layout", torch.strided),
            lambda: delattr(tensor, "layout"),
            lambda: descriptor.__set__(tensor, torch.strided),
            lambda: descriptor.__delete__(tensor),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'layout' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )


if __name__ == "__main__":
    unittest.main()
