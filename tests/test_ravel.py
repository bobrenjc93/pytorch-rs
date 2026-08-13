import gc
import inspect
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorRavelTests(unittest.TestCase):
    def assert_tensor(self, tensor, values, *, shape, stride, offset=0):
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.stride(), stride)
        self.assertEqual(tensor.storage_offset(), offset)
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        np.testing.assert_array_equal(
            np.asarray(tensor), np.asarray(values, dtype=np.float32).reshape(shape)
        )

    def test_scalar_vector_ordinary_and_empty_inputs_return_new_vectors(self):
        cases = (
            (torch.tensor(-0.0), [-0.0], (1,)),
            (torch.tensor([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0], (1,)),
            (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                [1.0, 2.0, 3.0, 4.0],
                (1,),
            ),
            (torch.zeros((2, 0, 3)), [], (1,)),
        )
        for source, values, stride in cases:
            with self.subTest(shape=source.shape):
                output = source.ravel()
                self.assertIsNot(output, source)
                self.assert_tensor(
                    output,
                    values,
                    shape=(source.numel(),),
                    stride=stride,
                    offset=source.storage_offset(),
                )

        scalar_bits = np.asarray(cases[0][0].ravel()).view(np.uint32).item()
        self.assertEqual(scalar_bits, 0x8000_0000)

    def test_contiguous_offsets_alias_and_strided_inputs_materialize(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = torch.tensor(values.tolist())

        offset_matrix = source[1]
        offset_ravel = offset_matrix.ravel()
        self.assertIsNot(offset_ravel, offset_matrix)
        self.assert_tensor(
            offset_ravel,
            values[1].reshape(-1),
            shape=(12,),
            stride=(1,),
            offset=12,
        )

        strided_vector = source.transpose(0, 2)[0][0]
        self.assertEqual(strided_vector.stride(), (12,))
        packed_vector = strided_vector.ravel()
        self.assert_tensor(
            packed_vector,
            values[:, 0, 0],
            shape=(2,),
            stride=(1,),
        )

        transposed = source.transpose(0, 2)
        packed = transposed.ravel()
        self.assert_tensor(
            packed,
            values.transpose(2, 1, 0).reshape(-1),
            shape=(24,),
            stride=(1,),
        )

        singleton_source = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        singleton = singleton_source.transpose(0, 1)[2]
        self.assertEqual(singleton.stride(), (4,))
        singleton_ravel = singleton.ravel()
        self.assert_tensor(
            singleton_ravel,
            [2.0],
            shape=(1,),
            stride=(4,),
            offset=2,
        )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        empty_ravel = empty.ravel()
        self.assert_tensor(
            empty_ravel,
            [],
            shape=(0,),
            stride=(1,),
            offset=1,
        )

        def ravel_after_source_drops():
            temporary = torch.tensor(values.tolist())
            return temporary.transpose(0, 2).ravel()

        surviving_copy = ravel_after_source_drops()
        gc.collect()
        np.testing.assert_array_equal(
            np.asarray(surviving_copy), values.transpose(2, 1, 0).reshape(-1)
        )

    def test_autograd_and_no_grad_follow_view_or_copy_behavior(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        output = leaf.transpose(0, 1).ravel()
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        (output * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]],
        )

        scalar = torch.tensor(2.0, requires_grad=True)
        (scalar.ravel() * 7.0).sum().backward()
        self.assertEqual(scalar.grad.item(), 7.0)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty.ravel().sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))

        no_grad_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        non_contiguous = no_grad_source.transpose(0, 1)
        with torch.no_grad():
            alias = no_grad_source.ravel()
            copied = non_contiguous.ravel()
        self.assertTrue(alias.requires_grad)
        self.assertTrue(alias.is_leaf)
        self.assertFalse(copied.requires_grad)
        self.assertTrue(copied.is_leaf)

    def test_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "ravel")
        bound = tensor.ravel
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "ravel")
        self.assertEqual(bound.__name__, "ravel")
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertEqual(
            descriptor.__doc__, "\nravel() -> Tensor\n\nsee :func:`torch.ravel`\n"
        )
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        output = descriptor(tensor)
        self.assertIsNot(output, tensor)
        self.assertEqual(output.shape, (6,))

        calls = (
            (lambda: tensor.ravel(1), "Tensor.ravel() takes no arguments (1 given)"),
            (lambda: tensor.ravel(1, 2), "Tensor.ravel() takes no arguments (2 given)"),
            (lambda: tensor.ravel(dim=0), "Tensor.ravel() takes no keyword arguments"),
            (lambda: descriptor(), "unbound method Tensor.ravel() needs an argument"),
            (
                lambda: descriptor(tensor, 1),
                "Tensor.ravel() takes no arguments (1 given)",
            ),
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        with self.assertRaises(TypeError):
            descriptor([1.0])


if __name__ == "__main__":
    unittest.main()
