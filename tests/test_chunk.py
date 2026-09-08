import inspect
import pickle
import re
import types
import unittest

import numpy as np

import torch_rs as torch


METHOD_DOC = "\nchunk(chunks, dim=0) -> List of Tensors\n\nSee :func:`torch.chunk`\n"
FUNCTION_DOC = (
    "\nchunk(input: Tensor, chunks: int, dim: int = 0) -> Tuple[Tensor, ...]\n\n"
    "Attempts to split a tensor into the specified number of chunks. Each chunk is a view of\n"
    "the input tensor.\n\n\n"
    ".. note::\n\n"
    "    This function may return fewer than the specified number of chunks!\n\n"
    ".. seealso::\n\n"
    "    :func:`torch.tensor_split` a function that always returns exactly the specified number of chunks\n\n"
    "If the tensor size along the given dimension :attr:`dim` is divisible by :attr:`chunks`,\n"
    "all returned chunks will be the same size.\n"
    "If the tensor size along the given dimension :attr:`dim` is not divisible by :attr:`chunks`,\n"
    "all returned chunks will be the same size, except the last one.\n"
    "If such division is not possible, this function may return fewer\n"
    "than the specified number of chunks.\n\n"
    "Arguments:\n"
    "    input (Tensor): the tensor to split\n"
    "    chunks (int): number of chunks to return\n"
    "    dim (int): dimension along which to split the tensor\n\n"
    "Example:\n"
    "    >>> torch.arange(11).chunk(6)\n"
    "    (tensor([0, 1]),\n"
    "     tensor([2, 3]),\n"
    "     tensor([4, 5]),\n"
    "     tensor([6, 7]),\n"
    "     tensor([8, 9]),\n"
    "     tensor([10]))\n"
    "    >>> torch.arange(12).chunk(6)\n"
    "    (tensor([0, 1]),\n"
    "     tensor([2, 3]),\n"
    "     tensor([4, 5]),\n"
    "     tensor([6, 7]),\n"
    "     tensor([8, 9]),\n"
    "     tensor([10, 11]))\n"
    "    >>> torch.arange(13).chunk(6)\n"
    "    (tensor([0, 1, 2]),\n"
    "     tensor([3, 4, 5]),\n"
    "     tensor([6, 7, 8]),\n"
    "     tensor([ 9, 10, 11]),\n"
    "     tensor([12]))\n"
)


def offset_noncontiguous_source(*, requires_grad=False):
    values = [float(value) for value in range(120)]
    return torch.tensor(values, requires_grad=requires_grad).reshape(2, 3, 4, 5)[
        1
    ].transpose(0, 1)


def chunk_lengths(size, chunks):
    if size == 0:
        return (0,) * chunks
    chunk_size = (size + chunks - 1) // chunks
    return tuple(
        min(chunk_size, size - start) for start in range(0, size, chunk_size)
    )


class TensorChunkTests(unittest.TestCase):
    def assert_chunk_outputs_match_narrows(self, source, outputs, chunks, dimension=0):
        axis = dimension + source.dim() if dimension < 0 else dimension
        lengths = chunk_lengths(source.shape[axis], chunks)
        self.assertIs(type(outputs), tuple)
        self.assertEqual(len(outputs), len(lengths))
        start = 0
        for index, (output, length) in enumerate(zip(outputs, lengths, strict=True)):
            direct = source.narrow(axis, start, length)
            with self.subTest(index=index):
                self.assertEqual(output.tolist(), direct.tolist())
                self.assertEqual(output.shape, direct.shape)
                self.assertEqual(output.stride(), direct.stride())
                self.assertEqual(output.storage_offset(), direct.storage_offset())
                self.assertEqual(output.data_ptr(), direct.data_ptr())
                self.assertTrue(output.is_set_to(direct))
                self.assertIs(output.dtype, source.dtype)
                self.assertEqual(output.device, source.device)
            start += length

    def test_method_and_top_level_chunk_return_shared_narrow_views(self):
        contiguous = torch.tensor([float(value) for value in range(24)]).reshape(
            2, 3, 4
        )
        transposed = contiguous.transpose(0, 2)
        offset = offset_noncontiguous_source()
        empty_middle = torch.zeros((2, 0, 3))

        cases = (
            ("contiguous uneven", contiguous, 3, 1),
            ("transposed uneven", transposed, 3, 0),
            ("offset noncontiguous", offset, 2, 1),
            ("empty dimension", empty_middle, 3, 1),
            ("negative dimension", contiguous, 2, -1),
            ("too many chunks", contiguous, 5, 0),
        )
        for case, source, chunks, dimension in cases:
            with self.subTest(case=case, surface="method"):
                self.assert_chunk_outputs_match_narrows(
                    source, source.chunk(chunks, dimension), chunks, dimension
                )
            with self.subTest(case=case, surface="top-level"):
                self.assert_chunk_outputs_match_narrows(
                    source, torch.chunk(source, chunks, dimension), chunks, dimension
                )

    def test_call_forms_input_aliases_and_integer_protocol(self):
        source = offset_noncontiguous_source()
        calls = (
            ("method default", lambda: source.chunk(2)),
            ("method positional", lambda: source.chunk(2, 1)),
            ("method keyword", lambda: source.chunk(chunks=2, dim=1)),
            ("method reordered", lambda: source.chunk(dim=1, chunks=2)),
            ("top default", lambda: torch.chunk(source, 2)),
            ("top positional", lambda: torch.chunk(source, 2, 1)),
            ("top mixed", lambda: torch.chunk(source, chunks=2, dim=1)),
            ("top keywords", lambda: torch.chunk(input=source, chunks=2, dim=1)),
            ("top input alias x", lambda: torch.chunk(x=source, chunks=2, dim=1)),
            ("top input alias a", lambda: torch.chunk(a=source, chunks=2, dim=1)),
            ("top input alias x1", lambda: torch.chunk(x1=source, chunks=2, dim=1)),
        )
        for case, call in calls:
            with self.subTest(case=case):
                dimension = 0 if "default" in case else 1
                self.assert_chunk_outputs_match_narrows(source, call(), 2, dimension)

        class IntegerSubclass(int):
            pass

        self.assertEqual(len(source.chunk(np.int64(2), IntegerSubclass(1))), 2)
        self.assertEqual(len(torch.chunk(source, np.uint32(2), IntegerSubclass(1))), 2)

        conversion_order = []

        class ChunkCount:
            def __index__(self):
                conversion_order.append("chunks")
                return 2

        self.assertEqual(len(source.chunk(ChunkCount(), 1)), 2)
        self.assertEqual(conversion_order, ["chunks", "chunks", "chunks"])

    def test_backward_through_sum_no_grad_and_empty_dimensions(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = torch.tensor(values.reshape(-1).tolist(), requires_grad=True)
        source = (leaf * 2.0).reshape(2, 3, 4).transpose(0, 1)
        chunks = source.chunk(2, 0)

        self.assertEqual(tuple(chunk.output_nr for chunk in chunks), (0, 1))
        self.assertTrue(all(chunk.requires_grad for chunk in chunks))
        self.assertTrue(all(not chunk.is_leaf for chunk in chunks))
        chunks[1].sum().backward()

        expected_gradient = np.zeros_like(values)
        expected_gradient[:, 2:3, :] = 2.0
        np.testing.assert_array_equal(
            np.asarray(leaf.grad).reshape(values.shape),
            expected_gradient,
        )

        top_level_leaf = torch.tensor(values.reshape(-1).tolist(), requires_grad=True)
        top_level_source = (top_level_leaf * 2.0).reshape(2, 3, 4)
        top_level_chunks = torch.chunk(top_level_source, 2, 1)
        top_level_chunks[1].sum().backward()
        expected_top_level_gradient = np.zeros_like(values)
        expected_top_level_gradient[:, 2:3, :] = 2.0
        np.testing.assert_array_equal(
            np.asarray(top_level_leaf.grad).reshape(values.shape),
            expected_top_level_gradient,
        )

        combined_leaf = torch.tensor(values.reshape(-1).tolist(), requires_grad=True)
        combined_source = (combined_leaf * 2.0).reshape(2, 3, 4)
        first, second = combined_source.chunk(2, 1)
        (first.sum() + second.sum()).backward()
        np.testing.assert_array_equal(
            np.asarray(combined_leaf.grad).reshape(values.shape),
            np.full_like(values, 2.0),
        )

        no_grad_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        with torch.no_grad():
            no_grad_chunks = no_grad_source.chunk(2)
        self.assertEqual(tuple(chunk.output_nr for chunk in no_grad_chunks), (0, 0))
        self.assertTrue(all(chunk.requires_grad for chunk in no_grad_chunks))
        self.assertTrue(all(chunk.is_leaf for chunk in no_grad_chunks))

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty_chunks = empty.chunk(3, 1)
        self.assertEqual(tuple(chunk.output_nr for chunk in empty_chunks), (0, 1, 2))
        self.assertEqual(
            tuple(chunk.shape for chunk in empty_chunks),
            ((2, 0, 3), (2, 0, 3), (2, 0, 3)),
        )
        empty_chunks[2].sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])

    def test_errors_and_surface_limits(self):
        tensor = torch.zeros((2, 3))
        scalar = torch.tensor(1.0)
        cases = (
            (
                lambda: tensor.chunk(),
                TypeError,
                'chunk() missing 1 required positional arguments: "chunks"',
            ),
            (
                lambda: tensor.chunk(2, 0, 0),
                TypeError,
                "chunk() takes from 1 to 2 positional arguments but 3 were given",
            ),
            (
                lambda: tensor.chunk(2, chunks=2),
                TypeError,
                "chunk() got multiple values for argument 'chunks'",
            ),
            (
                lambda: tensor.chunk(2, extra=0),
                TypeError,
                "chunk() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.chunk(0),
                RuntimeError,
                "chunk expects `chunks` to be greater than 0, got: 0",
            ),
            (
                lambda: tensor.chunk(-1),
                RuntimeError,
                "chunk expects `chunks` to be greater than 0, got: -1",
            ),
            (
                lambda: torch.chunk(tensor, 0),
                RuntimeError,
                "chunk expects `chunks` to be greater than 0, got: 0",
            ),
            (
                lambda: tensor.chunk(True),
                TypeError,
                "chunk(): argument 'chunks' (position 1) must be int, not bool",
            ),
            (
                lambda: torch.chunk(tensor, True),
                TypeError,
                "chunk(): argument 'chunks' (position 2) must be int, not bool",
            ),
            (
                lambda: tensor.chunk(2, True),
                TypeError,
                "chunk(): argument 'dim' (position 2) must be int, not bool",
            ),
            (
                lambda: tensor.chunk(2, 2),
                IndexError,
                "Dimension out of range (expected to be in range of [-2, 1], but got 2)",
            ),
            (
                lambda: scalar.chunk(1),
                RuntimeError,
                "chunk expects at least a 1-dimensional tensor",
            ),
            (
                lambda: scalar.chunk(0),
                RuntimeError,
                "chunk expects at least a 1-dimensional tensor",
            ),
            (
                lambda: torch.chunk([1.0], 2),
                TypeError,
                "chunk(): argument 'input' (position 1) must be Tensor, not list",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message), self.assertRaises(error_type) as raised:
                call()
            self.assertEqual(str(raised.exception), message)

        for call in (
            lambda: tensor.chunk(2**100),
            lambda: tensor.chunk(2, 2**100),
            lambda: torch.chunk(tensor, 2**100),
            lambda: torch.chunk(tensor, 2, 2**100),
        ):
            with self.assertRaisesRegex(ValueError, "^Overflow when unpacking long long$"):
                call()

    def test_callable_metadata_exports_pickle_and_torch_function_modes(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "chunk")
        bound = tensor.chunk

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "chunk")
        self.assertEqual(bound.__name__, "chunk")
        self.assertEqual(descriptor.__qualname__, "TensorBase.chunk")
        self.assertEqual(bound.__qualname__, "Tensor.chunk")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)
        with self.assertRaises(ValueError):
            inspect.signature(bound)

        function = torch.chunk
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "chunk")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.chunk")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertIn("chunk", torch.__all__)
        self.assertIs(pickle.loads(pickle.dumps(function)), function)

        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            self.assertIs(tensor.chunk(2, dim=1), marker)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor, 2))
        self.assertEqual(kwargs, {"dim": 1})

        top_level_mode = RecordingMode(marker)
        with top_level_mode:
            self.assertIs(torch.chunk(input=tensor, chunks=2, dim=1), marker)
        function, dispatch_types, args, kwargs = top_level_mode.calls[0]
        self.assertIs(function, torch.chunk)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "chunks": 2, "dim": 1})


if __name__ == "__main__":
    unittest.main()
