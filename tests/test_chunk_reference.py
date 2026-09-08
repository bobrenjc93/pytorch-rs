import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorChunkReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("chunk differentials require pinned PyTorch 2.13.0")

    def offset_noncontiguous_source(self, module, *, requires_grad=False):
        values = [float(value) for value in range(120)]
        return module.tensor(values, dtype=module.float32, requires_grad=requires_grad).reshape(
            2, 3, 4, 5
        )[1].transpose(0, 1)

    def layout_cases(self, module):
        contiguous = module.tensor(
            [float(value) for value in range(24)], dtype=module.float32
        ).reshape(2, 3, 4)
        transposed = contiguous.transpose(0, 2)
        offset = self.offset_noncontiguous_source(module)
        empty_middle = module.zeros((2, 0, 3), dtype=module.float32)
        return (
            ("contiguous uneven", contiguous, 3, 1),
            ("transposed uneven", transposed, 3, 0),
            ("offset noncontiguous", offset, 2, 1),
            ("empty dimension", empty_middle, 3, 1),
            ("negative dimension", contiguous, 2, -1),
            ("too many chunks", contiguous, 5, 0),
        )

    def output_contract(self, tensor, outputs, chunks, dimension=0):
        axis = dimension + tensor.dim() if dimension < 0 else dimension
        start = 0
        output_details = []
        for output in outputs:
            length = output.shape[axis]
            direct = tensor.narrow(axis, start, length)
            source_ptr = tensor.data_ptr()
            output_ptr = output.data_ptr()
            output_details.append(
                (
                    output.tolist(),
                    tuple(output.shape),
                    output.stride(),
                    output.storage_offset(),
                    None
                    if source_ptr == 0 or output_ptr == 0
                    else output_ptr - source_ptr,
                    output.data_ptr() == direct.data_ptr(),
                    output.is_set_to(direct),
                    output.output_nr,
                    output.requires_grad,
                    output.is_leaf,
                    str(output.dtype),
                    str(output.device),
                )
            )
            start += length
        return {
            "type": type(outputs).__name__,
            "requested_chunks": chunks,
            "source": (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                str(tensor.dtype),
                str(tensor.device),
            ),
            "outputs": tuple(output_details),
        }

    def call_method(self, tensor, chunks, dimension, form):
        if form == "default":
            return tensor.chunk(chunks)
        if form == "positional":
            return tensor.chunk(chunks, dimension)
        if form == "keyword":
            return tensor.chunk(chunks=chunks, dim=dimension)
        if form == "reordered":
            return tensor.chunk(dim=dimension, chunks=chunks)
        raise AssertionError(f"unknown method form: {form}")

    def call_top_level(self, module, tensor, chunks, dimension, form):
        if form == "default":
            return module.chunk(tensor, chunks)
        if form == "positional":
            return module.chunk(tensor, chunks, dimension)
        if form == "mixed":
            return module.chunk(tensor, chunks=chunks, dim=dimension)
        if form == "keyword":
            return module.chunk(input=tensor, chunks=chunks, dim=dimension)
        if form == "alias":
            return module.chunk(x=tensor, chunks=chunks, dim=dimension)
        raise AssertionError(f"unknown top-level form: {form}")

    def test_values_layout_offsets_aliasing_and_counts_match_pytorch_2_13(self):
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for (case, actual, chunks, dimension), (
            expected_case,
            expected,
            expected_chunks,
            expected_dimension,
        ) in zip(actual_cases, expected_cases, strict=True):
            self.assertEqual(case, expected_case)
            self.assertEqual(chunks, expected_chunks)
            self.assertEqual(dimension, expected_dimension)
            for form in ("default", "positional", "keyword", "reordered"):
                if form == "default" and dimension != 0:
                    continue
                with self.subTest(case=case, surface="method", form=form):
                    self.assertEqual(
                        self.output_contract(
                            actual,
                            self.call_method(actual, chunks, dimension, form),
                            chunks,
                            0 if form == "default" else dimension,
                        ),
                        self.output_contract(
                            expected,
                            self.call_method(expected, chunks, dimension, form),
                            expected_chunks,
                            0 if form == "default" else expected_dimension,
                        ),
                    )
            for form in ("default", "positional", "mixed", "keyword", "alias"):
                if form == "default" and dimension != 0:
                    continue
                with self.subTest(case=case, surface="top-level", form=form):
                    self.assertEqual(
                        self.output_contract(
                            actual,
                            self.call_top_level(torch, actual, chunks, dimension, form),
                            chunks,
                            0 if form == "default" else dimension,
                        ),
                        self.output_contract(
                            expected,
                            self.call_top_level(
                                reference_torch, expected, chunks, dimension, form
                            ),
                            expected_chunks,
                            0 if form == "default" else expected_dimension,
                        ),
                    )

    def autograd_contract(self, module, *, top_level=False):
        call = (
            (lambda input, chunks, dim: module.chunk(input, chunks, dim))
            if top_level
            else lambda input, chunks, dim: input.chunk(chunks, dim)
        )
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.reshape(-1).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        source = (leaf * 2.0).reshape(2, 3, 4).transpose(0, 1)
        outputs = call(source, 2, 0)
        output_metadata = tuple(
            (
                output.output_nr,
                output.requires_grad,
                output.is_leaf,
                tuple(output.shape),
                output.stride(),
                output.storage_offset(),
                output.is_set_to(source.narrow(0, 2, 1)),
            )
            for output in outputs[-1:]
        )
        outputs[-1].sum().backward()

        combined_leaf = module.tensor(
            values.reshape(-1).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        combined_source = (combined_leaf * 2.0).reshape(2, 3, 4)
        first, second = call(combined_source, 2, 1)
        (first.sum() + second.sum()).backward()

        no_grad_source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        with module.no_grad():
            no_grad_outputs = call(no_grad_source, 2, 0)

        empty = module.zeros((2, 0, 3), dtype=module.float32, requires_grad=True)
        empty_outputs = call(empty, 3, 1)
        empty_metadata = tuple(
            (
                output.output_nr,
                output.requires_grad,
                output.is_leaf,
                tuple(output.shape),
                output.stride(),
                output.storage_offset(),
                output.is_set_to(empty.narrow(1, 0, 0)),
            )
            for output in empty_outputs
        )
        empty_outputs[-1].sum().backward()

        return {
            "output_metadata": output_metadata,
            "gradient": leaf.grad.tolist(),
            "combined_gradient": combined_leaf.grad.tolist(),
            "no_grad": tuple(
                (
                    output.output_nr,
                    output.requires_grad,
                    output.is_leaf,
                )
                for output in no_grad_outputs
            ),
            "empty_metadata": empty_metadata,
            "empty_gradient_shape": tuple(empty.grad.shape),
            "empty_gradient": empty.grad.tolist(),
        }

    def test_backward_through_sum_no_grad_and_empty_match_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )

    def test_top_level_backward_through_sum_no_grad_and_empty_match_pytorch_2_13(
        self,
    ):
        self.assertEqual(
            self.autograd_contract(torch, top_level=True),
            self.autograd_contract(reference_torch, top_level=True),
        )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("chunk unexpectedly accepted an invalid call")

    def error_contract(self, module):
        tensor = module.zeros((2, 3), dtype=module.float32)
        scalar = module.tensor(1.0, dtype=module.float32)
        return (
            self.error(lambda: tensor.chunk()),
            self.error(lambda: tensor.chunk(2, 0, 0)),
            self.error(lambda: tensor.chunk(2, chunks=2)),
            self.error(lambda: tensor.chunk(2, extra=0)),
            self.error(lambda: tensor.chunk(0)),
            self.error(lambda: tensor.chunk(-1)),
            self.error(lambda: tensor.chunk(True)),
            self.error(lambda: tensor.chunk(2, True)),
            self.error(lambda: tensor.chunk(2**100)),
            self.error(lambda: tensor.chunk(2, 2**100)),
            self.error(lambda: tensor.chunk(2, 2)),
            self.error(lambda: scalar.chunk(1)),
            self.error(lambda: scalar.chunk(0)),
            self.error(lambda: module.chunk()),
            self.error(lambda: module.chunk(chunks=2)),
            self.error(lambda: module.chunk(tensor)),
            self.error(lambda: module.chunk(tensor, 2, 0, 0)),
            self.error(lambda: module.chunk(tensor, 2, input=tensor)),
            self.error(lambda: module.chunk(tensor, 2, extra=0)),
            self.error(lambda: module.chunk(x=tensor, chunks=2, extra=0)),
            self.error(lambda: module.chunk([], 2)),
            self.error(lambda: module.chunk(tensor, 0)),
            self.error(lambda: module.chunk(tensor, True)),
            self.error(lambda: module.chunk(tensor, 2, True)),
            self.error(lambda: module.chunk(tensor, 2**100)),
            self.error(lambda: module.chunk(tensor, 2, 2**100)),
            self.error(lambda: module.chunk(tensor, 2, 2)),
            self.error(lambda: module.chunk(scalar, 1)),
            self.error(lambda: module.chunk(scalar, 0)),
            len(tensor.chunk(np.int64(2))),
            len(module.chunk(tensor, np.uint32(2))),
        )

    def test_errors_and_integer_arguments_match_pytorch_2_13(self):
        self.assertEqual(self.error_contract(torch), self.error_contract(reference_torch))

    def descriptor_contract(self, module):
        tensor = module.zeros((2, 3), dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "chunk")
        bound = tensor.chunk

        def signature_error(callable_object):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                return type(error).__name__
            self.fail(f"{module.__name__} exposed an inspectable chunk signature")

        return {
            "descriptor_type": type(descriptor).__name__,
            "is_method_descriptor": type(descriptor) is types.MethodDescriptorType,
            "bound_type": type(bound).__name__,
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "repr": repr(descriptor),
            "class_identity": module.Tensor.chunk is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor) is descriptor,
            "descriptor_signature": signature_error(descriptor),
            "bound_signature": signature_error(bound),
            "call_length": len(descriptor(tensor, 2)),
            "function_doc": module.chunk.__doc__,
            "function_module": module.chunk.__module__,
            "function_qualname": module.chunk.__qualname__,
        }

    def test_descriptor_and_doc_metadata_match_pytorch_2_13(self):
        actual = self.descriptor_contract(torch)
        expected = self.descriptor_contract(reference_torch)
        actual["repr"] = re.sub(r"0x[0-9a-f]+", "0xADDR", actual["repr"])
        expected["repr"] = re.sub(r"0x[0-9a-f]+", "0xADDR", expected["repr"])
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
