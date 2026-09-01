import copy
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference PyTorch package")
class TensorNarrowReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("narrow differentials require pinned PyTorch 2.13.0")

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("narrow unexpectedly accepted an invalid call")

    def source(self, module, *, requires_grad=False):
        values = [float(value) for value in range(48)]
        return module.tensor(values, requires_grad=requires_grad).reshape(2, 2, 3, 4)[
            1
        ].transpose(0, 1)

    def view_observation(self, tensor, view):
        return {
            "values": view.tolist(),
            "shape": tuple(view.shape),
            "stride": view.stride(),
            "offset": view.storage_offset(),
            "data_ptr_matches_first_row": view.data_ptr()
            == tensor.select(0, 1).data_ptr(),
            "same_dtype": view.dtype is tensor.dtype,
            "same_device": view.device == tensor.device,
        }

    def view_contract(self, module):
        contiguous = module.tensor([float(value) for value in range(12)]).reshape(4, 3)
        offset = module.tensor([float(value) for value in range(24)]).reshape(2, 3, 4)[
            1
        ]
        noncontiguous = self.source(module)
        empty_length = noncontiguous.narrow(0, 2, 0)
        zero_sized_source = module.zeros((0, 2))
        zero_sized_view = zero_sized_source.narrow(0, 0, 0)
        empty_inner = module.zeros((2, 0, 3)).narrow(0, 1, 1)
        return (
            self.view_observation(contiguous, contiguous.narrow(0, 1, 2)),
            contiguous.narrow(0, 0, 4).is_set_to(contiguous),
            self.view_observation(offset, offset.narrow(dim=0, start=1, length=2)),
            self.view_observation(noncontiguous, noncontiguous.narrow(-3, -2, 2)),
            {
                "values": empty_length.tolist(),
                "shape": tuple(empty_length.shape),
                "stride": empty_length.stride(),
                "offset": empty_length.storage_offset(),
                "data_ptr": empty_length.data_ptr(),
            },
            {
                "values": zero_sized_view.tolist(),
                "shape": tuple(zero_sized_view.shape),
                "stride": zero_sized_view.stride(),
                "offset": zero_sized_view.storage_offset(),
                "data_ptr": zero_sized_view.data_ptr(),
                "is_set_to_source": zero_sized_view.is_set_to(zero_sized_source),
            },
            {
                "values": empty_inner.tolist(),
                "shape": tuple(empty_inner.shape),
                "stride": empty_inner.stride(),
                "offset": empty_inner.storage_offset(),
                "data_ptr": empty_inner.data_ptr(),
            },
            tuple(contiguous.narrow(np.int64(0), np.int32(0), np.uint32(1)).shape),
        )

    def test_views_match_pytorch_2_13(self):
        self.assertEqual(self.view_contract(torch), self.view_contract(reference_torch))

    def top_level_view_contract(self, module):
        source = self.source(module)
        calls = (
            module.narrow(source, 0, 1, 2),
            module.narrow(source, 0, 1, length=2),
            module.narrow(source, dim=0, start=1, length=2),
            module.narrow(input=source, dim=0, start=1, length=2),
            module.narrow(length=2, input=source, start=1, dim=0),
            module.narrow(x=source, dim=0, start=1, length=2),
            module.narrow(a=source, dim=0, start=1, length=2),
            module.narrow(x1=source, dim=0, start=1, length=2),
            module.narrow(source, -3, 1, 2),
            module.narrow(source, 0, -2, 2),
        )
        return tuple(self.view_observation(source, view) for view in calls)

    def test_top_level_views_match_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_view_contract(torch),
            self.top_level_view_contract(reference_torch),
        )

    def autograd_contract(self, module):
        leaf = module.tensor(
            [float(value) for value in range(48)], requires_grad=True
        )
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        narrowed = source.narrow(0, 1, 2)
        metadata = (
            narrowed.requires_grad,
            narrowed.is_leaf,
            tuple(narrowed.shape),
            narrowed.stride(),
            narrowed.storage_offset(),
        )
        (narrowed * 3.0).sum().backward()

        no_grad_source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with module.no_grad():
            untracked = module.narrow(no_grad_source, 0, 1, 1)

        empty = module.zeros((2, 0, 3), requires_grad=True)
        empty.narrow(0, 1, 1).sum().backward()
        return {
            "metadata": metadata,
            "gradient": leaf.grad.tolist(),
            "no_grad": (
                untracked.requires_grad,
                untracked.is_leaf,
                tuple(untracked.shape),
                untracked.stride(),
                untracked.storage_offset(),
            ),
            "empty_gradient_shape": tuple(empty.grad.shape),
            "empty_gradient": empty.grad.tolist(),
        }

    def test_no_grad_and_backward_through_sum_match_pytorch_2_13(self):
        self.assertEqual(self.autograd_contract(torch), self.autograd_contract(reference_torch))

    def error_contract(self, module):
        tensor = module.zeros((2, 3, 4))
        scalar = module.tensor(1.0)
        empty = module.zeros((0, 2))
        return (
            self.error(lambda: tensor.narrow(3, 0, 1)),
            self.error(lambda: tensor.narrow(-4, 0, 1)),
            self.error(lambda: tensor.narrow(0, 3, 0)),
            self.error(lambda: tensor.narrow(0, -3, 1)),
            self.error(lambda: tensor.narrow(0, 0, -1)),
            self.error(lambda: tensor.narrow(0, 1, 2)),
            self.error(lambda: scalar.narrow(0, 0, 0)),
            self.error(lambda: empty.narrow(0, 0, 1)),
            self.error(lambda: empty.narrow(0, 1, 0)),
            self.error(lambda: module.narrow(tensor, 3, 0, 1)),
            self.error(lambda: module.narrow(tensor, 0, 1, 2)),
        )

    def test_supported_bounds_errors_match_pytorch_2_13(self):
        self.assertEqual(self.error_contract(torch), self.error_contract(reference_torch))

    def callable_contract(self, module):
        function = module.narrow
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)

        def signature_error(callable_object):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                return type(error).__name__
            self.fail("narrow unexpectedly exposed an inspectable signature")

        pickle_identity = []
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            pickle_identity.append(pickle.loads(pickle.dumps(function, protocol)) is function)

        return {
            "type": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "text_signature": function.__text_signature__,
            "signature_error": signature_error(function),
            "import_identity": __import__(module.__name__, fromlist=["narrow"]).narrow
            is function,
            "wildcard_identity": wildcard_namespace["narrow"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identity": tuple(pickle_identity),
        }

    def test_callable_import_wildcard_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
