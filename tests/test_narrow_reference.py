import copy
import importlib
import inspect
import pickle
import re
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

    def contiguous_source(self, module, *, requires_grad=False):
        return module.tensor(
            [float(value) for value in range(24)], requires_grad=requires_grad
        ).reshape(3, 2, 4)

    def offset_source(self, module, *, requires_grad=False):
        return module.tensor(
            [float(value) for value in range(24)], requires_grad=requires_grad
        ).reshape(2, 3, 4)[1]

    def noncontiguous_source(self, module, *, requires_grad=False):
        return module.tensor(
            [float(value) for value in range(48)], requires_grad=requires_grad
        ).reshape(2, 2, 3, 4)[1].transpose(0, 1)

    def view_observation(self, result, source, start):
        if result.numel() == 0:
            data_ptr_observation = result.data_ptr()
        else:
            data_ptr_observation = result.data_ptr() == source[start].data_ptr()
        return {
            "values": result.tolist(),
            "shape": tuple(result.shape),
            "stride": result.stride(),
            "offset": result.storage_offset(),
            "data_ptr": data_ptr_observation,
            "same_dtype": result.dtype is source.dtype,
            "same_device": result.device == source.device,
        }

    def view_contract(self, module):
        source = self.contiguous_source(module)
        method_calls = (
            source.narrow(0, 1, 2),
            source.narrow(0, start=1, length=2),
            source.narrow(dim=0, start=1, length=2),
            source.narrow(length=2, dim=0, start=1),
            source.narrow(-3, 1, 2),
            source.narrow(-3, -2, 2),
        )
        top_level_calls = (
            module.narrow(source, 0, 1, 2),
            module.narrow(source, 0, start=1, length=2),
            module.narrow(source, dim=0, start=1, length=2),
            module.narrow(input=source, dim=0, start=1, length=2),
            module.narrow(length=2, start=1, input=source, dim=0),
            module.narrow(x=source, dim=0, start=1, length=2),
            module.narrow(a=source, dim=0, start=1, length=2),
            module.narrow(x1=source, dim=0, start=1, length=2),
            module.narrow(source, -3, 1, 2),
            module.narrow(source, -3, -2, 2),
        )

        offset = self.offset_source(module)
        noncontiguous = self.noncontiguous_source(module)
        empty_length = source.narrow(0, 1, 0)
        zero_source = module.zeros((0, 2))
        zero_narrowed = zero_source.narrow(0, 0, 0)
        return {
            "method_contiguous": tuple(
                self.view_observation(result, source, 1) for result in method_calls
            ),
            "top_level_contiguous": tuple(
                self.view_observation(result, source, 1) for result in top_level_calls
            ),
            "offset": self.view_observation(offset.narrow(0, 1, 2), offset, 1),
            "noncontiguous": self.view_observation(
                noncontiguous.narrow(0, 1, 2), noncontiguous, 1
            ),
            "empty_length": self.view_observation(empty_length, source, 1),
            "zero_source": self.view_observation(zero_narrowed, zero_source, 0),
        }

    def test_values_layout_aliasing_and_empty_cases_match_pytorch_2_13(self):
        self.assertEqual(self.view_contract(torch), self.view_contract(reference_torch))

    def autograd_contract(self, module):
        leaf = module.tensor(
            [float(value) for value in range(48)], requires_grad=True
        )
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        selected = source.narrow(0, 1, 2)
        metadata = (
            selected.requires_grad,
            selected.is_leaf,
            selected.output_nr,
            tuple(selected.shape),
            selected.stride(),
            selected.storage_offset(),
            selected.data_ptr() == source[1].data_ptr(),
        )
        selected.sum().backward()

        no_grad_source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with module.no_grad():
            untracked = module.narrow(no_grad_source, dim=0, start=-1, length=1)

        empty = module.zeros((2, 0, 3), requires_grad=True)
        empty_selected = module.narrow(empty, 0, 1, 1)
        empty_selected.sum().backward()
        return {
            "metadata": metadata,
            "gradient": leaf.grad.tolist(),
            "no_grad": (
                untracked.requires_grad,
                untracked.is_leaf,
                untracked.output_nr,
                tuple(untracked.shape),
                untracked.stride(),
                untracked.storage_offset(),
                untracked.data_ptr() == no_grad_source[1].data_ptr(),
            ),
            "empty": (
                empty_selected.requires_grad,
                empty_selected.is_leaf,
                empty_selected.output_nr,
                tuple(empty_selected.shape),
                empty_selected.stride(),
                empty_selected.storage_offset(),
                empty_selected.data_ptr(),
            ),
            "empty_gradient_shape": tuple(empty.grad.shape),
            "empty_gradient": empty.grad.tolist(),
        }

    def test_no_grad_and_backward_through_sum_match_pytorch_2_13(self):
        self.assertEqual(self.autograd_contract(torch), self.autograd_contract(reference_torch))

    def error_contract(self, module):
        tensor = module.zeros((3, 2, 4))
        scalar = module.tensor(1.0)
        zero_source = module.zeros((0, 2))
        return (
            self.error(lambda: tensor.narrow(0, -4, 0)),
            self.error(lambda: tensor.narrow(0, 4, 0)),
            self.error(lambda: tensor.narrow(0, 0, -1)),
            self.error(lambda: tensor.narrow(0, -1, 2)),
            self.error(lambda: tensor.narrow(3, 0, 1)),
            self.error(lambda: tensor.narrow(-4, 0, 1)),
            self.error(lambda: scalar.narrow(0, 0, 1)),
            self.error(lambda: zero_source.narrow(0, 1, 0)),
            self.error(lambda: module.narrow(tensor, 0, 4, 0)),
            self.error(lambda: module.narrow(tensor, 0, 0, -1)),
            self.error(lambda: module.narrow(scalar, 0, 0, 1)),
            tuple(tensor.narrow(np.int64(0), np.int32(1), np.uint32(2)).shape),
            tuple(module.narrow(tensor, np.int64(0), np.int32(1), np.uint32(2)).shape),
        )

    def test_supported_bounds_and_errors_match_pytorch_2_13(self):
        self.assertEqual(self.error_contract(torch), self.error_contract(reference_torch))

    def callable_contract(self, module):
        tensor = module.zeros((2, 3))
        descriptor = inspect.getattr_static(module.Tensor, "narrow")
        bound = tensor.narrow
        function = module.narrow
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)

        def signature_error(callable_object):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                return type(error).__name__
            self.fail("narrow unexpectedly exposed an inspectable signature")

        return {
            "descriptor_type": type(descriptor).__name__,
            "descriptor_is_method": type(descriptor) is types.MethodDescriptorType,
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "descriptor_owner": (
                descriptor.__objclass__.__name__,
                descriptor.__objclass__.__module__,
            ),
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "descriptor_doc": descriptor.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "descriptor_repr": repr(descriptor),
            "descriptor_signature_error": signature_error(descriptor),
            "descriptor_copy_identity": copy.copy(descriptor) is descriptor,
            "descriptor_deepcopy_identity": copy.deepcopy(descriptor) is descriptor,
            "descriptor_pickle_identities": tuple(
                pickle.loads(pickle.dumps(descriptor, protocol)) is descriptor
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "bound_type": type(bound).__name__,
            "bound_is_builtin_method": type(bound) is types.BuiltinMethodType,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "bound_doc": bound.__doc__,
            "bound_module": bound.__module__,
            "bound_text_signature": bound.__text_signature__,
            "bound_signature_error": signature_error(bound),
            "bound_copy_identity": copy.copy(bound) is bound,
            "bound_deepcopy_identity": copy.deepcopy(bound) is bound,
            "descriptor_call_shape": tuple(descriptor(tensor, 0, 0, 1).shape),
            "function_type": type(function).__name__,
            "function_is_builtin": type(function) is types.BuiltinFunctionType,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_module": function.__module__,
            "function_doc": function.__doc__,
            "function_text_signature": function.__text_signature__,
            "function_repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "function_signature_error": signature_error(function),
            "function_copy_identity": copy.copy(function) is function,
            "function_deepcopy_identity": copy.deepcopy(function) is function,
            "function_pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.narrow is function,
            "all_count": module.__all__.count("narrow"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["narrow"] is function,
        }

    def test_callable_import_wildcard_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(self.callable_contract(torch), self.callable_contract(reference_torch))

    def reload_contract(self, module):
        function = module.narrow
        descriptor = inspect.getattr_static(module.Tensor, "narrow")
        try:
            reloaded = importlib.reload(module)
        except Exception as error:
            return {
                "reload_error": (
                    type(error).__name__,
                    re.sub(r"0x[0-9a-f]+", "0x...", str(error).splitlines()[0]),
                )
            }
        return {
            "reload_error": None,
            "module_identity": reloaded is module,
            "function_identity": module.narrow is function,
            "descriptor_identity": inspect.getattr_static(module.Tensor, "narrow")
            is descriptor,
            "pickle_identity": pickle.loads(pickle.dumps(module.narrow)) is module.narrow,
        }

    def test_reload_behavior_matches_pytorch_2_13_when_reference_reload_is_available(self):
        expected = self.reload_contract(reference_torch)
        if expected["reload_error"] is not None:
            self.skipTest(f"reference PyTorch reload failed: {expected['reload_error']}")
        self.assertEqual(self.reload_contract(torch), expected)


if __name__ == "__main__":
    unittest.main()
