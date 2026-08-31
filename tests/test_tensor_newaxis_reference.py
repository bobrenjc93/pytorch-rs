import copy
import gc
import importlib
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


METHOD_DOC = "\nunsqueeze(dim) -> Tensor\n\nSee :func:`torch.unsqueeze`\n"
FUNCTION_DOC = (
    "\nunsqueeze(input, dim) -> Tensor\n\n"
    "Returns a new tensor with a dimension of size one inserted at the\n"
    "specified position.\n\n"
    "The returned tensor shares the same underlying data with this tensor.\n\n"
    "A :attr:`dim` value within the range ``[-input.dim() - 1, input.dim() + 1)``\n"
    "can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`\n"
    "applied at :attr:`dim` = ``dim + input.dim() + 1``.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n"
    "    dim (int): the index at which to insert the singleton dimension\n\n"
    "Example::\n\n"
    "    >>> x = torch.tensor([1, 2, 3, 4])\n"
    "    >>> torch.unsqueeze(x, 0)\n"
    "    tensor([[ 1,  2,  3,  4]])\n"
    "    >>> torch.unsqueeze(x, 1)\n"
    "    tensor([[ 1],\n"
    "            [ 2],\n"
    "            [ 3],\n"
    "            [ 4]])\n"
)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorNewAxisIndexReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "new-axis indexing differentials require pinned PyTorch 2.13.0"
            )

    def namespace_contract(self, module):
        namespace = {}
        exec(f"from {module.__name__} import *", namespace)
        return {
            "is_none": module.newaxis is None,
            "all_count": module.__all__.count("newaxis"),
            "native_has_newaxis": hasattr(module._C, "newaxis"),
            "wildcard_is_none": namespace["newaxis"] is None,
        }

    def test_newaxis_namespace_matches_pytorch_2_13(self):
        self.assertEqual(
            self.namespace_contract(torch),
            self.namespace_contract(reference_torch),
        )

    def layout_cases(self, module):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = module.tensor(values.tolist(), dtype=module.float32)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            ("empty", module.zeros((2, 0, 3), dtype=module.float32)),
            ("contiguous", base),
            ("transposed", base.transpose(0, 3)),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 3)[1]),
        )

    def unsqueeze_contract(self, source, index):
        result = source[index]
        values = np.asarray(result.detach(), dtype=np.float32).reshape(-1)
        return {
            "distinct_wrapper": result is not source,
            "shape": tuple(result.shape),
            "stride": result.stride(),
            "storage_offset": result.storage_offset(),
            "shared_data_pointer": result.data_ptr() == source.data_ptr(),
            "same_logical_view": result.is_set_to(source),
            "dtype": str(result.dtype),
            "device": str(result.device),
            "requires_grad": result.requires_grad,
            "is_leaf": result.is_leaf,
            "values": result.tolist(),
            "value_bits": tuple(values.view(np.uint32).tolist()),
        }

    def test_layout_value_and_aliasing_contracts_match_pytorch_2_13(self):
        for spelling in ("none", "newaxis"):
            actual_cases = self.layout_cases(torch)
            expected_cases = self.layout_cases(reference_torch)
            for (actual_case, actual), (expected_case, expected) in zip(
                actual_cases, expected_cases, strict=True
            ):
                self.assertEqual(actual_case, expected_case)
                with self.subTest(spelling=spelling, case=actual_case):
                    actual_index = None if spelling == "none" else torch.newaxis
                    expected_index = (
                        None
                        if spelling == "none"
                        else reference_torch.newaxis
                    )
                    self.assertEqual(
                        self.unsqueeze_contract(actual, actual_index),
                        self.unsqueeze_contract(expected, expected_index),
                    )

    def test_exact_trailing_layout_value_and_aliasing_match_pytorch_2_13(self):
        for spelling in ("none", "newaxis"):
            actual_cases = self.layout_cases(torch)
            expected_cases = self.layout_cases(reference_torch)
            for (actual_case, actual), (expected_case, expected) in zip(
                actual_cases, expected_cases, strict=True
            ):
                self.assertEqual(actual_case, expected_case)
                with self.subTest(spelling=spelling, case=actual_case):
                    actual_newaxis = (
                        None if spelling == "none" else torch.newaxis
                    )
                    expected_newaxis = (
                        None
                        if spelling == "none"
                        else reference_torch.newaxis
                    )
                    self.assertEqual(
                        self.unsqueeze_contract(
                            actual, (Ellipsis, actual_newaxis)
                        ),
                        self.unsqueeze_contract(
                            expected, (Ellipsis, expected_newaxis)
                        ),
                    )

    def extreme_empty_contract(
        self, module, leading_dimension, trailing_dimension
    ):
        source = module.zeros((0,), dtype=module.float32).reshape(
            (leading_dimension, 0, trailing_dimension)
        )
        try:
            result = source[module.newaxis]
        except Exception as error:
            message = str(error)
            non_concrete = (
                "SymIntArrayRef expected to contain only concrete integers"
            )
            if non_concrete in message:
                message = non_concrete
            return ("error", type(error).__name__, message)
        return (
            "result",
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(source),
            str(result.dtype),
            str(result.device),
        )

    @unittest.skipUnless(
        sys.maxsize == (1 << 63) - 1,
        "signed 64-bit stride wrapping requires a 64-bit Python build",
    )
    def test_extreme_empty_stride_boundaries_match_pytorch_2_13(self):
        cases = (
            ((1 << 62) - 1, 2),
            (1 << 62, 2),
            ((1 << 62) + 1, 2),
            ((1 << 62) - 1, 3),
            (1 << 62, 3),
            (sys.maxsize, 2),
            (sys.maxsize, 3),
        )
        for leading_dimension, trailing_dimension in cases:
            with self.subTest(
                leading_dimension=leading_dimension,
                trailing_dimension=trailing_dimension,
            ):
                self.assertEqual(
                    self.extreme_empty_contract(
                        torch, leading_dimension, trailing_dimension
                    ),
                    self.extreme_empty_contract(
                        reference_torch,
                        leading_dimension,
                        trailing_dimension,
                    ),
                )

    def make_autograd_case(self, module, case):
        if case == "scalar":
            leaf = module.tensor(
                -2.0, dtype=module.float32, requires_grad=True
            )
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf

        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )
        if case == "contiguous":
            return leaf, leaf
        if case == "transposed":
            return leaf, leaf.transpose(0, 3)
        if case == "offset":
            return leaf, leaf[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 3)[1]
        raise AssertionError(f"unknown case: {case}")

    def autograd_contract(self, module, case, trailing=False):
        leaf, source = self.make_autograd_case(module, case)
        index = (Ellipsis, module.newaxis) if trailing else module.newaxis
        result = source[index]
        metadata = (
            result is not source,
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(source),
            result.requires_grad,
            result.is_leaf,
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
        )
        weights = module.ones(tuple(result.shape), dtype=module.float32)
        (result * weights).sum().backward()
        return metadata, np.asarray(leaf.grad, dtype=np.float32).copy()

    def no_grad_contract(self, module, case, trailing=False):
        leaf, source = self.make_autograd_case(module, case)
        index = (Ellipsis, None) if trailing else None
        with module.no_grad():
            result = source[index]
        return (
            result is not source,
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(source),
            result.requires_grad,
            result.is_leaf,
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            leaf.grad,
        )

    def unsqueeze_node_diagnostic(self, module, trailing=False):
        leaf = module.tensor(
            [2.0], dtype=module.float32, requires_grad=True
        )
        index = (Ellipsis, module.newaxis) if trailing else module.newaxis
        try:
            module.nn.functional.dropout(
                None, p=leaf[index], training=False
            )
        except ValueError as error:
            return str(error)
        self.fail("dropout unexpectedly accepted an out-of-range tensor probability")

    def assert_autograd_and_no_grad_match_pytorch_2_13(self, trailing):
        for case in (
            "scalar",
            "empty",
            "contiguous",
            "transposed",
            "offset",
            "noncontiguous",
        ):
            with self.subTest(case=case, mode="autograd"):
                actual_metadata, actual_gradient = self.autograd_contract(
                    torch, case, trailing
                )
                expected_metadata, expected_gradient = self.autograd_contract(
                    reference_torch, case, trailing
                )
                self.assertEqual(actual_metadata, expected_metadata)
                np.testing.assert_array_equal(actual_gradient, expected_gradient)

            with self.subTest(case=case, mode="no_grad"):
                self.assertEqual(
                    self.no_grad_contract(torch, case, trailing),
                    self.no_grad_contract(reference_torch, case, trailing),
                )

        self.assertEqual(
            self.unsqueeze_node_diagnostic(torch, trailing),
            self.unsqueeze_node_diagnostic(reference_torch, trailing),
        )

    def test_autograd_and_no_grad_match_pytorch_2_13_for_every_layout(self):
        self.assert_autograd_and_no_grad_match_pytorch_2_13(False)

    def test_trailing_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assert_autograd_and_no_grad_match_pytorch_2_13(True)

    def public_unsqueeze_contract(self, module, source, dimension, form):
        if form == "method":
            result = source.unsqueeze(dimension)
        elif form == "method_dim":
            result = source.unsqueeze(dim=dimension)
        elif form == "method_axis":
            result = source.unsqueeze(axis=dimension)
        elif form == "top_level":
            result = module.unsqueeze(source, dimension)
        elif form == "top_level_input_dim":
            result = module.unsqueeze(input=source, dim=dimension)
        elif form == "top_level_x_axis":
            result = module.unsqueeze(x=source, axis=dimension)
        else:
            raise AssertionError(f"unknown unsqueeze form: {form}")

        values = np.asarray(result.detach(), dtype=np.float32).reshape(-1)
        return {
            "distinct_wrapper": result is not source,
            "shape": tuple(result.shape),
            "stride": result.stride(),
            "storage_offset": result.storage_offset(),
            "shared_data_pointer": result.data_ptr() == source.data_ptr(),
            "same_logical_view": result.is_set_to(source),
            "dtype": str(result.dtype),
            "device": str(result.device),
            "requires_grad": result.requires_grad,
            "is_leaf": result.is_leaf,
            "values": result.tolist(),
            "value_bits": tuple(values.view(np.uint32).tolist()),
        }

    def test_public_boundary_unsqueeze_layouts_match_pytorch_2_13(self):
        forms = (
            "method",
            "method_dim",
            "method_axis",
            "top_level",
            "top_level_input_dim",
            "top_level_x_axis",
        )
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for (actual_case, actual), (expected_case, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(actual_case, expected_case)
            rank = len(actual.shape)
            dimensions = (
                ("front_zero", 0),
                ("front_negative", -rank - 1),
                ("back_positive", rank),
                ("back_negative", -1),
            )
            for label, dimension in dimensions:
                for form in forms:
                    with self.subTest(
                        case=actual_case, dimension=label, form=form
                    ):
                        self.assertEqual(
                            self.public_unsqueeze_contract(
                                torch, actual, dimension, form
                            ),
                            self.public_unsqueeze_contract(
                                reference_torch, expected, dimension, form
                            ),
                        )

    def public_unsqueeze_autograd_contract(self, module, case, trailing=False):
        leaf, source = self.make_autograd_case(module, case)
        dimension = -1 if trailing else 0
        result = module.unsqueeze(source, dimension) if trailing else source.unsqueeze(dimension)
        metadata = (
            result is not source,
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(source),
            result.requires_grad,
            result.is_leaf,
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
        )
        result.sum().backward()
        return metadata, np.asarray(leaf.grad, dtype=np.float32).copy()

    def public_unsqueeze_no_grad_contract(self, module, case, trailing=False):
        leaf, source = self.make_autograd_case(module, case)
        dimension = -1 if trailing else -len(source.shape) - 1
        with module.no_grad():
            result = source.unsqueeze(dimension) if trailing else module.unsqueeze(source, dimension)
        return (
            result is not source,
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(source),
            result.requires_grad,
            result.is_leaf,
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            leaf.grad,
        )

    def test_public_unsqueeze_autograd_and_no_grad_match_pytorch_2_13(self):
        for trailing in (False, True):
            for case, _ in self.layout_cases(torch):
                with self.subTest(case=case, trailing=trailing, mode="autograd"):
                    actual_metadata, actual_gradient = (
                        self.public_unsqueeze_autograd_contract(
                            torch, case, trailing
                        )
                    )
                    expected_metadata, expected_gradient = (
                        self.public_unsqueeze_autograd_contract(
                            reference_torch, case, trailing
                        )
                    )
                    self.assertEqual(actual_metadata, expected_metadata)
                    np.testing.assert_array_equal(
                        actual_gradient, expected_gradient
                    )

                with self.subTest(case=case, trailing=trailing, mode="no_grad"):
                    self.assertEqual(
                        self.public_unsqueeze_no_grad_contract(
                            torch, case, trailing
                        ),
                        self.public_unsqueeze_no_grad_contract(
                            reference_torch, case, trailing
                        ),
                    )

    def callable_contract(self, module):
        def normalize_message(message):
            return re.sub(r"0x[0-9a-f]+", "0x...", message).replace(
                "torch_rs.Tensor", "Tensor"
            )

        function = module.unsqueeze
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        direct_namespace = {}
        exec(
            f"from {module.__name__} import unsqueeze as imported_unsqueeze",
            direct_namespace,
        )
        descriptor = inspect.getattr_static(module.Tensor, "unsqueeze")
        bound = module.tensor([1.0], dtype=module.float32).unsqueeze
        try:
            inspect.signature(function)
        except Exception as error:
            function_signature_error = (
                type(error).__name__,
                normalize_message(str(error)),
            )
        else:
            function_signature_error = None
        try:
            inspect.signature(descriptor)
        except Exception as error:
            descriptor_signature_error = (
                type(error).__name__,
                normalize_message(str(error)),
            )
        else:
            descriptor_signature_error = None
        try:
            inspect.signature(bound)
        except Exception as error:
            bound_signature_error = (
                type(error).__name__,
                normalize_message(str(error)),
            )
        else:
            bound_signature_error = None

        return {
            "function_type": type(function).__name__,
            "function_is_builtin": type(function) is types.BuiltinFunctionType,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_module": function.__module__,
            "function_doc": function.__doc__,
            "function_doc_matches_expected": function.__doc__ == FUNCTION_DOC,
            "function_text_signature": function.__text_signature__,
            "function_repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "function_signature_error": function_signature_error,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.unsqueeze is function,
            "all_count": module.__all__.count("unsqueeze"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["unsqueeze"] is function,
            "direct_import_identity": direct_namespace["imported_unsqueeze"] is function,
            "function_copy_identity": copy.copy(function) is function,
            "function_deepcopy_identity": copy.deepcopy(function) is function,
            "function_pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "descriptor_type": type(descriptor).__name__,
            "descriptor_is_method_descriptor": type(descriptor)
            is types.MethodDescriptorType,
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "descriptor_objclass_name": descriptor.__objclass__.__name__,
            "descriptor_objclass_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "descriptor_doc": descriptor.__doc__,
            "descriptor_doc_matches_expected": descriptor.__doc__ == METHOD_DOC,
            "descriptor_text_signature": descriptor.__text_signature__,
            "descriptor_repr": repr(descriptor),
            "descriptor_signature_error": descriptor_signature_error,
            "descriptor_copy_identity": copy.copy(descriptor) is descriptor,
            "descriptor_deepcopy_identity": copy.deepcopy(descriptor) is descriptor,
            "descriptor_pickle_identities": tuple(
                pickle.loads(pickle.dumps(descriptor, protocol=protocol))
                is descriptor
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "bound_type": type(bound).__name__,
            "bound_is_builtin_method": type(bound) is types.BuiltinMethodType,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "bound_module": bound.__module__,
            "bound_doc": bound.__doc__,
            "bound_text_signature": bound.__text_signature__,
            "bound_signature_error": bound_signature_error,
            "bound_copy_identity": copy.copy(bound) is bound,
            "bound_deepcopy_identity": copy.deepcopy(bound) is bound,
            "descriptor_call_shape": tuple(
                descriptor(module.tensor([1.0], dtype=module.float32), 0).shape
            ),
            "bound_call_shape": tuple(bound(-1).shape),
        }

    def test_public_unsqueeze_callable_metadata_imports_copy_pickle_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def reload_contract(self, module):
        function = module.unsqueeze
        native = module._C
        reloaded = importlib.reload(native)
        return (
            reloaded is native,
            module._C is native,
            module._C._VariableFunctionsClass.unsqueeze is function,
            pickle.loads(pickle.dumps(module.unsqueeze)) is module.unsqueeze,
        )

    def test_public_unsqueeze_native_reload_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )

    def lifetime_contract(self, module, trailing=False):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )

        def make_view():
            source = (leaf * 2.0).transpose(0, 2)[1]
            index = (Ellipsis, module.newaxis) if trailing else module.newaxis
            return source[index]

        result = make_view()
        gc.collect()
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.tolist(),
            result.requires_grad,
            result.is_leaf,
        )
        values = (
            [[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]
            if trailing
            else [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
        )
        weights = module.tensor(values, dtype=module.float32)
        (result * weights).sum().backward()
        return metadata, np.asarray(leaf.grad, dtype=np.float32).copy()

    def test_source_lifetime_matches_pytorch_2_13(self):
        actual_metadata, actual_gradient = self.lifetime_contract(torch)
        expected_metadata, expected_gradient = self.lifetime_contract(
            reference_torch
        )
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_gradient, expected_gradient)

    def test_trailing_source_lifetime_matches_pytorch_2_13(self):
        actual_metadata, actual_gradient = self.lifetime_contract(torch, True)
        expected_metadata, expected_gradient = self.lifetime_contract(
            reference_torch, True
        )
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_gradient, expected_gradient)

    def mode_contract(self, module, trailing=False):
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func,
                        dispatch_types,
                        args,
                        kwargs,
                        len(module.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        descriptor = inspect.getattr_static(module.Tensor, "__getitem__")
        mode = RecordingMode()
        records = []
        forwarded_layouts = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        for case, source in self.layout_cases(module):
            index = (
                (Ellipsis, module.newaxis) if trailing else module.newaxis
            )
            mode.calls.clear()
            with mode:
                result = source[index]
                context_depth = len(
                    module.overrides._get_current_function_mode_stack()
                )
            function, dispatch_types, args, kwargs, handler_depth = mode.calls[0]
            records.append(
                (
                    case,
                    result is marker,
                    len(mode.calls),
                    type(function).__name__,
                    function.__name__,
                    function.__qualname__,
                    function.__objclass__.__name__,
                    function.__objclass__.__module__,
                    function is descriptor,
                    dispatch_types == (),
                    len(args),
                    args[0] is source,
                    args[1] is index,
                    kwargs is None,
                    handler_depth,
                    context_depth,
                )
            )

            forwarding_index = (Ellipsis, None) if trailing else None
            with ForwardingMode():
                forwarded = source[forwarding_index]
            forwarded_layouts.append(
                (
                    case,
                    tuple(forwarded.shape),
                    forwarded.stride(),
                    forwarded.storage_offset(),
                    forwarded.data_ptr() == source.data_ptr(),
                    forwarded.is_set_to(source),
                )
            )

        events = []

        class NestedMode(module.overrides.TorchFunctionMode):
            def __init__(self, label, replacement=None):
                self.label = label
                self.replacement = replacement

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                events.append(
                    (
                        self.label,
                        func.__qualname__,
                        dispatch_types == (),
                        len(module.overrides._get_current_function_mode_stack()),
                    )
                )
                if self.replacement is not None:
                    return self.replacement
                return func(*args, **(kwargs or {}))

        source = self.layout_cases(module)[0][1]
        lower = NestedMode("lower", marker)
        upper = NestedMode("upper")
        nested_index = (
            (Ellipsis, module.newaxis) if trailing else module.newaxis
        )
        with lower:
            with upper:
                nested_result = source[nested_index]

        return {
            "records": tuple(records),
            "forwarded_layouts": tuple(forwarded_layouts),
            "nested_replacement": nested_result is marker,
            "nested_events": tuple(events),
            "final_stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_torch_function_mode_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch), self.mode_contract(reference_torch)
        )

    def test_trailing_torch_function_mode_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch, True),
            self.mode_contract(reference_torch, True),
        )


if __name__ == "__main__":
    unittest.main()
