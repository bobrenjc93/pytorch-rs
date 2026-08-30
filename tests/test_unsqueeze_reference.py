import inspect
import pickle
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorUnsqueezeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("unsqueeze differentials require pinned PyTorch 2.13.0")

    def layout_cases(self, module, *, requires_grad=False):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=requires_grad
        )
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32, requires_grad=requires_grad)),
            ("empty", module.zeros((2, 0, 3), dtype=module.float32, requires_grad=requires_grad)),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 3)[1]),
        )

    def call_unsqueeze(self, module, source, form):
        rank = source.dim()
        if form == "method_front":
            return source.unsqueeze(0)
        if form == "method_front_negative":
            return source.unsqueeze(-rank - 1)
        if form == "method_front_keyword":
            return source.unsqueeze(dim=0)
        if form == "method_front_axis":
            return source.unsqueeze(axis=0)
        if form == "method_back":
            return source.unsqueeze(rank)
        if form == "method_back_negative":
            return source.unsqueeze(-1)
        if form == "method_back_keyword":
            return source.unsqueeze(dim=rank)
        if form == "method_back_axis":
            return source.unsqueeze(axis=rank)
        if form == "top_level_front":
            return module.unsqueeze(source, 0)
        if form == "top_level_front_negative":
            return module.unsqueeze(source, -rank - 1)
        if form == "top_level_keywords":
            return module.unsqueeze(input=source, dim=0)
        if form == "top_level_reordered":
            return module.unsqueeze(dim=0, input=source)
        if form == "top_level_x_axis":
            return module.unsqueeze(x=source, axis=0)
        if form == "top_level_a":
            return module.unsqueeze(a=source, dim=0)
        if form == "top_level_x1":
            return module.unsqueeze(x1=source, dim=0)
        if form == "top_level_back":
            return module.unsqueeze(source, rank)
        if form == "top_level_back_negative":
            return module.unsqueeze(source, -1)
        if form == "top_level_back_keywords":
            return module.unsqueeze(input=source, dim=rank)
        if form == "top_level_back_axis":
            return module.unsqueeze(input=source, axis=rank)
        raise AssertionError(f"unknown unsqueeze form: {form}")

    def tensor_array(self, tensor):
        return np.asarray(tensor.detach(), dtype=np.float32)

    def view_contract(self, source, result):
        values = self.tensor_array(result).reshape(-1)
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
            "output_nr": result.output_nr,
            "values": result.tolist(),
            "value_bits": tuple(values.view(np.uint32).tolist()),
        }

    def test_edge_values_layout_aliasing_offsets_and_metadata_match_pytorch_2_13(
        self,
    ):
        forms = (
            "method_front",
            "method_front_negative",
            "method_front_keyword",
            "method_front_axis",
            "method_back",
            "method_back_negative",
            "method_back_keyword",
            "method_back_axis",
            "top_level_front",
            "top_level_front_negative",
            "top_level_keywords",
            "top_level_reordered",
            "top_level_x_axis",
            "top_level_a",
            "top_level_x1",
            "top_level_back",
            "top_level_back_negative",
            "top_level_back_keywords",
            "top_level_back_axis",
        )
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for (actual_case, actual), (expected_case, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(actual_case, expected_case)
            for form in forms:
                with self.subTest(case=actual_case, form=form):
                    actual_result = self.call_unsqueeze(torch, actual, form)
                    expected_result = self.call_unsqueeze(reference_torch, expected, form)
                    self.assertEqual(
                        self.view_contract(actual, actual_result),
                        self.view_contract(expected, expected_result),
                    )
                    self.assertEqual(
                        expected_result.untyped_storage().data_ptr(),
                        expected.untyped_storage().data_ptr(),
                    )

    def make_autograd_case(self, module, case):
        if case == "scalar":
            leaf = module.tensor(-2.0, dtype=module.float32, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros((2, 0, 3), dtype=module.float32, requires_grad=True)
            return leaf, leaf

        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        leaf = module.tensor(values.tolist(), dtype=module.float32, requires_grad=True)
        if case == "offset":
            return leaf, leaf[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 3)[1]
        raise AssertionError(f"unknown case: {case}")

    def autograd_contract(self, module, case, form):
        leaf, source = self.make_autograd_case(module, case)
        result = self.call_unsqueeze(module, source, form)
        metadata = self.view_contract(source, result)
        weights = module.ones(tuple(result.shape), dtype=module.float32)
        (result * weights).sum().backward()
        return metadata, np.asarray(leaf.grad, dtype=np.float32).copy()

    def no_grad_contract(self, module, case, form):
        leaf, source = self.make_autograd_case(module, case)
        with module.no_grad():
            result = self.call_unsqueeze(module, source, form)
        metadata = self.view_contract(source, result)
        return metadata, leaf.grad

    def unsqueeze_node_diagnostic(self, module, form):
        leaf = module.tensor([2.0], dtype=module.float32, requires_grad=True)
        result = self.call_unsqueeze(module, leaf, form)
        try:
            module.nn.functional.dropout(None, p=result, training=False)
        except ValueError as error:
            return str(error)
        self.fail("dropout unexpectedly accepted an out-of-range tensor probability")

    def test_autograd_and_no_grad_match_pytorch_2_13_for_supported_edges(self):
        forms = (
            "method_front",
            "method_back_negative",
            "top_level_front",
            "top_level_back",
        )
        for form in forms:
            for case in ("scalar", "empty", "offset", "noncontiguous"):
                with self.subTest(form=form, case=case, mode="autograd"):
                    actual_metadata, actual_gradient = self.autograd_contract(
                        torch, case, form
                    )
                    expected_metadata, expected_gradient = self.autograd_contract(
                        reference_torch, case, form
                    )
                    self.assertEqual(actual_metadata, expected_metadata)
                    np.testing.assert_array_equal(actual_gradient, expected_gradient)

                with self.subTest(form=form, case=case, mode="no_grad"):
                    self.assertEqual(
                        self.no_grad_contract(torch, case, form),
                        self.no_grad_contract(reference_torch, case, form),
                    )

        for form in ("method_front", "top_level_back"):
            with self.subTest(form=form, diagnostic="node"):
                self.assertEqual(
                    self.unsqueeze_node_diagnostic(torch, form),
                    self.unsqueeze_node_diagnostic(reference_torch, form),
                )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("unsqueeze unexpectedly accepted the operation")

    def test_supported_binding_errors_match_pytorch_2_13(self):
        actual = torch.zeros((2, 3), dtype=torch.float32)
        expected = reference_torch.zeros((2, 3), dtype=reference_torch.float32)
        scalar = torch.tensor(1.0, dtype=torch.float32)
        expected_scalar = reference_torch.tensor(1.0, dtype=reference_torch.float32)
        cases = (
            (lambda: actual.unsqueeze(), lambda: expected.unsqueeze()),
            (lambda: actual.unsqueeze(0, 0), lambda: expected.unsqueeze(0, 0)),
            (lambda: actual.unsqueeze(0, dim=0), lambda: expected.unsqueeze(0, dim=0)),
            (lambda: actual.unsqueeze(extra=0), lambda: expected.unsqueeze(extra=0)),
            (lambda: actual.unsqueeze(None), lambda: expected.unsqueeze(None)),
            (lambda: actual.unsqueeze([0]), lambda: expected.unsqueeze([0])),
            (lambda: actual.unsqueeze((0,)), lambda: expected.unsqueeze((0,))),
            (lambda: actual.unsqueeze(True), lambda: expected.unsqueeze(True)),
            (lambda: actual.unsqueeze(0, out=None), lambda: expected.unsqueeze(0, out=None)),
            (lambda: actual.unsqueeze(2**100), lambda: expected.unsqueeze(2**100)),
            (lambda: actual.unsqueeze(3), lambda: expected.unsqueeze(3)),
            (lambda: actual.unsqueeze(-4), lambda: expected.unsqueeze(-4)),
            (lambda: scalar.unsqueeze(1), lambda: expected_scalar.unsqueeze(1)),
            (lambda: torch.unsqueeze(), lambda: reference_torch.unsqueeze()),
            (lambda: torch.unsqueeze(actual), lambda: reference_torch.unsqueeze(expected)),
            (lambda: torch.unsqueeze([], 0), lambda: reference_torch.unsqueeze([], 0)),
            (lambda: torch.unsqueeze(actual, [0]), lambda: reference_torch.unsqueeze(expected, [0])),
            (lambda: torch.unsqueeze(actual, (0,)), lambda: reference_torch.unsqueeze(expected, (0,))),
            (lambda: torch.unsqueeze(actual, True), lambda: reference_torch.unsqueeze(expected, True)),
            (
                lambda: torch.unsqueeze(x=actual, a=actual, dim=0),
                lambda: reference_torch.unsqueeze(x=expected, a=expected, dim=0),
            ),
            (
                lambda: torch.unsqueeze(a=actual, x=actual, dim=0),
                lambda: reference_torch.unsqueeze(a=expected, x=expected, dim=0),
            ),
            (
                lambda: torch.unsqueeze(actual, 0, out=None),
                lambda: reference_torch.unsqueeze(expected, 0, out=None),
            ),
            (lambda: torch.unsqueeze(actual, 2**100), lambda: reference_torch.unsqueeze(expected, 2**100)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def callable_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "unsqueeze")
        tensor = module.zeros((2, 3), dtype=module.float32)
        bound = tensor.unsqueeze
        function = module.unsqueeze
        owner = function.__reduce__()[1][0]
        namespace = {}
        exec(f"from {module.__name__} import *", namespace)
        return {
            "descriptor_type": type(descriptor).__name__,
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "descriptor_owner_name": descriptor.__objclass__.__name__,
            "descriptor_owner_module": descriptor.__objclass__.__module__,
            "descriptor_doc": descriptor.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "bound_qualname": bound.__qualname__,
            "bound_doc": bound.__doc__,
            "bound_text_signature": bound.__text_signature__,
            "function_type": type(function).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_module": function.__module__,
            "function_doc": function.__doc__,
            "function_text_signature": function.__text_signature__,
            "all_count": module.__all__.count("unsqueeze"),
            "wildcard_is_function": namespace["unsqueeze"] is function,
            "owner_name": owner.__name__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_attr_is_function": owner.unsqueeze is function,
            "public_owner_missing": not hasattr(module, "_VariableFunctionsClass"),
            "pickle_roundtrip": all(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def test_public_surface_intentionally_keeps_middle_dims_modes_and_overrides_unsupported(
        self,
    ):
        tensor = torch.zeros((2, 3), dtype=torch.float32)
        for call, message in (
            (lambda: tensor.unsqueeze(1), "Tensor.unsqueeze"),
            (lambda: tensor.unsqueeze(-2), "Tensor.unsqueeze"),
            (lambda: torch.unsqueeze(tensor, 1), "torch.unsqueeze"),
            (lambda: torch.unsqueeze(tensor, -2), "torch.unsqueeze"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                RuntimeError,
                rf"^{message} only supports leading and trailing dimensions$",
            ):
                call()

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return object()

        with self.assertRaisesRegex(
            TypeError,
            r"^unsqueeze\(\): argument 'input' \(position 1\) must be Tensor, not Override$",
        ):
            torch.unsqueeze(Override(), 0)

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return object()

        for call, message in (
            (lambda: tensor.unsqueeze(0), "Tensor.unsqueeze"),
            (lambda: torch.unsqueeze(tensor, 0), "torch.unsqueeze"),
        ):
            with self.subTest(message=message):
                with RecordingMode(), self.assertRaisesRegex(
                    NotImplementedError,
                    rf"^{message} does not support TorchFunctionMode$",
                ):
                    call()


if __name__ == "__main__":
    unittest.main()
