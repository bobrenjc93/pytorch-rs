import gc
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


class IndexSize:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorUnflattenReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.unflatten differentials require pinned PyTorch 2.13.0"
            )

    def tensor_array(self, tensor, module):
        detached = tensor.detach()
        if module is reference_torch:
            return detached.cpu().numpy()
        return np.asarray(detached)

    def make_layout_cases(self, module):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        base = module.tensor(values.tolist(), dtype=module.float32)
        return (
            ("contiguous", base, 2, (2, -1)),
            ("offset", base.transpose(0, 1)[1], 1, (2, 2)),
            ("noncontiguous", base.transpose(0, 1), -1, (1, 5)),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32)
                .transpose(0, 2)[1],
                0,
                (0, 1),
            ),
        )

    def layout_contract(self, module, source, dimension, sizes):
        result = source.unflatten(dimension, sizes)
        return (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.data_ptr() == source.data_ptr(),
            result.requires_grad,
            result.is_leaf,
            str(result.dtype),
            str(result.device),
            self.tensor_array(result, module).copy(),
        )

    def test_shapes_strides_offsets_aliasing_and_values_match_pytorch_2_13(self):
        actual_cases = self.make_layout_cases(torch)
        expected_cases = self.make_layout_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source, dimension, sizes = actual_case
            expected_name, expected_source, expected_dimension, expected_sizes = (
                expected_case
            )
            self.assertEqual(
                (case, dimension, sizes),
                (expected_name, expected_dimension, expected_sizes),
            )
            for form in ("tuple", "list", "Size"):
                with self.subTest(case=case, form=form):
                    if form == "tuple":
                        actual_argument = tuple(sizes)
                        expected_argument = tuple(expected_sizes)
                    elif form == "list":
                        actual_argument = list(sizes)
                        expected_argument = list(expected_sizes)
                    else:
                        actual_argument = torch.Size(sizes)
                        expected_argument = reference_torch.Size(expected_sizes)
                    actual = self.layout_contract(
                        torch, actual_source, dimension, actual_argument
                    )
                    expected = self.layout_contract(
                        reference_torch,
                        expected_source,
                        expected_dimension,
                        expected_argument,
                    )
                    self.assertEqual(actual[:-1], expected[:-1])
                    np.testing.assert_array_equal(actual[-1], expected[-1])

    def lifetime_contract(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def make_view():
            return module.tensor(
                values.tolist(), dtype=module.float32
            )[1].unflatten(1, (2, 2))

        result = make_view()
        gc.collect()
        return (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            self.tensor_array(result, module).copy(),
        )

    def test_temporary_owner_lifetime_matches_pytorch_2_13(self):
        actual = self.lifetime_contract(torch)
        expected = self.lifetime_contract(reference_torch)
        self.assertEqual(actual[:-1], expected[:-1])
        np.testing.assert_array_equal(actual[-1], expected[-1])

    def autograd_contract(self, module):
        leaf = module.tensor(
            np.arange(12, dtype=np.float32).reshape(2, 6).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        result = leaf.unflatten(1, (2, 3))
        weights = module.tensor(
            np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3).tolist(),
            dtype=module.float32,
        )
        (result * weights).sum().backward()
        if module is reference_torch:
            node = type(result.grad_fn).__name__
        else:
            suffix = module._C._nn_functional_dropout_tensor_autograd_suffix(
                result
            )
            node = suffix.removeprefix(", grad_fn=<").removesuffix(">")
        return (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.data_ptr() == leaf.data_ptr(),
            result.requires_grad,
            result.is_leaf,
            node,
            self.tensor_array(leaf.grad, module).copy(),
        )

    def repeated_backward_contract(self, module):
        leaf = module.zeros((2, 6), dtype=module.float32, requires_grad=True)
        loss = leaf.unflatten(1, (2, 3)).sum()
        loss.backward()
        loss.backward()
        return self.tensor_array(leaf.grad, module).copy()

    def no_grad_contract(self, module):
        leaf = module.zeros((2, 6), dtype=module.float32, requires_grad=True)
        source = leaf.transpose(0, 1)
        with module.no_grad():
            result = source.unflatten(0, (2, 3))
        return (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.data_ptr() == source.data_ptr(),
            result.requires_grad,
            result.is_leaf,
            leaf.grad,
        )

    def test_view_backward_repeated_backward_and_no_grad_match(self):
        actual = self.autograd_contract(torch)
        expected = self.autograd_contract(reference_torch)
        self.assertEqual(actual[:-1], expected[:-1])
        np.testing.assert_array_equal(actual[-1], expected[-1])
        np.testing.assert_array_equal(
            self.repeated_backward_contract(torch),
            self.repeated_backward_contract(reference_torch),
        )
        self.assertEqual(
            self.no_grad_contract(torch),
            self.no_grad_contract(reference_torch),
        )

    def error(self, action, *, lines=None):
        try:
            action()
        except Exception as error:
            message = str(error)
            if lines is not None:
                message = "\n".join(message.splitlines()[:lines])
            return type(error).__name__, message
        self.fail("Tensor.unflatten unexpectedly accepted an invalid call")

    def test_binding_conversion_and_errors_match_pytorch_2_13(self):
        actual = torch.zeros((2, 3, 4), dtype=torch.float32)
        expected = reference_torch.zeros(
            (2, 3, 4), dtype=reference_torch.float32
        )
        exact_cases = (
            (lambda: actual.unflatten(True, (1, 3)), lambda: expected.unflatten(True, (1, 3))),
            (lambda: actual.unflatten(1.0, (1, 3)), lambda: expected.unflatten(1.0, (1, 3))),
            (lambda: actual.unflatten(1, 3), lambda: expected.unflatten(1, 3)),
            (lambda: actual.unflatten(1, (True, 3)), lambda: expected.unflatten(1, (True, 3))),
            (lambda: actual.unflatten(1, (1, 3.0)), lambda: expected.unflatten(1, (1, 3.0))),
            (lambda: actual.unflatten(3, (1, 3)), lambda: expected.unflatten(3, (1, 3))),
            (lambda: actual.unflatten(-4, (1, 3)), lambda: expected.unflatten(-4, (1, 3))),
            (lambda: actual.unflatten(-2, (1, 2)), lambda: expected.unflatten(-2, (1, 2))),
            (lambda: actual.unflatten(True, ()), lambda: expected.unflatten(True, ())),
            (lambda: actual.unflatten(True, None), lambda: expected.unflatten(True, None)),
            (lambda: actual.unflatten(), lambda: expected.unflatten()),
            (lambda: actual.unflatten(1), lambda: expected.unflatten(1)),
            (lambda: actual.unflatten(1, (1, 3), 0), lambda: expected.unflatten(1, (1, 3), 0)),
            (lambda: actual.unflatten(1, (1, 3), dim=1), lambda: expected.unflatten(1, (1, 3), dim=1)),
            (lambda: actual.unflatten(1, (1, 3), sizes=(1, 3)), lambda: expected.unflatten(1, (1, 3), sizes=(1, 3))),
            (lambda: actual.unflatten(1, (1, 3), unexpected=True), lambda: expected.unflatten(1, (1, 3), unexpected=True)),
        )
        for case, (actual_call, expected_call) in enumerate(exact_cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

        prefix_cases = (
            (lambda: actual.unflatten(1, (-1, -1)), lambda: expected.unflatten(1, (-1, -1))),
            (lambda: actual.unflatten(1, (-2,)), lambda: expected.unflatten(1, (-2,))),
            (
                lambda: torch.zeros((0,), dtype=torch.float32).unflatten(0, (0, -1)),
                lambda: reference_torch.zeros((0,), dtype=reference_torch.float32).unflatten(0, (0, -1)),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(prefix_cases):
            with self.subTest(prefix_case=case):
                self.assertEqual(
                    self.error(actual_call, lines=2),
                    self.error(expected_call, lines=2),
                )

        self.assertEqual(
            tuple(actual.unflatten(np.int64(-2), (np.int32(1), np.uint64(3))).shape),
            tuple(expected.unflatten(np.int64(-2), (np.int32(1), np.uint64(3))).shape),
        )
        actual_first, actual_second = IndexSize(1), IndexSize(3)
        expected_first, expected_second = IndexSize(1), IndexSize(3)
        self.assertEqual(
            tuple(actual.unflatten(1, (actual_first, actual_second)).shape),
            tuple(expected.unflatten(1, (expected_first, expected_second)).shape),
        )
        self.assertEqual(
            (actual_first.calls, actual_second.calls),
            (expected_first.calls, expected_second.calls),
        )

    def callable_contract(self, module):
        tensor = module.zeros((2, 3), dtype=module.float32)
        function = inspect.getattr_static(module.Tensor, "unflatten")
        bound = tensor.unflatten
        return {
            "function_type": type(function).__name__,
            "bound_type": type(bound).__name__,
            "is_python_function": type(function) is types.FunctionType,
            "is_bound_method": type(bound) is types.MethodType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__.replace("torch_rs", "torch"),
            "doc": function.__doc__,
            "annotations": function.__annotations__,
            "text_signature": getattr(function, "__text_signature__", "missing"),
            "signatures": (
                str(inspect.signature(function)),
                str(inspect.signature(bound)),
            ),
            "owned_by_tensor": "unflatten" in module.Tensor.__dict__,
            "module_identity": module._tensor.Tensor.unflatten is function,
            "keyword_shape": tuple(
                bound(dim=1, sizes=(1, 3)).shape
            ),
        }

    def test_python_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.zeros((2, 3), dtype=module.float32)
        function = inspect.getattr_static(module.Tensor, "unflatten")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        recording = RecordingMode(marker)
        with recording:
            intercepted = tensor.unflatten(dim=1, sizes=[1, 3])
        called_function, dispatch_types, args, kwargs = recording.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.unflatten(1, (1, 3))

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                tensor.unflatten(1, (1, 3))
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(
                    r"0x[0-9a-f]+",
                    "0x...",
                    str(error).replace("torch_rs", "torch"),
                ),
            )
        else:
            declining_error = None

        return {
            "intercepted": intercepted is marker,
            "call_count": len(recording.calls),
            "function_identity": called_function is function,
            "types": dispatch_types == (module.Tensor,),
            "args": (
                len(args) == 3
                and args[0] is tensor
                and args[1] == 1
                and args[2] == [1, 3]
            ),
            "kwargs": kwargs,
            "forwarding_order": order,
            "forwarded": (tuple(forwarded.shape), forwarded.stride()),
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
