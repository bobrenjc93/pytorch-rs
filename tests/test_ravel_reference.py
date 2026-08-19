import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorRavelReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_array_equal(
                np.asarray(actual), expected.cpu().detach().numpy()
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_layouts_identity_aliasing_and_lifetimes_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_base = torch.tensor(values.tolist(), requires_grad=True)
        expected_base = reference_torch.tensor(values, requires_grad=True)
        actual_singleton_base = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        expected_singleton_base = reference_torch.tensor([[0.0, 1.0, 2.0, 3.0]])

        cases = (
            ("scalar", actual_base[0][0][0], expected_base[0][0][0]),
            ("vector", actual_base[0][1], expected_base[0][1]),
            ("ordinary", actual_base, expected_base),
            ("offset", actual_base[1], expected_base[1]),
            ("transpose", actual_base.transpose(0, 2), expected_base.transpose(0, 2)),
            (
                "strided-vector",
                actual_base.transpose(0, 2)[0][0],
                expected_base.transpose(0, 2)[0][0],
            ),
            (
                "singleton-stride",
                actual_singleton_base.transpose(0, 1)[2],
                expected_singleton_base.transpose(0, 1)[2],
            ),
            (
                "empty-offset",
                torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1],
                reference_torch.zeros((2, 0, 3), requires_grad=True)
                .transpose(0, 2)[1],
            ),
        )
        retained = []
        for case, actual_source, expected_source in cases:
            actual = actual_source.ravel()
            expected = expected_source.ravel()
            self.assertIsNot(actual, actual_source)
            self.assertIsNot(expected, expected_source)
            self.assert_matches(actual, expected, case=case)
            self.assertEqual(
                expected.untyped_storage().data_ptr()
                == expected_source.untyped_storage().data_ptr(),
                expected_source.is_contiguous(),
            )
            retained.append((actual, expected))

        del actual_base, expected_base, actual_singleton_base, expected_singleton_base, cases
        self.assert_matches(retained[4][0], retained[4][1], case="lifetime-copy")
        self.assert_matches(retained[3][0], retained[3][1], case="lifetime-view")

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        gradients = []
        states = []
        scalar_gradients = []
        empty_gradients = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            output = leaf.transpose(0, 1).ravel()
            states.append((output.requires_grad, output.is_leaf))
            weights = module.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            (output * weights).sum().backward()
            gradients.append(np.asarray(leaf.grad).copy())

            scalar = module.tensor(2.0, requires_grad=True)
            (scalar.ravel() * 7.0).sum().backward()
            scalar_gradients.append(scalar.grad.item())

            empty = module.zeros((2, 0, 3), requires_grad=True)
            empty.ravel().sum().backward()
            empty_gradients.append((empty.grad.shape, np.asarray(empty.grad).copy()))

            source = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            non_contiguous = source.transpose(0, 1)
            with module.no_grad():
                alias = source.ravel()
                copied = non_contiguous.ravel()
            states.append(
                (
                    alias.requires_grad,
                    alias.is_leaf,
                    copied.requires_grad,
                    copied.is_leaf,
                )
            )

        np.testing.assert_array_equal(gradients[0], gradients[1])
        self.assertEqual(states[0], states[2])
        self.assertEqual(states[1], states[3])
        self.assertEqual(scalar_gradients[0], scalar_gradients[1])
        self.assertEqual(empty_gradients[0][0], empty_gradients[1][0])
        np.testing.assert_array_equal(empty_gradients[0][1], empty_gradients[1][1])

    def test_descriptor_and_argument_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 3))
        expected = reference_torch.zeros((2, 3))
        actual_descriptor = inspect.getattr_static(torch.Tensor, "ravel")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "ravel")
        for descriptor in (actual_descriptor, expected_descriptor):
            self.assertIs(type(descriptor), types.MethodDescriptorType)
            self.assertEqual(descriptor.__name__, "ravel")
            assert_no_argument_signature(self, descriptor, "(self, /)")
        self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)

        for bound in (actual.ravel, expected.ravel):
            self.assertIs(type(bound), types.BuiltinMethodType)
            assert_no_argument_signature(self, bound, "()")

        self.assert_matches(
            actual_descriptor(actual), expected_descriptor(expected), case="unbound-call"
        )

        actual_bound = actual.ravel
        expected_bound = expected.ravel
        for actual_call, expected_call in (
            (lambda: actual_bound(1), lambda: expected_bound(1)),
            (lambda: actual_bound(1, 2), lambda: expected_bound(1, 2)),
            (lambda: actual_bound(dim=0), lambda: expected_bound(dim=0)),
            (lambda: actual_bound(input=actual), lambda: expected_bound(input=expected)),
        ):
            self.assert_error_matches(actual_call, expected_call)

        for descriptor in (actual_descriptor, expected_descriptor):
            with self.assertRaises(TypeError):
                descriptor()
            with self.assertRaises(TypeError):
                descriptor([1.0])
            with self.assertRaises(TypeError):
                descriptor(actual if descriptor is actual_descriptor else expected, 1)

    def test_top_level_layouts_call_forms_and_lifetimes_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_base = torch.tensor(values.tolist(), requires_grad=True)
        expected_base = reference_torch.tensor(values, requires_grad=True)
        actual_singleton = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        expected_singleton = reference_torch.tensor(
            [[0.0, 1.0, 2.0, 3.0]], dtype=reference_torch.float32
        )
        cases = (
            ("scalar", actual_base[0][0][0], expected_base[0][0][0]),
            ("vector", actual_base[0][1], expected_base[0][1]),
            ("ordinary", actual_base, expected_base),
            ("offset", actual_base[1], expected_base[1]),
            ("transpose", actual_base.transpose(0, 2), expected_base.transpose(0, 2)),
            (
                "strided-vector",
                actual_base.transpose(0, 2)[0][0],
                expected_base.transpose(0, 2)[0][0],
            ),
            (
                "singleton-stride",
                actual_singleton.transpose(0, 1)[2],
                expected_singleton.transpose(0, 1)[2],
            ),
            (
                "empty-offset",
                torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1],
                reference_torch.zeros((2, 0, 3), requires_grad=True)
                .transpose(0, 2)[1],
            ),
        )
        retained = []
        for case, actual_source, expected_source in cases:
            for keyword in (None, "input", "x", "a", "x1"):
                actual = (
                    torch.ravel(actual_source)
                    if keyword is None
                    else torch.ravel(**{keyword: actual_source})
                )
                expected = (
                    reference_torch.ravel(expected_source)
                    if keyword is None
                    else reference_torch.ravel(**{keyword: expected_source})
                )
                self.assertIsNot(actual, actual_source)
                self.assertIsNot(expected, expected_source)
                self.assert_matches(actual, expected, case=(case, keyword))
                self.assertEqual(
                    actual.is_set_to(actual_source.ravel()),
                    expected.is_set_to(expected_source.ravel()),
                )
                retained.append((actual, expected))

        del actual_base, expected_base, actual_singleton, expected_singleton, cases
        self.assert_matches(retained[-6][0], retained[-6][1], case="lifetime-view")
        self.assert_matches(retained[-11][0], retained[-11][1], case="lifetime-copy")

    def test_top_level_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        gradients = []
        states = []
        scalar_gradients = []
        empty_gradients = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            output = module.ravel(leaf.transpose(0, 1))
            states.append((output.requires_grad, output.is_leaf))
            weights = module.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            (output * weights).sum().backward()
            gradients.append(np.asarray(leaf.grad).copy())

            scalar = module.tensor(2.0, requires_grad=True)
            (module.ravel(input=scalar) * 7.0).sum().backward()
            scalar_gradients.append(scalar.grad.item())

            empty = module.zeros((2, 0, 3), requires_grad=True)
            module.ravel(a=empty).sum().backward()
            empty_gradients.append((empty.grad.shape, np.asarray(empty.grad).copy()))

            source = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            with module.no_grad():
                alias = module.ravel(source)
                copied = module.ravel(source.transpose(0, 1))
            states.append(
                (
                    alias.requires_grad,
                    alias.is_leaf,
                    copied.requires_grad,
                    copied.is_leaf,
                )
            )

        np.testing.assert_array_equal(gradients[0], gradients[1])
        self.assertEqual(states[0], states[2])
        self.assertEqual(states[1], states[3])
        self.assertEqual(scalar_gradients[0], scalar_gradients[1])
        self.assertEqual(empty_gradients[0][0], empty_gradients[1][0])
        np.testing.assert_array_equal(empty_gradients[0][1], empty_gradients[1][1])

    def top_level_callable_contract(self, module):
        function = module.ravel
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.ravel is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("ravel"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["ravel"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_top_level_callable_and_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        call_pairs = (
            (lambda: torch.ravel(), lambda: reference_torch.ravel()),
            (
                lambda: torch.ravel(actual, actual),
                lambda: reference_torch.ravel(expected, expected),
            ),
            (
                lambda: torch.ravel(actual, input=actual),
                lambda: reference_torch.ravel(expected, input=expected),
            ),
            (
                lambda: torch.ravel(actual, extra=True, input=actual),
                lambda: reference_torch.ravel(expected, extra=True, input=expected),
            ),
            (
                lambda: torch.ravel(actual, input=actual, extra=True),
                lambda: reference_torch.ravel(expected, input=expected, extra=True),
            ),
            (
                lambda: torch.ravel(extra=actual),
                lambda: reference_torch.ravel(extra=expected),
            ),
            (
                lambda: torch.ravel(1, extra=True),
                lambda: reference_torch.ravel(1, extra=True),
            ),
            (lambda: torch.ravel(input=[]), lambda: reference_torch.ravel(input=[])),
            (lambda: torch.ravel(a=1), lambda: reference_torch.ravel(a=1)),
            (lambda: torch.ravel(x=[]), lambda: reference_torch.ravel(x=[])),
            (lambda: torch.ravel(x1=None), lambda: reference_torch.ravel(x1=None)),
            (
                lambda: torch.ravel(a=actual, x=actual),
                lambda: reference_torch.ravel(a=expected, x=expected),
            ),
            (
                lambda: torch.ravel(x=actual, a=actual),
                lambda: reference_torch.ravel(x=expected, a=expected),
            ),
            (
                lambda: torch.ravel(input=actual, x1=actual),
                lambda: reference_torch.ravel(input=expected, x1=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(call_pairs):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def top_level_dispatch_observation(self, module, keyword):
        tensor = module.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=module.float32)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = (
                module.ravel(tensor)
                if keyword is None
                else module.ravel(**{keyword: tensor})
            )
        function, dispatch_types, args, kwargs = mode.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = module.ravel(a=tensor)

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        override_result = module.ravel(x=value)
        override_function, override_types, override_args, override_kwargs = override_calls[0]
        return {
            "intercepted": result is marker,
            "function": function is module.ravel,
            "types": dispatch_types,
            "args": args == ((tensor,) if keyword is None else ()),
            "kwargs": kwargs is None
            if keyword is None
            else kwargs == {keyword: tensor},
            "forwarding_order": order,
            "forwarded_shape": tuple(forwarded.shape),
            "forwarded_stride": forwarded.stride(),
            "forwarded_values": forwarded.tolist(),
            "override_result": override_result is marker,
            "override_function": override_function is module.ravel,
            "override_types": override_types == (Override,),
            "override_args": override_args == (),
            "override_kwargs": override_kwargs == {"x": value},
        }

    def test_top_level_torch_function_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for keyword in (None, "input", "x", "a", "x1"):
            with self.subTest(keyword=keyword):
                self.assertEqual(
                    self.top_level_dispatch_observation(torch, keyword),
                    self.top_level_dispatch_observation(reference_torch, keyword),
                )


if __name__ == "__main__":
    unittest.main()
