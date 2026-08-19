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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelRavelReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("top-level ravel differentials require PyTorch 2.13.0")

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

    def call_ravel(self, module, source, form):
        if form == "positional":
            return module.ravel(source)
        return module.ravel(**{form: source})

    def test_layouts_call_forms_aliasing_and_lifetimes_match_pytorch_2_13(self):
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
            for form in ("positional", "input", "x", "a", "x1"):
                actual = self.call_ravel(torch, actual_source, form)
                expected = self.call_ravel(reference_torch, expected_source, form)
                self.assertIsNot(actual, actual_source)
                self.assertIsNot(expected, expected_source)
                self.assert_matches(actual, expected, case=(case, form))
                self.assertEqual(
                    actual.data_ptr() == actual_source.data_ptr(),
                    expected.data_ptr() == expected_source.data_ptr(),
                )
                if form == "positional":
                    retained.append((actual, expected))

        del actual_base, expected_base, actual_singleton_base, expected_singleton_base, cases
        self.assert_matches(retained[4][0], retained[4][1], case="lifetime-copy")
        self.assert_matches(retained[3][0], retained[3][1], case="lifetime-view")

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
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
            non_contiguous = source.transpose(0, 1)
            with module.no_grad():
                alias = module.ravel(x=source)
                copied = module.ravel(x1=non_contiguous)
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

    def callable_contract(self, module):
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

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    def test_binding_and_error_precedence_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
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
            (lambda: torch.ravel(1, extra=True), lambda: reference_torch.ravel(1, extra=True)),
            (lambda: torch.ravel(input=[]), lambda: reference_torch.ravel(input=[])),
            (lambda: torch.ravel(a=1), lambda: reference_torch.ravel(a=1)),
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
            (
                lambda: torch.ravel(actual, out=actual),
                lambda: reference_torch.ravel(expected, out=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def dispatch_observation(self, module):
        marker = object()
        observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for keyword in (None, "input", "x", "a", "x1"):
            value = Override()
            Override.calls.clear()
            result = (
                module.ravel(value)
                if keyword is None
                else module.ravel(**{keyword: value})
            )
            func, dispatch_types, args, kwargs = Override.calls[0]
            observations.append(
                (
                    "override",
                    keyword,
                    result is marker,
                    func is module.ravel,
                    dispatch_types == (Override,),
                    args == ((value,) if keyword is None else ()),
                    kwargs is None if keyword is None else kwargs == {keyword: value},
                )
            )

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        tensor = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        for keyword in (None, "input", "x", "a", "x1"):
            mode = Mode()
            with mode:
                result = (
                    module.ravel(tensor)
                    if keyword is None
                    else module.ravel(**{keyword: tensor})
                )
            func, dispatch_types, args, kwargs = mode.calls[0]
            observations.append(
                (
                    "mode",
                    keyword,
                    result is marker,
                    func is module.ravel,
                    dispatch_types,
                    len(args),
                    kwargs is None if keyword is None else tuple(kwargs),
                )
            )
        return observations

    def test_override_and_mode_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def forwarding_observation(self, module):
        order = []

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        tensor = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        with Mode("lower"):
            with Mode("upper"):
                result = module.ravel(tensor)
        return (
            order,
            result.shape,
            result.stride(),
            result.storage_offset(),
            result.data_ptr() == tensor.data_ptr(),
            result is tensor,
        )

    def declining_observation(self, module):
        marker = object()
        events = []

        class Mode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append("mode")
                return NotImplemented

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append("override")
                return marker

        with Mode():
            result = module.ravel(Override())
        return result is marker, events

    def test_forwarding_and_declining_modes_match_pytorch_2_13(self):
        self.assertEqual(
            self.forwarding_observation(torch),
            self.forwarding_observation(reference_torch),
        )
        self.assertEqual(
            self.declining_observation(torch),
            self.declining_observation(reference_torch),
        )

    def test_not_implemented_override_error_matches_pytorch_2_13(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        self.assert_error_matches(
            lambda: torch.ravel(Override()),
            lambda: reference_torch.ravel(Override()),
        )


if __name__ == "__main__":
    unittest.main()
