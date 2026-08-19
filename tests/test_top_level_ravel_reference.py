import gc
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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelRavelReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("ravel differentials require pinned PyTorch 2.13.0")

    def assert_matches(self, actual, expected, actual_source, expected_source, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertEqual(
                actual.data_ptr() == actual_source.data_ptr(),
                expected.data_ptr() == expected_source.data_ptr(),
            )
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_array_equal(
                np.asarray(actual), expected.detach().cpu().numpy()
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

    def test_layouts_aliasing_call_forms_and_lifetimes_match_pytorch_2_13(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_base = torch.tensor(values.tolist(), requires_grad=True)
        expected_base = reference_torch.tensor(values, requires_grad=True)
        actual_singleton = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        expected_singleton = reference_torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        cases = (
            ("scalar", actual_base[0][0][0], expected_base[0][0][0]),
            ("vector", actual_base[0][1], expected_base[0][1]),
            ("ordinary", actual_base, expected_base),
            ("offset", actual_base[1], expected_base[1]),
            (
                "transpose",
                actual_base.transpose(0, 2),
                expected_base.transpose(0, 2),
            ),
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
        for index, (case, actual_source, expected_source) in enumerate(cases):
            form = index % 5
            if form == 0:
                actual = torch.ravel(actual_source)
                expected = reference_torch.ravel(expected_source)
            elif form == 1:
                actual = torch.ravel(input=actual_source)
                expected = reference_torch.ravel(input=expected_source)
            elif form == 2:
                actual = torch.ravel(x=actual_source)
                expected = reference_torch.ravel(x=expected_source)
            elif form == 3:
                actual = torch.ravel(a=actual_source)
                expected = reference_torch.ravel(a=expected_source)
            else:
                actual = torch.ravel(x1=actual_source)
                expected = reference_torch.ravel(x1=expected_source)
            self.assertIsNot(actual, actual_source)
            self.assertIsNot(expected, expected_source)
            self.assert_matches(actual, expected, actual_source, expected_source, case)
            retained.append((actual, expected))

        del actual_base, expected_base, actual_singleton, expected_singleton, cases
        gc.collect()
        np.testing.assert_array_equal(
            np.asarray(retained[3][0]), retained[3][1].detach().cpu().numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(retained[4][0]), retained[4][1].detach().cpu().numpy()
        )

    def test_autograd_empty_backward_and_no_grad_match_pytorch_2_13(self):
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
            module.ravel(x=empty).sum().backward()
            empty_gradients.append((empty.grad.shape, np.asarray(empty.grad).copy()))

            source = module.tensor(
                [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
            )
            with module.no_grad():
                alias = module.ravel(a=source)
                copied = module.ravel(x1=source.transpose(0, 1))
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

    def test_binding_and_type_errors_match_pytorch_2_13(self):
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
                lambda: torch.ravel(actual, x=actual),
                lambda: reference_torch.ravel(expected, x=expected),
            ),
            (
                lambda: torch.ravel(extra=actual),
                lambda: reference_torch.ravel(extra=expected),
            ),
            (lambda: torch.ravel(1), lambda: reference_torch.ravel(1)),
            (
                lambda: torch.ravel(input=1),
                lambda: reference_torch.ravel(input=1),
            ),
            (
                lambda: torch.ravel(x=actual, extra=True),
                lambda: reference_torch.ravel(x=expected, extra=True),
            ),
            (
                lambda: torch.ravel(extra=True, x=actual),
                lambda: reference_torch.ravel(extra=True, x=expected),
            ),
            (
                lambda: torch.ravel(input=actual, x=actual),
                lambda: reference_torch.ravel(input=expected, x=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def dispatch_contract(self, module):
        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                override_calls.append((func, dispatch_types, args, kwargs))
                return "override"

        value = Override()
        override_result = module.ravel(x=value)
        override_function, override_types, override_args, override_kwargs = (
            override_calls[0]
        )

        mode_calls = []

        class Mode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                mode_calls.append((func, dispatch_types, args, kwargs))
                return "mode"

        tensor = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        with Mode():
            mode_result = module.ravel(input=tensor)
        mode_function, mode_types, mode_args, mode_kwargs = mode_calls[0]

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = module.ravel(tensor)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        try:
            module.ravel(DecliningOverride())
        except Exception as error:
            decline_error = (type(error).__name__, str(error))
        else:
            decline_error = None

        return {
            "override_result": override_result,
            "override_function": override_function is module.ravel,
            "override_types": tuple(value.__name__ for value in override_types),
            "override_args": len(override_args),
            "override_kwargs": tuple(override_kwargs),
            "override_value_identity": override_kwargs["x"] is value,
            "mode_result": mode_result,
            "mode_function": mode_function is module.ravel,
            "mode_types": tuple(value.__name__ for value in mode_types),
            "mode_args": len(mode_args),
            "mode_kwargs": tuple(mode_kwargs),
            "mode_value_identity": mode_kwargs["input"] is tensor,
            "forwarded_shape": tuple(forwarded.shape),
            "forwarded_stride": forwarded.stride(),
            "forwarded_values": np.asarray(forwarded).tolist(),
            "decline_error": decline_error,
        }

    def test_overrides_and_modes_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_contract(torch),
            self.dispatch_contract(reference_torch),
        )

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

    def test_callable_metadata_documentation_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
