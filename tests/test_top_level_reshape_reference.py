import gc
import inspect
import pickle
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelReshapeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("reshape differentials require pinned PyTorch 2.13.0")

    def assert_matches(self, actual, expected, actual_source, expected_source, case):
        with self.subTest(case=case, metadata=True):
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
        with self.subTest(case=case, values=True):
            if actual.numel() == 0:
                self.assertEqual(actual.tolist(), expected.tolist())
            else:
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
        actual_noncontiguous = actual_base.transpose(0, 1)
        expected_noncontiguous = expected_base.transpose(0, 1)
        cases = (
            ("scalar", torch.tensor(-0.0), reference_torch.tensor(-0.0), ()),
            ("contiguous", actual_base, expected_base, (6, 4)),
            ("contiguous-offset", actual_base[1], expected_base[1], (2, 6)),
            (
                "noncontiguous-same-shape",
                actual_noncontiguous,
                expected_noncontiguous,
                (3, 2, 4),
            ),
            (
                "noncontiguous-copy",
                actual_base.transpose(0, 2),
                expected_base.transpose(0, 2),
                (6, 4),
            ),
            (
                "empty-offset",
                torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1],
                reference_torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1],
                (2, 0),
            ),
            (
                "extreme-empty",
                torch.zeros((0,)),
                reference_torch.zeros((0,), dtype=reference_torch.float32),
                (0, sys.maxsize, 3),
            ),
        )

        retained = []
        for index, (case, actual_source, expected_source, shape) in enumerate(cases):
            style = index % 7
            if style == 0:
                actual = torch.reshape(actual_source, shape)
                expected = reference_torch.reshape(expected_source, shape)
            elif style == 1:
                actual = torch.reshape(actual_source, list(shape))
                expected = reference_torch.reshape(expected_source, list(shape))
            elif style == 2:
                actual = torch.reshape(actual_source, torch.Size(shape))
                expected = reference_torch.reshape(
                    expected_source, reference_torch.Size(shape)
                )
            elif style == 3:
                actual = torch.reshape(input=actual_source, shape=shape)
                expected = reference_torch.reshape(input=expected_source, shape=shape)
            elif style == 4:
                actual = torch.reshape(x=actual_source, shape=list(shape))
                expected = reference_torch.reshape(x=expected_source, shape=list(shape))
            elif style == 5:
                actual = torch.reshape(a=actual_source, shape=shape)
                expected = reference_torch.reshape(a=expected_source, shape=shape)
            else:
                actual = torch.reshape(x1=actual_source, shape=torch.Size(shape))
                expected = reference_torch.reshape(
                    x1=expected_source, shape=reference_torch.Size(shape)
                )
            self.assert_matches(
                actual, expected, actual_source, expected_source, case
            )
            retained.append((actual, expected))

        del actual_base, expected_base, actual_noncontiguous, expected_noncontiguous
        del cases
        gc.collect()
        for actual, expected in retained:
            if actual.numel() == 0:
                self.assertEqual(actual.tolist(), expected.tolist())
            else:
                np.testing.assert_array_equal(
                    np.asarray(actual), expected.detach().cpu().numpy()
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
            output = module.reshape(leaf.transpose(0, 1), (6,))
            states.append((output.requires_grad, output.is_leaf))
            weights = module.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            (output * weights).sum().backward()
            gradients.append(np.asarray(leaf.grad).copy())

            scalar = module.tensor(2.0, requires_grad=True)
            (module.reshape(input=scalar, shape=()) * 7.0).sum().backward()
            scalar_gradients.append(scalar.grad.item())

            empty = module.zeros((2, 0, 3), requires_grad=True)
            module.reshape(x=empty, shape=(0, 6)).sum().backward()
            empty_gradients.append((empty.grad.shape, np.asarray(empty.grad).copy()))

            source = module.tensor(
                [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
            )
            with module.no_grad():
                alias = module.reshape(source, (4,))
                copied = module.reshape(source.transpose(0, 1), (4,))
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

    def test_binding_shape_and_error_paths_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_two = torch.zeros((2,))
        expected_two = reference_torch.zeros((2,))
        cases = (
            (lambda: torch.reshape(), lambda: reference_torch.reshape()),
            (lambda: torch.reshape(actual), lambda: reference_torch.reshape(expected)),
            (
                lambda: torch.reshape(shape=(1,)),
                lambda: reference_torch.reshape(shape=(1,)),
            ),
            (
                lambda: torch.reshape(actual, (1,), None),
                lambda: reference_torch.reshape(expected, (1,), None),
            ),
            (
                lambda: torch.reshape(actual, (1,), input=actual),
                lambda: reference_torch.reshape(expected, (1,), input=expected),
            ),
            (
                lambda: torch.reshape(actual, (1,), shape=(1,)),
                lambda: reference_torch.reshape(expected, (1,), shape=(1,)),
            ),
            (
                lambda: torch.reshape(actual, (1,), extra=True),
                lambda: reference_torch.reshape(expected, (1,), extra=True),
            ),
            (
                lambda: torch.reshape(x=actual, shape=(1,), extra=True),
                lambda: reference_torch.reshape(x=expected, shape=(1,), extra=True),
            ),
            (
                lambda: torch.reshape(1, (1,)),
                lambda: reference_torch.reshape(1, (1,)),
            ),
            (
                lambda: torch.reshape(input=1, shape=(1,)),
                lambda: reference_torch.reshape(input=1, shape=(1,)),
            ),
            (
                lambda: torch.reshape(actual, 1),
                lambda: reference_torch.reshape(expected, 1),
            ),
            (
                lambda: torch.reshape(input=actual, shape=1),
                lambda: reference_torch.reshape(input=expected, shape=1),
            ),
            (
                lambda: torch.reshape(actual, (1.0,)),
                lambda: reference_torch.reshape(expected, (1.0,)),
            ),
            (
                lambda: torch.reshape(input=actual, shape=(1.0,)),
                lambda: reference_torch.reshape(input=expected, shape=(1.0,)),
            ),
            (
                lambda: torch.reshape(actual_two, (1, 1.0)),
                lambda: reference_torch.reshape(expected_two, (1, 1.0)),
            ),
            (
                lambda: torch.reshape(actual_two, (True, 2)),
                lambda: reference_torch.reshape(expected_two, (True, 2)),
            ),
            (
                lambda: torch.reshape(actual_two, (2, 2)),
                lambda: reference_torch.reshape(expected_two, (2, 2)),
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

        input_value = Override()
        input_result = module.reshape(x=input_value, shape=(1,))
        input_function, input_types, input_args, input_kwargs = override_calls[0]

        tensor = module.tensor([1.0, 2.0])
        shape_value = Override()
        override_calls.clear()
        shape_result = module.reshape(tensor, shape_value)
        shape_function, shape_types, shape_args, shape_kwargs = override_calls[0]

        mode_calls = []

        class Mode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                mode_calls.append((func, dispatch_types, args, kwargs))
                return "mode"

        with Mode():
            mode_result = module.reshape(input=tensor, shape=(2, 1))
        mode_function, mode_types, mode_args, mode_kwargs = mode_calls[0]

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = module.reshape(tensor, (2, 1))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        try:
            module.reshape(DecliningOverride(), (1,))
        except Exception as error:
            decline_error = (type(error).__name__, str(error))
        else:
            decline_error = None

        return {
            "input_result": input_result,
            "input_function": input_function is module.reshape,
            "input_types": tuple(value.__name__ for value in input_types),
            "input_args": len(input_args),
            "input_kwargs": tuple(input_kwargs),
            "input_identity": input_kwargs["x"] is input_value,
            "shape_result": shape_result,
            "shape_function": shape_function is module.reshape,
            "shape_types": tuple(value.__name__ for value in shape_types),
            "shape_args": len(shape_args),
            "shape_kwargs": shape_kwargs,
            "shape_identity": shape_args[1] is shape_value,
            "mode_result": mode_result,
            "mode_function": mode_function is module.reshape,
            "mode_types": tuple(mode_types),
            "mode_args": len(mode_args),
            "mode_kwargs": tuple(mode_kwargs),
            "forwarded_shape": tuple(forwarded.shape),
            "forwarded_values": tuple(np.asarray(forwarded).reshape(-1)),
            "decline_error": decline_error,
        }

    def test_override_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(self.dispatch_contract(torch), self.dispatch_contract(reference_torch))

    def test_callable_metadata_documentation_ownership_exports_and_pickling_match(self):
        actual = torch.reshape
        expected = reference_torch.reshape
        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__module__, expected.__module__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(actual)

        actual_owner = actual.__reduce__()[1][0]
        self.assertEqual(actual_owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(actual_owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(actual_owner.__module__, "torch_rs._C")
        self.assertIs(actual_owner, torch._C._VariableFunctionsClass)
        self.assertIs(actual_owner.reshape, actual)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)), actual
                )

        self.assertEqual(torch.__all__.count("reshape"), reference_torch.__all__.count("reshape"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["reshape"], actual)


if __name__ == "__main__":
    unittest.main()
