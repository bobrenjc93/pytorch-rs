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
            raise AssertionError("top-level ravel differentials require PyTorch 2.13.0")

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
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

    def call(self, module, source, form):
        if form == "positional":
            return module.ravel(source)
        return module.ravel(**{form: source})

    def test_layouts_aliasing_offsets_and_lifetimes_match_pytorch_2_13(self):
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
                "strided vector",
                actual_base.transpose(0, 2)[0][0],
                expected_base.transpose(0, 2)[0][0],
            ),
            (
                "singleton stride",
                actual_singleton.transpose(0, 1)[2],
                expected_singleton.transpose(0, 1)[2],
            ),
            (
                "empty offset",
                torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1],
                reference_torch.zeros((2, 0, 3), requires_grad=True)
                .transpose(0, 2)[1],
            ),
        )
        retained = []
        forms = ("positional", "input", "x", "a", "x1")
        for index, (case, actual_source, expected_source) in enumerate(cases):
            form = forms[index % len(forms)]
            actual = self.call(torch, actual_source, form)
            expected = self.call(reference_torch, expected_source, form)
            self.assertIsNot(actual, actual_source)
            self.assertIsNot(expected, expected_source)
            self.assert_matches(actual, expected, case=(case, form))
            self.assertEqual(
                actual.is_set_to(actual_source.ravel()),
                expected.is_set_to(expected_source.ravel()),
            )
            retained.append((actual, expected))

        del actual_base, expected_base, actual_singleton, expected_singleton, cases
        gc.collect()
        self.assert_matches(retained[3][0], retained[3][1], case="lifetime view")
        self.assert_matches(retained[4][0], retained[4][1], case="lifetime copy")

    def autograd_observation(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        output = module.ravel(leaf.transpose(0, 1))
        output_state = (output.requires_grad, output.is_leaf)
        weights = module.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        (output * weights).sum().backward()
        gradient = np.asarray(leaf.grad).copy()

        scalar = module.tensor(2.0, requires_grad=True)
        (module.ravel(input=scalar) * 7.0).sum().backward()

        empty = module.zeros((2, 0, 3), requires_grad=True)
        module.ravel(x=empty).sum().backward()

        source = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with module.no_grad():
            alias = module.ravel(a=source)
            copied = module.ravel(x1=source.transpose(0, 1))
        no_grad_state = (
            alias.requires_grad,
            alias.is_leaf,
            copied.requires_grad,
            copied.is_leaf,
        )
        return (
            output_state,
            gradient,
            scalar.grad.item(),
            empty.grad.shape,
            np.asarray(empty.grad).copy(),
            no_grad_state,
        )

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        actual = self.autograd_observation(torch)
        expected = self.autograd_observation(reference_torch)
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2], expected[2])
        self.assertEqual(actual[3], expected[3])
        np.testing.assert_array_equal(actual[4], expected[4])
        self.assertEqual(actual[5], expected[5])

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

    def dispatch_observation(self, module):
        tensor = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        function = module.ravel
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for form in ("positional", "input", "x", "a", "x1"):
            mode = RecordingMode()
            with mode:
                result = self.call(module, tensor, form)
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types,
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    args[0] is tensor if args else kwargs[form] is tensor,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for form in ("positional", "input", "x", "a", "x1"):
            value = Override()
            Override.calls.clear()
            result = self.call(module, value, form)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    args[0] is value if args else kwargs[form] is value,
                )
            )

        forwarding_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=tensor)

        fallback_events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_events.append("override")
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback = function(FallbackOverride())

        invalid = []
        for call in (
            lambda: function(),
            lambda: function(tensor, tensor),
            lambda: function([], extra=True),
            lambda: function(tensor, extra=True),
        ):
            mode = RecordingMode()
            try:
                with mode:
                    call()
            except Exception as error:
                invalid.append((type(error).__name__, str(error), len(mode.calls)))

        return (
            mode_observations,
            override_observations,
            forwarding_order,
            forwarded.tolist(),
            fallback is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid,
        )

    def test_callable_modes_and_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        self.assert_error_matches(
            lambda: torch.ravel(DecliningOverride()),
            lambda: reference_torch.ravel(DecliningOverride()),
        )

    def test_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.ravel(), lambda: reference_torch.ravel()),
            (
                lambda: torch.ravel(actual, actual),
                lambda: reference_torch.ravel(expected, expected),
            ),
            (lambda: torch.ravel([1.0]), lambda: reference_torch.ravel([1.0])),
            (lambda: torch.ravel(input=1), lambda: reference_torch.ravel(input=1)),
            (
                lambda: torch.ravel(actual, input=actual),
                lambda: reference_torch.ravel(expected, input=expected),
            ),
            (
                lambda: torch.ravel(actual, extra=True),
                lambda: reference_torch.ravel(expected, extra=True),
            ),
            (
                lambda: torch.ravel(extra=actual, input=actual),
                lambda: reference_torch.ravel(extra=expected, input=expected),
            ),
            (
                lambda: torch.ravel(extra=actual),
                lambda: reference_torch.ravel(extra=expected),
            ),
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
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
