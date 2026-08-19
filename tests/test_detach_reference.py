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
class TopLevelDetachReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("detach differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_detach_matches(self, actual, expected, actual_source, expected_source):
        self.assertIsNot(actual, actual_source)
        self.assertIsNot(expected, expected_source)
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
        self.assertIs(actual.dtype, torch.float32)
        self.assertEqual(actual.device, torch.device("cpu"))
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertTrue(actual_source.is_set_to(actual))
        self.assertTrue(expected_source.is_set_to(expected))
        np.testing.assert_array_equal(np.asarray(actual), expected.numpy())

    def test_layout_alias_call_forms_and_lifetimes_match_pytorch_2_13(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_base = torch.tensor(values.tolist(), requires_grad=True)
        expected_base = reference_torch.tensor(values, requires_grad=True)
        cases = (
            (
                "scalar",
                actual_base[0][0][0],
                expected_base[0][0][0],
            ),
            ("ordinary", actual_base, expected_base),
            ("offset", actual_base[1], expected_base[1]),
            (
                "strided-offset",
                actual_base.transpose(0, 2)[1],
                expected_base.transpose(0, 2)[1],
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
                actual = torch.detach(actual_source)
                expected = reference_torch.detach(expected_source)
            elif form == 1:
                actual = torch.detach(input=actual_source)
                expected = reference_torch.detach(input=expected_source)
            elif form == 2:
                actual = torch.detach(x=actual_source)
                expected = reference_torch.detach(x=expected_source)
            elif form == 3:
                actual = torch.detach(a=actual_source)
                expected = reference_torch.detach(a=expected_source)
            else:
                actual = torch.detach(x1=actual_source)
                expected = reference_torch.detach(x1=expected_source)
            with self.subTest(case=case, form=form):
                self.assert_detach_matches(
                    actual, expected, actual_source, expected_source
                )
            retained.append((actual, expected))

        del actual_base, expected_base, cases
        gc.collect()
        for actual, expected in retained:
            np.testing.assert_array_equal(np.asarray(actual), expected.numpy())

    def test_autograd_boundary_preserves_source_graph_like_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
            )
            source = (leaf * 3.0).transpose(0, 1)[1]
            detached = module.detach(source)
            method_detached = source.detach()
            detached_loss = (detached * detached).sum()
            source.sum().backward()
            outcomes.append(
                (
                    source.requires_grad,
                    detached.requires_grad,
                    detached.is_leaf,
                    detached.shape == method_detached.shape,
                    detached.stride() == method_detached.stride(),
                    detached.storage_offset() == method_detached.storage_offset(),
                    detached_loss.requires_grad,
                    np.asarray(leaf.grad).copy(),
                )
            )

        self.assertEqual(outcomes[0][:-1], outcomes[1][:-1])
        np.testing.assert_array_equal(outcomes[0][-1], outcomes[1][-1])

    def dispatch_contract(self, module):
        marker = object()
        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        for form, call in (
            ("positional", lambda value: module.detach(value)),
            ("input", lambda value: module.detach(input=value)),
            ("x", lambda value: module.detach(x=value)),
            ("a", lambda value: module.detach(a=value)),
            ("x1", lambda value: module.detach(x1=value)),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            function, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    form,
                    result is marker,
                    function is module.detach,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                )
            )

        mode_observations = []

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        tensor = module.tensor([1.0], requires_grad=True)
        for form, call in (
            ("positional", lambda: module.detach(tensor)),
            ("input", lambda: module.detach(input=tensor)),
            ("x", lambda: module.detach(x=tensor)),
            ("a", lambda: module.detach(a=tensor)),
            ("x1", lambda: module.detach(x1=tensor)),
        ):
            mode = Mode()
            with mode:
                result = call()
            function, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    form,
                    result is marker,
                    function is module.detach,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                )
            )

        forwarding_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = module.detach(x=tensor)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        try:
            module.detach(DecliningOverride())
        except Exception as error:
            decline_error = (type(error).__name__, str(error))
        else:
            decline_error = None

        return {
            "override_observations": override_observations,
            "mode_observations": mode_observations,
            "forwarding_order": forwarding_order,
            "forwarded_shape": tuple(forwarded.shape),
            "forwarded_stride": forwarded.stride(),
            "forwarded_offset": forwarded.storage_offset(),
            "forwarded_requires_grad": forwarded.requires_grad,
            "decline_error": decline_error,
        }

    def test_modes_subclass_overrides_and_not_implemented_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_contract(torch),
            self.dispatch_contract(reference_torch),
        )

    def callable_contract(self, module):
        function = module.detach
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
            "owner_callable_identity": owner.detach is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("detach"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["detach"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_ownership_pickling_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def test_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.detach(), lambda: reference_torch.detach()),
            (
                lambda: torch.detach(actual, actual),
                lambda: reference_torch.detach(expected, expected),
            ),
            (
                lambda: torch.detach(actual, input=actual),
                lambda: reference_torch.detach(expected, input=expected),
            ),
            (
                lambda: torch.detach(actual, x=actual),
                lambda: reference_torch.detach(expected, x=expected),
            ),
            (
                lambda: torch.detach(foo=actual),
                lambda: reference_torch.detach(foo=expected),
            ),
            (lambda: torch.detach(None), lambda: reference_torch.detach(None)),
            (
                lambda: torch.detach(input=1),
                lambda: reference_torch.detach(input=1),
            ),
            (
                lambda: torch.detach(x=actual, extra=True),
                lambda: reference_torch.detach(x=expected, extra=True),
            ),
            (
                lambda: torch.detach(extra=True, x=actual),
                lambda: reference_torch.detach(extra=True, x=expected),
            ),
            (
                lambda: torch.detach(input=actual, x=actual),
                lambda: reference_torch.detach(input=expected, x=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
