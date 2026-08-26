import copy
import inspect
import pickle
import re
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelNumelReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("numel differentials require pinned PyTorch 2.13.0")

    def make_cases(self, module):
        base = module.tensor(
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
                [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]],
                [[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
            ],
            dtype=module.float32,
        )
        return (
            module.tensor(-0.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            base[1],
            base.transpose(0, 2),
            module.zeros((0,), dtype=module.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2),
        )

    def metadata(self, tensor):
        return (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )

    def test_cardinality_call_forms_and_metadata_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            actual_metadata = self.metadata(actual)
            expected_metadata = self.metadata(expected)
            for keyword in (None, "input", "x", "a", "x1"):
                with self.subTest(
                    case=case,
                    keyword=keyword,
                    shape=actual.shape,
                    stride=actual.stride(),
                ):
                    actual_result = (
                        torch.numel(actual)
                        if keyword is None
                        else torch.numel(**{keyword: actual})
                    )
                    expected_result = (
                        reference_torch.numel(expected)
                        if keyword is None
                        else reference_torch.numel(**{keyword: expected})
                    )
                    self.assertEqual(actual_result, expected_result)
                    self.assertIs(type(actual_result), type(expected_result))
            self.assertEqual(self.metadata(actual), actual_metadata)
            self.assertEqual(self.metadata(expected), expected_metadata)

    def test_autograd_graph_is_unchanged_like_pytorch_2_13(self):
        actual_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_leaf = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32, requires_grad=True
        )
        actual_tracked = (actual_leaf * 3.0).transpose(0, 0)
        expected_tracked = (expected_leaf * 3.0).transpose(0, 0)

        self.assertEqual(torch.numel(actual_tracked), reference_torch.numel(expected_tracked))
        self.assertEqual(
            torch.numel(input=actual_tracked),
            reference_torch.numel(input=expected_tracked),
        )
        actual_tracked.sum().backward()
        expected_tracked.sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("numel unexpectedly accepted the invalid call")

    def test_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        calls = (
            (lambda: torch.numel(), lambda: reference_torch.numel()),
            (
                lambda: torch.numel(actual, actual),
                lambda: reference_torch.numel(expected, expected),
            ),
            (
                lambda: torch.numel(actual, input=actual),
                lambda: reference_torch.numel(expected, input=expected),
            ),
            (
                lambda: torch.numel(actual, x=actual),
                lambda: reference_torch.numel(expected, x=expected),
            ),
            (
                lambda: torch.numel(foo=actual),
                lambda: reference_torch.numel(foo=expected),
            ),
            (lambda: torch.numel(None), lambda: reference_torch.numel(None)),
            (
                lambda: torch.numel(input=1),
                lambda: reference_torch.numel(input=1),
            ),
            (
                lambda: torch.numel(x=actual, extra=True),
                lambda: reference_torch.numel(x=expected, extra=True),
            ),
            (
                lambda: torch.numel(extra=True, x=actual),
                lambda: reference_torch.numel(extra=True, x=expected),
            ),
            (
                lambda: torch.numel(input=actual, x1=actual),
                lambda: reference_torch.numel(input=expected, x1=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(calls):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def callable_contract(self, module):
        function = module.numel
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
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.numel is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "all_count": module.__all__.count("numel"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["numel"] is function,
        }

    def test_callable_metadata_exports_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def dispatch_contract(self, module):
        marker = object()
        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        for keyword in (None, "input", "x", "a", "x1"):
            value = Override()
            Override.calls.clear()
            result = (
                module.numel(value)
                if keyword is None
                else module.numel(**{keyword: value})
            )
            function, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    keyword,
                    result is marker,
                    function is module.numel,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                )
            )

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        tensor = module.tensor([1.0, 2.0, 3.0], dtype=module.float32)
        mode_observations = []
        for keyword in (None, "input", "x", "a", "x1"):
            mode = Mode()
            with mode:
                result = (
                    module.numel(tensor)
                    if keyword is None
                    else module.numel(**{keyword: tensor})
                )
            function, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    keyword,
                    result is marker,
                    function is module.numel,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                )
            )

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = module.numel(x=tensor)

        class DecliningMode(Mode):
            def __repr__(self):
                return "declining-numel-mode"

        with DecliningMode(NotImplemented):
            decline_error = self.error(lambda: module.numel(tensor))

        return {
            "overrides": override_observations,
            "modes": mode_observations,
            "forwarding_order": order,
            "forwarded": forwarded,
            "decline_error": decline_error,
        }

    def test_torch_function_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_contract(torch),
            self.dispatch_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
