import inspect
import pickle
import re
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ResultTypeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "result_type differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        base = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        return (
            module.tensor(-0.0),
            module.zeros((2, 0, 3)),
            base[1],
            base.transpose(0, 1),
            leaf,
            tracked,
            leaf.grad,
        )

    def tensor_state(self, tensor):
        return {
            "values": tensor.tolist(),
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
        }

    def test_tensor_tensor_forms_and_metadata_preservation_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                actual_state = self.tensor_state(actual)
                expected_state = self.tensor_state(expected)
                calls = (
                    (
                        lambda: torch.result_type(actual, actual),
                        lambda: reference_torch.result_type(expected, expected),
                    ),
                    (
                        lambda: torch.result_type(actual, other=actual),
                        lambda: reference_torch.result_type(
                            expected, other=expected
                        ),
                    ),
                    (
                        lambda: torch.result_type(tensor=actual, other=actual),
                        lambda: reference_torch.result_type(
                            tensor=expected, other=expected
                        ),
                    ),
                    (
                        lambda: torch.result_type(other=actual, tensor=actual),
                        lambda: reference_torch.result_type(
                            other=expected, tensor=expected
                        ),
                    ),
                )
                for actual_call, expected_call in calls:
                    self.assertIs(actual_call(), torch.float32)
                    self.assertIs(expected_call(), reference_torch.float32)
                self.assertEqual(self.tensor_state(actual), actual_state)
                self.assertEqual(self.tensor_state(expected), expected_state)
                self.assertEqual(actual_state, expected_state)

    def test_binding_errors_match_pytorch_2_13_before_dispatch(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.result_type(), lambda: reference_torch.result_type()),
            (
                lambda: torch.result_type(actual),
                lambda: reference_torch.result_type(expected),
            ),
            (
                lambda: torch.result_type(actual, actual, actual),
                lambda: reference_torch.result_type(
                    expected, expected, expected
                ),
            ),
            (
                lambda: torch.result_type([], actual),
                lambda: reference_torch.result_type([], expected),
            ),
            (
                lambda: torch.result_type(actual, []),
                lambda: reference_torch.result_type(expected, []),
            ),
            (
                lambda: torch.result_type(1, []),
                lambda: reference_torch.result_type(1, []),
            ),
            (
                lambda: torch.result_type([], 1),
                lambda: reference_torch.result_type([], 1),
            ),
            (
                lambda: torch.result_type(tensor=1, other=actual),
                lambda: reference_torch.result_type(tensor=1, other=expected),
            ),
            (
                lambda: torch.result_type(actual, tensor=actual),
                lambda: reference_torch.result_type(expected, tensor=expected),
            ),
            (
                lambda: torch.result_type(foo=actual, bar=actual),
                lambda: reference_torch.result_type(
                    foo=expected, bar=expected
                ),
            ),
            (
                lambda: torch.result_type(actual, actual, extra=True),
                lambda: reference_torch.result_type(
                    expected, expected, extra=True
                ),
            ),
            (
                lambda: torch.result_type(1, actual, extra=True),
                lambda: reference_torch.result_type(1, expected, extra=True),
            ),
            (
                lambda: torch.result_type(torch.float32, actual),
                lambda: reference_torch.result_type(
                    reference_torch.float32, expected
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def mode_observation(self, module, form):
        left = module.tensor([1.0])
        right = module.tensor([2.0])
        marker = object()

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = Mode()
        with mode:
            if form == "positional":
                result = module.result_type(left, right)
            elif form == "keywords":
                result = module.result_type(tensor=left, other=right)
            else:
                result = module.result_type(1, right)
        function, dispatch_types, args, kwargs = mode.calls[0]
        return (
            result is marker,
            function is module.result_type,
            dispatch_types,
            len(args),
            None if kwargs is None else tuple(kwargs),
        )

    def override_observation(self, module, second):
        tensor = module.tensor([1.0])
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        result = (
            module.result_type(tensor, value)
            if second
            else module.result_type(value, tensor)
        )
        function, dispatch_types, args, kwargs = Override.calls[0]
        return (
            result is marker,
            function is module.result_type,
            tuple(item.__name__ for item in dispatch_types),
            args[1] is value if second else args[0] is value,
            kwargs is None,
        )

    def test_modes_and_operand_overrides_match_pytorch_2_13(self):
        for form in ("positional", "keywords", "scalar"):
            with self.subTest(dispatch="mode", form=form):
                self.assertEqual(
                    self.mode_observation(torch, form),
                    self.mode_observation(reference_torch, form),
                )
        for second in (False, True):
            with self.subTest(dispatch="operand", second=second):
                self.assertEqual(
                    self.override_observation(torch, second),
                    self.override_observation(reference_torch, second),
                )

    def callable_contract(self, module):
        function = module.result_type
        owner = function.__reduce__()[1][0]
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
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "self": function.__self__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_suffix": owner.__module__.split(".")[-1],
            "owner_identity": owner.result_type is function,
            "signature_error": signature_error,
            "pickle": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol))
                is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "export_count": module.__all__.count("result_type"),
        }

    def test_callable_metadata_exports_and_pickling_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
