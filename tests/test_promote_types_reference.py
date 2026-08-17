import inspect
import pickle
import re
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class PromoteTypesReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "promote_types differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_float32_aliases_and_all_canonical_calls_match_pytorch_2_13(self):
        for actual_type1, expected_type1 in (
            (torch.float32, reference_torch.float32),
            (torch.float, reference_torch.float),
        ):
            for actual_type2, expected_type2 in (
                (torch.float32, reference_torch.float32),
                (torch.float, reference_torch.float),
            ):
                calls = (
                    (
                        lambda: torch.promote_types(actual_type1, actual_type2),
                        lambda: reference_torch.promote_types(
                            expected_type1, expected_type2
                        ),
                    ),
                    (
                        lambda: torch.promote_types(
                            actual_type1, type2=actual_type2
                        ),
                        lambda: reference_torch.promote_types(
                            expected_type1, type2=expected_type2
                        ),
                    ),
                    (
                        lambda: torch.promote_types(
                            type1=actual_type1, type2=actual_type2
                        ),
                        lambda: reference_torch.promote_types(
                            type1=expected_type1, type2=expected_type2
                        ),
                    ),
                    (
                        lambda: torch.promote_types(
                            type2=actual_type2, type1=actual_type1
                        ),
                        lambda: reference_torch.promote_types(
                            type2=expected_type2, type1=expected_type1
                        ),
                    ),
                )
                for actual_call, expected_call in calls:
                    self.assertIs(actual_call(), torch.float32)
                    self.assertIs(expected_call(), reference_torch.float32)

    def mode_observation(self, module):
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        forms = (
            lambda: module.promote_types(module.float32, module.float),
            lambda: module.promote_types(module.float, type2=module.float32),
            lambda: module.promote_types(
                type1=module.float32, type2=module.float
            ),
            lambda: module.promote_types(
                type2=module.float, type1=module.float32
            ),
        )
        observations = []
        for call in forms:
            mode = RecordingMode()
            with mode:
                result = call()
            function, dispatch_types, args, kwargs = mode.calls[0]
            observations.append(
                (
                    result is marker,
                    len(mode.calls),
                    function is module.promote_types,
                    type(function).__name__,
                    function.__qualname__,
                    dispatch_types,
                    len(args),
                    tuple(value is module.float32 for value in args),
                    None if kwargs is None else tuple(kwargs),
                    None
                    if kwargs is None
                    else tuple(value is module.float32 for value in kwargs.values()),
                )
            )

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(
                    (
                        self.label,
                        func is module.promote_types,
                        types,
                        args,
                        None if kwargs is None else tuple(kwargs),
                    )
                )
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = module.promote_types(
                    type2=module.float, type1=module.float32
                )

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        try:
            with DecliningMode():
                module.promote_types(module.float32, module.float32)
        except Exception as error:
            decline = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            decline = None

        class ClassMethodMode(module.overrides.TorchFunctionMode):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return module.float32

        try:
            with ClassMethodMode():
                module.promote_types(module.float32, module.float32)
        except Exception as error:
            classmethod_error = (type(error).__name__, str(error))
        else:
            classmethod_error = None

        return {
            "observations": observations,
            "forwarding_order": order,
            "forwarded_is_singleton": forwarded is module.float32,
            "decline": decline,
            "classmethod_error": classmethod_error,
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_observation(torch),
            self.mode_observation(reference_torch),
        )

    def test_validation_order_and_errors_match_pytorch_2_13(self):
        actual = torch.float32
        expected = reference_torch.float32

        class FalseEqualityKeyword(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                return False

        class RaisingEqualityKeyword(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                raise RuntimeError("keyword equality failed")

        class MismatchedHashKeyword(str):
            def __hash__(self):
                return 1

        cases = (
            (
                lambda: torch.promote_types(),
                lambda: reference_torch.promote_types(),
            ),
            (
                lambda: torch.promote_types(actual),
                lambda: reference_torch.promote_types(expected),
            ),
            (
                lambda: torch.promote_types(1),
                lambda: reference_torch.promote_types(1),
            ),
            (
                lambda: torch.promote_types(type2=actual),
                lambda: reference_torch.promote_types(type2=expected),
            ),
            (
                lambda: torch.promote_types(actual, actual, actual),
                lambda: reference_torch.promote_types(
                    expected, expected, expected
                ),
            ),
            (
                lambda: torch.promote_types(1, actual),
                lambda: reference_torch.promote_types(1, expected),
            ),
            (
                lambda: torch.promote_types(actual, None),
                lambda: reference_torch.promote_types(expected, None),
            ),
            (
                lambda: torch.promote_types(type1=1, type2=actual),
                lambda: reference_torch.promote_types(type1=1, type2=expected),
            ),
            (
                lambda: torch.promote_types(type1=actual, type2=True),
                lambda: reference_torch.promote_types(type1=expected, type2=True),
            ),
            (
                lambda: torch.promote_types(
                    actual, type1=actual, type2=actual
                ),
                lambda: reference_torch.promote_types(
                    expected, type1=expected, type2=expected
                ),
            ),
            (
                lambda: torch.promote_types(1, type1=actual, type2=actual),
                lambda: reference_torch.promote_types(
                    1, type1=expected, type2=expected
                ),
            ),
            (
                lambda: torch.promote_types(actual, actual, extra=True),
                lambda: reference_torch.promote_types(
                    expected, expected, extra=True
                ),
            ),
            (
                lambda: torch.promote_types(
                    actual, actual, **{"extra\x00suffix": True}
                ),
                lambda: reference_torch.promote_types(
                    expected, expected, **{"extra\x00suffix": True}
                ),
            ),
            (
                lambda: torch.promote_types(actual, 1, extra=True),
                lambda: reference_torch.promote_types(expected, 1, extra=True),
            ),
            (
                lambda: torch.promote_types(type1=actual, extra=True),
                lambda: reference_torch.promote_types(
                    type1=expected, extra=True
                ),
            ),
            (
                lambda: torch.promote_types(dtype1=actual, dtype2=actual),
                lambda: reference_torch.promote_types(
                    dtype1=expected, dtype2=expected
                ),
            ),
            (
                lambda: torch.promote_types(
                    actual, extra=True, type1=actual, type2=actual
                ),
                lambda: reference_torch.promote_types(
                    expected, extra=True, type1=expected, type2=expected
                ),
            ),
            (
                lambda: torch.promote_types(
                    **{
                        FalseEqualityKeyword("type1"): actual,
                        "type2": actual,
                    }
                ),
                lambda: reference_torch.promote_types(
                    **{
                        FalseEqualityKeyword("type1"): expected,
                        "type2": expected,
                    }
                ),
            ),
            (
                lambda: torch.promote_types(
                    actual, actual, **{FalseEqualityKeyword("type1"): actual}
                ),
                lambda: reference_torch.promote_types(
                    expected,
                    expected,
                    **{FalseEqualityKeyword("type1"): expected},
                ),
            ),
            (
                lambda: torch.promote_types(
                    **{
                        RaisingEqualityKeyword("type1"): actual,
                        "type2": actual,
                    }
                ),
                lambda: reference_torch.promote_types(
                    **{
                        RaisingEqualityKeyword("type1"): expected,
                        "type2": expected,
                    }
                ),
            ),
            (
                lambda: torch.promote_types(
                    actual,
                    actual,
                    **{RaisingEqualityKeyword("extra"): actual},
                ),
                lambda: reference_torch.promote_types(
                    expected,
                    expected,
                    **{RaisingEqualityKeyword("extra"): expected},
                ),
            ),
            (
                lambda: torch.promote_types(
                    1, actual, **{RaisingEqualityKeyword("extra"): actual}
                ),
                lambda: reference_torch.promote_types(
                    1,
                    expected,
                    **{RaisingEqualityKeyword("extra"): expected},
                ),
            ),
            (
                lambda: torch.promote_types(
                    **{
                        MismatchedHashKeyword("type1"): actual,
                        "type2": actual,
                    }
                ),
                lambda: reference_torch.promote_types(
                    **{
                        MismatchedHashKeyword("type1"): expected,
                        "type2": expected,
                    }
                ),
            ),
            (
                lambda: torch.promote_types(
                    actual,
                    actual,
                    **{MismatchedHashKeyword("type1"): actual},
                ),
                lambda: reference_torch.promote_types(
                    expected,
                    expected,
                    **{MismatchedHashKeyword("type1"): expected},
                ),
            ),
        )
        for actual_call, expected_call in cases:
            self.assert_error_matches(actual_call, expected_call)

        for actual_call, expected_call in (
            (
                lambda: torch.promote_types(1, actual),
                lambda: reference_torch.promote_types(1, expected),
            ),
            (
                lambda: torch.promote_types(actual),
                lambda: reference_torch.promote_types(expected),
            ),
            (
                lambda: torch.promote_types(actual, actual, extra=True),
                lambda: reference_torch.promote_types(
                    expected, expected, extra=True
                ),
            ),
        ):
            actual_calls = []
            expected_calls = []

            class ActualMode(torch.overrides.TorchFunctionMode):
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    actual_calls.append((func, types, args, kwargs))
                    return actual

            class ExpectedMode(reference_torch.overrides.TorchFunctionMode):
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    expected_calls.append((func, types, args, kwargs))
                    return expected

            with ActualMode():
                with self.assertRaises(Exception):
                    actual_call()
            with ExpectedMode():
                with self.assertRaises(Exception):
                    expected_call()
            self.assertEqual(actual_calls, [])
            self.assertEqual(expected_calls, [])

    def metadata_observation(self, module):
        function = module.promote_types
        reducer, (owner, name) = function.__reduce__()
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (type(error).__name__, str(error).split(" for ", 1)[0])
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
            "reducer_is_getattr": reducer is getattr,
            "reduce_name": name,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_identity": owner is module._C._VariableFunctionsClass,
            "owner_function_identity": owner.promote_types is function,
            "self_is_none": function.__self__ is None,
            "all_count": module.__all__.count("promote_types"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_exports_and_pickling_match_pytorch_2_13(self):
        self.assertEqual(
            self.metadata_observation(torch),
            self.metadata_observation(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
