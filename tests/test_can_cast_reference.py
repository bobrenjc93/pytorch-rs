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
class CanCastReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "can_cast differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_float32_aliases_and_all_canonical_calls_match_pytorch_2_13(self):
        for actual_from, expected_from in (
            (torch.float32, reference_torch.float32),
            (torch.float, reference_torch.float),
        ):
            for actual_to, expected_to in (
                (torch.float32, reference_torch.float32),
                (torch.float, reference_torch.float),
            ):
                calls = (
                    (
                        lambda: torch.can_cast(actual_from, actual_to),
                        lambda: reference_torch.can_cast(
                            expected_from, expected_to
                        ),
                    ),
                    (
                        lambda: torch.can_cast(actual_from, to=actual_to),
                        lambda: reference_torch.can_cast(
                            expected_from, to=expected_to
                        ),
                    ),
                    (
                        lambda: torch.can_cast(
                            from_=actual_from, to=actual_to
                        ),
                        lambda: reference_torch.can_cast(
                            from_=expected_from, to=expected_to
                        ),
                    ),
                    (
                        lambda: torch.can_cast(
                            to=actual_to, from_=actual_from
                        ),
                        lambda: reference_torch.can_cast(
                            to=expected_to, from_=expected_from
                        ),
                    ),
                )
                for actual_call, expected_call in calls:
                    actual = actual_call()
                    expected = expected_call()
                    self.assertIs(type(actual), type(expected))
                    self.assertIs(actual, expected)

    def test_validation_order_matches_pytorch_2_13(self):
        def keyword_order_observation(module, positional):
            events = []

            class Operand:
                pass

            class MutatingKeyword(str):
                __hash__ = str.__hash__

                def __eq__(self, other):
                    events.append("keyword equality")

                    @classmethod
                    def override(cls, func, types, args=(), kwargs=None):
                        events.append("override dispatch")
                        return True

                    Operand.__torch_function__ = override
                    return super().__eq__(other)

            operand = Operand()
            key = MutatingKeyword("to")
            try:
                if positional:
                    result = module.can_cast(
                        operand, **{key: module.float32}
                    )
                else:
                    result = module.can_cast(
                        from_=operand, **{key: module.float32}
                    )
            except Exception as error:
                outcome = (type(error).__name__, str(error))
            else:
                outcome = ("result", result)
            return outcome, tuple(events)

        for form, positional in (("positional", True), ("keyword", False)):
            with self.subTest(form=form):
                actual = keyword_order_observation(torch, positional)
                expected = keyword_order_observation(
                    reference_torch, positional
                )
                self.assertEqual(actual, expected)
                self.assertEqual(actual[1], ())

        def probe_observation(module):
            events = []

            class StatefulOperand:
                def __init__(self):
                    self.lookups = 0

                def __getattribute__(self, name):
                    if name == "__torch_function__":
                        lookups = object.__getattribute__(self, "lookups") + 1
                        object.__setattr__(self, "lookups", lookups)
                        events.append(("lookup", lookups))
                        if lookups == 1:
                            raise AttributeError("transient probe failure")

                        def handler(func, types, args=(), kwargs=None):
                            events.append(("dispatch",))
                            return True

                        return handler
                    return object.__getattribute__(self, name)

            operand = StatefulOperand()
            try:
                result = module.can_cast(operand, module.float32)
            except Exception as error:
                outcome = (type(error).__name__, str(error))
            else:
                outcome = ("result", result)
            return outcome, tuple(events), operand.lookups

        self.assertEqual(
            probe_observation(torch), probe_observation(reference_torch)
        )

    def mode_observation(self, module):
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        forms = (
            lambda: module.can_cast(module.float32, module.float),
            lambda: module.can_cast(module.float, to=module.float32),
            lambda: module.can_cast(
                from_=module.float32, to=module.float
            ),
            lambda: module.can_cast(
                to=module.float, from_=module.float32
            ),
        )
        calls = []
        for call in forms:
            mode = RecordingMode()
            with mode:
                result = call()
            function, dispatch_types, args, kwargs = mode.calls[0]
            calls.append(
                (
                    result is marker,
                    len(mode.calls),
                    function is module.can_cast,
                    dispatch_types,
                    tuple(argument is module.float32 for argument in args),
                    None
                    if kwargs is None
                    else tuple(
                        (key, value is module.float32)
                        for key, value in kwargs.items()
                    ),
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
                        func is module.can_cast,
                        types,
                        args,
                        None if kwargs is None else tuple(kwargs),
                    )
                )
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = module.can_cast(
                    from_=module.float32, to=module.float
                )

        invalid_calls = []

        class InvalidRecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return True

        for call in (
            lambda: module.can_cast(1, module.float32),
            lambda: module.can_cast(module.float32),
            lambda: module.can_cast(
                module.float32, module.float32, extra=True
            ),
        ):
            mode = InvalidRecordingMode()
            try:
                with mode:
                    call()
            except Exception as error:
                outcome = (type(error).__name__, str(error))
            else:
                outcome = ("result",)
            invalid_calls.append((outcome, len(mode.calls)))

        return (
            tuple(calls),
            forwarded,
            tuple(order),
            tuple(invalid_calls),
        )

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_observation(torch),
            self.mode_observation(reference_torch),
        )

    def test_invalid_argument_errors_match_pytorch_2_13(self):
        actual = torch.float32
        expected = reference_torch.float32

        class AlwaysEqualKeyword(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                return True

        cases = (
            (
                lambda: torch.can_cast(),
                lambda: reference_torch.can_cast(),
            ),
            (
                lambda: torch.can_cast(actual),
                lambda: reference_torch.can_cast(expected),
            ),
            (
                lambda: torch.can_cast(1),
                lambda: reference_torch.can_cast(1),
            ),
            (
                lambda: torch.can_cast(to=actual),
                lambda: reference_torch.can_cast(to=expected),
            ),
            (
                lambda: torch.can_cast(actual, actual, actual),
                lambda: reference_torch.can_cast(
                    expected, expected, expected
                ),
            ),
            (
                lambda: torch.can_cast(1, actual),
                lambda: reference_torch.can_cast(1, expected),
            ),
            (
                lambda: torch.can_cast(actual, None),
                lambda: reference_torch.can_cast(expected, None),
            ),
            (
                lambda: torch.can_cast(from_=1, to=actual),
                lambda: reference_torch.can_cast(from_=1, to=expected),
            ),
            (
                lambda: torch.can_cast(from_=actual, to=True),
                lambda: reference_torch.can_cast(from_=expected, to=True),
            ),
            (
                lambda: torch.can_cast(
                    actual, from_=actual, to=actual
                ),
                lambda: reference_torch.can_cast(
                    expected, from_=expected, to=expected
                ),
            ),
            (
                lambda: torch.can_cast(actual, actual, extra=True),
                lambda: reference_torch.can_cast(
                    expected, expected, extra=True
                ),
            ),
            (
                lambda: torch.can_cast(
                    actual, actual, **{"extra\x00suffix": True}
                ),
                lambda: reference_torch.can_cast(
                    expected, expected, **{"extra\x00suffix": True}
                ),
            ),
            (
                lambda: torch.can_cast(actual, 1, extra=True),
                lambda: reference_torch.can_cast(expected, 1, extra=True),
            ),
            (
                lambda: torch.can_cast(from_=actual, extra=True),
                lambda: reference_torch.can_cast(
                    from_=expected, extra=True
                ),
            ),
            (
                lambda: torch.can_cast(dtype1=actual, dtype2=actual),
                lambda: reference_torch.can_cast(
                    dtype1=expected, dtype2=expected
                ),
            ),
            (
                lambda: torch.can_cast(
                    actual,
                    actual,
                    **{AlwaysEqualKeyword("unexpected"): actual},
                ),
                lambda: reference_torch.can_cast(
                    expected,
                    expected,
                    **{AlwaysEqualKeyword("unexpected"): expected},
                ),
            ),
        )
        for actual_call, expected_call in cases:
            self.assert_error_matches(actual_call, expected_call)

    def metadata_observation(self, module):
        function = module.can_cast
        reducer, (owner, name) = function.__reduce__()
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                str(error).split(" for ", 1)[0],
            )
        else:
            signature_error = None
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
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
            "owner_function_identity": owner.can_cast is function,
            "self_is_none": function.__self__ is None,
            "all_count": module.__all__.count("can_cast"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(
                module, "_VariableFunctionsClass"
            ),
            "wildcard_identity": wildcard_namespace["can_cast"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol))
                is function
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
