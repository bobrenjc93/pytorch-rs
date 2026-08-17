import inspect
import pickle
import re
import types
import unittest
import warnings

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

    def test_hash_aware_string_subclass_keywords_match_pytorch_2_13(self):
        class AlwaysEqualKeyword(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                return True

        class RaisingKeyword(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                raise RuntimeError("later keyword equality should not run")

        class MismatchedHashKeyword(str):
            def __hash__(self):
                return 1

        actual_type2 = AlwaysEqualKeyword("type2")
        expected_type2 = AlwaysEqualKeyword("type2")
        self.assertIs(
            torch.promote_types(
                torch.float32, **{actual_type2: torch.float32}
            ),
            torch.float32,
        )
        self.assertIs(
            reference_torch.promote_types(
                reference_torch.float32,
                **{expected_type2: reference_torch.float32},
            ),
            reference_torch.float32,
        )

        self.assert_error_matches(
            lambda: torch.promote_types(
                torch.float32,
                torch.float32,
                **{AlwaysEqualKeyword("unexpected"): torch.float32},
            ),
            lambda: reference_torch.promote_types(
                reference_torch.float32,
                reference_torch.float32,
                **{AlwaysEqualKeyword("unexpected"): reference_torch.float32},
            ),
        )
        self.assert_error_matches(
            lambda: torch.promote_types(
                type1=torch.float32,
                type2=torch.float32,
                **{AlwaysEqualKeyword("unexpected"): torch.float32},
            ),
            lambda: reference_torch.promote_types(
                type1=reference_torch.float32,
                type2=reference_torch.float32,
                **{AlwaysEqualKeyword("unexpected"): reference_torch.float32},
            ),
        )
        self.assert_error_matches(
            lambda: torch.promote_types(
                torch.float32,
                **{
                    AlwaysEqualKeyword("type2"): torch.float32,
                    RaisingKeyword("type1"): torch.float32,
                },
            ),
            lambda: reference_torch.promote_types(
                reference_torch.float32,
                **{
                    AlwaysEqualKeyword("type2"): reference_torch.float32,
                    RaisingKeyword("type1"): reference_torch.float32,
                },
            ),
        )
        self.assert_error_matches(
            lambda: torch.promote_types(
                torch.float32,
                **{
                    MismatchedHashKeyword("type2"): torch.float32,
                    AlwaysEqualKeyword("type2"): torch.float32,
                },
            ),
            lambda: reference_torch.promote_types(
                reference_torch.float32,
                **{
                    MismatchedHashKeyword("type2"): reference_torch.float32,
                    AlwaysEqualKeyword("type2"): reference_torch.float32,
                },
            ),
        )

        def mode_observation(module):
            key = AlwaysEqualKeyword("type2")
            marker = object()

            class Mode(module.overrides.TorchFunctionMode):
                def __init__(self):
                    self.calls = []

                def __torch_function__(self, func, types, args=(), kwargs=None):
                    self.calls.append((func, types, args, kwargs))
                    return marker

            mode = Mode()
            with mode:
                result = module.promote_types(
                    module.float32, **{key: module.float32}
                )
            function, dispatch_types, args, kwargs = mode.calls[0]
            received_key = next(iter(kwargs))
            return (
                result is marker,
                function is module.promote_types,
                dispatch_types,
                args == (module.float32,),
                received_key is key,
                kwargs[received_key] is module.float32,
            )

        self.assertEqual(
            mode_observation(torch),
            mode_observation(reference_torch),
        )

    def test_operand_validation_precedes_later_keyword_lookup_in_pytorch_2_13(
        self,
    ):
        def observation(module, positional):
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
                        return module.float32

                    Operand.__torch_function__ = override
                    return super().__eq__(other)

            operand = Operand()
            key = MutatingKeyword("type2")
            try:
                if positional:
                    result = module.promote_types(
                        operand, **{key: module.float32}
                    )
                else:
                    result = module.promote_types(
                        type1=operand, **{key: module.float32}
                    )
            except Exception as error:
                outcome = (type(error).__name__, str(error))
            else:
                outcome = ("result", result is module.float32)
            return outcome, tuple(events)

        for form, positional in (("positional", True), ("keyword", False)):
            with self.subTest(form=form):
                actual = observation(torch, positional)
                expected = observation(reference_torch, positional)
                self.assertEqual(actual, expected)
                self.assertEqual(actual[1], ())

    def test_dtype_validation_one_shot_probe_matches_pytorch_2_13(self):
        def observation(module):
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
                            return module.float32

                        return handler
                    return object.__getattribute__(self, name)

            operand = StatefulOperand()
            try:
                result = module.promote_types(operand, module.float32)
            except Exception as error:
                outcome = (type(error).__name__, str(error))
            else:
                outcome = ("result", result is module.float32)
            return outcome, tuple(events), operand.lookups

        self.assertEqual(observation(torch), observation(reference_torch))

    def operand_override_observation(self, module):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        forms = (
            lambda value: module.promote_types(value, module.float32),
            lambda value: module.promote_types(module.float32, value),
            lambda value: module.promote_types(
                type1=value, type2=module.float32
            ),
            lambda value: module.promote_types(
                type1=module.float32, type2=value
            ),
            lambda value: module.promote_types(module.float32, type2=value),
            lambda value: module.promote_types(
                type2=value, type1=module.float32
            ),
        )
        calls = []
        for call in forms:
            value = Override()
            Override.calls = []
            result = call(value)
            function, dispatch_types, args, kwargs = Override.calls[0]
            calls.append(
                (
                    result is marker,
                    len(Override.calls),
                    function is module.promote_types,
                    tuple(dispatch_type.__name__ for dispatch_type in dispatch_types),
                    tuple(
                        "override" if argument is value else "dtype"
                        for argument in args
                    ),
                    None
                    if kwargs is None
                    else tuple(
                        (
                            key,
                            "override" if argument is value else "dtype",
                        )
                        for key, argument in kwargs.items()
                    ),
                )
            )

        events = []

        class Left:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("left", types))
                return NotImplemented

        class Right:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("right", types))
                return marker

        distinct_result = module.promote_types(Left(), Right())
        distinct = (
            distinct_result is marker,
            tuple(
                (
                    label,
                    tuple(dispatch_type.__name__ for dispatch_type in dispatch_types),
                )
                for label, dispatch_types in events
            ),
        )

        events.clear()

        class Same:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(tuple(
                    dispatch_type.__name__ for dispatch_type in types
                ))
                return marker

        same_result = module.promote_types(Same(), Same())
        same = (same_result is marker, tuple(events))

        events.clear()

        class Base:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("base", types))
                return marker

        class Derived(Base):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("derived", types))
                return marker

        subclass_result = module.promote_types(Base(), Derived())
        subclass = (
            subclass_result is marker,
            tuple(
                (
                    label,
                    tuple(dispatch_type.__name__ for dispatch_type in dispatch_types),
                )
                for label, dispatch_types in events
            ),
        )

        events.clear()

        class FallingOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("override", types, kwargs))
                return marker

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(("mode", types, kwargs))
                return NotImplemented

        with DecliningMode():
            fallthrough_result = module.promote_types(
                FallingOverride(), module.float32
            )
        fallthrough = (
            fallthrough_result is marker,
            tuple(
                (
                    label,
                    tuple(dispatch_type.__name__ for dispatch_type in dispatch_types),
                    kwargs,
                )
                for label, dispatch_types, kwargs in events
            ),
        )

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        try:
            module.promote_types(DecliningOverride(), module.float32)
        except Exception as error:
            decline = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            decline = None

        mode = DecliningMode()
        try:
            with mode:
                module.promote_types(DecliningOverride(), module.float32)
        except Exception as error:
            mode_decline = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            mode_decline = None

        return {
            "calls": tuple(calls),
            "distinct": distinct,
            "same": same,
            "subclass": subclass,
            "fallthrough": fallthrough,
            "decline": decline,
            "mode_decline": mode_decline,
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_operand_torch_function_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.operand_override_observation(torch),
            self.operand_override_observation(reference_torch),
        )

    def test_class_valued_override_ordering_matches_pytorch_2_13(self):
        def observation(module):
            events = []
            base_marker = object()
            derived_marker = object()

            class Base:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append(
                        ("base", tuple(dispatch_type.__name__ for dispatch_type in types))
                    )
                    return base_marker

            class Derived(Base):
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append(
                        (
                            "derived",
                            tuple(dispatch_type.__name__ for dispatch_type in types),
                        )
                    )
                    return derived_marker

            cases = (
                ("class pair", (Base, Derived), "base"),
                ("instance then class", (Base(), Derived), "base"),
                ("class then instance", (Base, Derived()), "derived"),
            )
            ordering = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for label, operands, expected_result in cases:
                    events.clear()
                    result = module.promote_types(*operands)
                    ordering.append(
                        (
                            label,
                            result is base_marker,
                            result is derived_marker,
                            expected_result,
                            tuple(events),
                        )
                    )

            events.clear()

            class Repeated:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append(
                        tuple(dispatch_type.__name__ for dispatch_type in types)
                    )
                    return NotImplemented

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    module.promote_types(Repeated, Repeated)
                except Exception as error:
                    repeated_error = (
                        type(error).__name__,
                        str(error).count("tensor subclass"),
                    )
                else:
                    repeated_error = None

            return {
                "ordering": tuple(ordering),
                "repeated_calls": tuple(events),
                "repeated_error": repeated_error,
            }

        self.assertEqual(observation(torch), observation(reference_torch))

    def handler_arity_observation(self, module):
        marker = object()

        class StrictMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=()):
                return marker

        def observe_mode(call):
            try:
                with StrictMode():
                    result = call()
            except Exception as error:
                return (type(error).__name__, str(error))
            return ("result", result is marker)

        mode = (
            observe_mode(
                lambda: module.promote_types(module.float32, module.float32)
            ),
            observe_mode(
                lambda: module.promote_types(
                    module.float32, module.float32, **{}
                )
            ),
            observe_mode(
                lambda: module.promote_types(
                    type1=module.float32, type2=module.float32
                )
            ),
        )

        class StrictOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=()):
                return marker

        value = StrictOverride()

        def observe_override(call):
            try:
                result = call()
            except Exception as error:
                return (type(error).__name__, str(error))
            return ("result", result is marker)

        override = (
            observe_override(
                lambda: module.promote_types(value, module.float32)
            ),
            observe_override(
                lambda: module.promote_types(value, module.float32, **{})
            ),
            observe_override(
                lambda: module.promote_types(
                    type1=value, type2=module.float32
                )
            ),
        )
        return {"mode": mode, "override": override}

    def test_handler_arity_matches_absent_and_empty_kwargs_in_pytorch_2_13(self):
        self.assertEqual(
            self.handler_arity_observation(torch),
            self.handler_arity_observation(reference_torch),
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

        class ValidationOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise AssertionError("invalid calls must not dispatch")

        actual_override = ValidationOverride()
        expected_override = ValidationOverride()

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
                lambda: torch.promote_types(actual_override, 1),
                lambda: reference_torch.promote_types(expected_override, 1),
            ),
            (
                lambda: torch.promote_types(1, actual_override),
                lambda: reference_torch.promote_types(1, expected_override),
            ),
            (
                lambda: torch.promote_types(actual_override),
                lambda: reference_torch.promote_types(expected_override),
            ),
            (
                lambda: torch.promote_types(type2=actual_override),
                lambda: reference_torch.promote_types(type2=expected_override),
            ),
            (
                lambda: torch.promote_types(
                    actual_override, actual, extra=True
                ),
                lambda: reference_torch.promote_types(
                    expected_override, expected, extra=True
                ),
            ),
            (
                lambda: torch.promote_types(
                    actual_override, type1=actual, type2=actual
                ),
                lambda: reference_torch.promote_types(
                    expected_override, type1=expected, type2=expected
                ),
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
