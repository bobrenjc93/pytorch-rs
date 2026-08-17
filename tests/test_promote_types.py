import copy
import inspect
import pickle
import re
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = """
promote_types(type1, type2) -> dtype

Returns the :class:`torch.dtype` with the smallest size and scalar kind that is
not smaller nor of lower kind than either `type1` or `type2`. See type promotion
:ref:`documentation <type-promotion-doc>` for more information on the type
promotion logic.

Args:
    type1 (:class:`torch.dtype`)
    type2 (:class:`torch.dtype`)

Example::

    >>> torch.promote_types(torch.int32, torch.float32)
    torch.float32
    >>> torch.promote_types(torch.uint8, torch.long)
    torch.long
"""


class PromoteTypesTests(unittest.TestCase):
    def assert_error(self, exception_type, message, call):
        with self.assertRaisesRegex(exception_type, f"^{re.escape(message)}$"):
            call()

    def test_float32_aliases_and_canonical_call_forms_return_the_singleton(self):
        self.assertIs(torch.float, torch.float32)
        for type1 in (torch.float32, torch.float):
            for type2 in (torch.float32, torch.float):
                calls = (
                    ("positional", lambda: torch.promote_types(type1, type2)),
                    ("mixed", lambda: torch.promote_types(type1, type2=type2)),
                    (
                        "keywords",
                        lambda: torch.promote_types(type1=type1, type2=type2),
                    ),
                    (
                        "reversed keywords",
                        lambda: torch.promote_types(type2=type2, type1=type1),
                    ),
                )
                for form, call in calls:
                    with self.subTest(type1=type1, type2=type2, form=form):
                        self.assertIs(call(), torch.float32)

        class AlwaysEqualKeyword(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                return True

        type2 = AlwaysEqualKeyword("type2")
        self.assertIs(
            torch.promote_types(
                torch.float32, **{type2: torch.float32}
            ),
            torch.float32,
        )
        self.assert_error(
            TypeError,
            "promote_types() got multiple values for argument 'unexpected'",
            lambda: torch.promote_types(
                torch.float32,
                torch.float32,
                **{AlwaysEqualKeyword("unexpected"): torch.float32},
            ),
        )
        self.assert_error(
            TypeError,
            "invalid keyword arguments",
            lambda: torch.promote_types(
                type1=torch.float32,
                type2=torch.float32,
                **{AlwaysEqualKeyword("unexpected"): torch.float32},
            ),
        )

    def test_keyword_validation_preserves_original_key_order(self):
        class AlwaysEqualKeyword(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                return True

        class RaisingKeyword(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                raise RuntimeError("later keyword equality should not run")

        self.assert_error(
            TypeError,
            "promote_types() got multiple values for argument 'type2'",
            lambda: torch.promote_types(
                torch.float32,
                **{
                    AlwaysEqualKeyword("type2"): torch.float32,
                    RaisingKeyword("type1"): torch.float32,
                },
            ),
        )

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        cases = (
            (
                "positional",
                lambda: torch.promote_types(torch.float32, torch.float),
                (torch.float32, torch.float),
                None,
            ),
            (
                "mixed",
                lambda: torch.promote_types(torch.float, type2=torch.float32),
                (torch.float,),
                {"type2": torch.float32},
            ),
            (
                "keywords",
                lambda: torch.promote_types(
                    type1=torch.float32, type2=torch.float
                ),
                (),
                {"type1": torch.float32, "type2": torch.float},
            ),
            (
                "reversed keywords",
                lambda: torch.promote_types(
                    type2=torch.float, type1=torch.float32
                ),
                (),
                {"type2": torch.float, "type1": torch.float32},
            ),
        )
        for case, call, expected_args, expected_kwargs in cases:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            with self.subTest(case=case):
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs = mode.calls[0]
                self.assertIs(function, torch.promote_types)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                result = torch.promote_types(
                    type2=torch.float, type1=torch.float32
                )

        self.assertIs(result, torch.float32)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, torch.promote_types)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, ())
            self.assertEqual(tuple(kwargs), ("type2", "type1"))
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_operand_torch_function_overrides_receive_all_call_forms(self):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        forms = (
            (
                "first positional",
                lambda value: torch.promote_types(value, torch.float32),
                ("override", "dtype"),
                None,
            ),
            (
                "second positional",
                lambda value: torch.promote_types(torch.float32, value),
                ("dtype", "override"),
                None,
            ),
            (
                "first keyword",
                lambda value: torch.promote_types(
                    type1=value, type2=torch.float32
                ),
                (),
                (("type1", "override"), ("type2", "dtype")),
            ),
            (
                "second keyword",
                lambda value: torch.promote_types(
                    type1=torch.float32, type2=value
                ),
                (),
                (("type1", "dtype"), ("type2", "override")),
            ),
            (
                "mixed",
                lambda value: torch.promote_types(torch.float32, type2=value),
                ("dtype",),
                (("type2", "override"),),
            ),
            (
                "reversed keywords",
                lambda value: torch.promote_types(
                    type2=value, type1=torch.float32
                ),
                (),
                (("type2", "override"), ("type1", "dtype")),
            ),
        )
        for case, call, expected_args, expected_kwargs in forms:
            value = Override()
            Override.calls = []
            self.assertIs(call(value), marker)
            self.assertEqual(len(Override.calls), 1)
            function, dispatch_types, args, kwargs = Override.calls[0]
            observed_args = tuple(
                "override" if argument is value else "dtype" for argument in args
            )
            observed_kwargs = (
                None
                if kwargs is None
                else tuple(
                    (
                        key,
                        "override" if argument is value else "dtype",
                    )
                    for key, argument in kwargs.items()
                )
            )
            with self.subTest(case=case):
                self.assertIs(function, torch.promote_types)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(observed_args, expected_args)
                self.assertEqual(observed_kwargs, expected_kwargs)

    def test_override_order_mode_fallthrough_and_decline_errors(self):
        marker = object()
        events = []

        class Left:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("left", func, types, args, kwargs))
                return NotImplemented

        class Right:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("right", func, types, args, kwargs))
                return marker

        self.assertIs(torch.promote_types(Left(), Right()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.promote_types)
            self.assertEqual(dispatch_types, (Left, Right))
            self.assertIsInstance(args[0], Left)
            self.assertIsInstance(args[1], Right)
            self.assertIsNone(kwargs)

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

        self.assertIs(torch.promote_types(Base(), Derived()), marker)
        self.assertEqual(events, [("derived", (Derived, Base))])

        events.clear()

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("override", types, args, kwargs))
                return marker

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(("mode", types, args, kwargs))
                return NotImplemented

        value = Override()
        with DecliningMode():
            self.assertIs(torch.promote_types(value, torch.float32), marker)
        self.assertEqual([event[0] for event in events], ["mode", "override"])
        for _, dispatch_types, args, kwargs in events:
            self.assertEqual(dispatch_types, (Override,))
            self.assertIs(args[0], value)
            self.assertIs(args[1], torch.float32)
            self.assertIsNone(kwargs)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.promote_types(DecliningOverride(), torch.float32)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.promote_types'; all "
            "__torch_function__ handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with "
            "TORCH_LOGS=not_implemented",
        )

    def test_torch_function_handler_arity_preserves_absent_and_empty_kwargs(self):
        marker = object()

        class StrictMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=()):
                return marker

        with StrictMode():
            self.assertIs(
                torch.promote_types(torch.float32, torch.float32), marker
            )
        with StrictMode():
            with self.assertRaises(TypeError):
                torch.promote_types(torch.float32, torch.float32, **{})

        class StrictOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=()):
                return marker

        value = StrictOverride()
        self.assertIs(torch.promote_types(value, torch.float32), marker)
        with self.assertRaises(TypeError):
            torch.promote_types(value, torch.float32, **{})

    def test_modes_run_after_validation_and_match_decline_errors(self):
        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return torch.float32

        invalid_calls = (
            (
                lambda: torch.promote_types(1, torch.float32),
                "promote_types(): argument 'type1' (position 1) must be "
                "torch.dtype, not int",
            ),
            (
                lambda: torch.promote_types(torch.float32),
                'promote_types() missing 1 required positional arguments: "type2"',
            ),
            (
                lambda: torch.promote_types(
                    torch.float32, torch.float32, extra=True
                ),
                "promote_types() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.promote_types(
                    torch.float32, type1=torch.float32, type2=torch.float32
                ),
                "promote_types() got multiple values for argument 'type1'",
            ),
        )
        for call, message in invalid_calls:
            mode = RecordingMode()
            with mode:
                self.assert_error(TypeError, message, call)
            self.assertEqual(mode.calls, [])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        mode = DecliningMode()
        message = (
            "Multiple dispatch failed for 'torch.promote_types'; all "
            "__torch_function__ handlers returned NotImplemented:\n\n"
            f"  - mode object {mode!r}\n\n"
            "For more information, try re-running with "
            "TORCH_LOGS=not_implemented"
        )
        with mode:
            self.assert_error(
                TypeError,
                message,
                lambda: torch.promote_types(torch.float32, torch.float32),
            )

        class ClassMethodMode(torch.overrides.TorchFunctionMode):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return torch.float32

        with ClassMethodMode():
            self.assert_error(
                RuntimeError,
                "Defining your mode's `__torch_function__` as a classmethod is "
                "not supported, please make it a plain method",
                lambda: torch.promote_types(torch.float32, torch.float32),
            )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_binding_type_and_error_precedence_matches_pytorch_2_13(self):
        dtype = torch.float32
        cases = (
            (
                lambda: torch.promote_types(),
                'promote_types() missing 2 required positional argument: '
                '"type1", "type2"',
            ),
            (
                lambda: torch.promote_types(dtype),
                'promote_types() missing 1 required positional arguments: "type2"',
            ),
            (
                lambda: torch.promote_types(1),
                "promote_types(): argument 'type1' (position 1) must be "
                "torch.dtype, not int",
            ),
            (
                lambda: torch.promote_types(type2=dtype),
                'promote_types() missing 2 required positional argument: '
                '"type1", "type2"',
            ),
            (
                lambda: torch.promote_types(dtype, dtype, dtype),
                "promote_types() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.promote_types(1, dtype),
                "promote_types(): argument 'type1' (position 1) must be "
                "torch.dtype, not int",
            ),
            (
                lambda: torch.promote_types(dtype, None),
                "promote_types(): argument 'type2' (position 2) must be "
                "torch.dtype, not NoneType",
            ),
            (
                lambda: torch.promote_types(type1=1, type2=dtype),
                "promote_types(): argument 'type1' must be torch.dtype, not int",
            ),
            (
                lambda: torch.promote_types(type1=dtype, type2=True),
                "promote_types(): argument 'type2' must be torch.dtype, not bool",
            ),
            (
                lambda: torch.promote_types(dtype, type1=dtype, type2=dtype),
                "promote_types() got multiple values for argument 'type1'",
            ),
            (
                lambda: torch.promote_types(
                    1, type1=dtype, type2=dtype
                ),
                "promote_types(): argument 'type1' (position 1) must be "
                "torch.dtype, not int",
            ),
            (
                lambda: torch.promote_types(dtype, dtype, extra=True),
                "promote_types() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.promote_types(
                    dtype, dtype, **{"extra\x00suffix": True}
                ),
                "promote_types() got an unexpected keyword argument 'extra",
            ),
            (
                lambda: torch.promote_types(dtype, 1, extra=True),
                "promote_types(): argument 'type2' (position 2) must be "
                "torch.dtype, not int",
            ),
            (
                lambda: torch.promote_types(type1=dtype, extra=True),
                'promote_types() missing 1 required positional arguments: "type2"',
            ),
            (
                lambda: torch.promote_types(dtype1=dtype, dtype2=dtype),
                'promote_types() missing 2 required positional argument: '
                '"type1", "type2"',
            ),
            (
                lambda: torch.promote_types(
                    dtype, extra=True, type1=dtype, type2=dtype
                ),
                "promote_types() got an unexpected keyword argument 'extra'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                self.assert_error(TypeError, message, call)

    def test_callable_metadata_exports_and_pickling_match_pytorch(self):
        function = torch.promote_types
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "promote_types")
        self.assertEqual(
            function.__qualname__, "_VariableFunctionsClass.promote_types"
        )
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method promote_types of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        reducer, (owner, name) = function.__reduce__()
        self.assertIs(reducer, getattr)
        self.assertEqual(name, "promote_types")
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.promote_types, function)
        self.assertIsNone(function.__self__)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("promote_types"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["promote_types"], function)


if __name__ == "__main__":
    unittest.main()
