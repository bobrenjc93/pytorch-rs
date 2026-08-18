import copy
import inspect
import pickle
import re
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = r"""
can_cast(from_, to) -> bool

Determines if a type conversion is allowed under PyTorch casting rules
described in the type promotion :ref:`documentation <type-promotion-doc>`.

Args:
    from\_ (dtype): The original :class:`torch.dtype`.
    to (dtype): The target :class:`torch.dtype`.

Example::

    >>> torch.can_cast(torch.double, torch.float)
    True
    >>> torch.can_cast(torch.float, torch.int)
    False
"""


class CanCastTests(unittest.TestCase):
    def assert_error(self, exception_type, message, call):
        with self.assertRaisesRegex(exception_type, f"^{re.escape(message)}$"):
            call()

    def test_float32_aliases_and_canonical_call_forms_return_true(self):
        self.assertIs(torch.float, torch.float32)
        for from_dtype in (torch.float32, torch.float):
            for to_dtype in (torch.float32, torch.float):
                calls = (
                    ("positional", lambda: torch.can_cast(from_dtype, to_dtype)),
                    ("mixed", lambda: torch.can_cast(from_dtype, to=to_dtype)),
                    (
                        "keywords",
                        lambda: torch.can_cast(from_=from_dtype, to=to_dtype),
                    ),
                    (
                        "reversed keywords",
                        lambda: torch.can_cast(to=to_dtype, from_=from_dtype),
                    ),
                )
                for form, call in calls:
                    with self.subTest(
                        from_dtype=from_dtype, to_dtype=to_dtype, form=form
                    ):
                        self.assertIs(call(), True)

    def test_dtype_validation_uses_one_shot_torch_function_probe(self):
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
        self.assert_error(
            TypeError,
            "can_cast(): argument 'from_' (position 1) must be "
            "torch.dtype, not StatefulOperand",
            lambda: torch.can_cast(operand, torch.float32),
        )
        self.assertEqual(events, [("lookup", 1)])
        self.assertEqual(operand.lookups, 1)

    def test_first_operand_validation_precedes_second_keyword_lookup(self):
        for form, positional in (("positional", True), ("keyword", False)):
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
            expected_position = " (position 1)" if positional else ""
            call = (
                (lambda: torch.can_cast(operand, **{key: torch.float32}))
                if positional
                else (
                    lambda: torch.can_cast(
                        from_=operand, **{key: torch.float32}
                    )
                )
            )
            with self.subTest(form=form):
                self.assert_error(
                    TypeError,
                    "can_cast(): argument 'from_'"
                    f"{expected_position} must be torch.dtype, not Operand",
                    call,
                )
                self.assertEqual(events, [])

    def test_torch_function_mode_observes_original_call_and_forwards(self):
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        forms = (
            (
                lambda: torch.can_cast(torch.float32, torch.float),
                (torch.float32, torch.float32),
                None,
            ),
            (
                lambda: torch.can_cast(torch.float, to=torch.float32),
                (torch.float32,),
                {"to": torch.float32},
            ),
            (
                lambda: torch.can_cast(
                    from_=torch.float32, to=torch.float
                ),
                (),
                {"from_": torch.float32, "to": torch.float32},
            ),
            (
                lambda: torch.can_cast(
                    to=torch.float, from_=torch.float32
                ),
                (),
                {"to": torch.float32, "from_": torch.float32},
            ),
        )
        for call, expected_args, expected_kwargs in forms:
            mode = RecordingMode()
            with mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.can_cast)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                self.assertIs(
                    torch.can_cast(to=torch.float, from_=torch.float32),
                    True,
                )
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_operand_torch_function_override_dispatch(self):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        forms = (
            lambda value: torch.can_cast(value, torch.float32),
            lambda value: torch.can_cast(torch.float32, value),
            lambda value: torch.can_cast(from_=value, to=torch.float32),
            lambda value: torch.can_cast(from_=torch.float32, to=value),
        )
        for call in forms:
            value = Override()
            Override.calls = []
            self.assertIs(call(value), marker)
            self.assertEqual(len(Override.calls), 1)
            function, dispatch_types, _, _ = Override.calls[0]
            self.assertIs(function, torch.can_cast)
            self.assertEqual(dispatch_types, (Override,))

    def test_invalid_argument_errors_and_validation_precede_mode_dispatch(self):
        dtype = torch.float32
        cases = (
            (
                lambda: torch.can_cast(),
                'can_cast() missing 2 required positional argument: "from_", "to"',
            ),
            (
                lambda: torch.can_cast(dtype),
                'can_cast() missing 1 required positional arguments: "to"',
            ),
            (
                lambda: torch.can_cast(dtype, dtype, dtype),
                "can_cast() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.can_cast(1, dtype),
                "can_cast(): argument 'from_' (position 1) must be "
                "torch.dtype, not int",
            ),
            (
                lambda: torch.can_cast(dtype, None),
                "can_cast(): argument 'to' (position 2) must be "
                "torch.dtype, not NoneType",
            ),
            (
                lambda: torch.can_cast(from_=1, to=dtype),
                "can_cast(): argument 'from_' must be torch.dtype, not int",
            ),
            (
                lambda: torch.can_cast(from_=dtype, to=True),
                "can_cast(): argument 'to' must be torch.dtype, not bool",
            ),
            (
                lambda: torch.can_cast(dtype, from_=dtype, to=dtype),
                "can_cast() got multiple values for argument 'from_'",
            ),
            (
                lambda: torch.can_cast(dtype, dtype, to=dtype),
                "can_cast() got multiple values for argument 'to'",
            ),
            (
                lambda: torch.can_cast(dtype, dtype, extra=True),
                "can_cast() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.can_cast(dtype, 1, extra=True),
                "can_cast(): argument 'to' (position 2) must be "
                "torch.dtype, not int",
            ),
            (
                lambda: torch.can_cast(from_=dtype, extra=True),
                'can_cast() missing 1 required positional arguments: "to"',
            ),
            (
                lambda: torch.can_cast(to=dtype),
                'can_cast() missing 2 required positional argument: "from_", "to"',
            ),
            (
                lambda: torch.can_cast(from_=dtype, to_=dtype),
                'can_cast() missing 1 required positional arguments: "to"',
            ),
        )

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return True

        for call, message in cases:
            with self.subTest(message=message):
                self.assert_error(TypeError, message, call)
                mode = RecordingMode()
                with mode:
                    self.assert_error(TypeError, message, call)
                self.assertEqual(mode.calls, [])

    def test_callable_metadata_exports_and_pickling_match_pytorch(self):
        function = torch.can_cast
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "can_cast")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.can_cast")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method can_cast of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        reducer, (owner, name) = function.__reduce__()
        self.assertIs(reducer, getattr)
        self.assertEqual(name, "can_cast")
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.can_cast, function)
        self.assertIsNone(function.__self__)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("can_cast"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["can_cast"], function)


if __name__ == "__main__":
    unittest.main()
