import copy
import inspect
import pickle
import sys
import types
import unittest

import torch_rs as torch


FLOAT32_REPR = (
    "finfo(resolution=1e-06, min=-3.40282e+38, max=3.40282e+38, "
    "eps=1.19209e-07, smallest_normal=1.17549e-38, "
    "tiny=1.17549e-38, dtype=float32)"
)


class FInfoTests(unittest.TestCase):
    def assert_error(self, exception_type, message, call):
        with self.assertRaises(exception_type) as raised:
            call()
        self.assertEqual(str(raised.exception), message)

    def test_default_and_float32_aliases_return_equal_fresh_objects(self):
        constructors = (
            lambda: torch.finfo(),
            lambda: torch.finfo(torch.float32),
            lambda: torch.finfo(torch.float),
            lambda: torch.finfo(type=torch.float32),
            lambda: torch.finfo(torch.tensor(1.0).dtype),
            lambda: torch.finfo(torch.get_default_dtype()),
        )
        values = tuple(constructor() for constructor in constructors)

        self.assertEqual(len({id(value) for value in values}), len(values))
        for value in values:
            with self.subTest(value=value):
                self.assertIs(type(value), torch.finfo)
                self.assertEqual(value, values[0])
                self.assertEqual(repr(value), FLOAT32_REPR)
                self.assertEqual(str(value), FLOAT32_REPR)

        self.assertIs(torch.float, torch.float32)
        self.assertIsNot(torch.finfo(), torch.finfo())

    def test_float32_metadata_uses_exact_native_limits(self):
        info = torch.finfo(torch.float32)

        self.assertIs(type(info.bits), int)
        self.assertEqual(info.bits, 32)
        self.assertIs(type(info.dtype), str)
        self.assertEqual(info.dtype, "float32")
        expected = {
            "eps": float.fromhex("0x1.0000000000000p-23"),
            "max": float.fromhex("0x1.fffffe0000000p+127"),
            "min": float.fromhex("-0x1.fffffe0000000p+127"),
            "resolution": 1.0e-6,
            "smallest_normal": float.fromhex("0x1.0000000000000p-126"),
            "tiny": float.fromhex("0x1.0000000000000p-126"),
        }
        for name, expected_value in expected.items():
            with self.subTest(name=name):
                value = getattr(info, name)
                self.assertIs(type(value), float)
                self.assertEqual(value.hex(), expected_value.hex())

        self.assertEqual(info.tiny, info.smallest_normal)

    def test_type_and_descriptor_metadata_match_native_builtin_shape(self):
        self.assertIs(type(torch.finfo), type)
        self.assertEqual(torch.finfo.__name__, "finfo")
        self.assertEqual(torch.finfo.__qualname__, "finfo")
        self.assertEqual(torch.finfo.__module__, "torch_rs")
        self.assertIsNone(torch.finfo.__doc__)
        self.assertIn("finfo", torch.__all__)
        surface = set(vars(torch.finfo))
        surface.discard("__module__")
        self.assertEqual(
            surface,
            {
                "__doc__",
                "__eq__",
                "__ge__",
                "__gt__",
                "__hash__",
                "__le__",
                "__lt__",
                "__ne__",
                "__new__",
                "__repr__",
                "__str__",
                "bits",
                "dtype",
                "eps",
                "max",
                "min",
                "resolution",
                "smallest_normal",
                "tiny",
            },
        )
        with self.assertRaises(ValueError):
            inspect.signature(torch.finfo)

        constructor = inspect.getattr_static(torch.finfo, "__new__")
        self.assertIs(type(constructor), types.BuiltinFunctionType)
        self.assertEqual(constructor.__name__, "__new__")
        self.assertEqual(constructor.__qualname__, "finfo.__new__")
        self.assertEqual(constructor.__text_signature__, "($type, *args, **kwargs)")
        self.assertEqual(
            constructor.__doc__,
            "Create and return a new object.  See help(type) for accurate signature.",
        )

        for name in (
            "bits",
            "dtype",
            "eps",
            "max",
            "min",
            "resolution",
            "smallest_normal",
            "tiny",
        ):
            descriptor = inspect.getattr_static(torch.finfo, name)
            with self.subTest(name=name):
                self.assertIs(type(descriptor), types.GetSetDescriptorType)
                self.assertEqual(descriptor.__name__, name)
                self.assertEqual(descriptor.__qualname__, f"finfo.{name}")
                self.assertIs(descriptor.__objclass__, torch.finfo)
                self.assertIsNone(descriptor.__doc__)
                self.assertIs(
                    descriptor.__get__(None, torch.finfo), descriptor
                )

        info = torch.finfo()
        with self.assertRaises(AttributeError):
            info.__dict__
        with self.assertRaises(AttributeError):
            info.__weakref__
        self.assertFalse(hasattr(info, "__getnewargs__"))
        self.assert_error(
            TypeError,
            "type 'torch_rs.finfo' is not an acceptable base type",
            lambda: type("SubFInfo", (torch.finfo,), {}),
        )

    def test_type_object_is_immutable(self):
        actions = (
            ("extra", lambda: setattr(torch.finfo, "extra", 1)),
            (
                "__repr__",
                lambda: setattr(torch.finfo, "__repr__", lambda self: "changed"),
            ),
            ("bits", lambda: delattr(torch.finfo, "bits")),
        )
        for name, action in actions:
            with self.subTest(name=name):
                self.assert_error(
                    TypeError,
                    f"cannot set '{name}' attribute of immutable type 'torch_rs.finfo'",
                    action,
                )

    def test_metadata_is_read_only(self):
        info = torch.finfo()
        for name in (
            "bits",
            "dtype",
            "eps",
            "max",
            "min",
            "resolution",
            "smallest_normal",
            "tiny",
        ):
            descriptor = inspect.getattr_static(torch.finfo, name)
            actions = (
                lambda name=name: setattr(info, name, None),
                lambda name=name: delattr(info, name),
                lambda descriptor=descriptor: descriptor.__set__(info, None),
                lambda descriptor=descriptor: descriptor.__delete__(info),
            )
            for action in actions:
                with self.subTest(name=name, action=action):
                    self.assert_error(
                        AttributeError,
                        f"attribute '{name}' of 'torch_rs.finfo' objects is not writable",
                        action,
                    )

        extra_message = "'torch_rs.finfo' object has no attribute 'extra'"
        if sys.version_info >= (3, 13):
            extra_message += " and no __dict__ for setting new attributes"
        self.assert_error(
            AttributeError,
            extra_message,
            lambda: setattr(info, "extra", 1),
        )

    def test_equality_ordering_and_unhashability(self):
        left = torch.finfo()
        right = torch.finfo(torch.float)

        self.assertIsNot(left, right)
        self.assertEqual(left, right)
        self.assertFalse(left != right)
        self.assertNotEqual(left, None)
        self.assertNotEqual(left, object())
        self.assertFalse(left == (left.bits, left.dtype))

        class Foreign:
            def __init__(self):
                self.events = []

            def __eq__(self, other):
                self.events.append(("eq", other))
                return "reflected equality sentinel"

            def __ne__(self, other):
                self.events.append(("ne", other))
                raise RuntimeError("reflected inequality ran")

        foreign = Foreign()
        self.assertIs(left == foreign, False)
        self.assertEqual(foreign.events, [])
        self.assertIs(left != foreign, True)
        self.assertEqual(foreign.events, [])

        self.assert_error(
            TypeError,
            "unhashable type: 'torch_rs.finfo'",
            lambda: hash(left),
        )
        for symbol, comparison in (
            ("<", lambda: left < right),
            ("<=", lambda: left <= right),
            (">", lambda: left > right),
            (">=", lambda: left >= right),
        ):
            with self.subTest(symbol=symbol):
                self.assert_error(
                    TypeError,
                    f"'{symbol}' not supported between instances of "
                    "'torch_rs.finfo' and 'torch_rs.finfo'",
                    comparison,
                )

    def test_pickle_and_copy_are_unsupported(self):
        info = torch.finfo()
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            expected = (
                "cannot pickle 'finfo' object"
                if protocol < 2
                else "cannot pickle 'torch_rs.finfo' object"
            )
            with self.subTest(protocol=protocol):
                self.assert_error(
                    TypeError,
                    expected,
                    lambda protocol=protocol: pickle.dumps(
                        info, protocol=protocol
                    ),
                )

        for name, action in (
            ("copy", lambda: copy.copy(info)),
            ("deepcopy", lambda: copy.deepcopy(info)),
        ):
            with self.subTest(operation=name):
                self.assert_error(
                    TypeError,
                    "cannot pickle 'torch_rs.finfo' object",
                    action,
                )
        self.assert_error(
            TypeError,
            "cannot pickle 'finfo' object",
            info.__reduce__,
        )
        self.assert_error(
            TypeError,
            "cannot pickle 'torch_rs.finfo' object",
            lambda: info.__reduce_ex__(pickle.HIGHEST_PROTOCOL),
        )

    def test_constructor_errors_and_unsupported_dtypes(self):
        cases = (
            (
                lambda: torch.finfo(dtype=torch.float32),
                'finfo() missing 1 required positional arguments: "type"',
            ),
            (
                lambda: torch.finfo(unexpected=torch.float32),
                'finfo() missing 1 required positional arguments: "type"',
            ),
            (
                lambda: torch.finfo(None),
                "finfo(): argument 'type' (position 1) must be torch.dtype, not NoneType",
            ),
            (
                lambda: torch.finfo(type=None),
                "finfo(): argument 'type' must be torch.dtype, not NoneType",
            ),
            (
                lambda: torch.finfo("float32"),
                "finfo(): argument 'type' (position 1) must be torch.dtype, not str",
            ),
            (
                lambda: torch.finfo(object()),
                "finfo(): argument 'type' (position 1) must be torch.dtype, not object",
            ),
            (
                lambda: torch.finfo(torch.tensor(1.0)),
                "finfo(): argument 'type' (position 1) must be torch.dtype, not Tensor",
            ),
            (
                lambda: torch.finfo(float),
                "finfo(): argument 'type' (position 1) must be torch.dtype, not type",
            ),
            (
                lambda: torch.finfo(torch.float32, torch.float32),
                "finfo() received an invalid combination of arguments - got "
                "(torch.dtype, torch.dtype), but expected one of:\n"
                " * (torch.dtype type)\n * ()\n",
            ),
            (
                lambda: torch.finfo(torch.float32, type=torch.float32),
                "finfo() received an invalid combination of arguments - got "
                "(torch.dtype, type=torch.dtype), but expected one of:\n"
                " * (torch.dtype type)\n * ()\n",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                self.assert_error(TypeError, message, call)

        for name in ("float64", "double", "float16", "half", "bfloat16"):
            with self.subTest(dtype=name):
                self.assertFalse(hasattr(torch, name))


if __name__ == "__main__":
    unittest.main()
