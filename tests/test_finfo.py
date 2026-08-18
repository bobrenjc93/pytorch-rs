import copy
import inspect
import pickle
import re
import subprocess
import sys
import textwrap
import types
import unittest

import torch_rs as torch


FINFO_REPR = (
    "finfo(resolution=1e-06, min=-3.40282e+38, max=3.40282e+38, "
    "eps=1.19209e-07, smallest_normal=1.17549e-38, "
    "tiny=1.17549e-38, dtype=float32)"
)


class FInfoTests(unittest.TestCase):
    def assert_error(self, exception_type, message, call):
        with self.assertRaisesRegex(exception_type, f"^{re.escape(message)}$"):
            call()

    def test_default_and_float32_aliases_return_fresh_native_metadata(self):
        values = (
            torch.finfo(),
            torch.finfo(torch.float32),
            torch.finfo(torch.float),
            torch.finfo(type=torch.float32),
            torch.finfo(type=torch.float),
        )
        self.assertIs(torch.float, torch.float32)
        self.assertEqual(len({id(value) for value in values}), len(values))

        expected = {
            "dtype": "float32",
            "bits": 32,
            "eps": 1.1920928955078125e-07,
            "max": 3.4028234663852886e38,
            "min": -3.4028234663852886e38,
            "resolution": 1e-06,
            "smallest_normal": 1.1754943508222875e-38,
            "tiny": 1.1754943508222875e-38,
        }
        for value in values:
            with self.subTest(value=value):
                self.assertIs(type(value), torch.finfo)
                self.assertEqual(
                    {name: getattr(value, name) for name in expected}, expected
                )
                self.assertIs(type(value.dtype), str)
                self.assertIs(type(value.bits), int)
                for name in expected.keys() - {"dtype", "bits"}:
                    self.assertIs(type(getattr(value, name)), float)
                self.assertEqual(repr(value), FINFO_REPR)
                self.assertEqual(str(value), FINFO_REPR)
                self.assertFalse(hasattr(value, "__dict__"))
                self.assertFalse(hasattr(value, "__weakref__"))

        value = values[0]
        self.assertIsNot(value.dtype, value.dtype)
        self.assertIsNot(value.eps, value.eps)
        self.assertIsNot(value.tiny, value.smallest_normal)

    def test_equality_and_unhashability_match_the_float32_scalar_type(self):
        values = (
            torch.finfo(),
            torch.finfo(),
            torch.finfo(torch.float32),
            torch.finfo(torch.float),
        )
        for left in values:
            for right in values:
                self.assertIs(left == right, True)
                self.assertIs(left != right, False)

        value = values[0]
        for dtype in (torch.float32, torch.float):
            self.assertIs(value == dtype, True)
            self.assertIs(dtype == value, True)
            self.assertIs(value != dtype, False)
            self.assertIs(torch.finfo.__eq__(value, dtype), True)
            self.assertIs(torch.finfo.__ne__(value, dtype), False)
        for other in (None, object(), 32, "float32"):
            self.assertIs(value == other, False)
            self.assertIs(value != other, True)
            self.assertIs(torch.finfo.__eq__(value, other), False)
            self.assertIs(torch.finfo.__ne__(value, other), True)

        self.assertIs(torch.finfo.__hash__, None)
        self.assert_error(
            TypeError,
            "unhashable type: 'torch_rs.finfo'",
            lambda: hash(value),
        )

    def test_copy_and_pickle_are_rejected_for_every_protocol(self):
        value = torch.finfo()
        short_error = "cannot pickle 'finfo' object"
        qualified_error = "cannot pickle 'torch_rs.finfo' object"

        self.assert_error(TypeError, short_error, value.__reduce__)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            expected = short_error if protocol < 2 else qualified_error
            with self.subTest(protocol=protocol, operation="reduce_ex"):
                self.assert_error(
                    TypeError,
                    expected,
                    lambda protocol=protocol: value.__reduce_ex__(protocol),
                )
            with self.subTest(protocol=protocol, operation="pickle"):
                self.assert_error(
                    TypeError,
                    expected,
                    lambda protocol=protocol: pickle.dumps(
                        value, protocol=protocol
                    ),
                )

        self.assert_error(TypeError, qualified_error, lambda: copy.copy(value))
        self.assert_error(
            TypeError, qualified_error, lambda: copy.deepcopy(value)
        )

    def test_type_and_descriptor_metadata_are_builtin_and_immutable(self):
        finfo_type = torch.finfo
        self.assertIs(type(finfo_type), type)
        self.assertEqual(finfo_type.__name__, "finfo")
        self.assertEqual(finfo_type.__qualname__, "finfo")
        self.assertEqual(finfo_type.__module__, "torch_rs")
        self.assertIsNone(finfo_type.__doc__)
        self.assertIsNone(finfo_type.__text_signature__)
        self.assertEqual(finfo_type.__bases__, (object,))
        self.assertTrue(finfo_type.__flags__ & (1 << 8))
        self.assertFalse(finfo_type.__flags__ & (1 << 10))
        self.assertEqual(finfo_type.__basicsize__, 24)
        self.assertEqual(finfo_type.__itemsize__, 0)
        self.assertEqual(finfo_type.__dictoffset__, 0)
        self.assertEqual(finfo_type.__weakrefoffset__, 0)
        self.assertIs(torch._C.finfo, finfo_type)
        self.assertIn("finfo", torch.__all__)

        self.assertEqual(finfo_type.__new__.__text_signature__, "($type, *args, **kwargs)")
        self.assertEqual(
            finfo_type.__new__.__doc__,
            "Create and return a new object.  See help(type) for accurate signature.",
        )
        with self.assertRaisesRegex(
            ValueError,
            r"^no signature found for builtin type <class 'torch_rs\.finfo'>$",
        ):
            inspect.signature(finfo_type)

        value = finfo_type()
        for name in (
            "dtype",
            "bits",
            "eps",
            "max",
            "min",
            "resolution",
            "smallest_normal",
            "tiny",
        ):
            with self.subTest(name=name):
                descriptor = inspect.getattr_static(finfo_type, name)
                self.assertIs(type(descriptor), types.GetSetDescriptorType)
                self.assertIs(descriptor.__objclass__, finfo_type)
                self.assertEqual(descriptor.__name__, name)
                self.assertEqual(descriptor.__qualname__, f"finfo.{name}")
                self.assertIsNone(descriptor.__doc__)
                self.assertIs(descriptor.__get__(None, finfo_type), descriptor)
                self.assertEqual(
                    descriptor.__get__(value, finfo_type), getattr(value, name)
                )
                message = (
                    f"attribute '{name}' of 'torch_rs.finfo' objects "
                    "is not writable"
                )
                self.assert_error(
                    AttributeError,
                    message,
                    lambda name=name: setattr(value, name, 0),
                )
                self.assert_error(
                    AttributeError,
                    message,
                    lambda name=name: delattr(value, name),
                )

        self.assert_error(
            TypeError,
            "cannot set 'marker' attribute of immutable type 'torch_rs.finfo'",
            lambda: setattr(finfo_type, "marker", object()),
        )
        self.assert_error(
            TypeError,
            "type 'torch_rs.finfo' is not an acceptable base type",
            lambda: type("Derived", (finfo_type,), {}),
        )

    def test_type_diagnostics_use_native_names_without_metaclass_dispatch(self):
        values = (
            (torch.device("cpu"), "torch.device"),
            (torch.strided, "torch.layout"),
            (torch.contiguous_format, "torch.memory_format"),
            (torch.Size([1]), "torch.Size"),
            (torch.tensor([1.0]), "Tensor"),
            (torch.finfo(), "torch.finfo"),
        )
        for value, name in values:
            with self.subTest(name=name):
                self.assert_error(
                    TypeError,
                    "finfo(): argument 'type' (position 1) must be "
                    f"torch.dtype, not {name}",
                    lambda value=value: torch.finfo(value),
                )

        lookups = []

        class HostileMeta(type):
            def __getattribute__(cls, name):
                lookups.append(name)
                if name == "__module__":
                    raise RuntimeError("metaclass module trap")
                return super().__getattribute__(name)

        class Value(metaclass=HostileMeta):
            pass

        lookups.clear()
        self.assert_error(
            TypeError,
            "finfo(): argument 'type' (position 1) must be torch.dtype, not Value",
            lambda: torch.finfo(Value()),
        )
        self.assertEqual(lookups, [])
        self.assert_error(
            TypeError,
            "finfo() received an invalid combination of arguments - got "
            "(Value, object), but expected one of:\n"
            " * (torch.dtype type)\n * ()\n",
            lambda: torch.finfo(Value(), object()),
        )
        self.assertEqual(lookups, [])

    def test_constructor_errors_match_without_expanding_other_dtypes(self):
        cases = (
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
                lambda: torch.finfo(dtype=torch.float32),
                'finfo() missing 1 required positional arguments: "type"',
            ),
            (
                lambda: torch.finfo(unexpected=1),
                'finfo() missing 1 required positional arguments: "type"',
            ),
            (
                lambda: torch.finfo(torch.float32, torch.float32),
                "finfo() received an invalid combination of arguments - got "
                "(torch.dtype, torch.dtype), but expected one of:\n"
                " * (torch.dtype type)\n * ()\n",
            ),
            (
                lambda: torch.finfo(
                    type=torch.float32, dtype=torch.float32
                ),
                "finfo() received an invalid combination of arguments - got "
                "(dtype=torch.dtype, type=torch.dtype, ), but expected one of:\n"
                " * (torch.dtype type)\n * ()\n",
            ),
            (
                lambda: torch.finfo(
                    dtype=torch.float32, unexpected=1
                ),
                "finfo() received an invalid combination of arguments - got "
                "(unexpected=int, dtype=torch.dtype, ), but expected one of:\n"
                " * (torch.dtype type)\n * ()\n",
            ),
            (
                lambda: torch.finfo(first=1, second=2),
                "finfo() received an invalid combination of arguments - got "
                "(second=int, first=int, ), but expected one of:\n"
                " * (torch.dtype type)\n * ()\n",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                self.assert_error(TypeError, message, call)

        self.assertFalse(hasattr(torch, "float64"))
        self.assert_error(
            TypeError,
            "finfo(): argument 'type' (position 1) must be torch.dtype, not type",
            lambda: torch.finfo(float),
        )

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux RLIMIT_AS")
    def test_large_invalid_call_raises_bad_alloc_instead_of_aborting(self):
        script = textwrap.dedent(
            """\
            import os
            import resource

            import torch_rs as torch

            arguments = (None,) * 2_000_000
            with open("/proc/self/statm", encoding="ascii") as statm:
                virtual_pages = int(statm.read().split()[0])
            current_virtual_size = virtual_pages * os.sysconf("SC_PAGE_SIZE")
            limit = current_virtual_size + 8 * 1024 * 1024
            _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if hard_limit != resource.RLIM_INFINITY and limit > hard_limit:
                raise SystemExit(77)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard_limit))

            try:
                torch.finfo(*arguments)
            except RuntimeError as error:
                assert str(error) == "std::bad_alloc", repr(error)
            else:
                raise AssertionError("the constrained call unexpectedly succeeded")
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        if completed.returncode == 77:
            self.skipTest("process hard address-space limit is too low")
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
