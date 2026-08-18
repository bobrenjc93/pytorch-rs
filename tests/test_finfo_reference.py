import copy
import inspect
import pickle
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class FInfoReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("finfo differentials require pinned PyTorch 2.13.0")

    def normalize(self, module, value):
        return value.replace(module.finfo.__module__, "torch")

    def error(self, module, call):
        try:
            call()
        except Exception as error:
            return type(error).__name__, self.normalize(module, str(error))
        self.fail(f"{module.__name__} unexpectedly accepted an operation")

    def metadata(self, value):
        return {
            "type_exact": type(value) is value.__class__,
            "dtype": (type(value.dtype).__name__, value.dtype),
            "bits": (type(value.bits).__name__, value.bits),
            "eps": (type(value.eps).__name__, value.eps),
            "max": (type(value.max).__name__, value.max),
            "min": (type(value.min).__name__, value.min),
            "resolution": (
                type(value.resolution).__name__,
                value.resolution,
            ),
            "smallest_normal": (
                type(value.smallest_normal).__name__,
                value.smallest_normal,
            ),
            "tiny": (type(value.tiny).__name__, value.tiny),
            "repr": repr(value),
            "str": str(value),
            "has_dict": hasattr(value, "__dict__"),
            "has_weakref": hasattr(value, "__weakref__"),
        }

    def test_float32_values_aliases_freshness_and_equality_match(self):
        def contract(module):
            values = (
                module.finfo(),
                module.finfo(module.float32),
                module.finfo(module.float),
                module.finfo(type=module.float32),
            )
            return {
                "metadata": tuple(self.metadata(value) for value in values),
                "all_fresh": len({id(value) for value in values}) == len(values),
                "pairwise_equality": tuple(
                    tuple(
                        (left is right, left == right, left != right)
                        for right in values
                    )
                    for left in values
                ),
                "dtype_equality": tuple(
                    (
                        values[0] == dtype,
                        dtype == values[0],
                        values[0] != dtype,
                        module.finfo.__eq__(values[0], dtype),
                        module.finfo.__ne__(values[0], dtype),
                    )
                    for dtype in (module.float32, module.float)
                ),
                "other_equality": tuple(
                    (
                        values[0] == other,
                        values[0] != other,
                        module.finfo.__eq__(values[0], other),
                        module.finfo.__ne__(values[0], other),
                    )
                    for other in (None, 32, "float32")
                ),
                "fresh_attributes": (
                    values[0].dtype is not values[0].dtype,
                    values[0].eps is not values[0].eps,
                    values[0].tiny is not values[0].smallest_normal,
                ),
            }

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_builtin_type_and_getset_metadata_match(self):
        def contract(module):
            finfo_type = module.finfo
            value = finfo_type()
            descriptors = []
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
                descriptor = inspect.getattr_static(finfo_type, name)
                descriptors.append(
                    (
                        type(descriptor) is types.GetSetDescriptorType,
                        descriptor.__objclass__ is finfo_type,
                        descriptor.__name__,
                        descriptor.__qualname__,
                        descriptor.__doc__,
                        descriptor.__get__(None, finfo_type) is descriptor,
                        descriptor.__get__(value, finfo_type),
                        self.error(
                            module,
                            lambda name=name: setattr(value, name, 0),
                        ),
                        self.error(
                            module,
                            lambda name=name: delattr(value, name),
                        ),
                    )
                )
            try:
                inspect.signature(finfo_type)
            except Exception as error:
                signature_error = (
                    type(error).__name__,
                    self.normalize(module, str(error)),
                )
            else:
                signature_error = None
            return {
                "metatype": type(finfo_type) is type,
                "name": finfo_type.__name__,
                "qualname": finfo_type.__qualname__,
                "module": self.normalize(module, finfo_type.__module__),
                "repr": self.normalize(module, repr(finfo_type)),
                "doc": finfo_type.__doc__,
                "text_signature": finfo_type.__text_signature__,
                "bases": finfo_type.__bases__ == (object,),
                "immutable": bool(finfo_type.__flags__ & (1 << 8)),
                "base_type": bool(finfo_type.__flags__ & (1 << 10)),
                "basicsize": finfo_type.__basicsize__,
                "itemsize": finfo_type.__itemsize__,
                "dictoffset": finfo_type.__dictoffset__,
                "weakrefoffset": finfo_type.__weakrefoffset__,
                "new_text_signature": finfo_type.__new__.__text_signature__,
                "new_doc": finfo_type.__new__.__doc__,
                "signature_error": signature_error,
                "hash_is_none": finfo_type.__hash__ is None,
                "repr_descriptor": type(
                    inspect.getattr_static(finfo_type, "__repr__")
                ).__name__,
                "str_descriptor": type(
                    inspect.getattr_static(finfo_type, "__str__")
                ).__name__,
                "eq_descriptor": type(
                    inspect.getattr_static(finfo_type, "__eq__")
                ).__name__,
                "ne_descriptor": type(
                    inspect.getattr_static(finfo_type, "__ne__")
                ).__name__,
                "descriptors": tuple(descriptors),
                "in_all": "finfo" in module.__all__,
                "native_identity": module._C.finfo is finfo_type,
                "mutation_error": self.error(
                    module, lambda: setattr(finfo_type, "marker", object())
                ),
                "subclass_error": self.error(
                    module, lambda: type("Derived", (finfo_type,), {})
                ),
            }

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_constructor_errors_and_unpicklability_match(self):
        def contract(module):
            value = module.finfo()
            constructors = (
                lambda: module.finfo(None),
                lambda: module.finfo(type=None),
                lambda: module.finfo("float32"),
                lambda: module.finfo(dtype=module.float32),
                lambda: module.finfo(unexpected=1),
                lambda: module.finfo(module.float32, module.float32),
                lambda: module.finfo(
                    type=module.float32, dtype=module.float32
                ),
                lambda: module.finfo(module.float32, type=module.float32),
                lambda: module.finfo(
                    dtype=module.float32, unexpected=1
                ),
                lambda: module.finfo(first=1, second=2),
                lambda: module.finfo(
                    **{
                        "\ud800": module.float32,
                        "other": module.float32,
                    }
                ),
                lambda: module.finfo(
                    **{
                        f"key{index}": module.float32
                        for index in range(14)
                    }
                ),
                lambda: module.finfo(
                    **{
                        f"key{index}": module.float32
                        for index in range(258)
                    }
                ),
            )
            pickle_errors = tuple(
                self.error(
                    module,
                    lambda protocol=protocol: pickle.dumps(
                        value, protocol=protocol
                    ),
                )
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            )
            reduce_ex_errors = tuple(
                self.error(
                    module,
                    lambda protocol=protocol: value.__reduce_ex__(protocol),
                )
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            )
            return {
                "constructors": tuple(
                    self.error(module, constructor)
                    for constructor in constructors
                ),
                "hash": self.error(module, lambda: hash(value)),
                "reduce": self.error(module, value.__reduce__),
                "reduce_ex": reduce_ex_errors,
                "pickle": pickle_errors,
                "copy": self.error(module, lambda: copy.copy(value)),
                "deepcopy": self.error(
                    module, lambda: copy.deepcopy(value)
                ),
            }

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_public_and_hostile_metaclass_type_diagnostics_match(self):
        def contract(module):
            public_values = (
                module.device("cpu"),
                module.strided,
                module.contiguous_format,
                module.Size([1]),
                module.tensor([1.0]),
                module.finfo(),
            )
            public_errors = tuple(
                self.error(module, lambda value=value: module.finfo(value))
                for value in public_values
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
            metaclass_errors = (
                self.error(module, lambda: module.finfo(Value())),
                self.error(
                    module, lambda: module.finfo(Value(), object())
                ),
            )
            return public_errors, metaclass_errors, tuple(lookups)

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_float64_shorthand_remains_outside_the_supported_boundary(self):
        self.assertFalse(hasattr(torch, "float64"))
        actual = self.error(torch, lambda: torch.finfo(float))
        self.assertEqual(
            actual,
            (
                "TypeError",
                "finfo(): argument 'type' (position 1) must be torch.dtype, not type",
            ),
        )
        self.assertEqual(reference_torch.finfo(float).dtype, "float64")


if __name__ == "__main__":
    unittest.main()
