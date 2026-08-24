import copy
import importlib
import inspect
import pickle
import types
import unittest
from collections.abc import Mapping, Sequence

import torch_rs as torch

from torch_rs.utils.data import default_collate


UNSUPPORTED_ERROR = (
    "default_collate(): tensor, numeric, mapping, and nested sequence batches "
    "are not supported"
)


class Text(str):
    pass


class Blob(bytes):
    pass


class StringClassSpoof:
    def __init__(self):
        self.class_lookups = 0

    @property
    def __class__(self):
        self.class_lookups += 1
        return str


class SpoofedInt(int):
    @property
    def __class__(self):
        return str


class SpoofedFloat(float):
    @property
    def __class__(self):
        return str


class IntClassText(str):
    @property
    def __class__(self):
        return int


class RaisingText(str):
    @property
    def __class__(self):
        raise RuntimeError("hostile string class lookup")


class RaisingBlob(bytes):
    @property
    def __class__(self):
        raise RuntimeError("hostile bytes class lookup")


class MappingProbe(Mapping):
    def __getitem__(self, key):
        raise AssertionError("mapping contents must not be read")

    def __iter__(self):
        raise AssertionError("mapping contents must not be traversed")

    def __len__(self):
        raise AssertionError("mapping length must not be read")


class SequenceProbe(Sequence):
    def __getitem__(self, index):
        raise AssertionError("nested sequence contents must not be read")

    def __len__(self):
        raise AssertionError("nested sequence length must not be read")


class DefaultCollateTests(unittest.TestCase):
    def test_string_and_bytes_batches_preserve_exact_identity(self):
        mapping_probe = MappingProbe()
        sequence_probe = SequenceProbe()
        batches = (
            ["value"],
            ("value",),
            [b"value"],
            (b"value",),
            ["", 1, mapping_probe, sequence_probe],
            (b"", torch.tensor([1.0]), mapping_probe, sequence_probe),
            [Text("value"), object()],
            (Blob(b"value"), object()),
        )

        for batch in batches:
            with self.subTest(
                container=type(batch).__name__, leaf=type(batch[0]).__name__
            ):
                self.assertIs(default_collate(batch), batch)

        mutable = ["first", "second"]
        result = default_collate(mutable)
        result.append("third")
        self.assertEqual(mutable, ["first", "second", "third"])

    def test_ordered_type_dispatch_precedes_string_and_bytes_handlers(self):
        for first in (SpoofedInt(7), SpoofedFloat(1.5), IntClassText("value")):
            with self.subTest(type=type(first).__name__):
                with self.assertRaises(TypeError) as raised:
                    default_collate([first])
                self.assertEqual(raised.exception.args, (UNSUPPORTED_ERROR,))

        spoof = StringClassSpoof()
        batch = [spoof]
        self.assertIs(default_collate(batch), batch)
        self.assertGreater(spoof.class_lookups, 1)

        for first, message in (
            (RaisingText("value"), "hostile string class lookup"),
            (RaisingBlob(b"value"), "hostile bytes class lookup"),
        ):
            with self.subTest(type=type(first).__name__):
                with self.assertRaises(RuntimeError) as raised:
                    default_collate([first])
                self.assertEqual(raised.exception.args, (message,))

    def test_unsupported_collation_paths_are_rejected_without_traversal(self):
        unsupported = (
            torch.tensor([1.0, 2.0]),
            None,
            True,
            1,
            1.5,
            complex(2.0, -3.0),
            MappingProbe(),
            SequenceProbe(),
            ["nested"],
            (b"nested",),
        )

        for first in unsupported:
            with self.subTest(type=type(first).__name__):
                batch = [first, MappingProbe(), SequenceProbe()]
                with self.assertRaises(TypeError) as raised:
                    default_collate(batch)
                self.assertEqual(raised.exception.args, (UNSUPPORTED_ERROR,))

    def test_empty_and_non_subscriptable_inputs_keep_python_indexing_errors(self):
        cases = (
            (lambda: default_collate([]), IndexError, "list index out of range"),
            (lambda: default_collate(()), IndexError, "tuple index out of range"),
            (
                lambda: default_collate(None),
                TypeError,
                "'NoneType' object is not subscriptable",
            ),
            (
                lambda: default_collate(1),
                TypeError,
                "'int' object is not subscriptable",
            ),
            (
                lambda: default_collate(object()),
                TypeError,
                "'object' object is not subscriptable",
            ),
            (
                lambda: default_collate(iter(("value",))),
                TypeError,
                "'tuple_iterator' object is not subscriptable",
            ),
        )

        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_signature_call_errors_imports_and_narrow_exports(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        collate_module = importlib.import_module("torch_rs.utils.data._utils.collate")
        private_package = importlib.import_module("torch_rs.utils.data._utils")
        dataset_module = importlib.import_module("torch_rs.utils.data.dataset")

        self.assertIs(type(default_collate), types.FunctionType)
        self.assertEqual(str(inspect.signature(default_collate)), "(batch)")
        self.assertEqual(default_collate.__annotations__, {})
        self.assertEqual(
            default_collate.__module__, "torch_rs.utils.data._utils.collate"
        )
        self.assertEqual(default_collate.__name__, "default_collate")
        self.assertEqual(default_collate.__qualname__, "default_collate")
        self.assertIn(
            "Return a supported string or bytes batch unchanged",
            default_collate.__doc__,
        )
        self.assertIn("later elements are not inspected", default_collate.__doc__)
        self.assertIsNone(default_collate.__defaults__)
        self.assertIsNone(default_collate.__kwdefaults__)
        self.assertEqual(default_collate.__dict__, {})
        self.assertFalse(hasattr(default_collate, "__text_signature__"))

        self.assertIs(torch.utils.data.default_collate, default_collate)
        self.assertIs(data_module.default_collate, default_collate)
        self.assertIs(collate_module.default_collate, default_collate)
        self.assertNotIn("default_collate", private_package.__dict__)
        self.assertNotIn("default_collate", dataset_module.__dict__)
        self.assertEqual(data_module.__all__.count("default_collate"), 1)
        self.assertFalse(hasattr(collate_module, "__all__"))
        self.assertNotIn("default_collate", torch.__all__)

        namespace = {}
        exec("from torch_rs.utils.data import *", namespace)
        self.assertIs(namespace["default_collate"], default_collate)

        self.assertFalse(hasattr(data_module, "DataLoader"))
        for unsupported in (
            "collate",
            "collate_str_fn",
            "default_collate_fn_map",
            "default_convert",
        ):
            with self.subTest(unsupported=unsupported):
                self.assertFalse(hasattr(collate_module, unsupported))

        marker = ["value"]
        self.assertIs(default_collate(batch=marker), marker)
        for call, message in (
            (
                lambda: default_collate(),
                "default_collate() missing 1 required positional argument: 'batch'",
            ),
            (
                lambda: default_collate(None, None),
                "default_collate() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: default_collate(value=None),
                "default_collate() got an unexpected keyword argument 'value'",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_copy_and_pickle_preserve_the_global_function(self):
        self.assertIs(copy.copy(default_collate), default_collate)
        self.assertIs(copy.deepcopy(default_collate), default_collate)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(
                    pickle.dumps(default_collate, protocol=protocol)
                )
                self.assertIs(restored, default_collate)


if __name__ == "__main__":
    unittest.main()
