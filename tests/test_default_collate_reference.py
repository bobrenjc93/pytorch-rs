import copy
import importlib
import inspect
import pickle
import pickletools
import types
import unittest
from collections.abc import Mapping, Sequence

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


UNSUPPORTED_ERROR = (
    "default_collate(): tensor, numeric, mapping, and nested sequence batches "
    "are not supported"
)


class Text(str):
    pass


class Blob(bytes):
    pass


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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DefaultCollateReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "default_collate differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    @staticmethod
    def pickle_shape(function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            elif opcode.name == "FRAME":
                argument = None
            shape.append((opcode.name, argument))
        return shape

    def test_string_and_bytes_batch_identity_matches_pytorch_2_13(self):
        actual_mapping_probe = MappingProbe()
        expected_mapping_probe = MappingProbe()
        actual_sequence_probe = SequenceProbe()
        expected_sequence_probe = SequenceProbe()
        batch_pairs = (
            (["value"], ["value"]),
            (("value",), ("value",)),
            ([b"value"], [b"value"]),
            ((b"value",), (b"value",)),
            (
                ["", 1, actual_mapping_probe, actual_sequence_probe],
                ["", 1, expected_mapping_probe, expected_sequence_probe],
            ),
            (
                (b"", torch.tensor([1.0]), actual_mapping_probe),
                (
                    b"",
                    reference_torch.tensor([1.0]),
                    expected_mapping_probe,
                ),
            ),
            ([Text("value"), object()], [Text("value"), object()]),
            ((Blob(b"value"), object()), (Blob(b"value"), object())),
        )

        actual = torch.utils.data.default_collate
        expected = reference_torch.utils.data.default_collate
        for actual_batch, expected_batch in batch_pairs:
            with self.subTest(
                container=type(actual_batch).__name__,
                leaf=type(actual_batch[0]).__name__,
            ):
                self.assertIs(actual(actual_batch), actual_batch)
                self.assertIs(expected(expected_batch), expected_batch)

    def test_empty_non_subscriptable_and_call_errors_match_pytorch_2_13(self):
        actual = torch.utils.data.default_collate
        expected = reference_torch.utils.data.default_collate

        for actual_call, expected_call in (
            (lambda: actual([]), lambda: expected([])),
            (lambda: actual(()), lambda: expected(())),
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(1), lambda: expected(1)),
            (lambda: actual(object()), lambda: expected(object())),
            (
                lambda: actual(iter(("value",))),
                lambda: expected(iter(("value",))),
            ),
            (lambda: actual(), lambda: expected()),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(value=None), lambda: expected(value=None)),
        ):
            self.assert_error_matches(actual_call, expected_call)

        actual_batch = ["value"]
        expected_batch = ["value"]
        self.assertIs(actual(batch=actual_batch), actual_batch)
        self.assertIs(expected(batch=expected_batch), expected_batch)

    def test_metadata_import_identity_copying_and_pickling_match(self):
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_collate = importlib.import_module("torch_rs.utils.data._utils.collate")
        expected_collate = importlib.import_module("torch.utils.data._utils.collate")
        actual = actual_data.default_collate
        expected = expected_data.default_collate

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertNotEqual(actual.__doc__, expected.__doc__)
        self.assertIn("string or bytes batch unchanged", actual.__doc__)
        self.assertIn("nested sequence collation", actual.__doc__)
        self.assertIn("additional outer dimension", expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

        self.assertIs(actual_data.default_collate, actual_collate.default_collate)
        self.assertIs(expected_data.default_collate, expected_collate.default_collate)
        self.assertFalse(hasattr(actual_collate, "__all__"))
        self.assertEqual(
            hasattr(actual_collate, "__all__"),
            hasattr(expected_collate, "__all__"),
        )

        supported = {
            "BatchSampler",
            "ChainDataset",
            "ConcatDataset",
            "DataChunk",
            "Dataset",
            "DistributedSampler",
            "IterableDataset",
            "Sampler",
            "SequentialSampler",
            "StackDataset",
            "Subset",
            "TensorDataset",
            "default_collate",
            "get_worker_info",
        }
        self.assertEqual(
            actual_data.__all__,
            [name for name in expected_data.__all__ if name in supported],
        )
        namespace = {}
        exec("from torch_rs.utils.data import *", namespace)
        self.assertIs(namespace["default_collate"], actual)

        self.assertFalse(hasattr(actual_data, "DataLoader"))
        for unsupported in (
            "collate",
            "collate_str_fn",
            "default_collate_fn_map",
            "default_convert",
        ):
            with self.subTest(unsupported=unsupported):
                self.assertFalse(hasattr(actual_collate, unsupported))

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_broader_collation_paths_are_explicitly_unsupported(self):
        actual = torch.utils.data.default_collate
        expected = reference_torch.utils.data.default_collate

        actual_values = (
            [torch.tensor([1.0]), torch.tensor([2.0])],
            [1, 2],
            [1.0, 2.0],
            [{"value": 1}, {"value": 2}],
            [["a"], ["b"]],
            [("a",), ("b",)],
        )
        expected_values = (
            [reference_torch.tensor([1.0]), reference_torch.tensor([2.0])],
            [1, 2],
            [1.0, 2.0],
            [{"value": 1}, {"value": 2}],
            [["a"], ["b"]],
            [("a",), ("b",)],
        )

        for actual_batch, expected_batch in zip(
            actual_values, expected_values, strict=True
        ):
            with self.subTest(type=type(actual_batch[0]).__name__):
                with self.assertRaises(TypeError) as raised:
                    actual(actual_batch)
                self.assertEqual(raised.exception.args, (UNSUPPORTED_ERROR,))
                self.assertIsNot(expected(expected_batch), expected_batch)

        for first in (MappingProbe(), SequenceProbe()):
            with self.subTest(type=type(first).__name__):
                with self.assertRaises(TypeError) as raised:
                    actual([first])
                self.assertEqual(raised.exception.args, (UNSUPPORTED_ERROR,))


if __name__ == "__main__":
    unittest.main()
