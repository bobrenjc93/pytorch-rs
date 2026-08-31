import copy
import importlib
import pickle
import re
import unittest

import numpy as np
import torch_rs as torch


class IntSubclass(int):
    pass


class IndexDimension:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


class EmptyTests(unittest.TestCase):
    def tensor_metadata(self, tensor):
        return (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.numel(),
            tensor.dtype,
            tensor.device,
            tensor.layout,
            tensor.is_pinned(),
            tensor.requires_grad,
            tensor.is_leaf,
        )

    def assert_empty_matches_zeros_metadata(self, actual, shape, *, requires_grad=False):
        expected = torch.zeros(shape, requires_grad=requires_grad)
        self.assertEqual(self.tensor_metadata(actual), self.tensor_metadata(expected))

    def test_supported_shapes_match_existing_factory_metadata(self):
        cases = (
            ("one positional dimension", lambda: torch.empty(2), (2,)),
            ("scalar tuple", lambda: torch.empty(()), ()),
            ("scalar list", lambda: torch.empty([]), ()),
            ("empty vector", lambda: torch.empty((0,)), (0,)),
            ("empty middle", lambda: torch.empty([2, 0, 3]), (2, 0, 3)),
            ("multidimensional", lambda: torch.empty((2, 3)), (2, 3)),
            ("size keyword", lambda: torch.empty(size=[2]), (2,)),
        )
        for case, create, shape in cases:
            with self.subTest(case=case):
                self.assert_empty_matches_zeros_metadata(create(), shape)

    def test_integer_protocol_positional_dimension_matches_singleton_size(self):
        custom = IndexDimension(2)
        dimensions = (IntSubclass(2), np.int64(2), np.uint32(2), custom)
        for dimension in dimensions:
            with self.subTest(dimension=dimension):
                self.assertEqual(
                    self.tensor_metadata(torch.empty(dimension)),
                    self.tensor_metadata(torch.empty((2,))),
                )
        self.assertGreater(custom.calls, 0)

    def test_supported_metadata_options(self):
        option_cases = (
            {},
            {"out": None},
            {"dtype": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"layout": None},
            {"layout": torch.strided},
            {"device": None},
            {"device": "cpu"},
            {"device": "cpu:0"},
            {"device": torch.device("cpu")},
            {"device": torch.device("cpu", 2)},
            {"pin_memory": None},
            {"pin_memory": False},
            {"requires_grad": None},
            {"requires_grad": False},
            {"requires_grad": True},
            {
                "out": None,
                "dtype": torch.float32,
                "layout": torch.strided,
                "device": torch.device("cpu"),
                "pin_memory": False,
                "requires_grad": True,
            },
        )
        for options in option_cases:
            with self.subTest(options=options):
                with torch.no_grad():
                    result = torch.empty((2, 3), **options)
                self.assert_empty_matches_zeros_metadata(
                    result,
                    (2, 3),
                    requires_grad=options.get("requires_grad") is True,
                )

    def test_out_none_and_repeated_calls_use_fresh_storage(self):
        cases = (
            ("scalar", lambda keywords: torch.empty((), **keywords)),
            ("empty", lambda keywords: torch.empty((0,), **keywords)),
            ("multidimensional", lambda keywords: torch.empty((2, 3), **keywords)),
            (
                "requires grad",
                lambda keywords: torch.empty((2,), requires_grad=True, **keywords),
            ),
        )
        for case, factory in cases:
            with self.subTest(case=case):
                baseline = factory({})
                with_out_none = factory({"out": None})
                peer = factory({})
                self.assertEqual(
                    self.tensor_metadata(with_out_none),
                    self.tensor_metadata(baseline),
                )
                self.assertFalse(with_out_none.is_set_to(baseline))
                self.assertFalse(with_out_none.is_set_to(peer))
                if with_out_none.numel() > 0:
                    self.assertNotEqual(with_out_none.data_ptr(), peer.data_ptr())

    def test_unsupported_boundaries_are_rejected(self):
        out = torch.zeros((1,))
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("empty(): the 'out' argument is not supported"),
        ):
            torch.empty((1,), out=out)
        self.assertEqual(out.tolist(), [0.0])

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                "empty(): pin_memory=True is not supported; only unpinned CPU storage is implemented"
            ),
        ):
            torch.empty((1,), pin_memory=True)

        for pin_memory in (0, 1, "false", object()):
            with self.subTest(pin_memory=pin_memory):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'pin_memory' must be bool, not ",
                ):
                    torch.empty((1,), pin_memory=pin_memory)

        for dtype in (object(),):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'dtype' must be torch\.dtype, not ",
                ):
                    torch.empty((1,), dtype=dtype)

        for layout in (object(), torch.layout):
            with self.subTest(layout=layout):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'layout' must be torch\.layout, not ",
                ):
                    torch.empty((1,), layout=layout)

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("empty(): device 'meta' is not supported; only 'cpu' is implemented"),
        ):
            torch.empty((1,), device="meta")

        with self.assertRaisesRegex(
            TypeError,
            re.escape("empty() takes 1 positional argument but 2 were given"),
        ):
            torch.empty(2, 3)

        with self.assertRaisesRegex(
            TypeError,
            re.escape("empty() got an unexpected keyword argument 'shape'"),
        ):
            torch.empty(2, shape=(2,))

        self.assertFalse(hasattr(torch, "empty_like"))
        self.assertNotIn("empty_like", torch.__all__)

    def test_callable_import_and_wildcard_exports(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.empty
        import_namespace = {}
        wildcard_namespace = {}
        exec("from torch_rs import empty as imported_empty", import_namespace)
        exec("from torch_rs import *", wildcard_namespace)

        self.assertEqual(function.__name__, "empty")
        self.assertEqual(
            function.__text_signature__,
            "(size, *, out=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False)",
        )
        self.assertEqual(package.__all__.count("empty"), 1)
        self.assertIs(import_namespace["imported_empty"], function)
        self.assertIs(wildcard_namespace["empty"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.empty, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.empty, function)
        self.assertEqual(package.__all__.count("empty"), 1)


if __name__ == "__main__":
    unittest.main()
