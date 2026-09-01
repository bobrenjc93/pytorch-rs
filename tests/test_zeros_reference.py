import sys
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class IntSubclass(int):
    pass


class IndexDimension:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ZerosReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("zeros differentials require pinned PyTorch 2.13.0")

    def tensor_observation(self, module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "values": tensor.tolist(),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is module.float32,
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "layout_identity": tensor.layout is module.strided,
            "is_pinned": tensor.is_pinned(),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
        }

    def capture_error(self, call):
        with self.assertRaises(Exception) as raised:
            call()
        return type(raised.exception), str(raised.exception)

    def test_scalar_results_and_metadata_match_pytorch_2_13(self):
        dimension_factories = (
            lambda: 2,
            lambda: 0,
            lambda: IntSubclass(2),
            lambda: np.int64(2),
            lambda: np.uint32(2),
            lambda: IndexDimension(2),
        )
        metadata_factories = (
            lambda module: {},
            lambda module: {"out": None},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"dtype": module.float},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"pin_memory": None},
            lambda module: {"pin_memory": False},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": "cpu:0"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"device": module.device("cpu", 2)},
            lambda module: {
                "out": None,
                "dtype": module.float32,
                "layout": module.strided,
                "device": module.device("cpu"),
                "pin_memory": False,
                "requires_grad": True,
            },
        )

        for dimension_factory in dimension_factories:
            for metadata_factory in metadata_factories:
                actual_dimension = dimension_factory()
                expected_dimension = dimension_factory()
                actual_keywords = metadata_factory(torch)
                expected_keywords = metadata_factory(reference_torch)
                with self.subTest(
                    dimension=actual_dimension,
                    keywords=actual_keywords,
                ):
                    actual = torch.zeros(actual_dimension, **actual_keywords)
                    expected = reference_torch.zeros(
                        expected_dimension, **expected_keywords
                    )
                    self.assertEqual(
                        self.tensor_observation(torch, actual),
                        self.tensor_observation(reference_torch, expected),
                    )

    def test_sequence_shapes_and_default_metadata_match_pytorch_2_13(self):
        size_factories = (
            ("scalar tuple", lambda module: module.zeros(())),
            ("scalar list", lambda module: module.zeros([])),
            ("tuple", lambda module: module.zeros((2, 3))),
            ("list", lambda module: module.zeros([2, 1])),
            ("Size", lambda module: module.zeros(module.Size([2, 0, 3]))),
            ("empty", lambda module: module.zeros((0,))),
            ("zero-size middle", lambda module: module.zeros((2, 0, 3))),
            (
                "size keyword Size",
                lambda module: module.zeros(size=module.Size([0, 2])),
            ),
        )
        for case, factory in size_factories:
            with self.subTest(case=case):
                actual = factory(torch)
                expected = factory(reference_torch)
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )

        option_factories = (
            lambda module: {},
            lambda module: {"out": None},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"dtype": module.float},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": "cpu:0"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"device": module.device("cpu", 2)},
            lambda module: {"pin_memory": None},
            lambda module: {"pin_memory": False},
            lambda module: {"requires_grad": None},
            lambda module: {"requires_grad": False},
            lambda module: {"requires_grad": True},
            lambda module: {
                "out": None,
                "dtype": module.float32,
                "layout": module.strided,
                "device": module.device("cpu"),
                "pin_memory": False,
                "requires_grad": True,
            },
        )
        for option_factory in option_factories:
            actual_options = option_factory(torch)
            expected_options = option_factory(reference_torch)
            with self.subTest(options=actual_options):
                with torch.no_grad():
                    actual = torch.zeros((2, 0, 3), **actual_options)
                with reference_torch.no_grad():
                    expected = reference_torch.zeros(
                        (2, 0, 3), **expected_options
                    )
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )

    def test_out_none_results_and_storage_freshness_match_pytorch_2_13(self):
        cases = (
            ("scalar", lambda module: module.zeros(2, out=None)),
            ("scalar tensor", lambda module: module.zeros((), out=None)),
            ("tuple", lambda module: module.zeros((2, 3), out=None)),
            ("list", lambda module: module.zeros([2, 1], out=None)),
            ("Size", lambda module: module.zeros(module.Size([2, 0, 3]), out=None)),
            ("size keyword", lambda module: module.zeros(size=(2,), out=None)),
            (
                "requires grad",
                lambda module: module.zeros((2,), out=None, requires_grad=True),
            ),
            ("empty", lambda module: module.zeros((0,), out=None)),
            ("zero-size middle", lambda module: module.zeros((2, 0, 3), out=None)),
            (
                "layout and pin defaults",
                lambda module: module.zeros(
                    (2,),
                    out=None,
                    layout=module.strided,
                    pin_memory=False,
                ),
            ),
            (
                "none layout and pin",
                lambda module: module.zeros(
                    (2,),
                    out=None,
                    layout=None,
                    pin_memory=None,
                ),
            ),
        )

        for case, factory in cases:
            with self.subTest(case=case):
                actual = factory(torch)
                actual_peer = factory(torch)
                expected = factory(reference_torch)
                expected_peer = factory(reference_torch)
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )
                self.assertEqual(
                    actual.is_set_to(actual_peer),
                    expected.is_set_to(expected_peer),
                )
                self.assertEqual(
                    actual.data_ptr() == actual_peer.data_ptr(),
                    expected.data_ptr() == expected_peer.data_ptr(),
                )

    def test_dimension_errors_match_pytorch_2_13(self):
        exact_cases = (
            -1,
            IndexDimension(-1),
            True,
            False,
            np.bool_(True),
            sys.maxsize,
            IndexDimension(sys.maxsize),
        )
        for dimension in exact_cases:
            with self.subTest(dimension=dimension):
                actual_type, actual_message = self.capture_error(
                    lambda dimension=dimension: torch.zeros(dimension)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimension=dimension: reference_torch.zeros(dimension)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(actual_message, expected_message)

        overflow_cases = (
            2**63,
            -(2**63) - 1,
            np.uint64(2**63),
            IndexDimension(2**63),
        )
        for dimension in overflow_cases:
            with self.subTest(dimension=dimension):
                actual_type, actual_message = self.capture_error(
                    lambda dimension=dimension: torch.zeros(dimension)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimension=dimension: reference_torch.zeros(dimension)
                )
                self.assertIs(actual_type, expected_type)
                marker = "failed to unpack the object at pos 1 with error"
                self.assertIn(marker, actual_message)
                self.assertIn(marker, expected_message)
                self.assertIn("Overflow when unpacking long long", actual_message)
                self.assertIn("Overflow when unpacking long long", expected_message)

    def test_mixed_invalid_scalar_validation_order_matches_pytorch_2_13(self):
        cases = (
            (
                "negative and invalid dtype",
                lambda module: module.zeros(-1, dtype=object()),
            ),
            (
                "overflow and invalid dtype",
                lambda module: module.zeros(2**63, dtype=object()),
            ),
            (
                "negative and invalid device",
                lambda module: module.zeros(-1, device=object()),
            ),
            (
                "overflow and invalid device",
                lambda module: module.zeros(2**63, device=object()),
            ),
            (
                "negative and invalid layout",
                lambda module: module.zeros(-1, layout=object()),
            ),
            (
                "negative and invalid pin",
                lambda module: module.zeros(-1, pin_memory=0),
            ),
            (
                "invalid pin before requires_grad",
                lambda module: module.zeros(1, pin_memory=0, requires_grad=0),
            ),
            (
                "negative and invalid requires_grad",
                lambda module: module.zeros(-1, requires_grad=1),
            ),
            (
                "index negative and invalid requires_grad",
                lambda module: module.zeros(IndexDimension(-1), requires_grad=1),
            ),
            (
                "overflow and invalid requires_grad",
                lambda module: module.zeros(2**63, requires_grad=1),
            ),
            (
                "negative and duplicate size",
                lambda module: module.zeros(-1, size=(2,)),
            ),
            (
                "overflow and duplicate size",
                lambda module: module.zeros(2**63, size=(2,)),
            ),
            (
                "negative and unknown keyword",
                lambda module: module.zeros(-1, unexpected=True),
            ),
            (
                "index overflow and unknown keyword",
                lambda module: module.zeros(IndexDimension(2**63), unexpected=True),
            ),
            (
                "unknown after accepted layout",
                lambda module: module.zeros(
                    1,
                    layout=module.strided,
                    unexpected=True,
                ),
            ),
            (
                "duplicate size after accepted layout",
                lambda module: module.zeros(
                    1,
                    size=(1,),
                    layout=module.strided,
                ),
            ),
            (
                "boolean type before requires_grad",
                lambda module: module.zeros(True, requires_grad=1),
            ),
        )
        for case, call in cases:
            with self.subTest(case=case):
                actual_type, actual_message = self.capture_error(lambda: call(torch))
                expected_type, expected_message = self.capture_error(
                    lambda: call(reference_torch)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(
                    actual_message.replace("torch.device or str", "torch.device"),
                    expected_message,
                )

    def test_out_type_error_order_matches_pytorch_2_13(self):
        cases = (
            ("missing size", lambda module: module.zeros(out=[])),
            ("negative size", lambda module: module.zeros(-1, out=[])),
            ("invalid dtype", lambda module: module.zeros(2, dtype=object(), out=[])),
            ("unknown keyword", lambda module: module.zeros(2, unexpected=True, out=[])),
            ("duplicate size", lambda module: module.zeros(2, size=(2,), out=[])),
            ("bool dimension", lambda module: module.zeros(True, out=[])),
        )
        for case, call in cases:
            with self.subTest(case=case):
                actual_type, actual_message = self.capture_error(lambda: call(torch))
                expected_type, expected_message = self.capture_error(
                    lambda: call(reference_torch)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(actual_message, expected_message)

    def test_unsupported_dtype_device_layout_pin_out_and_variadic_are_pinned(self):
        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(reference_torch, "float64"))
        with self.assertRaisesRegex(
            TypeError,
            r"^zeros\(\): argument 'dtype' must be torch\.dtype, not dtype$",
        ):
            torch.zeros((1,), dtype=reference_torch.float64, layout=torch.strided)
        self.assertIs(
            reference_torch.zeros((1,), dtype=reference_torch.float64).dtype,
            reference_torch.float64,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"^zeros\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.zeros((1,), device="meta", layout=torch.strided, pin_memory=False)
        meta = reference_torch.zeros((1,), device="meta")
        self.assertEqual(str(meta.device), "meta")
        self.assertIs(meta.dtype, reference_torch.float32)
        self.assertIs(meta.layout, reference_torch.strided)

        for layout in (object(), reference_torch.strided, reference_torch.sparse_coo):
            with self.subTest(layout=layout):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^zeros\(\): argument 'layout' must be torch\.layout, not ",
                ):
                    torch.zeros((1,), layout=layout)

        for pin_memory in (0, 1, "false", object()):
            with self.subTest(pin_memory=pin_memory):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^zeros\(\): argument 'pin_memory' must be bool, not ",
                ):
                    torch.zeros((1,), pin_memory=pin_memory)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^zeros\(\): pin_memory=True is not supported; only unpinned CPU "
            r"storage is implemented$",
        ):
            torch.zeros((1,), layout=torch.strided, pin_memory=True)

        out = torch.ones((1,))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^zeros\(\): the 'out' argument is not supported$",
        ):
            torch.zeros(
                (1,),
                out=out,
                layout=torch.strided,
                pin_memory=False,
            )
        self.assertEqual(out.tolist(), [1.0])

        with self.assertRaisesRegex(
            TypeError,
            r"^zeros\(\) takes 1 positional argument but 2 were given$",
        ):
            torch.zeros(2, 3)
        self.assertEqual(tuple(reference_torch.zeros(2, 3).shape), (2, 3))

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "requires a CUDA-capable PyTorch runtime",
    )
    def test_negative_size_is_validated_before_cuda_device_resolution(self):
        actual_type, actual_message = self.capture_error(
            lambda: torch.zeros(-1, device="cuda")
        )
        expected_type, expected_message = self.capture_error(
            lambda: reference_torch.zeros(-1, device="cuda")
        )
        self.assertIs(actual_type, expected_type)
        self.assertEqual(actual_message, expected_message)


if __name__ == "__main__":
    unittest.main()
