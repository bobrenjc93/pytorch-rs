import array
import ctypes
import pickle
import struct
import sys
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorBufferReferenceTests(unittest.TestCase):
    def assert_matches(self, source, *, case):
        with self.subTest(case=case):
            actual = torch.tensor(source, dtype=torch.float32, device="cpu")
            expected = reference_torch.tensor(
                source,
                dtype=reference_torch.float32,
                device="cpu",
            )
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_array_equal(np.asarray(actual), expected.numpy())

    def assert_error_matches(self, source, *, explicit_dtype=True):
        actual_kwargs = {"dtype": torch.float32} if explicit_dtype else {}
        expected_kwargs = {"dtype": reference_torch.float32} if explicit_dtype else {}
        with self.assertRaises(BaseException) as actual_raised:
            torch.tensor(source, **actual_kwargs)
        with self.assertRaises(BaseException) as expected_raised:
            reference_torch.tensor(source, **expected_kwargs)
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_numeric_buffers_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        pointer_high_bit = 1 << (8 * struct.calcsize("@P") - 1)
        pointer_bytes = struct.pack("@PP", 123, pointer_high_bit)
        cases = []
        for format_code, values in (
            ("b", [-128, 0, 127]),
            ("B", [0, 128, 255]),
            ("h", [-32768, 0, 32767]),
            ("H", [0, 32768, 65535]),
            ("i", [-1234567, 0, 1234567]),
            ("I", [0, 1234567, 4_000_000_000]),
            ("l", [-1234567, 0, 1234567]),
            ("L", [0, 1234567, 4_000_000_000]),
            ("q", [-(2**40), 0, 2**40]),
            ("Q", [0, 2**40, 2**63]),
            ("f", [-2.5, -0.0, 3.25]),
            ("d", [-2.5, -0.0, 3.25]),
        ):
            exporter = array.array(format_code, values)
            cases.append((f"array {format_code}", exporter))
            cases.append((f"memoryview {format_code}", memoryview(exporter)))

        cases.extend(
            (
                ("bytearray", bytearray((0, 1, 127, 128, 255))),
                ("bool canonical bytes", memoryview(b"\x00\x01").cast("?")),
                (
                    "ssize_t",
                    memoryview(bytes(2 * ctypes.sizeof(ctypes.c_ssize_t))).cast("n"),
                ),
                (
                    "size_t",
                    memoryview(bytes(2 * ctypes.sizeof(ctypes.c_size_t))).cast("N"),
                ),
                (
                    "native-prefixed int32",
                    memoryview(struct.pack("@ii", -7, 9)).cast("@i"),
                ),
                (
                    "native-prefixed float32",
                    memoryview(struct.pack("@ff", -2.5, 3.25)).cast("@f"),
                ),
                ("native-prefixed bool canonical", memoryview(b"\x00\x01").cast("@?")),
                ("pointer", memoryview(pointer_bytes).cast("P")),
                ("native-prefixed pointer", memoryview(pointer_bytes).cast("@P")),
            )
        )
        if sys.version_info >= (3, 14):
            cases.extend(
                (
                    (
                        "bool including noncanonical bytes",
                        memoryview(b"\x00\x01\x02\x03\xfe\xff").cast("?"),
                    ),
                    ("native-prefixed bool", memoryview(b"\x02\x03").cast("@?")),
                )
            )
        if sys.version_info >= (3, 12):
            cases.append(
                (
                    "float16",
                    memoryview(np.asarray([1.0, -2.0], dtype=np.float16)),
                )
            )
        for case, source in cases:
            self.assert_matches(source, case=case)

    def test_numpy_and_ctypes_sequence_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for case, source in (
            (
                "rank-2 numpy array",
                np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
            ),
            ("direct ctypes array", (ctypes.c_int * 3)(-7, 0, 9)),
        ):
            self.assert_matches(source, case=case)

    def test_float16_edge_value_bits_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        half_bits = np.asarray(
            (
                0x0000,
                0x8000,
                0x0001,
                0x03FF,
                0x0400,
                0x7BFF,
                0x7C00,
                0xFC00,
                0x7C01,
                0xFFFF,
            ),
            dtype=np.uint16,
        )
        source = memoryview(half_bits.view(np.float16))
        if sys.version_info < (3, 12):
            self.assert_error_matches(source, explicit_dtype=True)
            return
        actual = np.asarray(torch.tensor(source, dtype=torch.float32))
        expected = reference_torch.tensor(source, dtype=reference_torch.float32).numpy()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_strided_reversed_and_empty_buffers_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        exporter = array.array("i", [-8, -4, 0, 4, 8, 12])
        for case, source in (
            ("positive stride", memoryview(exporter)[1::2]),
            ("negative stride", memoryview(exporter)[::-2]),
            ("empty numeric", memoryview(array.array("d"))),
            ("empty unsupported format", memoryview(np.asarray([], dtype="U1"))),
            (
                "empty multidimensional with zero first dimension",
                memoryview(np.empty((0, 3), dtype=np.uint8)),
            ),
        ):
            self.assert_matches(source, case=case)

    def test_buffer_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        cases = (
            (memoryview(bytearray(range(6))).cast("B", (2, 3)), True),
            (memoryview(array.array("i", [3])).cast("B").cast("i", ()), True),
            (memoryview((ctypes.c_int * 2)(1, 2)), True),
        )
        for source, explicit_dtype in cases:
            with self.subTest(format=source.format, shape=source.shape):
                self.assert_error_matches(source, explicit_dtype=explicit_dtype)

    def test_bytes_and_character_views_match_dtype_sensitive_pytorch_dispatch(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for source in (b"", b"ab"):
            with self.subTest(source=source, explicit_dtype=False):
                self.assert_error_matches(source, explicit_dtype=False)
            with self.subTest(source=source, explicit_dtype=True):
                self.assert_error_matches(source, explicit_dtype=True)

        characters = memoryview(b"ab").cast("c")
        self.assert_error_matches(characters, explicit_dtype=False)
        self.assert_matches(characters, case="character memoryview with float32 dtype")

    def test_buffer_only_exporters_are_rejected_like_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        class BufferOnly:
            def __init__(self):
                self.data = bytearray((1, 2, 3))

            def __buffer__(self, flags):
                return memoryview(self.data)

        exporters = [pickle.PickleBuffer(bytearray((1, 2, 3)))]
        if sys.version_info >= (3, 12):
            exporters.append(BufferOnly())

        for exporter in exporters:
            for explicit_dtype in (False, True):
                with self.subTest(
                    type=type(exporter).__name__,
                    explicit_dtype=explicit_dtype,
                ):
                    self.assert_error_matches(
                        exporter,
                        explicit_dtype=explicit_dtype,
                    )

    def test_sequence_length_exceptions_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for error_type in (RuntimeError, MemoryError, KeyboardInterrupt):
            class RaisingSequence:
                def __len__(self):
                    raise error_type("length failed")

                def __getitem__(self, index):
                    raise AssertionError("getitem must not be called")

            with self.subTest(error=error_type.__name__):
                self.assert_error_matches(
                    RaisingSequence(),
                    explicit_dtype=True,
                )


if __name__ == "__main__":
    unittest.main()
