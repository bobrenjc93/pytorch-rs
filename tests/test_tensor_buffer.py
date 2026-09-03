import array
import ctypes
import gc
import pickle
import struct
import sys
import unittest

import numpy as np
import torch_rs as torch


class TensorBufferTests(unittest.TestCase):
    def assert_tensor(self, source, expected):
        tensor = torch.tensor(source, dtype=torch.float32, device=torch.device("cpu"))
        self.assertEqual(tensor.shape, (len(expected),))
        self.assertEqual(tensor.stride(), (1,))
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        np.testing.assert_array_equal(
            np.asarray(tensor),
            np.asarray(expected, dtype=np.float32),
        )
        return tensor

    def test_numeric_array_and_memoryview_formats_copy_as_float32(self):
        boolean_values = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        cases = (
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
        )
        for format_code, values in cases:
            exporter = array.array(format_code, values)
            with self.subTest(format=format_code, input="array"):
                self.assert_tensor(exporter, values)
            with self.subTest(format=format_code, input="memoryview"):
                self.assert_tensor(memoryview(exporter), values)

        for format_code, raw, expected in (
            ("?", b"\x00\x01\x02\x03\xfe\xff", boolean_values),
            ("n", bytes(2 * ctypes.sizeof(ctypes.c_ssize_t)), [0.0, 0.0]),
            ("N", bytes(2 * ctypes.sizeof(ctypes.c_size_t)), [0.0, 0.0]),
        ):
            with self.subTest(format=format_code, input="cast memoryview"):
                self.assert_tensor(memoryview(raw).cast(format_code), expected)
        float16 = memoryview(np.asarray([1.0, -2.0], dtype=np.float16))
        if sys.version_info < (3, 12):
            with self.assertRaisesRegex(ValueError, "could not determine the shape"):
                torch.tensor(float16, dtype=torch.float32)
        else:
            self.assert_tensor(float16, [1.0, -2.0])

    def test_native_prefixed_formats(self):
        pointer_high_bit = 1 << (8 * struct.calcsize("@P") - 1)
        pointer_bytes = struct.pack("@PP", 123, pointer_high_bit)
        boolean_values = [1.0, 1.0]
        for format_code, raw, expected in (
            ("@i", struct.pack("@ii", -7, 9), [-7.0, 9.0]),
            ("@f", struct.pack("@ff", -2.5, 3.25), [-2.5, 3.25]),
            ("@?", b"\x02\x03", boolean_values),
            ("P", pointer_bytes, [123.0, pointer_high_bit]),
            ("@P", pointer_bytes, [123.0, pointer_high_bit]),
        ):
            with self.subTest(format=format_code):
                self.assert_tensor(memoryview(raw).cast(format_code), expected)

    def test_bytes_reject_but_bytearray_strides_and_reversed_views_work(self):
        for source in (b"", bytes((0, 1, 127, 128, 255))):
            for kwargs in ({}, {"dtype": torch.float32}):
                with self.subTest(source=source, kwargs=kwargs):
                    with self.assertRaisesRegex(TypeError, "invalid data type 'bytes'"):
                        torch.tensor(source, **kwargs)

        self.assert_tensor(bytearray((0, 1, 127, 128, 255)), [0, 1, 127, 128, 255])

        exporter = array.array("i", [-8, -4, 0, 4, 8, 12])
        self.assert_tensor(memoryview(exporter)[1::2], [-4, 4, 12])
        self.assert_tensor(memoryview(exporter)[::-2], [12, 4, -4])

    def test_numpy_and_ctypes_sequences_keep_sequence_dispatch(self):
        matrix = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        tensor = torch.tensor(matrix, dtype=torch.float32)
        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.tolist(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        integers = (ctypes.c_int * 3)(-7, 0, 9)
        self.assert_tensor(integers, [-7.0, 0.0, 9.0])

    def test_empty_buffers_and_explicit_metadata(self):
        for source in (
            memoryview(b""),
            bytearray(),
            array.array("q"),
            memoryview(array.array("d")),
            memoryview(np.asarray([], dtype="U1")),
            memoryview(np.empty((0, 3), dtype=np.uint8)),
        ):
            with self.subTest(format=memoryview(source).format):
                self.assert_tensor(source, [])

    def test_tensor_owns_copy_after_exporter_mutation_and_collection(self):
        exporter = bytearray((1, 2, 3, 4))
        view = memoryview(exporter)[::-1]
        tensor = self.assert_tensor(view, [4, 3, 2, 1])

        view.release()
        exporter[:] = b"\x09\x09\x09\x09"
        del view
        del exporter
        gc.collect()

        self.assertEqual(tensor.tolist(), [4.0, 3.0, 2.0, 1.0])

    def test_multidimensional_zero_dimensional_and_unsupported_buffers_fail(self):
        multidimensional = memoryview(bytearray(range(6))).cast("B", (2, 3))
        with self.assertRaisesRegex(
            ValueError,
            "could not determine the shape of object type 'memoryview'",
        ):
            torch.tensor(multidimensional)

        scalar = memoryview(array.array("i", [3])).cast("B").cast("i", ())
        if sys.version_info < (3, 12):
            with self.assertRaisesRegex(ValueError, "could not determine the shape"):
                torch.tensor(scalar)
        else:
            with self.assertRaisesRegex(TypeError, "0-dim memory has no length"):
                torch.tensor(scalar)

        nonnative = memoryview((ctypes.c_int * 2)(1, 2))
        with self.assertRaisesRegex(
            ValueError,
            "could not determine the shape of object type 'memoryview'",
        ):
            torch.tensor(nonnative)

        characters = memoryview(b"ab").cast("c")
        with self.assertRaisesRegex(TypeError, "invalid data type 'bytes'"):
            torch.tensor(characters)
        converted_characters = torch.tensor(characters, dtype=torch.float32)
        self.assertEqual(converted_characters.shape, (2, 1))
        self.assertEqual(converted_characters.tolist(), [[97.0], [98.0]])

        released = memoryview(bytearray((1, 2)))
        released.release()
        with self.assertRaisesRegex(ValueError, "released memoryview"):
            torch.tensor(released)

    def test_buffer_only_exporters_are_not_tensor_sequences(self):
        class BufferOnly:
            def __init__(self):
                self.data = bytearray((1, 2, 3))

            def __buffer__(self, flags):
                return memoryview(self.data)

        exporters = [pickle.PickleBuffer(bytearray((1, 2, 3)))]
        if sys.version_info >= (3, 12):
            exporters.append(BufferOnly())

        for exporter in exporters:
            type_name = (
                "pickle.PickleBuffer"
                if isinstance(exporter, pickle.PickleBuffer)
                else "BufferOnly"
            )
            with self.subTest(type=type_name, explicit_dtype=False):
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"Could not infer dtype of {type_name}",
                ):
                    torch.tensor(exporter)
            with self.subTest(type=type_name, explicit_dtype=True):
                with self.assertRaisesRegex(
                    TypeError,
                    f"must be real number, not {type_name}",
                ):
                    torch.tensor(exporter, dtype=torch.float32)

    def test_sequence_length_exceptions_propagate(self):
        for error_type in (RuntimeError, MemoryError, KeyboardInterrupt):
            class RaisingSequence:
                def __len__(self):
                    raise error_type("length failed")

                def __getitem__(self, index):
                    raise AssertionError("getitem must not be called")

            with self.subTest(error=error_type.__name__):
                with self.assertRaisesRegex(error_type, "length failed"):
                    torch.tensor(RaisingSequence(), dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
