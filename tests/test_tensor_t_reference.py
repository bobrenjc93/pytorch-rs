import json
import subprocess
import sys
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorTransposePropertyReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case, operation):
        with self.subTest(case=case, operation=operation):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            if actual.numel() != 0:
                np.testing.assert_allclose(
                    np.asarray(actual),
                    expected.cpu().numpy(),
                    rtol=2.0e-6,
                    atol=1.0e-6,
                    equal_nan=True,
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

    def test_seeded_t_and_mt_views_and_consumers_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0x7A6D_213)
        shapes = [(), (0,), (3,), (2, 3), (2, 0, 3), (1, 3, 1, 2)]
        for _ in range(28):
            rank = int(rng.integers(0, 7))
            shapes.append(tuple(int(value) for value in rng.integers(0, 5, rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.uniform(-2.0, 2.0, elements).astype(np.float32).reshape(shape)
            if elements == 0:
                actual = torch.zeros(shape)
                expected = reference_torch.zeros(shape, dtype=reference_torch.float32)
            else:
                data = values.item() if shape == () else values.tolist()
                actual = torch.tensor(data)
                expected = reference_torch.tensor(values, dtype=reference_torch.float32)

            if len(shape) >= 2 and shape[0] > 0 and shape[-1] > 0 and case % 3 == 0:
                actual = actual.transpose(0, -1)[0]
                expected = expected.transpose(0, -1)[0]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                actual_t = actual.T
                expected_t = expected.T
            self.assert_matches(actual_t, expected_t, case=case, operation="T")
            self.assertIsNot(actual_t, actual)
            self.assertIsNot(expected_t, expected)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                actual_tt = actual_t.T
                expected_tt = expected_t.T
            self.assert_matches(actual_tt, expected_tt, case=case, operation="T.T")
            self.assertEqual(actual_tt.shape, actual.shape)
            self.assertEqual(actual_tt.stride(), actual.stride())

            rank = len(actual.shape)
            if rank == 1:
                self.assert_error_matches(
                    lambda: actual.mT,
                    lambda: expected.mT,
                )
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    actual_mt = actual.mT
                    expected_mt = expected.mT
                self.assert_matches(actual_mt, expected_mt, case=case, operation="mT")
                if rank == 0:
                    self.assertIs(actual_mt, actual)
                    self.assertIs(expected_mt, expected)
                else:
                    explicit_actual = actual.transpose(-2, -1)
                    explicit_expected = expected.transpose(-2, -1)
                    self.assert_matches(
                        explicit_actual,
                        explicit_expected,
                        case=case,
                        operation="explicit transpose",
                    )
                    self.assertEqual(actual_mt.shape, explicit_actual.shape)
                    self.assertEqual(actual_mt.stride(), explicit_actual.stride())
                    self.assertEqual(
                        actual_mt.storage_offset(), explicit_actual.storage_offset()
                    )
                    self.assert_matches(
                        actual_mt.mT,
                        expected_mt.mT,
                        case=case,
                        operation="mT.mT",
                    )

            for operation, actual_output, expected_output in (
                ("clone", actual_t.clone(), expected_t.clone()),
                ("contiguous", actual_t.contiguous(), expected_t.contiguous()),
                ("arithmetic", actual_t + 1.25, expected_t + 1.25),
                ("sum", actual_t.sum(), expected_t.sum()),
                ("reshape", actual_t.reshape(-1), expected_t.reshape(-1)),
                ("flatten", actual_t.flatten(), expected_t.flatten()),
                ("squeeze", actual_t.squeeze(), expected_t.squeeze()),
            ):
                self.assert_matches(
                    actual_output,
                    expected_output,
                    case=case,
                    operation=operation,
                )
            self.assertEqual(actual_t.tolist(), expected_t.tolist())

    def test_warning_categories_messages_once_behavior_and_stack_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        script = r'''
import importlib, json, warnings
torch = importlib.import_module(MODULE)
outputs = []
for shape, attribute in [((2, 3, 4), "T"), ((), "T"), ((), "mT")]:
    tensor = torch.zeros(shape)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(tensor, attribute)
        getattr(tensor, attribute)
    outputs.append({
        "count": len(caught),
        "category": caught[0].category.__name__,
        "message": str(caught[0].message),
        "filename": caught[0].filename,
        "lineno": caught[0].lineno,
    })
vector = torch.zeros((2,))
try:
    vector.mT
except Exception as error:
    outputs.append({"error": type(error).__name__, "message": str(error)})
for attribute in ("T", "mT"):
    try:
        setattr(vector, attribute, vector)
    except Exception as error:
        outputs.append({"error": type(error).__name__, "message": str(error)})
print(json.dumps(outputs))
'''

        def run(module):
            module_script = f"MODULE = {module!r}\n" + script
            result = subprocess.run(
                [sys.executable, "-c", module_script],
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(result.stdout)

        actual = run("torch_rs")
        expected = run("torch")
        for index in range(3):
            self.assertEqual(actual[index]["count"], expected[index]["count"])
            self.assertEqual(actual[index]["category"], expected[index]["category"])
            self.assertEqual(actual[index]["message"], expected[index]["message"])
            self.assertEqual(actual[index]["filename"], expected[index]["filename"])
            self.assertEqual(actual[index]["lineno"], expected[index]["lineno"])

        self.assertEqual(actual[3], expected[3])
        for index, attribute in ((4, "T"), (5, "mT")):
            self.assertEqual(actual[index]["error"], expected[index]["error"])
            self.assertIn(f"attribute '{attribute}'", actual[index]["message"])
            self.assertTrue(actual[index]["message"].endswith("objects is not writable"))
            self.assertTrue(expected[index]["message"].endswith("objects is not writable"))

    def test_empty_offset_memory_format_and_overflow_metadata_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        maximum = sys.maxsize
        actual = torch.zeros((maximum, 0, maximum))
        expected = reference_torch.zeros((maximum, 0, maximum))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assert_matches(actual.T, expected.T, case="extreme", operation="T")
        self.assert_error_matches(lambda: actual.mT, lambda: expected.mT)

        actual_offset = torch.zeros((maximum, 0, 1))[maximum - 1]
        expected_offset = reference_torch.zeros((maximum, 0, 1))[maximum - 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assert_matches(
                actual_offset.T,
                expected_offset.T,
                case="offset",
                operation="T",
            )
        self.assert_matches(
            actual_offset.mT,
            expected_offset.mT,
            case="offset",
            operation="mT",
        )

        actual_channels_last = torch.zeros((1, 1, 2, 2)).transpose(1, 3)
        expected_channels_last = reference_torch.zeros((1, 1, 2, 2)).transpose(1, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actual_views = (actual_channels_last.T, actual_channels_last.mT)
            expected_views = (expected_channels_last.T, expected_channels_last.mT)
        formats = (
            (torch.contiguous_format, reference_torch.contiguous_format),
            (torch.channels_last, reference_torch.channels_last),
            (torch.channels_last_3d, reference_torch.channels_last_3d),
        )
        for case, (actual_view, expected_view) in enumerate(
            zip(actual_views, expected_views, strict=True)
        ):
            self.assert_matches(
                actual_view,
                expected_view,
                case=case,
                operation="memory-format view",
            )
            for actual_format, expected_format in formats:
                self.assertEqual(
                    actual_view.is_contiguous(memory_format=actual_format),
                    expected_view.is_contiguous(memory_format=expected_format),
                )


if __name__ == "__main__":
    unittest.main()
