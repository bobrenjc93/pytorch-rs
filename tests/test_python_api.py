import array
import math
import operator
import re
import sys
import unittest
from decimal import Decimal

import numpy as np
import torch_rs as torch


class PythonApiBaselineTests(unittest.TestCase):
    def assert_tensor_values(self, actual, expected, shape):
        self.assertEqual(actual.shape, shape)
        actual_values = np.asarray(actual.tolist(), dtype=np.float32).reshape(-1)
        expected_values = np.asarray(expected, dtype=np.float32).reshape(-1)
        self.assertEqual(actual_values.size, expected_values.size)
        for actual_value, expected_value in zip(actual_values, expected_values):
            if np.isnan(expected_value):
                self.assertTrue(np.isnan(actual_value))
            else:
                actual_bits = actual_value.view(np.uint32).item()
                expected_bits = expected_value.view(np.uint32).item()
                self.assertEqual(actual_bits, expected_bits)

    def test_readme_style_tensor_expression(self):
        x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
        y = torch.ones([2, 2])
        result = (x + y).relu()

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [[0.0, 3.0], [4.0, 0.0]])

    def test_sin_matches_pytorch_float32_reference_and_metadata(self):
        source = torch.tensor(
            [
                [[0.25, -0.5], [1.0, -2.0]],
                [[np.pi, 1.0e10], [-1.0e10, np.finfo(np.float32).max]],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        # PyTorch 2.x CPU float32 reference values, checked with the
        # differential suite's transcendental tolerance.
        pytorch_reference = np.array(
            [
                0.24740396,
                -0.47942555,
                0.84147096,
                -0.9092974,
                -8.742278e-8,
                -0.48750603,
                0.48750603,
                -0.5218765,
            ],
            dtype=np.float32,
        ).reshape(2, 2, 2)

        actual = source.sin()
        self.assertEqual(actual.shape, source.shape)
        self.assertEqual(actual.stride(), source.stride())
        self.assertIs(actual.dtype, source.dtype)
        self.assertEqual(actual.device, source.device)
        np.testing.assert_allclose(
            np.asarray(actual),
            pytorch_reference,
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_sin_handles_scalar_empty_signed_zero_and_non_finite_values(self):
        scalar = torch.tensor(0.5).sin()
        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.stride(), ())
        self.assertAlmostEqual(scalar.item(), 0.47942555, delta=1.0e-6)

        empty = torch.zeros((2, 0, 3))
        empty_output = empty.sin()
        self.assertEqual(empty_output.shape, empty.shape)
        self.assertEqual(empty_output.stride(), empty.stride())
        self.assertEqual(empty_output.tolist(), [[], []])

        unusual_layout = torch.zeros((0, 1)) + 1.0
        self.assertEqual(unusual_layout.stride(), (1, 0))
        self.assertEqual(unusual_layout.sin().stride(), (1, 1))

        special = np.asarray(
            torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf")]).sin()
        )
        self.assertEqual(special[0].view(np.uint32).item(), np.float32(0.0).view(np.uint32).item())
        self.assertEqual(special[1].view(np.uint32).item(), np.float32(-0.0).view(np.uint32).item())
        self.assertTrue(np.isnan(special[2:]).all())

    def test_exp_matches_pytorch_float32_values_and_special_cases(self):
        source = torch.tensor(
            [
                -80.0,
                -20.0,
                -2.0,
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
                2.0,
                10.0,
                20.0,
                80.0,
            ]
        ).reshape(2, 2, 3)
        pytorch_reference = np.array(
            [
                1.8048513e-35,
                2.0611537e-9,
                0.13533528,
                0.36787945,
                0.60653067,
                1.0,
                1.6487212,
                2.7182817,
                7.389056,
                22026.465,
                4.851652e8,
                5.5406225e34,
            ],
            dtype=np.float32,
        ).reshape(2, 2, 3)

        actual = source.exp()
        self.assertEqual(actual.shape, source.shape)
        self.assertEqual(actual.stride(), (6, 3, 1))
        self.assertIs(actual.dtype, source.dtype)
        self.assertEqual(actual.device, source.device)
        np.testing.assert_allclose(
            np.asarray(actual),
            pytorch_reference,
            rtol=2.0e-6,
            atol=np.nextafter(np.float32(0), np.float32(1)),
        )

        smallest_subnormal = np.nextafter(np.float32(0), np.float32(1)).item()
        special = np.asarray(
            torch.tensor(
                [
                    0.0,
                    -0.0,
                    smallest_subnormal,
                    -smallest_subnormal,
                    -100.0,
                    -104.0,
                    88.0,
                    89.0,
                    float("nan"),
                    float("inf"),
                    -float("inf"),
                ]
            ).exp()
        )
        self.assertTrue(np.all(special[:4].view(np.uint32) == np.float32(1).view(np.uint32)))
        self.assertTrue(0 < special[4] < np.finfo(np.float32).tiny)
        self.assertEqual(special[5].view(np.uint32).item(), np.float32(0).view(np.uint32).item())
        self.assertTrue(np.isfinite(special[6]))
        self.assertTrue(np.isposinf(special[7]))
        self.assertTrue(np.isnan(special[8]))
        self.assertTrue(np.isposinf(special[9]))
        self.assertEqual(special[10].view(np.uint32).item(), np.float32(0).view(np.uint32).item())

    def test_exp_materializes_views_and_checks_extreme_empty_metadata(self):
        source = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        indexed = source[1]
        self.assertEqual(indexed.storage_offset(), 3)
        indexed_output = indexed.exp()
        self.assertEqual(indexed_output.shape, (3,))
        self.assertEqual(indexed_output.stride(), (1,))
        self.assertEqual(indexed_output.storage_offset(), 0)
        np.testing.assert_allclose(
            np.asarray(indexed_output),
            np.array([20.085537, 54.59815, 148.41316], dtype=np.float32),
            rtol=2.0e-6,
        )

        reshaped_output = source.reshape(1, 2, 3).exp()
        self.assertEqual(reshaped_output.shape, (1, 2, 3))
        self.assertEqual(reshaped_output.stride(), (6, 3, 1))

        empty = torch.zeros((2, 0, 3)).exp()
        self.assertEqual(empty.shape, (2, 0, 3))
        self.assertEqual(empty.stride(), (3, 3, 1))
        self.assertEqual(empty.tolist(), [[], []])

        offset_view = torch.zeros((sys.maxsize, 0))[sys.maxsize - 1]
        self.assertGreater(offset_view.storage_offset(), 0)
        offset_output = offset_view.exp()
        self.assertEqual(offset_output.shape, (0,))
        self.assertEqual(offset_output.stride(), (1,))
        self.assertEqual(offset_output.storage_offset(), 0)

        extreme = torch.zeros((0,)).reshape((0, sys.maxsize, 3))
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            extreme.exp()

    def test_float32_descriptor_identity_type_and_repr(self):
        self.assertIs(torch.float, torch.float32)
        self.assertIsInstance(torch.float32, torch.dtype)
        self.assertEqual(repr(torch.float32), "torch.float32")
        self.assertEqual(str(torch.float32), "torch.float32")
        self.assertEqual(hash(torch.float), hash(torch.float32))

        with self.assertRaises(TypeError):
            torch.dtype()

    def test_cpu_device_constructor_value_repr_and_equality(self):
        cpu = torch.device("cpu")
        copied = torch.device(cpu)

        self.assertIsInstance(cpu, torch.device)
        self.assertEqual(cpu, copied)
        self.assertEqual(hash(cpu), hash(copied))
        self.assertNotEqual(cpu, "cpu")
        self.assertEqual(cpu.type, "cpu")
        self.assertIsNone(cpu.index)
        self.assertEqual(str(cpu), "cpu")
        self.assertEqual(repr(cpu), "device(type='cpu')")
        self.assertEqual(torch.device(type="cpu"), cpu)

    def test_device_constructor_rejects_unsupported_values_and_types(self):
        for specification in ("cuda", "meta", "cpu:0", "CPU", ""):
            with self.subTest(specification=specification):
                with self.assertRaisesRegex(RuntimeError, "only 'cpu' is implemented"):
                    torch.device(specification)

        for specification in (object(), 0, b"cpu", torch.float32):
            with self.subTest(specification=specification):
                with self.assertRaises(TypeError):
                    torch.device(specification)

        with self.assertRaises(TypeError):
            torch.device()

    def test_creation_metadata_keywords_preserve_values_for_all_shapes(self):
        creators = (
            ("tensor scalar", lambda **kw: torch.tensor(-2.5, **kw), (), -2.5),
            ("zeros empty", lambda **kw: torch.zeros((2, 0, 3), **kw), (2, 0, 3), [[], []]),
            ("ones ordinary", lambda **kw: torch.ones((2, 2), **kw), (2, 2), [[1.0, 1.0], [1.0, 1.0]]),
            ("full ordinary", lambda **kw: torch.full((2,), 3.25, **kw), (2,), [3.25, 3.25]),
        )
        metadata = (
            (None, None),
            (torch.float32, None),
            (torch.float, "cpu"),
            (None, torch.device("cpu")),
            (torch.float32, torch.device("cpu")),
        )

        for name, create, shape, values in creators:
            for dtype, device in metadata:
                with self.subTest(name=name, dtype=dtype, device=device):
                    tensor = create(dtype=dtype, device=device)
                    self.assertEqual(tensor.shape, shape)
                    self.assertEqual(tensor.tolist(), values)
                    self.assertIs(tensor.dtype, torch.float32)
                    self.assertEqual(tensor.device, torch.device("cpu"))

        self.assertEqual(torch.zeros(size=(2,), dtype=torch.float32).tolist(), [0.0, 0.0])
        self.assertEqual(torch.ones(size=(2,), device="cpu").tolist(), [1.0, 1.0])

    def test_zeros_and_ones_accept_size_and_legacy_shape_keywords(self):
        class CustomSequence:
            def __init__(self, values):
                self.values = values

            def __len__(self):
                return len(self.values)

            def __getitem__(self, index):
                return self.values[index]

        class LengthlessSequence:
            def __getitem__(self, index):
                if index >= 2:
                    raise IndexError
                return 2

        class FailingLengthSequence(LengthlessSequence):
            def __len__(self):
                raise RuntimeError("length is only a capacity hint")

        for name, create, expected in (
            ("zeros", torch.zeros, [[0.0, 0.0], [0.0, 0.0]]),
            ("ones", torch.ones, [[1.0, 1.0], [1.0, 1.0]]),
        ):
            for keyword in ("size", "shape"):
                with self.subTest(function=name, keyword=keyword):
                    tensor = create(
                        **{keyword: (2, 2)},
                        dtype=torch.float32,
                        device=torch.device("cpu"),
                    )
                    self.assertEqual(tensor.tolist(), expected)
                    self.assertIs(tensor.dtype, torch.float32)
                    self.assertEqual(tensor.device, torch.device("cpu"))

            legacy_sequences = (
                ("array", array.array("q", (2, 2))),
                ("numpy", np.array((2, 2), dtype=np.int64)),
                ("memoryview", memoryview(array.array("q", (2, 2)))),
                ("custom", CustomSequence((2, 2))),
                ("lengthless", LengthlessSequence()),
                ("failing length", FailingLengthSequence()),
            )
            for sequence_name, sequence in legacy_sequences:
                with self.subTest(function=name, legacy_sequence=sequence_name):
                    self.assertEqual(create(shape=sequence).shape, (2, 2))
                    with self.assertRaises(TypeError):
                        create(size=sequence)

            for case, call in (
                ("None shape with positional size", lambda: create((2,), shape=None)),
                (
                    "None positional size with shape alias",
                    lambda: create(None, shape=(2,)),
                ),
                (
                    "None size with shape alias",
                    lambda: create(size=None, shape=(2,)),
                ),
                ("None shape with size keyword", lambda: create(size=(2,), shape=None)),
            ):
                with self.subTest(function=name, compatibility_case=case):
                    self.assertEqual(call().shape, (2,))

            with self.subTest(function=name, error="conflicting aliases"):
                with self.assertRaises(TypeError):
                    create(size=(1,), shape=(1,))

            with self.subTest(function=name, error="missing size"):
                with self.assertRaises(TypeError):
                    create()
                with self.assertRaises(TypeError):
                    create(size=None)
                with self.assertRaises(TypeError):
                    create(None)

    def test_zeros_and_ones_accept_variadic_and_integer_subclass_sizes(self):
        class IntSubclass(int):
            pass

        class SizeLike(tuple):
            pass

        class RaisesOverflow:
            def __index__(self):
                raise OverflowError("deliberate overflow")

        cases = (
            ("single integer", lambda create: create(4), (4,)),
            ("variadic", lambda create: create(2, 3), (2, 3)),
            ("zero dimension", lambda create: create(2, 0, 3), (2, 0, 3)),
            ("scalar tuple", lambda create: create(()), ()),
            ("scalar list", lambda create: create([]), ()),
            ("size keyword", lambda create: create(size=[2, 3]), (2, 3)),
            (
                "integer subclasses",
                lambda create: create(IntSubclass(2), IntSubclass(3)),
                (2, 3),
            ),
            ("tuple subclass", lambda create: create(SizeLike((2, 3))), (2, 3)),
            ("high rank", lambda create: create(*([1] * 32)), (1,) * 32),
        )
        for name, create, fill_value in (
            ("zeros", torch.zeros, 0.0),
            ("ones", torch.ones, 1.0),
        ):
            for case, call, shape in cases:
                with self.subTest(function=name, case=case):
                    tensor = call(create)
                    self.assertEqual(tensor.shape, shape)
                    self.assertTrue(np.all(np.asarray(tensor) == fill_value))

            for case, call, error in (
                ("missing", lambda: create(), TypeError),
                ("size keyword integer", lambda: create(size=2), TypeError),
                ("first bool", lambda: create(True), TypeError),
                ("tuple first bool", lambda: create((True, 2)), TypeError),
                ("negative", lambda: create(2, -1), RuntimeError),
                ("duplicate size", lambda: create(2, size=(2,)), TypeError),
                ("unknown keyword", lambda: create(2, unknown=True), TypeError),
            ):
                with self.subTest(function=name, invalid_case=case):
                    with self.assertRaises(error):
                        call()

            with self.subTest(function=name, invalid_case="first index overflow"):
                with self.assertRaisesRegex(TypeError, "not RaisesOverflow"):
                    create(RaisesOverflow())
            with self.subTest(function=name, invalid_case="later index overflow"):
                with self.assertRaisesRegex(TypeError, "got RaisesOverflow"):
                    create(2, RaisesOverflow())

            overflow_shape = [1, 2, sys.maxsize]
            overflow_message = (
                f"Storage size calculation overflowed with sizes={overflow_shape}"
            )
            with self.subTest(function=name, invalid_case="leading singleton overflow"):
                with self.assertRaisesRegex(RuntimeError, re.escape(overflow_message)):
                    create(*overflow_shape)

    def test_factory_binding_defers_later_index_conversion(self):
        def new_probe(events):
            class Probe:
                def __index__(self):
                    events.append("index")
                    return 3

            return Probe()

        for name, create in (("zeros", torch.zeros), ("ones", torch.ones)):
            invalid_calls = (
                ("variadic dtype", lambda probe: create(2, probe, dtype="bad")),
                ("tuple dtype", lambda probe: create((2, probe), dtype="bad")),
                ("size keyword dtype", lambda probe: create(size=(2, probe), dtype="bad")),
                ("variadic device", lambda probe: create(2, probe, device=object())),
                ("duplicate size", lambda probe: create(2, probe, size=(2, 3))),
                ("unknown keyword", lambda probe: create(2, probe, unknown=True)),
            )
            for case, call in invalid_calls:
                events = []
                with self.subTest(function=name, invalid_case=case):
                    with self.assertRaises(TypeError):
                        call(new_probe(events))
                    self.assertEqual(events, [])

            for case, call in (
                ("variadic", lambda probe: create(2, probe)),
                ("tuple", lambda probe: create((2, probe))),
                ("size keyword", lambda probe: create(size=(2, probe))),
            ):
                events = []
                with self.subTest(function=name, successful_case=case):
                    tensor = call(new_probe(events))
                    self.assertEqual(tensor.shape, (2, 3))
                    self.assertEqual(events, ["index"])

            for case, keyword in (("list", False), ("size keyword list", True)):
                dimensions = []

                class MutatingProbe:
                    def __index__(self):
                        dimensions[2] = 4
                        return 2

                dimensions.extend((2, MutatingProbe(), 3))
                with self.subTest(function=name, mutation_case=case):
                    tensor = (
                        create(size=dimensions) if keyword else create(dimensions)
                    )
                    self.assertEqual(tensor.shape, (2, 2, 4))

            for mutation, expected_shape in (("pop", (2,)), ("append", (2, 3, 4))):
                for keyword in (False, True):
                    dimensions = []
                    events = []

                    class ResizingProbe:
                        def __index__(self):
                            events.append("index")
                            if len(events) == 1:
                                if mutation == "pop":
                                    dimensions.pop()
                                else:
                                    dimensions.append(4)
                            return 2

                    dimensions.extend((ResizingProbe(), 3))
                    with self.subTest(
                        function=name,
                        list_resize=mutation,
                        keyword=keyword,
                    ):
                        tensor = (
                            create(size=dimensions) if keyword else create(dimensions)
                        )
                        self.assertEqual(tensor.shape, expected_shape)

            for keyword in (False, True):
                dimensions = []

                class LaterShrinkingProbe:
                    def __index__(self):
                        dimensions.pop()
                        return 3

                dimensions.extend((2, LaterShrinkingProbe(), 7))
                with self.subTest(
                    function=name,
                    later_list_shrink=True,
                    keyword=keyword,
                ):
                    tensor = (
                        create(size=dimensions) if keyword else create(dimensions)
                    )
                    self.assertEqual(tensor.shape, (2, 3, 7))

            stateful_forms = (
                ("direct first", lambda probe: create(probe), 3),
                ("variadic first", lambda probe: create(probe, 3), 3),
                ("tuple first", lambda probe: create((probe, 3)), 2),
                ("list first", lambda probe: create([probe, 3]), 2),
                ("size tuple first", lambda probe: create(size=(probe, 3)), 2),
                ("size list first", lambda probe: create(size=[probe, 3]), 2),
            )
            for case, call, expected_calls in stateful_forms:
                events = []

                class StatefulProbe:
                    def __index__(self):
                        events.append("index")
                        return 1 if len(events) == 1 else -1

                with self.subTest(function=name, stateful_case=case):
                    with self.assertRaises(RuntimeError):
                        call(StatefulProbe())
                    self.assertEqual(events, ["index"] * expected_calls)

    def test_factory_size_index_protocol_ignores_operator_monkeypatch(self):
        import _operator
        import operator

        class Indexable:
            def __index__(self):
                return 3

        original_index = operator.index
        original_native_index = _operator.index
        try:
            operator.index = lambda _: 7
            _operator.index = lambda _: 8
            for create in (torch.zeros, torch.ones):
                self.assertEqual(create(Indexable()).shape, (3,))
                with self.assertRaises(TypeError):
                    create(object())

            def broken_index(_):
                raise RuntimeError("monkeypatched")

            operator.index = broken_index
            for create in (torch.zeros, torch.ones):
                self.assertEqual(create(Indexable()).shape, (3,))
        finally:
            operator.index = original_index
            _operator.index = original_native_index

    def test_factory_size_index_protocol_ignores_preimport_monkeypatch(self):
        import subprocess

        script = """
import _operator
_operator.index = lambda _: 99
import torch_rs

class Indexable:
    def __index__(self):
        return 3

assert torch_rs.zeros(Indexable()).shape == (3,)
assert torch_rs.ones(Indexable()).shape == (3,)
for factory in (torch_rs.zeros, torch_rs.ones):
    try:
        factory(object())
    except TypeError:
        pass
    else:
        raise AssertionError("object without __index__ was accepted")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_factory_index_protocol_preserves_strict_int_warnings(self):
        import warnings

        class IntSubclass(int):
            pass

        class ReturnsBool:
            def __index__(self):
                return True

        class ReturnsSubclass:
            def __index__(self):
                return IntSubclass(2)

        message_suffix = (
            "The ability to return an instance of a strict subclass of int is "
            "deprecated, and may be removed in a future version of Python."
        )
        for name, create in (("zeros", torch.zeros), ("ones", torch.ones)):
            for case, value, expected_shape in (
                ("bool", ReturnsBool(), (1,)),
                ("int subclass", ReturnsSubclass(), (2,)),
            ):
                with self.subTest(function=name, warning_case=case):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", DeprecationWarning)
                        self.assertEqual(create(value).shape, expected_shape)
                    self.assertEqual(len(caught), 3)
                    for warning in caught:
                        self.assertIs(warning.category, DeprecationWarning)
                        self.assertIn(message_suffix, str(warning.message))

                with self.subTest(function=name, warning_error=case):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", DeprecationWarning)
                        with self.assertRaises(TypeError):
                            create(value)

                with self.subTest(function=name, later_warning_error=case):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", DeprecationWarning)
                        with self.assertRaises(TypeError):
                            create(2, value)

    def test_factory_tensor_detection_ignores_spoofed_metadata(self):
        class BoolDType:
            def __str__(self):
                return "torch.bool"

        class Tensor:
            __module__ = "torch"
            dtype = BoolDType()

            def __index__(self):
                return 2

        for name, create in (("zeros", torch.zeros), ("ones", torch.ones)):
            dimension = Tensor()
            calls = (
                ("direct", lambda: create(dimension), (2,)),
                ("later", lambda: create(3, dimension), (3, 2)),
                ("tuple", lambda: create((dimension, 3)), (2, 3)),
                ("list", lambda: create([dimension, 3]), (2, 3)),
                ("size tuple", lambda: create(size=(dimension, 3)), (2, 3)),
                ("size list", lambda: create(size=[dimension, 3]), (2, 3)),
            )
            for form, call, expected_shape in calls:
                with self.subTest(function=name, form=form):
                    self.assertEqual(call().shape, expected_shape)

    def test_factory_size_diagnostics_qualify_extension_heap_types(self):
        for name, create in (("zeros", torch.zeros), ("ones", torch.ones)):
            with self.subTest(function=name, position="first"):
                with self.assertRaisesRegex(TypeError, "not decimal.Decimal"):
                    create(Decimal(2))
            with self.subTest(function=name, position="later"):
                with self.assertRaisesRegex(TypeError, "got decimal.Decimal"):
                    create(2, Decimal(3))

    def test_factory_size_index_results_use_bounded_integer_conversion(self):
        huge_integer = 1 << 8_000_000

        class HugeIndex:
            def __init__(self):
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return huge_integer

        class HostileLookup:
            def __getattribute__(self, name):
                if name == "__index__":
                    raise RuntimeError("instance lookup must not run")
                return super().__getattribute__(name)

            def __index__(self):
                return 3

        class InstanceOnly:
            pass

        class ShadowedIndex:
            def __index__(self):
                return 3

        for name, create in (("zeros", torch.zeros), ("ones", torch.ones)):
            for case, call, expected_calls in (
                ("first", lambda probe: create(probe), 3),
                ("later", lambda probe: create(2, probe), 1),
                ("tuple", lambda probe: create((probe, 2)), 2),
            ):
                probe = HugeIndex()
                with self.subTest(function=name, huge_index_case=case):
                    with self.assertRaisesRegex(
                        TypeError, "Overflow when unpacking long long"
                    ):
                        call(probe)
                    self.assertEqual(probe.calls, expected_calls)

            with self.subTest(function=name, special_lookup=True):
                self.assertEqual(create(HostileLookup()).shape, (3,))

            instance_only = InstanceOnly()
            instance_only.__index__ = lambda: 4
            with self.subTest(function=name, instance_only=True):
                with self.assertRaises(TypeError):
                    create(instance_only)

            shadowed = ShadowedIndex()
            shadowed.__index__ = lambda: 4
            with self.subTest(function=name, shadowed_index=True):
                self.assertEqual(create(shadowed).shape, (3,))

    def test_factory_size_conversion_ignores_python_visible_type_metadata(self):
        class NonStringModule:
            __module__ = 123

            def __init__(self):
                self.events = []

            def __index__(self):
                self.events.append("index")
                return 2

        class HostileMeta(type):
            def __getattribute__(cls, name):
                if name in {"__module__", "__flags__", "__name__"}:
                    raise RuntimeError(f"blocked {name}")
                return super().__getattribute__(name)

        class HostileIndex(metaclass=HostileMeta):
            def __init__(self):
                self.events = []

            def __index__(self):
                self.events.append("index")
                return 2

        class HostileList(list, metaclass=HostileMeta):
            pass

        class HostileTuple(tuple, metaclass=HostileMeta):
            pass

        class HostileInvalid(metaclass=HostileMeta):
            pass

        for name, create in (("zeros", torch.zeros), ("ones", torch.ones)):
            for case, probe in (
                ("non-string module", NonStringModule()),
                ("hostile metaclass", HostileIndex()),
            ):
                with self.subTest(function=name, index_case=case):
                    self.assertEqual(create(probe).shape, (2,))
                    self.assertEqual(probe.events, ["index"] * 3)

            for case, dimensions in (
                ("keyword list subclass", HostileList([2, 3])),
                ("keyword tuple subclass", HostileTuple((2, 3))),
            ):
                with self.subTest(function=name, container_case=case):
                    self.assertEqual(create(size=dimensions).shape, (2, 3))

            with self.subTest(function=name, invalid_type_name=True):
                with self.assertRaisesRegex(TypeError, "not HostileInvalid"):
                    create(HostileInvalid())

    def test_ones_preserves_the_reserved_symint_range(self):
        reserved = -(1 << 62) - 1
        calls = (
            ("scalar", lambda: torch.ones(reserved)),
            ("tuple", lambda: torch.ones((reserved,))),
            ("list", lambda: torch.ones([reserved])),
            ("size tuple", lambda: torch.ones(size=(reserved,))),
            ("size list", lambda: torch.ones(size=[reserved])),
            ("mixed variadic", lambda: torch.ones(-1, reserved)),
        )
        for case, call in calls:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "SymIntArrayRef expected to contain only concrete integers",
                ):
                    call()

        with self.assertRaisesRegex(
            RuntimeError, "zeros: Dimension size must be non-negative"
        ):
            torch.zeros(reserved)

        boundary = -(1 << 62)
        with self.assertRaisesRegex(RuntimeError, "negative dimension"):
            torch.ones(boundary)
        with self.assertRaisesRegex(RuntimeError, "negative dimension"):
            torch.ones(boundary + 1)

    def test_factory_metadata_and_binding_error_precedence(self):
        for name, create in (("zeros", torch.zeros), ("ones", torch.ones)):
            for case, kwargs, message in (
                (
                    "dtype before unknown",
                    {"bogus": True, "dtype": "bad"},
                    "argument 'dtype' must be torch.dtype",
                ),
                (
                    "dtype before duplicate",
                    {"size": (2,), "dtype": "bad"},
                    "argument 'dtype' must be torch.dtype",
                ),
                (
                    "device before unknown",
                    {"bogus": True, "device": object()},
                    "argument 'device' must be torch.device",
                ),
                (
                    "duplicate before unknown",
                    {"size": (2,), "bogus": True},
                    "multiple values for argument 'size'",
                ),
                (
                    "unknown before duplicate",
                    {"bogus": True, "size": (2,)},
                    "unexpected keyword argument 'bogus'",
                ),
            ):
                with self.subTest(function=name, precedence_case=case):
                    with self.assertRaisesRegex(TypeError, re.escape(message)):
                        create(2, 3, **kwargs)

    def test_numpy_timedelta_size_errors_are_deferred_until_after_binding(self):
        value = np.timedelta64(3, "D")
        for name, create in (("zeros", torch.zeros), ("ones", torch.ones)):
            cases = (
                (
                    "plain",
                    lambda: create(value),
                    "failed to unpack the object at pos 1",
                ),
                (
                    "dtype",
                    lambda: create(value, dtype="bad"),
                    "argument 'dtype' must be torch.dtype",
                ),
                (
                    "device",
                    lambda: create(value, device=object()),
                    "argument 'device' must be torch.device",
                ),
                (
                    "duplicate",
                    lambda: create(value, size=(2,)),
                    "multiple values for argument 'size'",
                ),
                (
                    "unknown",
                    lambda: create(value, bogus=True),
                    "unexpected keyword argument 'bogus'",
                ),
            )
            for case, call, message in cases:
                with self.subTest(function=name, precedence_case=case):
                    with self.assertRaisesRegex(TypeError, re.escape(message)):
                        call()

    def test_eye_creates_square_rectangular_and_empty_tensors(self):
        cases = (
            (lambda: torch.eye(3), (3, 3), (3, 1), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            (lambda: torch.eye(2, 4), (2, 4), (4, 1), [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            (lambda: torch.eye(4, m=2), (4, 2), (2, 1), [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
            (lambda: torch.eye(n=0), (0, 0), (1, 1), []),
            (lambda: torch.eye(3, 0), (3, 0), (1, 1), [[], [], []]),
            (lambda: torch.eye(0, 3), (0, 3), (3, 1), []),
        )
        for create, shape, stride, values in cases:
            with self.subTest(shape=shape):
                tensor = create()
                self.assertEqual(tensor.shape, shape)
                self.assertEqual(tensor.stride(), stride)
                self.assertEqual(tensor.tolist(), values)
                self.assertIs(tensor.dtype, torch.float32)
                self.assertEqual(tensor.device, torch.device("cpu"))

        for create in (lambda: torch.eye(2, None), lambda: torch.eye(2, m=None)):
            with self.subTest(explicit_none=create):
                with self.assertRaises(TypeError):
                    create()

    def test_eye_accepts_float32_cpu_metadata_and_rejects_unsupported_options(self):
        tensor = torch.eye(
            2,
            3,
            dtype=torch.float,
            device=torch.device("cpu"),
        )
        self.assertEqual(tensor.tolist(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))

        for dtype in ("float32", np.dtype("float32"), np.float32, float, object()):
            with self.subTest(argument="dtype", value=dtype):
                with self.assertRaises(TypeError):
                    torch.eye(1, dtype=dtype)
        for device in ("cuda", "meta", "mps", "cpu:0"):
            with self.subTest(argument="device", value=device):
                with self.assertRaises(RuntimeError):
                    torch.eye(1, device=device)
        for keyword in ("out", "layout", "requires_grad", "pin_memory"):
            with self.subTest(keyword=keyword):
                with self.assertRaises(TypeError):
                    torch.eye(1, **{keyword: None})

        with self.assertRaises(TypeError):
            torch.eye(1, 1, torch.float32)

    def test_eye_uses_the_integer_index_protocol_and_rejects_bools(self):
        class IntSubclass(int):
            pass

        class IndexDimension:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        class IntOnly:
            def __int__(self):
                return 2

        rows = IndexDimension(2)
        columns = IndexDimension(3)
        tensor = torch.eye(rows, columns)
        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.tolist(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual((rows.calls, columns.calls), (1, 1))

        self.assertEqual(torch.eye(IntSubclass(2)).shape, (2, 2))
        self.assertEqual(torch.eye(np.int64(2), np.uint32(3)).shape, (2, 3))

        for dimensions in ((True,), (2, False), (np.bool_(True),), (IntOnly(),), (2.0,)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(TypeError):
                    torch.eye(*dimensions)

    def test_eye_normalizes_integer_conversion_failures(self):
        class FailingIndex:
            def __init__(self):
                self.calls = 0

            def __index__(self):
                self.calls += 1
                raise RuntimeError("index conversion failed")

        for dimensions in (
            (2**63,),
            (-(2**63) - 1,),
            (1, 2**63),
            (1, -(2**63) - 1),
            (np.uint64(2**63),),
        ):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(ValueError, "Overflow when unpacking long long"):
                    torch.eye(*dimensions)

        for position in ("n", "m"):
            dimension = FailingIndex()
            arguments = (dimension,) if position == "n" else (1, dimension)
            with self.subTest(position=position):
                with self.assertRaises(TypeError):
                    torch.eye(*arguments)
                self.assertEqual(dimension.calls, 1)

    def test_eye_rejects_invalid_metadata_before_dimension_conversion(self):
        invalid_metadata = (
            {"dtype": object()},
            {"device": object()},
        )
        for dimensions in ((2**63,), (1, 2**63)):
            for metadata in invalid_metadata:
                with self.subTest(dimensions=dimensions, metadata=metadata):
                    with self.assertRaises(TypeError):
                        torch.eye(*dimensions, **metadata)

    def test_eye_reports_negative_dimensions_and_checked_overflow(self):
        with self.assertRaisesRegex(RuntimeError, "n must be greater or equal to 0, got -1"):
            torch.eye(-1)
        with self.assertRaisesRegex(RuntimeError, "m must be greater or equal to 0, got -2"):
            torch.eye(1, -2)

        with self.assertRaisesRegex(RuntimeError, "Storage size calculation overflowed"):
            torch.eye(sys.maxsize, 3)

        oversized = sys.maxsize // 4 + 1
        with self.assertRaisesRegex(RuntimeError, "exceeds the platform capacity"):
            torch.eye(oversized, 1)

        no_rows = torch.eye(0, sys.maxsize)
        self.assertEqual(no_rows.shape, (0, sys.maxsize))
        self.assertEqual(no_rows.stride(), (sys.maxsize, 1))
        self.assertEqual(no_rows.numel(), 0)

        no_columns = torch.eye(sys.maxsize, 0)
        self.assertEqual(no_columns.shape, (sys.maxsize, 0))
        self.assertEqual(no_columns.stride(), (1, 1))
        self.assertEqual(no_columns.numel(), 0)

    def test_metadata_survives_views_and_native_kernels(self):
        source = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], dtype=torch.float32, device="cpu")
        outputs = (
            source.reshape(4),
            source + torch.ones((2, 2)),
            source * 2.0,
            source.relu(),
            source @ torch.ones((2, 2)),
            source.sum(),
        )
        for output in outputs:
            with self.subTest(shape=output.shape):
                self.assertIs(output.dtype, torch.float32)
                self.assertEqual(output.device, torch.device("cpu"))

    def test_creation_rejects_invalid_dtype_and_device_types(self):
        creators = (
            lambda **kw: torch.tensor([1.0], **kw),
            lambda **kw: torch.zeros((1,), **kw),
            lambda **kw: torch.ones((1,), **kw),
            lambda **kw: torch.full((1,), 2.0, **kw),
        )
        invalid_dtypes = ("float32", np.dtype("float32"), np.float32, float, object(), torch.device("cpu"))
        invalid_devices = (object(), 0, b"cpu", torch.float32)

        for create in creators:
            for dtype in invalid_dtypes:
                with self.subTest(argument="dtype", value=dtype):
                    with self.assertRaises(TypeError):
                        create(dtype=dtype)
            for device in invalid_devices:
                with self.subTest(argument="device", value=device):
                    with self.assertRaises(TypeError):
                        create(device=device)

    def test_creation_rejects_every_unimplemented_device(self):
        creators = (
            lambda **kw: torch.tensor(1.0, **kw),
            lambda **kw: torch.zeros((), **kw),
            lambda **kw: torch.ones((), **kw),
            lambda **kw: torch.full((), 2.0, **kw),
        )
        for create in creators:
            for device in ("cuda", "meta", "mps", "cpu:0"):
                with self.subTest(device=device):
                    with self.assertRaises(RuntimeError):
                        create(device=device)

    def test_creation_metadata_parameters_are_keyword_only(self):
        with self.assertRaises(TypeError):
            torch.tensor([1.0], torch.float32)
        with self.assertRaises(TypeError):
            torch.zeros((1,), torch.float32)
        with self.assertRaises(TypeError):
            torch.ones((1,), torch.float32)
        with self.assertRaises(TypeError):
            torch.full((1,), 2.0, torch.float32)

    def test_scalar_reduction_and_item(self):
        value = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).sum()
        self.assertEqual(value.shape, ())
        self.assertEqual(value.item(), 10.0)

    def test_stride_reports_contiguous_row_major_layout(self):
        cases = (
            (torch.tensor(1.0), ()),
            (torch.zeros((2, 3, 4)), (12, 4, 1)),
            (torch.zeros((2, 0, 3)), (3, 3, 1)),
            (torch.zeros((1, 0, 1)), (1, 1, 1)),
        )
        for tensor, expected in cases:
            with self.subTest(shape=tensor.shape):
                self.assertEqual(tensor.stride(), expected)

    def test_stride_accepts_positive_and_negative_dimensions(self):
        class IntSubclass(int):
            pass

        class IndexLike:
            def __index__(self):
                return 0

        tensor = torch.zeros((2, 3, 4))
        self.assertEqual(tensor.stride(0), 12)
        self.assertEqual(tensor.stride(1), 4)
        self.assertEqual(tensor.stride(-1), 1)
        self.assertEqual(tensor.stride(dim=-3), 12)
        self.assertEqual(tensor.stride(IntSubclass(1)), 4)
        self.assertEqual(tensor.stride(np.int64(-1)), 1)
        self.assertEqual(tensor.stride(np.uint64(1)), 4)

        for dimension in (3, -4):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    IndexError,
                    r"Dimension out of range \(expected to be in range of \[-3, 2\]",
                ):
                    tensor.stride(dimension)

        scalar = torch.tensor(1.0)
        for dimension in (0, -1):
            with self.subTest(scalar_dimension=dimension):
                with self.assertRaisesRegex(IndexError, "tensor has no dimensions"):
                    scalar.stride(dimension)

        for dimension in (True, np.bool_(False), IndexLike()):
            with self.subTest(invalid_type=type(dimension).__name__):
                with self.assertRaises(TypeError):
                    tensor.stride(dimension)

        for dimension in (1 << 100, -(1 << 100), np.uint64(2**64 - 1)):
            with self.subTest(overflow=dimension):
                with self.assertRaisesRegex(ValueError, "Overflow when unpacking long long"):
                    tensor.stride(dimension)

    def test_integer_indexing_returns_shared_storage_views(self):
        source = torch.tensor(
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
                [
                    [12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0],
                ],
            ]
        )

        row = source[-1]
        self.assertEqual(row.shape, (3, 4))
        self.assertEqual(row.stride(), (4, 1))
        self.assertEqual(row.storage_offset(), 12)
        self.assertEqual(row.tolist(), source.tolist()[1])
        self.assertIs(row.dtype, source.dtype)
        self.assertEqual(row.device, source.device)

        partial = source[-1, 1]
        self.assertEqual(partial.shape, (4,))
        self.assertEqual(partial.stride(), (1,))
        self.assertEqual(partial.storage_offset(), 16)
        self.assertEqual(partial.tolist(), [16.0, 17.0, 18.0, 19.0])

        scalar = source[1, -1, -2]
        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.stride(), ())
        self.assertEqual(scalar.storage_offset(), 22)
        self.assertEqual(scalar.item(), 22.0)

        tuple_partial = source[(1,)]
        alias = source[()]
        self.assertEqual(tuple_partial.tolist(), row.tolist())
        self.assertEqual(alias.shape, source.shape)
        self.assertEqual(alias.stride(), source.stride())
        self.assertEqual(alias.storage_offset(), source.storage_offset())
        self.assertEqual(alias.tolist(), source.tolist())

        del source
        copied = np.asarray(row)
        copied[0, 0] = 99.0
        self.assertEqual(row.tolist()[0][0], 12.0)
        self.assertEqual((row + 1).tolist()[0][0], 13.0)

    def test_clone_matches_values_metadata_and_canonical_layouts(self):
        source = torch.tensor(
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        chained_view = source[1].reshape(3, 2)[1].reshape(1, 2)
        cases = (
            (source, source.tolist(), (2, 2, 3), (6, 3, 1)),
            (source[1], source.tolist()[1], (2, 3), (3, 1)),
            (chained_view, [[8.0, 9.0]], (1, 2), (2, 1)),
            (torch.tensor(-0.0), -0.0, (), ()),
            (torch.zeros((2, 0, 3)), [[], []], (2, 0, 3), (3, 3, 1)),
            (torch.zeros((0, 1)) + 1, [], (0, 1), (1, 0)),
        )

        for original, expected, shape, stride in cases:
            for operation in (lambda value: value.clone(), torch.clone):
                with self.subTest(shape=shape, operation=operation):
                    copied = operation(original)
                    self.assert_tensor_values(copied, expected, shape)
                    self.assertEqual(copied.stride(), stride)
                    self.assertEqual(copied.storage_offset(), 0)
                    self.assertIs(copied.dtype, original.dtype)
                    self.assertEqual(copied.device, original.device)

        self.assert_tensor_values(
            torch.clone(input=chained_view), [[8.0, 9.0]], (1, 2)
        )

    def test_clone_preserves_nan_infinity_and_signed_zero_bits(self):
        expected_bits = np.array(
            [0x7FC12345, 0x7F800000, 0xFF800000, 0x00000000, 0x80000000],
            dtype=np.uint32,
        )
        source = torch.tensor(expected_bits.view(np.float32).tolist())

        for copied in (source.clone(), torch.clone(source)):
            with self.subTest(operation=copied):
                actual_bits = np.asarray(copied).view(np.uint32)
                np.testing.assert_array_equal(actual_bits, expected_bits)

    def test_clone_resets_extreme_empty_view_offsets(self):
        maximum = sys.maxsize
        base = torch.zeros((maximum, 0))
        view = base[maximum - 1].reshape((2, 0, 3))
        self.assertEqual(view.storage_offset(), maximum - 1)

        for copied in (view.clone(), torch.clone(view)):
            with self.subTest(operation=copied):
                self.assertEqual(copied.shape, (2, 0, 3))
                self.assertEqual(copied.stride(), (3, 3, 1))
                self.assertEqual(copied.storage_offset(), 0)
                self.assertEqual(copied.tolist(), [[], []])

        extreme_shape = torch.zeros((0,)).reshape((0, maximum, 3))
        for copied in (extreme_shape.clone(), torch.clone(extreme_shape)):
            with self.subTest(extreme_shape=copied):
                self.assertEqual(copied.shape, extreme_shape.shape)
                self.assertEqual(copied.stride(), extreme_shape.stride())
                self.assertEqual(copied.storage_offset(), 0)

    def test_memory_format_descriptors_match_pytorch_surface(self):
        formats = (
            (torch.preserve_format, "torch.preserve_format"),
            (torch.contiguous_format, "torch.contiguous_format"),
            (torch.channels_last, "torch.channels_last"),
            (torch.channels_last_3d, "torch.channels_last_3d"),
        )
        for memory_format, expected in formats:
            with self.subTest(memory_format=expected):
                self.assertIsInstance(memory_format, torch.memory_format)
                self.assertEqual(repr(memory_format), expected)
                self.assertEqual(str(memory_format), expected)

    def test_clone_supports_keyword_only_memory_formats(self):
        source = torch.zeros((0, 1)) + 1
        formats = (
            (None, (1, 0)),
            (torch.preserve_format, (1, 0)),
            (torch.contiguous_format, (1, 1)),
        )
        for memory_format, expected_stride in formats:
            operations = (
                lambda: source.clone(memory_format=memory_format),
                lambda: torch.clone(source, memory_format=memory_format),
            )
            for operation in operations:
                with self.subTest(memory_format=memory_format, operation=operation):
                    copied = operation()
                    self.assertEqual(copied.shape, source.shape)
                    self.assertEqual(copied.stride(), expected_stride)
                    self.assertEqual(copied.storage_offset(), 0)
                    self.assertEqual(copied.tolist(), source.tolist())

        copied = torch.clone(
            input=torch.tensor([[1.0, 2.0]]),
            memory_format=torch.contiguous_format,
        )
        self.assert_tensor_values(copied, [[1.0, 2.0]], (1, 2))
        self.assertEqual(copied.stride(), (2, 1))

        extreme = torch.zeros((0,)).reshape((0, sys.maxsize, 3))
        self.assertEqual(
            extreme.clone(memory_format=torch.preserve_format).stride(),
            extreme.stride(),
        )
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            extreme.clone(memory_format=torch.contiguous_format)

    def test_clone_rejects_unsupported_formats_and_extra_arguments(self):
        tensor = torch.tensor([1.0])
        for memory_format in (torch.channels_last, torch.channels_last_3d):
            for operation in (
                lambda: tensor.clone(memory_format=memory_format),
                lambda: torch.clone(tensor, memory_format=memory_format),
            ):
                with self.subTest(memory_format=memory_format, operation=operation):
                    with self.assertRaises(RuntimeError):
                        operation()

        for memory_format in (object(), 1, "contiguous_format"):
            for operation in (
                lambda: tensor.clone(memory_format=memory_format),
                lambda: torch.clone(tensor, memory_format=memory_format),
            ):
                with self.subTest(memory_format=memory_format, operation=operation):
                    with self.assertRaisesRegex(TypeError, "must be torch.memory_format"):
                        operation()

        invalid_calls = (
            lambda: tensor.clone(None),
            lambda: tensor.clone(object(), object()),
            lambda: tensor.clone(unexpected=None),
            lambda: torch.clone(),
            lambda: torch.clone([1.0]),
            lambda: torch.clone(tensor, None),
            lambda: torch.clone(tensor, unexpected=None),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

    def test_contiguous_identity_materialization_and_memory_formats(self):
        source = torch.tensor(
            [
                [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]], [[8.0, 9.0], [10.0, 11.0]]],
                [[[12.0, 13.0], [14.0, 15.0]], [[16.0, 17.0], [18.0, 19.0]], [[20.0, 21.0], [22.0, 23.0]]],
            ]
        )
        self.assertIs(source.contiguous(), source)
        self.assertIs(
            source.contiguous(memory_format=torch.contiguous_format), source
        )
        self.assertIs(source.contiguous(memory_format=torch.preserve_format), source)

        view = source.transpose(0, 3)
        packed = view.contiguous()
        self.assertIsNot(packed, view)
        self.assertEqual(packed.shape, view.shape)
        self.assertEqual(packed.stride(), (12, 4, 2, 1))
        self.assertEqual(packed.storage_offset(), 0)
        self.assertEqual(packed.tolist(), view.tolist())
        self.assertIs(packed.contiguous(), packed)

        channels_last = source.contiguous(memory_format=torch.channels_last)
        self.assertIsNot(channels_last, source)
        self.assertEqual(channels_last.stride(), (12, 1, 6, 3))
        self.assertTrue(
            channels_last.is_contiguous(memory_format=torch.channels_last)
        )
        self.assertFalse(channels_last.is_contiguous())
        self.assertEqual(channels_last.tolist(), source.tolist())
        self.assertIs(
            channels_last.contiguous(memory_format=torch.channels_last), channels_last
        )
        row_major = channels_last.contiguous()
        self.assertEqual(row_major.stride(), source.stride())
        self.assertEqual(row_major.tolist(), source.tolist())

        volume = torch.tensor(np.arange(48, dtype=np.float32).reshape(2, 3, 2, 2, 2).tolist())
        channels_last_3d = volume.contiguous(memory_format=torch.channels_last_3d)
        self.assertEqual(channels_last_3d.stride(), (24, 1, 12, 6, 3))
        self.assertTrue(
            channels_last_3d.is_contiguous(memory_format=torch.channels_last_3d)
        )
        self.assertEqual(channels_last_3d.tolist(), volume.tolist())
        self.assertIs(
            channels_last_3d.contiguous(memory_format=torch.channels_last_3d),
            channels_last_3d,
        )

    def test_contiguous_edge_layouts_consumers_and_float_bits(self):
        expected_bits = np.array(
            [0x00000000, 0x80000000, 0x7FC12345, 0x7F800000, 0xFF800000, 0x40A00000],
            dtype=np.uint32,
        )
        source = torch.tensor(expected_bits.view(np.float32).reshape(2, 3).tolist())
        packed = source.transpose(0, 1).contiguous()
        transposed_bits = expected_bits.reshape(2, 3).T.reshape(-1)
        np.testing.assert_array_equal(
            np.asarray(packed).reshape(-1).view(np.uint32), transposed_bits
        )
        self.assertEqual(packed.clone().stride(), packed.stride())
        np.testing.assert_array_equal(
            np.asarray(packed.reshape(2, 3)).reshape(-1).view(np.uint32),
            np.asarray(packed).reshape(-1).view(np.uint32),
        )
        self.assertEqual((packed + 1).shape, packed.shape)
        self.assertTrue(np.isnan(packed.sum().item()))
        np.testing.assert_array_equal(
            np.asarray(packed.tolist(), dtype=np.float32).view(np.uint32),
            np.asarray(packed).view(np.uint32),
        )
        self.assertIn("shape=[3, 2]", repr(packed))

        singleton = torch.zeros((2, 1, 4, 5))
        self.assertIs(
            singleton.contiguous(memory_format=torch.channels_last), singleton
        )
        self.assertEqual(singleton.stride(), (20, 20, 5, 1))

        empty_cases = (
            ((2, 0, 4, 5), (0, 1, 0, 0)),
            ((2, 3, 0, 5), (0, 1, 15, 3)),
            ((2, 3, 4, 0), (0, 1, 0, 3)),
            ((0, 3, 4, 5), (60, 1, 15, 3)),
        )
        for shape, stride in empty_cases:
            with self.subTest(shape=shape):
                result = torch.zeros(shape).contiguous(
                    memory_format=torch.channels_last
                )
                self.assertEqual(result.stride(), stride)
                self.assertEqual(result.storage_offset(), 0)
                self.assertEqual(result.tolist(), torch.zeros(shape).tolist())

        scalar = torch.tensor(-0.0)
        self.assertIs(scalar.contiguous(), scalar)
        self.assertEqual(np.asarray(scalar).view(np.uint32).item(), 0x80000000)

    def test_contiguous_rejects_invalid_calls_with_pytorch_diagnostics(self):
        tensor = torch.zeros((2, 3))
        invalid_calls = (
            (lambda: tensor.contiguous(torch.contiguous_format), TypeError, "contiguous() takes 0 positional arguments but 1 was given"),
            (lambda: tensor.contiguous(None), TypeError, "contiguous() takes 0 positional arguments but 1 was given"),
            (lambda: tensor.contiguous(memory_format=None), TypeError, "contiguous(): argument 'memory_format' must be torch.memory_format, not NoneType"),
            (lambda: tensor.contiguous(memory_format=1), TypeError, "contiguous(): argument 'memory_format' must be torch.memory_format, not int"),
            (lambda: tensor.contiguous(unexpected=None), TypeError, "contiguous() got an unexpected keyword argument 'unexpected'"),
            (lambda: tensor.contiguous(memory_format=torch.channels_last), RuntimeError, "required rank 4 tensor to use channels_last format"),
            (lambda: tensor.contiguous(memory_format=torch.channels_last_3d), RuntimeError, "required rank 5 tensor to use channels_last_3d format"),
            (
                lambda: tensor.transpose(0, 1).contiguous(memory_format=torch.preserve_format),
                RuntimeError,
                "preserve memory format is unsupported by the contiguous operator",
            ),
        )
        for call, error_type, message in invalid_calls:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, re.escape(message)):
                    call()

    def test_integer_indexing_accepts_index_protocol_values(self):
        class IntSubclass(int):
            pass

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        first = IndexValue(-1)
        second = IndexValue(0)
        self.assertEqual(tensor[first, second].item(), 3.0)
        self.assertEqual((first.calls, second.calls), (1, 1))
        self.assertEqual(tensor[IntSubclass(1)].tolist(), [3.0, 4.0])
        self.assertEqual(tensor[np.int64(-1)].tolist(), [3.0, 4.0])
        self.assertEqual(tensor[np.uint64(0)].tolist(), [1.0, 2.0])

        scalar_index = IndexValue(0)
        with self.assertRaisesRegex(IndexError, "too many indices for tensor of dimension 0"):
            torch.tensor(1.0)[scalar_index]
        self.assertEqual(scalar_index.calls, 0)

        with self.assertRaisesRegex(IndexError, "invalid index of a 0-dim tensor"):
            torch.tensor(1.0)[np.int64(0)]

    def test_integer_indexing_stops_converting_after_the_first_axis_error(self):
        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        tensor = torch.zeros((2, 3))
        later = IndexValue(0)
        with self.assertRaisesRegex(
            IndexError, "index 2 is out of bounds for dimension 0 with size 2"
        ):
            tensor[2, later]
        self.assertEqual(later.calls, 0)

        first = IndexValue(2)
        later = IndexValue(0)
        with self.assertRaisesRegex(
            IndexError, "index 2 is out of bounds for dimension 0 with size 2"
        ):
            tensor[first, later]
        self.assertEqual(first.calls, 1)
        self.assertEqual(later.calls, 0)

        with self.assertRaisesRegex(
            IndexError, "index 2 is out of bounds for dimension 0 with size 2"
        ):
            tensor[2, 1 << 100]

    def test_integer_indexing_rejects_bool_and_non_integer_forms(self):
        class IntOnly:
            def __int__(self):
                return 0

        class FailingIndex:
            def __index__(self):
                raise RuntimeError("conversion failed")

        tensor = torch.zeros((2, 3))
        invalid = (
            True,
            False,
            np.bool_(True),
            1.0,
            np.float64(1.0),
            slice(None),
            [0],
            None,
            Ellipsis,
            IntOnly(),
            FailingIndex(),
        )
        for index in invalid:
            with self.subTest(index=repr(index)):
                with self.assertRaisesRegex(IndexError, "only integers"):
                    tensor[index]

        with self.assertRaisesRegex(IndexError, "only integers"):
            tensor[0, True]
        for index in (1 << 100, -(1 << 100), np.uint64(2**64 - 1)):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "Overflow when unpacking long long"):
                    tensor[index]

    def test_integer_indexing_matches_pytorch_errors_and_empty_offsets(self):
        tensor = torch.zeros((2, 3, 4))
        errors = (
            (2, "index 2 is out of bounds for dimension 0 with size 2"),
            (-3, "index -3 is out of bounds for dimension 0 with size 2"),
            ((0, 3), "index 3 is out of bounds for dimension 1 with size 3"),
        )
        for index, message in errors:
            with self.subTest(index=index):
                with self.assertRaisesRegex(IndexError, message):
                    tensor[index]

        with self.assertRaisesRegex(IndexError, "too many indices for tensor of dimension 3"):
            tensor[99, 0, 0, 0]

        scalar = torch.tensor(5.0)
        with self.assertRaisesRegex(IndexError, "invalid index of a 0-dim tensor"):
            scalar[0]
        with self.assertRaisesRegex(
            IndexError, "index -1 is out of bounds for dimension 0 with size 0"
        ):
            scalar[-1]
        with self.assertRaisesRegex(IndexError, "too many indices for tensor of dimension 0"):
            scalar[(0,)]

        empty = torch.zeros((2, 0, 3))
        view = empty[1]
        self.assertEqual(view.shape, (0, 3))
        self.assertEqual(view.stride(), (3, 1))
        self.assertEqual(view.storage_offset(), 3)
        self.assertEqual(view.tolist(), [])
        with self.assertRaisesRegex(
            IndexError, "index 0 is out of bounds for dimension 1 with size 0"
        ):
            empty[1, 0]

        maximum = sys.maxsize
        extreme = torch.zeros((maximum, 0))[maximum - 1]
        self.assertEqual(extreme.storage_offset(), maximum - 1)
        with self.assertRaisesRegex(RuntimeError, "Tensor: invalid storage offset -4"):
            extreme.reshape((maximum, 0))[maximum - 1]

    def test_empty_elementwise_results_match_pytorch_strides(self):
        scalar_cases = (
            ((1, 0), (1, 1)),
            ((0, 1), (1, 0)),
            ((1, 0, 1), (0, 1, 0)),
            ((2, 0, 3), (3, 3, 1)),
        )
        for shape, expected in scalar_cases:
            with self.subTest(operation="scalar", shape=shape):
                self.assertEqual((torch.zeros(shape) + 1).stride(), expected)

        empty = torch.zeros((1, 0, 1))
        self.assertEqual((empty + torch.ones((1, 0, 1))).stride(), (1, 1, 1))

        broadcast = empty + torch.ones((2, 1, 3))
        self.assertEqual(broadcast.shape, (2, 0, 3))
        self.assertEqual(broadcast.stride(), (3, 3, 1))

        compatible = torch.zeros((0, 1)) + torch.ones((1, 1))
        self.assertEqual(compatible.stride(), (1, 0))

        chained = torch.zeros((0, 1)) + 1
        self.assertEqual(chained.stride(), (1, 0))
        self.assertEqual(chained.relu().stride(), (1, 1))

    def test_extreme_empty_pointwise_outputs_match_pytorch_stride_boundaries(self):
        tensor = torch.zeros((0,)).reshape((0, sys.maxsize, 3))

        scalar_output = tensor + 1
        self.assertEqual(scalar_output.shape, (0, sys.maxsize, 3))
        self.assertEqual(scalar_output.stride(), (1, 0, 0))
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            tensor.relu()
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            tensor.sin()

        wrapped_shape = torch.zeros((0,)).reshape(
            (0, 2, sys.maxsize, sys.maxsize)
        )
        wrapped_output = wrapped_shape + 1
        self.assertEqual(wrapped_output.shape, wrapped_shape.shape)
        self.assertEqual(wrapped_output.stride(), (2, sys.maxsize, 1, 1))
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            wrapped_shape.sin()

        zeroed_byte_stride = torch.zeros((0,)).reshape((0, 1, 2, 1 << 61))
        self.assertEqual((zeroed_byte_stride + 1).stride(), (0, 0, 1, 2))

    def test_empty_reshape_preserves_compatible_source_strides(self):
        source = torch.zeros((0, 1)) + 1
        view = source.reshape((0, 1))

        self.assertEqual(source.stride(), (1, 0))
        self.assertEqual(view.stride(), (1, 0))
        self.assertEqual(view.shape, source.shape)
        self.assertEqual(view.tolist(), source.tolist())

    def test_reshape_accepts_variadic_and_sequence_signatures(self):
        source = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        variadic = source.reshape(3, 2)
        tuple_shape = source.reshape((1, 6))
        list_shape = source.reshape([6, 1])
        keyword_shape = source.reshape(shape=(2, 3))

        self.assertEqual(variadic.shape, (3, 2))
        self.assertEqual(variadic.stride(), (2, 1))
        self.assertEqual(variadic.tolist(), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(tuple_shape.tolist(), [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        self.assertEqual(list_shape.tolist(), [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        self.assertEqual(keyword_shape.tolist(), source.tolist())
        self.assertEqual(source.tolist(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with self.assertRaises(TypeError):
            source.reshape(shape=-1)

    def test_reshape_inference_scalar_and_empty_cases(self):
        source = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(source.reshape(2, -1).shape, (2, 3))
        self.assertEqual(source.reshape(-1).shape, (6,))

        scalar = torch.tensor([7.0]).reshape(())
        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.stride(), ())
        self.assertEqual(scalar.item(), 7.0)
        self.assertEqual(scalar.reshape([1]).tolist(), [7.0])

        empty = torch.zeros((0,))
        inferred = empty.reshape(2, -1, 3)
        self.assertEqual(inferred.shape, (2, 0, 3))
        self.assertEqual(inferred.stride(), (3, 3, 1))
        self.assertEqual(inferred.tolist(), [[], []])
        self.assertEqual(empty.reshape((0, 2)).shape, (0, 2))

        large = 2**32
        large_empty = empty.reshape((0, large, large))
        self.assertEqual(large_empty.shape, (0, large, large))
        self.assertEqual(large_empty.stride(), (0, large, 1))
        self.assertEqual(large_empty.numel(), 0)

        maximum = sys.maxsize
        wrapped_inference = empty.reshape(-1, maximum, maximum)
        self.assertEqual(wrapped_inference.shape, (0, maximum, maximum))
        self.assertEqual(wrapped_inference.stride(), (1, maximum, 1))
        self.assertEqual(wrapped_inference.tolist(), [])

        with self.assertRaisesRegex(RuntimeError, "element count overflowed"):
            torch.tensor([1.0]).reshape(maximum, maximum, -1)
        with self.assertRaisesRegex(RuntimeError, "element count overflowed"):
            empty.reshape(3, maximum, -1)

        with self.assertRaisesRegex(RuntimeError, "is invalid for input of size 0"):
            empty.reshape(2, -1, 1 << 62)

        self.assertEqual(empty.reshape((0, maximum, maximum)).tolist(), [])

        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            empty.reshape((0, 1 << 62, 3))

    def test_reshape_reports_pytorch_compatible_errors(self):
        tensor = torch.zeros((6,))
        invalid = (
            ((4, 2), "shape '\\[4, 2\\]' is invalid for input of size 6"),
            ((-1, -1), "only one dimension can be inferred"),
            ((-2, 3), "invalid shape dimension -2 at index 0 of shape \\[-2, 3\\]"),
        )
        for shape, message in invalid:
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(RuntimeError, message):
                    tensor.reshape(shape)

        with self.assertRaisesRegex(RuntimeError, "unspecified dimension size -1"):
            torch.zeros((0,)).reshape(0, -1)

        large = 2**62
        with self.assertRaisesRegex(RuntimeError, "invalid shape dimension -2"):
            tensor.reshape((large, 4, -2))
        with self.assertRaisesRegex(RuntimeError, "only one dimension can be inferred"):
            tensor.reshape((large, 4, -1, -1))

        for shape in ((2.0, 3), (True, 6), [[2, 3]]):
            with self.subTest(shape=shape):
                with self.assertRaises(TypeError):
                    tensor.reshape(shape)

        with self.assertRaises(TypeError):
            torch.tensor(1.0).reshape()

    def test_reshape_observables_survive_source_lifetime_and_numpy_mutation(self):
        source = torch.tensor([1.0, 2.0, 3.0, 4.0])
        view = source.reshape(2, 2)
        del source

        copied = np.asarray(view)
        copied[0, 0] = 99.0
        self.assertEqual(view.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual((view + 1.0).tolist(), [[2.0, 3.0], [4.0, 5.0]])

    def test_matrix_multiplication_operator(self):
        left = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        right = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        output = left @ right
        self.assertEqual(output.shape, (2, 2))
        self.assertEqual(output.tolist(), [[58.0, 64.0], [139.0, 154.0]])

    def test_binary_arithmetic_broadcasts_trailing_dimensions(self):
        left = torch.tensor([[[1.0, 2.0, 4.0]], [[8.0, 16.0, 32.0]]])
        right = torch.tensor([[1.0], [2.0], [4.0]])
        cases = (
            (
                operator.add,
                [
                    [[2.0, 3.0, 5.0], [3.0, 4.0, 6.0], [5.0, 6.0, 8.0]],
                    [
                        [9.0, 17.0, 33.0],
                        [10.0, 18.0, 34.0],
                        [12.0, 20.0, 36.0],
                    ],
                ],
            ),
            (
                operator.sub,
                [
                    [[0.0, 1.0, 3.0], [-1.0, 0.0, 2.0], [-3.0, -2.0, 0.0]],
                    [
                        [7.0, 15.0, 31.0],
                        [6.0, 14.0, 30.0],
                        [4.0, 12.0, 28.0],
                    ],
                ],
            ),
            (
                operator.mul,
                [
                    [[1.0, 2.0, 4.0], [2.0, 4.0, 8.0], [4.0, 8.0, 16.0]],
                    [
                        [8.0, 16.0, 32.0],
                        [16.0, 32.0, 64.0],
                        [32.0, 64.0, 128.0],
                    ],
                ],
            ),
            (
                operator.truediv,
                [
                    [[1.0, 2.0, 4.0], [0.5, 1.0, 2.0], [0.25, 0.5, 1.0]],
                    [
                        [8.0, 16.0, 32.0],
                        [4.0, 8.0, 16.0],
                        [2.0, 4.0, 8.0],
                    ],
                ],
            ),
        )

        for operation, expected in cases:
            with self.subTest(operation=operation):
                self.assert_tensor_values(operation(left, right), expected, (2, 3, 3))

    def test_binary_arithmetic_broadcasts_scalars_and_zero_dimensions(self):
        scalar = torch.tensor(2.0)
        matrix = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
        self.assert_tensor_values(matrix + scalar, [[3.0, 5.0], [7.0, 9.0]], (2, 2))
        self.assert_tensor_values(scalar - matrix, [[1.0, -1.0], [-3.0, -5.0]], (2, 2))

        empty = torch.zeros((2, 0, 3))
        row = torch.ones((1, 1, 3))
        for operation in (operator.add, operator.sub, operator.mul, operator.truediv):
            with self.subTest(operation=operation):
                self.assert_tensor_values(operation(empty, row), [[], []], (2, 0, 3))

        self.assertEqual((torch.zeros((0,)) + torch.ones((1,))).shape, (0,))

        large_empty = torch.full((sys.maxsize, 0), 1.0)
        large_output = large_empty + torch.tensor(2.0)
        self.assertEqual(large_output.shape, (sys.maxsize, 0))
        self.assertEqual(large_output.numel(), 0)

        large = sys.maxsize // 2 + 1
        left = torch.full((0, large, 1), 1.0)
        right = torch.tensor([[[1.0, 2.0]]])
        for operation in (operator.add, operator.sub, operator.mul, operator.truediv):
            with self.subTest(operation=operation, shape=(0, large, 2)):
                with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                    operation(left, right)

    def test_python_real_scalar_and_reverse_arithmetic(self):
        tensor = torch.tensor([1.0, -2.0, 4.0])
        cases = (
            (tensor + 2, [3.0, 0.0, 6.0]),
            (2 + tensor, [3.0, 0.0, 6.0]),
            (tensor - 2.0, [-1.0, -4.0, 2.0]),
            (2.0 - tensor, [1.0, 4.0, -2.0]),
            (tensor * np.float32(2.0), [2.0, -4.0, 8.0]),
            (np.float32(2.0) * tensor, [2.0, -4.0, 8.0]),
            (tensor / 2, [0.5, -1.0, 2.0]),
            (2 / tensor, [2.0, -1.0, 0.5]),
            (tensor + True, [2.0, -1.0, 5.0]),
        )
        for actual, expected in cases:
            with self.subTest(expected=expected):
                self.assert_tensor_values(actual, expected, (3,))

        zero = torch.tensor(0.0)
        self.assertEqual((zero + (-(2**63))).item(), -9223372036854775808.0)
        self.assertEqual((zero + (2**64 - 1)).item(), 18446744073709551616.0)
        self.assertEqual(
            (zero + np.uint64(2**63 - 1)).item(),
            9223372036854775808.0,
        )

    def test_wide_numpy_unsigned_scalars_delegate_to_numpy(self):
        tensor = torch.tensor([0.0])
        value = np.uint64(2**63 + 2048)

        result = tensor + value
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.dtype(np.float64))
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result[0], np.float64(2**63 + 2048))
        self.assertNotEqual(result[0], np.float64(2**63))

        for operation in (operator.add, operator.sub, operator.mul):
            with self.subTest(operation=operation):
                with self.assertRaises(TypeError):
                    operation(value, tensor)

        denominator = torch.tensor([2.0])
        for numerator in (
            np.uint64(2**63),
            np.uint64(2**63 + 2048),
            np.uint64(2**64 - 1),
        ):
            with self.subTest(numerator=numerator, operation=operator.truediv):
                result = numerator / denominator
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.dtype, np.dtype(np.float64))
                self.assertEqual(result.shape, (1,))
                self.assertEqual(result[0], np.float64(numerator) / np.float64(2.0))

    def test_numpy_array_conversion_rejects_requests_prohibiting_a_copy(self):
        tensor = torch.tensor([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "non-copying NumPy view"):
            np.array(tensor, copy=False)

        copied = np.array(tensor, copy=True)
        self.assertEqual(copied.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(copied, np.array([1.0, 2.0], dtype=np.float32))
        copied[0] = 9.0
        self.assertEqual(tensor.tolist(), [1.0, 2.0])

    def test_python_bool_subtraction_matches_pytorch_errors(self):
        tensor = torch.tensor([1.0, 2.0])
        for operation in (
            lambda: tensor - True,
            lambda: False - tensor,
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(RuntimeError, "bool tensor is not supported"):
                    operation()

        numpy_bool = np.bool_(True)
        self.assert_tensor_values(tensor - numpy_bool, [0.0, 1.0], (2,))
        self.assert_tensor_values(numpy_bool - tensor, [0.0, -1.0], (2,))

    def test_unsupported_operands_use_python_reflected_dispatch(self):
        class ReflectedArithmetic:
            def __init__(self):
                self.calls = []

            def reflected(self, name, tensor):
                self.calls.append(name)
                return name, tensor

            def __radd__(self, tensor):
                return self.reflected("add", tensor)

            def __rsub__(self, tensor):
                return self.reflected("sub", tensor)

            def __rmul__(self, tensor):
                return self.reflected("mul", tensor)

            def __rtruediv__(self, tensor):
                return self.reflected("truediv", tensor)

        tensor = torch.tensor([1.0])
        value = ReflectedArithmetic()
        for operation, expected_name in (
            (operator.add, "add"),
            (operator.sub, "sub"),
            (operator.mul, "mul"),
            (operator.truediv, "truediv"),
        ):
            with self.subTest(operation=operation):
                name, reflected_tensor = operation(tensor, value)
                self.assertEqual(name, expected_name)
                self.assertIs(reflected_tensor, tensor)
        self.assertEqual(value.calls, ["add", "sub", "mul", "truediv"])

    def test_recognized_scalar_errors_do_not_fall_back_to_reflection(self):
        class OverflowingInteger(int):
            def __new__(cls):
                instance = super().__new__(cls, 2**64)
                instance.reflected = False
                return instance

            def __rmul__(self, tensor):
                self.reflected = True
                return tensor

        value = OverflowingInteger()
        with self.assertRaises(OverflowError):
            torch.ones((1,)) * value
        self.assertFalse(value.reflected)

    def test_scalar_division_preserves_non_finite_and_signed_zero_results(self):
        tensor = torch.tensor([1.0, -1.0, 0.0, -0.0])
        self.assert_tensor_values(
            tensor / -0.0,
            [-math.inf, math.inf, math.nan, math.nan],
            (4,),
        )
        self.assert_tensor_values(
            -0.0 / tensor,
            [-0.0, 0.0, math.nan, math.nan],
            (4,),
        )
        self.assert_tensor_values(
            tensor + math.nan,
            [math.nan, math.nan, math.nan, math.nan],
            (4,),
        )

        self.assert_tensor_values(
            1.0e-38 / torch.tensor([1.0e-39]),
            [math.inf],
            (1,),
        )
        self.assert_tensor_values(
            0.0 / torch.tensor([1.0e-39]),
            [math.nan],
            (1,),
        )

        scalar = np.array([0xC25FB64C], dtype=np.uint32).view(np.float32)[0].item()
        denominator = (
            np.array([0xC27C80A7], dtype=np.uint32).view(np.float32)[0].item()
        )
        expected = np.array([0x3F62CF8F], dtype=np.uint32).view(np.float32)
        self.assert_tensor_values(
            scalar / torch.tensor([denominator]),
            expected,
            (1,),
        )

    def test_scalar_arithmetic_rejects_non_real_and_out_of_range_values(self):
        tensor = torch.ones((2,))
        for value in (object(), Decimal("1.0"), 1 + 2j, [1.0]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    operator.add(tensor, value)
                with self.assertRaises(TypeError):
                    operator.add(value, tensor)

        for value in (-(2**63) - 1, 2**64):
            with self.subTest(value=value):
                with self.assertRaises(OverflowError):
                    tensor * value

    def test_subtraction_and_division_cover_general_same_shapes(self):
        cases = (
            (torch.tensor(7.0), torch.tensor(2.0), (), 5.0, 3.5),
            (
                torch.tensor([[[12.0, -8.0]], [[3.0, 0.5]]]),
                torch.tensor([[[3.0, 2.0]], [[-1.5, 0.25]]]),
                (2, 1, 2),
                [[[9.0, -10.0]], [[4.5, 0.25]]],
                [[[4.0, -4.0]], [[-2.0, 2.0]]],
            ),
            (
                torch.full((2, 0, 3), 1.0),
                torch.full((2, 0, 3), 2.0),
                (2, 0, 3),
                [[], []],
                [[], []],
            ),
        )

        for left, right, shape, expected_sub, expected_div in cases:
            with self.subTest(shape=shape):
                self.assert_tensor_values(left - right, expected_sub, shape)
                self.assert_tensor_values(left / right, expected_div, shape)

    def test_subtraction_and_division_match_pytorch_special_values(self):
        cases = (
            (
                operator.sub,
                [math.nan, math.inf, -math.inf, math.inf, -math.inf, -0.0, 0.0],
                [1.0, math.inf, -math.inf, -math.inf, math.inf, 0.0, -0.0],
            ),
            (
                operator.truediv,
                [
                    math.nan,
                    math.inf,
                    -math.inf,
                    math.inf,
                    -math.inf,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    0.0,
                    -0.0,
                    0.0,
                    -0.0,
                ],
                [
                    1.0,
                    math.inf,
                    -math.inf,
                    2.0,
                    2.0,
                    0.0,
                    0.0,
                    -0.0,
                    -0.0,
                    2.0,
                    2.0,
                    -2.0,
                    -2.0,
                ],
            ),
        )

        expected = (
            [math.nan, math.nan, math.nan, math.inf, -math.inf, -0.0, 0.0],
            [
                math.nan,
                math.nan,
                math.nan,
                math.inf,
                -math.inf,
                math.inf,
                -math.inf,
                -math.inf,
                math.inf,
                0.0,
                -0.0,
                -0.0,
                0.0,
            ],
        )
        for (operation, left, right), expected_values in zip(cases, expected):
            with self.subTest(operation=operation):
                self.assert_tensor_values(
                    operation(torch.tensor(left), torch.tensor(right)),
                    expected_values,
                    (len(expected_values),),
                )

    def test_binary_arithmetic_rejects_incompatible_shapes(self):
        left = torch.zeros([2, 2])
        right = torch.zeros([3])

        for operation in (operator.add, operator.sub, operator.mul, operator.truediv):
            with self.subTest(operation=operation):
                with self.assertRaises(RuntimeError):
                    operation(left, right)

    def test_ragged_input_is_rejected(self):
        with self.assertRaises(ValueError):
            torch.tensor([[1.0], [2.0, 3.0]])

    def test_full_handles_scalar_empty_and_multidimensional_shapes(self):
        scalar = torch.full([], -2.5)
        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.numel(), 1)
        self.assertEqual(scalar.item(), -2.5)

        empty = torch.full([2, 0, 3], 7.0)
        self.assertEqual(empty.shape, (2, 0, 3))
        self.assertEqual(empty.numel(), 0)
        self.assertEqual(empty.tolist(), [[], []])

        matrix = torch.full((2, 3), 1.25)
        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(matrix.tolist(), [[1.25] * 3] * 2)

    def test_tolist_maps_zero_element_list_capacity_overflow_to_memory_error(self):
        tensor = torch.full((sys.maxsize, 0), 1.0)
        self.assertEqual(tensor.numel(), 0)

        with self.assertRaises(MemoryError):
            tensor.tolist()

    def test_full_preserves_nan_and_infinities(self):
        nan_values = torch.full([2], math.nan).tolist()
        self.assertTrue(all(math.isnan(value) for value in nan_values))
        self.assertEqual(torch.full([2], math.inf).tolist(), [math.inf, math.inf])
        self.assertEqual(torch.full([2], -math.inf).tolist(), [-math.inf, -math.inf])

    def test_full_accepts_pytorch_keyword_names(self):
        result = torch.full(size=[2], fill_value=3.0)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.tolist(), [3.0, 3.0])

    def test_full_rejects_negative_sizes_as_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "negative dimension -1"):
            torch.full([-1], 3.0)

    def test_full_rejects_storage_capacity_overflow(self):
        oversized = sys.maxsize // 4 + 1
        with self.assertRaisesRegex(RuntimeError, "exceeds the platform capacity"):
            torch.full([oversized], 1.0)

    def test_full_rejects_finite_fill_value_overflow(self):
        for fill_value in (1e40, -1e40):
            with self.subTest(fill_value=fill_value):
                with self.assertRaisesRegex(RuntimeError, "float32 without overflow"):
                    torch.full((2,), fill_value)

    def test_full_maps_shape_product_overflow_to_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "Storage size calculation overflowed"):
            torch.full((2**62, 4), 1.0)

    def test_full_rejects_invalid_size_arguments(self):
        for size in ([True], (False,), range(2)):
            with self.subTest(size=size):
                with self.assertRaises(TypeError):
                    torch.full(size, 3.0)

    def test_full_accepts_index_protocol_dimensions(self):
        class IndexDimension:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        dimension = IndexDimension(2)
        result = torch.full([dimension], 3.0)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.tolist(), [3.0, 3.0])
        self.assertEqual(dimension.calls, 1)

    def test_full_normalizes_invalid_index_dimensions_to_type_error(self):
        class FailingIndex:
            def __index__(self):
                raise RuntimeError("index conversion failed")

        for dimension in (2**63, -(2**63) - 1, FailingIndex()):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(TypeError, "size element at index 0"):
                    torch.full([dimension], 3.0)

    def test_full_accepts_scalar_tensor_fill_value(self):
        result = torch.full((2,), torch.tensor(3.0))
        self.assertEqual(result.tolist(), [3.0, 3.0])

        with self.assertRaises(TypeError):
            torch.full((2,), torch.tensor([3.0]))

    def test_full_accepts_real_numpy_scalar_fill_values(self):
        cases = (
            (np.longdouble(1.25), [1.25, 1.25]),
            (np.float32(1.25), [1.25, 1.25]),
            (np.int64(3), [3.0, 3.0]),
            (np.bool_(True), [1.0, 1.0]),
        )
        for fill_value, expected in cases:
            with self.subTest(fill_value=fill_value):
                self.assertEqual(torch.full((2,), fill_value).tolist(), expected)

    def test_full_rejects_zero_dimensional_buffer_fill_values(self):
        array = np.array(3.0)
        for fill_value in (array, memoryview(array)):
            with self.subTest(fill_value=fill_value):
                with self.assertRaises(TypeError):
                    torch.full((2,), fill_value)

    def test_full_enforces_numpy_integer_signed_boundary(self):
        accepted = (
            np.int64(-(2**63)),
            np.int64(2**63 - 1),
            np.uint64(2**63 - 1),
        )
        for fill_value in accepted:
            with self.subTest(fill_value=fill_value):
                self.assertEqual(torch.full((1,), fill_value).numel(), 1)

        for fill_value in (np.uint64(2**63), np.uint64(2**64 - 1)):
            with self.subTest(fill_value=fill_value):
                with self.assertRaises(TypeError):
                    torch.full((1,), fill_value)

    def test_full_rejects_non_scalar_numeric_coercions(self):
        class FloatLike:
            def __init__(self):
                self.calls = 0

            def __float__(self):
                self.calls += 1
                return 3.0

        float_like = FloatLike()
        for fill_value in (Decimal("3.0"), float_like):
            with self.subTest(fill_value=fill_value):
                with self.assertRaises(TypeError):
                    torch.full((2,), fill_value)
        self.assertEqual(float_like.calls, 0)

    def test_full_converts_integer_fill_values_without_double_rounding(self):
        class IntWithFloat(int):
            def __new__(cls, value):
                instance = super().__new__(cls, value)
                instance.float_calls = 0
                return instance

            def __float__(self):
                self.float_calls += 1
                return 0.0

        fill_value = IntWithFloat(9007199791611905)
        result = torch.full((1,), fill_value)
        self.assertEqual(result.item(), 9007200328482816.0)
        self.assertEqual(fill_value.float_calls, 0)

    def test_full_enforces_python_integer_scalar_boundaries(self):
        accepted = (
            (-(2**63), -9223372036854775808.0),
            (2**64 - 1, 18446744073709551616.0),
        )
        for fill_value, expected in accepted:
            with self.subTest(fill_value=fill_value):
                self.assertEqual(torch.full((1,), fill_value).item(), expected)

        for fill_value in (-(2**63) - 1, 2**64):
            with self.subTest(fill_value=fill_value):
                with self.assertRaises(OverflowError):
                    torch.full((1,), fill_value)

    def test_full_matches_pytorch_validation_order(self):
        with self.assertRaises(TypeError):
            torch.full([-1], object())

        with self.assertRaisesRegex(RuntimeError, "Storage size calculation overflowed"):
            torch.full((2**62, 4), 1e40)

    def test_full_validates_strides_for_empty_shapes(self):
        large = 2**62
        for size in ((0, large, 2), (2, 0, large, 2), (1, large, 2, 0)):
            with self.subTest(size=size):
                with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                    torch.full(size, 1.0)


if __name__ == "__main__":
    unittest.main()
