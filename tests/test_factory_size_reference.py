import sys
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class FactorySizeReferenceTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, compare_values=True):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.numel(), expected.numel())
        self.assertIs(actual.dtype, torch.float32)
        self.assertEqual(actual.device, torch.device("cpu"))
        if compare_values:
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32),
                expected.cpu().numpy(),
            )

    def assert_error_matches(self, actual_call, expected_call, *, exact=True):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        if exact:
            self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        else:
            actual_first_line = str(actual_raised.exception).splitlines()[0]
            expected_first_line = str(expected_raised.exception).splitlines()[0]
            self.assertTrue(expected_first_line.startswith(actual_first_line.rstrip('"')))

    def test_deterministic_and_generated_call_forms_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0x510E_213)
        shapes = [(), (0,), (3,), (2, 3), (2, 0, 4), (1,) * 96]
        for rank in range(17):
            shape = tuple(int(value) for value in rng.integers(0, 3, size=rank))
            if rank > 8:
                shape = shape[: rank // 2] + (0,) + shape[rank // 2 + 1 :]
            shapes.append(shape)
        for _ in range(24):
            rank = int(rng.integers(0, 13))
            shapes.append(tuple(int(value) for value in rng.integers(0, 3, size=rank)))

        for factory_name in ("zeros", "ones"):
            actual_factory = getattr(torch, factory_name)
            expected_factory = getattr(reference_torch, factory_name)
            for case, shape in enumerate(shapes):
                forms = (
                    ("tuple", lambda factory, shape=shape: factory(tuple(shape))),
                    ("list", lambda factory, shape=shape: factory(list(shape))),
                    (
                        "torch.Size",
                        lambda factory, shape=shape: factory(reference_torch.Size(shape)),
                    ),
                    ("size keyword", lambda factory, shape=shape: factory(size=tuple(shape))),
                )
                if shape:
                    forms += (
                        ("variadic", lambda factory, shape=shape: factory(*shape)),
                    )
                if len(shape) == 1:
                    forms += (("single integer", lambda factory, shape=shape: factory(shape[0])),)

                for form, call in forms:
                    with self.subTest(
                        factory=factory_name,
                        case=case,
                        rank=len(shape),
                        form=form,
                    ):
                        actual = call(actual_factory)
                        expected = call(expected_factory)
                        self.assert_tensor_matches(
                            actual,
                            expected,
                            compare_values=len(shape) <= 32,
                        )

    def test_integer_protocol_bool_and_invalid_inputs_match(self):
        class IntSubclass(int):
            pass

        class Indexable:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        class BadIndex:
            def __index__(self):
                raise RuntimeError("cannot index")

        class RaisesOverflow:
            def __index__(self):
                raise OverflowError("deliberate overflow")

        accepted_sizes = (
            (IntSubclass(2), IntSubclass(3)),
            (Indexable(2), Indexable(3)),
            (np.int64(2), np.int32(3)),
            (2, True, False),
        )
        for factory_name in ("zeros", "ones"):
            actual_factory = getattr(torch, factory_name)
            expected_factory = getattr(reference_torch, factory_name)
            for size in accepted_sizes:
                for form, actual, expected in (
                    (
                        "variadic",
                        lambda size=size: actual_factory(*size),
                        lambda size=size: expected_factory(*size),
                    ),
                    (
                        "tuple",
                        lambda size=size: actual_factory(size),
                        lambda size=size: expected_factory(size),
                    ),
                    (
                        "size keyword",
                        lambda size=size: actual_factory(size=size),
                        lambda size=size: expected_factory(size=size),
                    ),
                ):
                    with self.subTest(factory=factory_name, size=size, form=form):
                        self.assert_tensor_matches(actual(), expected())

            error_cases = (
                (lambda factory: factory(), True),
                (lambda factory: factory(True), True),
                (lambda factory: factory((True, 2)), True),
                (lambda factory: factory(np.bool_(True)), True),
                (lambda factory: factory(2.0), True),
                (lambda factory: factory(object()), True),
                (lambda factory: factory((2, object())), True),
                (lambda factory: factory(2, BadIndex()), True),
                (lambda factory: factory(RaisesOverflow()), True),
                (lambda factory: factory((RaisesOverflow(), 2)), True),
                (lambda factory: factory(2, RaisesOverflow()), True),
                (lambda factory: factory([2, RaisesOverflow()]), True),
                (lambda factory: factory(size=(2, RaisesOverflow())), True),
                (lambda factory: factory((2,), 3), True),
                (lambda factory: factory(size=2), True),
                (lambda factory: factory((2,), size=(2,)), True),
                (lambda factory: factory((2,), unknown=True), True),
                (lambda factory: factory((2,), dtype="float32"), True),
                (lambda factory: factory((2,), device=object()), True),
                (lambda factory: factory((2, object()), dtype="float32"), True),
                (lambda factory: factory(1 << 100), False),
                (lambda factory: factory((2, 1 << 100)), False),
            )
            for case, (call, exact) in enumerate(error_cases):
                with self.subTest(factory=factory_name, invalid_case=case):
                    self.assert_error_matches(
                        lambda call=call: call(actual_factory),
                        lambda call=call: call(expected_factory),
                        exact=exact,
                    )

    def test_later_index_conversion_order_matches_pytorch_2_13(self):
        def error_outcome(factory, call):
            events = []

            class Probe:
                def __index__(self):
                    events.append("index")
                    return 3

            try:
                call(factory, Probe())
            except Exception as error:
                return events, type(error).__name__, str(error)
            self.fail("invalid factory call unexpectedly succeeded")

        invalid_calls = (
            ("variadic dtype", lambda factory, probe: factory(2, probe, dtype="bad")),
            ("tuple dtype", lambda factory, probe: factory((2, probe), dtype="bad")),
            (
                "size keyword dtype",
                lambda factory, probe: factory(size=(2, probe), dtype="bad"),
            ),
            ("variadic device", lambda factory, probe: factory(2, probe, device=object())),
            ("duplicate size", lambda factory, probe: factory(2, probe, size=(2, 3))),
            ("unknown keyword", lambda factory, probe: factory(2, probe, unknown=True)),
        )
        for factory_name in ("zeros", "ones"):
            actual_factory = getattr(torch, factory_name)
            expected_factory = getattr(reference_torch, factory_name)
            for case, call in invalid_calls:
                with self.subTest(factory=factory_name, case=case):
                    actual = error_outcome(actual_factory, call)
                    expected = error_outcome(expected_factory, call)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual[0], [])

    def test_mutating_list_dimensions_match_pytorch_2_13(self):
        def create(factory, keyword):
            dimensions = []

            class Probe:
                def __index__(self):
                    dimensions[2] = 4
                    return 2

            dimensions.extend((2, Probe(), 3))
            tensor = factory(size=dimensions) if keyword else factory(dimensions)
            return tensor.shape, dimensions[2]

        for factory_name in ("zeros", "ones"):
            actual_factory = getattr(torch, factory_name)
            expected_factory = getattr(reference_torch, factory_name)
            for keyword in (False, True):
                with self.subTest(factory=factory_name, keyword=keyword):
                    actual = create(actual_factory, keyword)
                    expected = create(expected_factory, keyword)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual[0], (2, 2, 4))

            for keyword in (False, True):
                for mutation in ("pop", "append"):
                    def resize(factory):
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
                        tensor = (
                            factory(size=dimensions)
                            if keyword
                            else factory(dimensions)
                        )
                        return tensor.shape, len(dimensions), events

                    with self.subTest(
                        factory=factory_name,
                        keyword=keyword,
                        mutation=mutation,
                    ):
                        self.assertEqual(
                            resize(actual_factory),
                            resize(expected_factory),
                        )

    def test_stateful_first_dimension_is_reconverted_like_pytorch_2_13(self):
        def outcome(factory, call):
            events = []

            class Probe:
                def __index__(self):
                    events.append("index")
                    return 1 if len(events) == 1 else -1

            try:
                call(factory, Probe())
            except Exception as error:
                return type(error).__name__, str(error), events
            self.fail("stateful negative dimension unexpectedly succeeded")

        forms = (
            ("direct", lambda factory, probe: factory(probe)),
            ("variadic", lambda factory, probe: factory(probe, 3)),
            ("tuple", lambda factory, probe: factory((probe, 3))),
            ("list", lambda factory, probe: factory([probe, 3])),
            ("size tuple", lambda factory, probe: factory(size=(probe, 3))),
            ("size list", lambda factory, probe: factory(size=[probe, 3])),
        )
        for factory_name in ("zeros", "ones"):
            actual_factory = getattr(torch, factory_name)
            expected_factory = getattr(reference_torch, factory_name)
            for form, call in forms:
                with self.subTest(factory=factory_name, form=form):
                    self.assertEqual(
                        outcome(actual_factory, call),
                        outcome(expected_factory, call),
                    )

    def test_size_index_protocol_is_independent_of_operator_module(self):
        import operator

        class Indexable:
            def __index__(self):
                return 3

        def outcome(factory, value):
            try:
                tensor = factory(value)
                return "ok", tensor.shape
            except Exception as error:
                return type(error).__name__, str(error)

        original_index = operator.index
        try:
            for replacement in (
                lambda _: 7,
                lambda _: (_ for _ in ()).throw(RuntimeError("monkeypatched")),
            ):
                operator.index = replacement
                for factory_name in ("zeros", "ones"):
                    actual_factory = getattr(torch, factory_name)
                    expected_factory = getattr(reference_torch, factory_name)
                    for value in (Indexable(), object()):
                        with self.subTest(
                            factory=factory_name,
                            replacement=replacement,
                            value=type(value).__name__,
                        ):
                            self.assertEqual(
                                outcome(actual_factory, value),
                                outcome(expected_factory, value),
                            )
        finally:
            operator.index = original_index

    def test_combined_metadata_and_keyword_errors_match_pytorch_2_13(self):
        cases = (
            {"dtype": "bad", "bogus": True},
            {"bogus": True, "dtype": "bad"},
            {"dtype": "bad", "size": (2,)},
            {"size": (2,), "dtype": "bad"},
            {"device": object(), "bogus": True},
            {"bogus": True, "device": object()},
            {"size": (2,), "bogus": True},
            {"bogus": True, "size": (2,)},
        )
        for factory_name in ("zeros", "ones"):
            actual_factory = getattr(torch, factory_name)
            expected_factory = getattr(reference_torch, factory_name)
            for case, kwargs in enumerate(cases):
                with self.subTest(factory=factory_name, case=case):
                    self.assert_error_matches(
                        lambda kwargs=kwargs: actual_factory(2, 3, **kwargs),
                        lambda kwargs=kwargs: expected_factory(2, 3, **kwargs),
                    )

    def test_empty_metadata_negative_and_overflow_shapes_match(self):
        maximum = sys.maxsize
        metadata_shapes = (
            (maximum, 0),
            (0, maximum),
            (maximum, 0, 2),
            (2, 0, maximum),
        )
        error_shapes = (
            (0, maximum, maximum),
            (maximum // 2 + 1,),
            (1 << 32, 1 << 32),
            (1, 2, maximum),
            (1, 1, 2, maximum),
        )
        for factory_name in ("zeros", "ones"):
            actual_factory = getattr(torch, factory_name)
            expected_factory = getattr(reference_torch, factory_name)
            for shape in metadata_shapes:
                with self.subTest(factory=factory_name, metadata_shape=shape):
                    self.assert_tensor_matches(
                        actual_factory(*shape),
                        expected_factory(*shape),
                        compare_values=False,
                    )
            for shape in error_shapes:
                with self.subTest(factory=factory_name, overflow_shape=shape):
                    self.assert_error_matches(
                        lambda shape=shape: actual_factory(*shape),
                        lambda shape=shape: expected_factory(*shape),
                    )
            for shape in ((-1,), (2, -3, 4), (0, -1)):
                with self.subTest(factory=factory_name, negative_shape=shape):
                    self.assert_error_matches(
                        lambda shape=shape: actual_factory(*shape),
                        lambda shape=shape: expected_factory(*shape),
                    )

    def test_dtype_device_and_legacy_shape_alias(self):
        for factory_name, fill_value in (("zeros", 0.0), ("ones", 1.0)):
            actual_factory = getattr(torch, factory_name)
            expected_factory = getattr(reference_torch, factory_name)
            metadata_calls = (
                (
                    lambda: actual_factory(2, 3, dtype=torch.float32, device="cpu"),
                    lambda: expected_factory(
                        2, 3, dtype=reference_torch.float32, device="cpu"
                    ),
                ),
                (
                    lambda: actual_factory(
                        size=[2, 3], dtype=None, device=torch.device("cpu")
                    ),
                    lambda: expected_factory(
                        size=[2, 3], dtype=None, device=reference_torch.device("cpu")
                    ),
                ),
            )
            for actual_call, expected_call in metadata_calls:
                with self.subTest(factory=factory_name):
                    self.assert_tensor_matches(actual_call(), expected_call())

            legacy = actual_factory(shape=(2, 3), dtype=torch.float32, device="cpu")
            self.assertEqual(legacy.tolist(), [[fill_value] * 3, [fill_value] * 3])
            with self.assertRaises(TypeError):
                actual_factory((2, 3), shape=(2, 3))
            with self.assertRaises(TypeError):
                actual_factory(size=(2, 3), shape=(2, 3))


if __name__ == "__main__":
    unittest.main()
