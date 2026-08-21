import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class OuterReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.outer differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    @staticmethod
    def make_cases(module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(4, 6).tolist(),
            dtype=module.float32,
        )
        special_left_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special_right_bits = np.asarray(
            (0xBF80_0000, 0x0000_0000, 0x8000_0000, 0x3F00_0000),
            dtype=np.uint32,
        )
        return (
            ("contiguous", module.tensor([1.0, 2.0, 3.0]), module.tensor([4.0, 5.0])),
            ("offset", base[2], base[1]),
            ("strided", base.transpose(0, 1)[1], base.transpose(0, 1)[4]),
            ("empty left", module.zeros((3, 0))[1], module.tensor([1.0, 2.0])),
            ("empty right", module.tensor([1.0, 2.0]), module.zeros((3, 0))[2]),
            ("both empty", module.zeros((2, 0))[1], module.zeros((3, 0))[2]),
            (
                "non-finite",
                module.tensor(memoryview(special_left_bits.view(np.float32))),
                module.tensor(memoryview(special_right_bits.view(np.float32))),
            ),
        )

    @staticmethod
    def call_outer(module, left, right, form):
        if form == "positional":
            return module.outer(left, right)
        if form == "keywords":
            return module.outer(input=left, vec2=right)
        if form == "x alias":
            return module.outer(x=left, vec2=right)
        if form == "a alias":
            return module.outer(a=left, vec2=right)
        if form == "x1 alias":
            return module.outer(x1=left, vec2=right)
        if form == "out none":
            return module.outer(left, right, out=None)
        raise AssertionError(f"unknown call form: {form}")

    def test_values_strides_empties_offsets_and_non_finites_match_pytorch_2_13(self):
        forms = ("positional", "keywords", "x alias", "a alias", "x1 alias", "out none")
        for actual_case, expected_case in zip(
            self.make_cases(torch), self.make_cases(reference_torch), strict=True
        ):
            name, actual_left, actual_right = actual_case
            expected_name, expected_left, expected_right = expected_case
            self.assertEqual(name, expected_name)
            for form in forms:
                self.assert_matches(
                    self.call_outer(torch, actual_left, actual_right, form),
                    self.call_outer(reference_torch, expected_left, expected_right, form),
                    case=(name, form),
                )

    def test_paired_nan_payloads_match_pytorch_2_13_across_lanes(self):
        cases = (
            ("single lane", (0x7FC1_2345,), (0xFFC5_4321,), 1, 1),
            ("two quiet lanes", (0x7FC1_2345,), (0xFFC5_4321,) * 2, 1, 1),
            ("two signaling lanes", (0x7F81_2345,), (0xFF85_4321,) * 2, 1, 1),
            (
                "scalar tail",
                (0x7FC1_2345, 0x7F81_2345),
                (0xFFC5_4321,) * 4,
                1,
                1,
            ),
            ("vector boundary", (0x7FC1_2345,), (0xFF85_4321,) * 16, 1, 1),
            ("post-vector tail", (0x7F81_2345,), (0xFFC5_4321,) * 18, 1, 1),
            ("single column", (0x7FC1_2345,) * 6, (0xFF85_4321,), 1, 1),
            (
                "strided vec2",
                (0x7FC1_2345, 0x7F81_2345),
                (0xFFC5_4321, 0xFF85_4321) * 3,
                1,
                2,
            ),
            ("strided single column", (0x7FC1_2345,) * 6, (0xFF85_4321,), 2, 1),
        )

        def make_vector(module, values, stride):
            bits = np.full(len(values) * stride, 0x3F80_0000, dtype=np.uint32)
            bits[::stride] = values
            vector = module.tensor(memoryview(bits.view(np.float32)))
            if stride == 1:
                return vector
            return vector.reshape((len(values), stride)).transpose(0, 1)[0]

        for name, left_values, right_values, left_stride, right_stride in cases:
            actual = torch.outer(
                make_vector(torch, left_values, left_stride),
                make_vector(torch, right_values, right_stride),
            )
            expected = reference_torch.outer(
                make_vector(reference_torch, left_values, left_stride),
                make_vector(reference_torch, right_values, right_stride),
            )
            self.assert_matches(actual, expected, case=name)

    @staticmethod
    def make_autograd_inputs(module):
        left_leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        right_leaf = module.tensor(
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], requires_grad=True
        )
        return (
            left_leaf,
            left_leaf.transpose(0, 1)[1],
            right_leaf,
            right_leaf.transpose(0, 1)[1],
        )

    def test_autograd_empties_shared_operands_and_no_grad_match_pytorch_2_13(self):
        actual_left_leaf, actual_left, actual_right_leaf, actual_right = (
            self.make_autograd_inputs(torch)
        )
        expected_left_leaf, expected_left, expected_right_leaf, expected_right = (
            self.make_autograd_inputs(reference_torch)
        )
        actual = torch.outer(actual_left, actual_right)
        expected = reference_torch.outer(expected_left, expected_right)
        self.assert_matches(actual, expected, case="tracked strided views")
        actual.sum().backward()
        expected.sum().backward()
        self.assert_matches(
            actual_left_leaf.grad, expected_left_leaf.grad, case="left gradient"
        )
        self.assert_matches(
            actual_right_leaf.grad, expected_right_leaf.grad, case="right gradient"
        )

        actual_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_shared = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        torch.outer(actual_shared, actual_shared).sum().backward()
        reference_torch.outer(expected_shared, expected_shared).sum().backward()
        self.assert_matches(
            actual_shared.grad, expected_shared.grad, case="shared operand gradient"
        )

        actual_empty = torch.zeros((0,), requires_grad=True)
        expected_empty = reference_torch.zeros((0,), requires_grad=True)
        torch.outer(actual_empty, torch.tensor([2.0, 3.0])).sum().backward()
        reference_torch.outer(
            expected_empty, reference_torch.tensor([2.0, 3.0])
        ).sum().backward()
        self.assert_matches(actual_empty.grad, expected_empty.grad, case="empty gradient")

        actual_left = torch.tensor([1.0, 2.0], requires_grad=True)
        actual_right = torch.tensor([3.0, 4.0], requires_grad=True)
        expected_left = reference_torch.tensor([1.0, 2.0], requires_grad=True)
        expected_right = reference_torch.tensor([3.0, 4.0], requires_grad=True)
        with torch.no_grad():
            actual = torch.outer(actual_left, actual_right)
        with reference_torch.no_grad():
            expected = reference_torch.outer(expected_left, expected_right)
        self.assert_matches(actual, expected, case="no_grad")

    @staticmethod
    def dispatch_observation(module):
        left = module.tensor([1.0, 2.0])
        right = module.tensor([3.0])
        matrix = module.ones((1, 1))
        destination = module.zeros((2, 1))
        function = module.outer
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_observations = []
        for call, keywords in (
            (lambda: function(left, right), None),
            (lambda: function(input=left, vec2=right), ("input", "vec2")),
            (lambda: function(a=left, vec2=right, out=None), ("a", "vec2", "out")),
            (lambda: function(matrix, right), None),
            (lambda: function(left, right, out=destination), ("out",)),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keywords,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call in (
            lambda value: function(value, right),
            lambda value: function(left, value),
            lambda value: function(left, right, out=value),
            lambda value: function(input=left, vec2=value, out=None),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                )
            )

        subclass_events = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_events.append(("base", tuple(item.__name__ for item in types)))
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_events.append(
                    ("derived", tuple(item.__name__ for item in types))
                )
                return marker

        subclass_result = function(BaseOverride(), right, out=DerivedOverride())

        invalid_observations = []
        for call in (
            lambda: function([], right),
            lambda: function(left, []),
            lambda: function(left, right, out=[]),
            lambda: function(left, right, extra=True),
        ):
            mode = RecordingMode()
            try:
                with mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(mode.calls))
                )

        return (
            mode_observations,
            override_observations,
            subclass_result is marker,
            subclass_events,
            invalid_observations,
        )

    def test_override_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_rank_and_binding_errors_match_pytorch_2_13(self):
        actual_vector = torch.tensor([1.0, 2.0])
        expected_vector = reference_torch.tensor([1.0, 2.0])
        cases = (
            (lambda: torch.outer(), lambda: reference_torch.outer()),
            (lambda: torch.outer(actual_vector), lambda: reference_torch.outer(expected_vector)),
            (
                lambda: torch.outer(actual_vector, actual_vector, actual_vector),
                lambda: reference_torch.outer(expected_vector, expected_vector, expected_vector),
            ),
            (
                lambda: torch.outer([], actual_vector),
                lambda: reference_torch.outer([], expected_vector),
            ),
            (
                lambda: torch.outer(actual_vector, []),
                lambda: reference_torch.outer(expected_vector, []),
            ),
            (
                lambda: torch.outer(input=None, vec2=actual_vector),
                lambda: reference_torch.outer(input=None, vec2=expected_vector),
            ),
            (
                lambda: torch.outer(actual_vector, actual_vector, input=actual_vector),
                lambda: reference_torch.outer(
                    expected_vector, expected_vector, input=expected_vector
                ),
            ),
            (
                lambda: torch.outer(actual_vector, actual_vector, vec2=actual_vector),
                lambda: reference_torch.outer(
                    expected_vector, expected_vector, vec2=expected_vector
                ),
            ),
            (
                lambda: torch.outer(input=actual_vector, x2=actual_vector),
                lambda: reference_torch.outer(input=expected_vector, x2=expected_vector),
            ),
            (
                lambda: torch.outer(actual_vector, actual_vector, extra=True),
                lambda: reference_torch.outer(expected_vector, expected_vector, extra=True),
            ),
            (
                lambda: torch.outer(actual_vector, actual_vector, out=[]),
                lambda: reference_torch.outer(expected_vector, expected_vector, out=[]),
            ),
            (
                lambda: torch.outer(torch.tensor(1.0), actual_vector),
                lambda: reference_torch.outer(reference_torch.tensor(1.0), expected_vector),
            ),
            (
                lambda: torch.outer(actual_vector, torch.ones((1, 1))),
                lambda: reference_torch.outer(expected_vector, reference_torch.ones((1, 1))),
            ),
            (
                lambda: torch.outer(
                    torch.ones((1, 1)), actual_vector, out=torch.zeros((1, 2))
                ),
                lambda: reference_torch.outer(
                    reference_torch.ones((1, 1)),
                    expected_vector,
                    out=reference_torch.zeros((1, 2)),
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    @staticmethod
    def callable_contract(module):
        function = module.outer
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.outer is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("outer"),
            "wildcard_identity": wildcard_namespace["outer"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
