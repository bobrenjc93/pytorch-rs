import copy
import inspect
import operator
import pickle
import unittest

import numpy as np
import torch_rs as torch


class IntegerSubclass(int):
    pass


class CustomIndex:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


class BadIndex:
    def __index__(self):
        raise RuntimeError("not an integer")


class SizeValueTests(unittest.TestCase):
    def test_public_type_is_a_positional_only_immutable_tuple_subtype(self):
        self.assertIs(torch.Size, torch._C.Size)
        self.assertEqual(torch.__all__.count("Size"), 1)
        self.assertEqual(torch._C.__all__.count("Size"), 1)
        self.assertEqual(str(inspect.signature(torch.Size)), "(iterable=(), /)")
        self.assertTrue(issubclass(torch.Size, tuple))

        empty = torch.Size()
        self.assertIs(type(empty), torch.Size)
        self.assertEqual(tuple(empty), ())
        self.assertEqual(repr(empty), "torch.Size([])")
        self.assertEqual(str(empty), "torch.Size([])")

        original = torch.Size([1, 2])
        clone = torch.Size(original)
        self.assertIsNot(clone, original)
        self.assertEqual(clone, original)

        for call, message in (
            (
                lambda: torch.Size(iterable=()),
                "tuple() takes no keyword arguments",
            ),
            (
                lambda: torch.Size((), ()),
                "tuple expected at most 1 argument, got 2",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        with self.assertRaisesRegex(
            AttributeError, "^'torch.Size' object has no attribute 'extra'$"
        ):
            empty.extra = 1
        with self.assertRaisesRegex(
            TypeError, "^'torch.Size' object does not support item assignment$"
        ):
            operator.setitem(original, 0, 3)
        with self.assertRaisesRegex(
            TypeError, "^type 'Size' is not an acceptable base type$"
        ):
            type("SizeSubclass", (torch.Size,), {})

    def test_construction_normalizes_bool_and_defers_boundary_unpacking(self):
        custom = CustomIndex(7)
        integer_subclass = IntegerSubclass(3)
        numpy_integer = np.int64(4)
        value = torch.Size(
            [True, False, integer_subclass, numpy_integer, custom]
        )

        self.assertEqual(value, (1, 0, 3, 4, 7))
        self.assertIs(type(value[0]), int)
        self.assertIs(type(value[1]), int)
        self.assertIs(value[2], integer_subclass)
        self.assertIs(value[3], numpy_integer)
        self.assertIs(type(value[4]), int)
        self.assertEqual(custom.calls, 1)
        self.assertEqual(repr(value), "torch.Size([1, 0, 3, 4, 7])")

        boundaries = torch.Size([-(2**63), 0, 2**63 - 1])
        self.assertEqual(
            repr(boundaries),
            "torch.Size([-9223372036854775808, 0, 9223372036854775807])",
        )

        invalid = (
            (np.bool_(True), "numpy.bool"),
            (1.5, "float"),
            ("1", "str"),
            (None, "NoneType"),
            (BadIndex(), "BadIndex"),
        )
        for item, type_name in invalid:
            with self.subTest(item=repr(item)):
                with self.assertRaises(TypeError) as raised:
                    torch.Size([0, item])
                self.assertEqual(
                    str(raised.exception),
                    "torch.Size() takes an iterable of 'int' "
                    f"(item 1 is '{type_name}')",
                )

        for item in (2**63, -(2**63) - 1, 2**100, np.uint64(2**63)):
            with self.subTest(item=repr(item)):
                value = torch.Size([item])
                self.assertIs(value[0], item)
                self.assertEqual(tuple(value), (item,))
                self.assertEqual(value, (item,))
                self.assertEqual(hash(value), hash((item,)))
                self.assertEqual(value.__reduce__(), (torch.Size, ((item,),)))
                for operation in (lambda: repr(value), value.numel):
                    with self.assertRaises(ValueError) as raised:
                        operation()
                    self.assertEqual(
                        str(raised.exception),
                        "Overflow when unpacking long long",
                    )

    def test_slicing_concatenation_and_repetition_preserve_size(self):
        value = torch.Size([1, 2, 3])
        results = (
            value[:],
            value[1:],
            value[::-1],
            value + (4,),
            (0,) + value,
            value + torch.Size([4]),
            value * 2,
            2 * value,
            value * 0,
            value * -1,
        )
        expected = (
            (1, 2, 3),
            (2, 3),
            (3, 2, 1),
            (1, 2, 3, 4),
            (0, 1, 2, 3),
            (1, 2, 3, 4),
            (1, 2, 3, 1, 2, 3),
            (1, 2, 3, 1, 2, 3),
            (),
            (),
        )
        for result, expected_value in zip(results, expected, strict=True):
            self.assertIs(type(result), torch.Size)
            self.assertEqual(result, expected_value)

        repeat = CustomIndex(2)
        self.assertEqual(value * repeat, value * 2)
        self.assertEqual(repeat.calls, 1)

        with self.assertRaises(TypeError) as raised:
            value + [4]
        self.assertEqual(
            str(raised.exception),
            "can only concatenate tuple (not list) to torch.Size",
        )
        with self.assertRaises(TypeError) as raised:
            value * 1.5
        self.assertEqual(
            str(raised.exception),
            "can't multiply sequence by non-int of type 'float'",
        )

    def test_numel_equality_hashing_and_signed_overflow_match_tuple_contract(self):
        cases = (
            (torch.Size(), 1),
            (torch.Size([0]), 0),
            (torch.Size([True, 7]), 7),
            (torch.Size([2, 3, 4]), 24),
            (torch.Size([2**31, 2**32]), -(2**63)),
            (torch.Size([2**62, 4]), 0),
            (torch.Size([2**63 - 1, 2]), -2),
            (torch.Size([-(2**63), -1]), -(2**63)),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(value.numel(), expected)
                self.assertIs(type(value.numel()), int)
                self.assertEqual(value, tuple(value))
                self.assertEqual(tuple(value), value)
                self.assertEqual(hash(value), hash(tuple(value)))

        with self.assertRaises(TypeError) as raised:
            torch.Size([1]).numel(1)
        self.assertIn("takes", str(raised.exception))

    def test_reduction_copy_and_all_pickle_protocols_reconstruct_size(self):
        for original in (
            torch.Size(),
            torch.Size([0]),
            torch.Size([True, 2**63 - 1, -(2**63)]),
            torch.Size([2**63]),
            torch.Size([-(2**63) - 1]),
        ):
            expected_reduction = (torch.Size, (tuple(original),))
            with self.subTest(original=tuple(original)):
                self.assertEqual(original.__reduce__(), expected_reduction)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertEqual(
                        original.__reduce_ex__(protocol), expected_reduction
                    )
                    restored = pickle.loads(
                        pickle.dumps(original, protocol=protocol)
                    )
                    self.assertIsNot(restored, original)
                    self.assertIs(type(restored), torch.Size)
                    self.assertEqual(restored, original)
                    self.assertEqual(hash(restored), hash(original))

                for restored in (copy.copy(original), copy.deepcopy(original)):
                    self.assertIsNot(restored, original)
                    self.assertIs(type(restored), torch.Size)
                    self.assertEqual(restored, original)

    def test_tensor_metadata_return_types_remain_unchanged(self):
        tensor = torch.zeros((2, 3))
        self.assertIs(type(tensor.shape), tuple)
        self.assertIs(type(tensor.stride()), tuple)
        self.assertIs(type(tensor.size(0)), int)
        with self.assertRaises(TypeError) as raised:
            tensor.size()
        self.assertEqual(
            str(raised.exception),
            'size() missing 1 required positional arguments: "dim"',
        )


if __name__ == "__main__":
    unittest.main()
