import copy
import importlib
import inspect
import operator
import pickle
import subprocess
import sys
import textwrap
import unittest
import warnings

import numpy as np
import torch_rs as torch


class IntegerSubclass(int):
    pass


class NumpyIntegerSubclass(np.int64):
    def marker(self):
        return "numpy integer subclass"


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


class integer:
    __module__ = "numpy"


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

        if sys.version_info[:2] == (3, 10):
            self.assertEqual(torch.Size(iterable=(1, 2)), ())
            self.assertEqual(torch.Size([1, 2], ignored=()), (1, 2))
        else:
            with self.assertRaises(TypeError) as raised:
                torch.Size(iterable=())
            self.assertEqual(
                str(raised.exception),
                "tuple() takes no keyword arguments",
            )

        with self.assertRaises(TypeError) as raised:
            torch.Size((), ())
        self.assertEqual(
            str(raised.exception),
            "tuple expected at most 1 argument, got 2",
        )

        with self.assertRaises(AttributeError) as raised:
            empty.extra = 1
        expected_attribute_error = "'torch.Size' object has no attribute 'extra'"
        if sys.version_info >= (3, 13):
            expected_attribute_error += (
                " and no __dict__ for setting new attributes"
            )
        self.assertEqual(
            str(raised.exception).replace("torch_rs.Size", "torch.Size"),
            expected_attribute_error,
        )
        with self.assertRaises(TypeError) as raised:
            operator.setitem(original, 0, 3)
        self.assertEqual(
            str(raised.exception).replace("torch_rs.Size", "torch.Size"),
            "'torch.Size' object does not support item assignment",
        )
        with self.assertRaisesRegex(
            TypeError, "^type 'torch_rs.Size' is not an acceptable base type$"
        ):
            type("SizeSubclass", (torch.Size,), {})

    def test_construction_normalizes_bool_and_defers_boundary_unpacking(self):
        custom = CustomIndex(7)
        integer_subclass = IntegerSubclass(3)
        numpy_integer = np.int64(4)
        numpy_integer_subclass = NumpyIntegerSubclass(5)
        value = torch.Size(
            [
                True,
                False,
                integer_subclass,
                numpy_integer,
                numpy_integer_subclass,
                custom,
            ]
        )

        self.assertEqual(value, (1, 0, 3, 4, 5, 7))
        self.assertIs(type(value[0]), int)
        self.assertIs(type(value[1]), int)
        self.assertIs(value[2], integer_subclass)
        self.assertIs(value[3], numpy_integer)
        self.assertIs(value[4], numpy_integer_subclass)
        self.assertEqual(value[4].marker(), "numpy integer subclass")
        self.assertIs(type(value[5]), int)
        self.assertEqual(custom.calls, 1)
        self.assertEqual(repr(value), "torch.Size([1, 0, 3, 4, 5, 7])")

        boundaries = torch.Size([-(2**63), 0, 2**63 - 1])
        self.assertEqual(
            repr(boundaries),
            "torch.Size([-9223372036854775808, 0, 9223372036854775807])",
        )

        invalid = (
            (1.5, "float"),
            ("1", "str"),
            (None, "NoneType"),
            (BadIndex(), "BadIndex"),
            (integer(), "integer"),
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

        numpy_boolean = np.bool_(True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                indexed_boolean = operator.index(numpy_boolean)
            except TypeError:
                with self.assertRaises(TypeError):
                    torch.Size([numpy_boolean])
            else:
                normalized_boolean = torch.Size([numpy_boolean])
                self.assertEqual(normalized_boolean, (indexed_boolean,))
                self.assertIs(type(normalized_boolean[0]), int)

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

    def test_operator_index_patching_cannot_poison_validation(self):
        script = """
import operator
import torch_rs as torch

original_index = operator.index
operator.index = lambda value: object()
try:
    for value in (1.5, object()):
        try:
            torch.Size([value])
        except TypeError:
            pass
        else:
            raise AssertionError("patched operator.index bypassed validation")
finally:
    operator.index = original_index

for value in (1.5, object()):
    try:
        torch.Size([value])
    except TypeError:
        pass
    else:
        raise AssertionError("Size validation remained poisoned")

assert torch.Size([True, 2]) == (1, 2)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

    def test_numpy_globals_do_not_control_dimension_validation(self):
        original_integer = np.integer

        class MutableIndex:
            def __init__(self):
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return 7

        try:
            np.integer = int
            boolean_size = torch.Size([True])
            self.assertIs(type(boolean_size[0]), int)

            np.integer = MutableIndex
            custom = MutableIndex()
            custom_size = torch.Size([custom])
            self.assertEqual(custom_size, (7,))
            self.assertIs(type(custom_size[0]), int)
            self.assertIsNot(custom_size[0], custom)
            self.assertEqual(custom.calls, 1)
        finally:
            np.integer = original_integer

        spoofed_type = type(
            "numpy.integer",
            (object,),
            {"__index__": lambda self: 9},
        )
        spoofed = spoofed_type()
        spoofed_size = torch.Size([spoofed])
        self.assertEqual(spoofed_size, (9,))
        self.assertIs(type(spoofed_size[0]), int)
        self.assertIsNot(spoofed_size[0], spoofed)
        invalid_spoofed_type = type(
            "numpy.integer",
            (object,),
            {"__index__": lambda self: object()},
        )
        with self.assertRaises(TypeError):
            torch.Size([invalid_spoofed_type()])

        script = """
import sys

class BrokenNumpyImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise RuntimeError("NumPy import was attempted")
        return None

sys.meta_path.insert(0, BrokenNumpyImport())
import torch_rs as torch

value = torch.Size([True, False])
assert value == (1, 0)
assert type(value[0]) is int
assert type(value[1]) is int
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

    def test_numpy_integer_indexing_is_deferred_until_numeric_use(self):
        calls = []

        class DeferredIndex(np.int64):
            def __index__(self):
                calls.append("index")
                raise RuntimeError("deferred NumPy index")

        dimension = DeferredIndex(7)
        value = torch.Size([dimension])
        self.assertIs(value[0], dimension)
        self.assertEqual(calls, [])

        for operation in (lambda: repr(value), value.numel):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(RuntimeError, "^deferred NumPy index$"):
                    operation()
        self.assertEqual(calls, ["index", "index"])

    def test_construction_releases_normalized_dimensions_in_order(self):
        released = []

        class LifecycleDimension:
            def __init__(self, position):
                self.position = position

            def __index__(self):
                if self.position == 0:
                    return 1
                return len(released)

            def __del__(self):
                if self.position == 0:
                    released.append("first")

        def dimensions():
            yield LifecycleDimension(0)
            yield LifecycleDimension(1)

        value = torch.Size(dimensions())
        self.assertEqual(value, (1, 1))
        self.assertEqual(released, ["first"])

    def test_hash_recomputes_retained_item_hashes(self):
        calls = []

        class CountingHash(int):
            def __hash__(self):
                calls.append("hash")
                return super().__hash__()

        value = torch.Size([CountingHash(3)])
        self.assertEqual(hash(value), hash((3,)))
        self.assertEqual(hash(value), hash((3,)))
        self.assertEqual(calls, ["hash", "hash"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux RLIMIT_AS")
    def test_large_size_allocation_failures_raise_memory_error(self):
        script = textwrap.dedent(
            """\
            import os
            import resource
            import sys

            import torch_rs as torch

            mode = sys.argv[1]
            if mode == "construct_oom":
                source = (1,) * 4_000_000
            elif mode == "construct_success":
                source = (1,) * 2_000_000
            else:
                value = torch.Size((-(2**63),) * 1_000_000)

            with open("/proc/self/statm", encoding="ascii") as statm:
                virtual_pages = int(statm.read().split()[0])
            current_virtual_size = virtual_pages * os.sysconf("SC_PAGE_SIZE")
            allowance = {
                "construct_oom": 12,
                "construct_success": 28,
                "repr_reserve": 12,
                "unicode": 40,
            }[mode]
            limit = current_virtual_size + allowance * 1024 * 1024
            _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if hard_limit != resource.RLIM_INFINITY and limit > hard_limit:
                raise SystemExit(77)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard_limit))

            try:
                if mode in {"construct_oom", "construct_success"}:
                    result = torch.Size(source)
                else:
                    repr(value)
            except MemoryError as error:
                if mode == "construct_success":
                    raise AssertionError("in-place construction exhausted memory") from error
                message = str(error)
                if mode == "unicode":
                    assert message != "unable to allocate torch.Size representation", message
            else:
                if mode == "construct_success":
                    assert len(result) == 2_000_000
                    assert result[0] == 1 and result[-1] == 1
                else:
                    raise AssertionError(f"constrained {mode} unexpectedly succeeded")
            """
        )
        for mode in (
            "construct_oom",
            "construct_success",
            "repr_reserve",
            "unicode",
        ):
            with self.subTest(mode=mode):
                completed = subprocess.run(
                    [sys.executable, "-c", script, mode],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=60,
                )
                if completed.returncode == 77:
                    self.skipTest("process hard address-space limit is too low")
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=(
                        f"stdout:\n{completed.stdout}\n"
                        f"stderr:\n{completed.stderr}"
                    ),
                )

    def test_tuple_mutation_protocol_is_not_extended(self):
        value = torch.Size([1, 2])
        self.assertFalse(hasattr(value, "__setitem__"))
        self.assertFalse(hasattr(value, "__delitem__"))

        operations = (
            (operator.setitem, (value, 0, 3), TypeError),
            (operator.delitem, (value, 0), TypeError),
            (object.__setattr__, (value, "extra", 1), AttributeError),
            (object.__delattr__, (value, "extra"), AttributeError),
            (object.__setattr__, (value, "__class__", tuple), TypeError),
            (object.__delattr__, (value, "__class__"), TypeError),
        )
        for operation, arguments, error_type in operations:
            with self.subTest(operation=operation, arguments=arguments[1:]):
                with self.assertRaises(error_type):
                    operation(*arguments)

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

    def test_binary_operators_preserve_reflected_dispatch(self):
        value = torch.Size([1, 2, 3])
        np.testing.assert_array_equal(value + np.array([4]), [5, 6, 7])
        np.testing.assert_array_equal(
            value * np.array([2, 3, 4]), [2, 6, 12]
        )
        np.testing.assert_array_equal(
            np.array([2, 3, 4]) * value, [2, 6, 12]
        )
        self.assertIs(type(value * np.int64(2)), torch.Size)
        self.assertEqual(value * np.int64(2), value * 2)
        self.assertIs(type(np.int64(2) * value), torch.Size)
        self.assertEqual(np.int64(2) * value, value * 2)

        class ReflectedAdd:
            def __init__(self, result):
                self.result = result
                self.calls = 0

            def __radd__(self, other):
                self.calls += 1
                return self.result

        add_marker = object()
        reflected_add = ReflectedAdd(add_marker)
        self.assertIs(value + reflected_add, add_marker)
        self.assertEqual(reflected_add.calls, 1)

        declining_add = ReflectedAdd(NotImplemented)
        with self.assertRaises(TypeError) as raised:
            value + declining_add
        self.assertEqual(
            str(raised.exception),
            "can only concatenate tuple (not ReflectedAdd) to torch.Size",
        )
        self.assertEqual(declining_add.calls, 1)

        class ReflectedMultiply:
            def __init__(self, result):
                self.result = result
                self.reflected_calls = 0
                self.index_calls = 0

            def __rmul__(self, other):
                self.reflected_calls += 1
                return self.result

            def __index__(self):
                self.index_calls += 1
                return 2

        multiply_marker = object()
        reflected_multiply = ReflectedMultiply(multiply_marker)
        self.assertIs(value * reflected_multiply, multiply_marker)
        self.assertEqual(reflected_multiply.reflected_calls, 1)
        self.assertEqual(reflected_multiply.index_calls, 0)

        declining_multiply = ReflectedMultiply(NotImplemented)
        self.assertEqual(value * declining_multiply, value * 2)
        self.assertEqual(declining_multiply.reflected_calls, 1)
        self.assertEqual(declining_multiply.index_calls, 1)

        class LeftMultiply:
            def __init__(self):
                self.multiply_calls = 0
                self.index_calls = 0

            def __mul__(self, other):
                self.multiply_calls += 1
                return NotImplemented

            def __index__(self):
                self.index_calls += 1
                return 2

        left_multiply = LeftMultiply()
        self.assertEqual(left_multiply * value, value * 2)
        self.assertEqual(left_multiply.multiply_calls, 1)
        self.assertEqual(left_multiply.index_calls, 1)

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

    def test_public_type_is_immutable_and_stable_across_reinitialization(self):
        original_type = torch.Size
        original = torch.Size([1, 2, 3])
        original_attributes = {
            name: getattr(original_type, name)
            for name in ("__new__", "__repr__", "numel")
        }
        for name in original_attributes:
            for operation in (
                lambda name=name: setattr(original_type, name, None),
                lambda name=name: type.__setattr__(
                    original_type, name, None
                ),
                lambda name=name: delattr(original_type, name),
                lambda name=name: type.__delattr__(original_type, name),
            ):
                with self.subTest(name=name, operation=operation):
                    with self.assertRaises(TypeError) as raised:
                        operation()
                    self.assertEqual(
                        str(raised.exception),
                        f"cannot set '{name}' attribute of immutable type "
                        "'torch_rs.Size'",
                    )
            self.assertIs(getattr(original_type, name), original_attributes[name])

        saved_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == "torch_rs" or name.startswith("torch_rs.")
        }
        try:
            for name in saved_modules:
                sys.modules.pop(name, None)
            reinitialized = importlib.import_module("torch_rs")
            self.assertIs(reinitialized.Size, original_type)
            restored = pickle.loads(pickle.dumps(original))
            self.assertIs(type(restored), original_type)
            self.assertEqual(restored, original)
        finally:
            for name in tuple(sys.modules):
                if name == "torch_rs" or name.startswith("torch_rs."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

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
