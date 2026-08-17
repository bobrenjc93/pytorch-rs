import copy
import inspect
import pickle
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SizeValueReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Size differentials require pinned PyTorch 2.13.0"
            )

    def outcome(self, action):
        try:
            value = action()
            contract = self.value_contract(value)
        except Exception as error:
            return "error", type(error).__name__, str(error)
        return "return", contract

    def scalar_outcome(self, action):
        try:
            value = action()
        except Exception as error:
            return "error", type(error).__name__, str(error)
        return "return", type(value).__name__, value

    def value_contract(self, value):
        if isinstance(value, np.ndarray):
            return type(value).__name__, value.tolist()
        if isinstance(value, tuple):
            return (
                type(value).__name__,
                tuple(value),
                tuple(type(item).__name__ for item in value),
                repr(value),
                str(value),
            )
        return type(value).__name__, value

    def construction_contract(self, module):
        custom = CustomIndex(7)
        integer_subclass = IntegerSubclass(3)
        numpy_integer = np.int64(4)
        numpy_integer_subclass = NumpyIntegerSubclass(5)
        constructors = (
            lambda: module.Size(),
            lambda: module.Size(()),
            lambda: module.Size(range(3)),
            lambda: module.Size(iter([1, 2])),
            lambda: module.Size(
                [
                    True,
                    False,
                    integer_subclass,
                    numpy_integer,
                    numpy_integer_subclass,
                    custom,
                ]
            ),
            lambda: module.Size([-(2**63), 0, 2**63 - 1]),
            lambda: module.Size("12"),
            lambda: module.Size([np.bool_(True)]),
            lambda: module.Size([BadIndex()]),
            lambda: module.Size([integer()]),
            lambda: module.Size([2**63]),
            lambda: module.Size([-(2**63) - 1]),
            lambda: module.Size(iterable=()),
            lambda: module.Size((), ()),
        )
        results = tuple(self.outcome(constructor) for constructor in constructors)
        return {
            "signature": str(inspect.signature(module.Size)),
            "tuple_subtype": issubclass(module.Size, tuple),
            "in_all": module.__all__.count("Size"),
            "native_identity": module._C.Size is module.Size,
            "results": results,
            "custom_calls": custom.calls,
        }

    def test_construction_namespace_and_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.construction_contract(torch),
            self.construction_contract(reference_torch),
        )

    def operation_contract(self, module):
        value = module.Size([1, 2, 3])
        repeat = CustomIndex(2)

        class ReflectedAdd:
            def __init__(self, result):
                self.result = result
                self.calls = 0

            def __radd__(self, other):
                self.calls += 1
                return self.result

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

        add_marker = object()
        reflected_add = ReflectedAdd(add_marker)
        declining_add = ReflectedAdd(NotImplemented)
        multiply_marker = object()
        reflected_multiply = ReflectedMultiply(multiply_marker)
        declining_multiply = ReflectedMultiply(NotImplemented)
        left_multiply = LeftMultiply()
        operations = (
            lambda: value[0],
            lambda: value[:],
            lambda: value[1:],
            lambda: value[::-1],
            lambda: value + (4,),
            lambda: (0,) + value,
            lambda: value + module.Size([4]),
            lambda: value + [4],
            lambda: value * 0,
            lambda: value * -1,
            lambda: value * 2,
            lambda: 2 * value,
            lambda: value * repeat,
            lambda: value * 1.5,
            lambda: value * (2**100),
            lambda: value + np.array([4]),
            lambda: value * np.array([2, 3, 4]),
            lambda: np.array([2, 3, 4]) * value,
            lambda: value * np.int64(2),
            lambda: np.int64(2) * value,
            lambda: value + reflected_add is add_marker,
            lambda: value + declining_add,
            lambda: value * reflected_multiply is multiply_marker,
            lambda: value * declining_multiply,
            lambda: left_multiply * value,
        )
        results = tuple(self.outcome(operation) for operation in operations)
        return {
            "results": results,
            "repeat_calls": repeat.calls,
            "reflected_add_calls": reflected_add.calls,
            "declining_add_calls": declining_add.calls,
            "reflected_multiply_calls": (
                reflected_multiply.reflected_calls,
                reflected_multiply.index_calls,
            ),
            "declining_multiply_calls": (
                declining_multiply.reflected_calls,
                declining_multiply.index_calls,
            ),
            "left_multiply_calls": (
                left_multiply.multiply_calls,
                left_multiply.index_calls,
            ),
        }

    def test_slicing_concatenation_and_repetition_match_pytorch_2_13(self):
        self.assertEqual(
            self.operation_contract(torch),
            self.operation_contract(reference_torch),
        )

    def numeric_contract(self, module):
        values = (
            module.Size(),
            module.Size([0]),
            module.Size([True, 7]),
            module.Size([2, 3, 4]),
            module.Size([2**31, 2**32]),
            module.Size([2**62, 4]),
            module.Size([2**63 - 1, 2]),
            module.Size([-(2**63), -1]),
        )
        return tuple(
            (
                value.numel(),
                value == tuple(value),
                tuple(value) == value,
                hash(value) == hash(tuple(value)),
                self.outcome(lambda value=value: value.numel(1)),
            )
            for value in values
        )

    def test_numel_equality_hashing_and_boundaries_match_pytorch_2_13(self):
        self.assertEqual(
            self.numeric_contract(torch),
            self.numeric_contract(reference_torch),
        )

    def pickle_contract(self, module):
        values = (
            module.Size(),
            module.Size([0]),
            module.Size([True, 2**63 - 1, -(2**63)]),
            module.Size([2**63]),
            module.Size([-(2**63) - 1]),
        )
        results = []
        for original in values:
            reduction = original.__reduce__()
            reduce_ex = tuple(
                original.__reduce_ex__(protocol)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            )
            restored = (
                copy.copy(original),
                copy.deepcopy(original),
                *(
                    pickle.loads(pickle.dumps(original, protocol=protocol))
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ),
            )
            results.append(
                {
                    "reduce": (
                        reduction[0] is module.Size,
                        reduction[1],
                    ),
                    "reduce_ex": tuple(
                        (constructor is module.Size, arguments)
                        for constructor, arguments in reduce_ex
                    ),
                    "restored": tuple(
                        (
                            value is original,
                            type(value) is module.Size,
                            value == original,
                            tuple(value),
                            self.scalar_outcome(lambda value=value: repr(value)),
                            hash(value) == hash(original),
                        )
                        for value in restored
                    ),
                }
            )
        return tuple(results)

    def test_reduction_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(
            self.pickle_contract(torch),
            self.pickle_contract(reference_torch),
        )

    def immutability_contract(self, module):
        results = []
        for name in ("__new__", "__repr__", "numel"):
            original = getattr(module.Size, name)
            operations = (
                lambda name=name: setattr(module.Size, name, None),
                lambda name=name: type.__setattr__(module.Size, name, None),
                lambda name=name: delattr(module.Size, name),
                lambda name=name: type.__delattr__(module.Size, name),
            )
            outcomes = []
            for operation in operations:
                try:
                    operation()
                except Exception as error:
                    outcomes.append(
                        (
                            type(error).__name__,
                            str(error).replace("torch_rs.Size", "torch.Size"),
                        )
                    )
                else:
                    outcomes.append(None)
            results.append(
                (
                    name,
                    tuple(outcomes),
                    getattr(module.Size, name) is original,
                )
            )
        return tuple(results)

    def test_public_type_immutability_matches_pytorch_2_13(self):
        self.assertEqual(
            self.immutability_contract(torch),
            self.immutability_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
