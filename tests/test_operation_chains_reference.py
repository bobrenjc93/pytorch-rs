import unittest
from typing import NamedTuple

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SEED = 0xC205_0213
CHAIN_COUNT = 12
MAX_CHAIN_OPERATIONS = 7


class ChainSpec(NamedTuple):
    input_kind: str
    leaf_values: np.ndarray
    view_name: str
    view_dims: tuple[int, int] | None
    first_unary: str
    multiplier_values: np.ndarray
    multiply_form: str
    second_unary: str

    def describe(self):
        view = (
            f"{self.view_name}()"
            if self.view_dims is None
            else f"{self.view_name}{self.view_dims}"
        )
        return (
            f"{self.input_kind} -> {view} -> {self.first_unary}() -> "
            f"{self.multiply_form}(broadcast_shape={self.multiplier_values.shape}) -> "
            f"{self.second_unary}() -> sum() -> backward()"
        )

    def operation_count(self):
        input_view = self.input_kind in {"offset", "transposed"}
        return 6 + int(input_view)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class OperationChainReferenceTests(unittest.TestCase):
    INPUT_SHAPES = {
        "scalar": (),
        "empty": (2, 0, 3),
        "offset": (3, 4),
        "transposed": (4, 3, 2),
    }
    LEAF_SHAPES = {
        "scalar": (),
        "empty": (2, 0, 3),
        "offset": (2, 3, 4),
        "transposed": (2, 3, 4),
    }

    @staticmethod
    def balanced_choices(rng, choices):
        selected = [choices[index % len(choices)] for index in range(CHAIN_COUNT)]
        rng.shuffle(selected)
        return selected

    @staticmethod
    def encoded_dimension(rng, dimension, rank):
        return dimension - rank if rng.integers(0, 2) else dimension

    @classmethod
    def view_spec(cls, rng, input_shape, view_name):
        if view_name == "squeeze":
            return None, tuple(size for size in input_shape if size != 1)

        rank = len(input_shape)
        if rank == 0:
            return (0, -1), ()
        if rank == 1:
            return (0, -1), input_shape

        first, second = (
            int(value) for value in rng.choice(rank, size=2, replace=False)
        )
        dimensions = (
            cls.encoded_dimension(rng, first, rank),
            cls.encoded_dimension(rng, second, rank),
        )
        output_shape = list(input_shape)
        output_shape[first], output_shape[second] = (
            output_shape[second],
            output_shape[first],
        )
        return dimensions, tuple(output_shape)

    @staticmethod
    def multiplier_shape(rng, input_shape):
        if not input_shape:
            return (int(rng.integers(2, 5)),)

        shape = [1 if size == 0 else size for size in input_shape]
        for axis, size in enumerate(input_shape):
            if size != 1 and rng.integers(0, 2):
                shape[axis] = 1
        if tuple(shape) == input_shape:
            axis = next(
                axis for axis, size in enumerate(input_shape) if size != 1
            )
            shape[axis] = 1
        return tuple(shape)

    @classmethod
    def generate_chains(cls):
        rng = np.random.default_rng(SEED)
        input_kinds = cls.balanced_choices(
            rng, ("scalar", "empty", "offset", "transposed")
        )
        views = cls.balanced_choices(rng, ("transpose", "swapdims", "squeeze"))
        first_unaries = cls.balanced_choices(rng, ("neg", "sin", "relu"))
        multiply_forms = cls.balanced_choices(
            rng, ("operator_mul", "mul", "multiply")
        )
        next_unary = {"neg": "sin", "sin": "relu", "relu": "neg"}

        chains = []
        for input_kind, view_name, first_unary, multiply_form in zip(
            input_kinds, views, first_unaries, multiply_forms
        ):
            leaf_shape = cls.LEAF_SHAPES[input_kind]
            if 0 in leaf_shape:
                leaf_values = np.empty(leaf_shape, dtype=np.float32)
            else:
                leaf_values = rng.uniform(-0.75, 0.75, size=leaf_shape).astype(
                    np.float32
                )

            view_dims, view_shape = cls.view_spec(
                rng, cls.INPUT_SHAPES[input_kind], view_name
            )
            factor_shape = cls.multiplier_shape(rng, view_shape)
            magnitudes = rng.uniform(0.25, 1.25, size=factor_shape).astype(
                np.float32
            )
            signs = rng.choice(
                np.asarray([-1.0, 1.0], dtype=np.float32), factor_shape
            )
            multiplier_values = magnitudes * signs
            chains.append(
                ChainSpec(
                    input_kind=input_kind,
                    leaf_values=leaf_values,
                    view_name=view_name,
                    view_dims=view_dims,
                    first_unary=first_unary,
                    multiplier_values=multiplier_values,
                    multiply_form=multiply_form,
                    second_unary=next_unary[first_unary],
                )
            )
        return chains

    @staticmethod
    def make_tensor(module, values, *, requires_grad):
        if values.size == 0:
            return module.zeros(tuple(values.shape), requires_grad=requires_grad)
        data = values.item() if values.shape == () else values.tolist()
        return module.tensor(data, requires_grad=requires_grad)

    @classmethod
    def make_input(cls, module, chain):
        leaf = cls.make_tensor(module, chain.leaf_values, requires_grad=True)
        if chain.input_kind == "offset":
            return leaf, leaf[1]
        if chain.input_kind == "transposed":
            return leaf, leaf.transpose(0, 2)
        return leaf, leaf

    @staticmethod
    def apply_view(tensor, chain):
        if chain.view_dims is None:
            return getattr(tensor, chain.view_name)()
        return getattr(tensor, chain.view_name)(*chain.view_dims)

    @staticmethod
    def apply_multiply(tensor, multiplier, form):
        if form == "operator_mul":
            return tensor * multiplier
        return getattr(tensor, form)(multiplier)

    @classmethod
    def run_forward(cls, module, chain):
        leaf, input_tensor = cls.make_input(module, chain)
        view = cls.apply_view(input_tensor, chain)
        first_unary = getattr(view, chain.first_unary)()
        multiplier = cls.make_tensor(
            module, chain.multiplier_values, requires_grad=True
        )
        multiplied = cls.apply_multiply(
            first_unary, multiplier, chain.multiply_form
        )
        second_unary = getattr(multiplied, chain.second_unary)()
        loss = second_unary.sum()
        return {
            "leaf": leaf,
            "input": input_tensor,
            "view": view,
            "first_unary": first_unary,
            "multiplier": multiplier,
            "multiplied": multiplied,
            "second_unary": second_unary,
            "loss": loss,
        }

    def assert_tensor_matches(self, actual, expected, *, stage):
        with self.subTest(stage=stage, comparison="metadata"):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertEqual(actual.numel(), expected.numel())
            self.assertEqual(actual.ndim, expected.ndim)
            self.assertIs(actual.dtype, torch.float32)
            self.assertIs(expected.dtype, reference_torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertEqual(expected.device, reference_torch.device("cpu"))
            self.assertIs(actual.layout, torch.strided)
            self.assertIs(expected.layout, reference_torch.strided)

        with self.subTest(stage=stage, comparison="values"):
            np.testing.assert_allclose(
                np.asarray(actual, dtype=np.float32),
                expected.detach().cpu().numpy(),
                rtol=5.0e-6,
                atol=2.0e-6,
                equal_nan=True,
            )

    def assert_storage_alias_matches(
        self, actual, expected, actual_source, expected_source, *, stage
    ):
        def storage_start(tensor):
            return tensor.data_ptr() - tensor.storage_offset() * tensor.element_size()

        with self.subTest(stage=stage, comparison="view alias"):
            expected_alias = storage_start(expected) == storage_start(expected_source)
            actual_alias = storage_start(actual) == storage_start(actual_source)
            self.assertTrue(expected_alias)
            self.assertEqual(actual_alias, expected_alias)

    def assert_input_kind(self, actual, expected, kind):
        with self.subTest(stage="input", input_kind=kind):
            if kind == "scalar":
                self.assertEqual(actual.ndim, 0)
            elif kind == "empty":
                self.assertEqual(actual.numel(), 0)
            elif kind == "offset":
                self.assertGreater(actual.storage_offset(), 0)
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
            elif kind == "transposed":
                self.assertFalse(actual.is_contiguous())
                self.assertEqual(actual.stride(), expected.stride())

    def test_fixed_seed_bounded_cross_operation_chains_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        chains = self.generate_chains()
        self.assertEqual(len(chains), CHAIN_COUNT)
        self.assertEqual(
            {chain.input_kind for chain in chains}, set(self.INPUT_SHAPES)
        )

        for chain_index, chain in enumerate(chains):
            description = f"{chain_index}: {chain.describe()}"
            with self.subTest(seed=f"{SEED:#x}", chain=description):
                self.assertLessEqual(
                    chain.operation_count(), MAX_CHAIN_OPERATIONS
                )
                actual = self.run_forward(torch, chain)
                expected = self.run_forward(reference_torch, chain)
                self.assertNotEqual(
                    tuple(actual["view"].shape),
                    tuple(actual["multiplier"].shape),
                )

                for stage in (
                    "leaf",
                    "input",
                    "view",
                    "first_unary",
                    "multiplier",
                    "multiplied",
                    "second_unary",
                    "loss",
                ):
                    self.assert_tensor_matches(
                        actual[stage], expected[stage], stage=stage
                    )

                self.assert_input_kind(
                    actual["input"], expected["input"], chain.input_kind
                )
                self.assert_storage_alias_matches(
                    actual["input"],
                    expected["input"],
                    actual["leaf"],
                    expected["leaf"],
                    stage="input-to-leaf alias",
                )
                self.assert_storage_alias_matches(
                    actual["view"],
                    expected["view"],
                    actual["input"],
                    expected["input"],
                    stage="generated view alias",
                )

                actual["loss"].backward()
                expected["loss"].backward()
                self.assert_tensor_matches(
                    actual["leaf"].grad,
                    expected["leaf"].grad,
                    stage="input leaf gradient",
                )
                self.assert_tensor_matches(
                    actual["multiplier"].grad,
                    expected["multiplier"].grad,
                    stage="broadcast leaf gradient",
                )


if __name__ == "__main__":
    unittest.main()
