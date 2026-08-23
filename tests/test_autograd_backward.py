import copy
import importlib
import inspect
import operator
import pickle
import re
import subprocess
import sys
import types
import typing
import unittest
from collections.abc import Sequence

import torch_rs as torch


class CustomSequence(Sequence):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


class ListSubclass(list):
    pass


class TupleSubclass(tuple):
    pass


def wrap_root(root, sequence_type):
    if sequence_type is None:
        return root
    return sequence_type((root,))


def default_grad_tensors(sequence_type):
    if sequence_type is None:
        return None
    return sequence_type((None,))


class AutogradBackwardTests(unittest.TestCase):
    def test_single_root_calls_return_none_and_accumulate_gradients(self):
        calls = (
            lambda loss: torch.autograd.backward(loss),
            lambda loss: torch.autograd.backward(tensors=loss),
            lambda loss: torch.autograd.backward(
                loss,
                grad_tensors=None,
                retain_graph=None,
                create_graph=False,
                grad_variables=None,
                inputs=None,
            ),
            lambda loss: torch.autograd.backward(
                loss, None, False, False, None, None
            ),
            lambda loss: torch.autograd.backward(
                loss, None, operator.index(False), 0
            ),
            lambda loss: torch.autograd.backward(loss, (None,)),
            lambda loss: torch.autograd.backward(
                loss, grad_tensors=[None]
            ),
        )

        for sequence_type in (None, tuple, list):
            for case, call in enumerate(calls):
                with self.subTest(sequence_type=sequence_type, case=case):
                    leaf = torch.tensor([2.0, -3.0], requires_grad=True)
                    loss = (leaf * leaf).sum()
                    roots = wrap_root(loss, sequence_type)
                    self.assertIsNone(call(roots))
                    self.assertEqual(leaf.grad.tolist(), [4.0, -6.0])

    def test_empty_root_calls_are_non_mutating_noops(self):
        calls = (
            ("positional", lambda roots: torch.autograd.backward(roots)),
            (
                "keyword",
                lambda roots: torch.autograd.backward(tensors=roots),
            ),
            (
                "explicit defaults",
                lambda roots: torch.autograd.backward(
                    roots,
                    grad_tensors=None,
                    retain_graph=None,
                    create_graph=False,
                    grad_variables=None,
                    inputs=None,
                ),
            ),
            (
                "positional defaults",
                lambda roots: torch.autograd.backward(
                    roots, None, False, False, None, None
                ),
            ),
            (
                "integer false",
                lambda roots: torch.autograd.backward(roots, None, 0, 0),
            ),
            (
                "empty tuple grad_tensors",
                lambda roots: torch.autograd.backward(roots, ()),
            ),
            (
                "empty list grad_tensors",
                lambda roots: torch.autograd.backward(
                    roots, grad_tensors=[]
                ),
            ),
            (
                "tuple singleton None grad_tensors",
                lambda roots: torch.autograd.backward(roots, (None,)),
            ),
            (
                "list singleton None grad_tensors",
                lambda roots: torch.autograd.backward(
                    roots, grad_tensors=[None]
                ),
            ),
        )

        for root_sequence_type in (tuple, list):
            for label, call in calls:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=label
                ):
                    leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                    leaf.sum().backward()
                    self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])
                    loss = (leaf * leaf).sum()

                    self.assertIsNone(call(root_sequence_type()))
                    self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])

                    loss.backward()
                    self.assertEqual(leaf.grad.tolist(), [5.0, 7.0])

    def test_graph_reuse_freeing_and_accumulation_follow_tensor_backward(self):
        for sequence_type in (None, tuple, list):
            with self.subTest(sequence_type=sequence_type):
                reusable_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
                reusable_loss = reusable_leaf.transpose(0, 0).sum()
                torch.autograd.backward(wrap_root(reusable_loss, sequence_type))
                torch.autograd.backward(wrap_root(reusable_loss, sequence_type))
                self.assertEqual(reusable_leaf.grad.tolist(), [2.0, 2.0])

                scalar_leaf = torch.tensor(7.0, requires_grad=True)
                torch.autograd.backward(wrap_root(scalar_leaf, sequence_type))
                torch.autograd.backward(wrap_root(scalar_leaf, sequence_type))
                self.assertEqual(scalar_leaf.grad.item(), 2.0)

                freed_leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                freed_loss = (freed_leaf * freed_leaf).sum()
                torch.autograd.backward(wrap_root(freed_loss, sequence_type))
                self.assertEqual(freed_leaf.grad.tolist(), [4.0, 6.0])
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    torch.autograd.backward(wrap_root(freed_loss, sequence_type))

                torch.autograd.backward(
                    wrap_root((freed_leaf * freed_leaf).sum(), sequence_type)
                )
                self.assertEqual(freed_leaf.grad.tolist(), [8.0, 12.0])

    def test_singleton_none_grad_tensors_preserve_backward_semantics(self):
        for root_sequence_type in (None, tuple, list):
            for grad_sequence_type in (tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    grad_tensors = default_grad_tensors(grad_sequence_type)

                    reusable_leaf = torch.tensor(
                        [1.0, 2.0], requires_grad=True
                    )
                    reusable_loss = reusable_leaf.transpose(0, 0).sum()
                    torch.autograd.backward(
                        wrap_root(reusable_loss, root_sequence_type),
                        grad_tensors=grad_tensors,
                    )
                    torch.autograd.backward(
                        wrap_root(reusable_loss, root_sequence_type),
                        grad_tensors=grad_tensors,
                    )
                    self.assertEqual(
                        reusable_leaf.grad.tolist(), [2.0, 2.0]
                    )

                    scalar_leaf = torch.tensor(7.0, requires_grad=True)
                    torch.autograd.backward(
                        wrap_root(scalar_leaf, root_sequence_type),
                        grad_tensors=grad_tensors,
                    )
                    torch.autograd.backward(
                        wrap_root(scalar_leaf, root_sequence_type),
                        grad_tensors=grad_tensors,
                    )
                    self.assertEqual(scalar_leaf.grad.item(), 2.0)

                    freed_leaf = torch.tensor(
                        [2.0, 3.0], requires_grad=True
                    )
                    freed_leaf.sum().backward()
                    freed_loss = (freed_leaf * freed_leaf).sum()
                    torch.autograd.backward(
                        wrap_root(freed_loss, root_sequence_type),
                        grad_tensors=grad_tensors,
                    )
                    self.assertEqual(freed_leaf.grad.tolist(), [5.0, 7.0])
                    with self.assertRaisesRegex(
                        RuntimeError, "backward through the graph a second time"
                    ):
                        torch.autograd.backward(
                            wrap_root(freed_loss, root_sequence_type),
                            grad_tensors=grad_tensors,
                        )
                    torch.autograd.backward(
                        wrap_root(
                            (freed_leaf * freed_leaf).sum(),
                            root_sequence_type,
                        ),
                        grad_tensors=grad_tensors,
                    )
                    self.assertEqual(
                        freed_leaf.grad.tolist(), [9.0, 13.0]
                    )

                    nonscalar_leaf = torch.tensor(
                        [2.0, 3.0], requires_grad=True
                    )
                    nonscalar = nonscalar_leaf * nonscalar_leaf
                    with self.assertRaisesRegex(
                        RuntimeError, "implicitly created only for scalar"
                    ):
                        torch.autograd.backward(
                            wrap_root(nonscalar, root_sequence_type),
                            grad_tensors=grad_tensors,
                        )
                    self.assertIsNone(nonscalar_leaf.grad)
                    nonscalar.sum().backward()
                    self.assertEqual(
                        nonscalar_leaf.grad.tolist(), [4.0, 6.0]
                    )

    def test_tensor_backward_errors_are_preserved(self):
        for sequence_type in (None, tuple, list):
            with self.subTest(sequence_type=sequence_type):
                with self.assertRaisesRegex(RuntimeError, "does not require grad"):
                    torch.autograd.backward(
                        wrap_root(torch.tensor(1.0), sequence_type)
                    )
                with self.assertRaisesRegex(
                    RuntimeError, "implicitly created only for scalar"
                ):
                    torch.autograd.backward(
                        wrap_root(
                            torch.tensor([1.0, 2.0], requires_grad=True),
                            sequence_type,
                        )
                    )

    def test_unsupported_forms_fail_before_gradients_or_graph_state_change(self):
        root_error = (
            "torch_rs.autograd.backward only supports an exact native Tensor, "
            "directly or in an exact one-element tuple or list"
        )
        unsupported = (
            (
                "multiple tuple roots",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward((loss, loss)),
            ),
            (
                "multiple list roots",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward([loss, loss]),
            ),
            (
                "custom sequence",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(
                    CustomSequence((loss,))
                ),
            ),
            (
                "empty custom sequence",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(
                    CustomSequence(())
                ),
            ),
            (
                "tuple subclass",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(
                    TupleSubclass((loss,))
                ),
            ),
            (
                "empty tuple subclass",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(TupleSubclass()),
            ),
            (
                "list subclass",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(
                    ListSubclass([loss])
                ),
            ),
            (
                "empty list subclass",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(ListSubclass()),
            ),
            (
                "non-tensor singleton",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward((object(),)),
            ),
            (
                "retained graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support retain_graph=True",
                lambda leaf, loss: torch.autograd.backward(
                    loss, retain_graph=True
                ),
            ),
            (
                "higher-order graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support create_graph=True",
                lambda leaf, loss: torch.autograd.backward(
                    loss, create_graph=True
                ),
            ),
            (
                "grad_variables",
                NotImplementedError,
                "torch_rs.autograd.backward does not support grad_variables",
                lambda leaf, loss: torch.autograd.backward(
                    loss, grad_variables=torch.tensor(1.0)
                ),
            ),
            (
                "inputs",
                NotImplementedError,
                "torch_rs.autograd.backward does not support inputs",
                lambda leaf, loss: torch.autograd.backward(loss, inputs=leaf),
            ),
        )

        for label, error_type, message, call in unsupported:
            with self.subTest(label=label):
                leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                leaf.sum().backward()
                self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])
                loss = (leaf * leaf).sum()
                with self.assertRaisesRegex(
                    error_type, f"^{re.escape(message)}$"
                ):
                    call(leaf, loss)
                self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])

                loss.backward()
                self.assertEqual(leaf.grad.tolist(), [5.0, 7.0])

    def test_non_default_grad_tensors_forms_are_rejected_before_backward(self):
        grad_tensors = (
            ("tensor", lambda: torch.tensor(1.0)),
            ("tuple with tensor", lambda: (torch.tensor(1.0),)),
            ("list with tensor", lambda: [torch.tensor(1.0)]),
            ("empty tuple", tuple),
            ("empty list", list),
            ("multiple tuple", lambda: (None, None)),
            ("multiple list", lambda: [None, None]),
            ("custom sequence", lambda: CustomSequence((None,))),
            ("tuple subclass", lambda: TupleSubclass((None,))),
            ("list subclass", lambda: ListSubclass([None])),
        )
        message = (
            "torch_rs.autograd.backward does not support explicit gradients"
        )

        for sequence_type in (None, tuple, list):
            for label, make_grad_tensors in grad_tensors:
                with self.subTest(
                    sequence_type=sequence_type, grad_tensors=label
                ):
                    leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                    leaf.sum().backward()
                    loss = (leaf * leaf).sum()
                    with self.assertRaisesRegex(
                        NotImplementedError, f"^{re.escape(message)}$"
                    ):
                        torch.autograd.backward(
                            wrap_root(loss, sequence_type),
                            grad_tensors=make_grad_tensors(),
                        )
                    self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])
                    loss.backward()
                    self.assertEqual(leaf.grad.tolist(), [5.0, 7.0])

    def test_empty_roots_reject_non_default_grad_tensors(self):
        grad_tensors = (
            ("tensor", lambda: torch.tensor(1.0)),
            ("tuple with tensor", lambda: (torch.tensor(1.0),)),
            ("list with tensor", lambda: [torch.tensor(1.0)]),
            ("multiple tuple", lambda: (None, None)),
            ("multiple list", lambda: [None, None]),
            ("custom empty sequence", lambda: CustomSequence(())),
            ("custom singleton None", lambda: CustomSequence((None,))),
            ("empty tuple subclass", TupleSubclass),
            ("empty list subclass", ListSubclass),
        )
        message = (
            "torch_rs.autograd.backward does not support explicit gradients"
        )

        for root_sequence_type in (tuple, list):
            for label, make_grad_tensors in grad_tensors:
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_tensors=label,
                ):
                    leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                    leaf.sum().backward()
                    loss = (leaf * leaf).sum()
                    with self.assertRaisesRegex(
                        NotImplementedError, f"^{re.escape(message)}$"
                    ):
                        torch.autograd.backward(
                            root_sequence_type(),
                            grad_tensors=make_grad_tensors(),
                        )
                    self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])
                    loss.backward()
                    self.assertEqual(leaf.grad.tolist(), [5.0, 7.0])

    def test_graph_option_conversion_errors_are_non_mutating(self):
        cases = (
            ("retain_graph", 0.5),
            ("create_graph", None),
        )
        for grad_sequence_type in (None, tuple, list):
            for name, value in cases:
                with self.subTest(
                    grad_sequence_type=grad_sequence_type,
                    name=name,
                    value=value,
                ):
                    leaf = torch.tensor(2.0, requires_grad=True)
                    loss = leaf * leaf
                    with self.assertRaises(TypeError) as raised:
                        torch.autograd.backward(
                            loss,
                            grad_tensors=default_grad_tensors(
                                grad_sequence_type
                            ),
                            **{name: value},
                        )
                    self.assertEqual(
                        str(raised.exception),
                        f"'{type(value).__name__}' object cannot be "
                        "interpreted as an integer",
                    )
                    self.assertIsNone(leaf.grad)
                    loss.backward()
                    self.assertEqual(leaf.grad.item(), 4.0)

    def test_singleton_none_grad_tensors_reach_later_option_validation(self):
        cases = (
            (
                "retain_graph",
                "torch_rs.autograd.backward does not support "
                "retain_graph=True",
                lambda leaf: {"retain_graph": True},
            ),
            (
                "create_graph",
                "torch_rs.autograd.backward does not support "
                "create_graph=True",
                lambda leaf: {"create_graph": True},
            ),
            (
                "grad_variables",
                "torch_rs.autograd.backward does not support grad_variables",
                lambda leaf: {"grad_variables": torch.tensor(1.0)},
            ),
            (
                "inputs",
                "torch_rs.autograd.backward does not support inputs",
                lambda leaf: {"inputs": leaf},
            ),
        )
        for root_sequence_type in (None, tuple, list):
            for grad_sequence_type in (tuple, list):
                for label, message, make_options in cases:
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        grad_sequence_type=grad_sequence_type,
                        option=label,
                    ):
                        leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                        leaf.sum().backward()
                        loss = (leaf * leaf).sum()
                        with self.assertRaisesRegex(
                            NotImplementedError, f"^{re.escape(message)}$"
                        ):
                            torch.autograd.backward(
                                wrap_root(loss, root_sequence_type),
                                grad_tensors=default_grad_tensors(
                                    grad_sequence_type
                                ),
                                **make_options(leaf),
                            )
                        self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])
                        loss.backward()
                        self.assertEqual(leaf.grad.tolist(), [5.0, 7.0])

    def test_empty_roots_reject_non_default_options(self):
        cases = (
            (
                "retain_graph",
                "torch_rs.autograd.backward does not support "
                "retain_graph=True",
                lambda leaf: {"retain_graph": True},
            ),
            (
                "create_graph",
                "torch_rs.autograd.backward does not support "
                "create_graph=True",
                lambda leaf: {"create_graph": True},
            ),
            (
                "grad_variables",
                "torch_rs.autograd.backward does not support grad_variables",
                lambda leaf: {"grad_variables": torch.tensor(1.0)},
            ),
            (
                "inputs",
                "torch_rs.autograd.backward does not support inputs",
                lambda leaf: {"inputs": leaf},
            ),
        )
        supported_grad_tensors = (None, (), [], (None,), [None])

        for root_sequence_type in (tuple, list):
            for grad_tensors in supported_grad_tensors:
                for label, message, make_options in cases:
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        grad_tensors=grad_tensors,
                        option=label,
                    ):
                        leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                        leaf.sum().backward()
                        loss = (leaf * leaf).sum()
                        with self.assertRaisesRegex(
                            NotImplementedError, f"^{re.escape(message)}$"
                        ):
                            torch.autograd.backward(
                                root_sequence_type(),
                                grad_tensors=grad_tensors,
                                **make_options(leaf),
                            )
                        self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])
                        loss.backward()
                        self.assertEqual(leaf.grad.tolist(), [5.0, 7.0])

    def test_root_and_gradient_validation_precede_graph_options(self):
        root_error = (
            "torch_rs.autograd.backward only supports an exact native Tensor, "
            "directly or in an exact one-element tuple or list"
        )
        gradient_error = (
            "torch_rs.autograd.backward does not support explicit gradients"
        )
        leaf = torch.tensor(2.0, requires_grad=True)
        loss = leaf * leaf

        with self.assertRaisesRegex(TypeError, f"^{re.escape(root_error)}$"):
            torch.autograd.backward(
                (loss, loss),
                grad_tensors=(torch.tensor(1.0),),
                retain_graph=True,
            )
        with self.assertRaisesRegex(
            NotImplementedError, f"^{re.escape(gradient_error)}$"
        ):
            torch.autograd.backward(
                (), grad_tensors=(torch.tensor(1.0),), retain_graph=True
            )
        with self.assertRaisesRegex(
            NotImplementedError, f"^{re.escape(gradient_error)}$"
        ):
            torch.autograd.backward(
                loss,
                grad_tensors=(torch.tensor(1.0),),
                retain_graph=True,
            )
        self.assertIsNone(leaf.grad)
        loss.backward()
        self.assertEqual(leaf.grad.item(), 4.0)

    def test_metadata_imports_copying_pickling_and_exports(self):
        module = importlib.import_module("torch_rs.autograd")
        function = module.backward

        self.assertIs(module, torch.autograd)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "backward")
        self.assertEqual(function.__qualname__, "backward")
        self.assertEqual(function.__module__, "torch_rs.autograd")
        self.assertEqual(
            tuple(inspect.signature(function).parameters),
            (
                "tensors",
                "grad_tensors",
                "retain_graph",
                "create_graph",
                "grad_variables",
                "inputs",
            ),
        )
        self.assertEqual(function.__defaults__, (None, None, False, None, None))
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(
            tuple(function.__annotations__),
            (
                "tensors",
                "grad_tensors",
                "retain_graph",
                "create_graph",
                "grad_variables",
                "inputs",
                "return",
            ),
        )
        self.assertIs(
            typing.get_args(function.__annotations__["tensors"])[0],
            torch.Tensor,
        )
        self.assertIs(function.__annotations__["create_graph"], bool)
        self.assertIsNone(function.__annotations__["return"])
        self.assertIn("Compute the sum of gradients", function.__doc__)

        self.assertEqual(module.__all__.count("backward"), 1)
        wildcard_namespace = {}
        exec("from torch_rs.autograd import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["backward"], function)
        explicit_namespace = {}
        exec("from torch_rs.autograd import backward", explicit_namespace)
        self.assertIs(explicit_namespace["backward"], function)

        self.assertFalse(hasattr(torch, "backward"))
        self.assertNotIn("backward", torch.__all__)
        self.assertFalse(hasattr(module, "grad"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_signature_binding_errors_are_non_mutating(self):
        leaf = torch.tensor(2.0, requires_grad=True)
        loss = leaf * leaf
        calls = (
            lambda: torch.autograd.backward(),
            lambda: torch.autograd.backward(loss, tensors=loss),
            lambda: torch.autograd.backward(loss, unexpected=True),
            lambda: torch.autograd.backward(
                loss, None, None, False, None, None, None
            ),
        )
        for case, call in enumerate(calls):
            with self.subTest(case=case), self.assertRaises(TypeError):
                call()
        self.assertIsNone(leaf.grad)
        loss.backward()
        self.assertEqual(leaf.grad.item(), 4.0)

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch
from torch_rs.autograd import backward

leaf = torch.tensor(2.0, requires_grad=True)
assert backward(leaf * leaf) is None
assert leaf.grad.item() == 4.0
tuple_leaf = torch.tensor(3.0, requires_grad=True)
assert backward((tuple_leaf * tuple_leaf,)) is None
assert tuple_leaf.grad.item() == 6.0
list_leaf = torch.tensor(4.0, requires_grad=True)
assert backward([list_leaf * list_leaf]) is None
assert list_leaf.grad.item() == 8.0
tuple_grad_leaf = torch.tensor(5.0, requires_grad=True)
assert backward(tuple_grad_leaf * tuple_grad_leaf, (None,)) is None
assert tuple_grad_leaf.grad.item() == 10.0
list_grad_leaf = torch.tensor(6.0, requires_grad=True)
assert backward((list_grad_leaf * list_grad_leaf,), [None]) is None
assert list_grad_leaf.grad.item() == 12.0
assert backward(()) is None
assert backward([], ()) is None
assert backward((), [None]) is None
assert not hasattr(torch.autograd, "grad")
assert not hasattr(torch, "backward")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
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


if __name__ == "__main__":
    unittest.main()
