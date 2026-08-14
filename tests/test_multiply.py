import array
import collections
import inspect
import io
import numbers
import re
import subprocess
import sys
import textwrap
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorMultiplyTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def test_tensor_and_real_scalar_calls_match_mul(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [3.0], [4.0]])
        self.assert_tensor_matches(
            left.multiply(right), left.mul(right), case="tensor positional"
        )
        self.assert_tensor_matches(
            left.multiply(other=right),
            left.mul(other=right),
            case="tensor keyword",
        )
        self.assert_tensor_matches(
            left.multiply(x2=right), left.mul(other=right), case="tensor x2 keyword"
        )

        offset_view = left[1]
        for scalar in (
            True,
            -2,
            2.5,
            np.bool_(False),
            np.int64(3),
            np.float32(-0.0),
        ):
            self.assert_tensor_matches(
                offset_view.multiply(scalar),
                offset_view.mul(scalar),
                case=("scalar positional", scalar),
            )
            self.assert_tensor_matches(
                offset_view.multiply(other=scalar),
                offset_view.mul(other=scalar),
                case=("scalar keyword", scalar),
            )
        self.assert_tensor_matches(
            offset_view.multiply(x2=np.float32(-2.5)),
            offset_view.mul(other=np.float32(-2.5)),
            case="scalar x2 keyword",
        )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            empty.multiply(other=broadcast),
            empty.mul(other=broadcast),
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        self.assert_tensor_matches(
            special.multiply(-0.0),
            special.mul(-0.0),
            case="signed zero and non-finites",
        )

    def test_autograd_shared_operands_empties_and_no_grad_match_mul(self):
        alias_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        alias_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        mul_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        mul_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)

        alias_output = alias_left.transpose(0, 1).multiply(
            other=alias_right.transpose(0, 1)
        )
        mul_output = mul_left.transpose(0, 1).mul(
            other=mul_right.transpose(0, 1)
        )
        self.assert_tensor_matches(alias_output, mul_output, case="tracked views")
        alias_output.sum().backward()
        mul_output.sum().backward()
        self.assert_tensor_matches(alias_left.grad, mul_left.grad, case="left gradient")
        self.assert_tensor_matches(
            alias_right.grad, mul_right.grad, case="right gradient"
        )

        alias_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        mul_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        alias_shared.multiply(alias_shared).sum().backward()
        mul_shared.mul(mul_shared).sum().backward()
        self.assert_tensor_matches(
            alias_shared.grad, mul_shared.grad, case="shared operand gradient"
        )

        alias_empty = torch.zeros((2, 0, 3), requires_grad=True)
        mul_empty = torch.zeros((2, 0, 3), requires_grad=True)
        alias_empty.multiply(other=torch.ones((1, 1, 3))).sum().backward()
        mul_empty.mul(other=torch.ones((1, 1, 3))).sum().backward()
        self.assert_tensor_matches(
            alias_empty.grad, mul_empty.grad, case="empty gradient"
        )

        no_grad_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        no_grad_right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            tensor_output = no_grad_left.transpose(0, 1).multiply(
                no_grad_right.transpose(0, 1)
            )
            scalar_output = no_grad_left.multiply(other=2.0)
        self.assertFalse(tensor_output.requires_grad)
        self.assertFalse(scalar_output.requires_grad)
        self.assertTrue(
            no_grad_left.multiply(no_grad_right.transpose(0, 1)).requires_grad
        )

    def test_descriptor_metadata_unbound_call_and_argument_errors(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "multiply")
        bound = tensor.multiply

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "multiply")
        self.assertEqual(bound.__name__, "multiply")
        self.assertEqual(descriptor.__qualname__, "TensorBase.multiply")
        self.assertEqual(bound.__qualname__, "Tensor.multiply")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertEqual(
            descriptor.__doc__,
            "\nmultiply(value) -> Tensor\n\nSee :func:`torch.multiply`.\n",
        )
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)
        self.assert_tensor_matches(
            descriptor(tensor, other=tensor),
            tensor.mul(tensor),
            case="unbound call",
        )

        overloads = "but expected one of:\n * (Tensor other)\n * (Number other)\n"
        cases = (
            (
                lambda: tensor.multiply(),
                f"multiply() received an invalid combination of arguments - got (), {overloads}",
            ),
            (
                lambda: tensor.multiply(tensor, tensor),
                "multiply() received an invalid combination of arguments - got "
                f"(Tensor, Tensor), {overloads}",
            ),
            (
                lambda: tensor.multiply(tensor, other=tensor),
                "multiply() received an invalid combination of arguments - got "
                f"(Tensor, other=Tensor), {overloads}",
            ),
            (
                lambda: tensor.multiply(tensor, out=tensor),
                "multiply() received an invalid combination of arguments - got "
                f"(Tensor, out=Tensor), {overloads}",
            ),
            (
                lambda: tensor.multiply(other=tensor, wat=tensor),
                "multiply() received an invalid combination of arguments - got "
                f"(wat=Tensor, other=Tensor, ), {overloads}",
            ),
            (
                lambda: tensor.multiply(wat=tensor),
                "multiply() received an invalid combination of arguments - got "
                "(wat=Tensor, ), but expected one of:\n"
                " * (Tensor other)\n"
                "      didn't match because some of the keywords were incorrect: wat\n"
                " * (Number other)\n"
                "      didn't match because some of the keywords were incorrect: wat\n",
            ),
            (
                lambda: tensor.multiply([]),
                "multiply() received an invalid combination of arguments - got "
                "(list), but expected one of:\n"
                " * (Tensor other)\n"
                "      didn't match because some of the arguments have invalid types: "
                "(!list of []!)\n"
                " * (Number other)\n"
                "      didn't match because some of the arguments have invalid types: "
                "(!list of []!)\n",
            ),
            (
                lambda: tensor.multiply(other=None),
                "multiply() received an invalid combination of arguments - got "
                "(other=NoneType, ), but expected one of:\n"
                " * (Tensor other)\n"
                "      didn't match because some of the arguments have invalid types: "
                "(!other=NoneType!, )\n"
                " * (Number other)\n"
                "      didn't match because some of the arguments have invalid types: "
                "(!other=NoneType!, )\n",
            ),
            (
                lambda: tensor.multiply(x2=[]),
                "multiply() received an invalid combination of arguments - got "
                "(x2=list, ), but expected one of:\n"
                " * (Tensor other)\n"
                "      didn't match because some of the keywords were incorrect: x2\n"
                " * (Number other)\n"
                "      didn't match because some of the keywords were incorrect: x2\n",
            ),
            (
                lambda: tensor.multiply([], out=tensor),
                "multiply() received an invalid combination of arguments - got "
                f"(list, out=Tensor), {overloads}",
            ),
            (lambda: tensor.multiply(np.uint64(2**63)), "an integer is required"),
            (lambda: tensor.multiply(2**64), "int too big to convert"),
            (
                lambda: tensor.multiply(-(2**63) - 1),
                "can't convert negative int to unsigned",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        with self.assertRaisesRegex(
            TypeError,
            r"^mul\(\): argument 'other' must be Tensor, not list$",
        ):
            tensor.mul(x2=[], wat=tensor)
        for call in (
            lambda: tensor.mul(x2=tensor, other=[]),
            lambda: tensor.mul(other=[], x2=tensor),
        ):
            with self.subTest(mul_x2_fallback=call):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^mul\(\): argument 'other' must be Tensor, not list$",
                ):
                    call()

        descriptor_cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.multiply() needs an argument",
            ),
            (
                lambda: descriptor(1, tensor),
                "descriptor 'multiply' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor, other=tensor),
                "unbound method TensorBase.multiply() needs an argument",
            ),
        )
        for call, message in descriptor_cases:
            with self.subTest(descriptor_error=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_invalid_sequence_subclasses_and_keyword_order_match_pytorch(self):
        class NamedList(list):
            pass

        class NamedTuple(tuple):
            pass

        class IterationBombList(list):
            def __iter__(self):
                raise RuntimeError("list iteration must not be invoked")

        class IterationBombTuple(tuple):
            def __iter__(self):
                raise RuntimeError("tuple iteration must not be invoked")

        class ProtocolList(list):
            def __iter__(self):
                raise RuntimeError("list iteration must not be invoked")

            def __len__(self):
                self.calls.append("len")
                return 1

            def __getitem__(self, index):
                self.calls.append(("getitem", index))
                return 3.5

        class ProtocolTuple(tuple):
            def __iter__(self):
                raise RuntimeError("tuple iteration must not be invoked")

            def __len__(self):
                self.calls.append("len")
                return 1

            def __getitem__(self, index):
                self.calls.append(("getitem", index))
                return 3.5

        class RaisingLengthList(list):
            def __len__(self):
                self.calls.append("len")
                raise RuntimeError("length must be cleared")

        class InvalidLengthTuple(tuple):
            def __len__(self):
                self.calls.append("len")
                return -1

        tensor = torch.tensor([1.0])
        for value, detail in (
            (NamedList([1, "x"]), "NamedList of [int, str]"),
            (NamedTuple((1, "x")), "NamedTuple of (int, str)"),
            (IterationBombList([1, "x"]), "IterationBombList of [int, str]"),
            (
                IterationBombTuple((1, "x")),
                "IterationBombTuple of (int, str)",
            ),
        ):
            message = (
                "multiply() received an invalid combination of arguments - got "
                f"({type(value).__name__}), but expected one of:\n"
                " * (Tensor other)\n"
                "      didn't match because some of the arguments have invalid types: "
                f"(!{detail}!)\n"
                " * (Number other)\n"
                "      didn't match because some of the arguments have invalid types: "
                f"(!{detail}!)\n"
            )
            with self.subTest(sequence=type(value).__name__):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    tensor.multiply(value)

        for value, detail in (
            (ProtocolList([1, "x"]), "ProtocolList of [float]"),
            (ProtocolTuple((1, "x")), "ProtocolTuple of (float,)"),
        ):
            value.calls = []
            message = (
                "multiply() received an invalid combination of arguments - got "
                f"({type(value).__name__}), but expected one of:\n"
                " * (Tensor other)\n"
                "      didn't match because some of the arguments have invalid types: "
                f"(!{detail}!)\n"
                " * (Number other)\n"
                "      didn't match because some of the arguments have invalid types: "
                f"(!{detail}!)\n"
            )
            with self.subTest(protocol_sequence=type(value).__name__):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    tensor.multiply(value)
                self.assertEqual(
                    value.calls,
                    ["len", ("getitem", 0), "len", ("getitem", 0)],
                )

        for value, detail in (
            (RaisingLengthList([1, "x"]), "RaisingLengthList of []"),
            (InvalidLengthTuple((1, "x")), "InvalidLengthTuple of ()"),
        ):
            value.calls = []
            message = (
                "multiply() received an invalid combination of arguments - got "
                f"({type(value).__name__}), but expected one of:\n"
                " * (Tensor other)\n"
                "      didn't match because some of the arguments have invalid types: "
                f"(!{detail}!)\n"
                " * (Number other)\n"
                "      didn't match because some of the arguments have invalid types: "
                f"(!{detail}!)\n"
            )
            with self.subTest(invalid_length=type(value).__name__):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    tensor.multiply(value)
                self.assertEqual(value.calls, ["len", "len"])

        keyword_order = None
        if sys.platform == "darwin":
            keyword_order = "d=Tensor, b=Tensor, a=Tensor"
        elif sys.platform == "win32":
            keyword_order = "a=Tensor, b=Tensor, d=Tensor"
        elif sys.platform.startswith("linux"):
            keyword_order = "b=Tensor, d=Tensor, a=Tensor"
        if keyword_order is not None:
            keyword_message = (
                "multiply() received an invalid combination of arguments - got "
                f"({keyword_order}, ), but expected one of:\n"
                " * (Tensor other)\n"
                " * (Number other)\n"
            )
            with self.assertRaisesRegex(
                TypeError, f"^{re.escape(keyword_message)}$"
            ):
                tensor.multiply(a=tensor, b=tensor, d=tensor)

    def test_extension_type_names_do_not_dispatch_metaclass_hooks(self):
        tensor = torch.tensor([1.0])
        values = (
            (re.compile("x"), "re.Pattern"),
            (array.array("i"), "array.array"),
            (collections.deque(), "collections.deque"),
            (io.BytesIO(), "_io.BytesIO"),
        )
        for value, expected_name in values:
            with self.subTest(type_name=expected_name):
                with self.assertRaises(TypeError) as raised:
                    tensor.multiply(value)
                message = str(raised.exception)
                self.assertIn(f"got ({expected_name}),", message)
                self.assertEqual(message.count(f"(!{expected_name}!)"), 2)

        metaclass_accesses = []

        class GuardedMeta(type):
            def __getattribute__(cls, name):
                if name in {"__name__", "__module__", "__flags__"}:
                    metaclass_accesses.append(name)
                    raise RuntimeError(f"metaclass hook invoked for {name}")
                return super().__getattribute__(name)

        class Guarded(metaclass=GuardedMeta):
            pass

        with self.assertRaises(TypeError) as raised:
            tensor.multiply(Guarded())
        self.assertIn("got (Guarded),", str(raised.exception))
        self.assertEqual(metaclass_accesses, [])

        SpoofedTensor = type("torch_rs.Tensor", (), {})
        with self.assertRaises(TypeError) as raised:
            tensor.multiply(SpoofedTensor())
        self.assertIn("got (torch_rs.Tensor),", str(raised.exception))

    def test_number_overload_markers_and_nul_keyword_match_pytorch(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: tensor.multiply(numbers.Number()),
                "multiply() received an invalid combination of arguments - got "
                "(Number), but expected one of:\n"
                " * (Tensor other)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (!Number!)\n"
                " * (Number other)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (Number)\n",
            ),
            (
                lambda: tensor.multiply(other=numbers.Number()),
                "multiply() received an invalid combination of arguments - got "
                "(other=Number, ), but expected one of:\n"
                " * (Tensor other)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (!other=Number!, )\n"
                " * (Number other)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (other=Number, )\n",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        with self.assertRaises(TypeError) as raised:
            tensor.multiply(**{"bad\x00tail": tensor})
        self.assertEqual(
            str(raised.exception),
            "multiply() received an invalid combination of arguments - got (bad",
        )

    def test_overflow_sign_and_surrogate_keyword_errors_match_pytorch(self):
        comparison_calls = []

        class ComparisonBombInt(int):
            def __lt__(self, other):
                comparison_calls.append(other)
                raise RuntimeError("comparison override must not be invoked")

        tensor = torch.tensor([1.0])
        for value, message in (
            (ComparisonBombInt(2**64), "int too big to convert"),
            (
                ComparisonBombInt(-(2**63) - 1),
                "can't convert negative int to unsigned",
            ),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    OverflowError, f"^{re.escape(message)}$"
                ):
                    tensor.multiply(value)
        self.assertEqual(comparison_calls, [])

        with self.assertRaisesRegex(
            RuntimeError, r"^error unpacking string as utf-8$"
        ):
            tensor.multiply(**{"\ud800": tensor})

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux RLIMIT_AS")
    def test_large_keyword_error_returns_bad_alloc_instead_of_aborting(self):
        script = textwrap.dedent(
            """\
            import os
            import resource

            import torch_rs as torch

            tensor = torch.tensor([1.0])
            keywords = {f"key{index}": tensor for index in range(50_000)}
            with open("/proc/self/statm", encoding="ascii") as statm:
                virtual_pages = int(statm.read().split()[0])
            current_virtual_size = virtual_pages * os.sysconf("SC_PAGE_SIZE")
            limit = current_virtual_size + 4 * 1024 * 1024
            _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if hard_limit != resource.RLIM_INFINITY and limit > hard_limit:
                raise SystemExit(77)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard_limit))

            try:
                tensor.multiply(**keywords)
            except RuntimeError as error:
                assert str(error) == "std::bad_alloc", repr(error)
            else:
                raise AssertionError("the constrained call unexpectedly succeeded")
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
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
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
