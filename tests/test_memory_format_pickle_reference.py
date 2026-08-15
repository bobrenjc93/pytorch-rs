import copy
import pickle
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


FORMAT_NAMES = (
    "preserve_format",
    "contiguous_format",
    "channels_last",
    "channels_last_3d",
)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class MemoryFormatPickleReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "memory-format pickle differentials require pinned PyTorch 2.13.0"
            )

    def format_contract(self, module):
        formats = tuple(getattr(module, name) for name in FORMAT_NAMES)
        results = []
        for name, value in zip(FORMAT_NAMES, formats, strict=True):
            metadata = (
                repr(value),
                str(value),
                tuple(value == other for other in formats),
                tuple(value != other for other in formats),
                tuple(hash(value) == hash(other) for other in formats),
                type(hash(value)).__name__,
            )
            results.append(
                {
                    "reduce": value.__reduce__(),
                    "reduce_ex": tuple(
                        value.__reduce_ex__(protocol)
                        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                    ),
                    "public_name_identity": getattr(module.torch, name) is value,
                    "copy_identities": (
                        copy.copy(value) is value,
                        copy.deepcopy(value) is value,
                    ),
                    "copy_metadata": tuple(
                        (
                            repr(copied),
                            str(copied),
                            tuple(copied == other for other in formats),
                            tuple(copied != other for other in formats),
                            tuple(
                                hash(copied) == hash(other) for other in formats
                            ),
                            type(hash(copied)).__name__,
                        )
                        for copied in (copy.copy(value), copy.deepcopy(value))
                    ),
                    "pickle_results": tuple(
                        (
                            restored is value,
                            repr(restored),
                            str(restored),
                            tuple(restored == other for other in formats),
                            tuple(
                                hash(restored) == hash(other) for other in formats
                            ),
                        )
                        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                        for restored in (
                            pickle.loads(pickle.dumps(value, protocol=protocol)),
                        )
                    ),
                    "metadata": metadata,
                }
            )
        return {
            "module_self_alias": module.torch is module,
            "torch_in_all": "torch" in module.__all__,
            "formats": tuple(results),
        }

    def tensor_layout_contract(self, module):
        formats = tuple(getattr(module, name) for name in FORMAT_NAMES)
        matrix = module.zeros((2, 3, 2, 4)).transpose(2, 3)
        volume = module.zeros((2, 3, 2, 2, 4)).transpose(3, 4)
        tensors = (
            matrix,
            matrix.clone(memory_format=module.preserve_format),
            matrix.clone(memory_format=module.contiguous_format),
            matrix.contiguous(memory_format=module.channels_last),
            volume.contiguous(memory_format=module.channels_last_3d),
        )
        return tuple(
            (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                tuple(
                    tensor.is_contiguous(memory_format=memory_format)
                    for memory_format in formats
                ),
            )
            for tensor in tensors
        )

    def test_pickle_copy_and_metadata_match_pytorch_2_13(self):
        self.assertEqual(
            self.format_contract(torch),
            self.format_contract(reference_torch),
        )

    def test_tensor_layout_behavior_remains_unchanged(self):
        self.assertEqual(
            self.tensor_layout_contract(torch),
            self.tensor_layout_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
