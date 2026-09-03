import json
from pathlib import Path
import unittest


class CompileCorpusTests(unittest.TestCase):
    def test_versioned_unary_relu_entry_has_explicit_accounting(self):
        corpus_path = Path(__file__).with_name("compile_corpus_v1.json")
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

        self.assertEqual(corpus["version"], 1)
        self.assertEqual(
            corpus["entries"],
            [
                {
                    "id": "torch_compile_unary_relu_cpu_float32",
                    "category": "unary_relu",
                    "description": (
                        "Exact Python callable with one positional exact native "
                        "CPU float32 Tensor argument returning x.relu() or "
                        "torch.relu(x)."
                    ),
                    "weight": 1,
                    "eligible": 1,
                    "passed": 1,
                }
            ],
        )
        self.assertEqual(
            corpus["totals"],
            {
                "weight": sum(entry["weight"] for entry in corpus["entries"]),
                "eligible": sum(entry["eligible"] for entry in corpus["entries"]),
                "passed": sum(entry["passed"] for entry in corpus["entries"]),
            },
        )


if __name__ == "__main__":
    unittest.main()
