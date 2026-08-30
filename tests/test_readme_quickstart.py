import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README = REPOSITORY_ROOT / "README.md"
BENCHMARKING = REPOSITORY_ROOT / "BENCHMARKING.md"
CONTRIBUTING = REPOSITORY_ROOT / "CONTRIBUTING.md"
FEATURES = REPOSITORY_ROOT / "FEATURES.md"
SUPPORTED_SURFACE = REPOSITORY_ROOT / "docs" / "supported-surface.md"
SUPPORTED_SURFACE_ANCHORS = (
    ("Tensors", "tensors"),
    ("Creation and math", "creation-and-math"),
    ("NN and data", "nn-and-data"),
    (
        "Backends, compiler, and distributed",
        "backends-compiler-and-distributed",
    ),
    ("Unsupported boundaries", "unsupported-boundaries"),
)
SUPPORTED_SURFACE_SUBSECTION_ANCHORS = (
    ("Metadata and views", "metadata-and-views", "####"),
    ("Creation", "creation", "####"),
    ("Elementwise and reductions", "elementwise-and-reductions", "####"),
    ("NN and data helpers", "nn-and-data-helpers", "####"),
    ("Backend and compiler metadata", "backend-and-compiler-metadata", "####"),
    ("Unsupported boundaries", "unsupported-boundaries", "###"),
)
SUPPORTED_SURFACE_INDEX_SUMMARIES = (
    "CPU `float32` tensors",
    "inference-only `torch.nn.functional.softsign`",
    "Functional linear, loss, and deterministic dropout paths",
    "autocast cache state helpers",
    "eager JIT helper decorators and state queries",
    "Explicit unsupported APIs",
)


class ReadmeQuickstartTests(unittest.TestCase):
    def test_readme_keeps_quickstart_scope_and_evaluation_route(self):
        readme = README.read_text(encoding="utf-8")
        route_headings = (
            "# pytorch-rs",
            "## Quickstart",
            "### First success",
            "## Scope",
            "## Evaluation",
            "## Development",
        )

        previous_position = -1
        for heading in route_headings:
            with self.subTest(heading=heading):
                self.assertEqual(readme.count(heading), 1)
                position = readme.index(heading)
                self.assertGreater(position, previous_position)
                previous_position = position

    def test_source_install_commands_are_locked(self):
        readme = README.read_text(encoding="utf-8")
        match = re.search(
            r"^## Quickstart\n.*?^```bash\n(?P<commands>.*?)^```$",
            readme,
            flags=re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match, "README quickstart shell block is missing")

        commands = match.group("commands")
        self.assertIn("uv sync --locked", commands)
        self.assertIn("maturin develop --release --locked", commands)

    def test_first_success_example_is_short_and_runs(self):
        readme = README.read_text(encoding="utf-8")
        matches = list(
            re.finditer(
                r"^### First success\n\n```python\n(?P<source>.*?)^```$",
                readme,
                flags=re.MULTILINE | re.DOTALL,
            )
        )
        self.assertEqual(len(matches), 1, "expected one first-success example")

        source = matches[0].group("source").rstrip()
        self.assertLessEqual(len(source.splitlines()), 15)
        exec(compile(source, f"{README}#first-success", "exec"), {})

    def test_readme_routes_benchmark_policy_to_focused_docs(self):
        readme = README.read_text(encoding="utf-8")
        match = re.search(
            r"^## Evaluation\n(?P<section>.*?)^## Development$",
            readme,
            flags=re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match, "README evaluation section is missing")

        section = match.group("section")
        self.assertLessEqual(len(section.strip().splitlines()), 5)
        self.assertIn("[BENCHMARKING.md](BENCHMARKING.md)", section)
        self.assertIn("[FEATURES.md](FEATURES.md)", section)
        self.assertTrue(BENCHMARKING.is_file())
        self.assertTrue(FEATURES.is_file())

        benchmarking = BENCHMARKING.read_text(encoding="utf-8")
        normalized_benchmarking = re.sub(r"\s+", " ", benchmarking)
        for policy_text in (
            "outputs, shapes, dtypes, errors, aliasing, and edge cases",
            "never removed from the denominator",
            "fixed seeds",
            "generated or held-out shapes",
            "compile time, and dependency-installation time",
            "may not weaken, delete, skip, special-case, or rewrite evaluation infrastructure",
            "Benchmark changes are separate, human-reviewed campaign changes",
        ):
            with self.subTest(policy_text=policy_text):
                self.assertIn(policy_text, normalized_benchmarking)

        features = FEATURES.read_text(encoding="utf-8")
        normalized_features = re.sub(r"\s+", " ", features)
        self.assertIn(
            "production tensor operations forwarded to Python or PyTorch",
            normalized_features,
        )

    def test_readme_links_contributing_guide(self):
        readme = README.read_text(encoding="utf-8")
        self.assertIn("[CONTRIBUTING.md](CONTRIBUTING.md)", readme)
        self.assertTrue(CONTRIBUTING.is_file())

        contributing = CONTRIBUTING.read_text(encoding="utf-8")
        self.assertLess(len(contributing.splitlines()), 120)

    def test_readme_routes_to_supported_surface_anchors(self):
        readme = README.read_text(encoding="utf-8")
        route = "docs/supported-surface.md"
        self.assertRegex(readme, rf"\[[^\]]+\]\({re.escape(route)}\)")
        self.assertTrue(SUPPORTED_SURFACE.is_file())

        supported = SUPPORTED_SURFACE.read_text(encoding="utf-8")
        self.assertIn("## Category index", supported)
        self.assertIn("## Current baseline", supported)
        self.assertLess(
            supported.index("## Category index"),
            supported.index("## Current baseline"),
        )

        category_index = supported[
            supported.index("## Category index") : supported.index(
                "## Current baseline"
            )
        ]
        self.assertIn(
            "| Surface area | Supported summary | Contract section |",
            category_index,
        )
        self.assertIn("| --- | --- | --- |", category_index)
        for summary in SUPPORTED_SURFACE_INDEX_SUMMARIES:
            with self.subTest(summary=summary):
                self.assertIn(summary, category_index)
        for title, anchor in SUPPORTED_SURFACE_ANCHORS:
            with self.subTest(anchor=anchor):
                self.assertIn(f"[{title}](#{anchor})", category_index)
                self.assertRegex(
                    supported,
                    rf"(?m)^### {re.escape(title)}$",
                    msg=f"missing supported-surface anchor: {anchor}",
                )

        previous_position = supported.index("## Current baseline")
        for title, anchor, heading_level in SUPPORTED_SURFACE_SUBSECTION_ANCHORS:
            with self.subTest(anchor=anchor):
                heading = f"{heading_level} {title}"
                self.assertRegex(
                    supported,
                    rf"(?m)^{re.escape(heading)}$",
                    msg=f"missing supported-surface subsection anchor: {anchor}",
                )
                position = supported.index(heading)
                self.assertGreater(position, previous_position)
                previous_position = position

if __name__ == "__main__":
    unittest.main()
