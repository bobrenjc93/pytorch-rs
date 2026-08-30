import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README = REPOSITORY_ROOT / "README.md"
CONTRIBUTING = REPOSITORY_ROOT / "CONTRIBUTING.md"
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
