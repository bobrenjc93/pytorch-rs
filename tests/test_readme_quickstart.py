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
        for title, anchor in SUPPORTED_SURFACE_ANCHORS:
            with self.subTest(anchor=anchor):
                self.assertIn(f"- [{title}](#{anchor})", supported)
                self.assertRegex(
                    supported,
                    rf"(?m)^### {re.escape(title)}$",
                    msg=f"missing supported-surface anchor: {anchor}",
                )


if __name__ == "__main__":
    unittest.main()
