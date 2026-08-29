import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README = REPOSITORY_ROOT / "README.md"
CONTRIBUTING = REPOSITORY_ROOT / "CONTRIBUTING.md"


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


if __name__ == "__main__":
    unittest.main()
