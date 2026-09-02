import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README = REPOSITORY_ROOT / "README.md"
BENCHMARKING = REPOSITORY_ROOT / "BENCHMARKING.md"
CONTRIBUTING = REPOSITORY_ROOT / "CONTRIBUTING.md"
FEATURES = REPOSITORY_ROOT / "FEATURES.md"
SUPPORTED_SURFACE = REPOSITORY_ROOT / "docs" / "supported-surface.md"
HISTORICAL_TIMING_REPORTS = (
    (
        "Rank-1 `Tensor.sum` release timings",
        "docs/rank1-sum-release-timings.md",
    ),
    (
        "Rank-9 `Tensor.sum` release timings",
        "docs/rank9-sum-release-timings.md",
    ),
    (
        "Rank-10 `Tensor.sum` release timings",
        "docs/rank10-sum-release-timings.md",
    ),
    (
        "Rank-11 `Tensor.sum` release timings",
        "docs/rank11-sum-release-timings.md",
    ),
    (
        "Rank-12 `Tensor.sum` release timings",
        "docs/rank12-sum-release-timings.md",
    ),
    (
        "`+` and `Tensor.add` release timings",
        "docs/tensor-add-release-timings.md",
    ),
    (
        "`*`, `Tensor.mul`/`Tensor.multiply`, and "
        "`torch.mul`/`torch.multiply` release timings",
        "docs/tensor-mul-release-timings.md",
    ),
    (
        "Rank-2 `@`, `Tensor.matmul`, and `torch.matmul` release timings",
        "docs/rank2-matmul-release-timings.md",
    ),
    (
        "`Tensor.abs` and `torch.abs` release timings",
        "docs/tensor-abs-release-timings.md",
    ),
    (
        "`Tensor.sqrt` and `torch.sqrt` release timings",
        "docs/tensor-sqrt-release-timings.md",
    ),
    (
        "`Tensor.reciprocal` and `torch.reciprocal` release timings",
        "docs/tensor-reciprocal-release-timings.md",
    ),
    (
        "`Tensor.view`, reshape, flatten, ravel, unbind, and edge-unsqueeze release timings",
        "docs/tensor-view-release-timings.md",
    ),
    (
        "`torch.nn.functional.mse_loss` release timings",
        "docs/mse-loss-release-timings.md",
    ),
    (
        '`torch.nn.functional.l1_loss(reduction="none")` release timings',
        "docs/l1-loss-release-timings.md",
    ),
)
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
    "grad/autograd state queries",
    "eager JIT helper decorators and state queries",
    "Explicit unsupported APIs",
)
SUPPORTED_SURFACE_NAMESPACE_SUMMARIES = (
    (
        "torch",
        (
            "`torch.tensor`",
            "`torch.sum`",
            "`torch.autograd.backward`",
            "[Creation](#creation)",
            "[Backend and compiler metadata](#backend-and-compiler-metadata)",
        ),
    ),
    (
        "Tensor",
        (
            "`Tensor.view`",
            "`Tensor.backward`",
            "[Metadata and views](#metadata-and-views)",
        ),
    ),
    (
        "torch.nn.functional",
        (
            "`torch.nn.functional.linear`",
            "`torch.nn.functional.dropout3d`",
            "`torch.nn.functional.softsign`",
        ),
    ),
    (
        "torch.cuda",
        (
            "`torch.cuda.device_count`",
            "`torch.cuda.is_available`",
            "`torch.cuda.is_initialized`",
            "`torch.cuda.max_memory_allocated`",
        ),
    ),
    (
        "torch.backends",
        (
            "`torch.backends.cpu.get_cpu_capability`",
            "`torch.backends.cuda.sdp_kernel`",
            "`torch.backends.mha.get_fastpath_enabled`",
        ),
    ),
    (
        "torch.compiler",
        (
            "`torch.compiler.disable`",
            "`torch.compiler.skip_all_guards_unsafe`",
        ),
    ),
    (
        "torch.jit",
        (
            "`torch.jit.Attribute`",
            "`torch.jit.optimized_execution`",
        ),
    ),
    (
        "torch.distributed",
        (
            "`torch.distributed.is_available`",
            "`torch.distributed.get_node_local_rank`",
        ),
    ),
    (
        "torch.utils.data",
        (
            "`torch.utils.data.TensorDataset`",
            "`torch.utils.data.DistributedSampler`",
            "`torch.utils.data.get_worker_info`",
        ),
    ),
)
SUPPORTED_SURFACE_TASK_INDEX_ROWS = (
    (
        "Create CPU `float32` tensors",
        ("`torch.tensor`", "`torch.as_tensor`", "`torch.zeros`"),
        ("dtype conversions", "accelerator or meta devices", "concrete `out`"),
        ("[Tensors](#tensors)", "[Creation](#creation)"),
    ),
    (
        "Preserve or change tensor layout",
        (
            "`Tensor.select`",
            "`torch.select`",
            "`Tensor.unbind`",
            "`torch.unbind`",
            "`Tensor.view`",
            "`Tensor.reshape`",
            "`torch.reshape`",
            "`Tensor.cpu`",
        ),
        (
            "range slicing",
            "advanced indexing",
            "sequence `movedim` axes",
            "cross-dtype views",
        ),
        ("[Metadata and views](#metadata-and-views)",),
    ),
    (
        "Run eager math and reductions",
        (
            "Python `+`, `-`, `*`, and `/` operators",
            "`Tensor.add`",
            "`torch.add`",
            "`torch.matmul`",
            "`torch.sum`",
        ),
        (
            "scalar-left and scalar-only `torch.add` calls",
            "in-place variants",
            "dimension reductions",
        ),
        ("[Elementwise and reductions](#elementwise-and-reductions)",),
    ),
    (
        "Use functional NN helpers",
        (
            "`torch.nn.functional.linear`",
            "`torch.nn.functional.mse_loss`",
            "`torch.nn.functional.dropout1d`",
            "`torch.nn.functional.softsign`",
        ),
        (
            "Module layers",
            '`mse_loss` reductions other than `"none"`/`"mean"`/`"sum"`',
            "loss `weight` arguments",
            "mutating initializers",
        ),
        (
            "[NN/data helpers](#nn-and-data-helpers)",
            "[math activations](#elementwise-and-reductions)",
        ),
    ),
    (
        "Reuse data and state helpers",
        (
            "`torch.utils.data.TensorDataset`",
            "`torch.utils.data.DistributedSampler`",
            "`torch.serialization.get_default_load_endianness`",
        ),
        ("`DataLoader`", "worker processes", "`torch.load`"),
        ("[NN/data helpers](#nn-and-data-helpers)",),
    ),
    (
        "Check dtype, device, and backend state",
        (
            "`torch.float32`",
            "`torch.finfo`",
            "`torch.get_device`",
            "`torch.cpu.current_device`",
            "`torch.cpu.synchronize`",
            "`torch.cpu.set_device`",
            "`torch.cuda.max_memory_allocated`",
            "`torch.accelerator.empty_cache`",
            "`torch.accelerator.reset_accumulated_memory_stats`",
            "`torch.accelerator.reset_peak_memory_stats`",
            "`torch.accelerator.memory_allocated`",
            "`torch.accelerator.max_memory_allocated`",
            "`torch.accelerator.memory_reserved`",
            "`torch.accelerator.max_memory_reserved`",
            "`torch.backends.nnpack.set_flags`",
            "`torch.backends.cuda.enable_flash_sdp`",
            "`torch.backends.cuda.enable_cudnn_sdp`",
            "`torch.backends.cudnn.benchmark_limit`",
            "`torch.backends.mha.get_fastpath_enabled`",
        ),
        (
            "Additional dtypes",
            "CUDA tensors/transfers/streams/events/synchronization/runtime/kernels",
            "memory-management APIs outside the named helper set",
            "backend APIs outside [Backend and compiler metadata]",
        ),
        (
            "[tensor metadata](#metadata-and-views)",
            "[backend metadata](#backend-and-compiler-metadata)",
        ),
    ),
    (
        "Control eager autograd state",
        (
            "`Tensor.backward`",
            "`torch.is_grad_enabled`",
            "`torch.no_grad`",
            "`torch.autograd.is_view_replay_enabled`",
        ),
        ("Concrete gradients", "`torch.autograd.grad`", "inference-mode contexts"),
        (
            "[Metadata and views](#metadata-and-views)",
            "[Backend and compiler metadata](#backend-and-compiler-metadata)",
        ),
    ),
    (
        "Integrate eager compiler, JIT, and distributed probes",
        (
            "`torch.compiler.disable`",
            "`torch.jit.annotate`",
            "`torch.distributed.is_available`",
            "`torch.distributed.is_gloo_available`",
            "`torch.distributed.is_mpi_available`",
            "`torch.distributed.is_nccl_available`",
            "`torch.distributed.is_ucc_available`",
            "`torch.distributed.is_xccl_available`",
            "`torch.distributed.get_backend_config`",
            "`torch.distributed.get_backend`",
            "`torch.distributed.get_rank`",
            "`torch.distributed.get_world_size`",
            "`torch.distributed.get_process_group_ranks`",
            "`torch.distributed.get_node_local_rank`",
        ),
        (
            "Actual `torch.compile`",
            "TorchScript compilation",
            "process-group creation",
            "initialized backend/config/rank/world-size access",
            "distributed APIs outside [Backend and compiler metadata]",
        ),
        ("[Backend and compiler metadata](#backend-and-compiler-metadata)",),
    ),
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

    def test_benchmarking_indexes_historical_release_timing_reports(self):
        benchmarking = BENCHMARKING.read_text(encoding="utf-8")
        match = re.search(
            r"^## Historical release timing reports\n(?P<section>.*?)^## ",
            benchmarking,
            flags=re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(
            match, "benchmarking historical timing report section is missing"
        )

        section = match.group("section")
        normalized_section = re.sub(r"\s+", " ", section).lower()
        self.assertIn("historical release evidence snapshots", normalized_section)
        self.assertIn("not live gates", normalized_section)
        self.assertIn("burner-managed evaluation progress", normalized_section)

        links = dict(re.findall(r"\[([^\]]+)\]\(([^)]+)\)", section))
        expected_targets = {target for _, target in HISTORICAL_TIMING_REPORTS}
        indexed_targets = set(links.values())
        existing_reports = {
            path.relative_to(REPOSITORY_ROOT).as_posix()
            for path in (REPOSITORY_ROOT / "docs").glob("*-release-timings.md")
        }
        self.assertEqual(indexed_targets, expected_targets)
        self.assertEqual(indexed_targets, existing_reports)

        for label, target in HISTORICAL_TIMING_REPORTS:
            with self.subTest(report=target):
                self.assertEqual(links.get(label), target)
                path = (REPOSITORY_ROOT / target).resolve()
                self.assertTrue(path.is_relative_to(REPOSITORY_ROOT))
                self.assertTrue(path.is_file())

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
        scope_match = re.search(
            r"^## Scope\n(?P<section>.*?)^## Evaluation$",
            readme,
            flags=re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(scope_match, "README scope section is missing")
        scope = scope_match.group("section")
        self.assertIn(
            "[exhaustive supported surface](docs/supported-surface.md)", scope
        )
        self.assertNotRegex(scope, r"docs/supported-surface\.md#")
        self.assertTrue(SUPPORTED_SURFACE.is_file())

        supported = SUPPORTED_SURFACE.read_text(encoding="utf-8")
        self.assertIn("## Common adopter task quick index", supported)
        self.assertIn("## Namespace summary", supported)
        self.assertIn("## Category index", supported)
        self.assertIn("## Current baseline", supported)
        self.assertLess(
            supported.index("## Common adopter task quick index"),
            supported.index("## Namespace summary"),
        )
        self.assertLess(
            supported.index("## Namespace summary"),
            supported.index("## Category index"),
        )
        self.assertLess(
            supported.index("## Category index"),
            supported.index("## Current baseline"),
        )

        task_index = supported[
            supported.index("## Common adopter task quick index") : supported.index(
                "## Namespace summary"
            )
        ]
        self.assertIn(
            "| Adopter task | Supported APIs | Unsupported boundaries to verify |",
            task_index,
        )
        self.assertIn("| --- | --- | --- |", task_index)
        self.assertNotIn("other accelerator memory-management APIs", task_index)
        self.assertNotIn("unlisted backend APIs", task_index)
        self.assertNotIn("remaining distributed APIs", task_index)
        for label, calls, boundaries, links in SUPPORTED_SURFACE_TASK_INDEX_ROWS:
            with self.subTest(task_index_row=label):
                self.assertIn(f"| {label} |", task_index)
                for call in calls:
                    self.assertIn(call, task_index)
                for boundary in boundaries:
                    self.assertIn(boundary, task_index)
                for link in links:
                    self.assertIn(link, task_index)

        namespace_summary = supported[
            supported.index("## Namespace summary") : supported.index(
                "## Category index"
            )
        ]
        self.assertIn(
            "| Focus | APIs at a glance | Detailed contract |",
            namespace_summary,
        )
        self.assertEqual(
            namespace_summary.count(
                "| Focus | APIs at a glance | Detailed contract |"
            ),
            len(SUPPORTED_SURFACE_NAMESPACE_SUMMARIES),
        )
        for title, snippets in SUPPORTED_SURFACE_NAMESPACE_SUMMARIES:
            with self.subTest(namespace_summary=title):
                self.assertRegex(
                    namespace_summary,
                    rf"(?m)^### {re.escape(title)}$",
                )
                for snippet in snippets:
                    self.assertIn(snippet, namespace_summary)

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
