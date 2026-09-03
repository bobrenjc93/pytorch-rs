import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README = REPOSITORY_ROOT / "README.md"
BENCHMARKING = REPOSITORY_ROOT / "BENCHMARKING.md"
CONTRIBUTING = REPOSITORY_ROOT / "CONTRIBUTING.md"
FEATURES = REPOSITORY_ROOT / "FEATURES.md"
DOCS_README = REPOSITORY_ROOT / "docs" / "README.md"
TROUBLESHOOTING = REPOSITORY_ROOT / "docs" / "troubleshooting.md"
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
        "`Tensor.mean` and `torch.mean` full-reduction release timings",
        "docs/tensor-mean-release-timings.md",
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
        "`torch.div` and `torch.divide` release timings",
        "docs/top-level-division-release-timings.md",
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
        "`torch.nn.functional.softsign` release timings",
        "docs/softsign-release-timings.md",
    ),
    (
        "`torch.compile` eager CPU release timings",
        "docs/torch-compile-cpu-release-timings.md",
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
    (
        '`torch.nn.functional.l1_loss(reduction="sum")` release timings',
        "docs/l1-loss-sum-release-timings.md",
    ),
)
HISTORICAL_TIMING_GROUPS = (
    (
        "Reductions",
        (
            "docs/rank1-sum-release-timings.md",
            "docs/rank9-sum-release-timings.md",
            "docs/rank10-sum-release-timings.md",
            "docs/rank11-sum-release-timings.md",
            "docs/rank12-sum-release-timings.md",
            "docs/tensor-mean-release-timings.md",
        ),
    ),
    (
        "Elementwise ops",
        (
            "docs/tensor-add-release-timings.md",
            "docs/tensor-mul-release-timings.md",
            "docs/top-level-division-release-timings.md",
            "docs/tensor-abs-release-timings.md",
            "docs/tensor-sqrt-release-timings.md",
            "docs/tensor-reciprocal-release-timings.md",
            "docs/softsign-release-timings.md",
        ),
    ),
    (
        "Compilation",
        ("docs/torch-compile-cpu-release-timings.md",),
    ),
    (
        "Layout/view ops",
        ("docs/tensor-view-release-timings.md",),
    ),
    (
        "Linear algebra",
        ("docs/rank2-matmul-release-timings.md",),
    ),
    (
        "NN losses",
        (
            "docs/mse-loss-release-timings.md",
            "docs/l1-loss-release-timings.md",
            "docs/l1-loss-sum-release-timings.md",
        ),
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
    ("CPU and default device", "cpu-and-default-device", "#####"),
    ("Accelerator memory", "accelerator-memory", "#####"),
    ("Grad and autocast state", "grad-and-autocast-state", "#####"),
    ("Backend flags", "backend-flags", "#####"),
    ("JIT and compiler", "jit-and-compiler", "#####"),
    ("Distributed support", "distributed-support", "#####"),
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
            "`torch.compile`",
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
            "`torch.compiler.register_backend`",
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
            "scalar-only `torch.add` calls",
            "in-place variants",
            "dimension reductions",
        ),
        ("[Elementwise and reductions](#elementwise-and-reductions)",),
    ),
    (
        "Use functional NN helpers",
        (
            "`torch.nn.functional.linear`",
            "`torch.nn.functional.l1_loss`",
            "`torch.nn.functional.mse_loss`",
            "`torch.nn.functional.dropout1d`",
            "`torch.nn.functional.softsign`",
        ),
        (
            "Module layers",
            '`l1_loss` reductions other than `"none"`/`"sum"`',
            '`mse_loss` reductions other than `"none"`/`"mean"`/`"sum"`',
            "loss `weight` arguments",
            "legacy loss reduction arguments",
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
            "`torch.compile`",
            "`torch.compiler.disable`",
            "`torch.compiler.register_backend`",
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
            "`torch.compile` graph capture/execution",
            "installed-PyTorch forwarding",
            "TorchScript compilation",
            "process-group creation",
            "initialized backend/config/rank/world-size access",
            "distributed APIs outside [Backend and compiler metadata]",
        ),
        ("[Backend and compiler metadata](#backend-and-compiler-metadata)",),
    ),
)
README_SCOPE_ROW_LABELS = (
    "Eager CPU tensors",
    "CPU-build device probes",
    "CPU-build backend probes",
    "`torch.compile` eager subset",
    "Larger PyTorch stacks",
)
README_SCOPE_REQUIRED_SNIPPETS = (
    "CPU `float32` tensors",
    "core construction and layout/view operations",
    "selected math and neural-network functions",
    "limited first-order autograd",
    "`torch.cuda.device_count() == 0`",
    "`torch.cuda.is_available() is False`",
    "`torch.cuda.is_initialized() is False`",
    "`torch.set_default_device(...)`",
    "CPU-equivalent no-op",
    "`None` or `\"cpu\"`",
    "`torch.backends.cuda` preference flags",
    "`enable_flash_sdp(...)`",
    "`enable_cudnn_sdp(...)`",
    "`sdp_kernel(...)` as a context manager/decorator",
    "`torch.nn.functional.scaled_dot_product_attention`",
    "CUDA tensors",
    "actual attention-kernel dispatch",
    "CUDA `torch.compile` execution",
    "Device selection",
    "mutable default-device routing",
    "streams",
    "events",
    "synchronization",
    "allocator APIs",
    "runtime initialization",
    "Additional tensor dtypes",
    "non-CPU tensor execution",
    "PyTorch 2.13-shaped argument binding",
    "`disable=True` pass-through",
    "backend default/name resolution through the `torch.compiler` registry",
    "native `backend=\"eager\", fullgraph=True` execution",
    "straight-line one- or two-input CPU `float32` Tensor functions",
    "Tensor `neg`, `abs`, and binary broadcasting `add`",
    "broader graph capture/execution",
    "active `__torch_function__` modes",
    "eager fallback",
    "installed-PyTorch forwarding",
    "callable backend invocation",
    "inductor/CUDA compilation",
    "Full module",
    "optimizer",
    "model-serialization",
    "compiler execution",
    "distributed stacks",
)
DOCS_INDEX_CONTRACTS = (
    (
        "Supported surface",
        "supported-surface.md",
        "Exhaustive Python API coverage and unsupported boundary contract.",
    ),
    (
        "Feature coverage contract",
        "../FEATURES.md",
        "Weighted feature areas and what counts toward coverage.",
    ),
    (
        "Benchmark policy",
        "../BENCHMARKING.md",
        "Correctness gates, measurement rules, provenance, and anti-gaming policy.",
    ),
)
DOCS_INDEX_GUIDES = (
    (
        "Repository README",
        "../README.md",
        "Install commands, first-success example, scope summary, and validation entry points.",
    ),
    (
        "Contributing guide",
        "../CONTRIBUTING.md",
        "Locked setup, environment expectations, test selection, draft workflow, and documentation ownership.",
    ),
    (
        "Setup troubleshooting",
        "troubleshooting.md",
        "Short fixes for common environment, import, reference dependency, and stale wheel failures.",
    ),
    (
        "Architecture map",
        "../ARCHITECTURE.md",
        "Source map for the Rust core, Python bindings, wrappers, and test layout.",
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

    def test_readme_scope_is_scan_friendly_table(self):
        readme = README.read_text(encoding="utf-8")
        match = re.search(
            r"^## Scope\n(?P<section>.*?)^## Evaluation$",
            readme,
            flags=re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match, "README scope section is missing")

        section = match.group("section")
        table, route = section.strip().split("\n\n", maxsplit=1)
        table_lines = table.splitlines()
        self.assertEqual(
            table_lines[0],
            "| Surface | Supported today | Unsupported boundary |",
        )
        self.assertEqual(table_lines[1], "| --- | --- | --- |")
        self.assertEqual(len(table_lines), len(README_SCOPE_ROW_LABELS) + 2)
        self.assertIn(
            "[exhaustive supported surface](docs/supported-surface.md)", route
        )
        self.assertNotIn("The current native backend supports eager CPU", section)

        for row_label in README_SCOPE_ROW_LABELS:
            with self.subTest(scope_row=row_label):
                self.assertIn(f"| {row_label} |", table)
        for snippet in README_SCOPE_REQUIRED_SNIPPETS:
            with self.subTest(scope_snippet=snippet):
                self.assertIn(snippet, table)

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

        previous_position = -1
        for group, group_targets in HISTORICAL_TIMING_GROUPS:
            with self.subTest(group=group):
                heading = f"### {group}"
                self.assertEqual(section.count(heading), 1)
                position = section.index(heading)
                self.assertGreater(position, previous_position)
                previous_position = position

                group_match = re.search(
                    rf"^### {re.escape(group)}\n(?P<group>.*?)(?=^### |\Z)",
                    section,
                    flags=re.MULTILINE | re.DOTALL,
                )
                self.assertIsNotNone(group_match)
                group_links = dict(
                    re.findall(r"\[([^\]]+)\]\(([^)]+)\)", group_match.group("group"))
                )
                self.assertEqual(set(group_links.values()), set(group_targets))

    def test_docs_readme_indexes_contracts_guides_and_timing_evidence(self):
        docs_readme = DOCS_README.read_text(encoding="utf-8")
        sections = (
            "## Current Contracts",
            "## Contributor Guides",
            "## Historical Timing Evidence",
        )
        previous_position = docs_readme.index("# Documentation Index")
        for heading in sections:
            with self.subTest(heading=heading):
                self.assertEqual(docs_readme.count(heading), 1)
                position = docs_readme.index(heading)
                self.assertGreater(position, previous_position)
                previous_position = position

        for line in docs_readme.splitlines():
            if line.startswith("- "):
                with self.subTest(line=line):
                    self.assertRegex(line, r"^- \[[^\]]+\]\([^)]+\): \S")

        current_contracts = docs_readme[
            docs_readme.index("## Current Contracts") : docs_readme.index(
                "## Contributor Guides"
            )
        ]
        for label, target, description in DOCS_INDEX_CONTRACTS:
            with self.subTest(contract=target):
                self.assertIn(
                    f"- [{label}]({target}): {description}",
                    current_contracts,
                )
                path = (DOCS_README.parent / target).resolve()
                self.assertTrue(path.is_relative_to(REPOSITORY_ROOT))
                self.assertTrue(path.is_file())

        contributor_guides = docs_readme[
            docs_readme.index("## Contributor Guides") : docs_readme.index(
                "## Historical Timing Evidence"
            )
        ]
        for label, target, description in DOCS_INDEX_GUIDES:
            with self.subTest(guide=target):
                self.assertIn(
                    f"- [{label}]({target}): {description}",
                    contributor_guides,
                )
                path = (DOCS_README.parent / target).resolve()
                self.assertTrue(path.is_relative_to(REPOSITORY_ROOT))
                self.assertTrue(path.is_file())

        timing_evidence = docs_readme[
            docs_readme.index("## Historical Timing Evidence") :
        ]
        normalized_timing_evidence = re.sub(r"\s+", " ", timing_evidence).lower()
        self.assertIn("historical release evidence snapshots", normalized_timing_evidence)
        self.assertIn("not live benchmark gates", normalized_timing_evidence)

        links = dict(re.findall(r"\[([^\]]+)\]\(([^)]+)\):", timing_evidence))
        expected_targets = {
            Path(target).name for _, target in HISTORICAL_TIMING_REPORTS
        }
        self.assertEqual(set(links.values()), expected_targets)
        for _, target in HISTORICAL_TIMING_REPORTS:
            with self.subTest(timing_report=target):
                path = DOCS_README.parent / Path(target).name
                self.assertTrue(path.is_file())

        previous_position = -1
        for group, group_targets in HISTORICAL_TIMING_GROUPS:
            with self.subTest(group=group):
                heading = f"### {group}"
                self.assertEqual(timing_evidence.count(heading), 1)
                position = timing_evidence.index(heading)
                self.assertGreater(position, previous_position)
                previous_position = position

                group_match = re.search(
                    rf"^### {re.escape(group)}\n(?P<group>.*?)(?=^### |\Z)",
                    timing_evidence,
                    flags=re.MULTILINE | re.DOTALL,
                )
                self.assertIsNotNone(group_match)
                group_links = dict(
                    re.findall(
                        r"\[([^\]]+)\]\(([^)]+)\):",
                        group_match.group("group"),
                    )
                )
                self.assertEqual(
                    set(group_links.values()),
                    {Path(target).name for target in group_targets},
                )

    def test_readme_links_contributing_guide(self):
        readme = README.read_text(encoding="utf-8")
        self.assertIn("[docs/README.md](docs/README.md)", readme)
        self.assertIn("[CONTRIBUTING.md](CONTRIBUTING.md)", readme)
        self.assertIn("setup preflight", readme)
        self.assertTrue(DOCS_README.is_file())
        self.assertTrue(CONTRIBUTING.is_file())

        contributing = CONTRIBUTING.read_text(encoding="utf-8")
        self.assertIn("[docs/README.md](docs/README.md)", contributing)
        self.assertIn(
            "[docs/troubleshooting.md](docs/troubleshooting.md)", contributing
        )
        self.assertIn("## Contributor Preflight", contributing)
        self.assertLess(len(contributing.splitlines()), 120)

        troubleshooting = TROUBLESHOOTING.read_text(encoding="utf-8")
        for snippet in (
            "Ambient Python Missing Pytest",
            "`PYTHONPATH=python` Finds Python Files But Not the Native Extension",
            "Missing Reference PyTorch 2.13",
            "Stale Wheel Installs",
        ):
            with self.subTest(troubleshooting=snippet):
                self.assertIn(snippet, troubleshooting)

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
