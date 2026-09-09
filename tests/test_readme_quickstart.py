import re
import runpy
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README = REPOSITORY_ROOT / "README.md"
BENCHMARKING = REPOSITORY_ROOT / "BENCHMARKING.md"
CONTRIBUTING = REPOSITORY_ROOT / "CONTRIBUTING.md"
FIRST_SUCCESS_EXAMPLE = REPOSITORY_ROOT / "examples" / "first_success.py"
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
        "`torch.empty`, `torch.zeros`, and `torch.ones` eager CPU factory timings",
        "docs/creation-factory-release-timings.md",
    ),
    (
        "`+` and `Tensor.add` release timings",
        "docs/tensor-add-release-timings.md",
    ),
    (
        "`torch.sub` and `torch.subtract` release timings",
        "docs/top-level-subtract-release-timings.md",
    ),
    (
        "`torch.stack` release timings",
        "docs/top-level-stack-release-timings.md",
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
        "`torch.compile` H100 CUDA prepared-executor timings",
        "docs/torch-compile-cuda-h100-release-timings.md",
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
        "Creation",
        ("docs/creation-factory-release-timings.md",),
    ),
    (
        "Elementwise ops",
        (
            "docs/tensor-add-release-timings.md",
            "docs/top-level-subtract-release-timings.md",
            "docs/top-level-stack-release-timings.md",
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
        (
            "docs/torch-compile-cpu-release-timings.md",
            "docs/torch-compile-cuda-h100-release-timings.md",
        ),
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
        "Create CPU `float32` tensors and narrow CUDA zeros",
        ("`torch.tensor`", "`torch.as_tensor`", "`torch.zeros`"),
        (
            "dtype conversions",
            "accelerator or meta devices",
            "narrow CUDA zeros path",
            "concrete `out`",
        ),
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
            "variadic top-level reshape dimensions",
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
            '`l1_loss` reductions other than `"none"`/`"mean"`/`"sum"`',
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
            "`torch.cuda.device_count`",
            "`torch.cuda.is_available`",
            "`torch.cuda.is_initialized`",
            "`torch.backends.nnpack.set_flags`",
            "`torch.backends.cuda.enable_flash_sdp`",
            "`torch.backends.cuda.enable_cudnn_sdp`",
            "`torch.backends.cudnn.benchmark_limit`",
            "`torch.backends.mha.get_fastpath_enabled`",
        ),
        (
            "Additional dtypes",
            "CUDA factories beyond 1-D float32 zeros and transfers beyond explicit CPU/CUDA float32 copies without autograd",
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
    "CPU tensors",
    "NVIDIA CUDA",
    "`torch.compile`",
    "Compatibility helpers",
)
README_SCOPE_REQUIRED_SNIPPETS = (
    "Native `float32`",
    "limited first-order autograd",
    "synchronous CPU transfers",
    "same-shape contiguous `float32` addition",
    "1-D zeros only",
    "No general CUDA math or accelerator training",
    "Bounded eager CPU capture",
    "[CUDA neg/add capture](docs/compile-cuda-add.md)",
    "without fusion",
    "No full Inductor compiler",
    "general graph capture, or eager fallback",
    "No additional tensor dtypes or full training stack",
    "device/backend probes",
    "No full module, `DataLoader`, optimizer, model-serialization, or distributed stacks",
)
# Detailed contracts belong in the focused guide, not in the README overview.
SUPPORTED_SURFACE_DETAIL_SNIPPETS = (
    'synchronous `Tensor.to("cuda:N")`',
    "Scalars, empty tensors, contiguous inputs, offset views, transposes",
    "only rank-1 float32 `zeros` can create CUDA storage directly",
    "Autograd inputs (including under `no_grad`), asynchronous copies, dtype changes",
    "Dtype-changing conversions, CUDA-to-CUDA copies, unindexed CUDA targets",
    '`x + y`, `x.add(y)`, and `torch.add(x, y)`',
    "both inputs have identical shapes, are contiguous",
    "numeric default-equivalent `alpha=1`",
    "CUDA broadcasting, noncontiguous operands, Python scalar arithmetic",
    "`torch.cuda.device_count()`",
    "`torch.cuda.is_available()`",
    "`torch.cuda.is_initialized()`",
    "report runtime CUDA visibility without importing PyTorch",
    "current-device allocation for unindexed",
    "streams, events, synchronization APIs, allocator APIs",
    "general runtime management",
    "`torch.set_default_device(device)`",
    "CPU-only compatibility no-op",
    '(`None`, `"cpu"`, unindexed `torch.device("cpu")`',
    "mutable default-device routing",
    "enable_flash_sdp(enabled)",
    "`torch.backends.cuda.sdp_kernel(enable_flash=True",
    "context-manager/decorator",
    "actual attention-kernel dispatch",
    "`torch.nn.functional.scaled_dot_product_attention`",
    "`Tensor.split(split_size, dim=0)`",
    "`torch.split(tensor, split_size_or_sections, dim=0)`",
    "`torch.functional.split`",
    "tuple of shared-storage slice views",
    "last may be shorter",
    "empty split dimension produces one empty view",
    "chunk multi-output backward machinery",
    "list/tuple section sizes",
    "Zero-length sections are preserved",
    "CUDA splits",
    "unsupported dtype/device metadata",
    "`disable=True` returns the original callable without invoking the backend",
    "resolves registered backend names to their registered callables",
    '`backend="eager"` with exact `fullgraph=True`',
    "no-break `fullgraph=False`",
    "one or two positional exact native CPU `float32` Tensor inputs",
    "Tensor `neg`/`negative`, Tensor `abs`/`absolute`, Tensor `relu`",
    "Tensor `square`, Tensor `detach`, zero-argument Tensor `float`",
    "one same-module exact Python helper call",
    "module-global exact native CPU `float32` Tensor constants",
    "tuple/list output pytrees with Tensor leaves",
    "Tensor broadcasting",
    "square decomposition graphlets",
    "storage-aliasing detach graphlets with `requires_grad=False` outputs",
    "preserve values, shape, stride, storage offset, dtype, device, and `requires_grad`",
    "no-grad inference ReLU graphlets",
    "CUDA cache keys additionally guard the exact storage offset and device ordinal",
    "native Rust/CUDA addition kernel without fusion",
    "Negation outputs own fresh CUDA storage with canonical contiguous strides and offset zero",
    "not a general Inductor compiler or a performance-parity claim",
    "private benchmark-only H100 pointwise-reduce workload",
    '`backend="inductor"`, `fullgraph=True`, and `dynamic=False`',
    "global binding identity, and global metadata",
    "exact non-negative integer `recompile_limit` values",
    "`torch.compiler.reset()` clears those native graph caches",
    "active `__torch_function__` modes",
    "`Tensor.float(...)` memory-format arguments, `Tensor.to(...)`",
    "dtype-changing conversion methods, non-Tensor globals",
    "`isolate_recompiles=True`",
    "eager fallback, installed-PyTorch forwarding, backend invocation",
    "CUDA compile autograd/training workloads",
    "general CUDA compile execution",
    "`torch.utils.data.default_collate`",
    "`torch.utils.data.default_convert`",
    "`DataLoader`",
    "optimizers",
    "`torch.save`",
    "`torch.load`",
    "distributed backend initialization/execution",
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
DOCS_INDEX_EXAMPLES = (
    (
        "First-success example",
        "../examples/first_success.py",
        "Runnable version of the README first-success assertions.",
    ),
)
DOCS_INDEX_GUIDES = (
    (
        "CUDA neg/add graph capture",
        "compile-cuda-add.md",
        "Public bounded eager compilation, device/cache guards, unsupported cases, and reproduction.",
    ),
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
        self.assertEqual(table_lines[0], "| Surface | Supported today | Limits |")
        self.assertEqual(table_lines[1], "| --- | --- | --- |")
        self.assertEqual(len(table_lines), len(README_SCOPE_ROW_LABELS) + 2)
        self.assertLessEqual(len(section.split()), 300)
        for line in table_lines[2:]:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            self.assertEqual(len(cells), 3)
            for cell in cells:
                self.assertLessEqual(len(cell.split()), 25, cell)
        for row_label in README_SCOPE_ROW_LABELS:
            with self.subTest(scope_row=row_label):
                self.assertIn(f"| {row_label} |", table)
        for snippet in README_SCOPE_REQUIRED_SNIPPETS:
            with self.subTest(scope_snippet=snippet):
                self.assertIn(snippet, table)
        for snippet in (
            "[exhaustive supported surface](docs/supported-surface.md)",
            "docs/supported-surface.md#jit-and-compiler",
            "private H100 compile benchmark path",
            "fixed workload",
            "`torch.cuda.is_available()`",
            "`torch.cuda.device_count()`",
            "report runtime GPU visibility",
            "backend build flags do not",
            "docs/troubleshooting.md#optional-native-cuda-runtime",
            "`CUDA_VISIBLE_DEVICES=0`",
            "hardware-only cases skip when unavailable",
        ):
            with self.subTest(scope_route=snippet):
                self.assertIn(snippet, " ".join(route.split()))
        self.assertIn("not a general PyTorch replacement", readme)

    def test_detailed_support_contracts_live_in_focused_docs(self):
        supported = " ".join(SUPPORTED_SURFACE.read_text(encoding="utf-8").split())
        for snippet in SUPPORTED_SURFACE_DETAIL_SNIPPETS:
            with self.subTest(supported_contract=snippet):
                self.assertIn(snippet, supported)

    def test_entry_point_links_resolve(self):
        # Check hand-authored entry points, including cross-document anchors.
        sources = {
            README: README.read_text(encoding="utf-8").split("## License", 1)[0],
            SUPPORTED_SURFACE: SUPPORTED_SURFACE.read_text(encoding="utf-8"),
            TROUBLESHOOTING: TROUBLESHOOTING.read_text(encoding="utf-8"),
            REPOSITORY_ROOT / "docs/compile-cuda-add.md": (
                REPOSITORY_ROOT / "docs/compile-cuda-add.md"
            ).read_text(encoding="utf-8"),
        }
        for source, content in sources.items():
            for target in re.findall(r"\[[^\]\n]+\]\(([^)]+)\)", content):
                if "://" in target:
                    continue
                with self.subTest(source=source.name, target=target):
                    filename, _, anchor = target.partition("#")
                    destination = source.parent / filename if filename else source
                    self.assertTrue(destination.is_file(), target)
                    if anchor:
                        headings = re.findall(
                            r"^#{1,6} (.+)$",
                            destination.read_text(encoding="utf-8"),
                            flags=re.MULTILINE,
                        )
                        slugs = {
                            re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-")
                            for heading in headings
                        }
                        self.assertIn(anchor, slugs, target)

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
        self.assertIn(
            "[examples/first_success.py](examples/first_success.py)", readme
        )
        example_source = FIRST_SUCCESS_EXAMPLE.read_text(encoding="utf-8").rstrip()
        self.assertEqual(example_source, source)

        exec(compile(source, f"{README}#first-success", "exec"), {})
        runpy.run_path(str(FIRST_SUCCESS_EXAMPLE))

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
        self.assertIn("## Quick-scan map", features)
        quick_scan_start = features.index("## Quick-scan map")
        detailed_start = features.index("Full-tensor `Tensor.mean")
        weights_start = features.index("Fixed top-level weights")
        self.assertLess(quick_scan_start, detailed_start)
        self.assertLess(detailed_start, weights_start)

        quick_scan = features[quick_scan_start:detailed_start]
        table_lines = [
            line for line in quick_scan.splitlines() if line.startswith("| ")
        ]
        self.assertEqual(
            table_lines[0],
            "| Weighted area | Weight | Primary supported examples | Main unsupported boundary |",
        )
        self.assertEqual(table_lines[1], "| --- | ---: | --- | --- |")

        feature_quick_scan_rows = (
            (
                "tensor storage, shapes, strides, views, indexing",
                "15%",
                ("shared-storage views", "contiguous materialization"),
                ("advanced indexing", "storage-object APIs"),
            ),
            (
                "dtypes, promotion, devices, dispatch",
                "10%",
                ("runtime CUDA probes", "float32 dtype helpers",
                 "synchronous float32 CPU uploads to explicit indexed CUDA devices"),
                ("Broader CUDA factories/operations/runtime", "autograd or asynchronous transfers"),
            ),
            (
                "creation, elementwise, reductions",
                "15%",
                ("full-tensor `sum`/`mean`",),
                ("dimension reductions",),
            ),
            (
                "linear algebra and signal operations",
                "10%",
                ("Rank-2 `matmul`/`mm`",),
                ("`bmm`", "spectral ops"),
            ),
            (
                "autograd and higher-order differentiation",
                "15%",
                ("`Tensor.backward`", "grad-mode helpers"),
                ("`autograd.grad`",),
            ),
            (
                "neural-network functional API and modules",
                "15%",
                ("`l1_loss`/`mse_loss`", "`linear`"),
                ("Modules/parameters",),
            ),
            (
                "optimizers, initialization, data utilities",
                "5%",
                ("`torch.nn.init.calculate_gain`", "dataset and sampler helpers"),
                ("Optimizers", "`DataLoader`"),
            ),
            (
                "serialization, state dictionaries, model interchange",
                "5%",
                ("Serialization option state", "state-dict prefix removal"),
                ("`torch.save`", "`torch.load`"),
            ),
            (
                "compilation, parallelism, distributed execution",
                "5%",
                ("Eager JIT helpers", "narrow eager `torch.compile`"),
                ("TorchScript", "process groups/collectives"),
            ),
            (
                "ergonomics, diagnostics, documentation, ecosystem integration",
                "5%",
                ("Rank-0 `Tensor.__format__`", "native warning policy"),
                ("Deterministic enforcement",),
            ),
        )
        self.assertEqual(len(table_lines), len(feature_quick_scan_rows) + 2)
        for area, weight, supported_examples, unsupported_boundaries in (
            feature_quick_scan_rows
        ):
            with self.subTest(feature_quick_scan_area=area):
                row_prefix = f"| {area} | {weight} |"
                self.assertIn(row_prefix, quick_scan)
                row_start = quick_scan.index(row_prefix)
                row = quick_scan[row_start : quick_scan.index("\n", row_start)]
                for snippet in supported_examples + unsupported_boundaries:
                    self.assertIn(snippet, row)

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
            "## Examples",
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
            docs_readme.index("## Current Contracts") : docs_readme.index("## Examples")
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

        examples = docs_readme[
            docs_readme.index("## Examples") : docs_readme.index(
                "## Contributor Guides"
            )
        ]
        for label, target, description in DOCS_INDEX_EXAMPLES:
            with self.subTest(example=target):
                self.assertIn(
                    f"- [{label}]({target}): {description}",
                    examples,
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

    def test_stale_wheel_recovery_uses_wheel_test_workflow(self):
        troubleshooting = TROUBLESHOOTING.read_text(encoding="utf-8")
        match = re.search(
            r"^## Stale Wheel Installs\n(?P<section>.*?)(?=^## |\Z)",
            troubleshooting,
            flags=re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match, "stale-wheel recovery section is missing")
        commands = re.findall(
            r"^```bash\n(.*?)^```$",
            match.group("section"),
            flags=re.MULTILINE | re.DOTALL,
        )
        # A mention in prose is insufficient: the runnable recovery must build
        # and install a wheel, not feed an editable install to the verifier.
        self.assertEqual(
            [block.strip() for block in commands],
            ["unset PYTHONPATH\n./scripts/test-python.sh"],
        )
        self.assertTrue((REPOSITORY_ROOT / "scripts/test-python.sh").is_file())

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
        self.assertIn("docs/supported-surface.md#jit-and-compiler", scope)
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
