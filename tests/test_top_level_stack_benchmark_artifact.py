import copy
from collections import Counter
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_top_level_stack.py"
VALIDATOR_SCRIPT = (
    REPOSITORY_ROOT / "scripts" / "validate_top_level_stack_benchmark.py"
)

spec = importlib.util.spec_from_file_location(
    "_torch_rs_top_level_stack_benchmark_for_tests",
    BENCHMARK_SCRIPT,
)
benchmark_top_level_stack = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = benchmark_top_level_stack
spec.loader.exec_module(benchmark_top_level_stack)

validator_spec = importlib.util.spec_from_file_location(
    "_torch_rs_top_level_stack_validator_for_tests",
    VALIDATOR_SCRIPT,
)
validate_top_level_stack_benchmark = importlib.util.module_from_spec(validator_spec)
assert validator_spec.loader is not None
sys.modules[validator_spec.name] = validate_top_level_stack_benchmark
validator_spec.loader.exec_module(validate_top_level_stack_benchmark)


def _has_reference_torch_2_13():
    try:
        import torch as reference_torch
    except ImportError:
        return False
    return reference_torch.__version__.split("+", 1)[0] == "2.13.0"


class StackRuntimePathTests(unittest.TestCase):
    def setUp(self):
        target = REPOSITORY_ROOT / "target"
        target.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="stack-paths-", dir=target)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.venv = self.root / ".venv"
        self.packages = self.venv / "lib" / "python-test" / "site-packages"
        self.packages.mkdir(parents=True)
        self.module = self.packages / "module.py"
        self.module.write_text("# fixture\n", encoding="utf-8")
        self.other = self.packages / "other.py"
        self.other.write_bytes(self.module.read_bytes())
        self.alias = self.venv / "lib64"
        try:
            self.alias.symlink_to("lib", target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks unavailable: {error}")
        self.alias_module = self.alias / self.module.relative_to(self.venv / "lib")

    def check_paths(self, recorded, current, *, resolve_aliases=True):
        errors = []
        validate_top_level_stack_benchmark._validate_recorded_path(
            errors,
            "package.path",
            str(recorded),
            str(current),
            root=self.venv,
            resolve_aliases=resolve_aliases,
        )
        return "\n".join(errors)

    def test_package_aliases_compare_by_resolved_identity(self):
        for recorded in (self.module, self.alias_module):
            for current in (self.module, self.alias_module):
                with self.subTest(recorded=recorded, current=current):
                    self.assertEqual(self.check_paths(recorded, current), "")

    def test_different_existing_files_are_not_aliases(self):
        self.assertIn("mismatch", self.check_paths(self.alias_module, self.other))

    def test_missing_broken_looping_and_directory_paths_are_rejected(self):
        broken = self.packages / "broken.py"
        broken.symlink_to("absent.py")
        loop = self.packages / "loop.py"
        loop.symlink_to(loop.name)
        for invalid in (self.packages / "absent.py", broken, loop, self.packages):
            for recorded, current in ((invalid, self.module), (self.module, invalid)):
                with self.subTest(recorded=recorded, current=current):
                    self.assertTrue(self.check_paths(recorded, current))

    def test_outside_origins_and_symlink_escapes_are_rejected(self):
        # All simulated parent, sibling, and global installs stay in the fixture.
        for name in ("parent", "sibling", "global"):
            outside = self.root / name / "site-packages" / "module.py"
            outside.parent.mkdir(parents=True)
            outside.write_bytes(self.module.read_bytes())
            escape = self.packages / f"{name}.py"
            escape.symlink_to(outside)
            for recorded, current in (
                (outside, outside),
                (escape, escape),
                (escape, self.module),
                (self.module, escape),
            ):
                with self.subTest(recorded=recorded, current=current):
                    self.assertIn("outside", self.check_paths(recorded, current))

    def test_external_alias_back_into_environment_is_rejected(self):
        external_alias = self.root / "external.py"
        external_alias.symlink_to(self.module)
        self.assertIn("outside", self.check_paths(external_alias, self.module))
        self.assertIn("outside", self.check_paths(self.module, external_alias))

    def test_symlinked_environment_is_rejected(self):
        actual = self.root / "external-venv"
        self.venv.rename(actual)
        self.venv.symlink_to(actual, target_is_directory=True)
        self.assertIn("resolves outside", self.check_paths(self.module, self.module))

    def test_parent_traversal_and_relative_paths_are_rejected(self):
        traversal = self.venv / ".." / ".venv" / self.module.relative_to(self.venv)
        self.assertIn("outside", self.check_paths(traversal, self.module))
        self.assertIn("not absolute", self.check_paths("module.py", self.module))
        self.assertIn("not absolute", self.check_paths(self.module, "module.py"))

    def test_interpreter_symlink_keeps_lexical_identity(self):
        base = self.root / "base-python"
        base.write_bytes(b"interpreter fixture")
        executable = self.venv / "bin" / "python"
        executable.parent.mkdir()
        executable.symlink_to(base)
        other_executable = executable.with_name("python3")
        other_executable.symlink_to(base)
        self.assertEqual(
            self.check_paths(executable, executable, resolve_aliases=False), ""
        )
        self.assertIn("outside", self.check_paths(base, executable, resolve_aliases=False))
        self.assertIn("outside", self.check_paths(executable, base, resolve_aliases=False))
        self.assertIn(
            "mismatch", self.check_paths(other_executable, executable, resolve_aliases=False)
        )
        self.assertIn("resolves outside", self.check_paths(executable, executable))

    def test_runtime_origin_prefix_loader_and_native_identity_checks(self):
        validator = validate_top_level_stack_benchmark
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
        native_path = self.packages / f"torch_rs{suffix}"
        native_path.write_bytes(b"native fixture")

        def module(path, *, native=False):
            loader = (
                importlib.machinery.ExtensionFileLoader("torch_rs.torch_rs", str(path))
                if native
                else None
            )
            return SimpleNamespace(
                __spec__=SimpleNamespace(origin=str(path), loader=loader),
                __version__="fixture",
            )

        alias_native = self.alias / native_path.relative_to(self.venv / "lib")
        native = module(alias_native, native=True)
        package = module(self.module)
        package._C = module(native_path, native=True)
        modules = {
            "numpy": module(self.module),
            "torch": module(self.module),
            "torch_rs": package,
            "torch_rs.torch_rs": native,
        }
        executable = self.venv / "python"
        executable.write_bytes(b"interpreter fixture")
        environment = {
            "python_executable": str(executable),
            **{
                name: {"version": "fixture", "path": str(self.alias_module)}
                for name in ("numpy", "pytorch", "torch_rs")
            },
        }
        environment["torch_rs"]["extension_path"] = str(native_path)

        def validate():
            errors = []
            validator._validate_current_runtime_paths(errors, environment)
            return "\n".join(errors)

        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(validator, "REPOSITORY_ROOT", self.root),
            mock.patch.object(sys, "executable", str(executable)),
            mock.patch.object(sys, "prefix", str(self.venv)),
            mock.patch.object(
                validator.benchmark_top_level_stack,
                "_package_version",
                return_value="fixture",
            ),
        ):
            self.assertEqual(validate(), "")
            with mock.patch.object(sys, "prefix", str(self.root)):
                self.assertIn("prefix mismatch", validate())
            with mock.patch.object(modules["numpy"].__spec__, "origin", str(self.other)):
                self.assertIn("numpy.path mismatch", validate())
            with mock.patch.object(native.__spec__, "loader", None):
                self.assertIn("not a native extension", validate())
            with mock.patch.object(package, "_C", module(self.other)):
                self.assertIn("torch_rs._C identity mismatch", validate())
            with mock.patch.object(native.__spec__, "origin", str(self.other)):
                self.assertIn("unrecognized ABI suffix", validate())
            with mock.patch.object(
                native.__spec__, "origin", str(self.packages / f"absent{suffix}")
            ):
                self.assertIn("cannot resolve", validate())
            with mock.patch.object(modules["numpy"].__spec__, "origin", None):
                self.assertIn("current numpy.path is unavailable", validate())
            with mock.patch.dict(environment["numpy"], {"path": None}):
                self.assertIn("missing provenance field numpy.path", validate())
            with mock.patch.dict(environment["numpy"], {"version": "wrong"}):
                self.assertIn("numpy.version mismatch", validate())
            outside = self.root / f"global{suffix}"
            outside.write_bytes(b"outside environment fixture")
            for section, key, imported in (
                ("numpy", "path", modules["numpy"]),
                ("pytorch", "path", modules["torch"]),
                ("torch_rs", "path", package),
                ("torch_rs", "extension_path", native),
            ):
                with (
                    self.subTest(section=section, key=key),
                    mock.patch.dict(environment[section], {key: str(outside)}),
                    mock.patch.object(imported.__spec__, "origin", str(outside)),
                ):
                    self.assertIn(f"{section}.{key} is outside", validate())


class TopLevelStackBenchmarkArtifactTests(unittest.TestCase):
    def test_checked_in_raw_artifact_matches_markdown_summary(self):
        benchmark_top_level_stack.validate_artifact(
            benchmark_top_level_stack.DEFAULT_ARTIFACT_PATH,
            benchmark_top_level_stack.DEFAULT_MARKDOWN_REPORT_PATH,
        )

    def test_workload_matrix_covers_stack_policy_categories(self):
        categories = {workload.category for workload in benchmark_top_level_stack.WORKLOADS}

        self.assertEqual(
            categories,
            {
                "scalar",
                "vector",
                "matrix",
                "empty",
                "offset",
                "noncontiguous",
                "autograd forward",
                "autograd forward+backward",
            },
        )
        self.assertEqual(len(benchmark_top_level_stack.WORKLOADS), 8)

    def test_boundary_rows_cover_stack_policy_cases(self):
        self.assertEqual(
            {
                cell.name
                for cell in benchmark_top_level_stack.UNSUPPORTED_CELLS
                if cell.credit == benchmark_top_level_stack.CREDIT_ZERO
            },
            {
                "mixed_metadata",
                "concrete_out",
            },
        )
        self.assertEqual(
            {
                cell.name
                for cell in benchmark_top_level_stack.UNSUPPORTED_CELLS
                if cell.credit == benchmark_top_level_stack.CREDIT_ERROR_PARITY
            },
            {
                "empty_input_sequence",
                "mixed_shapes",
            },
        )

    def test_generated_validator_cases_are_held_out_and_deterministic(self):
        first = validate_top_level_stack_benchmark.generate_cases(
            seed=20260908,
            cases_per_category=1,
            max_elements=4096,
        )
        second = validate_top_level_stack_benchmark.generate_cases(
            seed=20260908,
            cases_per_category=1,
            max_elements=4096,
        )
        self.assertEqual(first, second)

        self.assertEqual(
            {case.category for case in first},
            set(validate_top_level_stack_benchmark.REQUIRED_CATEGORIES),
        )
        self.assertEqual(len({case.name for case in first}), len(first))
        for case in first:
            with self.subTest(case=case.name):
                self.assertNotIn(
                    case.shape,
                    validate_top_level_stack_benchmark.PUBLIC_INPUT_SHAPES,
                )
                self.assertLessEqual(
                    validate_top_level_stack_benchmark._product(case.shape),
                    4096,
                )
                self.assertEqual(
                    len(case.input_source_indices),
                    3
                    if case.category == "autograd forward+backward"
                    else len(case.seeds),
                )

        workloads = validate_top_level_stack_benchmark._workloads_for_cases(first)
        self.assertEqual(len(workloads), len(first))
        self.assertEqual(
            {workload.name for workload in workloads},
            {case.name for case in first},
        )
        self.assertTrue(all(callable(workload.make_operands) for workload in workloads))

    @unittest.skipUnless(
        _has_reference_torch_2_13(),
        "requires pinned PyTorch 2.13 reference dependency",
    )
    def test_generated_validator_smoke_artifact_validates(self):
        target_dir = REPOSITORY_ROOT / "target"
        target_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="stack-validator-",
            dir=target_dir,
        ) as temporary_directory:
            artifact_path = Path(temporary_directory) / "stack-validator.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_SCRIPT),
                    "--seed",
                    "20260908",
                    "--cases-per-category",
                    "1",
                    "--max-elements",
                    "4096",
                    "--warmups",
                    "1",
                    "--samples",
                    "1",
                    "--output",
                    str(artifact_path),
                ],
                check=False,
                capture_output=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                text=True,
                timeout=120,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )

            report = json.loads(artifact_path.read_text(encoding="utf-8"))
            for section, key in validate_top_level_stack_benchmark.RUNTIME_PACKAGE_PATHS:
                path = Path(report["environment"][section][key])
                self.assertEqual(path, path.resolve(strict=True))
                self.assertTrue(path.is_relative_to(REPOSITORY_ROOT / ".venv"))
            self.assertEqual(report["environment"]["python_executable"], sys.executable)
            validate_top_level_stack_benchmark.validate_artifact_dict(
                report,
                expected_seed=20260908,
                expected_cases_per_category=1,
                expected_max_elements=4096,
                expected_warmups=1,
                expected_samples=1,
                expected_threads=1,
                require_clean_git=False,
            )
            current_git = benchmark_top_level_stack._git_provenance()
            clean_current_git = {
                "head": current_git["head"],
                "status_short": "",
                "diff_stat": "",
            }
            report = copy.deepcopy(report)
            report["environment"]["git"] = clean_current_git
            artifact_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            validate_top_level_stack_benchmark.validate_artifact_dict(
                report,
                expected_seed=20260908,
                expected_cases_per_category=1,
                expected_max_elements=4096,
                expected_warmups=1,
                expected_samples=1,
                expected_threads=1,
                current_git_provenance=clean_current_git,
            )
            self.assertEqual(report["validator"]["seed"], 20260908)
            self.assertEqual(report["validator"]["cases_per_category"], 1)
            self.assertEqual(
                report["environment"]["benchmark_integrity"]["workload_set"],
                validate_top_level_stack_benchmark.WORKLOAD_SET,
            )
            self.assertEqual(report["aggregates"]["timed_supported_cell_count"], 6)
            self.assertEqual(
                report["aggregates"]["generated_category_counts"],
                {
                    category: 1
                    for category in validate_top_level_stack_benchmark.REQUIRED_CATEGORIES
                },
            )
            self.assertEqual(
                set(report["aggregates"]["groups"]),
                {
                    "all supported cells",
                    *{
                        f"{category} cells"
                        for category in validate_top_level_stack_benchmark.REQUIRED_CATEGORIES
                    },
                },
            )
            self.assertEqual(
                report["aggregates"]["groups"]["contiguous cells"]["cell_count"],
                1,
            )
            for case in report["cases"]:
                with self.subTest(case=case["workload"]):
                    self.assertTrue(case["generated"])
                    self.assertTrue(case["validation"]["held_out_generated_shape"])
                    self.assertTrue(
                        case["validation"]["same_shape_cpu_float32_inputs"]
                    )
                    self.assertNotIn(
                        tuple(case["shape"]),
                        validate_top_level_stack_benchmark.PUBLIC_INPUT_SHAPES,
                    )

            def rebuild_aggregates_from_rows(artifact):
                supported = artifact["cases"]
                unsupported = artifact["zero_credit_unsupported_cells"]
                error_parity = artifact["boundary_error_parity_cells"]
                aggregates = (
                    validate_top_level_stack_benchmark._aggregate_generated_rows(
                        supported
                    )
                )
                aggregates["zero_credit_unsupported_cell_count"] = len(unsupported)
                aggregates["boundary_error_parity_cell_count"] = len(error_parity)
                aggregates["combined_capped_with_zero_credit_unsupported"] = (
                    benchmark_top_level_stack._geomean(
                        [
                            min(
                                10.0,
                                max(
                                    0.10,
                                    row["ratios"][
                                        "steady_torch_rs_over_pytorch"
                                    ],
                                ),
                            )
                            for row in supported
                        ]
                        + [10.0] * len(unsupported)
                    )
                )
                aggregates["generated_category_counts"] = dict(
                    sorted(Counter(row["category"] for row in supported).items())
                )
                artifact["aggregates"] = aggregates

            def tamper_ratio_and_rebuild_aggregates(artifact):
                artifact["cases"][0]["ratios"][
                    "steady_torch_rs_over_pytorch"
                ] = 123.0
                rebuild_aggregates_from_rows(artifact)

            def strip_measured_checksums(artifact):
                for row in artifact["cases"]:
                    for implementation in ("torch_rs", "pytorch"):
                        for pass_result in row["implementations"][implementation][
                            "passes"
                        ]:
                            pass_result["steady_checksums"] = []
                            pass_result["warmup_checksums"] = []

            def forge_output_checksums(artifact):
                fake_checksum = "fabricated-checksum"

                def checksum_sink(count):
                    sink = "0"
                    for _ in range(count):
                        sink = benchmark_top_level_stack._roll_checksum(
                            sink,
                            fake_checksum,
                        )
                    return sink

                steady_sink = checksum_sink(artifact["environment"]["samples"])
                warmup_sink = checksum_sink(artifact["environment"]["warmups"])
                for row in artifact["cases"]:
                    row["validation"]["reference_checksum"] = fake_checksum
                    for implementation in ("torch_rs", "pytorch"):
                        implementation_result = row["implementations"][implementation]
                        implementation_result["checksums"] = [fake_checksum]
                        for pass_result in implementation_result["passes"]:
                            pass_result["cold_checksum"] = fake_checksum
                            pass_result["steady_checksums"] = [fake_checksum]
                            pass_result["steady_checksum_sink"] = steady_sink
                            pass_result["warmup_checksums"] = [fake_checksum]
                            pass_result["warmup_checksum_sink"] = warmup_sink

            tamper_cases = (
                (
                    "seed",
                    lambda artifact: (
                        artifact["validator"].__setitem__("seed", 1),
                        artifact["environment"]["validator"].__setitem__("seed", 1),
                    ),
                    "validator generated cases do not match seed/config",
                ),
                (
                    "aggregate",
                    lambda artifact: artifact["aggregates"].__setitem__(
                        "steady_geomean_torch_rs_over_pytorch",
                        123.0,
                    ),
                    "aggregates.steady_geomean_torch_rs_over_pytorch mismatch",
                ),
                (
                    "pass-summary",
                    lambda artifact: artifact["cases"][0]["implementations"][
                        "torch_rs"
                    ]["passes"][0]["steady"].__setitem__(
                        "median_us",
                        123.0,
                    ),
                    "steady.median_us mismatch",
                ),
                (
                    "implementation-median",
                    lambda artifact: artifact["cases"][0]["implementations"][
                        "torch_rs"
                    ].__setitem__(
                        "steady_median_us",
                        123.0,
                    ),
                    "torch_rs.steady_median_us mismatch",
                ),
                (
                    "ratio-with-rebuilt-aggregates",
                    tamper_ratio_and_rebuild_aggregates,
                    "ratios.steady_torch_rs_over_pytorch mismatch",
                ),
                (
                    "row-shape",
                    lambda artifact: artifact["cases"][0].__setitem__(
                        "shape",
                        [999],
                    ),
                    "shape mismatch",
                ),
                (
                    "row-layout",
                    lambda artifact: artifact["cases"][0].__setitem__(
                        "layout",
                        "fabricated",
                    ),
                    "layout mismatch",
                ),
                (
                    "measured-checksums-stripped",
                    strip_measured_checksums,
                    "steady_checksums mismatch",
                ),
                (
                    "output-checksums-forged",
                    forge_output_checksums,
                    "reference checksum mismatch",
                ),
                (
                    "steady-checksum-sink",
                    lambda artifact: artifact["cases"][0]["implementations"][
                        "torch_rs"
                    ]["passes"][0].__setitem__(
                        "steady_checksum_sink",
                        "0",
                    ),
                    "steady_checksum_sink mismatch",
                ),
                (
                    "driver-sha",
                    lambda artifact: artifact["environment"]["driver"].__setitem__(
                        "sha256",
                        "0" * 64,
                    ),
                    "driver SHA-256 does not match the checked-in script",
                ),
                (
                    "git-status",
                    lambda artifact: artifact["environment"]["git"].__setitem__(
                        "status_short",
                        " M fabricated.py",
                    ),
                    "git status_short is not clean",
                ),
                (
                    "git-diff-stat",
                    lambda artifact: artifact["environment"]["git"].__setitem__(
                        "diff_stat",
                        " scripts/validate_top_level_stack_benchmark.py | 1 +",
                    ),
                    "git diff_stat is not clean",
                ),
                (
                    "git-missing-head",
                    lambda artifact: artifact["environment"]["git"].__setitem__(
                        "head",
                        None,
                    ),
                    "git head is not a full commit hash",
                ),
                (
                    "unsupported-credit",
                    lambda artifact: artifact["zero_credit_unsupported_cells"][
                        0
                    ].__setitem__(
                        "credit",
                        benchmark_top_level_stack.CREDIT_ERROR_PARITY,
                    ),
                    "credit mismatch",
                ),
                (
                    "python-executable",
                    lambda artifact: artifact.__getitem__("environment").__setitem__(
                        "python_executable",
                        str(REPOSITORY_ROOT.parent / "stale-venv" / "bin" / "python"),
                    ),
                    "python_executable is outside",
                ),
                (
                    "numpy-path",
                    lambda artifact: artifact["environment"]["numpy"].__setitem__(
                        "path",
                        str(REPOSITORY_ROOT.parent / "stale" / "numpy.py"),
                    ),
                    "numpy.path is outside",
                ),
                (
                    "torch-rs-path",
                    lambda artifact: artifact["environment"]["torch_rs"].__setitem__(
                        "path",
                        str(
                            REPOSITORY_ROOT.parent
                            / "stale"
                            / "torch_rs"
                            / "__init__.py"
                        ),
                    ),
                    "torch_rs.path is outside",
                ),
                (
                    "torch-rs-extension-path",
                    lambda artifact: artifact["environment"]["torch_rs"].__setitem__(
                        "extension_path",
                        str(REPOSITORY_ROOT.parent / "stale" / "torch_rs.abi3.so"),
                    ),
                    "torch_rs.extension_path is outside",
                ),
            )
            for label, mutate, expected_message in tamper_cases:
                with self.subTest(tamper=label):
                    tampered = copy.deepcopy(report)
                    mutate(tampered)
                    with self.assertRaises(AssertionError) as raised:
                        validate_top_level_stack_benchmark.validate_artifact_dict(
                            tampered,
                            expected_seed=20260908,
                            expected_cases_per_category=1,
                            expected_max_elements=4096,
                            expected_warmups=1,
                            expected_samples=1,
                            expected_threads=1,
                            current_git_provenance=clean_current_git,
                        )
                    self.assertIn(expected_message, str(raised.exception))

            current_git_tamper_cases = (
                (
                    "current-git-status",
                    {
                        **clean_current_git,
                        "status_short": " M scripts/validate_top_level_stack_benchmark.py",
                    },
                    "current git status_short is not clean",
                ),
                (
                    "current-git-diff-stat",
                    {
                        **clean_current_git,
                        "diff_stat": (
                            " scripts/validate_top_level_stack_benchmark.py | 1 +"
                        ),
                    },
                    "current git diff_stat is not clean",
                ),
            )
            for label, current_git_override, expected_message in current_git_tamper_cases:
                with self.subTest(tamper=label):
                    with self.assertRaises(AssertionError) as raised:
                        validate_top_level_stack_benchmark.validate_artifact_dict(
                            report,
                            expected_seed=20260908,
                            expected_cases_per_category=1,
                            expected_max_elements=4096,
                            expected_warmups=1,
                            expected_samples=1,
                            expected_threads=1,
                            current_git_provenance=current_git_override,
                        )
                    self.assertIn(expected_message, str(raised.exception))

            default_validation_completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_SCRIPT),
                    "--validate-artifact",
                    str(artifact_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(default_validation_completed.returncode, 0)
            self.assertIn(
                "environment warmups mismatch",
                default_validation_completed.stdout
                + default_validation_completed.stderr,
            )
            self.assertIn(
                "validator cases_per_category mismatch",
                default_validation_completed.stdout
                + default_validation_completed.stderr,
            )

            strict_validation_completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_SCRIPT),
                    "--validate-artifact",
                    str(artifact_path),
                    "--seed",
                    "20260908",
                    "--cases-per-category",
                    "1",
                    "--max-elements",
                    "4096",
                    "--warmups",
                    "1",
                    "--samples",
                    "1",
                    "--threads",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            strict_validation_output = (
                strict_validation_completed.stdout + strict_validation_completed.stderr
            )
            if (
                current_git.get("status_short") == ""
                and current_git.get("diff_stat") == ""
            ):
                self.assertEqual(
                    strict_validation_completed.returncode,
                    0,
                    msg=strict_validation_output,
                )
            else:
                self.assertNotEqual(strict_validation_completed.returncode, 0)
                self.assertIn(
                    "current git status_short is not clean",
                    strict_validation_output,
                )

            wrong_seed_validation = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_SCRIPT),
                    "--validate-artifact",
                    str(artifact_path),
                    "--seed",
                    "222",
                    "--cases-per-category",
                    "1",
                    "--max-elements",
                    "4096",
                    "--warmups",
                    "1",
                    "--samples",
                    "1",
                    "--threads",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(wrong_seed_validation.returncode, 0)
            self.assertIn(
                "validator seed mismatch",
                wrong_seed_validation.stdout + wrong_seed_validation.stderr,
            )

            tampered_path = Path(temporary_directory) / "tampered-stack-validator.json"
            tampered = copy.deepcopy(report)
            tampered["aggregates"][
                "steady_geomean_capped_0_10_10_0"
            ] = 456.0
            tampered_path.write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tampered_validation = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_SCRIPT),
                    "--validate-artifact",
                    str(tampered_path),
                    "--seed",
                    "20260908",
                    "--cases-per-category",
                    "1",
                    "--max-elements",
                    "4096",
                    "--warmups",
                    "1",
                    "--samples",
                    "1",
                    "--threads",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(tampered_validation.returncode, 0)
            self.assertIn(
                "aggregates.steady_geomean_capped_0_10_10_0 mismatch",
                tampered_validation.stdout + tampered_validation.stderr,
            )


if __name__ == "__main__":
    unittest.main()
