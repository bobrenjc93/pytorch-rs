"""Validate historical setup disclosure without running or scoring workloads.

Schema v1 is specific to the four reports refreshed from e2f40ff. It records a
later same-code rerun; its durations must never be presented as original setup.
Uses only the standard library so CI needs neither a native wheel nor a GPU.
"""

import datetime
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("docs/benchmark-setup/2026-09-09-e2f40ff/setup.json")
REPORTS = {
    "docs/top-level-stack-release-timings.md":
        "docs/benchmark-data/top-level-stack-release-timings.json",
    "docs/top-level-subtract-release-timings.md":
        "docs/benchmark-data/top-level-subtract-release-timings.json",
    "docs/torch-compile-cpu-release-timings.md":
        "docs/benchmark-data/torch-compile-cpu-v4.json",
    "docs/torch-compile-cuda-h100-release-timings.md":
        "docs/benchmark-data/torch-compile-cuda-h100-shape-matrix-v11.json",
}
STEPS = {
    "tools": "provenance",
    "venv": "environment",
    "python-dependencies": "dependency-installation",
    "rust-dependencies": "dependency-installation",
    "release-build": "compile",
    "wheel-install": "project-installation",
    "verify-native": "verification",
    "runtime": "provenance",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def digest(value, length=64):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_file(root, relative):
    require(nonempty(relative), "missing file reference")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "nonlocal file reference")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()) and resolved.is_file(),
            f"missing or nonlocal file: {relative}")
    return resolved


def _validate(record, root):
    require(type(record.get("schema_version")) is int and record["schema_version"] == 1,
            "unsupported setup schema")
    require(record.get("measurement_kind") == "same-code-setup-rerun",
            "setup must be labeled a same-code rerun")
    require(record.get("original_setup_timings") == "unavailable",
            "original setup timings must remain unavailable")
    source = record["source"]
    for name in ("commit", "tree", "capture_checkout_commit"):
        require(digest(source.get(name), 40), f"invalid source {name}")
    require(source.get("archive_command") == ["git", "archive", source["commit"]],
            "source archive command mismatch")
    require(digest(source.get("archive_sha256")), "missing source archive hash")
    require(nonempty(source.get("cwd")), "missing setup cwd")
    require(source.get("files_unchanged_after_setup") is True, "setup changed source files")
    require(type(source.get("archive_files_verified_unchanged")) is int
            and source["archive_files_verified_unchanged"] > 0, "missing archive verification")
    for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml", "uv.lock", "rust-toolchain.toml"):
        require(digest(source["files_sha256"].get(name)), f"missing source hash: {name}")
    for name in ("timing_method", "cache_baseline", "platform", "cpu"):
        require(nonempty(record.get(name)), f"missing {name}")
    for name in ("PATH", "CARGO_HOME", "CARGO_TARGET_DIR", "UV_CACHE_DIR",
                 "TMPDIR", "XDG_CACHE_HOME", "VIRTUAL_ENV", "PYO3_PYTHON",
                 "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        require(nonempty(record["environment"].get(name)), f"missing environment {name}")
    require(record["build_profile"] == {
        "profile": "release", "lto": "thin", "codegen_units": 1,
        "features": ["extension-module"], "strip": True,
    }, "release build profile mismatch")
    require(nonempty(record["wheel"].get("filename")) and digest(record["wheel"].get("sha256")),
            "missing wheel provenance")
    require([s["id"] for s in record["steps"]] == list(STEPS), "missing or reordered setup steps")
    previous_end = None
    for step in record["steps"]:
        label = step["id"]
        require(step["category"] == STEPS[label], f"wrong category: {label}")
        require(type(step.get("exit_status")) is int and step["exit_status"] == 0,
                f"unsuccessful setup step: {label}")
        seconds = step.get("wall_seconds")
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
                f"invalid wall duration: {label}")
        start = datetime.datetime.fromisoformat(step["started_at"])
        end = datetime.datetime.fromisoformat(step["ended_at"])
        require(start.utcoffset() == end.utcoffset() == datetime.timedelta(0),
                f"timestamps must be UTC: {label}")
        require(end >= start and abs((end - start).total_seconds() - seconds) < 0.01,
                f"wall duration and timestamps disagree: {label}")
        require(previous_end is None or start >= previous_end, "setup step timestamps overlap")
        previous_end = end
        require(isinstance(step.get("command"), list) and step["command"]
                and all(nonempty(arg) for arg in step["command"]), f"missing command: {label}")
        require(nonempty(step.get("cache_state")), f"missing cache state: {label}")
        log = local_file(root / EVIDENCE.parent, step["log"])
        require(digest(step.get("log_sha256")) and sha256(log) == step["log_sha256"],
                f"log hash mismatch: {label}")
    require(set(record["reports"]) == set(REPORTS), "setup must reference all four reports")
    for report, artifact in REPORTS.items():
        reference = record["reports"][report]
        require(reference["artifact"] == artifact, f"wrong artifact for {report}")
        artifact_path = local_file(root, artifact)
        require(sha256(artifact_path) == reference["artifact_sha256"],
                f"historical workload artifact changed: {artifact}")
        raw = json.loads(artifact_path.read_text())
        require(raw["environment"]["git"]["head"] == source["commit"],
                f"setup source differs from workload source: {report}")
        if "ended_epoch_seconds" in raw:
            rerun_start = datetime.datetime.fromisoformat(record["steps"][0]["started_at"])
            require(rerun_start.timestamp() > raw["ended_epoch_seconds"],
                    "setup rerun must postdate historical workload capture")
        markdown = local_file(root, report).read_text()
        require(source["commit"] in markdown, f"report source mismatch: {report}")
        target = EVIDENCE.relative_to("docs").as_posix()
        require(f"]({target})" in markdown and "same-code setup rerun" in markdown,
                f"missing labeled setup reference: {report}")
        summary = markdown[markdown.index("\n## "):]
        require(hashlib.sha256(summary.encode()).hexdigest() == reference["results_sha256"],
                f"historical results changed: {report}")


def validate(record, root=ROOT):
    """Raise ValueError for incomplete, inconsistent, or misattributed evidence."""
    try:
        _validate(record, root)
    except (KeyError, TypeError, AttributeError, OSError) as error:
        raise ValueError(f"incomplete setup evidence: {error}") from error


def main():
    try:
        validate(json.loads((ROOT / EVIDENCE).read_text()))
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise SystemExit(f"Invalid setup evidence: {error}") from error
    print("Validated setup evidence and references from all four historical reports")


if __name__ == "__main__":
    main()
