"""Read-only campaign contract for the rank-2/CUDA diagnostic runner.

Scoring campaigns are reviewer-owned files outside the candidate worktree.
The repository's public matrix and local fixtures are diagnostic inputs only.
"""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SEED = 2511894
PUBLIC_CAMPAIGN = ROOT / "scripts/campaigns/rank2_sum_cuda_diagnostic.json"


def validate_campaign(data):
    if not isinstance(data, dict) or set(data) != {"schema_version", "id", "full", "quick"}:
        raise ValueError("campaign requires schema_version, id, full, and quick")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1 or not isinstance(data["id"], str) or not data["id"]:
        raise ValueError("campaign requires schema_version=1 and a nonempty id")
    full, quick = data.get("full"), data.get("quick")
    if not isinstance(full, list) or not full or not isinstance(quick, list):
        raise ValueError("campaign requires full cells and quick cell IDs")
    ids = []
    for cell in full:
        if not isinstance(cell, dict) or not isinstance(cell.get("id"), str) or not cell["id"]:
            raise ValueError("every cell requires a nonempty id")
        ids.append(cell["id"])
        threads = cell.get("pytorch_threads")
        if type(threads) is not int or threads < 1:
            raise ValueError("pytorch_threads must be a positive integer")
        kind = cell.get("kind")
        if kind in ("sum", "backward_accumulate"):
            shape = cell.get("shape")
            if not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError("CPU cells require two positive dimensions")
            if type(cell.get("axis")) is not int or cell["axis"] not in (0, 1):
                raise ValueError("axis must be 0 or 1")
            if cell.get("layout") not in ("contiguous", "transposed", "offset", "selected"):
                raise ValueError("unknown layout")
            if cell["layout"] == "offset" and shape[1] <= 5:
                raise ValueError("offset layout requires at least six columns")
            if kind == "backward_accumulate" and cell["layout"] != "contiguous":
                raise ValueError("backward cells require contiguous leaves")
            keys = {"id", "kind", "shape", "layout", "axis", "pytorch_threads"}
        elif kind in ("zeros", "to_cpu", "roundtrip"):
            if type(cell.get("elements")) is not int or cell["elements"] < 0:
                raise ValueError("CUDA cells require nonnegative elements")
            keys = {"id", "kind", "elements", "pytorch_threads"}
        else:
            raise ValueError("unknown workload kind")
        if set(cell) != keys:
            raise ValueError("unexpected cell fields")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate full cell IDs")
    if any(not isinstance(value, str) for value in quick) or len(set(quick)) != len(quick):
        raise ValueError("quick IDs must be unique strings")
    if not quick or not set(quick) < set(ids):
        raise ValueError("quick must be a nonempty strict subset of full")
    return data


def load_campaign(path, *, scoring, seed, quick):
    path = Path(path).resolve(strict=True)
    if scoring and path.is_relative_to(ROOT):
        raise ValueError("scoring campaign must be independently owned outside the candidate worktree; use --diagnostic-campaign for local fixtures")
    if type(seed) is not int or seed < 0 or (scoring and seed == PUBLIC_SEED):
        raise ValueError("scoring requires a nonnegative reviewer-supplied held-out seed, distinct from the public diagnostic seed")
    raw = path.read_bytes()
    data = validate_campaign(json.loads(raw))
    selected = [cell for cell in data["full"] if not quick or cell["id"] in data["quick"]]
    return selected, {
        "id": data["id"], "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
        "mode": "quick" if quick else "full", "scoring": scoring, "seed": seed,
        "seed_source": "reviewer_supplied" if scoring else "diagnostic",
        "full_ids": [cell["id"] for cell in data["full"]], "quick_ids": data["quick"],
        "selected_ids": [cell["id"] for cell in selected],
    }


def cell_rng_seed(seed, cell_id):
    # Stable per-cell inputs: quick/full and execution order share identical data.
    return int.from_bytes(hashlib.sha256(f"{seed}:{cell_id}".encode()).digest()[:8], "little")
