"""Untimed selected GELU/package evidence; no score or clean-commit claim.

Run after installing a release wheel. The frozen consumer remains unchanged.
This companion capture also works after Burner's implementation commit.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PINS = {
    "docs/diagnostics/gelu-program-integration/protocol.md": "24bb4d8924b821ef4a94efa1ee89c85db0af46afa138e7c814d502eec9ccbe17",
    "docs/diagnostics/program-identity/protocol.md": "d6e8f51d024b451f8afed33d743647d46396425724b918a599f7b32274f95d5b",
    "docs/diagnostics/program-identity/consumer.py": "1861fca73674d8c89f6285b07883ab9a83e6719c1b6a0c12cd2be1ae7c4c8b42",
    "tests/test_program_identity_consumer.py": "b907c94b00f35fbb18e53510ec692b5fb83f94d37d509fe662418844bc2af5b3",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def packages(wheel, sdist):
    import torch_rs
    import torch_rs.torch_rs as native
    installed = Path(torch_rs.__file__).parent.parent
    required = [
        "src/cuda/erf_provider.ptx", "src/cuda/erf_provider.ll", "src/cuda/gelu.ptx",
        "src/cuda/gelu.cu", "src/cuda/NOTICE.md", "LICENSE",
        "docs/diagnostics/compile-gelu-vendor/notices/requested-CUDA11.8-LICENSE.txt",
        "docs/diagnostics/compile-gelu-vendor/notices/compiler-CUDA12.8-LICENSE.txt",
        "docs/diagnostics/compile-gelu-linked/notices/CUDA13-LICENSE.txt",
    ]
    with zipfile.ZipFile(wheel) as archive:
        files = {n: archive.read(n) for n in archive.namelist()}
        binary = next(data for name, data in files.items() if name.endswith(".so"))
        for path in required:
            content = (ROOT / path).read_bytes()
            if path.endswith(".ptx"):
                assert content in binary, f"provider image missing in wheel extension: {path}"
            elif path.endswith((".md", ".txt")) or path == "LICENSE":
                assert content in files.values(), f"license/notice missing from wheel: {path}"
        for name, data in files.items():
            if name.startswith("torch_rs/") and name.endswith((".py", ".so")):
                assert (installed / name).read_bytes() == data, name
                if name.endswith(".py"):
                    assert (ROOT / "python" / name).read_bytes() == data, name
    with tarfile.open(sdist) as archive:
        for path in required:
            member = next(m for m in archive.getmembers() if m.name.endswith("/" + path))
            assert archive.extractfile(member).read() == (ROOT / path).read_bytes(), path
    return {"wheel": {"path": str(wheel), "sha256": sha(wheel.read_bytes())},
            "sdist": {"path": str(sdist), "sha256": sha(sdist.read_bytes())},
            "native": {"path": native.__file__, "sha256": sha(Path(native.__file__).read_bytes())},
            "required": {path: sha((ROOT / path).read_bytes()) for path in required}}


def main():
    wheel, sdist, output = map(Path, sys.argv[1:])
    assert all(path.resolve().is_relative_to(ROOT) for path in (wheel, sdist, output))
    assert not output.exists()
    for path, digest in PINS.items():
        assert sha((ROOT / path).read_bytes()) == digest, path
    spec = importlib.util.spec_from_file_location("frozen_consumer", ROOT / "docs/diagnostics/program-identity/consumer.py")
    consumer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consumer)
    import torch_rs as native
    assert "torch" not in sys.modules
    record = {"protocolBindings": PINS, "timingData": False, "score": None,
              "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "status": subprocess.check_output(["git", "status", "--short"], text=True),
              "captures": [], "passed": False,
              "unobservableCounters": ["module loads", "pending link destruction", "eager image loads"],
              "ptxMeaning": "NVRTC pre-link PTX; not final linked cubin",
              "nonfiniteEncoding": "Python float transport does not guarantee original NaN payload/sign"}
    try:
        record["packages"] = packages(wheel, sdist)
        sync = consumer.Synchronizer(Path(os.environ["TORCH_RS_CUDART"]))
        compiler = consumer.CompilerPin(Path(os.environ["TORCH_RS_NVRTC"]))
        record.update(gpuBefore=consumer.gpu_identity(), runtime=sync.metadata, compiler=compiler.metadata)
        texts = [
            "def f(x):\n return fw.nn.functional.gelu(x)",
            "def f(x):\n p=fw.nn.functional.gelu(x)\n return (p, fw.nn.functional.gelu(p), p-x)",
            "def f(x):\n p=fw.nn.functional.gelu(x)\n for _ in range(260):\n  p=p.sin()\n return p",
            "def f(x):\n return fw.nn.functional.gelu(x)*0",
        ]
        for text in texts:
            scope = {"fw": native}
            exec(text, scope)
            fn = scope["f"]
            compiled = native.compile(fn)
            for count in (13, 0, 13):
                values = ([-3., -0., 0., 1e-38, .5, 3.] * 3)[:count]
                attempt = {"function": text, "inputValues": values, "shape": [count],
                           "companionInvocation": True, "passed": False}
                record["captures"].append(attempt)
                x = native.tensor(values).to("cuda:0")
                attempt["input"] = consumer.encode(x, [x])
                old_profile = sys.getprofile()
                def reject(frame, event, arg):
                    if event == "call" and frame.f_code is fn.__code__:
                        raise AssertionError("Python body executed")
                    if event == "c_call" and arg is native.nn.functional.gelu:
                        raise AssertionError("eager GELU executed")
                try:
                    sys.setprofile(reject)
                    result, prepared = compiled._torch_rs_pointwise_receipt(x)
                finally:
                    sys.setprofile(old_profile)
                attempt["selected"] = consumer.selected(prepared)
                sync()
                results = result if isinstance(result, tuple) else (result,)
                bits = lambda tensor: [struct.unpack("<I", struct.pack("<f", v))[0] for v in tensor.cpu().tolist()]
                attempt.update(inputs=bits(x), outputs=[bits(t) for t in results],
                               output=consumer.encode(result, [x]), passed=True)
        assert "torch" not in sys.modules
        record["passed"] = True
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        # A failed final observation must not discard successful earlier calls
        # or the input of an interrupted/failed invocation.
        for name, query in (("gpuAfter", consumer.gpu_identity), ("libraries", consumer.libraries)):
            try:
                record[name] = query()
            except BaseException:
                record[name + "Failure"] = traceback.format_exc()
                record["passed"] = False
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    assert record["passed"], "Incomplete capture; see retained partial record"
    print(f"Verified wheel/sdist provider and licenses; captured {len(record['captures'])} selected invocations")


if __name__ == "__main__":
    main()
